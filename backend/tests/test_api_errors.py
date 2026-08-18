"""Failure paths for POST /predict.

Responses are always ``{error: {code, message}}``. Technical detail stays in logs.
"""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.errors import (
    ErrorCode,
    ModelLoadError,
    ModelNotFoundError,
    TokenizerLoadError,
)
from app.services.sentiment_service import get_sentiment_service
from tests.conftest import POSITIVE_TEXT

LEAK_MARKERS = ("Traceback", "File \"", ".py\", line", "RuntimeError", "CUDA")


def _assert_error(response, status_code: int, code: ErrorCode) -> dict:
    assert response.status_code == status_code
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message"}
    assert body["error"]["code"] == code.value
    assert body["error"]["message"]
    leaked = str(body)
    for marker in LEAK_MARKERS:
        assert marker not in leaked
    return body


@pytest.fixture
def client_without_model(monkeypatch):
    def unavailable():
        raise ModelNotFoundError("Model directory not found: /nowhere")

    monkeypatch.setattr(main, "init_model", lambda *args, **kwargs: None)
    main.app.dependency_overrides[get_sentiment_service] = unavailable
    with TestClient(main.app, raise_server_exceptions=False) as test_client:
        yield test_client
    main.app.dependency_overrides.clear()


def test_missing_model_returns_503(client_without_model):
    response = client_without_model.post("/predict", json={"text": POSITIVE_TEXT})

    body = _assert_error(response, 503, ErrorCode.MODEL_NOT_FOUND)
    assert "nowhere" not in str(body)
    assert body["error"]["message"] == "The sentiment model is not available."


def test_tokenizer_failure_returns_503(monkeypatch):
    def unavailable():
        raise TokenizerLoadError("Failed to load the tokenizer from /secret/path")

    monkeypatch.setattr(main, "init_model", lambda *args, **kwargs: None)
    main.app.dependency_overrides[get_sentiment_service] = unavailable
    with TestClient(main.app, raise_server_exceptions=False) as client:
        response = client.post("/predict", json={"text": POSITIVE_TEXT})
    main.app.dependency_overrides.clear()

    body = _assert_error(response, 503, ErrorCode.TOKENIZER_FAILURE)
    assert "/secret/path" not in str(body)


def test_model_loading_failure_returns_503(monkeypatch):
    def unavailable():
        raise ModelLoadError("Failed to load the model weights from /home/user/weights.bin")

    monkeypatch.setattr(main, "init_model", lambda *args, **kwargs: None)
    main.app.dependency_overrides[get_sentiment_service] = unavailable
    with TestClient(main.app, raise_server_exceptions=False) as client:
        response = client.post("/predict", json={"text": POSITIVE_TEXT})
    main.app.dependency_overrides.clear()

    body = _assert_error(response, 503, ErrorCode.MODEL_LOAD_FAILURE)
    assert "weights.bin" not in str(body)
    assert "/home/user" not in str(body)


def test_health_still_works_without_a_model(client_without_model):
    assert client_without_model.get("/health").status_code == 200


def test_inference_failure_returns_500(client, fake_classifier):
    fake_classifier.raises = RuntimeError("CUDA out of memory at /secret")

    response = client.post("/predict", json={"text": POSITIVE_TEXT})

    body = _assert_error(response, 500, ErrorCode.INFERENCE_FAILURE)
    assert "CUDA" not in str(body)
    assert "/secret" not in str(body)


def test_unexpected_error_returns_500_without_internals(client, service, monkeypatch):
    def boom(_text: str):
        raise RuntimeError("secret path /home/user/weights.bin")

    monkeypatch.setattr(service, "predict", boom)

    response = client.post("/predict", json={"text": POSITIVE_TEXT})

    body = _assert_error(response, 500, ErrorCode.INTERNAL_ERROR)
    assert "secret path" not in str(body)
    assert "weights.bin" not in str(body)


def test_error_response_does_not_leak_internals(client, fake_classifier):
    fake_classifier.raises = RuntimeError("secret path /home/user/weights.bin")

    body = client.post("/predict", json={"text": POSITIVE_TEXT}).json()

    assert "secret path" not in str(body)
    assert "detail" not in body
    assert "traceback" not in str(body).lower()


def test_service_level_invalid_text_returns_422(client, service):
    """Defence in depth: if a blank string reached the service, it is a 422."""
    with pytest.raises(Exception):
        service.predict("   ")

    response = client.post("/predict", json={"text": "   "})
    _assert_error(response, 422, ErrorCode.EMPTY_TEXT)


def test_cors_headers_for_the_frontend(client):
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_rejects_unknown_origin(client):
    response = client.post(
        "/predict",
        json={"text": POSITIVE_TEXT},
        headers={"Origin": "http://evil.example"},
    )

    assert "access-control-allow-origin" not in response.headers
