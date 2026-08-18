"""Failure paths for POST /predict."""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.models.model_loader import ModelLoadError
from app.services.sentiment_service import InferenceError, get_sentiment_service
from tests.conftest import POSITIVE_TEXT


@pytest.fixture
def client_without_model(monkeypatch):
    """The app is up, but the model is unavailable."""

    def unavailable():
        raise ModelLoadError("Model directory not found: /nowhere")

    monkeypatch.setattr(main, "init_model", lambda *args, **kwargs: None)
    main.app.dependency_overrides[get_sentiment_service] = unavailable
    with TestClient(main.app, raise_server_exceptions=False) as test_client:
        yield test_client
    main.app.dependency_overrides.clear()


def test_missing_model_returns_503(client_without_model):
    response = client_without_model.post("/predict", json={"text": POSITIVE_TEXT})

    assert response.status_code == 503
    assert response.json() == {"detail": "Sentiment model is not available."}


def test_health_still_works_without_a_model(client_without_model):
    assert client_without_model.get("/health").status_code == 200


def test_inference_failure_returns_500(client, fake_classifier):
    fake_classifier.raises = RuntimeError("CUDA out of memory")

    response = client.post("/predict", json={"text": POSITIVE_TEXT})

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to run inference on the provided text."


def test_error_response_does_not_leak_internals(client, fake_classifier):
    fake_classifier.raises = RuntimeError("secret path /home/user/weights.bin")

    body = client.post("/predict", json={"text": POSITIVE_TEXT}).json()

    assert "secret path" not in str(body)


def test_service_level_invalid_text_returns_422(client, service):
    """Defence in depth: if a blank string reached the service, it is a 422."""
    with pytest.raises(Exception):
        service.predict("   ")

    assert client.post("/predict", json={"text": "   "}).status_code == 422


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
