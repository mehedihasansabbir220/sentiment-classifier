"""Input validation for POST /predict."""

import pytest

from app.errors import ErrorCode
from app.schemas import MAX_TEXT_LENGTH

UNPROCESSABLE = 422


def _assert_error(response, code: ErrorCode) -> None:
    assert response.status_code == UNPROCESSABLE
    body = response.json()
    assert body == {
        "error": {
            "code": code.value,
            "message": body["error"]["message"],
        }
    }
    assert "detail" not in body
    assert "loc" not in str(body)


def test_empty_text_is_rejected(client):
    response = client.post("/predict", json={"text": ""})

    _assert_error(response, ErrorCode.EMPTY_TEXT)


@pytest.mark.parametrize("text", ["   ", "\t", "\n", " \t\n  "])
def test_whitespace_only_text_is_rejected(client, text):
    response = client.post("/predict", json={"text": text})

    _assert_error(response, ErrorCode.EMPTY_TEXT)


def test_excessively_long_text_is_rejected(client):
    response = client.post("/predict", json={"text": "a" * (MAX_TEXT_LENGTH + 1)})

    _assert_error(response, ErrorCode.TEXT_TOO_LONG)


def test_text_at_the_length_limit_is_accepted(client):
    response = client.post("/predict", json={"text": "a" * MAX_TEXT_LENGTH})

    assert response.status_code == 200


def test_missing_text_field_is_rejected(client):
    response = client.post("/predict", json={})

    _assert_error(response, ErrorCode.INVALID_REQUEST)


def test_null_text_is_rejected(client):
    response = client.post("/predict", json={"text": None})

    _assert_error(response, ErrorCode.INVALID_REQUEST)


@pytest.mark.parametrize("value", [123, 4.5, True, ["a"], {"nested": "object"}])
def test_wrong_text_type_is_rejected(client, value):
    response = client.post("/predict", json={"text": value})

    _assert_error(response, ErrorCode.INVALID_REQUEST)


def test_invalid_json_body_is_rejected(client):
    response = client.post(
        "/predict",
        content="not json at all",
        headers={"Content-Type": "application/json"},
    )

    _assert_error(response, ErrorCode.INVALID_REQUEST)


def test_json_array_body_is_rejected(client):
    response = client.post("/predict", json=[{"text": "hello"}])

    _assert_error(response, ErrorCode.INVALID_REQUEST)


def test_empty_body_is_rejected(client):
    response = client.post("/predict")

    _assert_error(response, ErrorCode.INVALID_REQUEST)


def test_unknown_fields_are_ignored(client):
    response = client.post(
        "/predict",
        json={"text": "I really enjoyed this movie.", "unexpected": "field"},
    )

    assert response.status_code == 200
    assert "unexpected" not in response.json()


def test_rejected_requests_never_reach_the_model(client, fake_tokenizer):
    client.post("/predict", json={"text": "  "})
    client.post("/predict", json={})

    assert fake_tokenizer.calls == []


def test_get_on_predict_is_not_allowed(client):
    response = client.get("/predict")

    assert response.status_code == 405
    body = response.json()
    assert body["error"]["code"] == ErrorCode.INVALID_REQUEST.value
    assert "detail" not in body
