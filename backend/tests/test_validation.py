"""Input validation for POST /predict."""

import pytest

from app.schemas import MAX_TEXT_LENGTH

UNPROCESSABLE = 422


def test_empty_text_is_rejected(client):
    response = client.post("/predict", json={"text": ""})

    assert response.status_code == UNPROCESSABLE


@pytest.mark.parametrize("text", ["   ", "\t", "\n", " \t\n  "])
def test_whitespace_only_text_is_rejected(client, text):
    response = client.post("/predict", json={"text": text})

    assert response.status_code == UNPROCESSABLE


def test_excessively_long_text_is_rejected(client):
    response = client.post("/predict", json={"text": "a" * (MAX_TEXT_LENGTH + 1)})

    assert response.status_code == UNPROCESSABLE


def test_text_at_the_length_limit_is_accepted(client):
    response = client.post("/predict", json={"text": "a" * MAX_TEXT_LENGTH})

    assert response.status_code == 200


def test_missing_text_field_is_rejected(client):
    response = client.post("/predict", json={})

    assert response.status_code == UNPROCESSABLE
    assert any(error["loc"][-1] == "text" for error in response.json()["detail"])


def test_null_text_is_rejected(client):
    response = client.post("/predict", json={"text": None})

    assert response.status_code == UNPROCESSABLE


@pytest.mark.parametrize("value", [123, 4.5, True, ["a"], {"nested": "object"}])
def test_wrong_text_type_is_rejected(client, value):
    response = client.post("/predict", json={"text": value})

    assert response.status_code == UNPROCESSABLE


def test_invalid_json_body_is_rejected(client):
    response = client.post(
        "/predict",
        content="not json at all",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == UNPROCESSABLE


def test_json_array_body_is_rejected(client):
    response = client.post("/predict", json=[{"text": "hello"}])

    assert response.status_code == UNPROCESSABLE


def test_empty_body_is_rejected(client):
    response = client.post("/predict")

    assert response.status_code == UNPROCESSABLE


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
    assert client.get("/predict").status_code == 405
