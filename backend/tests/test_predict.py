import pytest

from tests.conftest import NEGATIVE_TEXT, POSITIVE_TEXT

PROBABILITY_SUM_TOLERANCE = 1e-3


def _assert_valid_payload(body: dict, text: str, expected_sentiment: str) -> None:
    assert set(body) == {"text", "sentiment", "confidence", "probabilities"}
    assert body["text"] == text
    assert body["sentiment"] == expected_sentiment

    probabilities = body["probabilities"]
    assert set(probabilities) == {"positive", "negative"}

    confidence = body["confidence"]
    positive = probabilities["positive"]
    negative = probabilities["negative"]

    assert 0.0 <= confidence <= 1.0
    assert 0.0 <= positive <= 1.0
    assert 0.0 <= negative <= 1.0
    assert positive + negative == pytest.approx(1.0, abs=PROBABILITY_SUM_TOLERANCE)

    # The reported confidence is the winning label's probability.
    assert confidence == pytest.approx(probabilities[expected_sentiment])
    assert confidence == pytest.approx(max(positive, negative))


def test_predict_positive_text(client):
    response = client.post("/predict", json={"text": POSITIVE_TEXT})

    assert response.status_code == 200
    body = response.json()
    _assert_valid_payload(body, POSITIVE_TEXT, "positive")
    assert body["probabilities"]["positive"] > body["probabilities"]["negative"]


def test_predict_negative_text(client):
    response = client.post("/predict", json={"text": NEGATIVE_TEXT})

    assert response.status_code == 200
    body = response.json()
    _assert_valid_payload(body, NEGATIVE_TEXT, "negative")
    assert body["probabilities"]["negative"] > body["probabilities"]["positive"]


def test_predict_echoes_input_text(client):
    text = "I really enjoyed this movie."
    body = client.post("/predict", json={"text": text}).json()

    assert body["text"] == text


def test_predict_is_deterministic(client):
    first = client.post("/predict", json={"text": POSITIVE_TEXT}).json()
    second = client.post("/predict", json={"text": POSITIVE_TEXT}).json()

    assert first == second


def test_predict_does_not_reload_the_model(client, fake_classifier):
    """Every request reuses the model loaded at startup."""
    fake_classifier.eval_called = False

    for _ in range(5):
        assert client.post("/predict", json={"text": POSITIVE_TEXT}).status_code == 200

    assert fake_classifier.eval_called is False


def test_predict_runs_in_eval_mode(client, fake_classifier):
    client.post("/predict", json={"text": POSITIVE_TEXT})

    assert fake_classifier.training is False


def test_predict_truncates_long_input_at_model_limit(client, fake_tokenizer):
    client.post("/predict", json={"text": "great " * 500})

    call = fake_tokenizer.calls[-1]
    assert call["truncation"] is True
    assert call["max_length"] == 512
