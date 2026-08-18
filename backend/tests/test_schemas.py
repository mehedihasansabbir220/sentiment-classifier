"""Contract-level tests for the request/response models."""

import pytest
from pydantic import ValidationError

from app.schemas import MAX_TEXT_LENGTH, HealthResponse, PredictRequest, PredictResponse


def test_valid_request():
    assert PredictRequest(text="I really enjoyed this movie.").text == (
        "I really enjoyed this movie."
    )


@pytest.mark.parametrize("text", ["", "   ", "\t", "\n", " \t\n "])
def test_blank_text_is_rejected_by_the_schema(text):
    """The schema rejects blanks on its own, without reaching the service."""
    with pytest.raises(ValidationError):
        PredictRequest(text=text)


def test_text_longer_than_the_limit_is_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(text="a" * (MAX_TEXT_LENGTH + 1))


def test_text_at_the_limit_is_accepted():
    assert len(PredictRequest(text="a" * MAX_TEXT_LENGTH).text) == MAX_TEXT_LENGTH


@pytest.mark.parametrize("value", [None, 123, 4.5, ["a"], {"a": 1}])
def test_non_string_text_is_rejected(value):
    with pytest.raises(ValidationError):
        PredictRequest(text=value)


def test_missing_text_is_rejected():
    with pytest.raises(ValidationError):
        PredictRequest()


def test_response_requires_confidence_in_range():
    for bad in (-0.01, 1.01):
        with pytest.raises(ValidationError):
            PredictResponse(
                text="x",
                sentiment="positive",
                confidence=bad,
                probabilities={"positive": 1.0, "negative": 0.0},
            )


def test_response_round_trips():
    payload = {
        "text": "I really enjoyed this movie.",
        "sentiment": "positive",
        "confidence": 0.97,
        "probabilities": {"positive": 0.97, "negative": 0.03},
    }

    assert PredictResponse(**payload).model_dump() == payload


def test_health_response_shape():
    assert HealthResponse(status="ok").model_dump() == {"status": "ok"}
