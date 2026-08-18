"""Unit tests for the inference pipeline, independent of FastAPI."""

import pytest
import torch

from app.services.sentiment_service import (
    InferenceError,
    InvalidTextError,
    SentimentService,
)
from tests.conftest import NEGATIVE_TEXT, POSITIVE_TEXT


def test_service_puts_model_in_eval_mode(fake_loaded_model):
    assert fake_loaded_model.model.training is True

    SentimentService(fake_loaded_model)

    assert fake_loaded_model.model.training is False


def test_predict_returns_full_payload(service):
    result = service.predict(POSITIVE_TEXT)

    assert result.to_dict() == {
        "text": POSITIVE_TEXT,
        "sentiment": "positive",
        "confidence": result.confidence,
        "probabilities": result.probabilities,
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [(POSITIVE_TEXT, "positive"), (NEGATIVE_TEXT, "negative")],
)
def test_predict_picks_the_highest_probability(service, text, expected):
    result = service.predict(text)

    assert result.sentiment == expected
    assert result.confidence == max(result.probabilities.values())
    assert result.confidence == result.probabilities[expected]


def test_probabilities_are_a_distribution(service):
    result = service.predict(POSITIVE_TEXT)

    assert set(result.probabilities) == {"positive", "negative"}
    assert all(0.0 <= value <= 1.0 for value in result.probabilities.values())
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_text_is_rejected(service, text):
    with pytest.raises(InvalidTextError):
        service.predict(text)


@pytest.mark.parametrize("value", [None, 123, 4.5, ["text"], {"text": "x"}])
def test_non_string_text_is_rejected(service, value):
    with pytest.raises(InvalidTextError):
        service.predict(value)


def test_batch_prediction(service):
    results = service.predict_batch([POSITIVE_TEXT, NEGATIVE_TEXT])

    assert [result.sentiment for result in results] == ["positive", "negative"]


def test_empty_batch_returns_nothing(service, fake_tokenizer):
    assert service.predict_batch([]) == []
    assert fake_tokenizer.calls == []


def test_labels_come_from_the_model_config(fake_loaded_model):
    """A checkpoint that maps 0 -> positive must not be read positionally."""
    fake_loaded_model.model.config.id2label = {0: "POSITIVE", 1: "NEGATIVE"}

    service = SentimentService(fake_loaded_model)

    assert service.labels == {0: "positive", 1: "negative"}
    # Logits favour index 1, which this config calls negative.
    assert service.predict(POSITIVE_TEXT).sentiment == "negative"


def test_falls_back_to_loader_labels_when_config_has_none(fake_loaded_model):
    fake_loaded_model.model.config.id2label = None

    service = SentimentService(fake_loaded_model)

    assert service.labels == {0: "negative", 1: "positive"}


def test_inference_runs_without_gradients(service, fake_loaded_model):
    seen = {}
    inner = fake_loaded_model.model

    class GradProbe:
        training = False
        config = inner.config

        def eval(self):
            return self

        def __call__(self, **encoded):
            seen["grad_enabled"] = torch.is_grad_enabled()
            return inner(**encoded)

    service._model = GradProbe()

    with torch.enable_grad():
        service.predict(POSITIVE_TEXT)

    assert seen["grad_enabled"] is False


def test_model_failure_becomes_inference_error(service, fake_classifier):
    fake_classifier.raises = RuntimeError("CUDA out of memory")

    with pytest.raises(InferenceError):
        service.predict(POSITIVE_TEXT)


def test_tokenizer_is_called_with_truncation(service, fake_tokenizer):
    service.predict(POSITIVE_TEXT)

    call = fake_tokenizer.calls[-1]
    assert call["truncation"] is True
    assert call["max_length"] == 512
    assert call["return_tensors"] == "pt"


def test_text_is_echoed_untrimmed_but_model_sees_trimmed(service, fake_tokenizer):
    result = service.predict("  I really enjoyed this movie.  ")

    assert result.text == "  I really enjoyed this movie.  "
    assert fake_tokenizer.calls[-1]["texts"] == ["I really enjoyed this movie."]


def test_winning_label_is_listed_first(service):
    result = service.predict(NEGATIVE_TEXT)

    assert list(result.probabilities)[0] == "negative"


def test_service_uses_the_loaded_device(service):
    assert service.device == torch.device("cpu")
