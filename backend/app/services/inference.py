"""Inference service: turns text into a sentiment prediction.

This is the only place that knows how to go from a string to probabilities.
Routes call this service; the service uses the model loaded at startup by
:mod:`app.models.model_loader`. Nothing here loads or trains anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from app.models.model_loader import SentimentModel, get_model

logger = logging.getLogger(__name__)

# DistilBERT's positional limit; longer inputs are truncated.
MAX_LENGTH = 512

# Probabilities are rounded for a stable, readable API response.
_ROUND_TO = 4


class InferenceError(RuntimeError):
    """Raised when a prediction cannot be produced."""


@dataclass(frozen=True)
class Prediction:
    text: str
    sentiment: str
    confidence: float
    probabilities: dict[str, float]


class SentimentService:
    """Stateless wrapper around a loaded checkpoint."""

    def __init__(self, model: SentimentModel, max_length: int = MAX_LENGTH) -> None:
        self._model = model
        self._max_length = max_length

    @property
    def labels(self) -> dict[int, str]:
        return self._model.id2label

    @property
    def device(self) -> torch.device:
        return self._model.device

    def predict(self, text: str) -> Prediction:
        """Classify a single text."""
        return self.predict_batch([text])[0]

    @torch.inference_mode()
    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        """Classify a batch of texts. Runs without gradients, in eval mode."""
        if not texts:
            return []

        try:
            encoded = self._model.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length,
                padding=True,
            )
            encoded = {key: value.to(self._model.device) for key, value in encoded.items()}

            logits = self._model.model(**encoded).logits
            scores = torch.softmax(logits, dim=-1).cpu()
        except Exception as exc:  # noqa: BLE001 - surfaced as InferenceError
            logger.exception("Inference failed")
            raise InferenceError(f"Failed to run inference: {exc}") from exc

        predictions: list[Prediction] = []
        for text, row in zip(texts, scores):
            probabilities = {
                self._model.id2label[index]: round(float(score), _ROUND_TO)
                for index, score in enumerate(row)
            }
            sentiment = self._model.id2label[int(torch.argmax(row))]
            predictions.append(
                Prediction(
                    text=text,
                    sentiment=sentiment,
                    confidence=probabilities[sentiment],
                    probabilities=probabilities,
                )
            )
        return predictions


_service: SentimentService | None = None


def get_sentiment_service() -> SentimentService:
    """FastAPI dependency returning the shared service.

    Wraps the single model instance loaded at startup — this never loads
    weights, so it is cheap to call per request.
    """
    global _service
    if _service is None:
        _service = SentimentService(get_model())
    return _service


def reset_sentiment_service() -> None:
    """Drop the cached service. Intended for tests and reloads."""
    global _service
    _service = None
