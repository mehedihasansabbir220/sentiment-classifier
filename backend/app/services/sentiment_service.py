"""Sentiment inference pipeline.

text -> validate -> tokenize -> DistilBERT -> softmax -> label + confidence

This module is deliberately framework-free: it imports no FastAPI, knows
nothing about HTTP, and can be used from a script, a notebook, or a worker.
It only *runs* the model that :mod:`app.models.model_loader` loaded at
startup — it never trains, downloads, or writes weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from app.errors import EmptyTextError, InferenceError, InvalidTextError
from app.models.model_loader import SentimentModel, get_model

logger = logging.getLogger(__name__)

# DistilBERT's positional limit; longer inputs are truncated, not rejected.
MAX_LENGTH = 512

# Probabilities are rounded so the API returns stable, readable numbers.
ROUND_TO = 4

# Fallback only — the real names come from the checkpoint's config.
_FALLBACK_LABELS = {0: "negative", 1: "positive"}

# Re-exported so existing imports keep working.
SentimentServiceError = InferenceError


@dataclass(frozen=True)
class SentimentPrediction:
    text: str
    sentiment: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
        }


class SentimentService:
    """Runs the fine-tuned classifier. Stateless apart from the model it wraps."""

    def __init__(
        self,
        model: SentimentModel,
        max_length: int = MAX_LENGTH,
        round_to: int = ROUND_TO,
    ) -> None:
        self._loaded = model
        self._tokenizer = model.tokenizer
        self._model = model.model
        self._device = model.device
        self._max_length = max_length
        self._round_to = round_to
        self._id2label = self._resolve_labels(model)

        # Idempotent, but keeps the guarantee local: dropout off, no training.
        self._model.eval()

    # ------------------------------------------------------------------
    # labels
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_labels(model: SentimentModel) -> dict[int, str]:
        """Read label names from the checkpoint config, never from position.

        The trained `config.json` carries `id2label`; using it means a
        checkpoint that maps 0 -> positive would still be read correctly.
        """
        raw = getattr(getattr(model.model, "config", None), "id2label", None) or model.id2label
        if not raw:
            logger.warning("Checkpoint exposes no id2label; falling back to %s", _FALLBACK_LABELS)
            return dict(_FALLBACK_LABELS)
        return {int(key): str(value).strip().lower() for key, value in raw.items()}

    @property
    def labels(self) -> dict[int, str]:
        return dict(self._id2label)

    @property
    def device(self) -> torch.device:
        """The device inference runs on: CUDA when available, else CPU."""
        return self._device

    # ------------------------------------------------------------------
    # 1. validate
    # ------------------------------------------------------------------
    @staticmethod
    def validate_text(text: str) -> str:
        """Reject anything that is not a non-empty string."""
        if not isinstance(text, str):
            raise InvalidTextError(f"text must be a string, got {type(text).__name__}")
        cleaned = text.strip()
        if not cleaned:
            raise EmptyTextError("text must not be empty or whitespace only")
        return cleaned

    # ------------------------------------------------------------------
    # 2-8. tokenize -> model -> softmax -> label + confidence
    # ------------------------------------------------------------------
    def predict(self, text: str) -> SentimentPrediction:
        """Classify a single text."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[SentimentPrediction]:
        """Classify several texts in one forward pass."""
        if not texts:
            return []

        cleaned = [self.validate_text(text) for text in texts]
        probabilities = self._forward(cleaned)

        predictions: list[SentimentPrediction] = []
        for original, row in zip(texts, probabilities):
            scores = self._as_label_scores(row)
            # 7. highest probability wins
            sentiment = max(scores, key=scores.__getitem__)
            predictions.append(
                SentimentPrediction(
                    text=original,
                    sentiment=sentiment,
                    confidence=scores[sentiment],  # 8. confidence
                    probabilities=scores,
                )
            )
        return predictions

    def _forward(self, texts: list[str]) -> torch.Tensor:
        """Tokenize and run the model without gradients. Returns probabilities."""
        try:
            # 2. tokenize with the trained model's own tokenizer
            encoded = self._tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length,
                padding=True,
            )
            # CPU or CUDA — whichever the model was loaded onto
            encoded = {key: value.to(self._device) for key, value in encoded.items()}

            # 3 + 4. forward pass with gradients disabled
            with torch.no_grad():
                logits = self._model(**encoded).logits

            # 5. logits -> probabilities
            return torch.softmax(logits, dim=-1).cpu()
        except Exception as exc:  # noqa: BLE001 - surfaced as InferenceError
            logger.exception("Inference failed")
            raise InferenceError(f"Failed to run inference: {type(exc).__name__}") from exc

    def _as_label_scores(self, row: torch.Tensor) -> dict[str, float]:
        """6. Map each class index to its configured label name.

        Ordered highest probability first, so the winning label leads.
        """
        scores = {
            self._id2label.get(index, f"label_{index}"): round(float(score), self._round_to)
            for index, score in enumerate(row)
        }
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


_service: SentimentService | None = None


def get_sentiment_service() -> SentimentService:
    """Return the shared service wrapping the model loaded at startup.

    Cheap to call per request: it never loads weights.
    """
    global _service
    if _service is None:
        _service = SentimentService(get_model())
    return _service


def reset_sentiment_service() -> None:
    """Drop the cached service. Intended for tests and shutdown."""
    global _service
    _service = None
