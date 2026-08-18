"""Shared test fixtures.

The suite never trains, downloads, or loads real weights by default: the API
tests run against a fake DistilBERT whose tensors flow through the *real*
service code (tokenize -> logits -> softmax -> argmax). Swapping the fake for
another one is a fixture change, not a rewrite.

Tests that do exercise the real checkpoint are marked `integration` and skip
themselves when it is absent.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

# Belt and braces: even a mistake in a test cannot reach the Hugging Face Hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app import main
from app.models.model_loader import SentimentModel
from app.services.sentiment_service import SentimentService, get_sentiment_service

POSITIVE_TEXT = "I really enjoyed this movie."
NEGATIVE_TEXT = "This was a boring, terrible waste of time."

_POSITIVE_WORDS = {"enjoyed", "great", "love", "loved", "wonderful", "excellent", "good"}
_NEGATIVE_WORDS = {"boring", "terrible", "awful", "hate", "hated", "waste", "bad"}

# Class index -> logits. Mirrors the checkpoint's id2label ordering.
_NEUTRAL, _NEGATIVE, _POSITIVE = 0, 1, 2
_LOGITS = {
    _NEUTRAL: torch.tensor([0.0, 0.0]),
    _NEGATIVE: torch.tensor([3.0, -3.0]),
    _POSITIVE: torch.tensor([-3.0, 3.0]),
}


def _code_for(text: str) -> int:
    words = {word.strip(".,!?").lower() for word in text.split()}
    if words & _POSITIVE_WORDS:
        return _POSITIVE
    if words & _NEGATIVE_WORDS:
        return _NEGATIVE
    return _NEUTRAL


class FakeTokenizer:
    """Encodes each text to a single sentiment code, the way a tokenizer would
    encode it to ids — the fake model reads it back out of the tensor."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        codes = torch.tensor([[_code_for(text)] for text in texts])
        return {"input_ids": codes, "attention_mask": torch.ones_like(codes)}


class FakeSequenceClassifier:
    """Stands in for DistilBertForSequenceClassification."""

    def __init__(self, id2label: dict[int, str]) -> None:
        self.config = SimpleNamespace(id2label=dict(id2label), num_labels=len(id2label))
        self.training = True
        self.eval_called = False
        self.raises: Exception | None = None

    def eval(self):
        self.training = False
        self.eval_called = True
        return self

    def to(self, device):
        return self

    def __call__(self, **encoded):
        if self.raises is not None:
            raise self.raises
        codes = encoded["input_ids"][:, 0]
        return SimpleNamespace(logits=torch.stack([_LOGITS[int(code)] for code in codes]))


@pytest.fixture
def id2label() -> dict[int, str]:
    return {0: "negative", 1: "positive"}


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def fake_classifier(id2label) -> FakeSequenceClassifier:
    return FakeSequenceClassifier(id2label)


@pytest.fixture
def fake_loaded_model(fake_tokenizer, fake_classifier, id2label) -> SentimentModel:
    """A SentimentModel with no real weights behind it."""
    return SentimentModel(
        tokenizer=fake_tokenizer,
        model=fake_classifier,
        device=torch.device("cpu"),
        id2label=id2label,
        model_path=Path("/fake/checkpoint"),
    )


@pytest.fixture
def service(fake_loaded_model) -> SentimentService:
    """The real service, wrapping the fake model."""
    return SentimentService(fake_loaded_model)


@pytest.fixture
def client(service, monkeypatch) -> TestClient:
    """TestClient whose app loads nothing at startup and uses the fake service."""
    monkeypatch.setattr(main, "init_model", lambda *args, **kwargs: None)
    main.app.dependency_overrides[get_sentiment_service] = lambda: service
    with TestClient(main.app) as test_client:
        yield test_client
    main.app.dependency_overrides.clear()
