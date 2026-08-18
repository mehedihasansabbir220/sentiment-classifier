"""End-to-end checks against the real fine-tuned checkpoint.

Skipped when the checkpoint is absent (a fresh clone has no weights — they are
gitignored). Loading is offline-only, so these tests never download anything.

Run with:  pytest -m integration
"""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import settings
from app.models.model_loader import load_model
from app.services.sentiment_service import SentimentService

pytestmark = pytest.mark.integration

CHECKPOINT = settings.resolved_model_path

requires_checkpoint = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").is_file(),
    reason=f"No checkpoint at {CHECKPOINT}",
)


@pytest.fixture(scope="module")
def real_service():
    return SentimentService(load_model())


@requires_checkpoint
def test_checkpoint_declares_the_expected_labels(real_service):
    assert real_service.labels == {0: "negative", 1: "positive"}


@requires_checkpoint
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I really enjoyed this movie.", "positive"),
        ("An absolute masterpiece, I loved every second.", "positive"),
        ("This was a boring, terrible waste of time.", "negative"),
        ("Awful acting and a dreadful script.", "negative"),
    ],
)
def test_real_model_classifies_clear_sentiment(real_service, text, expected):
    result = real_service.predict(text)

    assert result.sentiment == expected
    assert 0.0 <= result.confidence <= 1.0
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-3)


@requires_checkpoint
def test_real_model_leaves_weights_untouched(real_service):
    import torch

    weights = real_service._model.classifier.weight
    before = weights.detach().clone()

    real_service.predict("I really enjoyed this movie.")

    assert torch.equal(before, weights)
    assert real_service._model.training is False
    assert all(param.grad is None for param in real_service._model.parameters())


@requires_checkpoint
def test_api_end_to_end_with_the_real_model():
    with TestClient(main.app) as client:
        response = client.post("/predict", json={"text": "I really enjoyed this movie."})

    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == "positive"
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"]) == {"positive", "negative"}
