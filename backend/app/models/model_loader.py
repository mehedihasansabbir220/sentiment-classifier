"""Loads the fine-tuned DistilBERT sentiment checkpoint.

The checkpoint was produced in Google Colab and lives on disk under
``MODEL_PATH`` (default ``model/sentiment-distilbert``). This module only
*loads* it: nothing here trains, downloads, or writes model weights. Hugging
Face is called with ``local_files_only=True`` so a missing or malformed
checkpoint fails loudly instead of silently pulling a model from the Hub.

The model is loaded exactly once, at FastAPI startup, via :func:`init_model`.
The inference service calls :func:`get_model` to reuse that single instance;
inference itself lives in :mod:`app.services.sentiment_service`.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from app.config import settings
from app.errors import ModelLoadError, ModelNotFoundError, TokenizerLoadError

logger = logging.getLogger(__name__)

# Label mapping the fine-tuned checkpoint is expected to carry.
EXPECTED_LABELS: dict[int, str] = {0: "negative", 1: "positive"}

# At least one of these must exist for the tokenizer to be loadable offline.
_TOKENIZER_FILES = (
    "tokenizer.json",
    "vocab.txt",
    "vocab.json",
    "sentencepiece.bpe.model",
    "spiece.model",
)

# At least one of these must exist for the weights to be loadable.
_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "tf_model.h5",
    "flax_model.msgpack",
)


@dataclass(frozen=True)
class SentimentModel:
    """A loaded checkpoint and the context needed to run it.

    Deliberately a plain container: inference lives in
    :mod:`app.services.sentiment_service`, not here.
    """

    tokenizer: object
    model: object
    device: torch.device
    id2label: dict[int, str]
    model_path: Path


def resolve_device() -> torch.device:
    """CUDA when available, CPU otherwise."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validate_checkpoint_dir(model_path: Path) -> None:
    """Fail with an actionable message before Hugging Face is even called."""
    if not model_path.exists():
        raise ModelNotFoundError(
            f"Model directory not found: {model_path}. "
            "Copy the fine-tuned checkpoint exported from Colab into that "
            "directory, or point MODEL_PATH at its location."
        )
    if not model_path.is_dir():
        raise ModelNotFoundError(
            f"MODEL_PATH must be a directory containing the checkpoint, "
            f"but {model_path} is a file."
        )

    if not (model_path / "config.json").is_file():
        raise ModelNotFoundError(
            f"Model configuration not found: {model_path / 'config.json'}. "
            "The directory does not look like a Hugging Face checkpoint saved "
            "with save_pretrained()."
        )

    if not any((model_path / name).is_file() for name in _TOKENIZER_FILES):
        raise TokenizerLoadError(
            f"Tokenizer files are missing in {model_path}. Expected at least one "
            f"of: {', '.join(_TOKENIZER_FILES)}. Re-export the tokenizer with "
            "tokenizer.save_pretrained()."
        )

    if not any((model_path / name).is_file() for name in _WEIGHT_FILES):
        raise ModelNotFoundError(
            f"Model weights are missing in {model_path}. Expected at least one "
            f"of: {', '.join(_WEIGHT_FILES)}. Note that weights are gitignored, "
            "so a fresh clone will not contain them."
        )


def _normalize_id2label(raw: dict | None) -> dict[int, str]:
    """Coerce config id2label keys to ints (JSON stores them as strings)."""
    if not raw:
        return {}
    normalized: dict[int, str] = {}
    for key, value in raw.items():
        try:
            normalized[int(key)] = str(value).strip().lower()
        except (TypeError, ValueError) as exc:
            raise ModelLoadError(
                f"Invalid id2label entry in the model configuration: {key!r} -> {value!r}."
            ) from exc
    return normalized


def _verify_labels(config, model_path: Path) -> dict[int, str]:
    """Check the checkpoint really is the 0=negative / 1=positive classifier."""
    id2label = _normalize_id2label(getattr(config, "id2label", None))

    if id2label != EXPECTED_LABELS:
        raise ModelLoadError(
            f"Unexpected label mapping in {model_path / 'config.json'}: "
            f"{id2label or '<missing>'}. Expected {EXPECTED_LABELS} "
            "(0 = negative, 1 = positive)."
        )

    num_labels = getattr(config, "num_labels", None)
    if num_labels != len(EXPECTED_LABELS):
        raise ModelLoadError(
            f"Model configuration declares num_labels={num_labels}, "
            f"expected {len(EXPECTED_LABELS)}."
        )

    return id2label


def load_model(model_path: Path | str | None = None) -> SentimentModel:
    """Load tokenizer + model from disk and return them ready for inference.

    Raises :class:`ModelLoadError` for any missing or invalid checkpoint.
    """
    path = Path(model_path) if model_path is not None else settings.resolved_model_path
    _validate_checkpoint_dir(path)

    path_str = str(path)
    logger.info("Loading sentiment checkpoint from %s", path_str)

    try:
        config = AutoConfig.from_pretrained(path_str, local_files_only=True)
    except ModelLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced as ModelLoadError below
        raise ModelLoadError(
            f"Invalid model configuration in {path / 'config.json'}: {exc}"
        ) from exc

    id2label = _verify_labels(config, path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(path_str, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        raise TokenizerLoadError(f"Failed to load the tokenizer from {path}: {exc}") from exc

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            path_str,
            config=config,
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"Failed to load the model weights from {path}: {exc}") from exc

    device = resolve_device()
    try:
        model.to(device)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"Failed to place the model on {device}: {exc}") from exc
    model.eval()  # disable dropout; inference only

    logger.info(
        "Sentiment model ready on %s with labels %s",
        device.type,
        id2label,
    )
    return SentimentModel(
        tokenizer=tokenizer,
        model=model,
        device=device,
        id2label=id2label,
        model_path=path,
    )


_model: SentimentModel | None = None
_lock = threading.Lock()


def init_model(model_path: Path | str | None = None) -> SentimentModel:
    """Load the model once, at application startup.

    Safe to call more than once: subsequent calls return the already loaded
    instance instead of reloading the weights.
    """
    global _model
    with _lock:
        if _model is None:
            _model = load_model(model_path)
        return _model


def get_model() -> SentimentModel:
    """Return the model loaded at startup.

    Never loads on demand — a request must not pay for model loading.
    """
    model = _model
    if model is None:
        raise ModelLoadError(
            "The sentiment model is not loaded. init_model() runs during "
            "FastAPI startup; check the startup logs for the load failure."
        )
    return model


def is_loaded() -> bool:
    return _model is not None


def unload_model() -> None:
    """Drop the loaded model. Intended for tests and shutdown."""
    global _model
    with _lock:
        _model = None
