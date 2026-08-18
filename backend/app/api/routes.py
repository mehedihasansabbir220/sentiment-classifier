"""HTTP routes.

Routes only validate input, delegate, and shape the response. All model and
tensor work lives in :mod:`app.services.sentiment_service`. Failures bubble
as :class:`~app.errors.AppError` subclasses and are turned into structured
JSON by the handlers in :mod:`app.errors`.
"""

import logging

from fastapi import APIRouter, Depends

from app.errors import AppError
from app.schemas import HealthResponse, PredictRequest, PredictResponse
from app.services.sentiment_service import SentimentService, get_sentiment_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(
    payload: PredictRequest,
    service: SentimentService = Depends(get_sentiment_service),
) -> PredictResponse:
    """Classify a single text as positive or negative."""
    try:
        result = service.predict(payload.text)
    except AppError:
        raise
    except Exception:
        logger.exception("Unexpected error during prediction")
        raise AppError() from None

    return PredictResponse(
        text=result.text,
        sentiment=result.sentiment,
        confidence=result.confidence,
        probabilities=result.probabilities,
    )
