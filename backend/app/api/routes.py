"""HTTP routes.

Routes only validate input, delegate, and shape the response. All model and
tensor work lives in :mod:`app.services.inference`.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas import HealthResponse, PredictRequest, PredictResponse
from app.services.inference import InferenceError, SentimentService, get_sentiment_service

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
    # A missing model raises ModelLoadError from the dependency above and is
    # turned into a 503 by the handler registered in app.main.
    try:
        result = service.predict(payload.text)
    except InferenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run inference on the provided text.",
        ) from exc

    return PredictResponse(
        text=result.text,
        sentiment=result.sentiment,
        confidence=result.confidence,
        probabilities=result.probabilities,
    )
