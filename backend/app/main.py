"""FastAPI application for sentiment inference.

Inference only: the fine-tuned checkpoint is loaded from disk at startup and
reused for every request. Nothing in this app trains or downloads models.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import settings
from app.models.model_loader import ModelLoadError, init_model
from app.services.inference import reset_sentiment_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the fine-tuned checkpoint once, here — never per request.
    try:
        init_model()
    except ModelLoadError:
        logger.exception("Startup aborted: the sentiment model could not be loaded")
        raise
    yield
    reset_sentiment_service()


app = FastAPI(title=settings.app_name, lifespan=lifespan)

# CORS for the Next.js frontend (http://localhost:3000 by default).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(ModelLoadError)
async def model_unavailable_handler(request: Request, exc: ModelLoadError) -> JSONResponse:
    """The model could not be loaded — report it as unavailable, not as a bug."""
    logger.error("Request rejected, sentiment model unavailable: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Sentiment model is not available."},
    )


app.include_router(router)
