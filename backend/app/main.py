"""FastAPI application for sentiment inference.

Inference only: the fine-tuned checkpoint is loaded from disk at startup and
reused for every request. Nothing in this app trains or downloads models.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from app.errors import ModelLoadError, register_exception_handlers
from app.models.model_loader import init_model
from app.services.sentiment_service import reset_sentiment_service

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


app = FastAPI(title=settings.app_name, lifespan=lifespan, debug=False)

register_exception_handlers(app)

# CORS for the Next.js frontend (http://localhost:3000 by default).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
