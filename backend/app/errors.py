"""Structured API errors.

Technical detail belongs in logs. Clients only ever see ``{error: {code, message}}``.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class ErrorCode(StrEnum):
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    TOKENIZER_FAILURE = "TOKENIZER_FAILURE"
    MODEL_LOAD_FAILURE = "MODEL_LOAD_FAILURE"
    INVALID_REQUEST = "INVALID_REQUEST"
    EMPTY_TEXT = "EMPTY_TEXT"
    TEXT_TOO_LONG = "TEXT_TOO_LONG"
    INFERENCE_FAILURE = "INFERENCE_FAILURE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_USER_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.MODEL_NOT_FOUND: "The sentiment model is not available.",
    ErrorCode.TOKENIZER_FAILURE: "The sentiment model is not available.",
    ErrorCode.MODEL_LOAD_FAILURE: "The sentiment model is not available.",
    ErrorCode.INVALID_REQUEST: "The request is invalid. Send JSON with a non-empty text field.",
    ErrorCode.EMPTY_TEXT: "Text must not be empty.",
    ErrorCode.TEXT_TOO_LONG: "Text is too long. Keep it under 10,000 characters.",
    ErrorCode.INFERENCE_FAILURE: "Failed to analyze this text. Please try again.",
    ErrorCode.INTERNAL_ERROR: "An unexpected error occurred. Please try again.",
}

_HTTP_FALLBACKS: dict[int, tuple[ErrorCode, str]] = {
    status.HTTP_400_BAD_REQUEST: (ErrorCode.INVALID_REQUEST, _USER_MESSAGES[ErrorCode.INVALID_REQUEST]),
    status.HTTP_404_NOT_FOUND: (ErrorCode.INVALID_REQUEST, "That endpoint was not found."),
    status.HTTP_405_METHOD_NOT_ALLOWED: (ErrorCode.INVALID_REQUEST, "This method is not allowed."),
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: (ErrorCode.INVALID_REQUEST, _USER_MESSAGES[ErrorCode.INVALID_REQUEST]),
    status.HTTP_422_UNPROCESSABLE_CONTENT: (ErrorCode.INVALID_REQUEST, _USER_MESSAGES[ErrorCode.INVALID_REQUEST]),
    status.HTTP_503_SERVICE_UNAVAILABLE: (
        ErrorCode.MODEL_LOAD_FAILURE,
        _USER_MESSAGES[ErrorCode.MODEL_LOAD_FAILURE],
    ),
}


def error_payload(code: ErrorCode, message: str | None = None) -> dict[str, dict[str, str]]:
    return {"error": {"code": code.value, "message": message or _USER_MESSAGES[code]}}


class AppError(Exception):
    """Domain error that maps to a structured JSON response."""

    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    user_message: str = _USER_MESSAGES[ErrorCode.INTERNAL_ERROR]

    def __init__(self, log_message: str | None = None) -> None:
        self.log_message = log_message or self.user_message
        super().__init__(self.log_message)

    def to_payload(self) -> dict[str, dict[str, str]]:
        return error_payload(self.code, self.user_message)


class ModelLoadError(AppError):
    """Checkpoint exists but could not be loaded (config, labels, or weights)."""

    code = ErrorCode.MODEL_LOAD_FAILURE
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    user_message = _USER_MESSAGES[ErrorCode.MODEL_LOAD_FAILURE]


class ModelNotFoundError(ModelLoadError):
    code = ErrorCode.MODEL_NOT_FOUND
    user_message = _USER_MESSAGES[ErrorCode.MODEL_NOT_FOUND]


class TokenizerLoadError(ModelLoadError):
    code = ErrorCode.TOKENIZER_FAILURE
    user_message = _USER_MESSAGES[ErrorCode.TOKENIZER_FAILURE]


class InvalidTextError(AppError, ValueError):
    """Input the model cannot classify."""

    code = ErrorCode.INVALID_REQUEST
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    user_message = _USER_MESSAGES[ErrorCode.INVALID_REQUEST]


class EmptyTextError(InvalidTextError):
    code = ErrorCode.EMPTY_TEXT
    user_message = _USER_MESSAGES[ErrorCode.EMPTY_TEXT]


class TextTooLongError(InvalidTextError):
    code = ErrorCode.TEXT_TOO_LONG
    user_message = _USER_MESSAGES[ErrorCode.TEXT_TOO_LONG]


class InferenceError(AppError):
    code = ErrorCode.INFERENCE_FAILURE
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    user_message = _USER_MESSAGES[ErrorCode.INFERENCE_FAILURE]


def _validation_code(exc: RequestValidationError) -> ErrorCode:
    errors = exc.errors()
    types = {str(error.get("type", "")) for error in errors}
    text_types = {
        str(error.get("type", "")) for error in errors if "text" in error.get("loc", ())
    }

    if any("string_too_long" in error_type for error_type in types):
        return ErrorCode.TEXT_TOO_LONG
    if text_types & {"string_too_short", "value_error"}:
        return ErrorCode.EMPTY_TEXT
    return ErrorCode.INVALID_REQUEST


def _safe_validation_summary(exc: RequestValidationError) -> list[dict[str, object]]:
    """Log loc/type only — never the submitted text or traceback."""
    return [{"type": error.get("type"), "loc": list(error.get("loc", ()))} for error in exc.errors()]


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.error(
        "%s on %s %s: %s",
        exc.code,
        request.method,
        request.url.path,
        exc.log_message,
    )
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload())


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    code = _validation_code(exc)
    logger.info(
        "Invalid request on %s %s: %s",
        request.method,
        request.url.path,
        _safe_validation_summary(exc),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=error_payload(code),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    logger.warning(
        "HTTP %s on %s %s",
        exc.status_code,
        request.method,
        request.url.path,
    )
    code, message = _HTTP_FALLBACKS.get(
        exc.status_code,
        (ErrorCode.INTERNAL_ERROR, _USER_MESSAGES[ErrorCode.INTERNAL_ERROR]),
    )
    if exc.status_code >= 500 and exc.status_code != status.HTTP_503_SERVICE_UNAVAILABLE:
        code, message = ErrorCode.INTERNAL_ERROR, _USER_MESSAGES[ErrorCode.INTERNAL_ERROR]
    return JSONResponse(status_code=exc.status_code, content=error_payload(code, message))


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_payload(ErrorCode.INTERNAL_ERROR),
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
