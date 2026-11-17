#!/usr/bin/env python3
"""Shared response envelopes and exception handling for GNN FastAPI apps."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Any, Dict, Literal

try:
    from fastapi import FastAPI, Request
    from fastapi.encoders import jsonable_encoder
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
except ImportError as exc:
    raise ImportError(
        "GNN API requires the api extra. Install generalized-notation-notation[api]."
    ) from exc

from pydantic import BaseModel, ConfigDict, Field

try:
    from starlette.exceptions import HTTPException as StarletteHTTPException
except ImportError as exc:
    raise ImportError(
        "GNN API requires the api extra. Install generalized-notation-notation[api]."
    ) from exc

logger = logging.getLogger(__name__)


class APIError(BaseModel):
    """Machine-readable API error details."""

    code: str
    message: str
    details: Any = None

    model_config = ConfigDict(extra="forbid")


class APIEnvelope(BaseModel):
    """Canonical JSON response shape for all non-streaming API endpoints."""

    status: Literal["success", "error"]
    data: Any = Field(default_factory=dict)
    error: APIError | None = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


def response_meta(**extra: Any) -> Dict[str, Any]:
    """Return standard response metadata with a timezone-aware timestamp."""
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        **extra,
    }


def success_envelope(data: Any = None, **meta: Any) -> APIEnvelope:
    """Build a successful canonical response envelope."""
    return APIEnvelope(
        status="success",
        data={} if data is None else data,
        error=None,
        meta=response_meta(**meta),
    )


def error_envelope(
    code: str,
    message: str,
    *,
    details: Any = None,
    **meta: Any,
) -> APIEnvelope:
    """Build a failed canonical response envelope."""
    return APIEnvelope(
        status="error",
        data={},
        error=APIError(code=code, message=message, details=details),
        meta=response_meta(**meta),
    )


def error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    details: Any = None,
    headers: Mapping[str, str] | None = None,
    **meta: Any,
) -> JSONResponse:
    """Return a JSON response using the canonical error envelope."""
    payload = error_envelope(
        code,
        message,
        details=jsonable_encoder(
            details,
            custom_encoder={
                bytes: lambda value: value.decode("utf-8", errors="backslashreplace")
            },
        ),
        status_code=status_code,
        **meta,
    )
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(payload),
        headers=headers,
    )


def _http_error_code(status_code: int) -> str:
    """Map common HTTP statuses to stable machine-readable codes."""
    return {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        422: "validation_error",
        429: "rate_limited",
    }.get(status_code, "http_error")


def install_exception_handlers(app: FastAPI) -> None:
    """Install canonical handlers for validation, HTTP, and internal errors."""

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return error_response(
            422,
            "validation_error",
            "Request validation failed",
            details=exc.errors(),
            path=request.url.path,
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        try:
            default_message = HTTPStatus(exc.status_code).phrase
        except ValueError:
            default_message = "HTTP error"
        message = exc.detail if isinstance(exc.detail, str) else default_message
        details = None if isinstance(exc.detail, str) else jsonable_encoder(exc.detail)
        return error_response(
            exc.status_code,
            _http_error_code(exc.status_code),
            message,
            details=details,
            headers=exc.headers,
            path=request.url.path,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled API error on %s", request.url.path, exc_info=exc)
        return error_response(
            500,
            "internal_error",
            "Internal server error",
            path=request.url.path,
        )


__all__ = [
    "APIEnvelope",
    "APIError",
    "error_envelope",
    "error_response",
    "install_exception_handlers",
    "response_meta",
    "success_envelope",
]
