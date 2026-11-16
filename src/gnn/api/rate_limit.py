"""Per-client rate limiting middleware for the GNN API server.

Protects the FastAPI application from request floods by tracking each
client's request timestamps in an in-memory sliding 60-second window,
mirroring the dict-based limiter already used by the MCP HTTP server
(``mcp.server_http``). No Redis or external state is required.

Configuration
-------------
- ``GNN_RATE_LIMIT``: maximum requests per client per 60-second window.
  Default: ``60``. Set to ``0`` to disable rate limiting entirely.
  Invalid values fall back to the default (fail-safe: protection stays on).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Dict, List

try:
    from fastapi import Request
    from starlette.responses import Response
except ImportError as exc:
    raise ImportError(
        "GNN API requires the api extra. Install generalized-notation-notation[api]."
    ) from exc

from gnn.api.responses import error_response

logger = logging.getLogger(__name__)

ENV_RATE_LIMIT = "GNN_RATE_LIMIT"
DEFAULT_RATE_LIMIT = 60
RATE_LIMIT_WINDOW_SECONDS = 60.0
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_STATE: Dict[str, List[float]] = {}


def get_rate_limit() -> int:
    """Return the configured per-client rate limit, or 0 when disabled."""
    raw_value = os.environ.get(ENV_RATE_LIMIT, str(DEFAULT_RATE_LIMIT))
    try:
        return max(0, int(raw_value))
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to default %d",
            ENV_RATE_LIMIT,
            raw_value,
            DEFAULT_RATE_LIMIT,
        )
        return DEFAULT_RATE_LIMIT


def is_rate_limited(client_id: str, *, now: float | None = None) -> bool:
    """Return True when ``client_id`` has exceeded the configured rate limit."""
    limit = get_rate_limit()
    if limit <= 0:
        return False
    timestamp = time.time() if now is None else now
    cutoff = timestamp - RATE_LIMIT_WINDOW_SECONDS
    with _RATE_LIMIT_LOCK:
        recent = [
            seen_at
            for seen_at in _RATE_LIMIT_STATE.get(client_id, [])
            if seen_at >= cutoff
        ]
        if len(recent) >= limit:
            _RATE_LIMIT_STATE[client_id] = recent
            return True
        recent.append(timestamp)
        _RATE_LIMIT_STATE[client_id] = recent
    return False


def get_client_id(request: Request) -> str:
    """Resolve the per-client identity for rate limiting.

    Prefers the first (leftmost) address in ``X-Forwarded-For`` when present,
    falling back to the direct peer address (``request.client.host``).
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    client = request.client
    if client is not None and client.host:
        return client.host
    return "unknown"


async def rate_limit_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """ASGI middleware rejecting clients that exceed ``GNN_RATE_LIMIT``.

    Registered outermost in the middleware stack so it applies even when
    API-key auth is disabled, preventing DoS on the local research surface.
    """
    if is_rate_limited(get_client_id(request)):
        return error_response(
            429,
            "rate_limited",
            "Rate limit exceeded. Try again later.",
            headers={"Retry-After": str(int(RATE_LIMIT_WINDOW_SECONDS))},
            path=request.url.path,
        )
    return await call_next(request)


__all__ = [
    "ENV_RATE_LIMIT",
    "DEFAULT_RATE_LIMIT",
    "RATE_LIMIT_WINDOW_SECONDS",
    "get_rate_limit",
    "is_rate_limited",
    "get_client_id",
    "rate_limit_middleware",
]
