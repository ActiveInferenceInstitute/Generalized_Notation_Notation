"""Optional API-key authentication for the local GNN API server.

The API is documented as a local research tool with no authentication. This
module turns that from a hard-coded property into a configurable boundary:

- ``GNN_API_KEY`` unset  → authentication disabled (loopback research use);
  ``is_auth_enabled()`` returns ``False``.
- ``GNN_API_KEY`` set    → every route except ``/docs``, ``/redoc``,
  ``/openapi.json``, and ``/api/v1/health`` requires a matching ``X-API-Key``
  header. Comparisons use ``hmac.compare_digest`` (constant-time).

A server bound to a non-loopback address while authentication is disabled is
refused by ``run_server`` unless ``GNN_ALLOW_INSECURE_BIND=1`` is set.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Awaitable, Callable
from typing import Optional

try:
    from fastapi import Request
    from starlette.responses import Response
except ImportError as exc:
    raise ImportError(
        "GNN API requires the api extra. Install generalized-notation-notation[api]."
    ) from exc

from gnn.api.responses import error_response

ENV_API_KEY = "GNN_API_KEY"
ENV_ALLOW_INSECURE_BIND = "GNN_ALLOW_INSECURE_BIND"

#: Routes that never require an API key (public introspection/health surface).
_PUBLIC_PATHS = frozenset(
    {
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/health",
    }
)

#: Loopback hosts that are always acceptable to bind without authentication.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def get_api_key() -> Optional[str]:
    """Return the configured API key, or ``None`` when unset/blank."""
    key = os.environ.get(ENV_API_KEY, "").strip()
    return key or None


def is_auth_enabled() -> bool:
    """Whether API-key authentication is active (a key is configured)."""
    return get_api_key() is not None


def _keys_match(provided: str, expected: str) -> bool:
    """Constant-time comparison of a provided key against the expected key."""
    return hmac.compare_digest(provided.encode(), expected.encode())


def key_matches(provided: Optional[str]) -> bool:
    """Return True if ``provided`` satisfies the configured key.

    When authentication is disabled this always returns True (no-op boundary).
    """
    if not is_auth_enabled():
        return True
    expected = get_api_key()
    if not provided or expected is None:
        return False
    return _keys_match(provided, expected)


async def api_key_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """ASGI middleware enforcing ``X-API-Key`` outside the public path set."""
    if request.url.path in _PUBLIC_PATHS:
        return await call_next(request)
    if not is_auth_enabled():
        return await call_next(request)
    provided = request.headers.get("x-api-key")
    if not key_matches(provided):
        return error_response(
            401,
            "unauthorized",
            "Missing or invalid X-API-Key",
            path=request.url.path,
        )
    return await call_next(request)


def require_secure_bind(host: str) -> bool:
    """Return True if binding to ``host`` is acceptable under current auth state.

    Non-loopback binds are rejected unless authentication is enabled or the
    operator has explicitly set ``GNN_ALLOW_INSECURE_BIND=1``.
    """
    if host in _LOOPBACK_HOSTS:
        return True
    if is_auth_enabled():
        return True
    allow = os.environ.get(ENV_ALLOW_INSECURE_BIND, "").strip().lower()
    return allow in ("1", "true", "yes")


__all__ = [
    "ENV_API_KEY",
    "ENV_ALLOW_INSECURE_BIND",
    "get_api_key",
    "is_auth_enabled",
    "key_matches",
    "api_key_middleware",
    "require_secure_bind",
]
