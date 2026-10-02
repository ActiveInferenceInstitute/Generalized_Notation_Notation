"""Freeze noncredential endpoint identity for requests, caches and resume."""

from __future__ import annotations

import hashlib
import json
import os
from urllib.parse import urlsplit

from .providers.base_provider import ProviderType


def resolved_endpoint(provider: ProviderType) -> tuple[str, str]:
    """Return the actual endpoint and a canonical hash without credential data."""
    defaults = {
        ProviderType.OLLAMA: ("OLLAMA_HOST", "http://127.0.0.1:11434"),
        ProviderType.OPENAI: ("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        ProviderType.OPENROUTER: (None, "https://openrouter.ai/api/v1"),
        ProviderType.PERPLEXITY: (None, "https://api.perplexity.ai"),
    }
    variable, default = defaults[provider]
    raw = (os.getenv(variable) if variable else None) or default
    endpoint = raw if "://" in raw else f"http://{raw}"
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("LLM endpoint must be HTTP(S) without URL credentials")
    identity = {
        "scheme": parsed.scheme,
        "host": parsed.hostname.lower(),
        "port": parsed.port or (443 if parsed.scheme == "https" else 80),
        "path": parsed.path.rstrip("/"),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return endpoint, digest
