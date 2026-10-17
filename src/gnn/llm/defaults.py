"""
Central defaults for local Ollama usage.

Override precedence (highest first):
``OLLAMA_MODEL`` env → ``input/config.yaml`` key ``llm.model`` (read in
``llm.processor``) → this constant.
"""

import os
from typing import Any

DEFAULT_OLLAMA_MODEL = "smollm2:135m-instruct-q4_K_S"


def ollama_generation_options(
    options: dict[str, Any] | None = None, *, inherit_environment: bool = True
) -> dict[str, int]:
    """Freeze optional local context/seed settings without contacting a daemon."""
    supplied = options or {}
    unknown = set(supplied) - {"num_ctx", "seed"}
    if unknown:
        raise ValueError(f"Unknown Ollama generation options: {sorted(unknown)}")
    resolved = {}
    for name, variable in (("num_ctx", "OLLAMA_NUM_CTX"), ("seed", "OLLAMA_SEED")):
        value = supplied.get(name, os.getenv(variable) if inherit_environment else None)
        if value is None:
            continue
        if isinstance(value, str) and value.lstrip("-").isdecimal():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Ollama {name} must be an integer")
        if name == "num_ctx" and value <= 0:
            raise ValueError("Ollama num_ctx must be positive")
        resolved[name] = value
    return resolved
