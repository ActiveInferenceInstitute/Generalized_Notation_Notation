"""Lazy scientific adapters for explicitly selected THRML sampling results."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["adapt_result", "analyze_payload", "generate_analysis_from_logs"]


def __getattr__(name: str) -> Any:
    """Load core analysis helpers without importing THRML or JAX."""
    if name not in __all__:
        raise AttributeError(name)
    module = "adapter" if name == "adapt_result" else "analyzer"
    return getattr(import_module(f"{__name__}.{module}"), name)
