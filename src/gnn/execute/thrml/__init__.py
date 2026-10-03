"""Lazy public execution surface for the experimental THRML backend."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "execute_thrml_script",
    "find_thrml_scripts",
    "is_thrml_available",
    "run_thrml_records",
    "run_thrml_scripts",
    "validate_native_result",
]


def __getattr__(name: str) -> Any:
    """Import orchestration helpers only when explicitly requested."""
    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.thrml_runner"), name)
