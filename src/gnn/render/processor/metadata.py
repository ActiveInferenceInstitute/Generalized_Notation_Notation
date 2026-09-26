#!/usr/bin/env python3
"""Module metadata for the render processor package."""

from typing import Any, Dict

from gnn.render.framework_registry import (
    get_available_renderers as _registry_get_available_renderers,
)
from gnn.render.framework_registry import (
    get_supported_frameworks as _registry_get_supported_frameworks,
)


def get_module_info() -> Dict[str, Any]:
    """Get information about the render module."""
    # Import here (not at module top) to avoid circular init when this module is
    # imported before ``render/__init__.py`` finishes loading.
    try:
        from gnn import __version__ as _version
    except Exception:  # noqa: BLE001
        _version = "unknown"
    return {
        "name": "Render Module",
        "version": _version,
        "description": "POMDP-aware code generation for GNN specifications",
        "supported_targets": _registry_get_supported_frameworks(),
        "available_targets": _registry_get_supported_frameworks(),
        "features": [
            "POMDP state space extraction",
            "Modular framework injection",
            "Implementation-specific output directories",
            "PyMDP code generation",
            "RxInfer.jl code generation",
            "ActiveInference.jl code generation",
            "JAX code generation",
            "DisCoPy categorical diagram generation",
            "PyTorch code generation",
            "NumPyro code generation",
            "Stan model generation",
            "bnlearn bayesian network generation",
            "Structured documentation generation",
        ],
        "supported_formats": ["python", "julia", "stan", "python_script"],
        "processing_modes": ["basic", "pomdp_aware"],
    }


def get_available_renderers() -> Dict[str, Dict[str, Any]]:
    """Get information about available renderers (delegates to the registry)."""
    return _registry_get_available_renderers()
