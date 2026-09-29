#!/usr/bin/env python3
"""
Renderer Health Check — Reports importability of renderer generator modules.

Provides:
  - check_renderers(): returns Dict[str, RendererStatus]
  - RendererStatus: availability, version, import path

Behaviour with the framework registry:
  Frameworks that carry ``available=False`` in the registry (e.g. ``bnlearn``,
  which stays out of the default lock) are reported as unavailable *without*
  attempting an import.  This avoids confusing ``ImportError`` stack traces
  from modules that are intentionally absent.
"""

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .framework_registry import (
    get_framework_availability,
    get_supported_frameworks,
)

logger = logging.getLogger(__name__)


@dataclass
class RendererStatus:
    """Status of a single renderer."""

    name: str
    available: bool
    version: Optional[str] = None
    module_path: Optional[str] = None
    error: Optional[str] = None
    remediation: Optional[str] = None

    def to_dict(self) -> dict:
        """Provide to dict behavior."""
        return {
            "name": self.name,
            "available": self.available,
            "version": self.version,
            "module_path": self.module_path,
            "error": self.error,
            "remediation": self.remediation,
        }


# Renderer import paths derived from the canonical framework inventory.
_RENDERERS: dict[str, Any] = {
    name: f"gnn.render.{name}" for name in get_supported_frameworks()
}

_FRAMEWORK_REMEDIATIONS: dict[str, str] = {
    "julia": "Install Julia from https://julialang.org/downloads/ or run: sudo apt install julia",
    "rxinfer": "Install Julia and instantiate the committed env: julia --project=src/gnn/execute/rxinfer -e 'using Pkg; Pkg.instantiate()'",
    "activeinference_jl": "Install Julia and instantiate the committed env: julia --project=src/gnn/execute/activeinference_jl -e 'using Pkg; Pkg.instantiate()'",
    "pytorch": "Add torch: uv add torch",
    "bnlearn": "Add bnlearn (or R bnlearn bridge): uv add bnlearn",
    "numpyro": "Add numpyro: uv add numpyro",
    "stan": "Add cmdstanpy: uv add cmdstanpy",
    "discopy": "Add discopy: uv add discopy",
    "jax": "Add jax: uv add jax jaxlib",
    "pymdp": "Add pymdp: uv add pymdp",
    "ngclearn": "Add ngclearn: uv sync --extra ngclearn",
    "cpomdp": "Add cpomdp: uv sync --extra cpomdp",
}


def get_remediation(framework: str) -> Optional[str]:
    """Return the install-remediation hint for ``framework``, if any."""
    return _FRAMEWORK_REMEDIATIONS.get(framework)


def check_renderers() -> Dict[str, RendererStatus]:
    """
    Check importability of all renderer generator modules.

    Frameworks that are explicitly marked ``available=False`` in the canonical
    registry (see :data:`~render.framework_registry.FRAMEWORK_REGISTRY`) are
    reported as unavailable immediately, without attempting a Python import.
    This ensures the health check never produces a spurious ``ImportError`` for
    intentionally gated frameworks.

    Returns:
        Dict mapping renderer name → RendererStatus.
    """
    results: dict[Any, Any] = {}

    for name, module_path in _RENDERERS.items():
        remediation = _FRAMEWORK_REMEDIATIONS.get(name)
        # ── Registry-level gate ──────────────────────────────────────────
        available_in_registry, registry_reason = get_framework_availability(name)
        if not available_in_registry:
            results[name] = RendererStatus(
                name=name,
                available=False,
                error=(
                    f"Intentionally unavailable: {registry_reason}"
                    if registry_reason
                    else "Marked unavailable in framework registry"
                ),
                module_path=module_path,
                remediation=remediation,
            )
            continue

        # ── Import-level check ───────────────────────────────────────────
        try:
            mod = importlib.import_module(module_path)
            version = getattr(mod, "__version__", None)
            results[name] = RendererStatus(
                name=name,
                available=True,
                version=version,
                module_path=module_path,
                remediation=None,
            )
        except ImportError as e:
            results[name] = RendererStatus(
                name=name,
                available=False,
                error=str(e),
                module_path=module_path,
                remediation=remediation,
            )
        except Exception as e:
            results[name] = RendererStatus(
                name=name,
                available=False,
                error=f"Load error: {e}",
                module_path=module_path,
                remediation=remediation,
            )

    available = sum(1 for r in results.values() if r.available)
    total = len(results)
    logger.info(
        "🔧 Renderer generator modules: %s/%s importable",
        available,
        total,
    )

    return results


def get_available_frameworks() -> list:
    """Return list of available framework names."""
    return [name for name, status in check_renderers().items() if status.available]
