#!/usr/bin/env python3
"""
Render Module

This module provides rendering capabilities for GNN specifications to various
target languages and simulation environments.
"""

# Phase 6: render submodules are in-tree; unconditional imports.
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .activeinference_jl import render_gnn_to_activeinference_jl
from .discopy import render_gnn_to_discopy
from .framework_registry import get_supported_frameworks
from .generators import (
    generate_activeinference_jl_code,
    generate_discopy_code,
    generate_pymdp_code,
    generate_rxinfer_code,
)
from .numpyro import render_gnn_to_numpyro
from .pomdp_processor import POMDPRenderProcessor, process_pomdp_for_frameworks
from .processor import (
    find_render_artifact,
    get_available_renderers,
    get_module_info,
    process_render,
    render_gnn_spec,
    render_processing_succeeded,
)
from .pymdp import render_gnn_to_pymdp
from .pymdp.pymdp_renderer import PyMDPRenderer
from .pytorch import render_gnn_to_pytorch
from .rxinfer import render_gnn_to_rxinfer
from .stan import render_stan


class JAXRenderer:
    """Facade over ``render_gnn_to_jax`` exposed as a class for callers that
    want polymorphic dispatch. The real rendering work is in
    ``render/jax/jax_renderer.py`` — this class forwards ``render`` to it."""

    def render(
        self,
        spec: Dict[str, Any],
        output_path: Path,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, List[str]]:
        """Render operation."""
        from .jax.jax_renderer import render_gnn_to_jax

        return render_gnn_to_jax(spec, output_path, options)


def validate_render(result: Any, framework: Any = None) -> Any:
    """Validate render output.

    Args:
        result: The render result to validate.
        framework: Optional framework name for framework-specific validation.

    Returns:
        True if validation passes.

    Raises:
        ValueError: If validation fails.
    """
    if result is None:
        raise ValueError("Render result is None")
    if isinstance(result, str) and len(result) == 0:
        raise ValueError("Render result is empty string")
    return True


__all__: list[Any] = [
    # Core functions
    "process_render",
    "render_gnn_spec",
    "get_module_info",
    "get_available_renderers",
    # Generator functions
    "generate_pymdp_code",
    "generate_rxinfer_code",
    "generate_activeinference_jl_code",
    "generate_discopy_code",
    # Specific renderer functions
    "render_gnn_to_pymdp",
    "render_gnn_to_rxinfer",
    "render_gnn_to_discopy",
    "render_gnn_to_activeinference_jl",
    "render_gnn_to_pytorch",
    "render_gnn_to_numpyro",
    "render_stan",
    # Renderer classes
    "PyMDPRenderer",
    "JAXRenderer",
    # POMDP processing
    "POMDPRenderProcessor",
    "process_pomdp_for_frameworks",
    # Utility functions
    "find_render_artifact",
    "render_processing_succeeded",
    "get_supported_frameworks",
    "validate_render",
]


from gnn import __version__

FEATURES: dict[str, Any] = {
    "pymdp_rendering": True,
    "rxinfer_rendering": True,
    "activeinference_jl_rendering": True,
    "discopy_rendering": True,
    "jax_rendering": True,
    "pytorch_rendering": True,
    "numpyro_rendering": True,
    "stan_rendering": True,
    "mcp_integration": True,
    "pomdp_processing": True,
    "state_space_extraction": True,
    "modular_injection": True,
    "framework_specific_outputs": True,
    "structured_documentation": True,
}

from .render import main  # expose CLI entry as attribute for tests
