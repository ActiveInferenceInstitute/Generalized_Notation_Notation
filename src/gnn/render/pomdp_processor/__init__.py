#!/usr/bin/env python3
"""
POMDP Processor for Render Module

This module provides specialized processing capabilities for injecting POMDP state spaces
into various rendering implementations (PyMDP, RxInfer, ActiveInference.jl, etc.).

Numeric/matrix helpers live in ``pomdp_math``; generic code-metrics counting
lives in ``utils.config_io.code_metrics``.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from gnn.extract.pomdp_extractor import POMDPStateSpace

logger = logging.getLogger(__name__)

from gnn.render.framework_registry import get_pomdp_framework_configs
from gnn.render.naming import safe_output_stem
from gnn.render.pomdp_contract import (
    ModelKind,
    build_canonical_pomdp_spec,
    detect_pomdp_space_model_kind,
    detect_pomdp_space_model_kinds,
    unsupported_composition_reason,
    unsupported_nonstationary_reason,
)
from gnn.render.pomdp_math import (
    _factor_action_counts,
    _is_kronecker_factorized_spec,
    _mixed_radix_digit,
    _normalise_columns,
    _normalise_prob_vector,
)
from gnn.utils.config_io.code_metrics import count_code_metrics

from ._canonical import _CanonicalSpecMixin
from ._documentation import _DocumentationMixin
from ._flow import _ProcessFlowMixin
from ._renderer_dispatch import _RendererDispatchMixin
from ._routes import _continuous_shape, _safe_output_stem, RendererRoute, RENDERER_ROUTES
from ._spec_generation import _SpecGenerationMixin
from ._validation import _CompatibilityValidationMixin


class POMDPRenderProcessor(
    _ProcessFlowMixin,
    _CompatibilityValidationMixin,
    _CanonicalSpecMixin,
    _SpecGenerationMixin,
    _RendererDispatchMixin,
    _DocumentationMixin,
):
    """
    Processes POMDP state spaces and injects them into framework-specific renderers.

    Features:
    - Modular injection of POMDP state spaces into renderers
    - Framework-specific output directory management
    - Structured approach to render coordination
    - Validation of POMDP-renderer compatibility
    """


def pomdp_to_gnn_spec(pomdp_space: "POMDPStateSpace", **kwargs: Any) -> Dict[str, Any]:
    """Public spec conversion: POMDP state space -> renderer GNN spec dict.

    Callers outside the render package (e.g. analysis-side cross-framework
    execution) must use this instead of reaching into
    ``POMDPRenderProcessor._pomdp_to_gnn_spec``. Conversion is pure — no
    output directory is created or written.
    """
    return POMDPRenderProcessor(Path("."))._pomdp_to_gnn_spec(pomdp_space, **kwargs)


def process_pomdp_for_frameworks(
    pomdp_space: "POMDPStateSpace",
    output_dir: Union[str, Path],
    frameworks: Optional[List[str]] = None,
    gnn_file_path: Optional[Path] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Convenience function to process POMDP for multiple frameworks.

    Args:
        pomdp_space: POMDP state space data
        output_dir: Base output directory
        frameworks: List of frameworks to process (default: all)
        gnn_file_path: Original GNN file path
        **kwargs: Additional processing options

    Returns:
        Processing results dictionary
    """
    processor = POMDPRenderProcessor(Path(output_dir))
    return processor.process_pomdp_for_all_frameworks(
        pomdp_space, gnn_file_path, frameworks, **kwargs
    )
