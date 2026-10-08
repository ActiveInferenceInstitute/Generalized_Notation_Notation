"""
JAX Renderer for GNN Specifications

Implements rendering of GNN models to JAX code for POMDPs and related Active Inference models.
Leverages JAX's JIT, vmap, pmap, and supports Optax/Flax integration.

@Web: https://github.com/google/jax
@Web: https://optax.readthedocs.io
@Web: https://flax.readthedocs.io
@Web: https://pfjax.readthedocs.io
@Web: https://juliapomdp.github.io/POMDPs.jl/latest/def_pomdp/
@Web: https://arxiv.org/abs/1304.1118
@Web: https://www.cs.cmu.edu/~ggordon/jpineau-ggordon-thrun.ijcai03.pdf
"""

import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np

from gnn.render.pomdp_contract import build_canonical_pomdp_spec

logger = logging.getLogger(__name__)


def _render_to_path(
    generator_fn: Callable[[Dict[str, Any], Optional[Dict[str, Any]]], str],
    label: str,
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]],
) -> Tuple[bool, str, List[str]]:
    """Shared scaffold: generate code, mkdir, write, return (success, msg, paths)."""
    try:
        code = generator_fn(gnn_spec, options)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        logger.info(f"{label} code written to {output_path}")
        return True, f"{label} generated successfully.", [str(output_path)]
    except Exception as e:
        logger.error(f"Failed to render GNN to {label}: {e}")
        return False, f"{label} rendering failed: {e}", []


def render_gnn_to_jax(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]:
    """Render a GNN specification to a general JAX model implementation.

    Kronecker-factorized specs (``A_fN``/``B_fN`` per-factor matrices) route
    to the native factorized generator (MAJ-02): a standalone script that
    runs sparse mean-field active inference without materialising the joint
    state space.

    @Web: https://github.com/google/jax
    @Web: https://flax.readthedocs.io
    """
    from gnn.render.continuous_common import (
        extract_continuous_spec,
        extract_factored_continuous_spec,
        is_continuous_spec,
    )
    from gnn.render.execution_contracts import (
        execution_contract,
        generate_contract_script,
    )

    if execution_contract(gnn_spec) is not None:
        return _render_to_path(
            generate_contract_script,
            "JAX execution contract",
            gnn_spec,
            output_path,
            options,
        )

    # Refuse an explicitly malformed parameter collection before classifiers
    # inspect its keys. Omitted collections still retain existing defaults.
    if "initialparameterization" in gnn_spec and not isinstance(
        gnn_spec["initialparameterization"], dict
    ):
        return (
            False,
            "JAX model rendering failed: initialparameterization must be a dictionary",
            [],
        )

    if is_continuous_spec(gnn_spec):
        # Continuous-state (linear-Gaussian) branch: no A/B/C/D exist, so the
        # discrete extractors below must never run on this path.
        from gnn.render.continuous_script import (
            generate_continuous_script,
            generate_factored_continuous_script,
        )
        from gnn.render.pomdp_contract import ModelKind, detect_model_kinds

        if detect_model_kinds(gnn_spec) == frozenset(
            {ModelKind.MULTI_AGENT, ModelKind.CONTINUOUS}
        ):
            from gnn.render.multi_agent_continuous import (
                generate_multi_agent_continuous_script,
            )

            return _render_to_path(
                lambda spec, _opts: generate_multi_agent_continuous_script(spec, "jax"),
                "JAX native independent continuous agents",
                gnn_spec,
                output_path,
                options,
            )

        if detect_model_kinds(gnn_spec) == frozenset(
            {ModelKind.FACTORED, ModelKind.CONTINUOUS}
        ):
            # Per-factor LGSSM path: independent per-factor blocks rendered by
            # the factored generator; the flat extractor must never see them.
            return _render_to_path(
                lambda spec, _opts: generate_factored_continuous_script(
                    extract_factored_continuous_spec(spec), "jax"
                ),
                "JAX factored-continuous LGSSM (per-factor)",
                gnn_spec,
                output_path,
                options,
            )

        return _render_to_path(
            lambda spec, _opts: generate_continuous_script(
                extract_continuous_spec(spec), "jax"
            ),
            "JAX continuous LGSSM",
            gnn_spec,
            output_path,
            options,
        )
    if _is_factorized_spec(gnn_spec):
        return render_gnn_to_jax_factorized(gnn_spec, output_path, options)
    return _render_to_path(
        _generate_jax_model_code, "JAX model", gnn_spec, output_path, options
    )


def render_gnn_to_jax_factorized(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]:
    """Render a Kronecker-factorized GNN spec to a sparse JAX script.

    The emitted script embeds the per-factor ``A_fN``/``B_fN``/``C_fN``/
    ``D_fN`` matrices and drives ``execute.jax.kronecker_factorized``
    (``run_factorized_active_inference``), which never builds the joint state
    space (``joint_state_space_size = prod(factor_sizes)`` is reported but not
    allocated). Results are written as ``jax_kronecker_factorized_v1``
    ``simulation_results.json`` under ``GNN_OUTPUT_DIR`` so Step 12 collects
    and Step 16 (analysis) consumes them.
    """
    from gnn.render.execution_contracts import execution_contract

    if execution_contract(gnn_spec) is not None:
        return render_gnn_to_jax(gnn_spec, output_path, options)
    return _render_to_path(
        _generate_jax_factorized_code,
        "JAX Kronecker-factorized",
        gnn_spec,
        output_path,
        options,
    )


def render_gnn_to_jax_pomdp(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]:
    """Render a GNN POMDP specification to a JAX POMDP solver implementation.

    @Web: https://pfjax.readthedocs.io
    @Web: https://arxiv.org/abs/1304.1118
    @Web: https://www.cs.cmu.edu/~ggordon/jpineau-ggordon-thrun.ijcai03.pdf
    """
    from gnn.render.execution_contracts import execution_contract

    if execution_contract(gnn_spec) is not None:
        return render_gnn_to_jax(gnn_spec, output_path, options)
    return _render_to_path(
        _generate_jax_pomdp_code, "JAX POMDP", gnn_spec, output_path, options
    )


def render_gnn_to_jax_combined(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]:
    """Render a GNN specification to a combined JAX implementation (hierarchical, multi-agent, or continuous).

    @Web: https://github.com/google/jax
    @Web: https://optax.readthedocs.io
    """
    from gnn.render.execution_contracts import execution_contract

    if execution_contract(gnn_spec) is not None:
        return render_gnn_to_jax(gnn_spec, output_path, options)
    return _render_to_path(
        _generate_jax_combined_code, "JAX combined", gnn_spec, output_path, options
    )


from .jax_combined_generator import (
    _generate_jax_combined_code,
)
from .jax_factorized_generator import (
    _FACTOR_MATRIX_RE,
    _canonicalise_factor_b,
    _factor_action_count,
    _factor_matrix_groups,
    _generate_jax_factorized_code,
    _is_factorized_spec,
)
from .jax_model_generator import (
    EFE_CONVENTION_JAX,
    EFE_CONVENTION_JAX_JSON,
    EFE_CONVENTION_JAX_JSON_LITERAL,
    _generate_jax_model_code,
    _json_dumps,
)
from .jax_pomdp_generator import (
    _generate_jax_pomdp_code,
)
from .jax_spec_extract import (
    _create_fallback_matrix,
    _create_improved_default_matrix,
    _extract_gnn_matrices,
    _infer_matrix_from_context,
    _jax_model_name,
    _parse_gnn_matrix_string,
    _parse_matrix_string,
    _parse_vector_string,
    _validated_jax_matrices,
)
