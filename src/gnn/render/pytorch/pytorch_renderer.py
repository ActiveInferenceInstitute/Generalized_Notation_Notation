#!/usr/bin/env python3
"""
PyTorch Renderer for GNN Specifications

Renders GNN POMDP models to standalone PyTorch simulation scripts.
Discrete models emit ``torch.nn.Module`` components named after the GNN
variables they compute (see ``discrete_script.py`` and ``sites.py``), driven
by one seeded ``torch.Generator``; the same rollout can be traced op by op
with TorchLens when ``GNN_TORCHLENS_CAPTURE`` is set at execution time.

@Web: https://pytorch.org/docs/stable/
"""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gnn.render.naming import atomic_write_text
from gnn.render.pomdp_contract import normalise_vector
from gnn.render.pytorch.discrete_script import generate_discrete_script
from gnn.render.pytorch.sites import declared_connections, gnn_variable_sites
from gnn.render.spec_matrices import (
    extract_abcd_matrices,
    format_array_literal,
    parse_gnn_matrix_value,
)

logger = logging.getLogger(__name__)


def render_gnn_to_pytorch(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]:
    """Render a GNN specification to a PyTorch POMDP simulation script.

    Args:
        gnn_spec: Parsed GNN model specification.
        output_path: Path to write the generated PyTorch code.
        options: Optional rendering options.

    Returns:
        Tuple of (success: bool, message: str, artifact_paths: List[str])
    """
    from gnn.render.execution_contracts import unsupported_contract

    refusal = unsupported_contract(gnn_spec, "pytorch")
    if refusal:
        return False, refusal, []

    try:
        model_name = gnn_spec.get("modelName", "pytorch_pomdp")
        logger.info(f"Rendering GNN spec to PyTorch: {model_name}")

        # Continuous-state (LGSSM) branch: no A/B/C/D exist on this path, so
        # it must never reach the discrete extractors below.
        if _is_continuous_spec(gnn_spec):
            from gnn.render.pomdp_contract import ModelKind, detect_model_kinds

            if detect_model_kinds(gnn_spec) == frozenset(
                {ModelKind.FACTORED, ModelKind.CONTINUOUS}
            ):
                return (
                    False,
                    "unsupported-factored-continuous: pytorch renders the flat "
                    "linear-Gaussian family only; per-factor compositions are "
                    "refused rather than silently rendered flat",
                    [],
                )
            return _render_continuous(gnn_spec, Path(output_path), options, model_name)

        # Extract matrices
        provenance = deepcopy(gnn_spec.get("matrix_provenance", {}))
        A, B, C, D = _extract_matrices(gnn_spec, provenance=provenance)

        # Validate shapes
        from gnn.render.matrix_utils import validate_abcd_shapes

        valid, msg = validate_abcd_shapes(A, B, C, D)
        if not valid:
            logger.warning(f"Shape validation warning: {msg}")

        num_actions = B.shape[2] if B.ndim == 3 else 2
        habit, habit_source = _extract_habit(gnn_spec, num_actions)

        # Generate code
        code = _generate_pytorch_code(
            model_name,
            A,
            B,
            C,
            D,
            _extract_num_timesteps(gnn_spec, options),
            matrix_provenance=provenance,
            E=habit,
            habit_source=habit_source,
            random_seed=_extract_random_seed(gnn_spec, options),
            gnn_spec=gnn_spec,
        )

        # Write output
        output_path = Path(output_path)
        atomic_write_text(output_path, code)

        logger.info(f"✅ PyTorch script written to: {output_path}")
        return True, f"PyTorch script generated: {output_path}", [str(output_path)]

    except Exception as e:
        logger.error(f"❌ PyTorch rendering failed: {e}")
        return False, f"PyTorch rendering failed: {e}", []


def _is_continuous_spec(gnn_spec: Dict[str, Any]) -> bool:
    """True for continuous-state (linear-Gaussian) specs."""
    from gnn.render.continuous_common import is_continuous_spec

    return is_continuous_spec(gnn_spec)


def _render_continuous(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]],
    model_name: str,
) -> Tuple[bool, str, List[str]]:
    """Emit the standalone PyTorch Kalman-filter LGSSM script."""
    from gnn.render.continuous_common import extract_continuous_spec
    from gnn.render.continuous_script import generate_continuous_script

    code = generate_continuous_script(extract_continuous_spec(gnn_spec), "pytorch")
    atomic_write_text(output_path, code)
    logger.info(f"✅ PyTorch continuous script written to: {output_path}")
    return (
        True,
        f"PyTorch continuous LGSSM script generated: {output_path}",
        [str(output_path)],
    )


def _extract_matrices(
    gnn_spec: Dict[str, Any],
    *,
    provenance: dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract A, B, C, D matrices from GNN spec.

    Delegates to the shared :func:`render.spec_matrices.extract_abcd_matrices`.
    """
    return extract_abcd_matrices(gnn_spec, transformation_provenance=provenance)


def _extract_num_timesteps(
    gnn_spec: Dict[str, Any], options: Optional[Dict[str, Any]] = None
) -> int:
    """Extract simulation horizon from render options or parsed GNN metadata."""
    options = options or {}
    model_params = gnn_spec.get("model_parameters", {})
    init_params = gnn_spec.get("initialparameterization", {})
    return int(
        options.get(
            "num_timesteps",
            model_params.get("num_timesteps", init_params.get("num_timesteps", 10)),
        )
    )


def _parameter_source(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Return the first non-empty parameter block (``extract_abcd_matrices`` order)."""
    for params in (
        (gnn_spec.get("stateSpace") or {}).get("parameters"),
        gnn_spec.get("initialparameterization"),
        gnn_spec.get("parameters"),
    ):
        if isinstance(params, dict) and params:
            return params
    return {}


def _extract_habit(
    gnn_spec: Dict[str, Any], num_actions: int
) -> Tuple[np.ndarray, str]:
    """Return the habit prior E over one-step actions and its source.

    A declared ``E`` must be a strict probability vector with one entry per
    action; without one the uniform habit is used, which leaves
    ``softmax(log E - G)`` equal to ``softmax(-G)``.
    """
    raw = _parameter_source(gnn_spec).get("E")
    if raw is None:
        return np.full(num_actions, 1.0 / num_actions), "uniform_default"
    parsed = parse_gnn_matrix_value(raw, None)
    if parsed is None:
        raise ValueError("E is not a valid numeric vector")
    habit = np.asarray(normalise_vector(parsed, name="E"), dtype=float)
    if habit.shape != (num_actions,):
        raise ValueError(
            f"E has {habit.size} entries but B declares {num_actions} actions"
        )
    return habit, "declared"


_MAX_SEED = 2**63 - 1


def _extract_random_seed(
    gnn_spec: Dict[str, Any], options: Optional[Dict[str, Any]] = None
) -> int:
    """Return the simulation seed: ``seed`` option, then model parameters, else 42."""
    options = options or {}
    model_params = gnn_spec.get("model_parameters") or {}
    raw: Any = options.get("seed")
    if raw is None:
        raw = model_params.get("random_seed", model_params.get("seed"))
    if raw is None:
        raw = 42
    while isinstance(raw, (list, tuple)) and len(raw) == 1:
        raw = raw[0]
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError("random_seed must be a nonnegative integer, not boolean")
    try:
        seed = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("random_seed must be a nonnegative integer") from exc
    if not np.isfinite(seed) or seed < 0 or seed != int(seed) or seed > _MAX_SEED:
        raise ValueError("random_seed must be a nonnegative 64-bit integer")
    return int(seed)


def _format_tensor(arr: np.ndarray, indent: int = 4) -> str:
    """Format a numpy array as a torch.tensor() literal."""
    return format_array_literal(
        arr, prefix="torch.tensor", suffix=", dtype=torch.float64", indent=indent
    )


def _b_init_block(B: np.ndarray, num_actions: int) -> str:
    """Return the emitted statements that bind the canonical 3-D ``B``."""
    if B.ndim == 3:
        slices = [
            f"    B_slices.append({_format_tensor(B[:, :, a], indent=4)})"
            for a in range(B.shape[2])
        ]
        return (
            "B_slices = []\n"
            + "\n".join(slices)
            + "\n    B = torch.stack(B_slices, dim=2)"
        )
    return f"B = {_format_tensor(B)}.unsqueeze(2).expand(-1, -1, {num_actions}).clone()"


def _generate_pytorch_code(
    model_name: str,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    num_timesteps: int = 10,
    matrix_provenance: Optional[Dict[str, Any]] = None,
    *,
    E: Optional[np.ndarray] = None,
    habit_source: str = "uniform_default",
    random_seed: int = 42,
    gnn_spec: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate the standalone module-based PyTorch POMDP simulation script."""
    num_states = A.shape[1] if A.ndim == 2 else 2
    num_obs = A.shape[0] if A.ndim == 2 else num_states
    num_actions = B.shape[2] if B.ndim == 3 else 2
    if int(num_timesteps) < 1:
        raise ValueError("num_timesteps must be at least 1")
    habit = (
        np.full(num_actions, 1.0 / num_actions)
        if E is None
        else np.asarray(E, dtype=float)
    )
    spec = gnn_spec or {}
    variable_sites, unmapped = gnn_variable_sites(spec)
    return generate_discrete_script(
        model_name=model_name,
        random_seed=random_seed,
        num_states=num_states,
        num_obs=num_obs,
        num_actions=num_actions,
        num_timesteps=int(num_timesteps),
        a_literal=_format_tensor(A),
        c_literal=_format_tensor(C),
        d_literal=_format_tensor(D),
        e_literal=_format_tensor(habit),
        b_init=_b_init_block(B, num_actions),
        habit_source=habit_source,
        matrix_provenance=matrix_provenance or {},
        declared_connections=declared_connections(spec),
        variable_sites=variable_sites,
        unmapped_variables=unmapped,
    )
