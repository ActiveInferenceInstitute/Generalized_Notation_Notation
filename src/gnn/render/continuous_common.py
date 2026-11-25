"""Model-kind routing, script literals and result keys for continuous-state (linear-Gaussian) renderers.

The validated specifications live in ``gnn.extract.continuous_spec``; this
module re-exports each of them unchanged, so ``gnn.render.continuous_common``
remains the import path renderers and callers already use.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from gnn.extract.continuous_spec import (
    AGENT_CONTINUOUS_KEY,
    REQUIRED_DECLARATIONS,
    REQUIRED_KEYS,
    ContinuousSpec,
    FactorContinuousSpec,
    FactoredContinuousSpec,
    extract_continuous_spec,
    extract_factored_continuous_spec,
    extract_multi_agent_continuous_specs,
)

__all__ = [
    "AGENT_CONTINUOUS_KEY",
    "ContinuousSpec",
    "FactorContinuousSpec",
    "FactoredContinuousSpec",
    "REQUIRED_DECLARATIONS",
    "REQUIRED_KEYS",
    "RESULT_KEYS",
    "extract_continuous_spec",
    "extract_factored_continuous_spec",
    "extract_multi_agent_continuous_specs",
    "is_continuous_spec",
    "literal_block",
    "literal_block_factored",
    "py_literal",
]


def is_continuous_spec(gnn_spec: Dict[str, Any]) -> bool:
    """True when the spec is a continuous linear-Gaussian model.

    Fails loud: ``detect_model_kind`` raises ``ValueError`` on a malformed
    ``initialparameterization`` and that error propagates to the caller. A
    spec that cannot be classified must never be silently misrouted to the
    discrete renderers, so there is deliberately no fallback here. The
    import stays function-local to keep this module cheap to import.
    """
    if gnn_spec.get("model_kind") == "continuous":
        return True
    from gnn.render.pomdp_contract import ModelKind, detect_model_kind

    return detect_model_kind(gnn_spec) == ModelKind.CONTINUOUS


def py_literal(arr: np.ndarray) -> str:
    """Render an array as a nested Python list literal with full precision."""
    return repr(np.asarray(arr, dtype=float).tolist())


def literal_block(spec: ContinuousSpec) -> Dict[str, str]:
    """Literals for every parameter, ready to splice into a template."""
    goal = py_literal(spec.goal_mean) if spec.goal_mean is not None else "None"
    gain = repr(float(spec.control_gain)) if spec.control_gain is not None else "None"
    return {
        "F": py_literal(spec.F),
        "H": py_literal(spec.H),
        "Q": py_literal(spec.Q),
        "R": py_literal(spec.R),
        "prior_mean": py_literal(spec.prior_mean),
        "prior_cov": py_literal(spec.prior_cov),
        "goal_mean": goal,
        "control_gain": gain,
    }


def literal_block_factored(spec: FactoredContinuousSpec) -> Dict[str, str]:
    """Literals for every per-factor ``_fN`` key, ready to splice into a template."""
    lits: Dict[str, str] = {}
    for factor in spec.factors:
        suffix = f"f{factor.index}"
        lits[f"F_{suffix}"] = py_literal(factor.F)
        lits[f"H_{suffix}"] = py_literal(factor.H)
        lits[f"Q_{suffix}"] = py_literal(factor.Q)
        lits[f"R_{suffix}"] = py_literal(factor.R)
        lits[f"prior_mean_{suffix}"] = py_literal(factor.prior_mean)
        lits[f"prior_cov_{suffix}"] = py_literal(factor.prior_cov)
        # Optional closed-loop control pair: omit None entries entirely.
        if factor.goal_mean is not None:
            lits[f"goal_mean_{suffix}"] = py_literal(factor.goal_mean)
        if factor.control_gain is not None:
            lits[f"control_gain_{suffix}"] = repr(float(factor.control_gain))
    return lits


RESULT_KEYS: List[str] = [
    "model_name",
    "framework",
    "model_kind",
    "num_timesteps",
    "num_states",
    "num_observations",
    "beliefs",
    "posterior_cov",
    "true_states_continuous",
    "observations_continuous",
    "controls",
    "rmse_vs_true",
    "validation",
]
