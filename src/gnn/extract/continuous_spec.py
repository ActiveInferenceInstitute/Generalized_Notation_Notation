"""Validated linear-Gaussian specifications read from a parsed GNN spec.

Extraction owns the source contract; renderers and analyzers consume the
validated values. Every consumer reads the same ``gnn_spec`` produced for
``model_kind == "continuous"``:

``initialparameterization`` holds ``F`` (n×n), ``H`` (m×n), ``Q`` (n×n),
``R`` (m×m), ``prior_mean`` (n), ``prior_cov`` (n×n) and optionally
``goal_mean`` (n) + ``control_gain`` (scalar). ``model_parameters`` holds
``num_timesteps``, ``dt`` and ``random_seed``.

The generative model these values declare:

    x_1 ~ N(prior_mean, prior_cov)
    x_t = F x_{t-1} + u_{t-1} + N(0, Q)
    y_t = H x_t + N(0, R)
    u_t = control_gain * (goal_mean - mu_t)   (mu_t = filtered mean; else 0)

Per-factor (``_fN``) and per-agent (``_agentN``) blocks are validated here as
well; an independent agent composition never clones a shared flat block.
Model-kind routing, script literals and result-key lists stay in
``gnn.render.continuous_common``, which re-exports every name defined here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

REQUIRED_KEYS = ("F", "H", "Q", "R", "prior_mean", "prior_cov")


@dataclass
class ContinuousSpec:
    """Validated numeric view of a continuous GNN spec."""

    model_name: str
    F: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    prior_mean: np.ndarray
    prior_cov: np.ndarray
    goal_mean: Optional[np.ndarray]
    control_gain: Optional[float]
    num_timesteps: int
    dt: float
    random_seed: int

    @property
    def n(self) -> int:
        return int(self.F.shape[0])

    @property
    def m(self) -> int:
        return int(self.H.shape[0])

    @property
    def has_control(self) -> bool:
        return self.goal_mean is not None and self.control_gain is not None


def _scalar(value: Any) -> float:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Scientific scalar parameters must be numeric, not boolean")
    return float(value)


def _positive_integer(value: Any, *, name: str) -> int:
    scalar = _scalar(value)
    if not np.isfinite(scalar) or scalar < 1 or scalar != int(scalar):
        raise ValueError(f"{name} must be a positive integer")
    return int(scalar)


def _validate_covariance(
    matrix: np.ndarray, *, name: str, positive_definite: bool
) -> None:
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain finite values")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if positive_definite and np.any(eigenvalues <= 0):
        raise ValueError(f"{name} must be positive definite for Gaussian inference")
    if not positive_definite and np.any(eigenvalues < 0):
        raise ValueError(f"{name} must be positive semidefinite")


def extract_continuous_spec(gnn_spec: Dict[str, Any]) -> ContinuousSpec:
    """Parse and shape-check the continuous parameter block."""
    initial = gnn_spec.get("initialparameterization") or gnn_spec.get(
        "initial_parameterization"
    )
    if not isinstance(initial, dict):
        raise ValueError("continuous spec requires an initialparameterization mapping")
    missing = [key for key in REQUIRED_KEYS if key not in initial]
    if missing:
        raise ValueError(f"continuous spec is missing {missing}")

    F = np.asarray(initial["F"], dtype=float)
    H = np.asarray(initial["H"], dtype=float)
    Q = np.asarray(initial["Q"], dtype=float)
    R = np.asarray(initial["R"], dtype=float)
    prior_mean = np.asarray(initial["prior_mean"], dtype=float).reshape(-1)
    prior_cov = np.asarray(initial["prior_cov"], dtype=float)
    if F.ndim != 2 or not F.shape[0] or F.shape[0] != F.shape[1]:
        raise ValueError(f"F must be square, got {F.shape}")
    n = F.shape[0]
    if H.ndim != 2 or not H.shape[0]:
        raise ValueError(f"H must be a nonempty matrix, got {H.shape}")
    m = H.shape[0]
    if H.shape != (m, n):
        raise ValueError(f"H must be [m, n]={m, n}, got {H.shape}")
    if Q.shape != (n, n) or R.shape != (m, m) or prior_cov.shape != (n, n):
        raise ValueError(
            f"covariance shapes mismatch: Q{Q.shape} R{R.shape} prior_cov{prior_cov.shape}"
        )
    if prior_mean.shape != (n,):
        raise ValueError(f"prior_mean must have {n} entries, got {prior_mean.shape}")
    for name, value in (("F", F), ("H", H), ("prior_mean", prior_mean)):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite values")
    # These backends use nonsingular Gaussian densities, Cholesky sampling,
    # and innovation solves. Degenerate Gaussian support needs its own route.
    for name, matrix in (("Q", Q), ("R", R), ("prior_cov", prior_cov)):
        _validate_covariance(matrix, name=name, positive_definite=True)

    goal_mean: Optional[np.ndarray] = None
    control_gain: Optional[float] = None
    if ("goal_mean" in initial) != ("control_gain" in initial):
        raise ValueError("goal_mean and control_gain must be declared together")
    if "goal_mean" in initial:
        goal_mean = np.asarray(initial["goal_mean"], dtype=float).reshape(-1)
        if goal_mean.shape != (n,):
            raise ValueError(f"goal_mean must have {n} entries, got {goal_mean.shape}")
        control_gain = _scalar(initial["control_gain"])
        if not np.isfinite(goal_mean).all() or not np.isfinite(control_gain):
            raise ValueError("goal_mean and control_gain must be finite")

    params = gnn_spec.get("model_parameters") or {}
    num_timesteps = _positive_integer(
        params.get("num_timesteps", 20), name="num_timesteps"
    )
    dt = _scalar(params.get("dt", 1.0))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    seed_raw = _scalar(params.get("random_seed", params.get("seed", 42)))
    if not np.isfinite(seed_raw) or seed_raw < 0 or seed_raw != int(seed_raw):
        raise ValueError("random_seed must be a nonnegative integer")
    seed = int(seed_raw)
    name = str(gnn_spec.get("model_name") or gnn_spec.get("name") or "continuous_model")
    return ContinuousSpec(
        model_name=name,
        F=F,
        H=H,
        Q=Q,
        R=R,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        goal_mean=goal_mean,
        control_gain=control_gain,
        num_timesteps=num_timesteps,
        dt=dt,
        random_seed=seed,
    )


@dataclass
class FactorContinuousSpec:
    """Validated numeric view of one per-factor LGSSM block (``_fN`` keys)."""

    index: int
    F: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    prior_mean: np.ndarray
    prior_cov: np.ndarray
    goal_mean: Optional[np.ndarray]
    control_gain: Optional[float]

    @property
    def n(self) -> int:
        return int(self.F.shape[0])

    @property
    def m(self) -> int:
        return int(self.H.shape[0])

    @property
    def has_control(self) -> bool:
        return self.goal_mean is not None and self.control_gain is not None


@dataclass
class FactoredContinuousSpec:
    """Validated numeric view of a factored-continuous spec (per-factor LGSSMs)."""

    model_name: str
    factors: List[FactorContinuousSpec]
    num_timesteps: int
    dt: float
    random_seed: int

    @property
    def num_factors(self) -> int:
        return len(self.factors)


def extract_factored_continuous_spec(
    gnn_spec: Dict[str, Any],
) -> FactoredContinuousSpec:
    """Parse and shape-check a factored-continuous parameter block (``_fN`` keys)."""
    # Subscripted-key convention mirrors the Kronecker factored regex
    # ^([ABCD])_f(\d+)$: one F/H/Q/R + prior block per factor (1-indexed,
    # contiguous), with optional goal_mean_fN + control_gain_fN (both-or-neither).
    initial = gnn_spec.get("initialparameterization") or gnn_spec.get(
        "initial_parameterization"
    )
    if not isinstance(initial, dict):
        raise ValueError(
            "factored-continuous spec requires an initialparameterization mapping"
        )
    params = gnn_spec.get("model_parameters") or {}
    num_factors = int(params.get("num_factors", 0))
    if num_factors < 2:
        raise ValueError(
            f"factored-continuous spec requires num_factors >= 2 in model_parameters, "
            f"got {num_factors}"
        )

    # Collect every missing subscripted key across all factors so one receipt
    # names the whole gap instead of failing factor by factor.
    missing = [
        f"{name}_f{factor}"
        for factor in range(1, num_factors + 1)
        for name in REQUIRED_KEYS
        if f"{name}_f{factor}" not in initial
    ]
    if missing:
        raise ValueError(f"factored-continuous spec is missing {missing}")

    factors: List[FactorContinuousSpec] = []
    for factor in range(1, num_factors + 1):
        block = {name: initial[f"{name}_f{factor}"] for name in REQUIRED_KEYS}
        for name in ("goal_mean", "control_gain"):
            if f"{name}_f{factor}" in initial:
                block[name] = initial[f"{name}_f{factor}"]
        if ("goal_mean" in block) != ("control_gain" in block):
            raise ValueError(
                f"goal_mean_f{factor} and control_gain_f{factor} must be declared together"
            )
        try:
            validated = extract_continuous_spec(
                {
                    "initialparameterization": block,
                    "model_parameters": params,
                }
            )
        except ValueError as exc:
            raise ValueError(f"Continuous factor {factor}: {exc}") from exc
        factors.append(
            FactorContinuousSpec(
                index=factor,
                F=validated.F,
                H=validated.H,
                Q=validated.Q,
                R=validated.R,
                prior_mean=validated.prior_mean,
                prior_cov=validated.prior_cov,
                goal_mean=validated.goal_mean,
                control_gain=validated.control_gain,
            )
        )
    num_timesteps, dt, seed = (
        validated.num_timesteps,
        validated.dt,
        validated.random_seed,
    )

    name = str(
        gnn_spec.get("model_name")
        or gnn_spec.get("name")
        or "factored_continuous_model"
    )
    return FactoredContinuousSpec(
        model_name=name,
        factors=factors,
        num_timesteps=num_timesteps,
        dt=dt,
        random_seed=seed,
    )


AGENT_CONTINUOUS_KEY = re.compile(
    r"^(F|H|Q|R|prior_mean|prior_cov|goal_mean|control_gain)_agent([1-9]\d*)$"
)
REQUIRED_DECLARATIONS = "declare agent_coupling: independent and complete F_agentN/H_agentN/Q_agentN/R_agentN/prior_mean_agentN/prior_cov_agentN blocks for every agent"


def extract_multi_agent_continuous_specs(
    spec: Dict[str, Any],
) -> Dict[str, ContinuousSpec]:
    """Validate all source agent identities, Gaussian blocks and control pairs."""
    initial = spec.get("initialparameterization") or spec.get(
        "initial_parameterization"
    )
    params = spec.get("model_parameters") or {}
    if not isinstance(initial, dict) or params.get("agent_coupling") != "independent":
        raise ValueError(f"unsupported-composition: {REQUIRED_DECLARATIONS}")
    if any(key in initial for key in REQUIRED_KEYS):
        raise ValueError(
            "unsupported-composition: a shared flat Gaussian block cannot be assigned independent agent semantics"
        )
    count = params.get("nr_agents", params.get("num_agents"))
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("Independent continuous agents require integer nr_agents >= 2")
    groups: Dict[int, Dict[str, Any]] = {}
    for key, value in initial.items():
        match = AGENT_CONTINUOUS_KEY.fullmatch(str(key))
        if match:
            groups.setdefault(int(match[2]), {})[match[1]] = value
    if set(groups) != set(range(1, count + 1)):
        raise ValueError(
            f"Agent blocks must match contiguous declared identities agent1..agent{count}"
        )
    base_seed = _scalar(params.get("random_seed", params.get("seed", 42)))
    if not np.isfinite(base_seed) or base_seed < 0 or base_seed != int(base_seed):
        raise ValueError("random_seed must be a nonnegative integer")
    result = {}
    for index in range(1, count + 1):
        agent_params = {
            **params,
            "random_seed": params.get(
                f"random_seed_agent{index}", int(base_seed) + index - 1
            ),
        }
        try:
            result[f"agent{index}"] = extract_continuous_spec(
                {
                    "model_name": f"{spec.get('model_name', 'Continuous agents')}: agent{index}",
                    "initialparameterization": groups[index],
                    "model_parameters": agent_params,
                }
            )
        except ValueError as exc:
            raise ValueError(f"agent{index}: {exc}") from exc
    return result
