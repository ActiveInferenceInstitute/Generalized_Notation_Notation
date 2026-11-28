"""Standalone native JAX renderer for two-agent joint Gaussian filtering."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from gnn.extract.joint_gaussian_spec import (
    JOINT_GAUSSIAN_CONTRACT,
    MAX_AXES,
    MAX_TIMESTEPS,
    extract_joint_gaussian_spec,
)
from gnn.render.continuous_script import KALMAN_STEP


def generate_joint_gaussian_script(spec: dict[str, Any]) -> str:
    """Preserve every bound joint parameter and exogenous input in the program."""
    model = extract_joint_gaussian_spec(spec)
    system = model.system
    declaration = {
        "contract": JOINT_GAUSSIAN_CONTRACT,
        "dt": system.dt,
        "agent_layout": model.agents,
        "F": system.F.tolist(),
        "H": system.H.tolist(),
        "Q": system.Q.tolist(),
        "R": system.R.tolist(),
        "prior_mean": system.prior_mean.tolist(),
        "prior_cov": system.prior_cov.tolist(),
        "control_matrix": model.control_matrix.tolist(),
        "controls": model.controls.tolist(),
        "observations": model.observations.tolist(),
    }
    identity = hashlib.sha256(
        json.dumps(declaration, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    record = {
        "schema_version": JOINT_GAUSSIAN_CONTRACT,
        "inference_contract": JOINT_GAUSSIAN_CONTRACT,
        "model_name": system.model_name,
        "model_kind": "multi_agent_continuous",
        "framework": "jax",
        "inference_mode": "gaussian_filtering",
        "initial_time": "first_observation_uses_prior",
        "agent_coupling": JOINT_GAUSSIAN_CONTRACT,
        "num_timesteps": system.num_timesteps,
        "num_states": system.n,
        "num_observations": system.m,
        "control_dimension": model.control_matrix.shape[1],
        "dt": system.dt,
        "source_spec_sha256": identity,
        "admission": {
            "max_joint_axes": MAX_AXES,
            "max_timesteps": MAX_TIMESTEPS,
        },
    }
    return (
        _HEADER
        + f"\nSOURCE = {declaration!r}\nRECORD = {record!r}\n"
        + KALMAN_STEP
        + _BODY
    )


_HEADER = '''#!/usr/bin/env python3
"""Joint Gaussian filtering of declared observations and fixed controls in JAX."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
solve = jnp.linalg.solve


def eye(n):
    return jnp.eye(n, dtype=jnp.float64)
'''

_BODY = """


def run_simulation():
    n, m, p = RECORD["num_states"], RECORD["num_observations"], RECORD["control_dimension"]
    steps = RECORD["num_timesteps"]
    F, H, Q, R, mu, covariance = (
        jnp.asarray(SOURCE[key], dtype=jnp.float64)
        for key in ("F", "H", "Q", "R", "prior_mean", "prior_cov")
    )
    observations = jnp.asarray(SOURCE["observations"], dtype=jnp.float64)
    control_matrix = jnp.asarray(SOURCE["control_matrix"], dtype=jnp.float64).reshape(n, p)
    controls = jnp.asarray(SOURCE["controls"], dtype=jnp.float64).reshape(steps - 1, p)
    means, covariances = [], []
    for step in range(steps):
        increment = jnp.zeros(n, dtype=jnp.float64)
        if step:
            increment = control_matrix @ controls[step - 1]
        mu, covariance = kalman_step(mu, covariance, observations[step], F, H, Q, R, increment, step == 0)
        means.append(np.asarray(mu))
        covariances.append(np.asarray(covariance))

    means, covariances = np.asarray(means), np.asarray(covariances)
    finite = bool(np.isfinite(means).all() and np.isfinite(covariances).all())
    symmetric = bool(np.allclose(covariances, covariances.swapaxes(1, 2), atol=1e-10, rtol=0))
    positive = finite and symmetric and bool(np.all(np.linalg.eigvalsh(covariances) > 0))
    if not positive:
        raise ValueError("Joint Gaussian filtering produced invalid posterior means or covariances")

    result = {
        **RECORD,
        "success": True,
        "dtype": str(means.dtype),
        "conditioning_horizon": list(range(1, steps + 1)),
        "agents": [agent["id"] for agent in SOURCE["agent_layout"]],
        "agent_layout": SOURCE["agent_layout"],
        "observations_continuous": SOURCE["observations"],
        "control_inputs": SOURCE["controls"],
        "state_control_increments": np.asarray(controls @ control_matrix.T).tolist(),
        "beliefs": means.tolist(),
        "posterior_cov": covariances.tolist(),
        "beliefs_by_agent": {},
        "posterior_cov_by_agent": {},
        "observations_continuous_by_agent": {},
        "control_inputs_by_agent": {},
        "state_control_increments_by_agent": {},
        "validation": {"all_valid": True, "means_and_covariance_finite": finite, "posterior_cov_positive_definite": positive},
        "runtime_metadata": {
            "jax_version": jax.__version__,
            "jax_platform": jax.default_backend(),
            "generated_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "control_timing": "control[t] moves state[t] into state[t+1]",
        },
    }
    for agent in SOURCE["agent_layout"]:
        ident = agent["id"]
        state = slice(*agent["state_slice"])
        obs = slice(*agent["observation_slice"])
        ctl = slice(*agent["control_slice"])
        result["beliefs_by_agent"][ident] = means[:, state].tolist()
        result["posterior_cov_by_agent"][ident] = covariances[:, state, state].tolist()
        result["observations_continuous_by_agent"][ident] = np.asarray(observations[:, obs]).tolist()
        result["control_inputs_by_agent"][ident] = np.asarray(controls[:, ctl]).tolist()
        result["state_control_increments_by_agent"][ident] = np.asarray(controls @ control_matrix[state].T).tolist()
    left, right = SOURCE["agent_layout"]
    result["posterior_cross_covariance"] = {
        "agents": (left["id"], right["id"]),
        "values": covariances[:, slice(*left["state_slice"]), slice(*right["state_slice"])].tolist(),
    }
    output = Path(os.environ.get("GNN_OUTPUT_DIR", "."))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "simulation_results.json"
    staging = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=output, delete=False) as pending:
            staging = Path(pending.name)
            json.dump(result, pending, indent=2, allow_nan=False)
        staging.replace(path)
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)
    return result


if __name__ == "__main__":
    raise SystemExit(0 if run_simulation()["success"] else 1)
"""
