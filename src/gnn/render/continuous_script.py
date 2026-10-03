"""Generate standalone continuous (linear-Gaussian) simulation scripts.

One generator, four Python backends (``jax``, ``numpyro``, ``pytorch``, and
``ngclearn``). The generated script simulates the LGSSM declared by the GNN
file, runs an online Kalman filter (closed-loop proportional control when the
spec declares ``goal_mean``/``control_gain``), and writes
``simulation_results.json`` with the continuous result schema documented in
:mod:`render.continuous_common`.
The NumPyro backend additionally fits the same generative model with NUTS.
The ngc-learn backend emits the same JAX Kalman numerics behind an
``ngclearn`` import guard: code generation only — executing the script
requires the ``ngclearn`` extra.
"""

from __future__ import annotations

import json
from typing import Dict

from gnn.render.continuous_common import (
    ContinuousSpec,
    FactoredContinuousSpec,
    literal_block,
    literal_block_factored,
)

_OUTPUT_ENV = {
    "jax": "GNN_OUTPUT_DIR",
    "numpyro": "NUMPYRO_OUTPUT_DIR",
    "pytorch": "PYTORCH_OUTPUT_DIR",
    "ngclearn": "NGCLEARN_OUTPUT_DIR",
}

_BACKEND_HEADER: Dict[str, str] = {
    "jax": """
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print("ERROR: JAX not installed. Install with: uv sync")
    sys.exit(1)

FRAMEWORK = "jax"
FRAMEWORK_VERSION = {"jax_version": jax.__version__}
_KEY = [jax.random.PRNGKey(RANDOM_SEED)]


def arr(x):
    return jnp.asarray(x, dtype=jnp.float64)


def eye(n):
    return jnp.eye(n, dtype=jnp.float64)


def solve(a, b):
    return jnp.linalg.solve(a, b)


def mvn_sample(mean, cov):
    _KEY[0], sub = jax.random.split(_KEY[0])
    return jax.random.multivariate_normal(sub, mean, cov)


def to_list(x):
    return np.asarray(x, dtype=float).tolist()


def is_psd(cov):
    return bool(jnp.all(jnp.linalg.eigvalsh((cov + cov.T) / 2.0) > -1e-9))
""",
    "pytorch": """
try:
    import torch
except ImportError:
    print("ERROR: PyTorch not installed. Install torch manually (see pyproject.toml note).")
    sys.exit(1)

FRAMEWORK = "pytorch"
FRAMEWORK_VERSION = {"torch_version": torch.__version__}
torch.manual_seed(RANDOM_SEED)


def arr(x):
    return torch.tensor(x, dtype=torch.float64)


def eye(n):
    return torch.eye(n, dtype=torch.float64)


def solve(a, b):
    return torch.linalg.solve(a, b)


def mvn_sample(mean, cov):
    return torch.distributions.MultivariateNormal(mean, covariance_matrix=cov).sample()


def to_list(x):
    return x.detach().cpu().numpy().astype(float).tolist()


def is_psd(cov):
    return bool(torch.all(torch.linalg.eigvalsh((cov + cov.T) / 2.0) > -1e-9))
""",
}
_BACKEND_HEADER["numpyro"] = _BACKEND_HEADER["jax"].replace(
    'FRAMEWORK = "jax"\nFRAMEWORK_VERSION = {"jax_version": jax.__version__}',
    """try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
except ImportError:
    print("ERROR: NumPyro not installed. Install with: uv sync")
    sys.exit(1)

FRAMEWORK = "numpyro"
FRAMEWORK_VERSION = {"jax_version": jax.__version__, "numpyro_version": numpyro.__version__}""",
)

_BACKEND_HEADER["ngclearn"] = _BACKEND_HEADER["jax"].replace(
    'FRAMEWORK = "jax"\nFRAMEWORK_VERSION = {"jax_version": jax.__version__}',
    """try:
    import ngclearn
except ImportError:
    print("ERROR: ngc-learn not installed. Install with: uv sync --extra ngclearn")
    sys.exit(1)

FRAMEWORK = "ngclearn"
FRAMEWORK_VERSION = {"ngclearn_version": ngclearn.__version__, "jax_version": jax.__version__}""",
)

_NUMPYRO_INFERENCE = '''

def lgssm_model(y, u, F, H, Q, R, prior_mean, prior_cov):
    """The same LGSSM as a NumPyro probabilistic program (controls observed)."""
    T = y.shape[0]
    x_prev = numpyro.sample("x_0", dist.MultivariateNormal(prior_mean, prior_cov))
    numpyro.sample("y_0", dist.MultivariateNormal(H @ x_prev, R), obs=y[0])
    for t in range(1, T):
        x_t = numpyro.sample(f"x_{t}", dist.MultivariateNormal(F @ x_prev + u[t - 1], Q))
        numpyro.sample(f"y_{t}", dist.MultivariateNormal(H @ x_t, R), obs=y[t])
        x_prev = x_t


def run_mcmc(y, u, F, H, Q, R, prior_mean, prior_cov, T, n):
    kernel = NUTS(lgssm_model)
    mcmc = MCMC(kernel, num_warmup=200, num_samples=200, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(RANDOM_SEED + 1), y, u, F, H, Q, R, prior_mean, prior_cov)
    samples = mcmc.get_samples()
    means = [np.asarray(samples[f"x_{t}"], dtype=float).mean(axis=0).tolist() for t in range(T)]
    from numpyro.diagnostics import summary as _summary

    stats = _summary(mcmc.get_samples(group_by_chain=True))
    r_hats = [float(np.max(np.asarray(v["r_hat"]))) for v in stats.values()]
    return means, (max(r_hats) if r_hats else float("nan"))
'''

_BODY_HEAD = '''

def kalman_step(mu, P, y_t, F, H, Q, R, u_prev, first):
    """One predict/update step. ``first`` skips prediction (prior is for x_1)."""
    if first:
        mu_pred, P_pred = mu, P
    else:
        mu_pred = F @ mu + u_prev
        P_pred = F @ P @ F.T + Q
    S = H @ P_pred @ H.T + R
    K = solve(S.T, (P_pred @ H.T).T).T  # P_pred H^T S^{-1}
    innovation = y_t - H @ mu_pred
    mu_new = mu_pred + K @ innovation
    I_KH = eye(P_pred.shape[0]) - K @ H
    P_new = I_KH @ P_pred @ I_KH.T + K @ R @ K.T  # Joseph form (stays PSD)
    return mu_new, P_new


def run_simulation():
    start = time.time()
    F, H, Q, R = arr(F_RAW), arr(H_RAW), arr(Q_RAW), arr(R_RAW)
    prior_mean, prior_cov = arr(PRIOR_MEAN_RAW), arr(PRIOR_COV_RAW)
    goal = arr(GOAL_MEAN_RAW) if GOAL_MEAN_RAW is not None else None
    gain = CONTROL_GAIN
    n, m, T = F.shape[0], H.shape[0], NUM_TIMESTEPS

    zero_u = arr([0.0] * n)
    true_states, observations, beliefs, covs, controls = [], [], [], [], []

    x = mvn_sample(prior_mean, prior_cov)
    y = H @ x + mvn_sample(arr([0.0] * m), R)
    mu, P = kalman_step(prior_mean, prior_cov, y, F, H, Q, R, zero_u, True)
    u = gain * (goal - mu) if goal is not None else zero_u
    for buf, val in ((true_states, x), (observations, y), (beliefs, mu), (covs, P), (controls, u)):
        buf.append(to_list(val))

    for _t in range(1, T):
        x = F @ x + u + mvn_sample(arr([0.0] * n), Q)
        y = H @ x + mvn_sample(arr([0.0] * m), R)
        mu, P = kalman_step(mu, P, y, F, H, Q, R, u, False)
        u = gain * (goal - mu) if goal is not None else zero_u
        for buf, val in ((true_states, x), (observations, y), (beliefs, mu), (covs, P), (controls, u)):
            buf.append(to_list(val))

    beliefs_np = np.asarray(beliefs)
    truth_np = np.asarray(true_states)
    rmse = float(np.sqrt(np.mean((beliefs_np - truth_np) ** 2)))
    psd = all(is_psd(arr(c)) for c in covs)
    validation = {
        "means_finite": bool(np.all(np.isfinite(beliefs_np))),
        "posterior_cov_psd": bool(psd),
        "rmse_finite": bool(np.isfinite(rmse)),
        "controls_finite": bool(np.all(np.isfinite(np.asarray(controls)))),
    }
    results = {
        "model_name": MODEL_NAME,
        "framework": FRAMEWORK,
        "model_kind": "continuous",
        "num_timesteps": T,
        "num_states": n,
        "num_observations": m,
        "beliefs": beliefs,
        "posterior_cov": covs,
        "true_states_continuous": true_states,
        "observations_continuous": observations,
        "controls": controls,
        "control_mode": "closed_loop_proportional" if goal is not None else "passive",
        "rmse_vs_true": rmse,
        # Discrete-schema slots stay empty: nothing categorical is defined here.
        "observations": [],
        "actions": [],
        "efe_history": [],
    }
'''


_NUMPYRO_RESULTS = """
    if FRAMEWORK == "numpyro":
        mcmc_means, r_hat_max = run_mcmc(
            arr(observations), arr(controls), F, H, Q, R, prior_mean, prior_cov, T, n
        )
        results["mcmc_posterior_means"] = mcmc_means
        results["mcmc_r_hat_max"] = r_hat_max
        results["mcmc_rmse_vs_kalman"] = float(
            np.sqrt(np.mean((np.asarray(mcmc_means) - beliefs_np) ** 2))
        )
        validation["mcmc_finite"] = bool(np.all(np.isfinite(np.asarray(mcmc_means))))
"""


_BODY_TAIL = """
    validation["all_valid"] = all(validation.values())
    results["validation"] = validation
    results["execution_time_seconds"] = round(time.time() - start, 4)
    results.update(FRAMEWORK_VERSION)

    output_dir = Path(os.environ.get(OUTPUT_ENV, "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "simulation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"{FRAMEWORK} continuous LGSSM simulation complete: {T} steps, rmse_vs_true={rmse:.4f}")
    print(f"Results saved to: {out}")
    print(f"Validation: {validation}")
    return results


if __name__ == "__main__":
    res = run_simulation()
    sys.exit(0 if res["validation"]["all_valid"] else 1)
"""


_FACTORED_BODY_HEAD = '''

def kalman_step(mu, P, y_t, F, H, Q, R, u_prev, first):
    """One predict/update step; numerics identical to the flat LGSSM path."""
    if first:
        mu_pred, P_pred = mu, P
    else:
        mu_pred = F @ mu + u_prev
        P_pred = F @ P @ F.T + Q
    S = H @ P_pred @ H.T + R
    K = solve(S.T, (P_pred @ H.T).T).T  # P_pred H^T S^{-1}
    innovation = y_t - H @ mu_pred
    mu_new = mu_pred + K @ innovation
    I_KH = eye(P_pred.shape[0]) - K @ H
    P_new = I_KH @ P_pred @ I_KH.T + K @ R @ K.T  # Joseph form (stays PSD)
    return mu_new, P_new


def simulate_factor(f_index, F, H, Q, R, prior_mean, prior_cov, goal, gain):
    """Simulate and Kalman-filter one factor's independent LGSSM."""
    n_f, m_f = F.shape[0], H.shape[0]

    zero_u = arr([0.0] * n_f)
    true_states, observations, beliefs, covs, controls = [], [], [], [], []

    x = mvn_sample(prior_mean, prior_cov)
    y = H @ x + mvn_sample(arr([0.0] * m_f), R)
    mu, P = kalman_step(prior_mean, prior_cov, y, F, H, Q, R, zero_u, True)
    u = gain * (goal - mu) if goal is not None else zero_u
    for buf, val in ((true_states, x), (observations, y), (beliefs, mu), (covs, P), (controls, u)):
        buf.append(to_list(val))

    for _t in range(1, NUM_TIMESTEPS):
        x = F @ x + u + mvn_sample(arr([0.0] * n_f), Q)
        y = H @ x + mvn_sample(arr([0.0] * m_f), R)
        mu, P = kalman_step(mu, P, y, F, H, Q, R, u, False)
        u = gain * (goal - mu) if goal is not None else zero_u
        for buf, val in ((true_states, x), (observations, y), (beliefs, mu), (covs, P), (controls, u)):
            buf.append(to_list(val))

    beliefs_np = np.asarray(beliefs)
    truth_np = np.asarray(true_states)
    rmse = float(np.sqrt(np.mean((beliefs_np - truth_np) ** 2)))
    record = {
        "factor_index": f_index,
        "num_states": n_f,
        "num_observations": m_f,
        "beliefs": beliefs,
        "posterior_cov": covs,
        "true_states_continuous": true_states,
        "observations_continuous": observations,
        "controls": controls,
        "control_mode": "closed_loop_proportional" if goal is not None else "passive",
        "rmse_vs_true": rmse,
    }
    psd = all(is_psd(arr(c)) for c in covs)
    return record, psd, beliefs_np, controls


def run_simulation():
    start = time.time()
    factors = []
    means_finite, psd_all, controls_finite = True, True, True
    for _f in range(NUM_FACTORS):
        F, H = arr(F_FACTORS[_f]), arr(H_FACTORS[_f])
        Q, R = arr(Q_FACTORS[_f]), arr(R_FACTORS[_f])
        prior_mean, prior_cov = arr(PRIOR_MEAN_FACTORS[_f]), arr(PRIOR_COV_FACTORS[_f])
        goal = arr(GOAL_FACTORS[_f]) if GOAL_FACTORS[_f] is not None else None
        gain = GAIN_FACTORS[_f]
        record, psd, beliefs_np, controls = simulate_factor(
            _f + 1, F, H, Q, R, prior_mean, prior_cov, goal, gain
        )
        factors.append(record)
        psd_all = psd_all and psd
        means_finite = means_finite and bool(np.all(np.isfinite(beliefs_np)))
        controls_finite = controls_finite and bool(np.all(np.isfinite(controls)))

    rmse_aggregate = float(np.mean([factor["rmse_vs_true"] for factor in factors]))
    validation = {
        "means_finite": means_finite,
        "posterior_cov_psd": psd_all,
        "rmse_finite": bool(np.isfinite(rmse_aggregate)),
        "controls_finite": controls_finite,
    }
    results = {
        "model_name": MODEL_NAME,
        "framework": FRAMEWORK,
        "model_kind": "factored_continuous",
        "num_timesteps": NUM_TIMESTEPS,
        "num_factors": NUM_FACTORS,
        "factors": factors,
        "rmse_vs_true_aggregate": rmse_aggregate,
        # Discrete-schema slots stay empty: nothing categorical is defined here.
        "observations": [],
        "actions": [],
        "efe_history": [],
    }
'''


_FACTORED_BODY_TAIL = """
    validation["all_valid"] = all(validation.values())
    results["validation"] = validation
    results["execution_time_seconds"] = round(time.time() - start, 4)
    results.update(FRAMEWORK_VERSION)

    output_dir = Path(os.environ.get(OUTPUT_ENV, "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "simulation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"{FRAMEWORK} factored-continuous LGSSM simulation complete: {NUM_FACTORS} per-factor LGSSMs, {NUM_TIMESTEPS} steps, rmse_vs_true_aggregate={rmse_aggregate:.4f}")
    print(f"Results saved to: {out}")
    print(f"Validation: {validation}")
    return results


if __name__ == "__main__":
    res = run_simulation()
    sys.exit(0 if res["validation"]["all_valid"] else 1)
"""


def generate_continuous_script(spec: ContinuousSpec, backend: str) -> str:
    """Return the full standalone script for ``backend``."""
    if backend not in _BACKEND_HEADER:
        raise ValueError(f"unsupported continuous backend: {backend}")
    lits = literal_block(spec)
    control_desc = (
        f"closed-loop u_t = {spec.control_gain} * (goal_mean - mu_t)"
        if spec.has_control
        else "passive (u_t = 0)"
    )
    header = f'''#!/usr/bin/env python3
"""
{backend} continuous (linear-Gaussian state-space) simulation: {json.dumps(spec.model_name)}

Auto-generated by the GNN pipeline — {backend} renderer, continuous branch.
Generative model:
    x_1 ~ N(prior_mean, prior_cov)
    x_t = F x_(t-1) + u_(t-1) + N(0, Q)      ({spec.n}-dim latent state)
    y_t = H x_t + N(0, R)                    ({spec.m}-dim observation)
Control: {control_desc}
Inference: online Kalman filter (Joseph-form covariance update).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

MODEL_NAME = {spec.model_name!r}
RANDOM_SEED = {spec.random_seed}
NUM_TIMESTEPS = {spec.num_timesteps}
DT = {spec.dt}
OUTPUT_ENV = {_OUTPUT_ENV[backend]!r}

F_RAW = {lits["F"]}
H_RAW = {lits["H"]}
Q_RAW = {lits["Q"]}
R_RAW = {lits["R"]}
PRIOR_MEAN_RAW = {lits["prior_mean"]}
PRIOR_COV_RAW = {lits["prior_cov"]}
GOAL_MEAN_RAW = {lits["goal_mean"]}
CONTROL_GAIN = {lits["control_gain"]}
'''
    if backend in ("jax", "numpyro", "ngclearn"):
        header += "\nfrom jax import config as _jax_config\n_jax_config.update('jax_enable_x64', True)\n"
    parts = [header, _BACKEND_HEADER[backend]]
    if backend == "numpyro":
        parts.append(_NUMPYRO_INFERENCE)
    parts.append(_BODY_HEAD)
    if backend == "numpyro":
        parts.append(_NUMPYRO_RESULTS)
    parts.append(_BODY_TAIL)
    return "".join(parts)


def generate_factored_continuous_script(
    spec: FactoredContinuousSpec, backend: str
) -> str:
    """Return the standalone per-factor LGSSM script (``jax`` backend only)."""
    if backend != "jax":
        raise ValueError(
            f"unsupported factored-continuous backend: {backend} (jax only)"
        )
    lits = literal_block_factored(spec)
    factor_indices = range(1, spec.num_factors + 1)
    f_list = ", ".join(lits[f"F_f{i}"] for i in factor_indices)
    h_list = ", ".join(lits[f"H_f{i}"] for i in factor_indices)
    q_list = ", ".join(lits[f"Q_f{i}"] for i in factor_indices)
    r_list = ", ".join(lits[f"R_f{i}"] for i in factor_indices)
    pm_list = ", ".join(lits[f"prior_mean_f{i}"] for i in factor_indices)
    pc_list = ", ".join(lits[f"prior_cov_f{i}"] for i in factor_indices)
    goal_list = ", ".join(lits.get(f"goal_mean_f{i}", "None") for i in factor_indices)
    gain_list = ", ".join(
        lits.get(f"control_gain_f{i}", "None") for i in factor_indices
    )
    control_desc = (
        "closed-loop per factor: u_f,t = gain_f * (goal_f - mu_f,t)"
        if all(f.has_control for f in spec.factors)
        else "mixed closed-loop/passive per factor"
        if any(f.has_control for f in spec.factors)
        else "passive (u_f,t = 0)"
    )
    header = rf'''#!/usr/bin/env python3
"""
{backend} factored-continuous (per-factor LGSSM) simulation: {json.dumps(spec.model_name)}

Auto-generated by the GNN pipeline — {backend} renderer, factored-continuous branch.
{spec.num_factors} independent linear-Gaussian state-space blocks (one per factor,
mirroring the ``^([ABCD])_f(\\d+)$`` key convention). Each factor runs the flat
continuous numerics on its own subscripted parameters:
    x_f1 ~ N(prior_mean_f, prior_cov_f)
    x_ft = F_f x_f,t-1 + u_f,t-1 + N(0, Q_f)
    y_ft = H_f x_ft + N(0, R_f)
Control: {control_desc}
Inference: per-factor online Kalman filter (Joseph-form covariance update).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

MODEL_NAME = {spec.model_name!r}
RANDOM_SEED = {spec.random_seed}
NUM_TIMESTEPS = {spec.num_timesteps}
DT = {spec.dt}
NUM_FACTORS = {spec.num_factors}
OUTPUT_ENV = {_OUTPUT_ENV["jax"]!r}

F_FACTORS = [{f_list}]
H_FACTORS = [{h_list}]
Q_FACTORS = [{q_list}]
R_FACTORS = [{r_list}]
PRIOR_MEAN_FACTORS = [{pm_list}]
PRIOR_COV_FACTORS = [{pc_list}]
GOAL_FACTORS = [{goal_list}]
GAIN_FACTORS = [{gain_list}]
'''
    header += "\nfrom jax import config as _jax_config\n_jax_config.update('jax_enable_x64', True)\n"
    parts = [header, _BACKEND_HEADER["jax"], _FACTORED_BODY_HEAD, _FACTORED_BODY_TAIL]
    return "".join(parts)
