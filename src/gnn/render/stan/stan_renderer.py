#!/usr/bin/env python3
"""
Stan Renderer — genuine Stan programs for GNN generative models.

Two model kinds are rendered:

* **Discrete POMDP / HMM** — a hidden Markov model whose latent chain is
  marginalised with the forward algorithm (``log_sum_exp``). Observations and
  actions are *data* (simulated from the GNN's own A/B/D by the driver); the
  per-state observation distributions ``A_est`` are *parameters* with
  Dirichlet priors centred on the GNN's declared ``A``. Generated quantities
  expose the filtered state posteriors.
* **Continuous linear-Gaussian state-space** — the Kalman-filter marginal
  likelihood written out explicitly (predict/update, ``multi_normal_lpdf`` of
  the innovations), with an observation-noise scale as the free parameter.

Each render emits three artifacts with one stem: ``<stem>.stan`` (program),
``<stem>.py`` (cmdstanpy driver: simulate → compile → sample → results JSON)
and, at run time, ``<stem>_data.json``. Step 12 executes the ``.py`` driver
like any other Python framework script; it skips with a dependency reason
when ``cmdstanpy``/CmdStan are absent.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_gnn_to_stan(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]:
    """Render a GNN spec to a Stan program plus its cmdstanpy driver.

    ``output_path`` is the driver ``.py`` path; the ``.stan`` program is written
    beside it with the same stem.
    """
    from gnn.render.execution_contracts import unsupported_contract

    refusal = unsupported_contract(gnn_spec, "stan")
    if refusal:
        return False, refusal, []

    try:
        output_path = Path(output_path)
        stan_path = output_path.with_suffix(".stan")
        from gnn.render.continuous_common import (
            extract_continuous_spec,
            is_continuous_spec,
        )

        if is_continuous_spec(gnn_spec):
            from gnn.render.pomdp_contract import ModelKind, detect_model_kinds

            if detect_model_kinds(gnn_spec) == frozenset(
                {ModelKind.FACTORED, ModelKind.CONTINUOUS}
            ):
                return (
                    False,
                    "unsupported-factored-continuous: stan renders the flat "
                    "linear-Gaussian family only; per-factor compositions are "
                    "refused rather than silently rendered flat",
                    [],
                )
            spec = extract_continuous_spec(gnn_spec)
            program = _continuous_program()
            driver = _continuous_driver(spec, stan_path.name)
            kind = "continuous LGSSM"
        else:
            params = _discrete_parameters(gnn_spec)
            program = _discrete_program()
            driver = _discrete_driver(params, stan_path.name)
            kind = "discrete POMDP/HMM"

        _atomic_write(stan_path, program)
        _atomic_write(output_path, driver)
        logger.info(f"✅ Stan {kind} program + driver written to: {output_path.parent}")
        return (
            True,
            f"Stan {kind} program generated: {stan_path.name} (+ driver {output_path.name})",
            [str(output_path), str(stan_path)],
        )
    except Exception as exc:
        logger.error(f"❌ Stan rendering failed: {exc}")
        return False, f"Stan rendering failed: {exc}", []


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as tmp:
        tmp.write(text)
    os.replace(tmp.name, str(path))


# ---------------------------------------------------------------------------
# Discrete POMDP / HMM
# ---------------------------------------------------------------------------


def _discrete_parameters(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Pull canonical A (O×S), B (S×S×U), D (S) and horizon from the spec."""
    initial = gnn_spec.get("initialparameterization") or gnn_spec.get(
        "initial_parameterization"
    )
    if not isinstance(initial, dict):
        raise ValueError("Stan renderer requires an initialparameterization mapping")
    for key in ("A", "B", "D"):
        if key not in initial:
            raise ValueError(f"Stan renderer requires canonical matrix {key!r}")
    A = np.asarray(initial["A"], dtype=float)
    B = np.asarray(initial["B"], dtype=float)
    D = np.asarray(initial["D"], dtype=float).reshape(-1)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D (obs × states), got shape {A.shape}")
    if B.ndim == 2:
        B = B[:, :, None]
    if B.ndim != 3:
        raise ValueError(f"B must be 3-D (next × prev × action), got shape {B.shape}")
    S = A.shape[1]
    O = A.shape[0]
    if B.shape[0] != S or B.shape[1] != S or D.shape[0] != S:
        raise ValueError(
            f"shape mismatch: A{A.shape} B{B.shape} D{D.shape} (states must agree)"
        )
    U = B.shape[2]
    # Column-normalise so every conditional distribution is a proper simplex.
    A = A / np.clip(A.sum(axis=0, keepdims=True), 1e-12, None)
    B = B / np.clip(B.sum(axis=0, keepdims=True), 1e-12, None)
    D = D / max(D.sum(), 1e-12)
    params = gnn_spec.get("model_parameters") or {}
    T = int(params.get("num_timesteps", 20))
    seed = int(params.get("random_seed", params.get("seed", 42)))
    name = str(gnn_spec.get("model_name") or gnn_spec.get("name") or "gnn_model")
    return {
        "model_name": name,
        "A": A,
        "B": B,
        "D": D,
        "S": S,
        "O": O,
        "U": U,
        "T": max(2, T),
        "seed": seed,
        "pseudo_count_strength": float(params.get("stan_dirichlet_strength", 20.0)),
        # S*O simplex parameters above this budget switch the driver from NUTS
        # to MAP (L-BFGS); overridable per model via ModelParameters.
        "nuts_param_budget": int(params.get("stan_nuts_param_budget", 1024)),
    }


def _discrete_program() -> str:
    return """// Generated by the GNN pipeline — Stan renderer (discrete POMDP / HMM).
//
// Latent categorical chain marginalised with the forward algorithm.
// Data: observed outcomes o[t] and the actions u[t] that drove the chain,
// the known action-conditioned transition tensor B[a][s_next, s_prev] and the
// prior D. Parameters: the per-state observation distributions A_est[s]
// (columns of the GNN likelihood A), with Dirichlet priors centred on the
// GNN's declared A. Transformed parameters expose the filtered posteriors.
data {
  int<lower=1> T;                              // timesteps
  int<lower=1> S;                              // hidden states
  int<lower=1> O;                              // observation outcomes
  int<lower=1> U;                              // actions
  array[T] int<lower=1, upper=O> o;            // observations (1-based)
  array[T - 1] int<lower=1, upper=U> u;        // actions taken at t=1..T-1
  array[U] matrix[S, S] B;                     // B[a][s_next, s_prev], columns sum to 1
  array[S] vector<lower=0>[O] alpha_A;         // Dirichlet pseudo-counts per state
  simplex[S] D;                                // prior over the initial state
}
parameters {
  array[S] simplex[O] A_est;                   // P(o | s) for each state s
}
transformed parameters {
  // Scaled forward algorithm, vectorised: alpha_t ∝ A[o_t, :] .* (B[u_{t-1}] alpha_{t-1}).
  // Scaling by the per-step normaliser c_t keeps the recursion in probability
  // space (no underflow) and yields log p(o_1..T) = sum_t log c_t.
  array[T] vector[S] filtered_state;           // P(s_t | o_1..t, u_1..t-1)
  real log_marginal = 0;
  {
    matrix[O, S] A_mat;                        // A_mat[o, s] = P(o | s)
    for (s in 1:S) {
      A_mat[:, s] = A_est[s];
    }
    vector[S] alpha = D .* (A_mat[o[1], :])';
    real c = sum(alpha) + 1e-300;
    log_marginal += log(c);
    alpha /= c;
    filtered_state[1] = alpha;
    for (t in 2:T) {
      alpha = (B[u[t - 1]] * alpha) .* (A_mat[o[t], :])';
      c = sum(alpha) + 1e-300;
      log_marginal += log(c);
      alpha /= c;
      filtered_state[t] = alpha;
    }
  }
}
model {
  for (s in 1:S) {
    A_est[s] ~ dirichlet(alpha_A[s]);
  }
  target += log_marginal;                      // marginal likelihood of o[1:T]
}
"""


def _discrete_driver(p: Dict[str, Any], stan_name: str) -> str:
    A_lit = repr(p["A"].tolist())
    B_lit = repr(p["B"].tolist())
    D_lit = repr(p["D"].tolist())
    return f'''#!/usr/bin/env python3
"""
Stan POMDP/HMM driver: {p["model_name"]}

Auto-generated by the GNN pipeline — Stan renderer (discrete branch).
1. Simulate T steps from the GNN generative model (A, B, D; exploratory
   uniform-random action policy) with a fixed seed.
2. Compile the sibling Stan program with CmdStan and sample the posterior over
   the per-state observation distributions A_est (Dirichlet prior centred on A).
3. Write simulation_results.json (filtered beliefs, learned A, diagnostics).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

MODEL_NAME = {p["model_name"]!r}
STAN_FILE = Path(__file__).resolve().with_name({stan_name!r})
RANDOM_SEED = {p["seed"]}
T = {p["T"]}
S, O, U = {p["S"]}, {p["O"]}, {p["U"]}
PSEUDO_COUNT_STRENGTH = {p["pseudo_count_strength"]}
NUTS_PARAM_BUDGET = {p["nuts_param_budget"]}
A = np.asarray({A_lit}, dtype=float)          # O x S, columns sum to 1
B = np.asarray({B_lit}, dtype=float)          # S(next) x S(prev) x U
D = np.asarray({D_lit}, dtype=float)          # S


def simulate(rng):
    s = int(rng.choice(S, p=D))
    obs, acts, states = [], [], []
    for t in range(T):
        o = int(rng.choice(O, p=A[:, s]))
        obs.append(o)
        states.append(s)
        if t < T - 1:
            a = int(rng.integers(U))
            acts.append(a)
            s = int(rng.choice(S, p=B[:, s, a]))
    return obs, acts, states


def main():
    start = time.time()
    try:
        import cmdstanpy
    except ImportError:
        print("ERROR: cmdstanpy not installed. Install with: uv sync --extra stan")
        return 1
    try:
        cmdstan_version = cmdstanpy.cmdstan_version()
    except Exception as exc:  # CmdStan toolchain missing
        print(f"ERROR: CmdStan toolchain not available: {{exc}}")
        return 1

    rng = np.random.default_rng(RANDOM_SEED)
    obs, acts, states = simulate(rng)
    output_dir = Path(os.environ.get("STAN_OUTPUT_DIR", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {{
        "T": T, "S": S, "O": O, "U": U,
        "o": [o + 1 for o in obs],
        "u": [a + 1 for a in acts],
        "B": [B[:, :, a].tolist() for a in range(U)],
        "alpha_A": [(1.0 + PSEUDO_COUNT_STRENGTH * A[:, s]).tolist() for s in range(S)],
        "D": D.tolist(),
    }}
    data_path = output_dir / (STAN_FILE.stem + "_data.json")
    with open(data_path, "w") as f:
        json.dump(data, f)

    model = cmdstanpy.CmdStanModel(stan_file=str(STAN_FILE))
    # Inference budget: full NUTS for models whose A_est parameter count stays
    # modest; MAP (L-BFGS with the Jacobian adjustment) for large joint
    # compositions (e.g. multi-agent products with hundreds of states), where
    # NUTS over tens of thousands of simplex parameters is impractical.
    num_params = S * O
    if num_params <= NUTS_PARAM_BUDGET:
        inference = "NUTS via CmdStan (1 chain, 300 warmup, 300 draws)"
        fit = model.sample(
            data=str(data_path), chains=1, iter_warmup=300, iter_sampling=300,
            seed=RANDOM_SEED, show_progress=False, show_console=False,
            output_dir=str(output_dir / "cmdstan"),
        )
        draws = fit.stan_variables()
        filtered = np.asarray(draws["filtered_state"])       # draws x T x S
        a_est = np.asarray(draws["A_est"])                    # draws x S x O
        beliefs = filtered.mean(axis=0)                       # T x S
        a_post = a_est.mean(axis=0).T                         # O x S
        summary = fit.summary()
        rhat_col = "R_hat" if "R_hat" in summary.columns else summary.columns[-1]
        a_rows = [idx for idx in summary.index if str(idx).startswith("A_est")]
        rhat_max = float(np.nanmax(summary.loc[a_rows, rhat_col])) if a_rows else float("nan")
        convergence_ok = bool(np.isfinite(rhat_max) and rhat_max < 1.1)
        convergence_key = "rhat_ok"
    else:
        inference = f"MAP via CmdStan optimize (L-BFGS, jacobian=True; {{num_params}} params > NUTS budget {{NUTS_PARAM_BUDGET}})"
        fit = model.optimize(
            data=str(data_path), algorithm="lbfgs", jacobian=True, iter=2000,
            seed=RANDOM_SEED, show_console=False,
            output_dir=str(output_dir / "cmdstan"),
        )
        draws = fit.stan_variables()
        beliefs = np.asarray(draws["filtered_state"], dtype=float)  # T x S
        a_post = np.asarray(draws["A_est"], dtype=float).T             # O x S
        rhat_max = float("nan")
        convergence_ok = bool(np.isfinite(float(fit.optimized_params_dict.get("lp__", np.nan))))
        convergence_key = "map_converged"
        draws["log_marginal"] = [float(draws["log_marginal"])]

    validation = {{
        "beliefs_in_range": bool(np.all((beliefs >= -1e-9) & (beliefs <= 1 + 1e-9))),
        "beliefs_sum_to_one": bool(np.allclose(beliefs.sum(axis=1), 1.0, atol=1e-6)),
        "a_posterior_columns_sum_to_one": bool(np.allclose(a_post.sum(axis=0), 1.0, atol=1e-6)),
        convergence_key: convergence_ok,
        "observations_in_range": all(0 <= o < O for o in obs),
    }}
    validation["all_valid"] = all(validation.values())
    results = {{
        "model_name": MODEL_NAME,
        "framework": "stan",
        "model_kind": "discrete",
        "num_timesteps": T,
        "num_states": S,
        "num_observations": O,
        "num_actions": U,
        "beliefs": beliefs.tolist(),
        "observations": obs,
        "actions": acts,
        "true_states": states,
        "A_posterior_mean": a_post.tolist(),
        "A_declared": A.tolist(),
        "A_posterior_abs_error_mean": float(np.mean(np.abs(a_post - A))),
        "log_marginal_mean": float(np.mean(np.asarray(draws["log_marginal"]))),
        "rhat_max_A_est": rhat_max,
        "policy": "uniform_random_exploration",
        "inference": inference,
        "num_a_est_parameters": num_params,
        "validation": validation,
        "cmdstan_version": str(cmdstan_version),
        "cmdstanpy_version": cmdstanpy.__version__,
        "execution_time_seconds": round(time.time() - start, 4),
    }}
    out = output_dir / "simulation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Stan HMM inference complete ({{inference}}): T={{T}} S={{S}} O={{O}} U={{U}} rhat_max={{rhat_max:.3f}}")
    print(f"Results saved to: {{out}}")
    print(f"Validation: {{validation}}")
    return 0 if validation["all_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
'''


# ---------------------------------------------------------------------------
# Continuous linear-Gaussian state-space model
# ---------------------------------------------------------------------------


def _continuous_program() -> str:
    return """// Generated by the GNN pipeline — Stan renderer (continuous LGSSM).
//
// x_1 ~ N(prior_mean, prior_cov);  x_t = F x_{t-1} + u_{t-1} + N(0, Q);
// y_t = H x_t + N(0, obs_noise_scale * R).
// The Kalman-filter marginal likelihood is written out explicitly so the
// controls u (data) enter the prediction step; obs_noise_scale is the free
// parameter. Filtered means/covariances are exposed as transformed parameters.
data {
  int<lower=1> T;
  int<lower=1> n;
  int<lower=1> m;
  matrix[n, n] F;
  matrix[m, n] H;
  cov_matrix[n] Q;
  cov_matrix[m] R;
  vector[n] prior_mean;
  cov_matrix[n] prior_cov;
  array[T] vector[m] y;
  array[T] vector[n] u;                        // u[t] applied when predicting t+1
}
parameters {
  real<lower=0> obs_noise_scale;
}
transformed parameters {
  array[T] vector[n] mu;
  array[T] matrix[n, n] P;
  real log_lik = 0;
  {
    matrix[m, m] Rs = obs_noise_scale * R;
    matrix[n, n] I_n = diag_matrix(rep_vector(1.0, n));
    vector[n] mu_pred = prior_mean;
    matrix[n, n] P_pred = prior_cov;
    for (t in 1:T) {
      if (t > 1) {
        mu_pred = F * mu[t - 1] + u[t - 1];
        P_pred = F * P[t - 1] * F' + Q;
      }
      matrix[m, m] S_t = H * P_pred * H' + Rs;
      matrix[n, m] K = P_pred * H' / S_t;
      vector[m] innovation = y[t] - H * mu_pred;
      log_lik += multi_normal_lpdf(y[t] | H * mu_pred, S_t);
      mu[t] = mu_pred + K * innovation;
      matrix[n, n] IKH = I_n - K * H;
      P[t] = IKH * P_pred * IKH' + K * Rs * K';   // Joseph form
    }
  }
}
model {
  obs_noise_scale ~ lognormal(0, 0.5);
  target += log_lik;
}
"""


def _continuous_driver(spec: Any, stan_name: str) -> str:
    from gnn.render.continuous_common import literal_block

    lits = literal_block(spec)
    return f'''#!/usr/bin/env python3
"""
Stan continuous (linear-Gaussian state-space) driver: {spec.model_name}

Auto-generated by the GNN pipeline — Stan renderer (continuous branch).
1. Simulate the LGSSM with a numpy Kalman filter in the loop (closed-loop
   proportional control when goal_mean/control_gain are declared).
2. Compile the sibling Stan program with CmdStan; sample obs_noise_scale and
   recover the filtered posterior means/covariances as transformed parameters.
3. Write simulation_results.json in the continuous result schema.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

MODEL_NAME = {spec.model_name!r}
STAN_FILE = Path(__file__).resolve().with_name({stan_name!r})
RANDOM_SEED = {spec.random_seed}
T = {spec.num_timesteps}
F = np.asarray({lits["F"]}, dtype=float)
H = np.asarray({lits["H"]}, dtype=float)
Q = np.asarray({lits["Q"]}, dtype=float)
R = np.asarray({lits["R"]}, dtype=float)
PRIOR_MEAN = np.asarray({lits["prior_mean"]}, dtype=float)
PRIOR_COV = np.asarray({lits["prior_cov"]}, dtype=float)
GOAL_MEAN = {"np.asarray(" + lits["goal_mean"] + ", dtype=float)" if spec.has_control else "None"}
CONTROL_GAIN = {lits["control_gain"]}
n, m = F.shape[0], H.shape[0]


def kalman_step(mu, P, y_t, u_prev, first):
    if first:
        mu_pred, P_pred = mu, P
    else:
        mu_pred = F @ mu + u_prev
        P_pred = F @ P @ F.T + Q
    S_t = H @ P_pred @ H.T + R
    K = np.linalg.solve(S_t.T, (P_pred @ H.T).T).T
    mu_new = mu_pred + K @ (y_t - H @ mu_pred)
    IKH = np.eye(n) - K @ H
    return mu_new, IKH @ P_pred @ IKH.T + K @ R @ K.T


def simulate(rng):
    xs, ys, mus, us = [], [], [], []
    x = rng.multivariate_normal(PRIOR_MEAN, PRIOR_COV)
    y = H @ x + rng.multivariate_normal(np.zeros(m), R)
    mu, P = kalman_step(PRIOR_MEAN, PRIOR_COV, y, np.zeros(n), True)
    u = CONTROL_GAIN * (GOAL_MEAN - mu) if GOAL_MEAN is not None else np.zeros(n)
    xs.append(x); ys.append(y); mus.append(mu); us.append(u)
    for _ in range(1, T):
        x = F @ x + u + rng.multivariate_normal(np.zeros(n), Q)
        y = H @ x + rng.multivariate_normal(np.zeros(m), R)
        mu, P = kalman_step(mu, P, y, u, False)
        u = CONTROL_GAIN * (GOAL_MEAN - mu) if GOAL_MEAN is not None else np.zeros(n)
        xs.append(x); ys.append(y); mus.append(mu); us.append(u)
    return np.asarray(xs), np.asarray(ys), np.asarray(mus), np.asarray(us)


def main():
    start = time.time()
    try:
        import cmdstanpy
    except ImportError:
        print("ERROR: cmdstanpy not installed. Install with: uv sync --extra stan")
        return 1
    try:
        cmdstan_version = cmdstanpy.cmdstan_version()
    except Exception as exc:
        print(f"ERROR: CmdStan toolchain not available: {{exc}}")
        return 1

    rng = np.random.default_rng(RANDOM_SEED)
    xs, ys, kf_mus, us = simulate(rng)
    output_dir = Path(os.environ.get("STAN_OUTPUT_DIR", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {{
        "T": T, "n": n, "m": m,
        "F": F.tolist(), "H": H.tolist(), "Q": Q.tolist(), "R": R.tolist(),
        "prior_mean": PRIOR_MEAN.tolist(), "prior_cov": PRIOR_COV.tolist(),
        "y": ys.tolist(), "u": us.tolist(),
    }}
    data_path = output_dir / (STAN_FILE.stem + "_data.json")
    with open(data_path, "w") as f:
        json.dump(data, f)

    model = cmdstanpy.CmdStanModel(stan_file=str(STAN_FILE))
    fit = model.sample(
        data=str(data_path), chains=1, iter_warmup=300, iter_sampling=300,
        seed=RANDOM_SEED, show_progress=False, show_console=False,
        output_dir=str(output_dir / "cmdstan"),
    )
    draws = fit.stan_variables()
    mu_draws = np.asarray(draws["mu"])                 # draws x T x n
    P_draws = np.asarray(draws["P"])                   # draws x T x n x n
    beliefs = mu_draws.mean(axis=0)
    covs = P_draws.mean(axis=0)
    scale_draws = np.asarray(draws["obs_noise_scale"])
    summary = fit.summary()
    rhat_col = "R_hat" if "R_hat" in summary.columns else summary.columns[-1]
    rhat = float(summary.loc["obs_noise_scale", rhat_col]) if "obs_noise_scale" in summary.index else float("nan")
    rmse = float(np.sqrt(np.mean((beliefs - xs) ** 2)))
    psd = all(np.all(np.linalg.eigvalsh((c + c.T) / 2) > -1e-9) for c in covs)
    validation = {{
        "means_finite": bool(np.all(np.isfinite(beliefs))),
        "posterior_cov_psd": bool(psd),
        "rmse_finite": bool(np.isfinite(rmse)),
        "controls_finite": bool(np.all(np.isfinite(us))),
        "rhat_ok": bool(np.isfinite(rhat) and rhat < 1.1),
    }}
    validation["all_valid"] = all(validation.values())
    results = {{
        "model_name": MODEL_NAME,
        "framework": "stan",
        "model_kind": "continuous",
        "num_timesteps": T,
        "num_states": n,
        "num_observations": m,
        "beliefs": beliefs.tolist(),
        "posterior_cov": covs.tolist(),
        "true_states_continuous": xs.tolist(),
        "observations_continuous": ys.tolist(),
        "controls": us.tolist(),
        "control_mode": "closed_loop_proportional" if GOAL_MEAN is not None else "passive",
        "kalman_filter_means": kf_mus.tolist(),
        "rmse_vs_true": rmse,
        "obs_noise_scale_posterior_mean": float(scale_draws.mean()),
        "obs_noise_scale_rhat": rhat,
        "observations": [],
        "actions": [],
        "efe_history": [],
        "validation": validation,
        "cmdstan_version": str(cmdstan_version),
        "cmdstanpy_version": cmdstanpy.__version__,
        "execution_time_seconds": round(time.time() - start, 4),
    }}
    out = output_dir / "simulation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Stan LGSSM inference complete: T={{T}} n={{n}} m={{m}} rmse_vs_true={{rmse:.4f}} rhat={{rhat:.3f}}")
    print(f"Results saved to: {{out}}")
    print(f"Validation: {{validation}}")
    return 0 if validation["all_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
'''


# ---------------------------------------------------------------------------
# Structural skeleton (declaration-only sketch of the variable graph; not executable)
# ---------------------------------------------------------------------------


def render_stan(
    variables: List[Dict[str, Any]],
    connections: List[Dict[str, Any]],
    model_name: str = "gnn_model",
) -> str:
    """Emit a *structural* Stan skeleton from variables/connections.

    This is a declaration-only sketch (data/parameters blocks plus commented
    edges) useful for inspecting a model's variable graph. It is not a
    runnable inference program — the pipeline uses
    :func:`render_gnn_to_stan` for that.
    """
    data_vars: list[str] = []
    param_vars: list[str] = []
    edge_lines: list[str] = []
    for v in variables:
        name = v.get("name", "x")
        stan_type = _stan_type(
            v.get("dtype", v.get("type", "real")), v.get("dimensions", [])
        )
        if name.lower() in ("o", "obs", "y", "data", "u", "t"):
            data_vars.append(f"  {stan_type} {name};")
        else:
            param_vars.append(f"  {stan_type} {name};")
    for conn in connections:
        src, tgt = conn.get("source", ""), conn.get("target", "")
        arrow = "→" if conn.get("directed", True) else "—"
        edge_lines.append(f"  // {src} {arrow} {tgt}")
    lines = [
        f"// Stan structural skeleton generated from GNN: {model_name}",
        f"// Variables: {len(variables)}, Connections: {len(connections)}",
        "// Declaration-only sketch; see render_gnn_to_stan() for the runnable program.",
        "",
        "data {",
        *(data_vars or ["  // No observed variables declared"]),
        "}",
        "",
        "parameters {",
        *(param_vars or ["  // No parameters declared"]),
        "}",
        "",
        "model {",
        *(edge_lines or ["  // No connections to model"]),
        "}",
    ]
    code = "\n".join(lines)
    logger.info(
        f"🔧 Stan skeleton generated: {len(data_vars)} data, {len(param_vars)} params"
    )
    return code


def _stan_type(dtype: str, dims: list) -> str:
    """Map GNN type+dims to Stan type declaration."""
    base = "real" if dtype in ("float", "double", "real") else "int"
    valid_dims = [d for d in dims if isinstance(d, int) and d > 0]
    if not valid_dims:
        return base
    if len(valid_dims) == 1:
        return f"vector[{valid_dims[0]}]"
    if len(valid_dims) == 2:
        return f"matrix[{valid_dims[0]}, {valid_dims[1]}]"
    inner = f"matrix[{valid_dims[-2]}, {valid_dims[-1]}]"
    for d in reversed(valid_dims[:-2]):
        inner = f"array[{d}] {inner}"
    return inner
