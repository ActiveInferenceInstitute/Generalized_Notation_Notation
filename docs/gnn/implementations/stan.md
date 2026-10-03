# Stan Framework Implementation

> **GNN Integration Layer**: Stan probabilistic programming language
> **Framework Base**: Stan (CmdStan ≥ 2.33 via `cmdstanpy`)
> **Documentation Version**: v3.2.0 Engine (Bundle v2.0.0)
> **Scope**: Runnable inference programs for both GNN model kinds — a
> forward-algorithm HMM/POMDP with likelihood learning for discrete models and
> an explicit Kalman marginal likelihood for continuous linear-Gaussian models —
> each paired with a cmdstanpy driver that Step 12 executes.

## Overview

The Stan renderer (`src/gnn/render/stan/stan_renderer.py`,
`render_gnn_to_stan(gnn_spec, output_path, options)`) emits two artifacts per
model with one stem: `<stem>_stan.stan` (the program) and `<stem>_stan.py`
(the driver). The driver simulates a trajectory from the GNN's own generative
model with `ModelParameters.random_seed`, writes `<stem>_stan_data.json`,
compiles the program with CmdStan, runs inference and writes
`simulation_results.json`. Step 12 discovers the driver like any other Python
framework script (framework directory `stan/`, env var `STAN_OUTPUT_DIR`).

Stan does not run an online Active Inference loop (no expected-free-energy
policy selection): the agent's actions are *data*. What Stan contributes is
posterior inference over the generative model given a trajectory — learning
the likelihood in the discrete case, the observation-noise scale in the
continuous case — with proper diagnostics.

## Discrete POMDP / HMM program

| Block | Contents |
|-------|----------|
| `data` | `T, S, O, U`; `o[T]`, `u[T-1]` (1-based); `B[U]` matrices `[s_next, s_prev]` (column-stochastic); `alpha_A[S]` Dirichlet pseudo-counts; `D` simplex |
| `parameters` | `A_est[S]` — one simplex over outcomes per hidden state (P(o \| s)) |
| `transformed parameters` | `filtered_state[T]`, `log_marginal`: vectorised scaled forward recursion `alpha_t ∝ A[o_t,:] .* (B[u_{t-1}] alpha_{t-1})` |
| `model` | `A_est[s] ~ dirichlet(alpha_A[s])`; `target += log_marginal` |

The pseudo-counts are `1 + strength · A[:, s]` with
`ModelParameters.stan_dirichlet_strength` (default 20). Inference is NUTS
(1 chain, 300 warmup, 300 draws) when `S·O ≤ stan_nuts_param_budget`
(default 1024) and L-BFGS MAP (`jacobian=True`) otherwise, so large joint
compositions (the 729-state stigmergic swarm: 46 656 simplex parameters)
finish in seconds. Results: `beliefs` (T×S filtered posterior means),
`A_posterior_mean` (O×S), `A_declared`, `rhat_max_A_est` (NUTS) or
`map_converged` (MAP), `log_marginal_mean`.

## Continuous linear-Gaussian program

| Block | Contents |
|-------|----------|
| `data` | `T, n, m`; `F, H, Q, R, prior_mean, prior_cov`; `y[T]` observations; `u[T]` controls |
| `parameters` | `obs_noise_scale > 0` (scales `R`) |
| `transformed parameters` | `mu[T]`, `P[T]`, `log_lik`: Kalman predict/update with Joseph-form covariance; `multi_normal_lpdf` of the innovations |
| `model` | `obs_noise_scale ~ lognormal(0, 0.5)`; `target += log_lik` |

The driver simulates the LGSSM with a numpy Kalman filter in the loop and, when
the GNN declares `goal_mean`/`control_gain`, closes the loop on beliefs
(`u_t = gain · (goal − μ_t)`), exactly as the JAX / NumPyro / PyTorch / RxInfer
continuous scripts do. Results follow the continuous schema (`beliefs`,
`posterior_cov`, `true_states_continuous`, `observations_continuous`,
`controls`, `rmse_vs_true`, `obs_noise_scale_posterior_mean`).

## Installation

```bash
uv sync --extra stan                                   # cmdstanpy
uv run python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"   # CmdStan toolchain (once)
uv run python src/gnn/main.py --only-steps "3,11,12" --target-dir input/gnn_files/discrete
```

CmdStan installs under `~/.cmdstan` by default and requires a working C++
compiler and build tools. The download is explicitly opt-in; neither Step 1
nor a readiness probe invokes `install_cmdstan()`. Record the installed
CmdStan version alongside execution receipts.

Without `cmdstanpy` *and* a CmdStan toolchain, Step 12 marks Stan scripts as
`skipped` with the install hint (`gnn.utils.runtime_safety.framework_availability` probes
`cmdstanpy.cmdstan_path()`); a missing toolchain is recorded with `reason_code: missing_toolchain`,
while a missing driver uses `missing_module`. An import failure or the bounded
30-second readiness timeout uses `probe_failed` or `probe_timeout`, without
claiming that the package is absent.

## Structural sketch (`render_stan`)

`render_stan(variables, connections, model_name)` remains available as a
declaration-only sketch of a model's variable graph (`data`/`parameters`
blocks plus commented edges). It is not an inference program and the pipeline
does not execute it.

## Verification

The exemplar census (`input/gnn_files/INDEX.md`, measured 2026-09-17) counts 30
runnable specs: 27 discrete (rendered to HMM programs, NUTS or MAP by budget)
and 3 continuous (rendered to LGSSM programs). Per-run render/execute outcomes
and per-model validation flags live in the Step 11/12 summaries, not here.
Tests: `tests/render/test_render_stan.py`,
`tests/execute/test_execute_stan.py`,
`tests/render/test_continuous_renderers.py` (compile/sample steps skip
when CmdStan is absent).
