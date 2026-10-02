# Stan Executor (`src/gnn/execute/stan/`)

## Purpose

Execute the Stan artifacts produced by `src/gnn/render/stan/`. Each rendered model
ships a `<stem>_stan.stan` program and a sibling `<stem>_stan.py` cmdstanpy
driver. Step 12 discovers the driver like any other Python framework script
(framework directory `stan/`, output env var `STAN_OUTPUT_DIR`) and runs it.

## What the driver does

1. Simulates a trajectory from the GNN generative model with a fixed seed
   (discrete: A/B/D with a uniform-random exploratory policy; continuous:
   LGSSM with an in-loop Kalman filter and optional closed-loop control).
2. Writes `<stem>_stan_data.json`, compiles the `.stan` program with CmdStan
   (`cmdstanpy.CmdStanModel`), then samples with NUTS (1 chain, 300 warmup,
   300 draws, seeded) or, for discrete models with more than
   `stan_nuts_param_budget` (default 1024) simplex parameters, fits the MAP
   with L-BFGS.
3. Writes `simulation_results.json` — discrete: filtered beliefs (T×S),
   learned `A_posterior_mean` (O×S), `rhat_max_A_est`; continuous: posterior
   means/covariances, `obs_noise_scale_posterior_mean`, `rmse_vs_true`.
   Exit code 0 iff `validation.all_valid`.

## Dependency gating

`gnn.utils.runtime_safety.framework_availability` maps `stan` → `cmdstanpy`. When it is not
importable Step 12 marks the script **skipped** with the reason
`uv sync --extra stan`. CmdStan itself is installed with
`python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"`.

## Public API

| Symbol | Purpose |
|---|---|
| `is_stan_available()` | cmdstanpy importable and CmdStan toolchain found |
| `find_stan_scripts(render_dir)` | every `*_stan.py` driver under a render tree |
| `execute_stan_script(script, out_dir)` | run one driver with `STAN_OUTPUT_DIR` set |
| `run_stan_scripts(render_dir, out_dir)` | run all drivers, or skip all with a reason |

Both runner functions accept an optional `cancel_token`. Preconditions precede
availability probes; readiness uses the shared bounded child probe in the
selected Python interpreter. The shared script security gate also precedes
readiness and execution; blocked scripts retain findings and never run.
Missing dependencies and toolchains produce skipped
receipts, while probe failures retain their cause as failed work. Cancellation
and timeout use the shared subprocess envelope; per-driver receipts preserve
the original error category, partial streams, and observed process cleanup
evidence. The invocation deadline bounds all work; a standalone driver's timeout
applies to its command and separately caps its readiness probe.

## Tests

`tests/execute/test_execute_stan.py` covers discovery and actual child-process
cancellation/timeout without CmdStan. Native compilation and sampling use the
explicit `needs_cmdstan` integration test.
