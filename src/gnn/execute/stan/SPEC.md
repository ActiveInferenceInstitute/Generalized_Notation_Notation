# Stan Executor — Specification

## Inputs

- Rendered driver `…/<model>/stan/<stem>_stan.py` and program `<stem>_stan.stan`
  (same stem, same directory), produced by `render.stan.render_gnn_to_stan`.
- Environment: `STAN_OUTPUT_DIR` (results directory; default cwd).

## Behaviour

| Condition | Outcome |
|---|---|
| `cmdstanpy` not importable | Step 12 pre-flight marks the script `skipped` ("Dependency not installed: cmdstanpy", `skip_category` `missing_module`, hint `uv sync --extra stan`) |
| CmdStan toolchain missing | Step 12 pre-flight marks the script `skipped` with `skip_category` `probe_failed` ("Dependency probe failed for cmdstanpy: …", hint `python -c 'import cmdstanpy; cmdstanpy.install_cmdstan()'`); a direct driver run exits 1 with `CmdStan toolchain not available` |
| compile or sampling error | driver exits 1; stderr carries the CmdStan message |
| success | `simulation_results.json` written; exit 0 iff `validation.all_valid` |

## Result schema

Discrete (`model_kind: "discrete"`): `beliefs` T×S, `observations`, `actions`,
`true_states`, `A_posterior_mean` O×S, `A_declared`, `rhat_max_A_est`,
`log_marginal_mean`, `validation{beliefs_in_range, beliefs_sum_to_one,
a_posterior_columns_sum_to_one, rhat_ok, observations_in_range, all_valid}`.

Continuous (`model_kind: "continuous"`): `beliefs` T×n, `posterior_cov` T×n×n,
`true_states_continuous`, `observations_continuous`, `controls`,
`kalman_filter_means`, `rmse_vs_true`, `obs_noise_scale_posterior_mean`,
`validation{means_finite, posterior_cov_psd, rmse_finite, controls_finite,
rhat_ok, all_valid}`; discrete slots (`observations`, `actions`) are empty.

Both carry `cmdstan_version`, `cmdstanpy_version`, `execution_time_seconds`.
