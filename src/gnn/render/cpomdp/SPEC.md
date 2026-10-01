# cpomdp Render Backend Specification

## Public API

`gnn.render.cpomdp` exports `render_gnn_to_cpomdp(gnn_spec, output_path, options=None) -> tuple[bool, str, list[str]]` and `UNSUPPORTED_MESSAGE`.

## Inputs

- `gnn_spec` is the continuous spec (`model_kind == "continuous"`). It carries `F`, `H`, `Q`, `R`, `prior_mean` and `prior_cov`, optionally `goal_mean` with `control_gain`, and `model_parameters` with `num_timesteps` and `random_seed`.
- `options` are listed in [README.md](README.md). An invalid value fails the render with a message naming the option. Unknown keys are ignored.

## Outputs

- A continuous spec writes one `.py` file at `output_path` and returns `(True, message, [str(output_path)])`.
- A discrete spec returns `(False, UNSUPPORTED_MESSAGE, [])` and writes nothing.

When run, the script writes `simulation_results.json` under `CPOMDP_OUTPUT_DIR` (default `.`). It holds the shared continuous schema: `model_name`, `framework` (`"cpomdp"`), `model_kind` (`"continuous"`), `num_timesteps`, `num_states`, `num_observations`, `beliefs`, `posterior_cov`, `true_states_continuous`, `observations_continuous`, `controls`, `control_mode`, `rmse_vs_true`, `observations` (`[]`), `actions` (`[]`), `validation` and `execution_time_seconds`.

cpomdp adds `efe_history` (one row of `|A|^H` values per step, empty when passive), `epistemic_term`, `pragmatic_term`, `selected_policy_index`, `horizon`, `action_scale`, `n_policies`, `cost_per_cycle`, `search_warrant`, `jax_version` and `cpomdp_version`.

`validation` holds `means_finite`, `posterior_cov_psd`, `rmse_finite`, `controls_finite`, `efe_finite` and `all_valid`. The script exits 1 when `all_valid` is false.
