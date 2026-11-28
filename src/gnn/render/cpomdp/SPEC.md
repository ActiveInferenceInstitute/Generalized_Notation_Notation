# cpomdp Render Backend Specification

## Public API

`gnn.render.cpomdp` exports `render_gnn_to_cpomdp(gnn_spec, output_path, options=None) -> tuple[bool, str, list[str]]` and `UNSUPPORTED_MESSAGE`.

## Inputs

- `gnn_spec` is the continuous spec (`model_kind == "continuous"`). It carries `F`, `H`, `Q`, `R`, `prior_mean` and `prior_cov`, optionally `goal_mean` with `control_gain`, and `model_parameters` with `num_timesteps` and `random_seed`.
- `options` are listed in [README.md](README.md). An invalid value fails the render with a message naming the option. Booleans, fractional horizons and nonfinite values are rejected. Admission ceilings can be lowered using `max_policies`, `max_step_evaluations` and `max_estimated_bytes`; they cannot be raised above the maintained limits. Unknown keys are rejected.

## Outputs

- A continuous spec writes one `.py` file at `output_path` and returns `(True, message, [str(output_path)])`.
- A discrete spec returns `(False, UNSUPPORTED_MESSAGE, [])` and writes nothing.

When run, the script writes `simulation_results.json` under `CPOMDP_OUTPUT_DIR` (default `.`). It holds the shared continuous schema: `model_name`, `framework` (`"cpomdp"`), `model_kind` (`"continuous"`), `num_timesteps`, `num_states`, `num_observations`, `beliefs`, `posterior_cov`, `true_states_continuous`, `observations_continuous`, `controls`, `control_mode`, `rmse_vs_true`, `observations` (`[]`), `actions` (`[]`), `validation` and `execution_time_seconds`.

cpomdp adds `efe_history` (one row with one value per scored policy, empty when passive), `epistemic_term`, `pragmatic_term`, `selected_policy_index`, `horizon`, `action_scale`, `n_policies`, `cost_per_cycle`, `search_mode`, `search_certificate`, `jax_version` and `cpomdp_version`. For exhaustive search, `search_certificate` carries the library's own `expected`, `visited`, `warrant`, `action_set_size`, `horizon` and `action_set_version` fields plus GNN's `claim`.

`validation` holds `means_finite`, `posterior_cov_psd`, `rmse_finite`, `controls_finite`, `efe_finite` and `all_valid`. The script exits 1 when `all_valid` is false.

## Admission and identity

The default versioned action set has nine actions for two control dimensions, otherwise `2p+1`. Passive models perform no search. The conservative estimate is recorded under `admission`, together with policy and step counts. An exhaustive run admits `num_timesteps + 1` planning cycles because the native `evaluate` pass used for its certificate adds a cycle. Candidate search has one cycle per timestep. `measured_resources` records peak process RSS and elapsed time.

`source_identity` binds the validated numeric contract; pipeline runs additionally bind the original relative path, source SHA and model ID. `generated_script_sha256`, `cpomdp_version`, `jax_version`, `action_set` and `selected_policy` retain execution identity and decisions. These hashes establish identity, not extraction correctness or universal equivalence.

## Source controls

The optional `initialparameterization.cpomdp_control` mapping follows the
[E3-FA1 contract](../../../../docs/gnn/implementations/cpomdp.md#source-declared-finite-controls-e3-fa1).
It admits only flat continuous EFE models, preserves ordered action identities
and explicit asymmetric control maps, and adds complete per-policy component
receipts. `controls` retain state increments while `control_inputs` retain
authored control coordinates. Other backends refuse this declaration.
The total evaluation ceiling is 262,144 across all timesteps; passive models
retain zero search cost. The numeric model digest includes all control metadata.

The optional `initialparameterization.cpomdp_search` mapping follows the
[candidate sequence contract](../../../../docs/gnn/implementations/cpomdp.md#source-declared-candidate-sequences).
It retains the ordered source declaration under `candidate_search_contract`, one
score validity flag per policy and timestep under `policy_score_valid`, and the
selected candidate's action IDs under `policy_action_ids`. The candidate horizon
and ordering contribute to the numeric model digest. Candidate receipts have a
null `search_certificate`.
