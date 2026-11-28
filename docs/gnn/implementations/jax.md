# JAX Framework Implementation

> **GNN Integration Layer**: Python / XLA
> **Framework Base**: `jax` & `jax.numpy`
> **Simulation Architecture**: Online True POMDP Generative Model
> **Documentation Version**: 2.0.0

## Overview

The Generalized Notation Notation (GNN) pipeline translates theoretical model specifications into heavily optimized, compiled Python code natively utilizing the `jax` ecosystem. As the primary high-performance numerical computing target, JAX serves to evaluate Active Inference mechanics across enormous vector spaces at computational speeds orders of magnitude faster than baseline python execution.

This document details the exact mechanisms through which a GNN JSON specification is extracted, built into a JAX pseudo-Agent node, evaluated continuously within an independent generative POMDP environment, and serialized into high-fidelity telemetry artifacts.
**Registry capability** (`src/gnn/render/framework_registry.py`): JAX covers both
model kinds — discrete categorical POMDP specs (A/B/C/D[/E]) and continuous
linear-Gaussian state-space models (`supports_continuous: True`, dispatched by
`detect_model_kind`) — and is fully executable (`supports_execution: True`):
Step 12 runs each generated script via `src/gnn/execute/jax/jax_runner.py`.

## Architecture

The JAX implementation operates entirely on stateless mathematical function transformations:

1. **Parameter Parsing**: `pomdp_processor.py` (Translating GNN variable states into multidimensional numpy matrices)
2. **Generative Loop Generation**: `jax_renderer.py` (Building the target python script defining functional state transitions and environmental loops)
3. **Execution Context**: `jax_runner.py` (Executing the generated python script and managing filesystem persistence)

### Model Initialization & Variable Extraction

Unlike object-oriented systems (like PyMDP), JAX initializes parameter matrices universally as immutable `jnp.ndarray` structures stored within a monolithic parameters dictionary `params`.

GNN dynamically calculates explicit dimensional mappings:

- **State Space (`num_states`)**: Inferred dynamically from `A_matrix.shape[1]`
- **Action Space (`num_actions`)**: Inferred dynamically from `B_matrix.shape[2]`
- **Observation Space (`num_obs`)**: Inferred dynamically from `A_matrix.shape[0]`

```python
# Functional architecture utilized by GNN to manage stateless execution
params = {
    "A_matrix": jnp.array(A_matrix),
    "B_matrix": jnp.array(B_matrix),
    "C_vector": jnp.array(C_vector),
    "D_vector": jnp.array(D_vector),
}
```

## Perception-Action Loop (The Generative Process)

JAX employs a pseudo-random number generation (PRNG) paradigm requiring explicit random keys (`jax.random.PRNGKey`). Maintaining true causal separation between the environment and the stateless agent functions requires tracking a generative loop managed completely independently of the JAX `belief` array.

1. **Initialize Environmental True State**:
   The `run_simulation` structure triggers the true environment by splitting the primary sequence key and sampling the Prior definition `D`.

   ```python
   # Stochastic draw from Prior distribution
   key, subkey = jax.random.split(key)
   true_state_idx = jax.random.categorical(subkey, jnp.log(params["D_vector"] + 1e-8))
   ```

2. **Step-wise Environment Generation**:
   The environment explicitly pulls an observation index representing an emission from the true state index using the Likelihood `A` matrix. JAX handles this via log-probability categorical mapping. This prevents arbitrary coupling of the agent's expectation and the genuine environment structure.

   ```python
   # Stochastic draw from A matrix conditioned on current True State
   key, subkey = jax.random.split(key)
   obs_probs = params["A_matrix"][:, true_state_idx]
   obs_idx = jax.random.categorical(subkey, jnp.log(obs_probs + 1e-8))

   # Translate integer emission to binary one-hot representation for inner agent equations
   obs_one_hot = jnp.zeros(params["A_matrix"].shape[0])
   obs_one_hot = obs_one_hot.at[obs_idx].set(1.0)
   ```

3. **Inference and Control Calculation (Agent)**:
   The pseudo-agent ingests the explicit one-hot `obs_one_hot` array inside `simulate_step`.

   ```python
   # JIT-optimized internal functional composition
   result = simulate_step(params, belief, obs_one_hot)

   # Explicit data serialization
   actions.append(result["action"])
   efes.append(result["all_efe_values"])
   ```

4. **Environment Transition**:
   Following the pure active inference formulation, the physical environment transitions inside real spacetime based on the selected action interacting against the generative structure (the `B` matrix probabilities mapped to the historical true state).

   ```python
   # Stochastic decay of the actual true_state index evaluated against JAX matrix
   key, subkey = jax.random.split(key)
   next_state_probs = params["B_matrix"][:, true_state_idx, action_idx]
   true_state_idx = jax.random.categorical(subkey, jnp.log(next_state_probs + 1e-8))
   ```

## Expected Free Energy Mechanics

JAX functional operations mathematically parallel standard active inference metrics for Expected Free Energy `G(π)`.

1. **Ambiguity** (Expected Uncertainty): `jnp.sum(A_matrix * jnp.log(A_matrix + 1e-16), axis=0)`
2. **Risk** (Expected Divergence): `jnp.sum(predicted_obs * jnp.log(predicted_obs / C_vector))`

*Important Tracking Detail*: As a native tensor execution engine, JAX computes expected free energy concurrently across the entire dimensional depth of the action space (`num_actions`). This results in explicit 1D arrays matching the length of available behaviors for every single timestep analyzed. JAX relies on `jax.nn.softmax` scaled by an inverse precision hyperparameter over the **negative** EFE array to evaluate categorical choices. `argmax` represents the final executed action index.

## Continuous (linear-Gaussian) models

When `is_continuous_spec` (`src/gnn/render/continuous_common.py`) recognises a
continuous specification (`F`/`H`/`Q`/`R`, `prior_mean`/`prior_cov`, optional
`u`, `goal_mean`, `control_gain`), `render_gnn_to_jax` delegates to the shared
generator `src/gnn/render/continuous_script.py` instead of the POMDP template. The
emitted script runs an online Kalman filter (Joseph-form covariance update) and,
when `goal_mean`/`control_gain` are present, closes the control loop with
`u_t = control_gain · (goal_mean − μ_t)`. Results follow the continuous schema:
`beliefs` (posterior means), `posterior_cov`, `true_states_continuous`,
`observations_continuous`, `controls`, and `rmse_vs_true`. The exemplars under
`input/gnn_files/continuous/` exercise this path.

### Two coupled Gaussian agents

Set `agent_coupling: gnn_joint_gaussian_filter_v1` and `nr_agents: 2` in
ModelParameters. InitialParameterization declares two ordered agent IDs and
their state, observation, and control axis labels. The resulting offsets
specify the rows and columns of the full joint F, H, Q, R, prior covariance,
and control matrix. Supply every cross-agent entry explicitly, including
zeros. Q, R, and the prior covariance are symmetric positive definite.

The prior describes the first observed state. Inference at time t uses
observations through t. Each fixed control at t contributes
`control_matrix @ control_sequence[t]` to the state at t+1. Supply one
observation per timestep and one control per transition. Process and
observation noise are independent across times and series. Their joint
covariances allow correlation between agents within each series.

The JAX runner records full posterior means, covariances, cross-agent
covariance blocks, per-agent views, and the ordered axis identities. Joint
state, observation, and control dimensions can each be at most 16. The
contract allows at most 128 observations. The output uses the
`gnn_joint_gaussian_filter_v1` schema. The independently declared agent
contract remains available for separate models. RxInfer and other targets
explicitly refuse the joint filtering contract. Read the complete
[coupled Gaussian source](../../../input/gnn_files/continuous/joint_gaussian_agents.md).

## Telemetry & Logging Output

At runtime, the JAX execution context compiles all trajectory metrics internally inside its loop arrays (such as the `observations_log`), avoiding arbitrary dictionary references prior to serialization. It writes telemetry to the directory named by the `JAX_OUTPUT_DIR` environment variable (set by Step 12 to the per-model `<model>/jax/` execution folder); the renderer's built-in default is `output/jax_simulations/<model>/simulation_data`.

**Logged Vectors**:

- `beliefs`: Array depth `[num_timesteps, num_states]` recording Bayesian posterior likelihoods.
- `actions`: Array length `[num_timesteps]` composed of integer behavior selections.
- `observations`: Array length `[num_timesteps]`. Explicit emissions produced by the causal observation environment step loop prior to agent filtering.
- `efe_history`: Tensor `[num_timesteps, num_actions]` recording all internal negative divergence limits calculated for every potential branch on every timestep.

```json
"simulation_trace": {
  "observations": [1, 1, 1, 0, 2],
  "beliefs": [[0.05, 0.89, 0.05], [0.003, 0.99, 0.003], ...],
  "actions": [0, 0, 0, 1, 2],
  "efe_history": [
    [-0.405, -0.405, -1.96], 
    [-0.405, -0.405, -1.45], 
    ...
  ] 
}
```

The compiled JAX simulation traces achieve deterministic equivalence against native PyMDP implementations operating with identical pseudo-random seeding indices.

---

## Source Code Connections

| Pipeline Stage | Module | Key Function | Lines |
|---|---|---|---|
| Rendering | [jax_renderer.py](../../../src/gnn/render/jax/jax_renderer.py) | `render_gnn_to_jax()` | Entry point |
| Simulation Code Gen | [jax_renderer.py](../../../src/gnn/render/jax/jax_renderer.py) | `run_simulation` (template) | — |
| Results Serialization | [jax_renderer.py](../../../src/gnn/render/jax/jax_renderer.py) | `save_simulation_results` (template) | — |
| Execution | [jax_runner.py](../../../src/gnn/execute/jax/jax_runner.py) | `execute_jax_script()` | — |
| Device Selection | [jax_runner.py](../../../src/gnn/execute/jax/jax_runner.py) | `initialize_jax_devices()` | — |
| Continuous Code Gen | [continuous_script.py](../../../src/gnn/render/continuous_script.py) | `generate_continuous_script()` | — |
| Analysis | [analyzer.py](../../../src/gnn/analysis/jax/analyzer.py) | `generate_analysis_from_logs()` | — |
| Raw Output Parsing | [analyzer.py](../../../src/gnn/analysis/jax/analyzer.py) | `parse_raw_output()` | — |
| Cross-Framework | [visualizations.py](../../../src/gnn/analysis/visualizations.py) | `generate_efe_convergence_comparison()` | — |

---

## Improvement Opportunities

| ID | Area | Description | Impact |
|---|---|---|---|
| J-1 | Execution | ~~JAX runner lacked execution log persistence~~ — now saves `execution_log.json`, `stdout.txt`, `stderr.txt` | ✅ FIXED |
| J-2 | Rendering | `save_simulation_results` uses f-string template with `{{` escaping — complex and error-prone for maintenance | Medium |
| J-3 | Analysis | EFE multi-dimensional array requires `np.mean(axis=1)` collapse — could be standardized at serialization time | Low |
| J-4 | Execution | ~~No `JAX_OUTPUT_DIR` pass-through~~ — now sets `JAX_OUTPUT_DIR` env var matching `PYMDP_OUTPUT_DIR` pattern | ✅ FIXED |

## See Also / Next Steps

- **[Cross-Framework Methodology](../integration/framework_integration_guide.md)**: Details on the correlation methodology and benchmarking metrics.
- **[Architecture Reference](../reference/architecture_reference.md)**: Deep dive into the pipeline orchestrator and module integration.
- **[GNN Implementations Index](README.md)**: Return to the master framework implementer manifest.
- **[Back to GNN START_HERE](../../START_HERE.md)**
