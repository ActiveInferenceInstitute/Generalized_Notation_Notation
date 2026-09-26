<!-- markdownlint-disable MD013 -->
# RxInfer.jl Framework Implementation

> **GNN Integration Layer**: Julia
> **Framework Base**: `RxInfer.jl` (genuine `@model` + `infer()` variational message passing)
> **Simulation Architecture**: POMDP generative model via `@model` + `infer()`
> **Documentation Version**: 3.0.0

## Overview

The Generalized Notation Notation (GNN) pipeline translates theoretical model
specifications into executable Julia code natively utilizing the `RxInfer.jl`
ecosystem. RxInfer is a probabilistic programming framework built on
Forney-style factor graphs, where belief propagation is expressed as message
passing over graphical model edges. Within the GNN cross-framework comparison,
RxInfer serves as the primary Bayesian message-passing reference implementation
and is the only framework in the pipeline that performs inference through a
declarative probabilistic programming model (via the `@model` macro).

The canonical renderer (`src/gnn/render/rxinfer/rxinfer_renderer.py`) emits a genuine Julia
script per exemplar model that runs `infer()` with `free_energy = true` — no hand-rolled
step simulator. It does not emit one flat model shape for every spec: it calls
`detect_model_kind()` (`src/gnn/render/pomdp_contract.py`) and dispatches by the detected
`ModelKind` to a per-kind strategy in `src/gnn/render/rxinfer/model_strategies.py`. Detection
is *structural* — the `GNNSection` value, per-level and per-agent matrix key patterns,
explicit `nr_agents`/`num_factors`, `F`/`H`/`Q`/`R` keys, and `dirichlet_[A-E]` keys —
never free-text scanning of the source.

| `ModelKind` | Strategy | Generated model |
|---|---|---|
| `FLAT` | `FlatStrategy` | `pomdp_model` — batch smoothing by default, or per-timestep filtering in [online mode](#online-mode) |
| `HIERARCHICAL` | `HierarchicalStrategy` | `hierarchical_pomdp_model` — native two-level; the context latent enters the fast-state prior through the column-normalized `A_level2` (the model's `A_ctx` argument). Mean-field constraints *and* marginal initialization are both required on RxInfer 5.5. Three or more declared levels render as the documented joint composition |
| `FACTORED` | `FactoredStrategy` | `factored_pomdp_model` — native mean-field two-factor model with a multi-parent likelihood, `DiscreteTransition(s1, A_m0, s2)` |
| `CONTINUOUS` | `ContinuousStrategy` | `continuous_pomdp_model` — native linear-Gaussian state space from `F`/`H`/`Q`/`R` plus `prior_mean`/`prior_cov`. Beliefs are posterior *means* alongside `posterior_cov`, and VFE validation is sign-agnostic because a Gaussian Bethe free energy is routinely negative |
| `LEARNING` | `LearningStrategy` | `learning_pomdp_model` — `A` is learned jointly with the states as `DirichletCollection(dirichlet_A)`; `a_learning_improved` is a hard validation gate |
| `MULTI_AGENT` | `MultiAgentStrategy` | joint composition stamping the true kind. There is no native multi-agent `@model`; per-agent marginals are recovered downstream by `compute_per_factor_beliefs()` from the `state_factors` echo |

The table is maintained next to the code in
[`src/gnn/render/rxinfer/README.md`](../../../src/gnn/render/rxinfer/README.md). The TOML-emitting
`toml_generator.py` is retired and kept only as a warning surface —
`render_gnn_to_rxinfer_toml()` raises a `DeprecationWarning`.

This document details the full data flow from GNN specification through Julia factor
graph construction, genuine variational message-passing inference, real variational free
energy (VFE) capture, explicit Expected Free Energy (EFE) computation, and JSON telemetry
serialization.

## Architecture

The RxInfer implementation consists of three interconnected layers:

1. **Parameter Parsing**: `pomdp_processor.py` → `rxinfer_renderer.py`
   (Translating GNN variable states into Julia matrix literals)
2. **Script Generation**: the selected strategy in `model_strategies.py`
   (Building the Julia script: matrices, EFE functions, the generative loop,
   and result serialization)
3. **Execution Context**: `rxinfer_runner.py`
   (Spawning a Julia subprocess to execute the generated script)

### Where the `@model` blocks live

The `@model` definitions are **not** inlined into each generated script. They live in the
committed Julia package `src/gnn/execute/rxinfer/src/GnnRxInferModels.jl`, which defines
`pomdp_model`, `continuous_pomdp_model`, `hierarchical_pomdp_model`,
`factored_pomdp_model`, and `learning_pomdp_model`. A rendered script imports the one it
needs and calls it:

```julia
using GnnRxInferModels: pomdp_model

result = infer(
    model = pomdp_model(A=A, B=B, D=D, u=model_actions, T=TIME_STEPS),
    data = (y = obs_seq,),
    iterations = INFERENCE_ITERATIONS,
    free_energy = true
)
```

Holding the models in a package is what allows the environment to precompile them ahead
of execution. The generated script supplies the matrices, the simulation loop, action
selection, and telemetry.

### Source File

[rxinfer_renderer.py](../../../src/gnn/render/rxinfer/rxinfer_renderer.py)

---

## GNN Parameter Ingestion

### Dimensional Extraction

RxInfer extracts model dimensions from the GNN specification using a multi-source priority chain:

```python
# Priority chain for num_actions
num_actions = (
    model_params.get("num_actions")  # Explicit GNN model param
    or model_params.get("num_controls")  # Alternative GNN naming
    or model_params.get("n_actions")  # Previous naming convention
    or inferred_actions  # Inferred from B matrix depth
    or 3  # Hardcoded default
)
```

| GNN Parameter       | Julia Constant     | Extraction Source                              |
| ------------------- | ------------------ | ---------------------------------------------- |
| `num_hidden_states` | `NUM_STATES`       | `model_parameters.num_hidden_states`           |
| `num_obs`           | `NUM_OBSERVATIONS` | `model_parameters.num_obs`                     |
| `num_actions`       | `NUM_ACTIONS`      | Priority chain (see above)                     |
| `num_timesteps`     | `TIME_STEPS`       | `model_parameters.num_timesteps` (default: 20) |

### Matrix Literal Injection

GNN parameters are injected directly as Julia literal expressions into the generated script. Two utility functions handle runtime conversion:

- **`to_matrix(raw)`**: Converts nested Julia `Vector{Vector}` or `Tuple` structures into a proper `Matrix{Float64}` via `hcat()`.
- **`to_tensor(raw)`**: Converts 3-level nested structures into
  `Array{Float64, 3}` tensors for the B transition matrix indexing scheme
  `[next_state, prev_state, action]`.

### Matrix Normalization

| Matrix     | Normalization Rule                 | Purpose                                |
| ---------- | ---------------------------------- | -------------------------------------- | ----- |
| `A_matrix` | Column-sum to 1.0                  | Valid conditional probability `P(o\ \  | s)`   |
| `B_matrix` | Column-sum to 1.0 per action slice | Valid transition prob `P(s'\ \         | s,a)` |
| `D_vector` | Sum to 1.0                         | Valid prior distribution               |

### Preference Vector Transformation

The `C_vector` undergoes a critical transformation unique to RxInfer:

The raw GNN `C` values are treated as **log-preferences**
(unnormalized log-probabilities). The softmax transformation converts
these into a proper probability distribution used in the KL-divergence
risk term of the EFE computation.

---

## Perception-Action Loop (The Generative Process)

RxInfer implements a true POMDP generative environment, fully decoupled from
the agent's internal belief state. The process mirrors PyMDP's architecture.

### Step 1: Initialize Environmental True State

```julia
current_state = rand(Categorical(D_vector))
current_belief = copy(D_vector)
```

### Step 2: Environment Generates Observation

The observation is sampled stochastically from the likelihood column of `A`
corresponding to the true hidden state. The agent never has access to
`current_state`.

This is where RxInfer differs fundamentally from all other frameworks.
Belief updating is performed via a **declarative probabilistic model**:

```julia
@model function pomdp_model(y, A, B, D, u, T)
    s[1] ~ Categorical(D)
    y[1] ~ DiscreteTransition(s[1], A)
    for t in 2:T
        s[t] ~ DiscreteTransition(s[t-1], B[:, :, u[t-1]])
        y[t] ~ DiscreteTransition(s[t], A)
    end
end
```

The `@model` macro compiles this into a Forney-style factor graph. RxInfer then
runs variational message passing over the whole trajectory via `infer()` with
`free_energy = true`:

````julia
result = infer(
    model = pomdp_model(A=A_matrix, B=B_matrix, D=D_vector, u=action_seq, T=T),
    data = (y = observation_seq,),
    free_energy = true
)
posterior = result.posteriors[:s]        # Vector of Categorical posterior beliefs
vfe_trace = [Float64(f) for f in result.free_energy]  # genuine VFE per iteration
````

`result.posteriors[:s]` yields the per-timestep posterior beliefs, and
`result.free_energy` supplies the real variational free energy trace that feeds
the `variational_free_energy` field. The pipeline records `Random.seed!(seed)` and the
script SHA256 in `runtime_metadata`.

**There is no inference fallback.** The `infer()` call is deliberately *not* wrapped in
`try`/`catch`; every strategy emits the comment

```julia
# NO try/catch — if infer() fails, the script crashes with a clear error.
# This is deliberate: real RxInfer inference or nothing.
```

If variational message passing fails, the run fails loudly. Nothing silently degrades to
hand-rolled Bayesian updating, because a result produced that way would be reported as
RxInfer inference while not being RxInfer inference. `try`/`catch` does appear elsewhere
in a generated script, but only around genuinely optional artifacts — the Plots backend
probe and the best-effort PNG rendering — where a missing plotting dependency must not
fail an otherwise valid run.

Step 12 additionally returns a non-zero exit code when `validation.all_valid` is false,
so invalid inference surfaces rather than passing silently.

### Step 4: Expected Free Energy Computation and Action Selection

RxInfer implements EFE from first principles as `G(a) = Ambiguity + Risk`:

#### Ambiguity (Expected Observation Uncertainty)

````julia
ambiguity = 0.0
for j in 1:length(predicted_state)
    if predicted_state[j] > 1e-16
        col = A[:, j]
        col = max.(col, 1e-16)
        ambiguity -= predicted_state[j] * sum(col .* log.(col))
This computes the expected entropy of `P(o\ | s)` weighted by the predicted
next-state distribution: `H[P(o \ | s')]`.

#### Risk (KL Divergence from Preferences)

```julia
C_safe = max.(C_pref, 1e-16)
risk = sum(predicted_obs .* (log.(predicted_obs) .- log.(C_safe)))
````

This computes `D_KL(P(o') \ | \ | C)`, the divergence between predicted and preferred observations.

#### Action Selection (Softmax Policy)

Action selection is `softmax(log E − γ·EFE)` — the Active Inference policy with a habit
prior:

```julia
policy = softmax(log.(max.(E_prior, 1e-16)) .- ACTION_PRECISION .* efe_values)
```

`E` is the habit (policy) prior from the GNN `E` vector, entering by log-addition. With
the uniform default `E` the log term is constant and cancels inside the softmax, so
behavior matches the E-less formula exactly; a non-uniform `E` biases action selection
toward habitual actions independently of EFE.

The `ACTION_PRECISION` constant (γ) is configurable via GNN
`ModelParameters.action_precision` or `ModelParameters.gamma`, default `4.0`. Higher
precision means more deterministic selection of the lowest-EFE action.

### Step 5: Environment Transition

### Step 6: Predictive Prior Update

After state transition, the agent updates its belief using the transition
model to form a predictive prior for the next timestep:

```julia
B_a = B_matrix[:, :, action]
current_belief = B_a * current_belief
current_belief = max.(current_belief, 1e-16)
current_belief = current_belief ./ sum(current_belief)
````

---

## Online mode

`FLAT` models default to **batch smoothing**: one `infer()` call over the whole
trajectory, whose posteriors are smoothed (each timestep's belief is informed by later
observations). That is the right object for offline analysis, but it is not what an agent
has available while acting.

Selecting `inference_mode: online` in the GNN file's `ModelParameters` section — or
passing it as a render option — switches `FlatStrategy` to
`_generate_online_code()`, which emits genuine online active inference: at each timestep
the script calls `infer()` on the observation *prefix* only, and the resulting **filtered**
posterior drives the EFE and habit-based action selection above. `batch` is the default,
and any value other than `batch` or `online` is rejected at render time.

The generated script records which path it took as
`const INFERENCE_MODE`, echoed into the result payload as `inference_mode`, so a
downstream consumer never has to guess whether beliefs are filtered or smoothed.

## Expected Free Energy: RxInfer vs PyMDP Convention

| Aspect           | RxInfer                       | PyMDP                          |
| ---------------- | ----------------------------- | ------------------------------ |
| **Sign Conv**    | Positive (Ambiguity + Risk)   | Negative (`neg_efe`)           |
| **Optimal Dir**  | Lower is better               | Higher (closer to 0) is better |
| **Selection**    | Softmax over `log E − γ·G`    | Softmax over `neg_efe`         |

Both are mathematically equivalent Active Inference implementations. The sign difference is purely conventional.

---

## Telemetry & Logging Output

RxInfer exports a comprehensive JSON artifact to `simulation_results.json`:

### Data Schema

| Field                        | Shape    | Description                                            |
| ---------------------------- | -------- | ------------------------------------------------------ |
| `schema_version`             | `string` | Always `"rxinfer_simulation_v1"`                       |
| `success`                    | `bool`   | Run completed                                           |
| `framework`                  | `string` | Always `"RxInfer.jl"`                                  |
| `model_name`                 | `string` | From GNN `ModelName`                                    |
| `num_timesteps`              | `int`    | Number of simulation steps                              |
| `true_states`                | `[T]`    | True hidden states (1-indexed Julia)                    |
| `observations`               | `[T]`    | Stochastic emissions from the environment               |
| `actions`                    | `[T]`    | Selected actions (1-indexed Julia)                      |
| `beliefs`                    | `[T, S]` | Full posterior belief distributions                     |
| `expected_free_energy`       | `[T]`    | EFE of the **selected** action                          |
| `efe_per_action`             | `[T, A]` | Full EFE vector across all actions                      |
| `policy_posterior`           | `[T, A]` | Softmax policy probabilities                            |
| `variational_free_energy`    | `[T]`    | Genuine VFE trace from `infer()` `free_energy`          |
| `vfe_per_iteration`          | `[I]`    | Per-iteration free energy for the final inference call  |
| `observations_by_modality`   | `obj`    | Per-modality view; flat models use `joint_observation`  |
| `hidden_states_by_factor`    | `obj`    | Per-factor view; flat models use `joint_state`          |
| `actions_by_control_factor`  | `obj`    | Per-control-factor view; flat models use `joint_action` |
| `beliefs_by_factor`          | `obj`    | Per-factor beliefs; flat models use `joint_state`       |
| `model_parameters`           | `obj`    | Matrix shapes, dimensions, `E`, and the `state_factors` / `observation_modalities` echo |
| `matrix_provenance`          | `obj`    | Where each matrix came from                             |
| `runtime_metadata`           | `obj`    | Seed, schema version, RxInfer/Julia versions, script SHA256, `uses_real_rxinfer`, `model_kind`, `b_tensor_order`, `belief_accuracy` |
| `metrics`                    | `obj`    | EFE, policy posterior, belief confidence, VFE           |
| `validation`                 | `obj`    | `all_beliefs_valid`, `beliefs_sum_to_one`, and the rolled-up `all_valid` |

A few conventions worth knowing before consuming this payload:

- **1-indexed.** RxInfer uses Julia's convention, so `actions`, `observations`, and
  `true_states` are 1-indexed. The downstream Python analysis handles the offset.
- **`true_states[t]` records the state that *emitted* observation `t`**, which makes it
  timing-aligned with `beliefs[t]`. Comparing `true_states[t]` against `beliefs[t]` is
  therefore the correct accuracy comparison — no manual shift.
- **`runtime_metadata.b_tensor_order`** carries
  `"next_state_previous_state_action"` from the script's `B_TENSOR_ORDER` constant, so
  the transition-tensor axis order is self-describing rather than assumed.
- **Continuous models echo `state_factors` and `observation_modalities` as empty**
  and report `controls`, `kalman_filter_means` and `control_mode` instead: a
  linear-Gaussian model has no categorical factors, and the closed-loop control
  declared by `goal_mean`/`control_gain` is honoured in the forward simulation.
- **`validation.all_valid` gates the exit code.** Step 12 returns non-zero when it is
  false, so invalid inference is surfaced rather than silently accepted.

---

## The Julia environment

`src/gnn/execute/rxinfer/` is a committed Julia environment, not something resolved at run
time. Its `Project.toml` + `Manifest.toml` pin **RxInfer 5.5** (Julia 1.10+) and declare
the `GnnRxInferModels` package that holds the `@model` blocks, which precompiles the
pomdp, continuous, hierarchical, factored, and learning models loudly — a precompilation
failure surfaces instead of being swallowed. `setup_environment.jl` activates and
instantiates it (`Pkg.activate()` + `Pkg.instantiate()`); there is no runtime `Pkg.add`.

Step 12 defaults `JULIA_PROJECT` to this directory for RxInfer scripts (see
`_build_execution_environment()` in `src/gnn/execute/processor.py`), so a script resolves its
packages without an ambient environment. An explicitly set `JULIA_PROJECT` still wins.

| Package            | Purpose                                            |
| ------------------ | -------------------------------------------------- |
| `RxInfer` (5.5)    | Genuine `@model` + `infer()` variational inference |
| `Distributions`    | `Categorical` distribution sampling                |
| `LinearAlgebra`    | Matrix operations                                   |
| `Random`           | PRNG seeding                                        |
| `StatsBase`        | Action distribution counting                        |
| `JSON`             | Telemetry serialization                             |
| `SHA`              | Script SHA256 for `runtime_metadata`               |
| `Plots`            | Best-effort Julia-native PNGs (never fatal)        |
| `PrecompileTools`  | Ahead-of-time model precompilation                 |
| `Base64`, `Dates`  | Artifact encoding and timestamps                   |

Verify the environment resolves:

```bash
julia --startup-file=no --project=src/gnn/execute/rxinfer \
  -e 'using RxInfer, JSON, Distributions, StatsBase'
```

---

## Source Code Connections

| Pipeline Stage | Module                                                                 | Key Function                      |
| -------------- | ---------------------------------------------------------------------- | --------------------------------- |
| Rendering      | [rxinfer_renderer.py](../../../src/gnn/render/rxinfer/rxinfer_renderer.py) | `render_gnn_to_rxinfer(...)`      |
| Kind detection | [pomdp_contract.py](../../../src/gnn/render/pomdp_contract.py)             | `detect_model_kind(...)`          |
| Strategies     | [model_strategies.py](../../../src/gnn/render/rxinfer/model_strategies.py) | per-`ModelKind` strategy classes  |
| Model blocks   | [GnnRxInferModels.jl](../../../src/gnn/execute/rxinfer/src/GnnRxInferModels.jl) | the five `@model` functions  |
| Entry Point    | [rendering.py](../../../src/gnn/render/processor/rendering.py)             | `render_gnn_spec(...)`            |
| Execution      | [rxinfer_runner.py](../../../src/gnn/execute/rxinfer/rxinfer_runner.py)    | `execute_rxinfer_script()`        |
| Julia Check    | [julia_setup.py](../../../src/gnn/execute/julia_setup.py)                  | `is_julia_available()`            |
| Analysis       | [analyzer.py](../../../src/gnn/analysis/rxinfer/analyzer.py)               | `generate_analysis_from_logs()`   |
| Per-factor     | [analyzer.py](../../../src/gnn/analysis/rxinfer/analyzer.py)               | `compute_per_factor_beliefs()`    |
| Visual         | [analyzer.py](../../../src/gnn/analysis/rxinfer/analyzer.py)               | `create_rxinfer_visualizations()` |
| Extraction     | [analyzer.py](../../../src/gnn/analysis/rxinfer/analyzer.py)               | `extract_simulation_data()`       |
| Cross-framework| [cross_framework.py](../../../src/gnn/analysis/rxinfer/cross_framework.py) | `run_cross_framework_comparison()`|

## See Also / Next Steps

- **[Cross-Framework Methodology](../integration/framework_integration_guide.md)**:
  Details on the correlation methodology and benchmarking metrics.
- **[Architecture Reference](../reference/architecture_reference.md)**:
  Deep dive into the pipeline orchestrator and module integration.
- **[GNN Implementations Index](README.md)**: Return to the master framework implementer manifest.
- **[Back to GNN START_HERE](../../START_HERE.md)**
