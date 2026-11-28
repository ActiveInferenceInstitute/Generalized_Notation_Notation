# pymdp 1.0.0 Compatibility Matrix

This matrix maps upstream `infer-actively/pymdp` 1.0.0 release claims to the
integration points in this repository.

## Scope

- Upstream sources reviewed:
    - <https://github.com/infer-actively/pymdp>
    - <https://github.com/infer-actively/pymdp/releases/tag/v1.0.0>
- Local surfaces reviewed:
    - `src/gnn/render/pymdp/`
    - `src/gnn/execute/pymdp/`
    - `docs/pymdp/`
    - `tests/execute/test_pymdp_*` / `tests/execute/test_execute_pymdp_*`

## Matrix

| Upstream 1.0.0 item | Local status | Action |
|---|---|---|
| **JAX-first Agent** (`equinox.Module`-based) | **Fully integrated** | `src/gnn/execute/pymdp/simulation._build_pymdp_agent` builds a real JAX-first Agent from GNN matrices. Exercised by `test_pymdp_1_0_0_upstream_api.py` and `test_pymdp_contracts.py`. |
| `Agent(A, B, C, D, E, num_controls, control_fac_idx, policy_len, batch_size, …)` | **Fully integrated** | Used verbatim by `_build_pymdp_agent`. Passive factors (num_controls[f] == 1) omit `control_fac_idx` as required by `Agent._validate`. |
| `infer_states(observations, empirical_prior, *, return_info=False)` → `qs [, info]` | **Fully integrated** | `simulation.run_pymdp_simulation` always calls with `empirical_prior=…, return_info=True` and extracts `info["vfe"]`. |
| `infer_policies(qs)` → `(q_pi, neg_efe)` | **Fully integrated** | The return tuple is unpacked as `(q_pi, neg_efe)` (upstream docstring calls the second value `G` = negative EFE per policy). |
| `sample_action(q_pi, rng_key=…)` with JAX PRNG keys | **Fully integrated** | Actions are drawn with `jr.split(key, batch_size + 1)[1:]`, matching the upstream quickstart. |
| `Agent.update_empirical_prior(action, qs)` — stateful rollout closure | **Fully integrated** | Called once per step in both `simulation` and `PyMDPSimulation`. Presence of this method is how this repo detects pymdp 1.0.0+. |
| Batched list-of-array models (leading batch dim on A/B/C/D/E) | **Fully integrated** | `_to_jax_batched` prepends a batch axis of size 1 (or broadcasts for `batch_size>1`). |
| `utils.random_A_array` / `random_B_array` / `list_array_uniform` | **Available, not used for GNN models** | GNN-provided numeric matrices take precedence over random generators. These utils are asserted present by `test_utils_public_surface_exists`. |
| `utils.norm_dist` (JAX-array normalisation helper) | **Available (informational)** | Pipeline uses its own numpy-based `_normalise_columns` / `_normalise_prob_vector` because GNN matrices enter as numpy. |
| `utils.obj_array` (0.x object-array helper) | **Removed upstream** | Gone from `pymdp.utils` in 1.0.0. Removed from this repo's code paths and docs. |
| `categorical_obs` naming standardisation | **Partial** | The pipeline always passes integer observation indices in 1-element `jnp.int32` arrays. `categorical_obs=True` mode (probability-vector observations) is supported by pymdp but not exercised in local tests. |
| Explicit PRNG key flow in JAX workflows | **Integrated for rollout** | `jax.random.PRNGKey` + `split` used for every `sample_action` call. Upstream learning / fitting flows (Dirichlet updates, JAX-native) are not locally wrapped. |
| `rollout()` / `infer_and_plan()` first-class APIs | **Not directly integrated** | The pipeline runs a plain Python loop for readability. Upstream helpers may be adopted later via a dedicated wrapper. |
| Model fitting via `pybefit` / NumPyro workflows | **Not integrated** | Upstream capability; not wrapped in pipeline code or tests. |
| `equinox` and `multimethod` runtime deps | **Added to `pyproject.toml`** | Pinned alongside `inferactively-pymdp>=1.0.0`. |
| Tiered notebook testing gates | **Not mirrored** | Upstream reference only. |

## Local Contract Clarifications

- **Render-side public API**
  (`src/gnn/render/pymdp/__init__.py`): `render_gnn_to_pymdp(gnn_spec, output_path, options=…)`.
  `options={"mode": "pipeline"}` (default) emits a pipeline runner; `options={"mode": "standalone"}`
  emits a fully self-contained pymdp 1.0.0 script.

- **Execute-side public API**
  (`src/gnn/execute/pymdp/__init__.py`): `execute_pymdp_simulation(gnn_spec, output_dir, correlation_id)` — canonical entry.
  Also exposes `PyMDPSimulation` (the GNN-driven wrapper class).

- **Visualisation** belongs to Step 16 (`src/gnn/analysis/pymdp/`), not Step 12.

## B Tensor Axis Order And Stochasticity (Canonical Contract)

The GNN transition tensor is stored in the pymdp 1.0.0 native axis order:

- **Axis order**: `B[next_state, previous_state, action]` — identical to
  pymdp 1.0.0 `B[f][s, v, u]` (see `pymdp/control.py`, "Each element `B[f][s, v, u]`
  stores the probability of hidden state level `s` at the current time, given
  hidden state level `v` and action `u` at the previous time").
- **Column-stochasticity**: each per-action slice `B[:, :, a]` is
  column-stochastic — rows are next states, columns are previous states, and
  each column (one previous state) sums to 1 over next states. pymdp's
  `Agent` validates this with `utils.validate_normalization(B[f], axis=1)`.
- **InitialParameterization writing convention**: the canonical tensor reads
  `B[next_state][previous_state][action]`. The pipeline accepts these layouts:
  - next-state-outer (the `pomdp_gridworld_3x3.md` convention): the outer
    axis is the next state; each slice has rows = previous states and
    columns = actions;
  - per-action-outer: one slice per action, with next-state rows and
    previous-state columns;
  - per-action-outer: one slice per action, with previous-state rows and
    next-state columns.
  Set `b_tensor_order` to declare the stored order explicitly.
- **Enforcement**: the shared resolver uses custody metadata for an already
  canonical tensor, then the executable `b_tensor_order` declaration.
  Shape and probability mass provide evidence for sources without one.
  The type checker
  (`type_checker.checking.dimensions.validate_dimension_compatibility`,
  code `GNN-E002`) flags comment-vs-comment orientation contradictions and
  row-stochastic-only slices as errors in strict mode.

## Locally regression-tested Agent API

Run:

```bash
uv run --extra dev python -m pytest \
    tests/execute/test_pymdp_1_0_0_upstream_api.py \
    tests/execute/test_pymdp_contracts.py \
    tests/execute/test_execute_pymdp_integration.py \
    -v
```

Covered surfaces (installed package exercised directly):

- `importlib.metadata.version("inferactively-pymdp") >= 1.0.0`
- `from pymdp.agent import Agent` / `from pymdp import utils`
- Presence of `Agent.update_empirical_prior` (the 1.0.0 signal)
- `utils.random_A_array`, `random_B_array`, `list_array_uniform`,
  `norm_dist`, `list_array_norm_dist`
- `Agent.infer_states(obs, empirical_prior=…, return_info=True)` return shapes
- `Agent.infer_policies(qs)` return tuple `(q_pi, neg_efe)`
- `Agent.sample_action(q_pi, rng_key=…)` batched action shape
- Multi-step rollout closed via `Agent.update_empirical_prior`
- Optional `E` habit vector length aligned with the policy count
- Import of `pymdp.control` and `pymdp.inference` (used internally by Agent)

Pipeline-level integration is covered by
`tests/execute/test_pymdp_contracts.py::test_actinf_pomdp_render_execute_analyze_e2e`,
which renders real GNN POMDP input → runs pymdp 1.0.0 via Step 12 → collects
the JSON result → analyses via Step 16.

## Validation Policy Used For This Alignment

- If a behaviour is not backed by local tests, docs label it as
  upstream context rather than a local guarantee.
- Examples in local docs use the canonical pymdp 1.0.0 import style:
  - `import jax.numpy as jnp`
  - `import jax.random as jr`
  - `from pymdp.agent import Agent`
  - `from pymdp import utils as pymdp_utils`
- Installation guidance uses the `inferactively-pymdp` PyPI name and pins
  the version floor to `1.0.0`.
