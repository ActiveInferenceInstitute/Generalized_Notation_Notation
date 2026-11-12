# PyMDP Documentation (pymdp 1.0.0 / JAX-first)

**Signposts:** [AGENTS.md](AGENTS.md) · [docs/INDEX.md](../INDEX.md) · [docs/SPEC.md](../SPEC.md) (versioning)

This folder documents how this repository integrates
[pymdp 1.0.0](https://github.com/infer-actively/pymdp) in the render/execute/
analysis pipeline (Steps 11, 12, 16).

pymdp 1.0.0 is the **JAX-first rewrite** of the library. This repository
uses the JAX-first API only (`Agent.infer_states(obs, empirical_prior=...)`,
JAX arrays, and explicit PRNG keys).

## Model Kind Coverage and Execution Status

- The pymdp renderer supports **discrete categorical models only**:
  `A`/`B`/`C`/`D` (plus the optional `E` habit/policy prior), with
  multi-modality and multi-factor POMDPs supported. Continuous
  linear-Gaussian models (`F`/`H`/`Q`/`R` system matrices or
  `prior_mean`/`prior_cov` parameterizations — the `CONTINUOUS` class from
  `detect_model_kind` in `src/gnn/render/pomdp_contract.py`) are **not
  supported** by this renderer (`framework_registry` sets
  `supports_continuous=False`).
- Rendered scripts are **executable**: Step 12 (execute) runs them via the
  `src/gnn/execute/pymdp/` package (`pymdp_runner.py`, rollout in
  `simulation.py`). pymdp is never render-only.
- Per pymdp 1.0.0 convention, `B` matrices are **column-stochastic** per
  action slice (`(next_state, prev_state, action)` layout — see
  [GNN B layout conversion](gnn_pymdp.md#gnn-b-layout-conversion)).

## Start Here

- [`gnn_pymdp.md`](gnn_pymdp.md) — GNN → pymdp 1.0.0 integration contract
  (matrix shapes, rollout loop, embedded code samples).
- [`pymdp_performance_guide.md`](pymdp_performance_guide.md) — performance
  notes (JAX/JIT, batching, memory) and **Systematic Scaling Studies**.
- [`run_pymdp_gnn_scaling_analysis.py`](../../scripts/experiments/run_pymdp_gnn_scaling_analysis.py) —
  The automated orchestrator for PyMDP performance sweeps.
- [`pymdp_1_0_0_alignment_matrix.md`](pymdp_1_0_0_alignment_matrix.md) —
  upstream 1.0.0 claim mapping and local status.
- [`pymdp_pomdp/README.md`](pymdp_pomdp/README.md) — reference scripts under
  `docs/pymdp/pymdp_pomdp/` and their boundaries.

## Version Scope

Local docs track the repository-tested behaviour of `inferactively-pymdp`
1.0.5, which resolves on every supported Python: 1.0.5 lifts the
`jax<0.11.1` cap that kept the Python 3.12+ splits, which need `jax>=0.11.1`
for flax and ngclearn, on 1.0.3 (see the requirement comment in
`pyproject.toml`).

- <https://github.com/infer-actively/pymdp>
- <https://github.com/infer-actively/pymdp/releases/tag/v1.0.0>
- <https://github.com/infer-actively/pymdp/releases/tag/v1.0.5>

Upstream features that this repository does not wire in (e.g. `infer_and_plan`,
full JAX PRNG-key learning loops, `pybefit` model-fitting) are documented as
upstream context only.

## Local Conventions

- Install: `uv pip install 'inferactively-pymdp>=1.0.0'`
- Import style:
  ```python
  import jax.numpy as jnp
  import jax.random as jr
  from pymdp.agent import Agent
  from pymdp import utils as pymdp_utils
  ```
- Agent construction uses **list-of-`jax.Array`** matrices with a leading
  batch dimension (`batch_size=1` by default):
  - `A[m].shape == (batch, num_obs[m], num_states...)`
  - `B[f].shape == (batch, num_states[f], num_states[f], num_controls[f])`
  - `C[m].shape == (batch, num_obs[m])`
  - `D[f].shape == (batch, num_states[f])`
  - `E.shape    == (batch, num_policies)` (optional)
- Rollout loop uses the 1.0.0 canonical pattern:
  ```python
  prior = agent.D
  for t in range(T):
      qs, info = agent.infer_states(obs, empirical_prior=prior, return_info=True)
      q_pi, neg_efe = agent.infer_policies(qs)
      action = agent.sample_action(q_pi, rng_key=jr.split(key, agent.batch_size + 1)[1:])
      prior = agent.update_empirical_prior(action, qs)
  ```
- A pure HMM (1-action / passive) must **omit** `control_fac_idx`; the 1.0.0
  Agent asserts that every indexed control factor has `num_controls[f] > 1`.

## Upstream Agent API regression tests

- `tests/execute/test_pymdp_1_0_0_upstream_api.py` — asserts the installed wheel's
  `Agent` / `utils` behaviour used by `execute/pymdp/simulation.py`
  through direct package calls. Covers:
    - Metadata version ≥ 1.0.0
    - `Agent.update_empirical_prior` presence
    - Single-step `infer_states` → `infer_policies` → `sample_action`
    - `empirical_prior` carry-through via `update_empirical_prior`
    - Optional `E` habit vector matches policy count
- `tests/execute/test_pymdp_contracts.py` — exercises the full GNN
  render → execute → analysis path against real pymdp 1.0.0.
- `tests/execute/test_execute_pymdp_integration.py` — exercises the JAX-first
  Agent directly and the pipeline's `run_pymdp_simulation`.
- `tests/execute/test_discrete_models_pymdp.py` — checks passive 2-D `B`,
  factored T-maze preservation, matrix provenance, and execution output schema.
