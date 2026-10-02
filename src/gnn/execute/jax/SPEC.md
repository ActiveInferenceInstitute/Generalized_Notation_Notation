# JAX Execution — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

## Execution Model

- Python subprocess execution for rendered JAX scripts
- Pre-flight: JAX availability + device check (CPU/GPU/TPU)
- JIT compilation caching for repeated runs
- Timeout: inherits from Step 12 timeout (1800s default)
- **First-class factorised Kronecker executor** (roadmap MAJ-02): the public
  `execute.jax` surface exposes `execute_kronecker_factorized` and
  `run_kronecker_factorized_execution` so the numbered pipeline can route
  sparse factor-separable active inference through Step 12 and emit the
  `jax_kronecker_factorized_v1` schema for Step 16 analysis consumption.

## Input

- JAX scripts from `output/11_render_output/jax/`
- A constructed/factorised model (via `execute.jax.kronecker_factorized`) for
  the sparse Kronecker path.

## Output

- `simulation_results.json` — Inference results and computation traces
  (`simulation_data/` carries the `jax_kronecker_factorized_v1` schema for
  Kronecker executions)
- `kronecker_execution_summary.json` — slim runtime metadata for factorised runs
- Execution logs (stdout/stderr)
- Device utilization metrics

## Kronecker executor contract

`execute.jax.kronecker_executor.execute_kronecker_factorized(config, output_dir)`
accepts a config with per-factor `factor_sizes` (joint size = product), and
writes:

- `output/simulation_data/simulation_results.json` — `jax_kronecker_factorized_v1`
- `output/kronecker_execution_summary.json` — runtime + validation summary

The joint state space is never materialised (`joint_materialized: False`).
Exactness (Kronecker identities, per-factor EFE decomposition) is pinned by
`tests/execute/test_kronecker_factorized.py`.

## Dependencies

- `jax[cpu]>=0.7.0,<0.12`, `jaxlib>=0.7.0,<0.12` from the supported frozen lock splits; GPU installation remains explicit.