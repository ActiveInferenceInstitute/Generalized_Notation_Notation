# cpomdp Render Backend - Agent Scaffolding

## Module Overview

**Purpose**: Render continuous (linear-Gaussian) GNN specifications to standalone cpomdp simulation scripts.

**Parent**: `src/gnn/render/` (Step 11: Render)

**Primary entrypoint**: `render_gnn_to_cpomdp` in `cpomdp_renderer.py`, re-exported by `__init__.py` with `UNSUPPORTED_MESSAGE`.

## Files

- `cpomdp_renderer.py` refuses non-continuous specs, parses the spec with `render.continuous_common.extract_continuous_spec`, validates the options and writes the script atomically.
- `finite_controls.py` validates source-declared finite controls and candidate sequences without importing inference libraries.
- `script_template.py` holds the emitted program. It reuses `continuous_common.literal_block` for the matrix literals. It does not reuse `continuous_script`, whose template hard-codes GNN's proportional controller.

## Design notes

- The compass-rose `LinearGaussianModel` uses `control = I_n`. E3-FA1 source declarations provide an explicit state-by-control matrix; recorded `controls` remain state increments and `control_inputs` preserve control coordinates.
- The prior is over `x_1`. Step 1 therefore runs through a `KalmanBackend` with identity dynamics and zero process noise, which folds the reading in without a predict step.
- Exhaustive policies come from `cpomdp.enumeration.EnumeratedEFESearch`. Explicit candidate policies come from ordered source action IDs. One `vmap` of the library's `policy_efe` scores all eligible search rows per step and returns G with its pragmatic and epistemic parts.
- Exhaustive search admits `num_timesteps + 1` planning cycles, including one native `evaluate` pass for the completeness certificate. Candidate search admits `num_timesteps` cycles and records the authored list and its score validity instead of a certificate.
- The per-step search cost is reported as `cost_per_cycle` (`|A|^H * H` step evaluations, the library's `cost_per_plan`). `parity` mode pays it too, because it records the scores it does not act on.
- No cpomdp source is vendored. Only the generated script imports cpomdp.

## Testing

`tests/render/test_cpomdp_contracts.py` compares real released-wheel runs against an independent NumPy Kalman and EFE implementation on both supported JAX lock splits. It checks admission caps, malformed options and all-invalid policy scores.

`admission.py` rejects searches before enumeration: at most 4,096 policies, 32,768 step evaluations and 128 MiB estimated allocation. The estimate excludes runtime/JIT overhead; the process watchdog remains authoritative. The optional dependency is exactly `cpomdp==0.5.1`; no Git source override is used.

## Documentation

- **[README](README.md)**: usage and options
- **[SPEC](SPEC.md)**: inputs and output schema
