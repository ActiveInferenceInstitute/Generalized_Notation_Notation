# cpomdp Render Backend - Agent Scaffolding

## Module Overview

**Purpose**: Render continuous (linear-Gaussian) GNN specifications to standalone cpomdp simulation scripts.

**Parent**: `src/gnn/render/` (Step 11: Render)

**Primary entrypoint**: `render_gnn_to_cpomdp` in `cpomdp_renderer.py`, re-exported by `__init__.py` with `UNSUPPORTED_MESSAGE`.

## Files

- `cpomdp_renderer.py` refuses non-continuous specs, parses the spec with `render.continuous_common.extract_continuous_spec`, validates the options and writes the script atomically.
- `script_template.py` holds the emitted program. It reuses `continuous_common.literal_block` for the matrix literals. It does not reuse `continuous_script`, whose template hard-codes GNN's proportional controller.

## Design notes

- The emitted `LinearGaussianModel` gets `control_matrix = I_n` when a goal is declared, because the GNN contract adds `u_t` straight to the state.
- The prior is over `x_1`. Step 1 therefore runs through a `KalmanBackend` with identity dynamics and zero process noise, which folds the reading in without a predict step.
- Policies come from `cpomdp.enumeration.EnumeratedEfeSearch` over a versioned compass-rose `FiniteActionSet`. One `vmap` of `policy_efe` scores every policy per step and returns G with its pragmatic and epistemic parts, so the chosen policy is not rolled out twice. `EFESelector` is not used because it only searches 1-D actions.
- The per-step search cost is reported as `cost_per_cycle` (`|A|^H * H` step evaluations). `parity` mode pays it too, because it records the scores it does not act on.
- No cpomdp source is vendored. Only the generated script imports cpomdp.

## Testing

`tests/render/test_continuous_renderers.py` renders and runs each mode, rejects bad options and discrete specs, and checks `parity` against the `jax` script's Kalman filter as the exact oracle.

## Documentation

- **[README](README.md)**: usage and options
- **[SPEC](SPEC.md)**: inputs and output schema
