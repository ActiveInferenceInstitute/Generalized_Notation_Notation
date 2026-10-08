# JAX Renderer Specification

## Overview

The `render.jax` submodule implements Step 11 (Render) backend support for generating **JAX-based Python scripts** from parsed GNN specifications.

This specification defines:

- the stable public API exported by `src/gnn/render/jax/__init__.py`,
- the input contract for `gnn_spec`,
- output artifacts and success criteria,
- dependency expectations for generated scripts.

## Public API

`src/gnn/render/jax/__init__.py` exports exactly:

- `render_gnn_to_jax(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`
- `render_gnn_to_jax_pomdp(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`
- `render_gnn_to_jax_combined(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`
- `render_gnn_to_jax_factorized(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`

No other functions are part of the supported surface unless re-exported there.

### Kronecker-factorized specs (MAJ-02)

Specs declaring per-factor matrices (`A_fN`/`B_fN`/`C_fN`/`D_fN` in
`structured_pomdp.matrices`, >= 2 factors) are detected by
`jax_renderer._is_factorized_spec` and routed to
`render_gnn_to_jax_factorized` even when `render_gnn_to_jax` is called
directly. The emitted script embeds the canonicalized per-factor matrices
(action-major B transposed to `(next, previous, action)` and
column-normalized, mirroring `POMDPRenderProcessor._canonicalise_factored_B`)
and drives `execute.jax.kronecker_factorized.run_factorized_active_inference`
— the joint state space (`prod(factor_sizes)`) is reported in the results but
never allocated. The script resolves the repository root via
`GNN_PROJECT_ROOT` (or upward walk), writes
`simulation_results.json` with schema `jax_kronecker_factorized_v1` under
`GNN_OUTPUT_DIR`, and is executed by Step 12 like any other rendered script.
`options` may carry `seed` and `action_precision` (defaults 42 / 4.0).

## Inputs

### `gnn_spec`

The renderer expects `gnn_spec` as a parsed dictionary, typically produced by earlier pipeline steps.

It supports multiple internal shapes for extracting \(A, B, C, D\) via `jax_renderer._extract_gnn_matrices`, including:

- POMDP-extractor style dicts with `model_parameters` and `initialparameterization`
- JSON export style dicts with `statespaceblock` and `raw_sections["InitialParameterization"]`
- older parsed dicts with `variables` and `InitialParameterization`

Complete `model_parameters` + `initialparameterization` inputs admit lists,
tuples, and numeric NumPy arrays without mutating the caller. Explicit
canonicalization failures propagate to a failed render before any output is
written; authored matrices are never replaced after failed admission.

Supplied existing braced literals are parsed strictly, including nested transition
tensors. Malformed or ragged literals fail rather than receiving padded rows or
recovery values. Supplied existing tables must match their declared dimensions;
`A` columns, each action's `B` columns, and `D` must contain finite nonnegative
probabilities with unit mass under the existing canonical rounding tolerance.
`C` is a finite real payoff vector and may contain negative preferences. Optional
`E` in complete canonical inputs retains the shared action-prior validation.

Historical dimension-based defaults for partial inputs remain compatible. The
complete-model value-preservation checks do not certify those defaults as
authored scientific parameters.

### `output_path`

- Must be a file path (typically ending in `.py`).
- Parent directories are created if missing.
- The function writes exactly one file per call and returns that path in `generated_files`.

### `options`

`options` is an optional dict passed through to the generator. Callers should not rely on specific keys unless explicitly documented by the implementation.

## Outputs

Each API call writes a single Python script to `output_path`:

- **General model**: pure JAX + NumPy
- **POMDP solver**: JAX; optional Optax import with explicit "continue without Optax" behavior in generated code
- **Combined model**: JAX + Flax + Optax

Step 11 may place these scripts under:

`output/11_render_output/<model_name>/jax/` (organization handled by the parent render processor).

## Dependencies

### Module runtime (render-time)

- Python: **3.11+** (repo policy)
- `numpy`

### Generated script runtime

- `jax`, `jaxlib` (required)
- `numpy`
- `optax` (required for combined model; optional for the POMDP solver, which records when Optax helpers are unavailable)
- `flax` (required for combined model)

## Non-goals

- This module does not define a thin orchestrator; Step 11 orchestration is handled by `src/gnn/11_render.py` calling `render.process_render`.
- `src/gnn/render/jax/templates/` is not part of the active code-generation path unless the implementation is explicitly refactored to use it.
