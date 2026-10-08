# JAX Render Submodule - Agent Scaffolding

## Module Overview

**Purpose**: Generate JAX-based simulation scripts from parsed GNN specifications.

**Parent Module**: Render Module (Step 11: Code rendering)

**Category**: Framework Code Generation / JAX

The JAX submodule focuses on two concrete use cases:

- a **general Active Inference–style model** (standalone JAX script with belief updates and expected free energy);
- a **POMDP solver and a combined hierarchical / multi-agent variant** for richer models.

All public entrypoints live in `jax_renderer.py` and are re-exported by `src/gnn/render/jax/__init__.py`.

---

## Public API (from `__init__.py`)

The submodule exports exactly three functions:

```python
from gnn.render.jax import (
    render_gnn_to_jax,
    render_gnn_to_jax_pomdp,
    render_gnn_to_jax_combined,
)
```

Internally these are thin wrappers around code generators in `jax_renderer.py`. They all share the same signature shape:

```python
def render_gnn_to_jax(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]: ...
```

The POMDP and combined variants differ only in which generator they call.

### `render_gnn_to_jax(gnn_spec, output_path, options=None) -> (bool, str, List[str])`

**Description**: Render a parsed GNN specification to a **general JAX Active Inference–style model**. The generated script is a standalone Python file that depends only on `jax`, `jax.numpy`, and `numpy`.

**Parameters**:
- `gnn_spec` (`Dict[str, Any]`): Parsed GNN model as produced by earlier pipeline steps (e.g. includes `ModelName`, optional `model_parameters`, and either `initialparameterization`, `statespaceblock`, or `variables` sections).
- `output_path` (`Path`): Exact path to the `.py` file that will be written.
- `options` (`Optional[Dict[str, Any]]`): Optional configuration passed through to the generator. Current options are interpreted inside `_generate_jax_model_code` and are intentionally kept minimal; callers may treat this as an opaque bag for future extensions.

**Returns**: `(success, message, generated_files)` where:
- `success` (`bool`): `True` if code was generated and written successfully.
- `message` (`str`): Short status message suitable for logs.
- `generated_files` (`List[str]`): List of file paths written (currently a single-element list containing `output_path`).

**Location**: `src/gnn/render/jax/jax_renderer.py` (`render_gnn_to_jax`).

**Notes**:
- The generator derives matrix dimensions from the GNN spec and emits a JAX script that exposes functions such as `create_params`, `simulate_step`, and `run_simulation`.

### `render_gnn_to_jax_pomdp(gnn_spec, output_path, options=None) -> (bool, str, List[str])`

**Description**: Render a GNN specification to a **JAX POMDP solver**. The generated script includes a `JAXPOMDPSolver` class with belief updates and alpha-vector–style value iteration.

**Parameters / Returns**:
- Same as `render_gnn_to_jax`; the output file usually has a `_pomdp`-oriented filename chosen by the caller.

**Location**: `src/gnn/render/jax/jax_renderer.py` (`render_gnn_to_jax_pomdp`).

**Notes**:
- Complete matrices \(A, B, C, D\) are admitted through the canonical probability
  and axis contract. NumPy array inputs are adapted on a copy, and explicit
  admission failures prevent artifact writes. Historical defaults for partial
  inputs remain a separate compatibility path.

### `render_gnn_to_jax_combined(gnn_spec, output_path, options=None) -> (bool, str, List[str])`

**Description**: Render a GNN specification to a **combined JAX model** suitable for hierarchical, multi-agent, or continuous extensions. The generated script uses Flax and Optax to define a `Combined` module and training-ready structure.

**Parameters / Returns**:
- Same as `render_gnn_to_jax`; `options` can be used to tune configuration in `_generate_jax_combined_code` as that function evolves.

**Location**: `src/gnn/render/jax/jax_renderer.py` (`render_gnn_to_jax_combined`).

---

## Input Expectations

The JAX renderer is designed to consume the **internal GNN representation** already produced by earlier pipeline steps. It supports several shapes:

- **POMDP extractor format**: dictionaries with a `model_parameters` section and an `initialparameterization` section containing numeric values for \(A, B, C, D\).
- **JSON export format**: specs with `statespaceblock` and `raw_sections["InitialParameterization"]`.
- **Older parsed format**: specs with `variables`, `parameters`, and `InitialParameterization`.

In all cases, `_extract_gnn_matrices` will:

- infer dimensionality for \(A, B, C, D\);
- normalize probability matrices where appropriate;
- preserve complete numeric lists, tuples, and NumPy arrays without caller mutation;
- reject supplied malformed/ragged text, nonfinite values, invalid probabilities,
  and disagreement with declared dimensions before writing an artifact;
- retain historical dimension-based defaults for partial inputs, outside the
  complete-model scientific value-preservation evidence.

`C` contains signed real preferences rather than probabilities. It is never
normalized as a probability vector. Legacy braced `B` tensors use the declared
`[next_state, previous_state, action]` axes; complete canonical inputs also honor
explicit supported action-major axis declarations.

Callers should treat `gnn_spec` as an opaque dict that has already passed type checking and validation at earlier pipeline steps.

---

## Generated Outputs

All three entrypoints ultimately write a **single Python script** to `output_path`:

- general JAX: `..._jax.py` (naming is handled by the caller);
- POMDP solver: JAX-based POMDP solver with `create_pomdp_solver` / `solve_pomdp`;
- combined model: Flax-based module for hierarchical / multi-agent / continuous setups.

From the perspective of Step 11:

- per-model JAX outputs are typically organized under  
  `output/11_render_output/<model_name>/jax/` by the higher-level POMDP render processor;
- these scripts are then consumed by Step 12 (execute) and later analysis steps.

---

## Dependencies

**At generation time (this module):**
- `numpy` for intermediate matrix handling.

**In the generated scripts:**
- general JAX and POMDP scripts: `jax`, `jax.numpy`, `numpy`;
- combined scripts: `jax`, `jax.numpy`, `flax.linen`, `optax`.

The generators attempt to make the produced files self-contained; installation of JAX/Flax/Optax happens in later steps or inside the generated scripts themselves where necessary.

---

## Integration Points

- **Orchestrated by**:
  - `src/gnn/render/processor/rendering.py` via the `"jax"` target in `AVAILABLE_RENDERERS`.
  - `src/gnn/11_render.py` (Step 11) through `render.process_render`.

- **Imported from**:
  - `render.jax` (this package) by tests and by the POMDP render processor.

- **Downstream**:
  - Step 12 (`src/gnn/12_execute.py`) executes the generated JAX scripts and collects results for analysis.

---

## Testing

JAX-specific behaviour is covered indirectly by the render integration tests:

- `tests/render/test_render_overall.py`
- `tests/render/test_render_integration.py`

These tests assert that:

- the `"jax"` backend is registered and reported by `get_available_renderers`;
- rendering runs to completion on representative GNN fixtures and produces syntactically valid Python files.

When adding new features to `jax_renderer.py`, prefer:

- small, focused tests that exercise matrix extraction and code generation;
- end-to-end tests that run Step 11 and confirm that JAX files are written to the expected locations.

---

## Versioning

- **Implementation version**: follows the parent `render` module (`__version__` in `src/gnn/render/__init__.py`).
- This document describes the current three-function API surface and should be kept in sync whenever new public entrypoints are added or signatures change.




