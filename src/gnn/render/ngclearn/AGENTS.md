# ngc-learn Render Backend - Agent Scaffolding

## Module Overview

**Purpose**: Render parsed GNN/POMDP specifications to **standalone ngc-learn simulation scripts** (continuous linear-Gaussian models only).

**Parent**: `src/gnn/render/` (Step 11: Render)

**Primary entrypoint**: `render_gnn_to_ngclearn` in `ngclearn_renderer.py` (re-exported by `__init__.py`).

---

## Public API

From `src/gnn/render/ngclearn/__init__.py`:

- `render_gnn_to_ngclearn(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`

**Contract**:
- continuous (linear-Gaussian) specs write exactly one Python file to `output_path` and return `(success, message, [output_path])`
- discrete POMDPs are first-class unsupported: `(False, "unsupported-discrete-state: discrete POMDP: ngclearn supports continuous linear-Gaussian models only", [])`

---

## Implementation notes

`ngclearn_renderer.py` is a thin codegen wrapper (pytorch precedent): the
renderer never imports `ngclearn` — the generated script does, behind an
import-or-exit guard. Script generation is delegated to
`render.continuous_script.generate_continuous_script(backend="ngclearn")`, so
the Kalman numerics stay byte-identical to the `jax` backend and
`rmse_vs_true` remains apples-to-apples across backends.

The generated script writes `simulation_results.json` under
`NGCLEARN_OUTPUT_DIR` (default: current directory).

---

## Integration points

- **Called by**: `render.processor._render_continuous_target` (continuous dispatch) and `render.pomdp_processor` framework dispatch when the `ngclearn` framework is selected.
- **Consumed by**: the Step-12 ngc-learn script path. Scripts skip without the `ngclearn` extra (`uv sync --extra ngclearn`; marker-gated to py3.12 via `gnn.utils.runtime_safety.framework_availability`).

---

## Testing

Preferred tests:

- formatting/syntax validation: the generated script compiles (`py_compile`) — no ngclearn import happens at render or test time
- integration: Step 11 produces `ngclearn/<model>_ngclearn.py` for continuous models

End-to-end execution tests should live under `tests/` and be resilient to optional dependency availability.
