# DisCoPy Render Submodule - Agent Scaffolding

## Module Overview

**Purpose**: Generate a self-contained Python script that builds a DisCoPy categorical (string) diagram for a parsed GNN specification.

**Parent Module**: `src/gnn/render/` (Step 11: Code rendering)

**Category**: Framework Code Generation / DisCoPy

**Status**: Production — single-entrypoint renderer backed by `DisCoPyRenderer`.

---

## Core Functionality

### Primary Responsibilities

1. Convert a parsed GNN spec into a runnable DisCoPy script.
2. Map GNN variables to DisCoPy `Box`es and connections to wires.
3. Emit a `main()` in the generated script that assembles and (when `discopy.drawing` is present) renders the diagram.
4. Surface warnings for missing `InitialParameterization`, `ModelParameters`, and `Connections` sections rather than hard-failing.
5. Validate and emit matrix permutation metadata when supplied by parsed GNN data; current permutation support is metadata-only and does not reorder generated diagram wires or boxes.

### What this module is **not**

- It does not execute DisCoPy — Step 12 (`execute.discopy`) runs the generated script.
- It does not emit JAX matrix diagrams. The historical `render_gnn_to_discopy_jax` entrypoint was removed in 2026-04 because the underlying implementation never landed.
- It does not provide separate `generate_discopy_visualization`, `convert_gnn_to_discopy`, or `create_discopy_diagram` helpers. Earlier revisions of this document claimed those — they have never existed in `discopy_renderer.py`.

---

## API Reference

Module path: `src/gnn/render/discopy/`

### `render_gnn_to_discopy(gnn_spec, output_path, options=None)`

```python
def render_gnn_to_discopy(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]:
```

**Location**: `src/gnn/render/discopy/discopy_renderer.py` (module-level function).

**Parameters**:
- `gnn_spec` — Parsed GNN spec dict (output of `gnn.parse_gnn_file`).
- `output_path` — Destination `.py` path for the generated script. Parent directories are created.
- `options` — Optional dict forwarded to `DisCoPyRenderer`. `matrix_permutations` is consumed for metadata validation; other keys are reserved for future template knobs.

**Returns**: `(success, message, warnings)` where `warnings` is a list of human-readable strings describing missing optional sections in the spec.

**Behavior on missing sections** (non-fatal warnings only):
- Missing `initial_parameterization` → `"No initial parameterization found - using defaults"`
- Missing `model_parameters` → `"No model parameters found - using inferred dimensions"`
- Missing `connections` → `"No explicit connections found - using default Active Inference structure"`

### `DisCoPyRenderer(options=None)`

Internal helper class used by `render_gnn_to_discopy`. It exposes `_generate_discopy_diagram_code(gnn_spec, model_name)` which produces the script source. Not part of the public API — callers should prefer the free function above.

---

## Usage

```python
from pathlib import Path
from gnn import parse_gnn_file
from gnn.render.discopy import render_gnn_to_discopy

spec = parse_gnn_file(Path("input/gnn_files/actinf_pomdp_agent.md"))
ok, message, warnings = render_gnn_to_discopy(
    spec,
    Path("output/11_render_output/actinf_pomdp_agent_discopy.py"),
)
for warning in warnings:
    print(f"[warn] {warning}")
assert ok, message
```

The CLI for the parent render module also exposes this path:

```bash
python -m render.render --target discopy --gnn-file input/gnn_files/actinf_pomdp_agent.md
```

---

## Output

A single Python file (`<output_path>`) containing:
- Imports (`discopy`, `discopy.drawing`, `numpy`).
- Box and type declarations derived from `gnn_spec.variables`.
- `MATRIX_PERMUTATION_METADATA` and `MATRIX_PERMUTATION_APPLIED_TO_DIAGRAM = False` when valid matrix permutation records are present.
- A `main()` that composes the diagram and calls `diagram.draw(...)` when DisCoPy drawing backends are available.
- A `__main__` guard so the script is runnable standalone.

No additional visualization artifacts are produced by the renderer — rendered images come from running the generated script in Step 12.

---

## Dependencies

### Required at generation time
- Standard library only (`pathlib`, `datetime`, `typing`).

### Required at runtime of the generated script
- `discopy` — categorical diagram library.
- `numpy` — tensor support used by DisCoPy.
- `matplotlib` — optional, used by `diagram.draw(...)` when present.

### Internal
- `gnn` package — consumers call `gnn.parse_gnn_file` before passing specs here.

---

## Error Handling

Any exception raised during code generation is caught and returned as `(False, "Error generating DisCoPy script: <exception>", [])`. The renderer never raises past the function boundary.

---

## Integration

### Orchestrated by
- `render/processor/rendering.py::render_gnn_spec` — dispatches when `target == "discopy"`.
- `11_render.py` — the numbered pipeline script.

### Consumed by
- `execute/discopy/` — runs the emitted script in Step 12.
- `tests/render/test_render_cli_targets.py` — exercises the CLI dispatch.
- `tests/test_render_discopy*.py` — focused unit tests.

### Data flow
```
GNN spec → render_gnn_to_discopy → <output>.py → execute.discopy runs script → image / logs
```

---

## Testing

- `uv run --extra dev python -m pytest tests/test_render_discopy*.py -v`
- `uv run --extra dev python -m pytest tests/render/test_render_cli_targets.py -v` — verifies the `discopy` target dispatches correctly.

---

## Known Limitations

- The generated script assumes DisCoPy's Python API; categorical features that require `pytket` or `lambeq` are not emitted. These were aspirational in prior drafts and have been removed from this document.
- JAX-backed diagram evaluation is not supported. See `render/jax/` for a JAX-specific renderer with a different code-path.
- Matrix permutation support is currently a validated metadata contract. The emitted DisCoPy diagram is still built from the parsed variable and connection structure; non-identity permutation metadata is not yet reflected as reordered diagram construction.

---

## References

- [Render Module](../AGENTS.md)
- [DisCoPy documentation](https://discopy.readthedocs.io/)

**Last Updated**: 2026-09-24
**Status**: Active
