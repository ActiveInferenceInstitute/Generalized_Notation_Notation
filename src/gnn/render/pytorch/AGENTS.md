# PyTorch Render Backend - Agent Scaffolding

## Module Overview

**Purpose**: Render parsed GNN/POMDP specifications to **standalone PyTorch simulation scripts**.

**Parent**: `src/gnn/render/` (Step 11: Render)

**Primary entrypoint**: `render_gnn_to_pytorch` in `pytorch_renderer.py` (re-exported by `__init__.py`).

---

## Public API

From `src/gnn/render/pytorch/__init__.py`:

- `render_gnn_to_pytorch(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`

**Contract**:
- writes exactly one Python file to `output_path`
- returns `(success, message, [output_path])` on success

---

## Implementation notes

`pytorch_renderer.py`:

- extracts \(A, B, C, D\) with `render.spec_matrices.extract_abcd_matrices`
  (strict probability validation, canonical B orientation, provenance)
- extracts the habit `E` (strict, one entry per action; uniform when absent)
  and the seed (`seed` option, then `model_parameters`, then 42)
- routes continuous specs to the shared LGSSM generator

`discrete_script.py` emits the discrete script: `torch.nn.Module` components
at the addresses in `sites.SITE_DESCRIPTIONS`, a seeded single-generator
rollout, and the opt-in `_torchlens_capture` block (`GNN_TORCHLENS_CAPTURE`,
`GNN_TORCHLENS_ARTIFACTS`). `sites.py` maps `## Connections` variables onto
those addresses and reports every unmappable variable with a reason.

The generated script writes `simulation_results.json` under `PYTORCH_OUTPUT_DIR` (default: current directory).

Rules for changes here:

- Keep `torchlens` out of module scope in the emitted script and out of the
  renderer; the pipeline parent never imports it.
- A capture failure must never change the simulation results or the exit code.
- Renaming a module or buffer changes the traced site addresses: update
  `SITE_DESCRIPTIONS`, `sites.py` and the fidelity tests together.

---

## Integration points

- **Called by**: `render.pomdp_processor.POMDPRenderProcessor` when the `pytorch` framework is selected and the backend is importable.
- **Consumed by**: `src/gnn/execute/pytorch/` runner (Step 12) and `src/gnn/analysis/pytorch/` analyzer (Step 16).

---

## Testing

Preferred tests:

- formatting/syntax validation (`python -m py_compile` on the generated file)
- integration: Step 11 produces `pytorch/<model>_pytorch.py` when the backend is available
- `tests/render/test_render_pytorch_torchlens.py`: module structure, habit and
  seed contracts, site mapping on corpus models, seeded reproducibility
  (`needs_torch`), and the real capture plus Step 16 fidelity report
  (`needs_torchlens`)

