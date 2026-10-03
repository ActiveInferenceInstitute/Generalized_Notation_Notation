# THRML execution

THRML is an experimental backend selected explicitly with `--frameworks thrml`.
Install its released runtime with `uv sync --extra thrml`; installation and
selection remain explicit. The supported adapter targets `thrml==0.1.4`.

```python
from pathlib import Path
from gnn.execute.thrml import execute_thrml_script

result = execute_thrml_script(
    Path("model_thrml.py"), output_dir=Path("scratch/thrml"), timeout=300
)
```

`execute_thrml_script` returns a structured process result. Dependency absence
is `skipped`; failed probes, deadlines, cancellation, invalid scientific
artifacts and unverified cleanup remain unsuccessful. Readiness imports THRML,
its released categorical API and JAX only inside a bounded child using the
execution interpreter. The parent imports neither THRML nor JAX.

`run_thrml_records` preserves every selected script's independent outcome;
`run_thrml_scripts` returns success only when all required records succeed.
`selected_scripts=[]` means no work, while `None` permits standalone discovery.
Default discovery that finds no scripts returns no records and an unsuccessful
Boolean result; it does not establish execution evidence.
One monotonic batch deadline covers readiness, dispatch, retrieval and receipts.

The script writes `THRML_OUTPUT_DIR/simulation_data/simulation_results.json`.
`GNN_THRML_EXECUTION_ID` and the current script SHA256 bind that result to its
execution. The runner leases the output directory, preserves process evidence,
validates the sampling witness, and rejects stale artifacts. Its default output
is a separate `<script_stem>_execution` directory beside the script.

[Analysis](../../analysis/thrml/README.md), [specification](SPEC.md), and
[implementation guide](../../../../docs/gnn/implementations/thrml.md) describe the
scientific interpretation and limitations.
