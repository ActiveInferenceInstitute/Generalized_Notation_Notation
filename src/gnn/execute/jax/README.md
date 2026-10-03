# JAX Executor for GNN Processing Pipeline

This module executes JAX scripts generated from GNN specifications by Step 11
(Render) as part of Step 12 (Execute), and provides the factorised Kronecker
executor for sparse factor-separable active inference.

## Features

- **Script Discovery**: Finds rendered JAX scripts under `output/11_render_output/<model>/jax/`
- **Device Selection**: `device` argument → `JAX_PLATFORM_NAME` (`cpu`, `gpu`, `tpu`); Step 12 honours `GNN_JAX_PLATFORM`
- **Output Routing**: `JAX_OUTPUT_DIR` / `GNN_OUTPUT_DIR` point the script at the Step 12 tree
- **Availability Check**: `is_jax_available()` uses a bounded child probe; optional packages stay outside the caller process
- **Kronecker Executor**: `execute_kronecker_factorized` / `run_kronecker_factorized_execution` (`jax_kronecker_factorized_v1`)

## Requirements

Pinned in `pyproject.toml` and installed by a plain `uv sync`:

- `jax[cpu]>=0.7.0,<0.12` and `jaxlib>=0.7.0,<0.12`
- `flax>=0.7.0,<0.12.7` on Python 3.11; `flax>=0.12.9` on Python 3.12 and later; `optax>=0.1.0` (used by the combined JAX template)
- `numpy`

GPU/TPU builds of `jaxlib` are a user-side install; the pipeline only requires the CPU build.

## Usage

### Command Line Interface

```bash
# Execute all JAX scripts below a render directory
python src/gnn/execute/jax/jax_runner.py --output-dir output/11_render_output --verbose

# Execute with a specific device
python src/gnn/execute/jax/jax_runner.py --output-dir output/11_render_output --device cpu

# Recursive search
python src/gnn/execute/jax/jax_runner.py --output-dir output/11_render_output --recursive --verbose
```

### Python API

```python
from gnn.execute.jax import run_jax_scripts, is_jax_available

if is_jax_available():
    success = run_jax_scripts(
        rendered_simulators_dir="output/11_render_output",
        execution_output_dir="output/12_execute_output",
        recursive_search=True,
        verbose=True,
        device="cpu",
    )
    print(f"Execution successful: {success}")
else:
    print("JAX not available")
```

### Individual Script Execution

```python
from pathlib import Path
from gnn.execute.jax import execute_jax_script

success = execute_jax_script(
    Path("output/11_render_output/actinf_pomdp_agent/jax/actinf_pomdp_agent_jax.py"),
    verbose=True,
    device="cpu",
    output_dir=Path("output/12_execute_output/actinf_pomdp_agent/jax/simulation_data"),
    timeout=300,
)
```

`execute_jax_script` and `run_jax_scripts` both return `bool`.

## Device Management

The runner does not pick hardware itself. Pass `device=` (or set `GNN_JAX_PLATFORM`
for Step 12) and the value is exported as `JAX_PLATFORM_NAME` to the subprocess:

```python
import jax

print([str(d) for d in jax.devices()])  # what the subprocess will see
```

There is no automatic GPU → CPU fallback; re-run with `device="cpu"` if a platform is missing.

## Integration with Pipeline

### Step 12 Integration

`execute.processor` executes each rendered JAX script listed in the Step 11
`render_processing_summary.json`, setting `GNN_OUTPUT_DIR` to
`output/12_execute_output/<model>/jax/simulation_data/` and (when
`GNN_JAX_PLATFORM` is set) `JAX_PLATFORM_NAME`.

### Output Directory Structure

```
output/12_execute_output/
├── <model>/jax/
│   └── simulation_data/
│       └── simulation_results.json
└── summaries/
    ├── execution_summary.json
    └── execution_report.md
```

Kronecker-factorised runs additionally write `kronecker_execution_summary.json`
next to their `simulation_data/`.

## Error Handling

1. **JAX Not Available** — `is_jax_available()` returns `False`; `run_jax_scripts` returns `False` and Step 12 reports the framework as skipped.
2. **Syntax Error** — the script fails `compile()` before execution.
3. **Non-zero Exit / Timeout** — `execute_jax_script` returns `False`; stderr is logged.

Individual script failures do not stop the rest of the batch.

## Debugging

```python
import logging
logging.getLogger("execute.jax").setLevel(logging.DEBUG)

# Inside a rendered script, JAX's own debug flags apply:
import jax
jax.config.update("jax_debug_nans", True)
```

Read per-script durations and statuses from `output/12_execute_output/summaries/execution_summary.json`.

## Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX Performance Guide](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)
- [Optax Documentation](https://optax.readthedocs.io)
- [Flax Documentation](https://flax.readthedocs.io)

## Contributing

When extending the JAX executor:

1. Change generated code in `src/gnn/render/jax/`; this module only runs what the renderer emits
2. Keep `jax_runner.py` signatures in sync with this README and `AGENTS.md`
3. Add tests under `tests/execute/` (see `test_kronecker_factorized.py`)

`execute_jax_script(..., output_dir=path)` resolves the requested directory
and supplies that same absolute path in both `JAX_OUTPUT_DIR` and
`GNN_OUTPUT_DIR`. Current continuous and factorized scripts therefore place
results beside their execution logs even when the script runs from a
different working directory.
