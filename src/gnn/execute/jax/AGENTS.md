# JAX Execute Submodule - Agent Scaffolding

## Module Overview

**Purpose**: Execution of JAX-based simulations generated from GNN specifications

**Parent Module**: Execute Module (Step 12: Simulation execution)

**Category**: Framework Execution / JAX

---

## Core Functionality

### Primary Responsibilities
1. Discover rendered JAX scripts under `output/11_render_output/<model>/jax/`
2. Execute each script as a Python subprocess with the requested JAX platform
3. Provide the first-class factorised Kronecker executor (`execute_kronecker_factorized`,
   `run_kronecker_factorized_execution`) for sparse factor-separable active inference

### Key Capabilities
- Subprocess execution of rendered scripts with syntax pre-check, timeout and captured stdout/stderr
- Device selection through `JAX_PLATFORM_NAME` (`device` argument; Step 12 honours `GNN_JAX_PLATFORM`)
- Output routing through `JAX_OUTPUT_DIR` / `GNN_OUTPUT_DIR` so scripts write into the Step 12 tree
- Kronecker-factorised execution that never materialises the joint state space
  (`jax_kronecker_factorized_v1` schema)
- Availability probe that logs the JAX version and visible devices

---

## API Reference

Public names re-exported from `execute.jax` (`__all__` in `__init__.py`):
`run_jax_scripts`, `execute_jax_script`, `find_jax_scripts`, `is_jax_available`,
`run_kronecker_factorized_execution`, `execute_kronecker_factorized`.

### Public Functions

#### `run_jax_scripts(rendered_simulators_dir: Union[str, Path], execution_output_dir: Optional[Union[str, Path]] = None, recursive_search: bool = True, verbose: bool = False, device: Optional[str] = None) -> bool`
**Description**: Find every JAX script below `<rendered_simulators_dir>/jax` and execute them in sequence.

**Parameters**:
- `rendered_simulators_dir`: Directory containing rendered simulators (Step 11 output)
- `execution_output_dir`: Where execution outputs go (optional)
- `recursive_search`: Whether to walk subdirectories
- `verbose`: Enable verbose logging
- `device`: JAX platform name (`"cpu"`, `"gpu"`, `"tpu"`); `None` leaves JAX's default

**Returns**: `True` if every script succeeded (or none were found), `False` otherwise.

**Example**:
```python
from pathlib import Path
from gnn.execute.jax import run_jax_scripts

success = run_jax_scripts(
    rendered_simulators_dir=Path("output/11_render_output"),
    execution_output_dir=Path("output/12_execute_output"),
    recursive_search=True,
    verbose=True,
    device="cpu",
)
```

#### `execute_jax_script(script_path: Path, verbose: bool = False, device: Optional[str] = None, output_dir: Optional[Path] = None, timeout: int = 300) -> bool`
**Description**: Execute one rendered JAX script with `sys.executable`. The script is
syntax-checked first; `device` sets `JAX_PLATFORM_NAME`, `output_dir` sets
`JAX_OUTPUT_DIR`, and the subprocess runs with `cwd` set to the script's directory.

**Returns**: `True` on exit code 0, `False` on syntax error, non-zero exit or timeout.

#### `find_jax_scripts(base_dir: Union[str, Path], recursive: bool = True) -> List[Path]`
**Description**: Collect JAX script files under `base_dir`.

#### `is_jax_available() -> bool`
**Description**: Return `True` when `jax` imports; logs the version and `jax.devices()`.
Takes no arguments — per-device probing is done by the rendered script itself.

### Kronecker Executor (`kronecker_executor.py`)

#### `execute_kronecker_factorized(config: Dict[str, Any], output_dir: Union[str, Path], *, factor_sizes: Optional[List[int]] = None, ...) -> Dict[str, Any]`
Runs the factorised model described by `config` and writes
`simulation_data/simulation_results.json` (`jax_kronecker_factorized_v1`) plus
`kronecker_execution_summary.json` under `output_dir`.

#### `run_kronecker_factorized_execution(model: Any, output_dir: Union[str, Path], model_name: Optional[str] = None) -> Dict[str, Any]`
Executes an already-constructed factorised model; same outputs as above.

See `tests/execute/test_kronecker_factorized.py` for the exactness contract.

---

## Dependencies

### Required Dependencies
- `jax`, `jaxlib` — pinned in `pyproject.toml` (`jax[cpu]>=0.7.0,<0.12`, `jaxlib>=0.7.0,<0.12`); installed by a plain `uv sync`
- `numpy`

### Optional Dependencies
- `optax`, `flax` — used by the combined JAX template; core dependencies in `pyproject.toml`

### Internal Dependencies
- `execute.processor` - Step 12 orchestration
- `render.jax` - Produces the scripts this module executes

---

## Configuration

This module carries no configuration dictionaries of its own. Behaviour is
driven by function arguments and two environment variables:

- `JAX_PLATFORM_NAME` — set from the `device` argument; Step 12 derives it from `GNN_JAX_PLATFORM`
- `JAX_OUTPUT_DIR` / `GNN_OUTPUT_DIR` — where the rendered script writes `simulation_results.json`

Timeouts and framework selection come from Step 12 (`--frameworks`, `timeout` kwarg in `execute.processor`).

---

## Usage Examples

### Execute one rendered script
```python
from pathlib import Path
from gnn.execute.jax import execute_jax_script, find_jax_scripts

for script in find_jax_scripts(Path("output/11_render_output"), recursive=True):
    ok = execute_jax_script(
        script,
        verbose=True,
        device="cpu",
        output_dir=Path("output/12_execute_output") / script.stem / "jax" / "simulation_data",
        timeout=600,
    )
    print(f"{script.name}: {'ok' if ok else 'failed'}")
```

### Availability check
```python
from gnn.execute.jax import is_jax_available

if not is_jax_available():
    print("JAX not importable; Step 12 will report the framework as skipped")
```

---

## Output Specification

### Output Products
- `simulation_results.json` - Per-run JAX simulation output written by the rendered script
  (the Kronecker executor emits schema `jax_kronecker_factorized_v1`; continuous
  linear-Gaussian scripts write the continuous result schema from `render.continuous_common`)
- `kronecker_execution_summary.json` - Runtime + validation summary for factorised runs
- Execution logs captured by Step 12

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

Read result fields from a generated `simulation_results.json`; the templates in
`src/gnn/render/jax/templates/` and `src/gnn/render/continuous_script.py` are the source
of truth for what each script writes.

---

## Performance Characteristics

Do not treat timing numbers in documentation as current measurements. The first
run of a script pays JIT compilation; read durations from
`summaries/execution_summary.json` after a run.

---

## Error Handling

### JAX Execution Errors
1. **JAX Not Importable**: `is_jax_available()` is `False`; `run_jax_scripts` returns `False` and Step 12 reports the framework as skipped
2. **Syntax Error**: the script fails `compile()` before execution
3. **Non-zero Exit**: the script raised at runtime (stderr is logged)
4. **Timeout**: the subprocess exceeded `timeout` seconds

### Recovery Strategies
- Re-run with `device="cpu"` when a GPU/TPU platform is not available
- There is no automatic device fallback, retry or code rewriting; read the captured stderr

---

## Integration Points

### Orchestrated By
- **Parent Module**: `src/gnn/execute/` (Step 12)
- **Main Script**: `12_execute.py`

### Imported By
- `execute.processor` - Main execution integration

### Data Flow
```
render.jax (.py script) → execute_jax_script (JAX_PLATFORM_NAME / JAX_OUTPUT_DIR) → simulation_results.json → Step 12 summaries → Step 16 analysis
```

---

## Testing

### Test Files
- `tests/execute/test_execute_overall.py` - Execute module tests (includes framework selection)
- `tests/execute/test_kronecker_factorized.py` - Kronecker executor exactness contract
- `tests/render/test_jax_renderer.py`, `tests/render/test_jax_factorized_pipeline.py` - Renderer-side contracts for the scripts this module executes

### Test Commands
```bash
uv run --extra dev python -m pytest tests/execute/test_kronecker_factorized.py \
    tests/execute/test_execute_overall.py -v

# With coverage
uv run --extra dev python -m pytest tests/execute/test_kronecker_factorized.py \
    --cov=src/gnn/execute/jax --cov-report=term-missing
```

---

## MCP Integration

This submodule registers no MCP tools of its own. The parent module
(`src/gnn/execute/mcp.py`) exposes `process_execute`, `execute_gnn_model`,
`execute_pymdp_simulation`, `check_execute_dependencies` and
`get_execute_module_info`; JAX scripts are reached through `process_execute`
with the `jax` framework selected.

---

## Development Guidelines

### Adding New JAX Features
1. Change the generated code in `src/gnn/render/jax/` (this module only runs what the renderer emits)
2. Update execution logic in `jax_runner.py` or `kronecker_executor.py`
3. Add or extend tests in `tests/execute/` and `tests/render/`

---

## Troubleshooting

### Common Issues

#### Issue 1: "JAX is not available"
**Symptom**: `run_jax_scripts` returns `False` immediately
**Cause**: `jax` not importable in the active environment
**Solution**: `uv sync` (JAX is a core dependency); check `src/gnn/utils/runtime_safety/jax_stack_validation.py`

#### Issue 2: Platform not found
**Symptom**: Script fails at start-up with an XLA platform error
**Cause**: `JAX_PLATFORM_NAME` names hardware that is not installed
**Solution**: Pass `device="cpu"` or unset `GNN_JAX_PLATFORM`

#### Issue 3: Script exits non-zero
**Symptom**: `execute_jax_script` returns `False`
**Cause**: Runtime error in the rendered script
**Solution**: Re-run with `verbose=True` and read the logged stderr

---

## Version History

### Current Version: [pyproject.toml](../../../../pyproject.toml) (canonical)

**Features**:
- JAX script discovery and subprocess execution with platform/output-dir routing
- First-class Kronecker-factorised executor (`jax_kronecker_factorized_v1`)
- Continuous linear-Gaussian scripts (Kalman filter) execute through the same runner

**Known Limitations**:
- TPU execution requires a Google Cloud environment
- No automatic device fallback

---

## References

### Related Documentation
- [Execute Module](../AGENTS.md) - Parent execute module
- [JAX Render](../../render/jax/AGENTS.md) - JAX code generation
- [JAX Documentation](https://jax.readthedocs.io/) - Official JAX docs

### External Resources
- [JAX GitHub](https://github.com/google/jax)
- [JAX Performance Guide](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)

---

**Last Updated**: 2026-09-24
**Maintainer**: Execute Module Team
**Status**: ✅ Production Ready
