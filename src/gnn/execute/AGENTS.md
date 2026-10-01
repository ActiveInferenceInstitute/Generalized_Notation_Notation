# Execute Module - Agent Scaffolding

## Module Overview
**Purpose**: Execute rendered simulation scripts across multiple frameworks (PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, Stan, Lean, bnlearn, ngc-learn, cpomdp). Per-folder `execution_summary.json` files are merged so the durable summary covers every input folder; frameworks a model's kind cannot use are reported `unsupported` by Step 11 and are never executed.

**Pipeline Step**: Step 12: Execution (src/gnn/12_execute.py)

**Category**: Simulation / Execution

**Status**: ✅ Production Ready

**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-26

---

## File Structure

```
processor/
├── __init__.py    # Package facade: module docstring, pre-split surface re-exports (detection/julia_env/metadata/data_extractors/types/subprocess_envelope aliases + moved defs)
├── envelope.py    # Per-script result envelope factories: _base_execution_envelope, _make_skipped_result, local/distributed failure factories, facade-routed availability probe
├── workers.py     # _coerce_execution_workers/_coerce_dispatch_retries, _execute_script_worker, _run_scripts_with_local_workers (ProcessPoolExecutor resolved through the facade)
├── summary.py     # _init_execution_summary, _update_framework_status, _classify_execute_outcome, _write_execution_summaries
├── single.py      # _aggregate_benchmark_samples, _new_execution_result, _build_execution_environment, sandbox helpers, _backend_version_from_runner_metadata, execute_single_script
└── pipeline.py    # process_execute (Step 12 orchestration) and execute_simulation_from_gnn
```

---

## Core Functionality

### Primary Responsibilities
1. Execute Python simulation scripts (PyMDP, JAX, DisCoPy, PyTorch, NumPyro, the cmdstanpy driver for Stan, and the generator-backed bnlearn programs via `execute/bnlearn/`)
2. Execute Julia simulation scripts (RxInfer.jl, ActiveInference.jl)
3. Capture simulation results and logs
4. Handle execution errors gracefully
5. Generate execution reports

### Key Capabilities
- Multi-framework execution support
- **Skip vs fail**: JAX, NumPyro, and DisCoPy are **core** dependencies (repair with `uv sync`); if the environment is incomplete, scripts are **skipped** (not run) and reported as "skipped" — they do not count as execution failures. PyTorch ships in the `torch` extra (`uv sync --extra torch`; torch>=2.13.0 resolves GHSA-rrmf-rvhw-rf47) — without it the PyTorch backend is reported skipped. Stan needs `uv sync --extra stan` plus a CmdStan toolchain. bnlearn needs the `bnlearn` extra (`uv sync --extra bnlearn`) or, for `.R` scripts, Rscript plus the R `bnlearn` package — without it bnlearn scripts are reported skipped. The ngc-learn runtime is the py3.12 marker-gated `ngclearn` extra (`uv sync --extra ngclearn`) — without it ngclearn scripts are reported skipped with an install hint, never failed. Julia backends still require a local Julia install.
- Graceful degradation when frameworks unavailable
- Automatic PyMDP package detection (distinguishes correct vs wrong package variants)
- Path collection with deduplication (prevents nested directory issues)
- Comprehensive error logging
- Result capture and validation
- Execution timeout handling
- **`summaries/execution_summary.json`** (slim aggregate): Per-script rows omit bulk fields (`stdout`/`stderr` bodies, `simulation_data`); lengths and pointers remain. Consumers needing full detail pass **`execution_summary_detail=True`** / **`--execution-summary-detail`** to also write **`summaries/execution_summary_detail.json`**.
- **`PipelineArguments` / main orchestrator**: Step 12 subprocess commands omit **`--backend`** unless **`distributed`** is true, and omit **`--execution-benchmark-repeats`** when the value is 1 (avoids implying benchmarking when repeats are disabled).
- **PyMDP subprocess environment**: Defaults `TF_CPP_MIN_LOG_LEVEL=3` for the child process when unset (quieter captured stderr). Set **`GNN_JAX_PLATFORM`** (e.g. `cpu`) on the host to pin JAX device selection for PyMDP runs; when unset, JAX uses normal platform discovery.
- **Julia subprocess environment**: every Julia subprocess GNN assembles carries `GKSwstype=100` (headless GR — Plots.jl renders without the `gksqt` Qt window, which hangs display-less hosts). A `GKSwstype` already set in the caller's environment wins. Shared helper: `execute.julia_env.julia_subprocess_env`; the Step 12 rendered-script env builder (`processor._build_execution_environment`) applies the same default for `julia` executors.
- Configurable script-level concurrency: local process workers by default, or Ray/Dask when `distributed=True`

---

## API Reference

### Public Functions

#### `process_execute(target_dir: Path, output_dir: Path, verbose: bool = False, frameworks: str = "all", **kwargs) -> Union[bool, int]`
**Description**: Main execution function called by orchestrator (src/gnn/12_execute.py). Executes rendered simulation scripts across multiple frameworks.

**Parameters**:
- `target_dir` (Path): Directory containing rendered scripts (typically output from Step 11)
- `output_dir` (Path): Output directory for execution results
- `verbose` (bool): Enable verbose logging (default: False)
- `frameworks` (str): Frameworks to execute ("all", "lite", or comma-separated list, default: "all")
  - `"all"`: Execute all configured executors
  - `"lite"`: Selects PyMDP, JAX, DisCoPy, and bnlearn (bnlearn scripts skip at the shared pre-flight probe when the `bnlearn` extra is absent; see below)
- `timeout` (int): Execution timeout per script in seconds (default: 3600)
- `render_output_dir` (Optional[Path]): Explicit Step 11 output directory to search. This is the safest way to keep Step 12 scoped to an isolated pipeline run.
- `execution_workers` (int): Number of rendered scripts to execute concurrently. `1` preserves serial execution; values above `1` use local process workers unless `distributed=True`.
- `distributed` (bool): Execute scripts through the distributed dispatcher when enabled (default: False)
- `backend` (str): Distributed execution backend, default `ray`
- `**kwargs`: Additional framework-specific options

**Returns**: `Union[bool, int]` - True when execution succeeds or there is nothing to execute, False on execution failure, and integer exit-style values where the caller uses them.

**Example**:
```python
from gnn.execute import process_execute
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
success = process_execute(
    target_dir=Path("output/11_render_output"),
    output_dir=Path("output/12_execute_output"),
    verbose=True,
    frameworks="pymdp,jax",
    timeout=600,
    render_output_dir=Path("output/11_render_output"),
    execution_workers=2,
)
```

#### `execute_simulation_from_gnn(gnn_file: Path, output_dir: Path) -> Dict[str, Any]`
**Description**: Execute simulation for a specific GNN file (defined in `execute/processor/pipeline.py`, re-exported from `execute`).

**Parameters**:
- `gnn_file` (Path): Path to GNN file
- `output_dir` (Path): Output directory for execution results

**Returns**: `Dict[str, Any]` - Execution results dictionary with:
- `success` (bool): Whether execution succeeded
- `return_code` (int): Process return code
- `stdout` (str): Standard output
- `stderr` (str): Standard error
- `duration` (float): Execution duration in seconds
- `output_files` (List[Path]): Generated output files

#### `execute_script_safely(script_path: Union[str, Path], timeout: int = 3600, capture_output: bool = True, cwd: Optional[Union[str, Path]] = None, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]`
**Description**: Run a single Python script via ``subprocess.run`` and return a uniform envelope. Rejects non-``.py`` paths up front and converts every failure mode (missing file, timeout, non-zero exit, unknown exception) into a dict so callers never need to differentiate.

**Returns**: `Dict[str, Any]` with keys:
- `success` (bool), `script_path` (str), `return_code` (int), `stdout` (str), `stderr` (str), `duration_seconds` (float), and `error`/`error_type` on failure.

**Example**:
```python
from gnn.execute import execute_script_safely

result = execute_script_safely(
    "output/11_render_output/.../model_pymdp.py", timeout=120
)
if not result["success"]:
    print(f"{result['error_type']}: {result.get('error') or result['stderr']}")
```


#### `execute_rendered_simulators(target_dir: Path, output_dir: Path, logger: logging.Logger, recursive: bool = False, verbose: bool = False, **kwargs) -> bool`
**Description**: Iterate over the `ExecutorFrameworkSpec` registry for every supported framework runner (PyMDP, RxInfer.jl, DisCoPy, ActiveInference.jl, JAX, NumPyro, PyTorch, ngc-learn, Lean, Stan, bnlearn) and write a summary JSON + markdown report under ``output_dir / "12_execute_output" / "summaries" /``. Stan runs as an availability-gated registry runner (cmdstanpy + CmdStan toolchain probe). bnlearn carries a render-only verdict: the registry never runs it and records ``"SKIPPED"`` with that reason, while rendered bnlearn scripts keep executing through the Step 12 script path (per-lane output env vars, `BNLEARN_OUTPUT_DIR`). Missing optional dependencies are recorded as ``"SKIPPED"`` instead of failures.

#### `plan_execute(target_dir: Path, output_dir: Path, frameworks: str = "all", **config) -> ExecutionPlan`
**Description**: Dry-run Step 12 planner (``execute.planning``). Composes the same discovery / render-contract / dependency primitives as `process_execute` but runs **no scripts and no Julia package probing** — it answers "what would Step 12 do?" for preflight checks, CI gates, and interactive debugging. Returns a typed `ExecutionPlan` (``execute.types``) with `requested_frameworks`, `render_output_dir`, `render_contract_found`, `status` (`"ready"` | `"no_render_output"` | `"no_executable_scripts"` | `"invalid_frameworks"`), `total_scripts`, and per-script disposition lists (`would_execute`, `would_skip_dependency`, `unknown_framework_scripts`), plus `missing_render_scripts` and `render_failures`. Raises `ValueError` on an invalid `frameworks` argument (the same exception `process_execute` catches and converts to `return False`).

**Example**:
```python
from gnn.execute import plan_execute

plan = plan_execute(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/12_execute_output"),
    frameworks="pymdp,jax",
    render_output_dir=Path("output/11_render_output"),
)
print(plan["status"], plan["total_scripts"], len(plan["would_execute"]))
```

#### `list_frameworks() -> list[dict]`
**Description**: Introspect the executor framework registry (`execute.executor`). Returns one record per registered backend with `framework`, `result_key` (the `*_executions` summary key), `available` (whether the runner is currently importable), and `operation` (the dispatch operation name). Useful for CLI/MCP diagnostics and tests that want to assert the registry shape without importing the private `_framework_specs` helper.

#### `collect_doctor_report(target_dir: Optional[Path] = None, output_dir: Optional[Path] = None, frameworks: str = "all") -> Dict[str, Any]`
**Description**: One structured capability report (`execute.doctor`, MCP tool `get_doctor_report`). Composes per-framework availability — the canonical `gnn.utils.runtime_safety.framework_availability` `FRAMEWORK_IMPORT_CHECK`/`check_framework` records plus the shared Julia PATH gate for the Julia frameworks (`rxinfer`, `activeinference_jl`, reusing `planning`'s classification) — with the `plan_execute` Step 12 dry-run when a `target_dir`/`output_dir` pair is supplied (both or neither; exactly one raises `ValueError`). Strictly offline: importability probes, one PATH lookup, directory reads; no scripts run and nothing is written. Returns `frameworks` (keyed records with `kind` `python_import` | `julia_toolchain`, `probe_module`, availability, `missing_module`/`install_hint` when unavailable, `toolchain_probe` for Stan), `julia`, `frameworks_available`/`frameworks_missing`, `execution` (the full plan or a `not_probed` record), and `execution_ready`. Invalid `frameworks` values raise the planner's `ValueError`; the MCP envelope converts it to the standard error shape.

**Example**:
```python
from gnn.execute import collect_doctor_report

report = collect_doctor_report(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/12_execute_output"),
)
print(report["frameworks_missing"], report["execution_ready"])
```

#### Framework Health Checking

Framework availability is assessed at execution time by the processor rather than a single standalone function. Key detection utilities:

- **`execute.pymdp.package_detector.detect_pymdp_installation()`** — Detect which PyMDP package variant is installed.
- **`execute.pymdp.package_detector.validate_pymdp_for_execution()`** — Validate PyMDP is ready for execution.
- **`execute.bnlearn.is_bnlearn_available()` / `execute.bnlearn.is_r_bnlearn_available()`** — bnlearn probes: the Python `bnlearn` module (shared `gnn.utils.runtime_safety.framework_availability` mapping) and, for `.R` scripts, Rscript + the R `bnlearn` package. Rendered bnlearn scripts skip with the install hint when the lane's runtime is missing.
- **`execute.ngclearn.ngclearn_runner.is_ngclearn_available()`** — ngclearn probe (shared `gnn.utils.runtime_safety.framework_availability` `FRAMEWORK_IMPORT_CHECK` row): `jax`, `ngclearn`, and `ngcsimlib` importability for the py3.12 marker-gated `ngclearn` extra. Rendered ngc-learn scripts skip with the install hint (`Add ngclearn: uv sync --extra ngclearn`) when the runtime is missing.
- **MCP tool**: `check_execute_dependencies` — Exposes framework availability via MCP (see `execute/mcp.py`).
- **MCP tool**: `get_doctor_report` — Exposes the composed availability + Step 12 readiness report via the `run_tool_envelope` dispatch pattern (see `execute/mcp.py`, `execute/doctor.py`).

#### PyMDP Package Detection Functions
**Module**: `execute.pymdp.package_detector`

**Functions**:
- `detect_pymdp_installation() -> Dict[str, Any]`: Detect which PyMDP package variant is installed
  - Returns detection results including `correct_package`, `wrong_package`, `has_agent`, `has_mdp_solver`
- `is_correct_pymdp_package() -> bool`: Check if correct package (inferactively-pymdp) is installed
- `get_pymdp_installation_instructions() -> str`: Get actionable installation instructions
- `validate_pymdp_for_execution() -> Dict[str, Any]`: Validate PyMDP is ready for execution
  - Returns `ready` status, detection results, and installation instructions

**Usage**:
```python
from gnn.execute.pymdp.package_detector import (
    detect_pymdp_installation,
    is_correct_pymdp_package,
)

detection = detect_pymdp_installation()
if detection.get("wrong_package"):
    print("Wrong PyMDP package installed - install inferactively-pymdp")
elif not detection.get("correct_package"):
    print("PyMDP not installed - install inferactively-pymdp")
```

---

## Configuration

### Configuration Options

#### Execution Parameters
- `timeout` (int): Execution timeout in seconds (default: `3600`)
- `capture_output` (bool): Capture stdout/stderr (default: `True`)
- `render_output_dir` (Path): Render output directory to search before default discovery
- `frameworks` (str): `"all"` (the executors: PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, Stan, Lean, bnlearn — bnlearn via `execute/bnlearn/`, skipping with the install hint when its runtime is absent), `"lite"` (PyMDP, JAX, DisCoPy, bnlearn), or a comma-separated subset — parsed by `parse_frameworks_parameter` in `execute/detection.py` (re-exported through the `execute/processor/` facade)
- `execution_workers` (int): Number of rendered scripts to execute concurrently. This parallelizes model/script runs, not timesteps within a single simulation.
- `distributed` (bool): Route scripts through the distributed dispatcher instead of the local process pool
- `backend` (str): Dispatcher backend, default `ray`

#### Security Configuration
- `GNN_ALLOW_UNSAFE_EXEC` (str): set to `"1"` to bypass the pre-execution
  security gate (RED_TEAM V-01/V-06). Default: gate active.
- `GNN_SANDBOX` (str): rendered-script sandbox mode — `"off"` (default, no
  wrapping), `"prefer"` (wrap when firejail/bwrap/nsjail is found, otherwise
  run unsandboxed with a warning), or `"require"` (refuse to run when no
  backend is found). See `execute.sandbox` (RED_TEAM V-10).
- `GNN_JAX_PLATFORM` (str): pins `JAX_PLATFORM_NAME` for JAX-backed child processes (see Key Capabilities).

---

## Dependencies

### Required Dependencies
- `subprocess` - Script execution
- `json` - Result serialization

### Optional Dependencies
- `inferactively-pymdp` - PyMDP simulation engine (package name: `inferactively-pymdp`, recovery: skip PyMDP)
  - **Note**: The correct package name is `inferactively-pymdp`, not `pymdp`
  - The execute module automatically detects wrong package variants
- `julia` - Julia runtime (recovery: skip Julia scripts)
- `jax` - JAX framework (recovery: skip JAX)

---

## Usage Examples

### Basic Usage
```python
from gnn.execute import process_execute

success = process_execute(
    target_dir=Path("output/11_render_output"),
    output_dir=Path("output/12_execute_output"),
    frameworks="all",
)
```

### Scoped Execution From Isolated Render Output
```bash
uv run python src/gnn/12_execute.py \
    --target-dir input/gnn_files/pymdp_scaling_study \
    --output-dir output/pymdp_scaling_pipeline \
    --frameworks pymdp \
    --timeout 1200 \
    --render-output-dir output/pymdp_scaling_pipeline/11_render_output \
    --execution-workers 2
```

When `--render-output-dir` is provided, Step 12 searches that directory for executable scripts and still filters by `--frameworks`. This prevents execution from reading stale artifacts in the default `output/11_render_output` directory.

Use `--distributed --backend ray` or `--distributed --backend dask` only when Ray/Dask is installed and the run should use that dispatcher. Without `--distributed`, Step 12 uses local process workers when `--execution-workers` is greater than `1`.

---

## Output Specification

### Output Products
- `summaries/execution_summary.json` - Slim per-script aggregate (stdout/stderr bodies and `simulation_data` omitted; pass `execution_summary_detail=True` / `--execution-summary-detail` to also write `summaries/execution_summary_detail.json`)
- `summaries/execution_report.md` - Human-readable report
- `<model>/<framework>/simulation_data/simulation_results.json` - Per-model, per-framework simulation output written by the rendered script

### Output Directory Structure
```
output/12_execute_output/
├── summaries/
│   ├── execution_summary.json
│   ├── execution_report.md
│   └── execution_summary_detail.json   # only with execution_summary_detail=True
└── <model>/
    └── <framework>/
        └── simulation_data/
            └── simulation_results.json
```

Successive invocations merge into the existing `summaries/execution_summary.json` (`_merge_prior_execution_summary`), so the durable summary covers every input folder.

---

## Performance Characteristics

### Latest Execution
Use the current `output/*/00_pipeline_summary/pipeline_execution_summary.json` and Step 12 summaries for exact timing, memory, and pass/fail counts. Do not treat stale benchmark numbers in documentation as current measurements.

---

## Error Handling

### Graceful Degradation
- **PyMDP unavailable**: Log warning, skip PyMDP scripts
- **Julia unavailable**: Log warning, skip Julia scripts
- **JAX unavailable**: Log warning, skip JAX scripts
- **Script errors**: Capture stderr, continue with other scripts
- **Timeout**: configurable per script, defaulting to 3600s in `process_execute`

### Error Categories
1. **Dependency Errors**: Framework not installed
2. **Syntax Errors**: Generated code has errors
3. **Runtime Errors**: Simulation crashes
4. **Timeout Errors**: Execution exceeds limit

---

## Integration Points

### Pipeline Integration
- **Input**: Receives rendered simulation scripts from Step 11 (render)
- **Output**: Generates execution results for Step 13 (llm analysis), Step 16 (analysis), and Step 23 (report generation)
- **Dependencies**: Requires rendered code from `src/gnn/11_render.py` output. Use `--render-output-dir` for isolated pipeline runs.

### Module Dependencies
- **render/**: Consumes rendered simulation scripts
- **llm/**: Provides execution results for LLM analysis
- **analysis/**: Provides execution data for statistical analysis
- **report/**: Provides execution summaries for reports

### External Integration
- **PyMDP**: Executes Python Active Inference simulations
- **Julia Runtime**: Executes Julia simulation scripts (RxInfer.jl, ActiveInference.jl). RxInfer.jl runs the genuine `@model` + `infer()` pipeline under the committed `Project.toml` + `Manifest.toml` at `src/gnn/execute/rxinfer/` via `julia --startup-file=no --project=src/gnn/execute/rxinfer <script>`.
- **JAX**: Executes JAX-based simulations
- **DisCoPy**: Executes categorical diagram computations

### Data Flow
```
src/gnn/11_render.py (Code generation)
  ↓
src/gnn/12_execute.py (Script execution; optional explicit render_output_dir)
  ↓
  ├→ src/gnn/13_llm.py (LLM analysis of results)
  ├→ src/gnn/16_analysis.py (Statistical analysis)
  ├→ src/gnn/23_report.py (Execution reports)
  └→ output/12_execute_output/ (Execution results)
```

---

## Testing

### Test Files
- `tests/execute/test_execute_overall.py`
- `tests/execute/test_execute_pymdp_integration.py`
- `tests/execute/test_execute_pymdp_package.py`

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest tests/test_execute*.py \
    --cov=src/gnn/execute --cov-report=term-missing
```
### Key Test Scenarios
1. Multi-framework execution
2. Error handling and recovery
3. Result capture and validation
4. Timeout handling

---

## MCP Integration

### Tools Registered
- `process_execute` - Execute rendered simulation scripts in a directory
- `execute_gnn_model` - Execute a rendered GNN model script
- `execute_pymdp_simulation` - Run a PyMDP simulation from a GNN specification
- `check_execute_dependencies` - Report framework availability
- `get_execute_module_info` - Module metadata

All five are registered by `register_tools()` in `src/gnn/execute/mcp.py`.

### MCP File Location
- `src/gnn/execute/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Julia execution fails
**Symptom**: Julia scripts fail to execute  
**Cause**: Julia not installed or not in PATH  
**Solution**: 
- Install Julia: `brew install julia` (macOS) or download from [julialang.org](https://julialang.org)
- Verify Julia installation: `julia --version`
- Check Julia is in PATH: `which julia`
- Install required Julia packages if needed

#### Issue 2: Framework dependencies missing
**Symptom**: Execution fails with import errors  
**Cause**: Required packages not installed in environment  
**Solution**:
- Install framework dependencies: `uv pip install inferactively-pymdp jax`
- **Note**: The correct PyMDP package name is `inferactively-pymdp`, not `pymdp`
- For Julia RxInfer: Instantiate committed environment: `julia --startup-file=no --project=src/gnn/execute/rxinfer -e 'using Pkg; Pkg.instantiate()'`
- Check framework-specific requirements in documentation

#### Issue 2a: Wrong PyMDP package installed
**Symptom**: Error message "Wrong pymdp package installed. Found 'pymdp' with MDP/MDPSolver"  
**Cause**: The wrong `pymdp` package (with MDP/MDPSolver) is installed instead of `inferactively-pymdp`  
**Solution**:
- Uninstall wrong package: `uv pip uninstall pymdp`
- Install correct package: `uv pip install inferactively-pymdp`
- Or use setup module: `python src/gnn/1_setup.py --install-optional --optional-groups pymdp`
- The execute module automatically detects wrong package variants and provides clear error messages

#### Issue 3: Execution timeout
**Symptom**: Scripts timeout before completion  
**Cause**: Simulation too complex or timeout too short  
**Solution**:
- Increase timeout: `--timeout 3600` (1 hour)
- Simplify model complexity
- Use faster frameworks (JAX) for large models
- Process models individually instead of batch

---

## Version History

### Current Version: [pyproject.toml](../../../pyproject.toml) (canonical)

**Features**:
- Multi-framework execution support
- Graceful degradation when frameworks unavailable
- Comprehensive error logging
- Result capture and validation
- Execution timeout handling

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Broader distributed execution coverage
- **Future**: Real-time execution monitoring

---

## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [Render Module](../render/AGENTS.md)
- [Execution Guide](../../../docs/execution/)

### External Resources
- [PyMDP Framework](https://github.com/infer-actively/pymdp)
- [RxInfer.jl](https://github.com/biaslab/RxInfer.jl)
- [ActiveInference.jl](https://github.com/ComputationalPsychiatry/ActiveInference.jl)
- [JAX Documentation](https://jax.readthedocs.io/)

---

**Last Updated**: 2026-09-24
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
