# RxInfer.jl Execute Submodule - Agent Scaffolding

## Module Overview

**Purpose**: Execution of RxInfer.jl (Julia) probabilistic models generated from GNN specifications

**Parent Module**: Execute Module (Step 12: Simulation execution)

**Category**: Framework Execution / RxInfer.jl Simulation

---

## Core Functionality

### Primary Responsibilities
1. Discover rendered RxInfer.jl scripts under `output/11_render_output/<model>/rxinfer/`
2. Run each script as a Julia subprocess under the committed project environment
3. Preserve canonical `simulation_results.json` artifacts for the Step 12 result
   extractor and Step 16 canonical analysis reader

### Key Capabilities
- Genuine ``@model`` + ``infer()`` variational message-passing inference execution (RxInfer.jl v5.5)
- **Offline batch inference (Bayesian smoothing)** — the pipeline runs ``infer()``
  on a pre-collected observation sequence, NOT online active inference. If
  ``infer()`` fails, the script crashes (no fallback).
- Reproducible Julia environment: `julia --startup-file=no --project=src/gnn/execute/rxinfer <script>`
  against the committed `Project.toml` + `Manifest.toml`; `setup_environment.jl`
  only runs `Pkg.instantiate()` (no runtime `Pkg.add`)
- Direct-file legacy result helpers (`rxinfer_results.py`) for saved
  `free_energy`/`posteriors` payloads; canonical results use the analysis reader
- Cross-platform compatibility (Linux/macOS/Windows)

---

## API Reference

The runner names below are re-exported from `execute.rxinfer` (`__all__` in
`__init__.py`) and defined in `rxinfer_runner.py`. Legacy result helpers are
separate direct-module APIs and are not wired into the numbered pipeline.

### Public Functions

#### `is_julia_available(min_version: tuple = (1, 9, 0)) -> bool`
**Description**: Return `True` when a `julia` executable is on `PATH` (shared helper from `execute.julia_setup`).

#### `find_rxinfer_scripts(base_dir: Union[str, Path], recursive: bool = True) -> List[Path]`
**Description**: Collect `*_rxinfer.jl` scripts (and any `*_config.toml` configs) under `base_dir`.

**Parameters**:
- `base_dir`: Directory to search (Step 12 passes `<render_dir>/rxinfer`)
- `recursive`: Whether to walk subdirectories

**Returns**: List of script paths (`.jl` scripts first, then `.toml` configs).

#### `execute_rxinfer_script(script_path: Path, verbose: bool = False, output_dir: Optional[Path] = None, timeout: int = 300) -> bool`
**Description**: Run one rendered script with
`julia --startup-file=no --project=<this directory> <script>`. `.toml` inputs are
handed to `rxinfer_runner.jl` instead.
The subprocess runs with the shared Julia environment: `GKSwstype=100` headless-GR default (a `GKSwstype` set in the caller's environment wins).

**Parameters**:
- `script_path`: Path to the rendered `.jl` script
- `verbose`: Log the script's stdout on success
- `output_dir`: Directory for execution evidence (`{stem}_stdout.txt`, `{stem}_stderr.txt`, `{stem}_execution_log.json`); defaults to the script's own directory
- `timeout`: Subprocess timeout in seconds

**Returns**: `True` on exit code 0, `False` on non-zero exit, timeout, missing or empty file.

#### `run_rxinfer_scripts(rendered_simulators_dir: Union[str, Path], execution_output_dir: Optional[Union[str, Path]] = None, recursive_search: bool = True, verbose: bool = False) -> bool`
**Description**: Find every RxInfer.jl script below `<rendered_simulators_dir>/rxinfer` and execute
them in sequence.

**Returns**: `True` only if every script succeeded (or none were found).

**Example**:
```python
from pathlib import Path
from gnn.execute.rxinfer import (
    execute_rxinfer_script,
    find_rxinfer_scripts,
    is_julia_available,
    run_rxinfer_scripts,
)

if is_julia_available():
    ok = run_rxinfer_scripts(
        rendered_simulators_dir=Path("output/11_render_output"),
        execution_output_dir=Path("output/12_execute_output"),
        verbose=True,
    )

    for script in find_rxinfer_scripts(Path("output/11_render_output"), recursive=True):
        execute_rxinfer_script(script, verbose=True, timeout=600)
```

### Legacy Result Helpers (`rxinfer_results.py`)
- `parse_rxinfer_output(output_path: Path) -> Optional[Dict]` — read one legacy `free_energy`/named `posteriors` JSON object; unreadable or canonical formats return `None` with contextual diagnosis
- `extract_convergence_metrics(parsed) -> Dict` — existing legacy free-energy trajectory and convergence flags
- `summarize_posteriors(parsed) -> Dict` — per-variable legacy posterior summaries, including numeric zero and scalar values
- `collect_rxinfer_results(output_dir: Path, model_name=None) -> List[Dict]` — collect legacy RxInfer-named JSON and `*simulation_results.json`, each matching path once; canonical artifacts are diagnosed and omitted
- `format_rxinfer_report(results) -> str` — Markdown report over collected legacy results

Canonical `rxinfer_simulation_v1` results belong to
[`gnn.analysis.rxinfer.result_ingestion.read_result_object`](../../analysis/rxinfer/result_ingestion.py)
and the [canonical analysis consumers](../../analysis/rxinfer/README.md).
`num_timesteps` is an observation count, not an inference-iteration count;
`model_parameters.inference_iterations` retains the latter identity.
Per-iteration `variational_free_energy`/`vfe_per_iteration` must remain separate
from per-step `expected_free_energy`. Gaussian `beliefs` are posterior means
paired with full per-timestep `posterior_cov` matrices. The legacy helper refuses
canonical schema or VFE/covariance fields rather than discarding these identities
and returning an apparently empty success. It does not add a new inference
contract or promote legacy helpers into the pipeline.

---

## Dependencies

### Required Dependencies
- `julia` (`Project.toml` compat `julia = "1.10"`)
- `RxInfer.jl` 5.5.0 plus `ReactiveMP.jl`, `GraphPPL.jl`, `JSON`, `Distributions`, `StatsBase` — all pinned by the committed `Project.toml` + `Manifest.toml` in this directory

### Internal Dependencies
- `execute.julia_setup` - Shared Julia availability check
- `execute.processor` - Step 12 orchestration (calls into this module per rendered script)
- `render.rxinfer` - Produces the scripts this module executes

---

## Configuration

This module carries no configuration dictionaries of its own. Timeouts and the
framework selection come from Step 12 (`--frameworks`, `timeout` kwarg in
`execute.processor`); the Julia environment is fixed by the committed
`Project.toml` + `Manifest.toml`.

---

## Output Specification

### Output Products
- `simulation_results.json` - Complete inference results (schema `rxinfer_simulation_v1`, written by the rendered Julia script)
- `simulation.log` / `simulation_log.json` - Best-effort runner logs (guarded; absence never fails a run)
- Step 12 folds every run into `output/12_execute_output/summaries/execution_summary.json` and `execution_report.md`

### Output Directory Structure
```
output/12_execute_output/
├── <model>/rxinfer/
│   ├── simulation_results.json
│   ├── simulation.log
│   └── simulation_log.json
└── summaries/
    ├── execution_summary.json
    └── execution_report.md
```

### Result Data Structure
The `rxinfer_simulation_v1` payload carries `schema_version`, `model_name`,
`runtime_metadata` (seed, script SHA256, Julia/RxInfer versions), the observation
and action sequences, per-timestep posterior beliefs, `variational_free_energy`
and convergence flags (`inference_converged`, `vfe_present`). Read the schema
from a generated file rather than from this document; the renderer strategies in
`src/gnn/render/rxinfer/_strategies_*.py` are the source of truth.

---

## Performance Characteristics

Do not treat timing numbers in documentation as current measurements. Julia
start-up and package precompilation dominate the first run; subsequent runs in
the same environment reuse the compiled cache. Read durations from
`summaries/execution_summary.json` after a run.

---

## Error Handling

### RxInfer.jl Execution Errors
1. **Julia Environment Issues**: `julia` missing from `PATH` (`is_julia_available()` returns `False`; Step 12 reports the framework as skipped)
2. **Environment Instantiation**: `setup_environment.jl` fails to `Pkg.instantiate()` the committed Manifest
3. **Script Errors**: the rendered script raises inside `@model`/`infer()`; the process exits non-zero and `execute_rxinfer_script` returns `False`
4. **Timeout**: the subprocess exceeds `timeout` seconds

### Recovery Strategies
- Install Julia and re-run `julia --startup-file=no --project=src/gnn/execute/rxinfer src/gnn/execute/rxinfer/setup_environment.jl`
- Inspect the stderr captured in the Step 12 log; there is no automatic retry or model simplification

---

## Integration Points

### Orchestrated By
- **Parent Module**: `src/gnn/execute/` (Step 12)
- **Main Script**: `12_execute.py`

### Imports From
- `execute.julia_setup` - Julia availability helper

### Imported By
- `execute.processor` - Main execution integration

### Data Flow
```
render.rxinfer (.jl script) → julia --project=src/gnn/execute/rxinfer → simulation_results.json → Step 12 summaries → Step 16 analysis
```

---

## Testing

### Test Files
- `tests/execute/test_execute_overall.py` - Execute module tests (includes framework selection)
- `tests/pipeline/test_pomdp_gridworld_cross_framework.py` - Cross-framework contract including RxInfer.jl
- `tests/render/test_rxinfer_*.py` - Renderer-side contracts for the scripts this module executes

### Test Commands
```bash
uv run --extra dev python -m pytest tests/execute/test_execute_overall.py \
    tests/pipeline/test_pomdp_gridworld_cross_framework.py -v

# With coverage
uv run --extra dev python -m pytest tests/execute/test_execute_overall.py \
    --cov=src/gnn/execute/rxinfer --cov-report=term-missing
```

---

## MCP Integration

This submodule registers no MCP tools of its own. The parent module
(`src/gnn/execute/mcp.py`) exposes `process_execute`, `execute_gnn_model`,
`execute_pymdp_simulation`, `check_execute_dependencies` and
`get_execute_module_info`; RxInfer.jl scripts are reached through
`process_execute` with the `rxinfer` framework selected.

---

## Development Guidelines

### Adding New RxInfer.jl Features
1. Update execution logic in `rxinfer_runner.py`
2. Change the generated Julia in `src/gnn/render/rxinfer/` (this module only runs what the renderer emits)
3. Update `Project.toml`/`Manifest.toml` together and re-run `setup_environment.jl`
4. Add or extend tests in `tests/execute/` and `tests/render/`

---

## Troubleshooting

### Common Issues

#### Issue 1: "Julia not found in PATH"
**Symptom**: Step 12 reports the RxInfer.jl framework as skipped
**Cause**: Julia not installed or not in system PATH
**Solution**: Install Julia and ensure `julia` resolves on PATH

#### Issue 2: "RxInfer.jl package not available"
**Symptom**: `using RxInfer` fails inside the script
**Cause**: Committed environment not instantiated
**Solution**: `julia --startup-file=no --project=src/gnn/execute/rxinfer -e 'using Pkg; Pkg.instantiate()'`

#### Issue 3: Script exits non-zero
**Symptom**: `execute_rxinfer_script` returns `False`
**Cause**: The rendered model raised during `infer()`; there is no fallback path
**Solution**: Re-run with `verbose=True`, read the captured stderr, fix the GNN specification or renderer

---

## Version History

### Current Version: [pyproject.toml](../../../../pyproject.toml) (canonical)

**Features**:
- RxInfer.jl script discovery and subprocess execution under the committed project
- Genuine `@model` + `infer()` variational message-passing inference
- Result parsing helpers in `rxinfer_results.py`

**Known Limitations**:
- Requires a local Julia runtime
- Offline batch inference only (no online active inference)

---

## References

### Related Documentation
- [Execute Module](../AGENTS.md) - Parent execute module
- [RxInfer.jl Render](../../render/rxinfer/AGENTS.md) - RxInfer.jl code generation
- [RxInfer.jl Documentation](https://docs.rxinfer.com/stable/) - Official RxInfer.jl docs

### External Resources
- [Julia Language](https://julialang.org/)
- [Message Passing](https://en.wikipedia.org/wiki/Belief_propagation)

---

**Last Updated**: 2026-10-08
**Maintainer**: Execute Module Team
**Status**: ✅ Production Ready
