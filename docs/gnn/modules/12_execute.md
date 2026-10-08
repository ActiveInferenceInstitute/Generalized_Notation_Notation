# Step 12: Execute

## Architectural Mapping

**Orchestrator**: `src/gnn/12_execute.py` (109 lines)
**Implementation Layer**: `src/gnn/execute/`

## Module Description

This module is responsible for running GNN models that have been rendered into framework-specific simulation code by Step 11 (`11_render.py`).


| Framework | Language | Subfolder | Script Pattern | Status |
|-----------|----------|-----------|----------------|--------|
| **PyMDP** | Python | `pymdp/` | `*_pymdp.py` | ✅ Full support |
| **RxInfer.jl** | Julia | `rxinfer/` | `*_rxinfer.jl` | ✅ Full support |
| **ActiveInference.jl** | Julia | `activeinference_jl/` | `*_activeinference.jl` | ✅ Full support |
| **JAX** | Python | `jax/` | `*_jax.py` | ✅ Full support |
| **DisCoPy** | Python | `discopy/` | `*_discopy.py` | ✅ Full support |
| **PyTorch** | Python | `pytorch/` | `*_pytorch.py` | ✅ Full support |
| **NumPyro** | Python | `numpyro/` | `*_numpyro.py` | ✅ Full support |
| **Stan** | Python driver (cmdstanpy) | `stan/` | `*_stan.py` | ✅ (skipped when cmdstanpy/CmdStan absent) |
| **ngc-learn** | Python | `ngclearn/` | `*_ngclearn.py` | ✅ (skipped without the py3.12 marker-gated `ngclearn` extra) |
| **bnlearn** | Python (generator-backed; `.R` lane via Rscript) | `bnlearn/` | `*_bnlearn.py`, `*.R` | ✅ (skipped when the `bnlearn` extra / R bnlearn runtime is absent; executor `src/gnn/execute/bnlearn/`) |

JAX, NumPyro and DisCoPy are **core** dependencies (`uv sync`); PyTorch needs the `torch` extra (`uv sync --extra torch`; torch>=2.13.0 resolves GHSA-rrmf-rvhw-rf47), bnlearn needs the `bnlearn` extra (`uv sync --extra bnlearn`), and Stan needs `uv sync --extra stan` plus a CmdStan toolchain. If the environment is incomplete, the affected scripts are **skipped** (not failed). Julia frameworks require Julia installed.

Two behaviours introduced in v3.2.0: `_merge_prior_execution_summary` (`src/gnn/execute/metadata.py`, re-exported through the `src/gnn/execute/processor/` facade) folds a previously written `execution_summary.json` into the current results so the durable summary covers every input folder rather than the last one processed; and script discovery only considers `.py`/`.jl` files, so companion artifacts such as `<stem>_stan.stan` and `<stem>_stan_data.json` are never treated as executables.

Continuous (linear-Gaussian) models reach Step 12 only for the backends that render them (JAX, NumPyro, PyTorch, Stan, RxInfer.jl); the categorical backends report render status `unsupported` in Step 11 and emit nothing to execute.

## Agent Identity & Capabilities

# Execute Module - Agent Scaffolding

## Module Overview

**Purpose**: Execute rendered simulation scripts across multiple frameworks (PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, Stan, Lean, bnlearn, ngc-learn).

**Pipeline Step**: Step 12: Execution (12_execute.py)

**Category**: Simulation / Execution

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-21

---

## Core Functionality

### Primary Responsibilities
1. Execute Python simulation scripts (PyMDP, JAX, DisCoPy)
2. Execute Julia simulation scripts (RxInfer.jl, ActiveInference.jl)
3. Capture simulation results and logs
4. Handle execution errors gracefully
5. Generate execution reports

### Key Capabilities
- Multi-framework execution support
- **Skip vs fail**: JAX, NumPyro and DisCoPy are **core** dependencies; PyTorch comes from the `torch` extra (GHSA-rrmf-rvhw-rf47 resolved in torch 2.13.0) and bnlearn comes from the locked `bnlearn` extra; Stan needs `uv sync --extra stan` plus a CmdStan toolchain; ngc-learn comes from the py3.12 marker-gated `ngclearn` extra (`uv sync --extra ngclearn`). If the environment is incomplete, scripts are **skipped** (not run) and reported as "skipped" — they do not count as execution failures. Repair with `uv sync` (plus the extras above). Julia backends still require a local Julia install.
- **Committed Julia environments** with `JULIA_PROJECT` defaulting (see below)
- Graceful degradation when frameworks unavailable
- Automatic PyMDP package detection (distinguishes correct vs wrong package variants)
- Path collection with deduplication (prevents nested directory issues)
- Comprehensive error logging
- Result capture and validation
- Execution timeout handling
- Distributed execution across a Ray or Dask cluster for parallel script/parameter-sweep dispatch (`src/gnn/execute/distributed.py`, `--distributed`, `--execution-workers`, `--backend {ray,dask}`)

---

## Julia Execution Environments

Both Julia backends run against **committed** environments checked into the repository, so a run never depends on an ambient Julia depot or a runtime `Pkg.add`.

| Framework | Environment | Pinned contents |
|---|---|---|
| RxInfer.jl | `src/gnn/execute/rxinfer/` | The `GnnRxInferModels` package — RxInfer 5.5.0 plus Distributions, JSON, Plots, StatsBase, PrecompileTools. Precompiles the `pomdp`, `continuous`, `hierarchical`, `factored`, and `learning` models loudly: a precompile failure surfaces rather than being swallowed. |
| ActiveInference.jl | `src/gnn/execute/activeinference_jl/` | A deliberately **minimal** environment — ActiveInference 0.1.2, Distributions, JSON, StatsBase, and nothing else. |

Environment hygiene (2026-09-22 re-check): both projects share one Julia
compatibility floor (`julia = "1.10"`), both committed `Manifest.toml` files
were regenerated with Julia 1.12.7, and both projects carry real generated
package UUIDs with a matching package source file (`src/GnnRxInferModels.jl`,
`src/GnnActiveInferenceModels.jl`) that `Pkg.instantiate()` precompiles. The
ActiveInference environment keeps its `Distributions = "0.25.100 - 0.25.125"`
compat pin: the archived DistributionsAD.jl repository has shipped no release
fixing the ReverseDiff-extension precompile break against Distributions >=
0.25.127 (re-checked 2026-09-22).

### `JULIA_PROJECT` defaulting

`_build_execution_environment` (`src/gnn/execute/processor/single.py`) sets `JULIA_PROJECT` to the committed environment matching the script's framework, using `setdefault` — **an explicitly exported `JULIA_PROJECT` still wins**. This is what lets `using GnnRxInferModels` / `using ActiveInference` resolve without an ambient environment, including under test runners whose temporary depot may not exist.

`setup_environment.jl` activates and instantiates the environment (`Pkg.activate()` + `Pkg.instantiate()`); there is no runtime `Pkg.add`.

### Skip semantics

A backend whose dependency is absent produces a **skipped** result (`skipped: true`) carrying an explicit dependency reason, not a failure. Skips are counted separately from failures in the step summary and are excluded from the failure threshold that determines step success — so the completion line reads, per model, in the shape `N succeeded, M skipped (dependency not installed)`. On a fully-provisioned run the remaining skips are the unprovisioned locked optional extras (bnlearn and PyTorch on a plain `uv sync`). Without the py3.12-gated `ngclearn` extra the ngc-learn backend records status `SKIPPED` (`skipped: true`) carrying the dependency reason and install hint (`uv sync --extra ngclearn`), with zero failures attributed.

### Exit-code contract

Rendered RxInfer scripts end with `return results["validation"]["all_valid"] ? 0 : 1`. The validation block — belief validity, normalisation, action range, and for continuous models VFE finiteness — therefore **drives the process exit code**, so inference that runs to completion but produces invalid posteriors is surfaced as a failed script rather than a silent success. The results payload is still written either way, so a failing run remains diagnosable.

## Backend Contract (Registry and Dispatch)

The `GNNExecutor` registry (`src/gnn/execute/executor.py`) carries one
`ExecutorFrameworkSpec` per backend — PyMDP, RxInfer.jl, DisCoPy,
ActiveInference.jl, JAX, NumPyro, PyTorch, ngc-learn, Lean, Stan, and bnlearn —
and the `execute_gnn_model` dispatch accepts all eleven execution types.

- **Stan** runs as an availability-gated registry runner: `is_stan_available()`
  probes `cmdstanpy` plus the CmdStan toolchain, and an unavailable toolchain
  produces a `"SKIPPED"` record with the `uv sync --extra stan` hint — never a
  failure.
- **bnlearn** carries a render-only verdict: the registry never executes it and
  records `"SKIPPED"` with that reason. Rendered bnlearn scripts continue to
  execute through the Step 12 script path (`BNLEARN_OUTPUT_DIR`), where the
  shared pre-flight probe applies.
- **RxInfer dispatch route**: `GNNExecutor._execute_rxinfer_config` executes
  both `.jl` scripts and `_config.toml` inputs through
  `gnn.execute.rxinfer.execute_rxinfer_script` — the committed `--project`
  environment, the security gate, and the `{stem}_stdout.txt` /
  `{stem}_stderr.txt` / `{stem}_execution_log.json` evidence sidecars — and
  synthesizes the dispatch envelope from the runner verdict and those
  sidecars. A cooperative `CancelToken` is threaded into the subprocess.
- **Cross-framework comparison** (`src/gnn/analysis/rxinfer/cross_framework.py`):
  `compare_with_status(gnn_file, output_dir, timeout, runtime)` renders one
  parsed spec to six backends — RxInfer.jl, PyMDP, ActiveInference.jl, JAX,
  PyTorch, NumPyro — executes each, and writes a self-contained comparison
  HTML plus per-framework status records. A backend whose dependency is
  missing records an `unavailable` status with the install hint (a skip
  receipt, never a failure) before any render is attempted. Exposed as the
  MCP tool `run_cross_framework_comparison`; Step 13 (LLM) injects a compact
  status summary of a prior comparison into its prompt context when the
  comparison artifact exists in the Step 12 output tree.
- **PyMDP success semantics** align with the other runners in
  `pymdp_runner.run_pymdp_scripts`: the run succeeds only when every executed
  script succeeds (`failure_count == 0`); a render tree with no scripts stays a
  success. Mixed-failure runs therefore fail the registry lane instead of
  counting as partial success.
- The markdown execution report (`summaries/execution_report.md`) writes one
  subsection per backend, including the Stan and bnlearn receipts.

---

## API Reference

### Public Functions

#### `process_execute(target_dir: Path, output_dir: Path, verbose: bool = False, logger: Optional[logging.Logger] = None, **kwargs) -> bool`
**Description**: Main execution function called by orchestrator (12_execute.py). Executes rendered simulation scripts across multiple frameworks.

**Parameters**:
- `target_dir` (Path): Directory containing rendered scripts (typically output from Step 11)
- `output_dir` (Path): Output directory for execution results
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Optional[logging.Logger]): Logger instance (default: None)
- `frameworks` (str): Frameworks to execute ("all", "lite", or comma-separated list, default: "all")
  - `"all"`: PyMDP, JAX, DisCoPy, RxInfer.jl, ActiveInference.jl, PyTorch, NumPyro, Stan, bnlearn
  - `"lite"`: the Python-only subset — PyMDP, JAX, DisCoPy, bnlearn (no Julia)
  - Comma-separated: `"pymdp,jax"` for specific frameworks; names outside the valid set are filtered out
- `render_output_dir` (str | Path): Explicit path to the Step 11 render output directory (default: filesystem heuristics via `_resolve_render_output_dir`, CLI: `--render-output-dir`). An explicit value is used verbatim even when nonexistent or empty (execution then records `skipped_reason: no_render_output`); the heuristic candidates — the sibling `11_render_output` of this step's output base, `target_dir` itself when it names a render output directory, then common pipeline/test locations — must exist and be non-empty to be selected.
- `execution_benchmark_repeats` (int): Sequential benchmark repeats per rendered script; when >1 the summary reports the median duration (default: 1, CLI: `--execution-benchmark-repeats`)
- `execution_summary_detail` (bool): Also write `summaries/execution_summary_detail.json` with full per-script payloads; the aggregate `execution_summary.json` stays slim (default: False, CLI: `--execution-summary-detail`)
- `require_render_summary` (bool): Enforce the Step 11 render summary contract: when `render_processing_summary.json` is missing or invalid (or the render output directory is unavailable), no scripts execute and the result records `missing_render_summary` (default: True; with False, scripts are discovered from the filesystem instead)
- `run_id` (str, optional): Correlates this invocation with the folder-scoped render summary contract and is recorded in the execution summary (default: None)
- `distributed_max_retries` (int): Retries for distributed script/parameter-sweep dispatch, used only when `distributed` is True (default: 3)
- `timeout` (int): Execution timeout per script in seconds (default: 3600, CLI: `--timeout`)
- `distributed` (bool): Run scripts and parameter sweeps in parallel across a Ray/Dask cluster (default: False, CLI: `--distributed`)
- `execution_workers` (int): Number of local or distributed workers for rendered script execution (default: 1, CLI: `--execution-workers`)
- `backend` (str): Backend for distributed execution, `"ray"` or `"dask"` (default: `"ray"`, CLI: `--backend {ray,dask}`)
- `**kwargs`: Additional framework-specific options

**Returns**: `bool` - True if execution succeeded, False otherwise

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
)
```

#### `execute_simulation_from_gnn(gnn_file: Path, output_dir: Path) -> Dict[str, Any]`
**Description**: Execute the simulation for one GNN file (`src/gnn/execute/processor/pipeline.py`, re-exported through the facade).

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

#### Framework health
There is no `get_execution_health_status` function in `src/gnn/execute/`. Framework availability is probed by `gnn.utils.runtime_safety.framework_availability` and surfaced through the `gnn health` CLI; the fields below describe that report.

**Returns**: `Dict[str, Any]` - Health status dictionary with:
- `pymdp_available` (bool): PyMDP availability
- `rxinfer_available` (bool): RxInfer.jl availability
- `activeinference_jl_available` (bool): ActiveInference.jl availability
- `jax_available` (bool): JAX availability
- `discopy_available` (bool): DisCoPy availability
- `julia_available` (bool): Julia installation status
- `python_version` (str): Python version
- `julia_version` (Optional[str]): Julia version if available

#### Capability doctor
**Function**: `collect_doctor_report(target_dir=None, output_dir=None, frameworks="all") -> Dict[str, Any]` (`execute.doctor`; MCP tool `get_doctor_report`). One structured offline probe composing per-framework availability (the canonical `gnn.utils.runtime_safety.framework_availability` records plus the shared Julia PATH gate) with the `plan_execute` Step 12 dry-run when a target/output directory pair is supplied (both or neither — exactly one raises `ValueError`).

**Returns**: `Dict[str, Any]` report with:
- `frameworks` (dict): per-framework records keyed by name; `kind` is `python_import` (with `probe_module`, availability, and `missing_module`/`install_hint` when unavailable; `toolchain_probe` for Stan) or `julia_toolchain` (mirrors the `julia` section)
- `julia` (dict): `available` plus the executable `path` when found
- `frameworks_available` / `frameworks_missing` (list[str]): the partitioned name lists
- `execution` (dict): the full `plan_execute` plan, or `{"status": "not_probed", ...}` when no directory pair was supplied
- `execution_ready` (Optional[bool]): `plan["status"] == "ready"`; `None` when not probed

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

#### Distributed Execution
- `distributed` (bool, CLI: `--distributed`): Run scripts and model parameter sweeps in parallel across a Ray/Dask cluster (default: `False`)
- `execution_workers` (int, CLI: `--execution-workers`): Number of local or distributed workers for rendered script execution (default: `1`)
- `backend` (str, CLI: `--backend {ray,dask}`): Backend to use for distributed execution (default: `"ray"`); implemented by `src/gnn/execute/distributed.py`'s `Dispatcher` class

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
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/12_execute_output"),
)
```

### Distributed Execution
```bash
# Fan out rendered scripts / parameter sweeps across a local or remote Ray cluster
uv run python src/gnn/12_execute.py --target-dir output/11_render_output --output-dir output \
  --distributed --execution-workers 4 --backend ray

# Use Dask instead of Ray
uv run python src/gnn/12_execute.py --target-dir output/11_render_output --output-dir output \
  --distributed --execution-workers 4 --backend dask
```

---

## Output Specification

### Output Products
- `summaries/execution_summary.json` - Execution summary, merged across per-folder invocations by `_merge_prior_execution_summary`
- `summaries/execution_summary_detail.json` - Per-script detail
- `summaries/execution_report.md` - Human-readable report
- `<model>/<framework>/execution_logs/` - Per-script results JSON and log
- `<model>/<framework>/simulation_data/simulation_results.json` - Simulation output data

### Output Directory Structure
```
output/12_execute_output/
├── summaries/
│   ├── execution_summary.json
│   ├── execution_summary_detail.json
│   └── execution_report.md
└── <model>/
    └── <framework>/
        ├── execution_logs/
        │   ├── <script>_results.json
        │   └── <script>_execution.log
        └── simulation_data/
            └── simulation_results.json
```

---

## Performance Characteristics

Per-run duration, memory, and per-script outcomes are recorded in `execution_summary.json` and the pipeline execution summary. Read those for current numbers rather than hard-coding a snapshot here.

### Framework Execution Times (indicative)
- **PyMDP**: ~1-5 seconds
- **RxInfer.jl**: ~10-20 seconds (JIT compilation)
- **ActiveInference.jl**: ~10-15 seconds
- **JAX**: ~2-8 seconds (with GPU)
- **DisCoPy**: ~1-3 seconds

---

## Error Handling

### Graceful Degradation
- **PyMDP unavailable**: Log warning, skip PyMDP scripts
- **Julia unavailable**: Log warning, skip Julia scripts
- **JAX unavailable**: Log warning, skip JAX scripts
- **Script errors**: Capture stderr, continue with other scripts
- **Timeout**: 3600s per script (configurable via `--timeout`)

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
- **Dependencies**: Requires rendered code from `11_render.py` output

### Module Dependencies
- **render/**: Consumes rendered simulation scripts
- **llm/**: Provides execution results for LLM analysis
- **analysis/**: Provides execution data for statistical analysis
- **report/**: Provides execution summaries for reports

### External Integration
- **PyMDP**: Executes Python Active Inference simulations
- **Julia Runtime**: Executes Julia simulation scripts (RxInfer.jl, ActiveInference.jl) under the committed environments described in [Julia Execution Environments](#julia-execution-environments). Equivalent by hand: `julia --startup-file=no --project=src/gnn/execute/rxinfer <script>`.
- **JAX**: Executes JAX-based simulations
- **DisCoPy**: Executes categorical diagram computations

### Data Flow
```
11_render.py (Code generation)
  ↓
12_execute.py (Script execution)
  ↓
  ├→ 13_llm.py (LLM analysis of results)
  ├→ 16_analysis.py (Statistical analysis)
  ├→ 23_report.py (Execution reports)
  └→ output/12_execute_output/ (Execution results)
```

---

## Testing

### Test Files
- `tests/execute/test_execute_overall.py`
- `tests/execute/test_execute_pymdp_integration.py`
- `tests/execute/test_execute_pymdp_package.py`

### Test Coverage
- Measure: `uv run --extra dev python -m pytest tests/execute/ --cov=gnn.execute --cov-report=term-missing` (do not treat fixed percentages in this doc as canonical).

### Key Test Scenarios
1. Multi-framework execution
2. Error handling and recovery
3. Result capture and validation
4. Timeout handling

---

## MCP Integration

### Tools Registered
- `process_execute` - Run Step 12 over a render output directory
- `execute_gnn_model` - Execute one rendered GNN model
- `execute_pymdp_simulation` - Execute one PyMDP simulation script
- `check_execute_dependencies` - Report framework/runtime availability
- `get_execute_module_info` - Module metadata
- `get_doctor_report` - Composed availability + Step 12 readiness report
- `run_cross_framework_comparison` - Render, execute, and compare one model across six backends; returns the comparison HTML path and per-framework statuses

### Tool Endpoints
```python
@mcp_tool("execute_gnn_model")
def run_simulation_tool(script_path: str, framework: str) -> Dict[str, Any]:
    """Execute simulation script"""
    # Implementation
```

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
- For Julia: the RxInfer.jl environment is pinned by the committed `Project.toml` + `Manifest.toml` under `src/gnn/execute/rxinfer/` (RxInfer 5.5.0). `setup_environment.jl` activates and instantiates it (`Pkg.activate()` + `Pkg.instantiate()`) — there is no runtime `Pkg.add`.
- Check framework-specific requirements in documentation

#### Issue 2a: Wrong PyMDP package installed
**Symptom**: Error message "Wrong pymdp package installed. Found 'pymdp' with MDP/MDPSolver"  
**Cause**: The wrong `pymdp` package (with MDP/MDPSolver) is installed instead of `inferactively-pymdp`  
**Solution**:
- Uninstall wrong package: `uv pip uninstall pymdp`
- Install correct package: `uv pip install inferactively-pymdp`
- Or use setup module: `uv run python src/gnn/1_setup.py --install-optional --optional-groups pymdp`
- The execute module automatically detects wrong package variants and provides clear error messages

#### Issue 3: Execution timeout
**Symptom**: Scripts timeout before completion  
**Cause**: Simulation too complex or timeout too short  
**Solution**:
- Increase timeout beyond the default 3600s (1 hour): `--timeout 7200` (2 hours)
- Simplify model complexity
- Use faster frameworks (JAX) for large models
- Process models individually instead of batch

---

## Version History

### Current package version

See [pyproject.toml](../../../pyproject.toml).

**Features**:
- Multi-framework execution support
- Graceful degradation when frameworks unavailable
- Comprehensive error logging
- Result capture and validation
- Execution timeout handling
- Distributed execution across a Ray/Dask cluster (`--distributed`, `--execution-workers`, `--backend`)

**Known Issues**:
- None currently

### Roadmap
- **Future**: Real-time execution monitoring

---

## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [Render Module](../../../src/gnn/execute/../render/AGENTS.md)
- [Execution Guide](../../../docs/execution/)

### External Resources
- [PyMDP Framework](https://github.com/infer-actively/pymdp)
- [RxInfer.jl](https://github.com/biaslab/RxInfer.jl)
- [ActiveInference.jl](https://github.com/ComputationalPsychiatry/ActiveInference.jl)
- [JAX Documentation](https://jax.readthedocs.io/)

---

**Last Updated**: 2026-09-21
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern


---
## Documentation
- **[README](../../../src/gnn/execute/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/execute/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/execute/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/execute/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/execute](../../../src/gnn/execute)
