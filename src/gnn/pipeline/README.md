# Pipeline Module

This module provides core pipeline orchestration, configuration management, and step coordination for the GNN processing pipeline. It manages the 25-step pipeline execution, configuration handling, and inter-module communication.

## Module Structure

```
src/gnn/pipeline/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
├── AGENTS.md                      # Agent scaffolding documentation
├── config.py                      # Pipeline configuration management
├── step_executor.py                # Consolidated in-process step execution (ADR 0001)
├── health_check.py               # Pipeline health checker
├── diagnostic_enhancer.py        # Diagnostic enhancement
├── execution.py                   # Pipeline execution
├── discovery.py                  # Pipeline discovery
├── mcp.py                        # MCP integration
├── verify_pipeline.py            # Pipeline verification
├── durable_streams.py            # v3: stream manifests + replayable execution traces
├── run_session.py                # v3: resumable run sessions (checkpoint/resume/cleanup)
├── container_plan.py             # v3: auditable container plans + static security review
├── dag.py                        # Dependency-graph construction and step ordering
├── step_registry.py              # Canonical step registry (names, numbers, scripts)
```

### Also exported

Beyond the API classes above, `gnn.pipeline.__all__` exports `STEP_METADATA`, `DEFAULT_TARGET_DIR`, `DEFAULT_OUTPUT_DIR`, `FEATURES`, `__version__`, `get_module_info`, `create_pipeline_config`, `get_pipeline_info`, `validate_pipeline_config`, and `EnhancedHealthChecker`.

### Runtime validation workspace

`python -m gnn.pipeline.pipeline_runtime_validator` runs the documented
validation selection against the caller's `input/gnn_files`, `input/config.yaml`
and `output`. The Python API `PipelineValidator.test_pipeline_execution(["3"])`
can select only parsing. The package must already be installed or importable;
the validator launches the same interpreter with safe-path module execution,
so a workspace `gnn/` cannot replace the package through the default cwd entry.
The subprocess retains its 300-second timeout. It does not redirect relative
data paths into the package checkout or installation directory.

Each validation execution supplies a fresh `GNN_RUN_ID` in the child environment
without changing the parent's identity or context. Step results are admitted
only from `output/00_pipeline_summary/pipeline_execution_summary.json` carrying
that exact identity. A missing, malformed or other-run summary cannot validate
success, including after failed admission; earlier-format summary locations
are not current-run evidence. This diagnostic interface does not add the
API's cancellation contract or an operating-system sandbox.

### v3.0.0 Long-Running Orchestration (safe-by-design)

Three modules provide the foundation for long-running, resumable, deployable-by-plan runs **without
any live infrastructure mutation** (no container execution, no cluster/sensor calls):

- **`durable_streams.py`** — `StreamManifest` (file/array, content-checksummed) and `ExecutionTrace`
  with integrity checks + deterministic replay.
- **`run_session.py`** — `RunSession` manifests with atomic checkpoint/resume, status inspection, and
  path-safe cancellation cleanup.
- **`container_plan.py`** — hardened container plan generation, static `security_review`
  (CRITICAL/HIGH/MEDIUM/LOW), rollback descriptors, and deterministic plan hashes.

Full reference: [`docs/pipeline/v3_orchestration.md`](../../../docs/pipeline/v3_orchestration.md). Acceptance
gate: `PYTHONPATH=src uv run python scripts/run_v3_orchestration_acceptance.py --strict`.

### Pipeline Flow

```mermaid
graph TD
    Config[Pipeline Config] --> Orchestrator[Pipeline Orchestrator]
    Steps[Steps Registry] --> Orchestrator
    
    Orchestrator -->|Sequence| Step1[Setup]
    Step1 --> Step2[GNN Parse]
    Step2 --> Step3[Validation]
    Step3 --> Step4[Export]
    Step4 --> Step5[...]
    
    Orchestrator -->|Tracks| State[Pipeline State]
    State --> Report[Final Report]
    
    subgraph "Orchestration Layer"
    Orchestrator
    State
    end
```

### Pipeline Orchestration Architecture

```mermaid
sequenceDiagram
    participant Main as main.py
    participant Config as Config Manager
    participant Orchestrator as Pipeline Orchestrator
    participant Step as Pipeline Step
    participant Monitor as Health Monitor
    participant Logger as Logger
    
    Main->>Config: Load pipeline configuration
    Config-->>Main: Configuration dict
    
    Main->>Orchestrator: Initialize orchestrator
    Orchestrator->>Monitor: Start health monitoring
    Orchestrator->>Logger: Setup logging
    
    loop For each step
        Main->>Orchestrator: Execute step
        Orchestrator->>Step: Validate prerequisites
        Step-->>Orchestrator: Prerequisites OK
        Orchestrator->>Step: Execute step function
        Step->>Logger: Log step start
        Step-->>Orchestrator: Step result
        Orchestrator->>Monitor: Update health status
        Orchestrator->>Logger: Log step completion
    end
    
    Orchestrator->>Monitor: Generate health report
    Orchestrator-->>Main: Pipeline complete
    Main->>Logger: Log pipeline summary
```

### Step Execution Flow

```mermaid
flowchart TD
    Start[Step Execution Start] --> Validate{Validate<br/>Prerequisites}
    Validate -->|Pass| Init[Initialize Step]
    Validate -->|Fail| Error[Log Error & Skip]
    
    Init --> Execute[Execute Step Function]
    Execute --> Check{Success?}
    
    Check -->|Yes| LogSuccess[Log Success]
    Check -->|No| Retry{Retries<br/>Remaining?}
    
    Retry -->|Yes| RetryStep[Retry Step]
    RetryStep --> Execute
    Retry -->|No| LogError[Log Error]
    
    LogSuccess --> UpdateState[Update Pipeline State]
    LogError --> UpdateState
    Error --> UpdateState
    
    UpdateState --> Next{More Steps?}
    Next -->|Yes| Start
    Next -->|No| Complete[Pipeline Complete]
```

## Core Components

### Pipeline Configuration Management (`config.py`)

#### `get_pipeline_config() -> dict`
Returns the pipeline configuration as a plain dict (steps, timeout, parallel).

#### `set_pipeline_config(config: PipelineConfig) -> None`
Saves a new pipeline configuration.

#### `get_output_dir_for_script(script_name: str, base_output_dir: Path) -> Path`
Returns the standardized per-step output directory (e.g. `output/3_gnn_output/`).

### Pipeline Execution (`execution.py`)

#### `run_pipeline(pipeline_data=None, *, target_dir=None, output_dir=None, steps=None, verbose=None, strict=None, parallel=None, consolidated_steps=None) -> dict`
Executes the pipeline through `main.py` and returns a compact summary dict
(`success`, `steps_executed`, `errors`, `exit_code`, `duration`, ...).
Explicit keyword flags override `pipeline_data`, including `False`; omitted
flags inherit it. Accepted data keys are paths (`target_dir`, `input_dir`,
`temp_dir`, `output_dir`), selections (`steps`, `only_steps`, `skip_steps`) and
the four flags above. Unknown keys fail admission. Steps accept registered
integer/name aliases; empty, unknown, duplicate, boolean and fractional execution
selections fail before dispatch. Omit steps or use `"all"` to select all.

Every invocation resolves prerequisites, freezes exact model bytes and original
source identities, and owns current-run artifact receipts. Serial, parallel and
matrix dispatch consume that shared plan. An explicitly empty **model** selection
is valid and yields skipped steps; it never expands to the input directory.
Results include `run_id`, `run_hash`, `model_selection` and per-step `status` and
artifact records. Inherited summaries are ignored. Exit 2 completes with warnings
unless `strict=True`.

#### `resolve_step_numbers(steps, pipeline_data=None) -> list[int]`
Normalizes step identifiers (`"11"`, `"11_render"`, `"3,5"`, iterables, `"all"`)
to sorted registered step numbers. This discovery helper retains its permissive
unknown-token filtering; execution independently performs strict admission.

#### `execute_pipeline_step(step_name, step_config, pipeline_data) -> StepExecutionResult`
Executes one step, returning a result object with `step_name`, `success`,
`duration`, `output`, `error`, `warnings`, and additive `status`, `run_id`, `artifacts`.

#### `execute_pipeline_steps(steps, pipeline_data)`
Executes the complete requested plan and prerequisites once. Requested outcomes
return in canonical pipeline order under one run identity. An empty convenience
batch returns no outcomes and performs no work.

#### `get_pipeline_status() -> dict`
Static readiness probe (`status`, `timestamp`, `steps_available`,
`steps_completed`); live progress lives in the per-run
`pipeline_execution_summary.json`.

#### `validate_pipeline_config(config: dict) -> bool`
Validates a pipeline configuration dict.

### Step Ordering (`dag.py`)

#### `resolve_execution_order(step_dependencies, total_steps=None, skip_steps=None, raise_on_circular=False) -> List[List[int]]`
Topological sort of **step numbers** into parallel tiers; circular
dependencies raise `ValueError` when `raise_on_circular=True`, otherwise form
a final tier.

#### `find_circular_dependencies(step_dependencies, nodes=None) -> Set[int]`
Returns the step numbers bound up in dependency cycles.

#### `visualize_dag(tiers, step_names=None) -> str`
Renders DAG tiers as a human-readable multi-line string.

### Step Registry (`step_registry.py`)

The canonical `STEPS` list (25 `StepInfo` entries) with lookups `step_for_name`,
`step_for_stem`, tag filters `get_core_steps()` / `get_llm_steps()`, and stage
definitions in `STAGE_DEFINITIONS`. Step-level prerequisites are validated by
`gnn.utils.pipeline_orchestration.pipeline_validator`; there are no `register_step` / `get_step_status`
/ `reset_step` runtime registration APIs — steps are declared in `STEPS`.

## Usage Examples

### Running Steps Through `main.py`

In practice the pipeline runs through `src/gnn/main.py` (CLI flags `--only-steps`,
`--skip-steps`, etc.); the `pipeline.execution` helpers below are programmatic
wrappers around that entry point.

```python
from gnn.pipeline.execution import run_pipeline, execute_pipeline_step, get_pipeline_status

# Execute the pipeline for a target directory (returns a compact summary dict)
summary = run_pipeline(
    target_dir="input/gnn_files",
    output_dir="output",
    steps="all",
)

# Execute a single step
result = execute_pipeline_step(
    step_name="6_validation",
    step_config={},
    pipeline_data={"target_dir": "input/gnn_files", "output_dir": "output"},
)

# Inspect pipeline status
status = get_pipeline_status()
```

### DAG-Based Ordering

```python
from gnn.pipeline.dag import find_circular_dependencies, resolve_execution_order
from gnn.utils.pipeline_orchestration.pipeline_step_dependencies import PIPELINE_STEP_DEPENDENCIES

# Resolve tiers over the canonical dependency table
tiers = resolve_execution_order(dict(PIPELINE_STEP_DEPENDENCIES))
print(f"Resolved tiers: {tiers}")

# Detect cycle-bound steps in an arbitrary graph
assert find_circular_dependencies({0: [1], 1: []}) == set()
```

### Configuration

```python
from gnn.pipeline.config import get_pipeline_config, get_output_dir_for_script
from pathlib import Path

config = get_pipeline_config()
print(f"Steps: {config['steps']}")

output_dir = get_output_dir_for_script("3_gnn.py", Path("output"))
print(f"GNN output directory: {output_dir}")
```

## Pipeline Structure

### 25-Step Pipeline (Current)
The pipeline consists of exactly 25 steps (steps 0-24), executed in order:

1. **0_template.py** → `src/gnn/template/` - Pipeline template and initialization
2. **1_setup.py** → `src/gnn/setup/` - UV environment setup and dependency management
3. **2_tests.py** → `tests/` - Comprehensive test suite execution
4. **3_gnn.py** → `src/gnn/` - GNN file discovery, multi-format parsing, and validation
5. **4_model_registry.py** → `src/gnn/model_registry/` - Model registry management and versioning
6. **5_type_checker.py** → `src/gnn/type_checker/` - GNN syntax validation and resource estimation
7. **6_validation.py** → `src/gnn/validation/` - Advanced validation and consistency checking
8. **7_export.py** → `src/gnn/export/` - Multi-format export (JSON, XML, GraphML, GEXF, Pickle)
9. **8_visualization.py** → `src/gnn/visualization/` - Graph and matrix visualization generation
10. **9_advanced_viz.py** → `src/gnn/advanced_visualization/` - Advanced visualization and interactive plots
11. **10_ontology.py** → `src/gnn/ontology/` - Active Inference Ontology processing and validation
12. **11_render.py** → `src/gnn/render/` - Code generation for PyMDP, RxInfer, ActiveInference.jl, JAX, PyTorch, NumPyro, DisCoPy, bnlearn, and Stan-supported render paths
13. **12_execute.py** → `src/gnn/execute/` - Execute rendered simulation scripts with result capture
14. **13_llm.py** → `src/gnn/llm/` - LLM-enhanced analysis, model interpretation, and AI assistance
15. **14_ml_integration.py** → `src/gnn/ml_integration/` - Machine learning integration and model training
16. **15_audio.py** → `src/gnn/audio/` - Audio generation (SAPF, Pedalboard, and other backends)
17. **16_analysis.py** → `src/gnn/analysis/` - Advanced analysis and statistical processing
18. **17_integration.py** → `src/gnn/integration/` - System integration and cross-module coordination
19. **18_security.py** → `src/gnn/security/` - Security validation and access control
20. **19_research.py** → `src/gnn/research/` - Research tools and experimental features
21. **20_website.py** → `src/gnn/website/` - Static HTML website generation from pipeline artifacts
22. **21_mcp.py** → `src/gnn/mcp/` - Model Context Protocol processing and tool registration
23. **22_gui.py** → `src/gnn/gui/` - Interactive GUI for constructing/editing GNN models
24. **23_report.py** → `src/gnn/report/` - Comprehensive analysis report generation
25. **24_intelligent_analysis.py** → `src/gnn/intelligent_analysis/` - AI-powered pipeline analysis and executive reports

### Execution Flow (conceptual)

1. Load configuration (`input/config.yaml` via `src/gnn/main.py`)
2. Discover steps (`pipeline/step_registry.py`) and resolve dependencies (`pipeline/dag.py`)
3. Execute routed steps per testing matrix, writing per-step outputs
4. Write `00_pipeline_summary/pipeline_execution_summary.json`

## Configuration Options

Real configuration lives in `input/config.yaml` (testing matrix, `pipeline.skip_steps`,
LLM settings) plus `PipelineConfig` defaults in `pipeline/config.py`
(`timeout: 3600`, `retries: 3`, `parallel: True`). The illustrative dicts that
previously appeared here are not literal schema — check `pipeline/config.py`
and `gnn.utils.arguments.arg_parsing` for the accepted keys.

## Error Handling

Step failures are handled by `main.py` per the safe-to-fail contract: steps
log, write artifacts, and return structured exit codes (0 / 1 / 2) rather than
raising. `pipeline/execution.py` `run_pipeline` returns a compact summary dict;
`execute_pipeline_step` returns a `StepExecutionResult` with success/duration/
errors. There are no `PipelineError` / `StepError` / `ConfigError` exception
classes in this module.

## Performance Optimization

- **Parallel Execution**: `execution_workers` controls script-level parallelism in Step 12; `PipelineConfig.parallel` covers orchestration
- **Incremental Processing**: matrix routing runs per-folder step lists from `input/config.yaml`

## Testing and Validation

```bash
uv run --extra dev python -m pytest tests/pipeline/ -q --tb=short
```

Key suites: `test_pipeline_integration.py`, `test_pipeline_functionality.py`,
`test_pipeline_performance.py`.

## Dependencies

### Required Dependencies
- **pathlib**: Path handling
- **json**: JSON configuration handling
- **logging**: Logging functionality
- **typing**: Type hints
- **time**: Time utilities

### Optional Dependencies
- **yaml**: YAML configuration parsing

## Performance

Per-step timeouts live in `pipeline/step_timeouts.py` (overridable via
`GNN_STEP_TIMEOUT_{N}` and `GNN_STEP_TIMEOUT_SCALE`). Run-to-run timing,
memory, and pass/fail counts are recorded in each run's
`00_pipeline_summary/pipeline_execution_summary.json` and per-step summaries —
consult those for measurements instead of the static ranges some older docs printed.

## Summary

The Pipeline module provides orchestration, configuration, step coordination,
and the canonical step registry for the 25-step GNN pipeline.

## License and Citation

This module is part of the GeneralizedNotationNotation project. See the main repository for license and citation information. 

## References

- Project overview: ../../../README.md
- Comprehensive docs: ../../../DOCS.md
- Architecture guide: ../../../ARCHITECTURE.md
- Pipeline details: ../../../docs/pipeline/README.md

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
