# Pipeline Module - Agent Scaffolding

## Module Overview

**Purpose**: Pipeline orchestration, configuration management, and execution coordination for the GNN processing system

**Pipeline Step**: Infrastructure module (not a numbered step)

**Category**: Pipeline Infrastructure / Orchestration

**Status**: ✅ Production Ready

**v3.0.0 orchestration (safe-by-design, no live mutation)**: `durable_streams.py` (stream
manifests + replayable execution traces), `run_session.py` (resumable run sessions with atomic
checkpoint/resume + path-safe cleanup), and `container_plan.py` (auditable container plans + static
security review + rollback). These generate/validate **data only** — no container or cluster is
executed. Acceptance: `scripts/run_v3_orchestration_acceptance.py --strict`. `run_session_wiring.py`
additionally wires run sessions + durable streams into the `gnn.main` composition (artifacts:
`00_pipeline_summary/run_session.json` + `v3_run_manifest/`). `run_manifest.py` emission also records
binary artifacts (`.png`/`.gif`/`.npy`/`.csv`) as additive index keys (`schema_version` `3.2`); consumers
must read new keys additively — the JSON inventory keys keep their prior semantics, and
`verify_run_manifests` accepts older 3.1 indexes. `session_acceptance.py` provides
`verify_session_artifacts(session, output_dir)`, which joins each unit's recorded `artifact_hashes`
inventory against the on-disk artifacts (units without a recorded inventory are outside the join).
Reference:
[`docs/pipeline/v3_orchestration.md`](../../../docs/pipeline/v3_orchestration.md).

**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-10

---

## Core Functionality

### Primary Responsibilities

1. Pipeline execution orchestration and step coordination
2. Configuration management and validation
3. Step discovery and dependency management
4. Pipeline health monitoring and diagnostics
5. Execution planning and resource estimation
6. Pipeline validation and verification

### Key Capabilities

- Multi-step pipeline orchestration
- Dynamic step discovery and configuration
- Pipeline health monitoring and alerting
- Resource estimation and allocation
- Execution plan generation
- Performance tracking and optimization
- Error recovery and retry mechanisms

### Autonomous Proposal Surface

`pipeline/autonomous.py` implements the bounded, proposal-only autonomy
surface (`AUTONOMY_POLICY`, mode `proposal_only`). `collect_observation_streams`
describes file-backed observation inputs, `build_container_plan` emits an
audited, never-executed container plan for `src/gnn/main.py --autonomous`
(static security review attached), and `run_autonomous_proposal_loop` writes
candidate patch templates, evaluation reports, review gates, and an audit log
under `output/autonomous/` without mutating source files. Patch artifacts are
proposal templates only (`patch_artifact_kind: proposal_only`);
review gates are stateless single-pass checks computed from static policy and
do not track approval state across runs. Application always requires recorded
human approval. The validator commands referenced by the review workflow live
in `VALIDATOR_COMMANDS` (capability contracts, docs audit, and the
`tests/pipeline/test_autonomous_contract.py` pytest suite). Entry point:
`uv run --extra dev python src/gnn/main.py --autonomous`.

### Shared Building Blocks

Cross-cutting behaviour is implemented once and delegated to; these are the
shared owners new code must reuse instead of re-implementing.

- **Atomic writes** — `pipeline/_io.py` owns the mkstemp +
  `os.replace` recipe as `atomic_write_text` / `atomic_write_bytes` (a unique
  temporary file in the destination directory, then replace, so the rename
  cannot cross a filesystem boundary). Stream manifests
  and traces (`durable_streams.py`), run-session checkpoints
  (`run_session.py`), the run-manifest index write (`run_manifest.py`), the
  history index (`hasher.index_run`), run-context snapshots
  (`run_context.py`), and finalization (`finalization.py`) all delegate to it,
  so the history index write is atomic too.
- **Single version source** — `pipeline/_version.py` re-exports
  `gnn.__version__` and exists so leaf modules can read the version without an
  import cycle. `pipeline/__init__.py`, `execution.get_pipeline_info()`, and
  `execution.create_pipeline_config()` report that value rather than a
  hardcoded string.
- **Shared family selection** —
  `model_family_acceptance.select_model_families()` is the single filter rule
  for accepting model families; `semantic_fidelity` and `session_acceptance`
  delegate to it instead of re-implementing the match.
- **Cycle detection** — `dag.find_circular_dependencies()` is the shared
  Kahn-peel detector; the `validate_pipeline_dependencies` MCP tool reports its
  result as the tool's `circular_dependencies` field.
- **Registry-derived counts** — `dag.resolve_execution_order()` defaults
  `total_steps` to `len(step_registry.STEPS)`; `get_pipeline_status()` and
  `get_pipeline_info()` count steps from the registry-derived
  `_STEP_SCRIPT_BY_NUM`, so no module hardcodes 25.
- **Path constants** — `config.DEFAULT_TARGET_DIR` (`"input/gnn_files"`) and
  `config.DEFAULT_OUTPUT_DIR` (`"output"`) are the canonical defaults;
  `execution` and the package defaults read them instead of repeating the
  literals.
- **Status set** — `execution._SUCCESS_STATUSES` (frozenset of `SUCCESS`,
  `SUCCESS_WITH_WARNINGS`, `SKIPPED`) is the one success predicate.
- **Step-list normalization** — `execution.resolve_step_numbers()` is the one
  parser for step identifiers, including the comma-separated `--only-steps`
  form (`"3,5"`).
- **Preflight** — `preflight.validate_config()` validates
  `pipeline.skip_steps` with the canonical `read_skip_steps` parser and
  reports an error with a fix hint, so bad values fail preflight instead of
  mid-run.
- **Package surface** — `pipeline/__init__.py` declares a typed, complete
  `__all__` that includes `resolve_step_numbers`, `get_module_info`,
  `validate_pipeline_step`, `discover_pipeline_steps`, `DEFAULT_TARGET_DIR`,
  and `DEFAULT_OUTPUT_DIR`.
- **Step status type** — `pipeline.schemas.StepStatus` is the status literal;
  the former `PipelineContext` carrier is gone and has no replacement.

---

## API Reference

### Configuration Functions

#### `get_pipeline_config() -> Dict[str, Any]`

**Description**: Get the complete pipeline configuration including step metadata and settings

**Returns**: `Dict[str, Any]` - Pipeline configuration dictionary with:

- `steps`: List of step names
- `output_base_dir`: Base output directory
- `log_level`: Logging level
- `step_configs`: Step-specific configurations

**Example**:

```python
from gnn.pipeline import get_pipeline_config

config = get_pipeline_config()
print(f"Steps: {config['steps']}")
```

#### `set_pipeline_config(config: PipelineConfig) -> None`

**Description**: Save a new pipeline configuration (see `pipeline/README.md`).

**Parameters**:

- `config` (PipelineConfig): The pipeline configuration instance to save

**Returns**: `None`

#### `get_output_dir_for_script(script_name: str, base_output_dir: Optional[Path] = None) -> Path`

**Description**: Get standardized output directory for a pipeline script

**Parameters**:

- `script_name` (str): Name of the pipeline script (e.g., "3_gnn.py")
- `base_output_dir` (Optional[Path]): Base output directory (default: Path("output"))

**Returns**: `Path` - Output directory path (e.g., "output/3_gnn_output/")

**Example**:

```python
from gnn.pipeline import get_output_dir_for_script
from pathlib import Path

output_dir = get_output_dir_for_script("3_gnn.py", Path("output"))
```

### Step Validation and DAG Functions

#### `validate_pipeline_step(step_name: str) -> bool`

Validate that a step name is known to the pipeline (present in
`STEP_METADATA`).

**Location**: `src/gnn/pipeline/__init__.py`

#### `discover_pipeline_steps() -> list[str]`

Return the ordered list of registered step script stems (derived from
`step_registry.STEP_METADATA_DICT`, the single source of truth).

**Location**: `src/gnn/pipeline/__init__.py`

#### `resolve_execution_order(step_dependencies, total_steps=None, skip_steps=None, raise_on_circular=False) -> List[List[int]]`

**Location**: `src/gnn/pipeline/dag.py`

Topologically sort **step numbers** into parallel execution tiers (Kahn's
algorithm with tier grouping). `step_dependencies` maps each step number to
the step numbers it depends on (`gnn.utils.pipeline_orchestration.pipeline_step_dependencies.PIPELINE_STEP_DEPENDENCIES`
is the canonical dependency table). `total_steps` defaults to the registry
step count. Circular dependencies are appended as a final tier unless
`raise_on_circular=True`, which raises `ValueError`.

#### `find_circular_dependencies(step_dependencies, nodes=None) -> Set[int]`

**Location**: `src/gnn/pipeline/dag.py` (added 2026-09-04)

Return the set of step numbers bound up in dependency cycles (cycle members
plus any step that transitively depends on one). Powers the
`circular_dependencies` field of the `validate_pipeline_dependencies` MCP
tool.

#### `visualize_dag(tiers, step_names=None) -> str`

Render DAG tiers (output of `resolve_execution_order`) as a human-readable
multi-line string for logging.

> **Note**: `validate_step_prerequisites` and `validate_pipeline_step_sequence` live in `gnn.utils.pipeline_orchestration.pipeline_validator` (prerequisite/output checking at execution time); `generate_execution_plan` from earlier documentation versions does not exist. Dependency ordering is in `pipeline/dag.py`, and E2E runtime testing is `pipeline/pipeline_runtime_validator.py`.

### Execution Planning

**Execution planning** is handled inline by `main.py`, not via a separate `pipeline_planner.py` module. `PipelineContext` removed 2026-09-15 (N-4: delete) — production-dead; `StepStatus` re-homed to `pipeline/schemas.py`.

**Example**:

```python
from gnn.pipeline.config import get_output_dir_for_script
from gnn.pipeline.dag import resolve_execution_order

output_dir = get_output_dir_for_script("3_gnn.py", Path("output"))
```

### Execution Functions

#### `run_pipeline(pipeline_data=None, *, target_dir=None, output_dir=None, steps="all", verbose=False) -> dict`

Execute the pipeline through ``main.py`` and return a compact summary dict
with `success`, `steps_executed`, `errors`, `warnings`, `target_dir`,
`output_dir`, `exit_code`, `duration`, and (when the run wrote a summary)
`summary_file` / `overall_status`. `steps` accepts `"all"`, a single step
name/number, a comma-separated list, or an iterable of those — normalized by
`resolve_step_numbers`.

#### `resolve_step_numbers(steps, pipeline_data=None) -> list[int]`

**Added 2026-09-04.** Normalize step identifiers ("11", "11_render",
"11_render.py", "3,5", iterables of those, or `"all"`/`None`) to a sorted,
de-duplicated list of registered step numbers. Unknown tokens are dropped.

#### `execute_pipeline_step(step_name: str, step_config: dict, pipeline_data: dict) -> StepExecutionResult`

Execute a single numbered pipeline step via ``main.execute_pipeline_step``.

**Returns**: `StepExecutionResult` — dataclass with `step_name`, `success`,
`duration`, `output`, `error`, `warnings`, `remediation`.

#### `get_pipeline_status() -> dict`

Static readiness probe: `{"status": "ready", "timestamp", "steps_available",
"steps_completed"}`. Step availability is derived from the registry; live
progress is reported by the per-run `pipeline_execution_summary.json` instead.

### Health Check Functions

#### `run_enhanced_health_check() -> Dict[str, Any]`

**Description**: Run the pipeline health check (components, dependencies, step discovery, config sanity). Returns a status dict instead of raising.

**Returns**: `Dict[str, Any]` - Health check results with:

- `overall_status` (str): "healthy", "degraded", or "unhealthy"
- `component_status` (Dict[str, str]): Status of each component
- `issues` (List[str]): Detected issues
- `recommendations` (List[str]): Recommended actions

---

## Dependencies

### Required Dependencies

- `pathlib` - Path manipulation
- `typing` - Type hints
- `logging` - Logging functionality

### Internal Dependencies

- `gnn.utils.arguments` - Argument parsing utilities
- `gnn.utils.logging_utils` - Structured (JSON-L + text) logging helpers
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline template utilities

---

## Configuration

### Environment Variables

`GNN_RUN_ID` identifies one top-level invocation. `src/gnn/main.py` preserves an
incoming API value or creates a UUID, passes it explicitly to step subprocesses,
and restores the caller environment in `finally`. Canonical summaries retain
`run_id` on success and failure; `run_hash` remains the stable input/config hash.
Programmatic top-level calls serialize their environment scopes; parallel step
execution uses explicit child environments. Other pipeline behavior is configured through
`input/config.yaml` (loaded by `src/gnn/main.py` via `gnn.utils.arguments.arg_parsing`) and
CLI flags.

`file_workers` (default 1; `--file-workers`, Steps 3/5) bounds spawned CPU
file jobs. Parents publish in selection order, and multiworker steps retain
the supervised subprocess boundary even with consolidated execution requested.
`step_workers` (`--step-workers`, default automatic) bounds topological tiers
under `--parallel`, retaining memory throttling and the eight-step ceiling.
Both admit exact integers from 1 to 256; omitted `step_workers` is automatic.
`execution_resources` and `parallel_tiers` in the summary retain configured
limits and actual admitted tier widths. These do not claim CPU affinity,
quotas or that every available worker executed a file.

### Configuration Files

- `pipeline_config.yaml` - Optional project-local pipeline configuration
  (default lookup path in `pipeline/config.py`)
- `input/config.yaml` - The pipeline's primary configuration file

### Default Settings

Configuration defaults (`timeout: 3600`, `retries: 3`) are set in
`pipeline/config.py` (`PipelineConfig`); step-level defaults live with each
module.

---

## Usage Examples

### Basic Pipeline Configuration

```python
from gnn.pipeline.config import get_pipeline_config, get_output_dir_for_script

# Get current configuration
config = get_pipeline_config()
print(f"Output directory: {config['output_dir']}")

# Get output directory for specific step
output_dir = get_output_dir_for_script("3_gnn.py", Path("output"))
print(f"GNN output directory: {output_dir}")
```

### Step Validation

```python
from gnn.pipeline import validate_pipeline_step, discover_pipeline_steps

# Verify a step is known (keys are script stems)
assert validate_pipeline_step("3_gnn")

# List all steps
for step in discover_pipeline_steps():
    print(step)
```

### DAG-Based Ordering

```python
from gnn.pipeline.dag import resolve_execution_order, find_circular_dependencies
from gnn.utils.pipeline_orchestration.pipeline_step_dependencies import PIPELINE_STEP_DEPENDENCIES

# Resolve tiers over the canonical dependency table
tiers = resolve_execution_order(dict(PIPELINE_STEP_DEPENDENCIES))
print(f"Resolved tiers: {tiers}")

# Detect cycle-bound steps in an arbitrary graph
assert find_circular_dependencies({0: [1], 1: []}) == set()
```

---

## Output Specification

### Output Products

- `pipeline_execution_summary.json` - Execution summary (written by `main.py` under `00_pipeline_summary/`)
- `pipeline_health_report_<timestamp>.json` - Health monitoring report (written by `gnn.utils.pipeline_orchestration.pipeline_monitor`)

### Output Directory Structure

```text
output/
├── pipeline_config.yaml
├── pipeline_execution_summary.json
├── pipeline_health_report.json
└── step_execution_reports/
    ├── 0_template_execution.json
    ├── 1_setup_execution.json
    └── ...
```

---

## Performance Characteristics

### Latest Execution

- **Duration**: Variable (depends on pipeline length)
- **Memory**: ~10-50MB for orchestration
- **Status**: ✅ Production Ready

### Expected Performance

- **Orchestration Overhead**: < 5% of total pipeline time
- **Configuration Loading**: < 100ms
- **Step Discovery**: < 500ms
- **Health Monitoring**: < 10ms per check

---

## Error Handling

### Pipeline Errors

1. **Configuration Errors**: Invalid pipeline configuration
2. **Dependency Errors**: Missing step dependencies
3. **Resource Errors**: Insufficient resources for execution
4. **Timeout Errors**: Step execution timeout
5. **Validation Errors**: Invalid step sequence or parameters

### Recovery Strategies

- **Auto-retry**: Automatic retry for transient failures
- **Graceful degradation**: Continue with available steps
- **Resource reallocation**: Adjust resource allocation
- **Configuration repair**: Attempt to fix configuration issues

---

## Integration Points

### Orchestrated By

- **Script**: `main.py` (Main pipeline orchestrator)
- **Function**: Pipeline execution coordination

### Imports From

- `gnn.utils.arguments` - Argument parsing
- `gnn.utils.logging_utils` - Structured logging
- `gnn.utils.pipeline_orchestration.pipeline_template` - Template utilities

### Imported By

- All pipeline scripts (0_template.py through 24_intelligent_analysis.py)
- `tests.test_pipeline_*` - Pipeline tests
- `mcp.pipeline_tools` - MCP pipeline tools

### Data Flow

```text
Configuration → Step Discovery → Dependency Validation → Execution Planning → Step Execution → Health Monitoring
```

---

## Testing

### Test Files

The pipeline test suite lives in `tests/pipeline/` (45+ files). Key
examples:

- `tests/pipeline/test_pipeline_orchestration.py` - Orchestration, config, DAG tiers
- `tests/pipeline/test_pipeline_integration.py` - Integration tests
- `tests/pipeline/test_pipeline_refactor_contracts.py` - Shared-building-block
  contracts (`pipeline._io`, `dag.find_circular_dependencies`,
  `execution.resolve_step_numbers`, `select_model_families`, preflight
  `skip_steps` gate) added 2026-09-04
- `tests/pipeline/test_pipeline_performance.py` - Performance tests

### Test Coverage

Measure on demand — no static number is kept in this file:

```bash
uv run --extra dev python -m pytest tests/pipeline/ --cov=src/gnn/pipeline --cov-report=term-missing
```

### Key Test Scenarios

1. Pipeline configuration validation
2. Step dependency resolution
3. Execution plan generation
4. Health monitoring functionality
5. Error recovery mechanisms

---

## MCP Integration

### Tools Registered

Registered by `register_tools()` in `pipeline/mcp.py`:

- `get_pipeline_steps` - Step metadata and dependencies
- `get_pipeline_status` - Current execution status, recent logs, statistics; includes `memory_receipts` (aggregate + per-step `memory_usage_mb`/`peak_memory_mb`/`memory_delta_mb`) when a summary exists
- `list_step_artifacts` - Registry-aware per-step artifact browser data (`exists`, `file_count`, `total_size_bytes`, up to 20 `files` with a `truncated` flag); optional `step_number`
- `validate_pipeline_dependencies` - Missing or circular dependency check
- `get_pipeline_config_info` - Detailed configuration information
- `get_v3_orchestration_capabilities`, `run_v3_container_security_review`, `run_v3_orchestration_self_check` - v3.0.0 safe-by-design orchestration tools (data only, no live mutation)
---

## Troubleshooting

### Common Issues

#### Issue 1: Step discovery fails

**Symptom**: Pipeline can't find or load pipeline steps  
**Cause**: Step scripts missing or naming convention mismatch  
**Solution**:

- Verify all step scripts exist in `src/` directory
- Check script naming follows pattern `N_module.py`
- Ensure scripts are executable and have correct imports
- Review step discovery logs

#### Issue 2: Configuration validation errors

**Symptom**: Pipeline fails with configuration errors  
**Cause**: Invalid configuration values or missing required settings  
**Solution**:

- Verify configuration file format is valid
- Check all required configuration keys are present
- Use default configuration if issues persist

---


## Version History

### Current Version: [pyproject.toml](../../../pyproject.toml) (canonical)

**Features**:

- Pipeline orchestration
- Configuration management
- Step discovery and dependency management
- Health monitoring

**Known Issues**:

- None currently

### Roadmap

- Candidate: DAG-based parallel execution for independent step clusters
- Candidate: per-step streaming metrics published via MCP

---

## References

### Related Documentation

- [Pipeline Overview](../../../README.md)
- [Utils Module](../utils/AGENTS.md)

### External Resources

- [v3 Orchestration](../../../docs/pipeline/v3_orchestration.md)
- [Stage Hardening Review](../../../docs/pipeline/pipeline_stage_hardening_review.md)


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

## Step Registry — `step_registry.py`

The canonical pipeline step registry lives in [`step_registry.py`](step_registry.py). Every other step list in the codebase derives from this single source of truth.

- **Key types:** `StepInfo` (frozen dataclass), `STEPS` (ordered list of 25 entries).
- **Lookup:** `step_for_name("11_render")`, `step_for_stem("11_render")`.
- **Filtering:** `get_core_steps()` → 24 steps, `get_llm_steps()` → 1 step.
- **Adding a new step:** Add one `StepInfo(...)` to `STEPS` — all downstream consumers update automatically.

## Consolidated In-Process Execution — `step_executor.py`

[`step_executor.py`](step_executor.py) implements the opt-in consolidated
execution selected by `main.py --consolidated-steps`: steps whitelisted in
`step_registry.CONSOLIDATED_IN_PROCESS_STEMS` run inside the main process and
return a receipt identical in shape to the subprocess mode. Limits and the
thread-pool caveat are recorded in ADR 0001
([docs/decisions/0001-consolidated-pipeline-execution.md](../../../docs/decisions/0001-consolidated-pipeline-execution.md)):
testing-matrix folder dispatch stays on the subprocess path, and the parallel
tier runs consolidated steps on threads that share one process — steps 7/8
render with matplotlib (Agg), thread-safe in-process, but concurrent
consolidated runs of step 8 should expect shared matplotlib state (no
behavioral change attempted for matplotlib).

## Adding a New Pipeline Step

The old copy-and-customize step-template module is deleted
(COMP-009); new steps follow the numbered-thin-orchestrator contract that
`step_registry.py` and `step_executor.py` enforce:

1. **Register the step**: add one `StepInfo(...)` entry to `STEPS` in
   [`step_registry.py`](step_registry.py) — script stem, number, module
   function name, tags, and stage. Every downstream consumer (DAG ordering,
   discovery, output-dir mapping, the consolidated executor) derives from
   the registry; never maintain a second step list.
2. **Implement the thin orchestrator**: create `src/gnn/<N>_<name>.py`
   exposing the registered `module_function` — resolved by
   `step_executor.resolve_step_function` via `import gnn.<stem>` — with the
   standard argument surface (`--target-dir`, `--output-dir`, `--verbose`,
   `--recursive` plus step-specific args from `StepConfiguration`) and
   per-step output directories from
   `gnn.pipeline.config.get_output_dir_for_script`.
3. **In-process eligibility**: only steps proven equivalent run under
   `--consolidated-steps`; extend `CONSOLIDATED_IN_PROCESS_STEMS` only
   after an in-process-safety audit (no subprocess spawns, no
   cwd-relative writes). Limits and the thread-pool caveat are recorded in
   ADR 0001
   ([docs/decisions/0001-consolidated-pipeline-execution.md](../../../docs/decisions/0001-consolidated-pipeline-execution.md)).
4. **Shared utilities**: logging goes through the single public entry
   `gnn.utils.logging_utils`; argument parsing through
   `gnn.utils.arguments` (`ArgumentParser`, `StepConfiguration`); the
   output-directory helper is `gnn.pipeline.get_output_dir_for_script`.
