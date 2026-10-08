# Pipeline Module Specification

## Overview
Pipeline orchestration, configuration, and execution utilities.
## Components

### Configuration
- `config.py` - Pipeline configuration management (`PipelineConfig`, `get_pipeline_config`, `get_output_dir_for_script`)

### Execution & Ordering
- `execution.py` - `run_pipeline`, `execute_pipeline_step(s)`, `get_pipeline_status`
- `dag.py` - `resolve_execution_order`, `visualize_dag`
- `step_executor.py` - Opt-in consolidated in-process step execution (ADR 0001)

### Validation & Health
- `pipeline_runtime_validator.py` - E2E pipeline validation in the caller's workspace;
  safe-path module launch and a fresh child invocation ID admit only its canonical
  current-run summary, under the existing 300-second subprocess timeout.
- `health_check.py` - `run_enhanced_health_check`
- `verify_pipeline.py` - Pipeline verification

### Registry & Discovery
- `step_registry.py` - Canonical `STEPS` list (25 entries) + `STAGE_DEFINITIONS`
- `discovery.py` - Step discovery helpers

### v3.0.0 Long-Running Orchestration (safe-by-design; no live mutation)
- `durable_streams.py` - `StreamManifest` (file/array, content-checksummed) + `ExecutionTrace`
  integrity & deterministic replay (`validate_stream_manifest`, `trace_integrity`, `replay_trace`)
- `run_session.py` - resumable `RunSession` manifests: atomic `checkpoint`/`load_session`,
  `remaining_units`/`status_report`, path-safe `cancel_safe_cleanup`
- `container_plan.py` - `generate_container_plan` (hardened) + static `security_review` +
  `RollbackDescriptor` + deterministic `compute_plan_hash` (no container/cluster is executed)
- Acceptance: `scripts/run_v3_orchestration_acceptance.py --strict`; reference:
  `docs/pipeline/v3_orchestration.md`

## Key Exports
```python
from gnn.pipeline import execute_pipeline_step, execute_pipeline_steps
from gnn.pipeline.config import PipelineConfig, get_pipeline_config, get_output_dir_for_script
from gnn.pipeline.step_registry import STEPS, step_for_name, get_core_steps
```

## Step Naming Convention
Steps follow `N_name.py` pattern (0-24)


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
