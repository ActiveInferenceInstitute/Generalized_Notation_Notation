# Execute Processor Package — Agent Scaffolding

## Module Overview

**Purpose**: Step 12 execution processing for rendered implementations: discover executable scripts under the render output, enforce the render-summary contract, run each script through a structured subprocess envelope (security gate, sandbox, benchmark repeats), and persist slim aggregate summaries plus per-script receipts.

**Pipeline Step**: Step 12: Execution (`src/gnn/12_execute.py`), re-exported through the parent [../AGENTS.md](../AGENTS.md) surface (`execute/__init__.py:44`)

**Category**: Simulation / Execution

**Status**: ✅ Production Ready

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-28

## File Structure

```
processor/
├── __init__.py    # Package facade: module docstring, pre-split surface re-exports (data_extractors/detection/julia_env/metadata/security_gate/subprocess_envelope/types/logging_utils + framework availability), moved-def re-exports from the five leaf modules
├── envelope.py    # Per-script result envelope factories: _base_execution_envelope, _make_skipped_result, local-pool/distributed dispatch failure factories, facade-routed dependency probe (_is_python_framework_dependency_available)
├── workers.py     # _coerce_execution_workers/_coerce_dispatch_retries, _execute_script_worker, _run_scripts_with_local_workers (ProcessPoolExecutor resolved through the facade at call time)
├── summary.py     # _init_execution_summary, _update_framework_status, _classify_execute_outcome, _write_execution_summaries (slim aggregate + optional detail + regenerated report)
├── single.py      # execute_single_script: subprocess envelopes, per-framework environment, security gate, sandbox receipts, benchmark aggregation, structured results/logs
└── pipeline.py    # process_execute (Step 12 orchestration) and execute_simulation_from_gnn
```

This package has no MCP surface; the Step 12 MCP tools live in the parent package (`src/gnn/execute/mcp.py`).

## Module Map

| Module | Owns |
| --- | --- |
| `__init__.py` | Facade: pre-split surface re-exports (`__init__.py:20-77`), moved-def re-exports (`__init__.py:82-117`), framework-availability aliases (`__init__.py:118-123`) |
| `envelope.py` | `_is_python_framework_dependency_available` (`envelope.py:15`), `_base_execution_envelope` (`envelope.py:31`), `_model_framework_from_path` (`envelope.py:66`), `_make_skipped_result` (`envelope.py:79`), `_make_local_worker_pool_failure_result` (`envelope.py:116`), `_make_distributed_dispatch_failure_result` (`envelope.py:138`) |
| `workers.py` | `_coerce_execution_workers` (`workers.py:14`), `_coerce_dispatch_retries` (`workers.py:23`), `_execute_script_worker` (`workers.py:32`), `_run_scripts_with_local_workers` (`workers.py:51`) |
| `summary.py` | `_init_execution_summary` (`summary.py:18`), `_update_framework_status` (`summary.py:47`), `_classify_execute_outcome` (`summary.py:95`), `_write_execution_summaries` (`summary.py:164`) |
| `single.py` | `_aggregate_benchmark_samples` (`single.py:53`), `_new_execution_result` (`single.py:70`), `_build_execution_environment` (`single.py:83`), `_framework_for_data_helpers` (`single.py:131`), `_GNN_ALLOW_MISSING_DEPS` (`single.py:136`), `_gnn_allow_missing_deps` (`single.py:139`), `_sandbox_mode` (`single.py:148`), `_sandbox_command_prefix` (`single.py:156`), `_backend_version_from_runner_metadata` (`single.py:180`), `execute_single_script` (`single.py:202`) |
| `pipeline.py` | `process_execute` (`pipeline.py:43`), `execute_simulation_from_gnn` (`pipeline.py:440`) |

## Public Surface (facade contract)

Import rule: callers import only through the facade — `from gnn.execute.processor import ...` or
`from gnn.execute import processor`. The leaf modules (`envelope`, `workers`, `summary`, `single`,
`pipeline`) and their `_*` members are private to the package; the facade is the only stable name
surface.

### Layer 1 — moved definitions re-exported from the split modules

| Source module | Names | Cite |
| --- | --- | --- |
| `processor.envelope` | `_base_execution_envelope`, `_is_python_framework_dependency_available`, `_make_distributed_dispatch_failure_result`, `_make_local_worker_pool_failure_result`, `_make_skipped_result`, `_model_framework_from_path` | `__init__.py:82-89` |
| `processor.pipeline` | `execute_simulation_from_gnn`, `process_execute` | `__init__.py:90-93` |
| `processor.single` | `_GNN_ALLOW_MISSING_DEPS`, `_aggregate_benchmark_samples`, `_backend_version_from_runner_metadata`, `_build_execution_environment`, `_framework_for_data_helpers`, `_gnn_allow_missing_deps`, `_new_execution_result`, `_sandbox_command_prefix`, `_sandbox_mode`, `execute_single_script` | `__init__.py:94-105` |
| `processor.summary` | `_classify_execute_outcome`, `_init_execution_summary`, `_update_framework_status`, `_write_execution_summaries` | `__init__.py:106-111` |
| `processor.workers` | `_coerce_dispatch_retries`, `_coerce_execution_workers`, `_execute_script_worker`, `_run_scripts_with_local_workers` | `__init__.py:112-117` |

### Layer 2 — pre-split surface preservation (execute siblings + logging)

The facade re-exports the full pre-split import surface verbatim, so every pre-split
`from gnn.execute.processor import <name>` site keeps working unchanged:

| Source module | Names | Cite |
| --- | --- | --- |
| `gnn.execute.data_extractors` | `collect_execution_outputs`, `_extract_simulation_data`, `_extract_simulation_data_from_files` | `__init__.py:20-28` |
| `gnn.execute.detection` | `_build_script_execution_context`, `_detect_accelerator_type`, `_resolve_render_output_dir`, `determine_script_framework`, `find_executable_scripts`, `parse_frameworks_parameter` | `__init__.py:29-36` |
| `gnn.execute.julia_env` | `GKSWSTYPE_HEADLESS`, `GKSWSTYPE_VAR`, `_build_script_execution_command`, `_julia_project_for_framework`, `check_julia_dependencies` | `__init__.py:37-43` |
| `gnn.execute.metadata` | `_STATUS_SEVERITY`, `_atomic_execution_json`, `_execution_detail_key`, `_execution_input_identity`, `_execution_script_identity`, `_load_render_summary_contract`, `_load_rxinfer_execution_metadata_from_script`, `_load_rxinfer_execution_metadata_sidecar`, `_merge_prior_execution_summary`, `_sha256_file`, `_slim_execution_detail`, `_summarize_collected_outputs`, `generate_execution_report` | `__init__.py:44-58` |
| `gnn.execute.security_gate` | `check_script_allowed` | `__init__.py:59` |
| `gnn.execute.subprocess_envelope` | `INTERNAL_ERROR`, `NEVER_STARTED`, `UNKNOWN_STATE`, `run_subprocess_envelope` | `__init__.py:60-65` |
| `gnn.execute.types` | `_EXECUTABLE_SUFFIXES`, `ExecutionFrameworkName`, `ExecutionOutcome`, `ScriptExecutionContext` | `__init__.py:66-71` |
| `gnn.utils.logging_utils` | `log_step_error`, `log_step_start`, `log_step_success`, `log_step_warning` | `__init__.py:72-77` |
| `gnn.utils.runtime_safety.framework_availability` | `FRAMEWORK_IMPORT_CHECK` (as `_FRAMEWORK_IMPORT_CHECK`), `is_framework_available` (as `_is_framework_available_by_name`) | `__init__.py:118-123` |

The facade also binds `ProcessPoolExecutor` at module top (`__init__.py:15`); the worker pool
resolves that attribute through the facade at call time (see the seam section below).

## Lazy-Facade Patch Seam

Leaf modules resolve selected facade attributes **inside function bodies** via
`from gnn.execute import processor as _processor_facade`, so tests that patch the facade
attribute are observed by the leaf code:

- `envelope.py:24-28` — `_is_python_framework_dependency_available` resolves
  `_processor_facade._is_framework_available_by_name` at call time.
- `pipeline.py:149-155` — `process_execute` resolves
  `_processor_facade._load_render_summary_contract` at call time.
- `single.py:228-236` — `execute_single_script` resolves
  `_processor_facade._is_python_framework_dependency_available` for the pre-flight skip check.
- `single.py:374-376` — the same call-time lookup supplies
  `_processor_facade._build_execution_environment` for the subprocess environment.
- `workers.py:87-90` — `_run_scripts_with_local_workers` resolves
  `_processor_facade.ProcessPoolExecutor` at pool-construction time.

Why call-time resolution matters: a module-top import would bind the original object into the
leaf module's namespace before any test runs, so a later `monkeypatch.setattr(processor, ...)`
would replace the facade attribute but not the frozen leaf binding. Resolving inside the function
body reads the attribute from the facade on every call, so patches apply. The consuming proof is
`tests/execute/test_execute_script_safely.py:194`
(`monkeypatch.setattr(processor, "ProcessPoolExecutor", BrokenPool)`), and the per-script failure
envelopes that test asserts can only be produced if `_run_scripts_with_local_workers` sees the
patched class.

Rule for new leaf code: any facade attribute a leaf needs at runtime must be resolved through the
`_processor_facade` lookup inside the function body, never bound at module top. Module-top imports
remain correct only for non-facade dependencies that are not patch seams.

Acyclicity invariant: `src/gnn/execute/types.py:6-7` states the contract — a leaf module must not
import from `execute.processor` or any execute sibling, which keeps the facade import graph
acyclic. The module-top sibling imports that are legitimate (pure intra-package wiring, no patch
seams): `pipeline.py:18-29` (envelope/single/summary/workers), `single.py:34-38` (metadata +
envelope), `workers.py:8-11` (envelope + single). `envelope.py` imports no sibling at module top —
only `gnn.execute.metadata` and the framework-availability module (`envelope.py:9-12`).

## Provenance

This package was extracted in-place from the former flat module `src/gnn/execute/processor.py`
into the six-file package above; the import surface is unchanged (Layer 2 above preserves it
verbatim, Layer 1 re-homes the moved definitions on the facade). Import sites now target the
package facade:

- `src/gnn/12_execute.py:36` imports `process_execute` via the parent facade
  (`execute/__init__.py:44`).
- `src/gnn/analysis/complexity/benchmark.py:289` imports `process_execute`.
- The test files in the table below import `_`-prefixed moved helpers through the facade.

## Testing & Verification

Test files that import this package (verified by grep at repo tip):

| Test | Import cite | Imports |
| --- | --- | --- |
| `tests/execute/test_execute_benchmark_samples.py` | `:9` | `_aggregate_benchmark_samples` |
| `tests/execute/test_execute_envelope_factories.py` | `:17-20` | `_base_execution_envelope`, `_make_distributed_dispatch_failure_result`, `_make_local_worker_pool_failure_result` |
| `tests/execute/test_execute_fallback_receipts.py` | `:32` | `_make_skipped_result` |
| `tests/execute/test_execute_outcome_classification.py` | `:19` | `_classify_execute_outcome` |
| `tests/execute/test_execute_overall.py` | `:36,38-41` | facade module, `collect_execution_outputs`, `determine_script_framework`, `execute_single_script` |
| `tests/execute/test_execute_script_safely.py` | `:111,194,302-304` | `process_execute`, facade `ProcessPoolExecutor` patch, `ScriptExecutionContext`, `_build_script_execution_command` |
| `tests/execute/test_execute_slim_detail.py` | `:9` | `_slim_execution_detail`, `_summarize_collected_outputs` |
| `tests/execute/test_execute_stan.py` | `:50` | `_load_render_summary_contract` |
| `tests/execute/test_receipt_reliability.py` | `:159` | `_write_execution_summaries` |
| `tests/execute/test_resource_measurement.py` | `:30` | `_write_execution_summaries` |
| `tests/execute/test_execute_bnlearn.py` | `:21` | facade module |
| `tests/execute/test_julia_env_headless.py` | `:21` | facade module |
| `tests/execute/test_ngclearn_runner.py` | `:26` | facade module |
| `tests/execute/test_discrete_models_pymdp.py` | `:312` | `process_execute` |
| `tests/execute/test_pymdp_contracts.py` | `:197` | `process_execute` |
| `tests/render/test_rxinfer_multiagent_contract.py` | `:18-21` | `_load_rxinfer_execution_metadata_from_script`, `execute_single_script` |

Run the facade-contract and seam coverage with:

```bash
uv run --extra dev python -m pytest \
    tests/execute/test_execute_benchmark_samples.py \
    tests/execute/test_execute_envelope_factories.py \
    tests/execute/test_execute_fallback_receipts.py \
    tests/execute/test_execute_outcome_classification.py \
    tests/execute/test_execute_overall.py \
    tests/execute/test_execute_script_safely.py \
    tests/execute/test_execute_slim_detail.py \
    tests/execute/test_execute_stan.py \
    tests/execute/test_receipt_reliability.py \
    tests/execute/test_resource_measurement.py \
    tests/render/test_rxinfer_multiagent_contract.py \
    -q
```

For parent-level coverage of the whole Step 12 module, run the execute suite:
`uv run --extra dev python -m pytest tests/execute -q`.

---

**Last Updated**: 2026-09-28
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern
