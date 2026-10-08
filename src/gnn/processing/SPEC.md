# Processing Module Specification

## Overview
The processing concern for GNN files: corpus discovery, a lightweight parse/validate/report surface, the five-phase orchestration engine, and the step-3 multi-format serialization body.

## Components
### Lightweight Surface
- `processor.py` - `discover_gnn_files`, `parse_gnn_file`, `check_gnn_file_structure`, `process_gnn_directory` / `process_gnn_directory_lightweight`, `generate_gnn_report`, `get_module_info`

### Orchestration
- `core_processor.py` - `GNNProcessor`: five-phase engine (`ProcessingPhase`: DISCOVERY → VALIDATION → ROUND_TRIP → CROSS_FORMAT → REPORTING) over `ProcessingContext`; plus module-level `process_gnn_directory` / `process_gnn_directory_lightweight` recovery wrappers (distinct semantics from the canonical `processor.py` functions — see README "Core-Processor Wrappers")

### Multi-Format
- `multi_format_processor.py` - `process_gnn_multi_format(target_dir, output_dir, logger, ...)`: step-3 discovery, parsing, and multi-format serialization orchestrator (`full` / `minimal` serialize presets via `_formats_for_serialize_preset`)

### Discovery
- `discovery.py` - `is_model_source_path` corpus filter, `FileDiscoveryStrategy` content-aware discovery, `DiscoveryResult`, `NON_MODEL_MARKDOWN_FILENAMES`, `NON_MODEL_MARKDOWN_SUFFIXES`
- `__init__.py` - Curated public surface (`__all__`, sixteen names)

## Invariants
- `GNNProcessor.process` preserves its Boolean public contract: report exceptions, returned write errors, or absent/empty required JSON, Markdown, or HTML reports return `False`. Failure context is recorded in `processing_results["report_error"]` and the reporting phase; completed partial report artifacts and returned metadata remain available.
- Numbered step scripts stay thin: `src/gnn/3_gnn.py` wraps `process_gnn_multi_format` with `create_standardized_pipeline_script`; the step body lives here.
- The root package lazily re-exports the lightweight surface and `process_gnn_multi_format` through `_EXPORT_MAP` in `src/gnn/__init__.py`; the orchestration engine (`GNNProcessor`) and discovery predicates must be imported from `gnn.processing` directly.
- `gnn.manuscript.variables` mirrors the non-model markdown rules rather than importing them; `test_producer_model_census_matches_pipeline_discovery` pins the two definitions equal.
- Discovery filtering is filename-based: `is_model_source_path` excludes the known non-model markdown names and the `.example.md` / `.template.md` suffixes.

## Key Exports
```python
from gnn.processing import (
    DiscoveryResult,
    FileDiscoveryStrategy,
    GNNProcessor,
    NON_MODEL_MARKDOWN_FILENAMES,
    NON_MODEL_MARKDOWN_SUFFIXES,
    ProcessingContext,
    ProcessingPhase,
    check_gnn_file_structure,
    discover_gnn_files,
    generate_gnn_report,
    get_module_info,
    is_model_source_path,
    parse_gnn_file,
    process_gnn_directory,
    process_gnn_directory_lightweight,
    process_gnn_multi_format,
)
```

## Receipts
```bash
uv run --extra dev python -m pytest tests/gnn/test_gnn_processor_validation.py \
  tests/gnn/test_gnn_overall.py tests/test_core_modules.py \
  tests/test_gnn_multi_format_preset.py tests/pipeline/test_pipeline_recovery.py -q
```

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
