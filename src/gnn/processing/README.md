# GNN Processing Module

The processing concern for GNN files: corpus discovery, a lightweight
parse/validate/report surface, the five-phase orchestration engine, and the
step-3 multi-format serialization body. Consolidated into one package in
3.2.0 so numbered step scripts stay thin.

## Module Structure

| File | Purpose |
|------|---------|
| `processor.py` | Lightweight surface: `discover_gnn_files`, `parse_gnn_file`, `check_gnn_file_structure`, directory processing, `generate_gnn_report`, `get_module_info` |
| `core_processor.py` | `GNNProcessor`: five-phase orchestration engine (discovery → validation → round-trip → cross-format → reporting) over a `ProcessingContext` |
| `multi_format_processor.py` | `process_gnn_multi_format`: step-3 discovery, parsing, and multi-format serialization orchestrator |
| `discovery.py` | `is_model_source_path` corpus filtering, `FileDiscoveryStrategy` content-aware discovery, `DiscoveryResult` |
| `__init__.py` | Curated public surface (`__all__`) |

## Public API

Re-exported from `gnn.processing` (see `__init__.py`):

- Lightweight surface (`processor.py`): `discover_gnn_files`,
  `parse_gnn_file`, `check_gnn_file_structure`, `process_gnn_directory`,
  `process_gnn_directory_lightweight`, `generate_gnn_report`,
  `get_module_info`
- Orchestration (`core_processor.py`): `GNNProcessor`, `ProcessingContext`,
  `ProcessingPhase`, plus the module-level `process_gnn_directory` /
  `process_gnn_directory_lightweight` entry points (see
  [Core-Processor Wrappers](#core-processor-wrappers-distinct-from-the-canonical-surface)
  for how they differ from the canonical `processor.py` functions)
- Multi-format (`multi_format_processor.py`): `process_gnn_multi_format`
- Discovery (`discovery.py`): `is_model_source_path`, `FileDiscoveryStrategy`,
  `DiscoveryResult`, `NON_MODEL_MARKDOWN_FILENAMES`,
  `NON_MODEL_MARKDOWN_SUFFIXES`

## Core-Processor Wrappers (distinct from the canonical surface)

`GNNProcessor.process(context)` returns `True` only after its required JSON,
Markdown, and HTML processing reports have been written as nonempty files.
Reporting exceptions, a returned write error, or a missing required format
return `False` and record `context.processing_results["report_error"]` with
reporting-phase context. Any returned partial report metadata and completed
files remain available in `context.processing_results["report"]`.

`core_processor.py` defines module-level `process_gnn_directory` and
`process_gnn_directory_lightweight` that intentionally differ from the
canonical `processor.py` functions of the same names (the ones the root
package re-exports through `_EXPORT_MAP`):

- `core_processor.process_gnn_directory` drives the five-phase
  `GNNProcessor` engine and — if the engine raises — falls back to a raw
  `_scan_files_lightweight` glob scan, always returning `status="SUCCESS"`
  with a `processing_mode` of `"full"` or `"lightweight"` plus a
  `valid_files` key, and writes `gnn_core_results.json`. The canonical
  `processor.process_gnn_directory` never invokes the engine: it parses
  each discovered file, returns `FAILED` for missing paths (Phase 1.3
  validation), and writes `gnn_processing_results.json`.
- `core_processor.process_gnn_directory_lightweight` is a pure
  `**/*.md` glob scan (single-`.md` paths accepted) returning a
  `{path: {status, format, size}}` mapping with the recursion flag
  ignored, writing `gnn_core_lightweight_results.json`. The canonical
  `processor.process_gnn_directory_lightweight` parses each file and
  returns an aggregate result dict with `parsed_files` /
  `validation_results`.

These recovery semantics are pinned by `tests/pipeline/test_pipeline_recovery.py`
(engine-failure fallback and lightweight scan), `tests/pipeline/test_pipeline_functionality.py`
(engine path), and `tests/infrastructure/test_coverage_overall.py`
(lightweight callable). Consumers wanting the canonical surface should
import from `gnn` or `gnn.processing`.

## Usage

```python
from pathlib import Path

from gnn.processing import (
    GNNProcessor,
    ProcessingContext,
    is_model_source_path,
    parse_gnn_file,
    process_gnn_multi_format,
    process_gnn_directory_lightweight,
    check_gnn_file_structure,
)

# Single-file parse and structure validation
parsed = parse_gnn_file("my_model.gnn")
check = check_gnn_file_structure("my_model.gnn")

# Lightweight directory sweep (no heavy dependencies)
results = process_gnn_directory_lightweight(
    Path("input"), output_dir=Path("output/3_gnn_output"), recursive=True
)

# Five-phase orchestration engine
processor = GNNProcessor()  # optional logger argument
ok = processor.process(
    ProcessingContext(
        target_dir=Path("input"),
        output_dir=Path("output/3_gnn_output"),
        enable_round_trip=True,
        enable_cross_format=True,
    )
)

# Step-3 multi-format serialization body
success = process_gnn_multi_format(
    Path("input"), Path("output"), logger, recursive=True
)

# Corpus filtering: is this path a maintained model source?
is_model_source_path(Path("input/active_inference.md"))  # True
is_model_source_path(Path("input/README.md"))  # False
```

## Facade

The root package re-exports the lightweight surface and the multi-format
entry point through the lazy `_EXPORT_MAP` in `src/gnn/__init__.py`:
`gnn.parse_gnn_file`, `gnn.discover_gnn_files`,
`gnn.process_gnn_directory`, `gnn.process_gnn_directory_lightweight`,
`gnn.check_gnn_file_structure`, `gnn.generate_gnn_report`,
`gnn.get_module_info`, and `gnn.process_gnn_multi_format` all resolve to
`processing.processor` / `processing.multi_format_processor`. Import the
orchestration engine (`GNNProcessor`) and the discovery predicates from
`gnn.processing` directly.

## Pipeline Wiring

None of these symbols are CLI subcommands. The pipeline entry is
`src/gnn/3_gnn.py` (step 3), which wraps `process_gnn_multi_format` with
`create_standardized_pipeline_script`; the step body lives in
`multi_format_processor.py`, not in the numbered script.

## See Also

- [Parent: gnn/README.md](../README.md)
- [AGENTS.md](AGENTS.md) — Agent scaffolding documentation
- [Parsers module](../parsers/README.md) — formal parsing system
- [Validation module](../validation/README.md) — semantic/quality validation
- [Package root](../../../README.md)
