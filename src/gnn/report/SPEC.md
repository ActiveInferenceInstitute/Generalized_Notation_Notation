# Report Module Specification

Consolidation of pipeline outputs into comprehensive HTML/Markdown/JSON analysis reports with health scoring. Step 23 of the GNN pipeline.

## Components

### Core
- `processor.py` - `process_report()` step entry, `analyze_gnn_file()`, HTML/Markdown renderers
- `generator.py` - `generate_comprehensive_report()` and report file writers (HTML/Markdown/JSON/summary/custom)
- `analyzer.py` - `collect_pipeline_data()` aggregation over step output directories
- `formatters.py` - HTML/Markdown section rendering (performance, errors, steps, visualizations)
- `pipeline_report.py` - Per-step status/timing/artifact/statistics sections
- `processing_report.py` - `ReportGenerator` processing-context reports: structural `valid/errors/warnings` mappings and typed `is_valid/errors/warnings` results retain actual Boolean counts and errors. Missing required fields or malformed validity refuse compilation; returned write failures retain completed format paths and `report_files["error"]`.
- `diff_report.py` - `compare_runs()` / `archive_run()` run-to-run diffing
- `model_family.py`, `semantic_fidelity.py`, `cross_framework_reliability.py` - ledger markdown renderers
- `mcp.py` - MCP tool registrations (5 tools)

## Features
- Multi-format report generation (HTML, Markdown, JSON)
- Pipeline report inputs must be existing directories. `report_formats=None` uses all three defaults; explicit lists are copied and honored. Empty or entirely unsupported selections return `False` with a generation summary and no report files. Mixed unsupported formats and partial writer failures retain the existing best-effort behavior: completion means at least one writer succeeded, and the saved generation summary names its completed files.
- Pipeline health score (0-100)
- Run-to-run diff reports and acceptance-ledger renderers

## Key Exports
```python
from gnn.report import process_report
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
