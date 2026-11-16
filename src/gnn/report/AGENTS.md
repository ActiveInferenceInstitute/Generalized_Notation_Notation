# Report Module - Agent Scaffolding

## Module Overview

**Purpose**: Consolidates pipeline outputs (per-step artifacts, logs, metrics) into HTML, Markdown, and JSON analysis reports with health scoring.

**Pipeline Step**: Step 23: Report generation (23_report.py)

**Category**: Documentation / Analysis Reporting

**Status**: Production Ready

**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-24

---

## Core Functionality

1. Aggregate results from all pipeline step output directories
2. Compute a pipeline health score (0-100)
3. Generate the comprehensive report in HTML, Markdown, and JSON formats
4. Write generation metadata summaries
5. Support run-to-run diffing (`diff_report.py`) and ledger renderers (model family, semantic fidelity, cross-framework reliability)

---

## API Reference

### Public Functions

#### `process_report(target_dir: Path, output_dir: Path, verbose: bool = False, logger: Optional[logging.Logger] = None, **kwargs) -> bool`

**Description**: Main report processing function called by orchestrator (23_report.py). Determines the pipeline output directory (parent of `output_dir`) and delegates to `generate_comprehensive_report()`.

**Parameters**:

- `target_dir` (Path): Directory containing pipeline results (typically "output/")
- `output_dir` (Path): Output directory for report results (typically "output/23_report_output")
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Optional[logging.Logger]): Logger instance (default: None)
- `report_formats` (List[str], via kwargs): Formats to generate, subset of `["html", "markdown", "json"]` (default: all three)
- `include_performance` (bool, via kwargs): Include performance metrics section (default: True)
- `include_errors` (bool, via kwargs): Include error analysis section (default: True)
- `include_dependencies` (bool, via kwargs): Include dependency analysis section (default: True)

**Returns**: `bool` - True if report generation succeeded, False otherwise

**Example**:

```python
from gnn.report import process_report
from pathlib import Path

success = process_report(
    target_dir=Path("output"),
    output_dir=Path("output/23_report_output"),
    verbose=True,
)
```


#### `generate_comprehensive_report(...)`

Two variants exist:

- `report.generator.generate_comprehensive_report(pipeline_output_dir, report_output_dir, logger, report_formats=None, include_performance=True, include_errors=True, include_dependencies=True) -> bool` — collects data from all step directories via `analyzer.collect_pipeline_data()`, computes the health score, and writes the report files.
- `report.processor.generate_comprehensive_report(target_dir, output_dir, format="json", **kwargs) -> Dict[str, Any]` — file-level analysis of GNN `.md` files in `target_dir` (`analyze_gnn_file()` per file); returns report data with `success`, `total_files`, `files_analyzed`, `summary`.

#### `generate_html_report(report_data: Dict[str, Any]) -> str`

**Description**: Renders report data to an HTML string (`report/processor.py`). For file writing, use `report.generator.generate_html_report_file(pipeline_data, report_output_dir, logger) -> bool`. A markdown counterpart `generate_markdown_report()` exists in both modules.

---

## Report Types and Formats

### Comprehensive Report

**Purpose**: Complete pipeline analysis with all details
**Features**:

- Pipeline summary and key findings
- Detailed step-by-step analysis
- Performance metrics
- Error analysis
- Health score (0-100)

### Summary Report

**Purpose**: High-level overview for quick review
**Features**:

- Key metrics and success indicators
- Performance highlights
- Critical findings and warnings
- Action items and next steps

### Technical Report

`generate_custom_report()` in `generator.py` supports custom report generation; `diff_report.compare_runs()` produces run-to-run diffs.

### Supported Formats

#### HTML Format

- Interactive web-based reports
- Embedded visualizations and charts
- Collapsible sections and navigation
- Responsive design for mobile devices

#### Markdown Format

- Human-readable structured documentation
- GitHub-compatible formatting
- Easy integration with documentation systems
- Print-friendly layout

#### JSON Format

- Machine-readable structured data
- API integration and automation
- Data analysis and processing
- Metadata and configuration export

#### PDF Format (via external tools)

- Professional document format
- Print-ready layout and styling
- Archival and sharing purposes
- Integration with document management systems

---

## Dependencies

### Required Dependencies

- `pathlib`, `json`, `logging`, `typing` - Standard library

### Optional Dependencies

None. HTML/Markdown generation is pure Python string templating.

### Internal Dependencies

- `gnn.utils.pipeline_orchestration.pipeline_template` - Standardized pipeline processing
- `pipeline.config` - Configuration management
- Submodules: `analyzer.py` (pipeline data collection), `formatters.py` (HTML/Markdown section rendering), `generator.py` (report file writers), `pipeline_report.py` (re-exports the per-step report renderer owned by `gnn.pipeline.summary_wiring`), `diff_report.py` (run-to-run comparison), `model_family.py`, `semantic_fidelity.py`, `cross_framework_reliability.py` (re-export ledger renderers owned by the `gnn.pipeline` producers)

---

## Configuration

### Environment Variables

None dedicated to this module. Report behavior is configured through
`process_report()` kwargs (e.g. `report_formats`, `include_performance`,
`include_errors`, `include_dependencies`) and `input/config.yaml` pipeline
settings.

### Default Settings

Default report formats (`["html", "markdown", "json"]`) and section inclusion
flags are set in `report/generator.py` (`generate_comprehensive_report`) and
`report/__init__.py` (`process_report`).


---

## Usage Examples

### Basic Report Generation

```python
from gnn.report.processor import process_report
from pathlib import Path
import logging

success = process_report(
    target_dir=Path("output"),
    output_dir=Path("output/23_report_output"),
    verbose=True,
)
```

### Custom Format Selection

```python
success = process_report(
    target_dir=Path("output"),
    output_dir=Path("output/23_report_output"),
    report_formats=["html", "json"],
    include_dependencies=False,
)
```

### File-Level GNN Analysis

```python
from gnn.report.processor import generate_comprehensive_report

report = generate_comprehensive_report(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/23_report_output"),
    format="html",
)
```

---

## Output Specification

### Output Products

- `comprehensive_analysis_report.html` - Full HTML analysis report
- `comprehensive_analysis_report.md` - Markdown analysis report
- `report_summary.json` - Structured JSON export of report data
- `report_generation_summary.json` - Generation metadata (health score, formats, options)
- `report_processing_summary.json` - Step processing summary

### Output Directory Structure

```text
output/23_report_output/
├── comprehensive_analysis_report.html
├── comprehensive_analysis_report.md
├── report_summary.json
├── report_generation_summary.json
└── report_processing_summary.json
```

---

## Error Handling

### Graceful Degradation

- **Missing pipeline directory**: Falls back to using `output_dir` as the pipeline directory (with warning)
- **Per-format generation failure**: Logged; remaining formats still attempted
- **Per-file analysis failure**: Recorded in `summary.errors`; report continues

### Error Categories

1. **Format Errors**: Unsupported output format requests (validated against `{html, json, markdown}`)
2. **Data Errors**: Invalid or corrupted pipeline data (per-file errors collected, non-fatal)
3. **Write Errors**: Report write failures (generation returns False if no files produced)

---

## Integration Points

### Orchestrated By

- **Script**: `23_report.py` (Step 23)
- **Function**: `process_report()`

### Imports From

- `gnn.utils.pipeline_orchestration.pipeline_template` - Standardized processing patterns
- `pipeline.config` - Configuration management

### Imported By

- `tests/report/*` - Report tests
- `src/gnn/main.py` - Runs `23_report.py` as a pipeline step; the final per-step report is written by `gnn.pipeline.summary_wiring` (re-exported as `report.pipeline_report.generate_pipeline_report`); `gnn.pipeline` never imports `gnn.report`

### Data Flow

```
Pipeline Results → Report Aggregation → Data Analysis → Format Generation → Multi-format Output
```

---

## Testing

### Test Files

- `tests/report/test_report_integration.py` - Integration tests
- `tests/report/test_report_generation.py` - Generation tests
- `tests/report/test_report_formats.py` - Format tests
- `tests/report/test_report_functional.py`, `test_report_generator_functional.py`, `test_report_overall.py` - Functional and module tests
- `tests/report/test_report_diff_and_ledgers.py`, `test_model_family_report.py` - Diff/ledger renderer tests
- `tests/report/test_report_mcp_wrappers.py` - MCP wrapper tests

### Test Coverage

Measure on demand:

```bash
uv run --extra dev python -m pytest tests/report/ \
    --cov=src/gnn/report --cov-report=term-missing
```

### Key Test Scenarios

1. Report generation across all supported formats
2. Health-score computation
3. Error handling with malformed data
4. Integration with pipeline results

---

## MCP Integration

### Tools Registered

- `generate_report` - Run the full report pipeline step
- `process_report` - Process report generation for a directory
- `list_report_formats` - Return all supported report output formats
- `read_report` - Read and return the contents of a generated report file
- `get_report_module_info` - Return module metadata (version, supported formats)

### MCP File Location

- `src/gnn/report/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Report generation fails

**Symptom**: Report files not generated or incomplete  
**Cause**: Missing pipeline artifacts or template issues  
**Solution**:

- Verify all pipeline steps completed successfully
- Check that required artifacts exist in output directories
- Use `--verbose` flag for detailed generation logs
- Review report template structure

#### Issue 2: Unexpected report content

**Symptom**: Report sections missing or incomplete
**Cause**: Pipeline artifacts absent for a step, or section-inclusion kwargs disabled
**Solution**:

- Verify the pipeline step output directories exist under the pipeline output dir
- Check `report_generation_summary.json` for formats and options actually used
- Re-run with `include_performance`/`include_errors`/`include_dependencies` left at defaults

---

## Version History

### Current Version: [pyproject.toml](../../../pyproject.toml) (canonical)

**Features**:

- Multi-format report generation (HTML, Markdown, JSON)
- Pipeline results aggregation with health scoring
- Run-to-run diff reports and ledger renderers

**Known Issues**:

- None currently

### Roadmap

- **Next Version**: Enhanced visualization integration
- **Future**: Real-time report updates

---

## References

### Related Documentation

- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [Documentation Index](../../../docs/README.md)

### External Resources

- [Markdown Specification](https://daringfireball.net/projects/markdown/)

---

**Last Updated**: 2026-09-24
**Maintainer**: GNN Pipeline Team
**Status**: Production Ready
**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: Thin Orchestrator Pattern


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
