# Step 23: Report

## Architectural Mapping

**Orchestrator**: `src/gnn/23_report.py` (52 lines)
**Implementation Layer**: `src/gnn/report/`

## Module Description

This module provides comprehensive report generation capabilities for GNN pipeline results, including analysis summaries, performance metrics, and detailed documentation.


```
src/gnn/report/
├── __init__.py                       # Module initialization and exports
├── README.md                         # This documentation
├── analyzer.py                       # Report analysis system
├── cross_framework_reliability.py    # Markdown rendering for cross-framework reliability ledgers
├── diff_report.py                    # Diff-aware reporting — compares runs to detect regressions
├── formatters.py                     # Report formatting utilities
├── generator.py                      # Report generation system
├── mcp.py                            # Model Context Protocol integration
├── model_family.py                   # Markdown rendering for model-family acceptance ledgers
├── pipeline_report.py                # Auto-generates PIPELINE_REPORT.md after each run
├── processor.py                      # Report processor module (process_report entry point)
└── semantic_fidelity.py              # Markdown rendering for the semantic fidelity gate
```

## Agent Identity & Capabilities

# Report Module - Agent Scaffolding

## Module Overview

**Purpose**: Comprehensive analysis report generation with multiple format support and automated documentation creation

**Pipeline Step**: Step 23: Report generation (23_report.py)

**Category**: Documentation / Analysis Reporting

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-01-21

---

## Core Functionality

### Primary Responsibilities

1. Generate comprehensive analysis reports from pipeline results
2. Create multi-format documentation (HTML, Markdown, JSON, PDF)
3. Aggregate results from all pipeline steps into unified reports
4. Provide automated report generation and formatting
5. Enable customizable report templates and styling

### Key Capabilities

- Multi-format report generation (HTML, Markdown, JSON, PDF)
- Pipeline results aggregation and analysis
- Automated documentation creation
- Customizable report templates and styling
- Interactive HTML reports with visualizations
- Cross-reference linking between pipeline steps

---

## API Reference

### Public Functions

#### `process_report(target_dir: Path, output_dir: Path, verbose: bool = False, logger: Optional[logging.Logger] = None, **kwargs) -> bool`

**Description**: Main report processing function called by orchestrator (23_report.py). Generates comprehensive analysis reports from pipeline results.

**Parameters**:

- `target_dir` (Path): Directory containing pipeline results (typically "output/")
- `output_dir` (Path): Output directory for report results
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Optional[logging.Logger]): Logger instance for progress reporting (default: None)
- `report_format` (str, optional): Report format ("comprehensive", "summary", "technical") (default: "comprehensive")
- `include_visualizations` (bool, optional): Include visualizations in report (default: True)
- `output_formats` (List[str], optional): Output formats ["html", "markdown", "json"] (default: ["html", "markdown"])
- `**kwargs`: Additional report-specific options

**Returns**: `bool` - True if report generation succeeded, False otherwise

**Example**:

```python
from report import process_report
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
success = process_report(
    target_dir=Path("output"),
    output_dir=Path("output/23_report_output"),
    logger=logger,
    verbose=True,
    report_format="comprehensive",
    include_visualizations=True,
    output_formats=["html", "markdown", "json"],
)
```

#### `generate_comprehensive_report(target_dir: Path, output_dir: Path, **kwargs) -> Dict[str, Any]`

**Description**: Generate comprehensive analysis report from pipeline results.

**Parameters**:

- `target_dir` (Path): Input directory with pipeline results
- `output_dir` (Path): Output directory for reports
- `format` (str, optional): Report format ("html", "markdown", "json") (default: "html")
- `include_executive_summary` (bool, optional): Include executive summary (default: True)
- `include_detailed_analysis` (bool, optional): Include detailed analysis (default: True)
- `**kwargs`: Additional formatting options

**Returns**: `Dict[str, Any]` - Report generation results with:

- `success` (bool): Whether generation succeeded
- `report_files` (List[Path]): Paths to generated report files
- `summary` (Dict): Report summary statistics

#### `generate_html_report(data: Dict[str, Any], output_path: Path, **kwargs) -> bool`

**Description**: Generate interactive HTML report from analysis data.

**Parameters**:

- `data` (Dict[str, Any]): Analysis data to include in report
- `output_path` (Path): Output path for HTML file
- `include_charts` (bool, optional): Include interactive charts (default: True)
- `theme` (str, optional): HTML theme ("default", "dark", "light") (default: "default")
- `**kwargs`: Additional HTML formatting options

**Returns**: `bool` - True if HTML report generation succeeded, False otherwise

---

## Report Types and Formats

### Comprehensive Report

**Purpose**: Complete pipeline analysis with all details
**Features**:

- Executive summary and key findings
- Detailed step-by-step analysis
- Performance metrics and benchmarks
- Error analysis and recommendations
- Cross-reference linking

### Summary Report

**Purpose**: High-level overview for quick review
**Features**:

- Key metrics and success indicators
- Performance highlights
- Critical findings and warnings
- Action items and next steps

### Technical Report

**Purpose**: Detailed technical documentation
**Features**:

- Implementation details and architecture
- Configuration and environment information
- Troubleshooting and debugging information
- API reference and usage examples

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

- `pathlib` - Path manipulation and file operations
- `json` - JSON data handling and serialization
- `typing` - Type annotations and validation

### Optional Dependencies

- `jinja2` - HTML template rendering (recovery: basic HTML)
- `markdown` - Markdown processing (recovery: text-based)
- `weasyprint` - PDF generation (recovery: HTML only)
- `plotly` - Interactive charts in HTML (recovery: static images)

### Internal Dependencies

- `gnn.utils.pipeline_orchestration.pipeline_template` - Standardized pipeline processing
- `pipeline.config` - Configuration management

---

## Configuration

### Environment Variables

- `REPORT_FORMAT` - Default report format ("html", "markdown", "json")
- `REPORT_TEMPLATE` - Custom report template path
- `REPORT_INCLUDE_VISUALS` - Include visualizations in reports (default: True)
- `REPORT_COMPRESSION` - Enable report compression for large outputs

### Configuration Files

- `report_config.yaml` - Report generation settings and templates

### Default Settings

```python
DEFAULT_REPORT_SETTINGS = {
    "format": "html",
    "template": "default",
    "include_visualizations": True,
    "include_raw_data": False,
    "compression": False,
    "max_file_size": "100MB",
    "sections": {
        "executive_summary": True,
        "detailed_analysis": True,
        "performance_metrics": True,
        "error_analysis": True,
        "recommendations": True,
        "appendices": False,
    },
}
```

---

## Usage Examples

### Basic Report Generation

```python
from report.processor import process_report

success = process_report(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/23_report_output"),
    logger=logger,
    report_format="comprehensive",
)
```

### HTML Report Generation

```python
from report.generator import generate_html_report

data = {"pipeline_results": {...}, "analysis_data": {...}, "performance_metrics": {...}}

success = generate_html_report(
    data, Path("output/23_report_output/comprehensive_report.html")
)
```

### Custom Report Configuration

```python
from report.processor import generate_comprehensive_report

report = generate_comprehensive_report(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/23_report_output"),
    format="html",
    include_visualizations=True,
    template="custom",
)
```

---

## Output Specification

### Output Products

- `comprehensive_report.html` - Interactive HTML report
- `comprehensive_report.md` - Markdown documentation
- `comprehensive_report.json` - Structured data export
- `report_generation_summary.json` - Generation metadata
- `report_assets/` - Images, charts, and supporting files

### Output Directory Structure

```text
output/23_report_output/
├── comprehensive_report.html
├── comprehensive_report.md
├── comprehensive_report.json
├── report_generation_summary.json
└── report_assets/
    ├── charts/
    ├── images/
    └── data/
```

---

## Performance Characteristics

### Latest Execution

- **Duration**: ~5-15 seconds (depending on data size)
- **Memory**: ~50-100MB for comprehensive reports
- **Status**: ✅ Production Ready

### Expected Performance

- **Fast Path**: ~2-5s for summary reports
- **Slow Path**: ~10-30s for comprehensive reports with visuals
- **Memory**: ~20-50MB for typical reports, ~100MB+ for large datasets

---

## Error Handling

### Graceful Degradation

- **No template engine**: Recovery to basic HTML templates
- **No visualization libraries**: Generate text-based reports
- **Large datasets**: Sampling and summary generation

### Error Categories

1. **Template Errors**: Invalid or missing report templates
2. **Format Errors**: Unsupported output format requests
3. **Data Errors**: Invalid or corrupted pipeline data
4. **Resource Errors**: Memory or disk space exhaustion

---

## Integration Points

### Orchestrated By

- **Script**: `23_report.py` (Step 23)
- **Function**: `process_report()`

### Imports From

- `gnn.utils.pipeline_orchestration.pipeline_template` - Standardized processing patterns
- `pipeline.config` - Configuration management

### Imported By

- `tests.test_report_integration.py` - Report generation tests
- `main.py` - Pipeline orchestration

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

### Test Coverage

- Measure: `uv run --extra dev python -m pytest tests/report/ --cov=gnn.report --cov-report=term-missing` (do not treat fixed percentages in this doc as canonical).

### Key Test Scenarios

1. Report generation across all supported formats
2. Template rendering and customization
3. Large dataset handling and performance
4. Error handling with malformed data
5. Integration with pipeline results

---

## MCP Integration

### Tools Registered

- `generate_report` - Generate reports from pipeline data
- `process_report` - Run Step 23 over a directory
- `list_report_formats` - List supported report formats
- `read_report` - Read a saved report
- `get_report_module_info` - Module metadata

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

#### Issue 2: HTML validation errors

**Symptom**: HTML reports have validation errors  
**Cause**: Template issues or missing content  
**Solution**:

- Check HTML structure in generated files
- Verify all referenced assets exist
- Review HTML validation logs
- Use markdown format if HTML issues persist

---

## Version History

### Current module status

**Features**:

- Multi-format report generation
- Pipeline results aggregation
- Automated documentation
- Customizable templates

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

**Last Updated**: 2026-01-21
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern


---
## Documentation
- **[README](../../../src/gnn/report/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/report/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/report/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/report/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/report](../../../src/gnn/report)
