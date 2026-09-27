# Step 20: Website

## Architectural Mapping

**Orchestrator**: `src/gnn/20_website.py` (54 lines)
**Implementation Layer**: `src/gnn/website/`

## Module Description

This module provides comprehensive static HTML website generation capabilities for the GNN pipeline, creating interactive websites from pipeline artifacts including visualizations, reports, analysis results, and documentation.


```
src/gnn/website/
├── __init__.py                     # Module initialization and exports
├── README.md                       # Module documentation
├── collection.py                   # collect_website_data: aggregates page inputs, copies visualization assets
├── generator.py                    # WebsiteGenerator / generate_website: builds the seven site pages (from SITE_PAGES) plus per-model pages under model/ and search-index.json
├── pages.py                        # One site page catalogue (SITE_PAGES): single source for page inventories
├── renderer.py                     # WebsiteRenderer, process_website, and HTML/asset embedding helpers
├── inspection.py                   # inspect_website / list_website_pages / read_website_page: site inventory + page-read queries
├── processor.py                    # Thin facade re-exporting process_website from renderer.py
└── mcp.py                          # Model Context Protocol integration

## Agent Identity & Capabilities

# Website Module - Agent Scaffolding

## Module Overview

**Purpose**: Static HTML website generation from pipeline artifacts and results

**Pipeline Step**: Step 20: Website generation (20_website.py)

**Category**: Documentation / Website Generation

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-01-21

---

## Core Functionality

### Primary Responsibilities
1. Generate static HTML websites from pipeline results
2. Create interactive documentation and reports
3. Organize and present pipeline artifacts
4. Generate cross-linked documentation
5. Create publication-ready websites

### Key Capabilities
- Static website generation from pipeline artifacts
- Interactive documentation and reports
- Cross-linked content organization
- Publication-ready HTML output
- Asset management

---

## API Reference

### Public Functions

#### `process_website(target_dir: Path, output_dir: Path, verbose: bool = False, pipeline_output_root: Optional[Path] = None, **kwargs) -> bool`
**Description**: Main website generation function called by orchestrator (20_website.py). Generates a seven-page static HTML website from pipeline artifacts plus one per-model detail page per parsed model under `model/` and `search-index.json`, and writes `website_results.json`.

**Parameters**:
- `target_dir` (Path): Directory containing pipeline artifacts
- `output_dir` (Path): Output directory for website files
- `verbose` (bool): Enable verbose logging (default: False)
- `pipeline_output_root` (Optional[Path]): Root of numbered pipeline output dirs (default: `output_dir.parent`)
- `**kwargs`: Additional website generation options (absorbed silently; the former `website_html_filename` orchestrator knob was removed end-to-end in v3.6.0 as dead)

**Returns**: `bool` - True if website generation succeeded, False otherwise

**Example**:
```python
from gnn.website import process_website
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
success = process_website(
    target_dir=Path("output"),
    output_dir=Path("output/20_website_output"),
    verbose=True,
)
```

#### `generate_html_report(content: str, output_file: Path) -> bool`
**Description**: Write an HTML report file wrapping the given content with default styling.

**Parameters**:
- `content` (str): Content to embed as the report body
- `output_file` (Path): Path of the HTML file to write

**Returns**: `bool` - True if the report file was written, False otherwise

#### `embed_image(image_path: Path, output_file: Path) -> bool`
**Description**: Write an HTML page referencing the given image file (path reference, not base64).

**Parameters**:
- `image_path` (Path): Path to image file
- `output_file` (Path): Output HTML file referencing the image

**Returns**: `bool` - True if the page was written, False otherwise

#### `embed_markdown_file(md_path, output_file) -> bool`
**Description**: Embed markdown file in HTML output

**Parameters**:
- `md_path`: Path to markdown file
- `output_file`: Output HTML file

**Returns**: `True` if embedding succeeded

---

## Dependencies

### Required Dependencies
Stdlib only: `logging`, `pathlib`, `json`, `shutil`, `datetime`, `html`.

### Optional Dependencies
None — the module imports no third-party packages: pages are built with inline CSS/HTML written directly to the output directory, with no templating engine.

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities
---

## Configuration

Generation requires no configuration: `WebsiteGenerator` accepts no settings and always builds the same seven site pages, plus one generated detail page per parsed model under `model/` and `search-index.json` (site pages carry breadcrumb navs; model pages render the full source). The only config surface is the validation helper:

```python
# Keys checked by validate_website_config (renderer.py)
{
    "output_dir": "output/20_website_output",  # required
    "input_dir": "output",                     # optional
}
```

---

## Usage Examples

### Basic Website Generation
```python
from gnn.website import process_website

success = process_website(target_dir="output/", output_dir="output/20_website_output")
```

### HTML Report Generation
```python
from gnn.website import generate_html_report
from pathlib import Path

generate_html_report(markdown_content, Path("report.html"))
```

### Asset Embedding
```python
from gnn.website import embed_image
from pathlib import Path

success = embed_image(
    image_path=Path("visualizations/network.png"), output_file=Path("website/index.html")
)
```

---

## Output Specification

### Output Products
- `index.html` - Main website page
- `pipeline.html`, `gnn_files.html`, `analysis.html`, `visualization.html`, `reports.html`, `mcp.html` - The remaining six of the seven site pages
- `model/<slug>.html` - One detail page per parsed model: model-name `h1`, a source link back to the model's GNN Files listing row, variables/edges tables, embedded visualization assets, and the model's FULL GNN source (no truncation — the 3000-character cap applies only to the aggregate GNN Files listing rows)
- `search-index.json` - Client-side search index (`{"generated", "pages": [{"title", "url", "snippet"}]}`, ≤200-char snippets) covering the 7 site pages plus all model pages
- `assets/` - Visualization PNG/HTML artifacts copied flat into this directory
- `website_results.json` - Generation manifest (success, pages_created, pages, errors, warnings, generated_at, plus model_pages_created and model_pages for the per-model pages)

Every generated page — the seven site pages and every model page — emits a breadcrumb nav in the page shell (`Home › <section>`; model pages: `Home › GNN Files › <Model Name>`; index: a single `Home` crumb). The GNN Files listing page carries a client-side search box driven by an inline copy of `search-index.json` (the payload is inlined because `fetch()` fails on `file://`).

### Output Directory Structure
```
output/20_website_output/
├── index.html
├── pipeline.html
├── gnn_files.html
├── analysis.html
├── visualization.html
├── reports.html
├── mcp.html
├── model/            # one detail page per parsed model (model/<slug>.html)
├── search-index.json
├── website_results.json
└── assets/            # visualization artifacts, written flat
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~2-5 seconds
- **Memory**: ~50-100MB
- **Status**: ✅ Production Ready

### Expected Performance
- **Basic Generation**: 1-2 seconds
- **Full Website**: 3-5 seconds
- **Content Processing**: 2-4 seconds

---

## Error Handling

### Website Errors
1. **Page Rendering Errors**: Failure building one of the seven site pages (or one of the per-model detail pages)
2. **Content Errors**: Content processing failures
3. **Asset Errors**: Asset copy failures
4. **File I/O**: File system operation failures

### Recovery Strategies
- **Per-Page Isolation**: Each page is rendered and written independently; a failing page is recorded in `errors` while the remaining pages stay intact
- **Content Simplification**: Simplify content processing
- **Asset Skip**: Skip problematic assets
- **Error Documentation**: Errors are recorded in `website_results.json`

---

## Integration Points

### Orchestrated By
- **Script**: `20_website.py` (Step 20)
- **Function**: `process_website()`

### Imports From
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

### Imported By
- `main.py` - Pipeline orchestration
- `tests/website/` - Website tests

### Data Flow
```
Pipeline Artifacts → Data Collection → Page Rendering → Asset Copying → Website Output
```

---

## Testing

### Test Files
- `tests/website/test_website_overall.py` - Module-level tests

### Test Coverage
- Measure: `uv run --extra dev python -m pytest tests/website/ --cov=src/gnn/website --cov-report=term-missing` (do not treat fixed percentages in this doc as canonical).

### Key Test Scenarios
1. Website generation from pipeline artifacts
2. HTML report creation and formatting
3. Asset embedding and management
4. Error handling and recovery

---

## MCP Integration

### Tools Registered

Registered in `register_tools` (`src/gnn/website/mcp.py`):

- `process_website` - Run Step 20 over pipeline artifacts
- `build_website_from_pipeline_output` - Build the site from a pipeline output tree
- `get_website_status` - Read website generation status
- `list_generated_website_pages` - List pages in a generated site
- `get_website_module_info` - Module metadata
- `get_website_page` - Read one page's HTML content by catalogue page key (capped read)

### MCP File Location
- `src/gnn/website/mcp.py` - MCP tool registrations


---

## Troubleshooting

### Common Issues

#### Issue 1: Website generation fails
**Symptom**: HTML files not generated or incomplete  
**Cause**: Missing pipeline artifacts or failed prior steps  
**Solution**: 
- Verify previous pipeline steps completed successfully
- Check that required artifacts exist in output directories
- Use `--verbose` flag for detailed generation logs

#### Issue 2: Embedded content missing
**Symptom**: Website generated but images or markdown not embedded  
**Cause**: File paths incorrect or files missing  
**Solution**:
- Verify all referenced files exist
- Check file paths are relative to website output directory
- Ensure images and markdown files are accessible
- Review embedding function logs

---

## Version History

### Current module status

**Features**:
- Static HTML website generation
- Interactive documentation
- Cross-linked content
- Asset management

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced interactivity
- **Future**: Dynamic content generation

---

## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [Website Module](../../../src/gnn/website/../website/README.md)

### External Resources
- [HTML5 Specification](https://html.spec.whatwg.org/)

---

**Last Updated**: 2026-01-21
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern

---
## Documentation
- **[README](../../../src/gnn/website/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/website/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/website/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/website/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/website](../../../src/gnn/website)
