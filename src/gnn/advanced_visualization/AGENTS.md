# Advanced Visualization Module - Agent Scaffolding

## Module Overview

**Purpose**: Advanced visualization artifact generation for GNN models: statistical panels, POMDP plots, network metrics, optional Plotly/HTML dashboards, and optional D2 diagrams

**Pipeline Step**: Step 9: Advanced visualization (src/gnn/9_advanced_viz.py)

**Category**: Advanced Visualization / Interactive Analysis

**Status**: Maintained

**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-22

---

## Core Functionality

### Primary Responsibilities

1. Generate 3D-style visualization artifacts
2. Create optional HTML dashboard artifacts
3. Produce advanced statistical plots
4. Generate optional interactive HTML visualizations when dependencies and flags allow
5. Provide multi-dimensional model summaries
6. Generate professional D2 (Declarative Diagramming) diagrams

### Key Capabilities

- 3D network topology visualization
- Interactive Plotly dashboards
- Multi-panel comparative analysis
- HTML-based interactive reports
- **D2 diagram generation for GNN models and pipeline architecture**

---

## API Reference

### Public Functions

#### `process_advanced_viz(target_dir, output_dir, logger, **kwargs) -> bool | int`

**Description**: Main advanced visualization processing function called by orchestrator ([src/gnn/9_advanced_viz.py](../9_advanced_viz.py)). Implementation: [processor.py](processor.py).

**Parameters**:

- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for visualizations
- `logger` (Logger): Logger instance
- `viz_type` (str): Visualization type ("all", "3d", "interactive", "dashboard", "d2", "diagrams", "pipeline", "statistical", "pomdp", "network", default: "all")
- `interactive` (bool): Enable interactive features (default: True)
- `export_formats` (List[str]): Export formats ["html", "json", "png"], default: ["html", "json"]
- `**kwargs**: Additional options

**Returns**: `True` when at least one advanced visualization artifact is
produced, `2` when the step completes with warning-only recovery such as missing
Step 3 model data or optional-only skips, and `False` for hard failures.

**Example**:

```python
from gnn.advanced_visualization.processor import process_advanced_viz

success = process_advanced_viz(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/9_advanced_viz_output"),
    logger=logger,
    viz_type="all",
    interactive=True,
    export_formats=["html", "json"],
)
```

---

## Visualization Types

### 3D Visualization

- Network topology in 3D space with semantic positioning
- State space visualization with force-directed layout
- Connection strength representation with real POMDP data
- Interactive hover information with variable details

### Statistical Analysis Plots

- Variable type distribution pie charts
- Variable dimension distribution analysis
- Scalar parameter value histograms
- Matrix size distribution analysis
- Matrix correlation heatmaps between all matrices
- Comprehensive statistical overview panels

### POMDP-Specific Visualizations

- **Transition Matrix Analysis**: B matrix visualization with action-specific slices
- **Policy Visualization**: Policy distribution over actions (π and E matrices)
- **3D Transition Visualization**: Multi-action transition matrix heatmaps
- **State-Action Relationships**: Visual representation of POMDP dynamics

### Network Analysis Visualizations

- **Network Metrics**: Node count, edge count, density, clustering coefficients
- **Centrality Analysis**: Degree centrality and node importance rankings
- **Network Graph Visualization**: Force-directed layout with connection visualization
- **Connection Strength Analysis**: Edge weight and connection pattern analysis
- **Network Statistics**: Comprehensive network topology metrics

### Interactive Plotly Dashboards

- **Multi-Panel Dashboard**: Variable types, matrix overview, network graph, statistics
- **Interactive Matrix Explorer**: Zoom, pan, and explore matrix heatmaps
- **Export Support**: HTML output when requested and available
- **Static Fallbacks**: Recorded skips or static artifacts when optional dependencies are unavailable

### Interactive Dashboard

- Model-summary dashboard artifacts when requested and `interactive=True`
- Multi-view reports assembled from extracted model data
- HTML-based interactive reports

### D2 Diagram Generation (NEW)

- **GNN Model Structure**: Visualize state space components, connections, and Active Inference ontology
- **POMDP Diagrams**: Generative model components (A, B, C, D, E matrices) and inference processes
- **Pipeline Architecture**: Complete 25-step pipeline flow with data dependencies
- **Framework Integration**: Mapping of GNN models to PyMDP, RxInfer.jl, ActiveInference.jl, DisCoPy, JAX
- **Active Inference Concepts**: Free Energy Principle, perception-action loops, belief updating
- **Multiple Output Formats**: SVG, PNG, PDF with professional themes
- **Layout Engines**: Dagre (fast), ELK (quality), TALA (advanced)

See [D2_README.md](D2_README.md) for comprehensive D2 integration documentation.

---

## Configuration

### Configuration Options

#### Visualization Type Selection

- `viz_type` (str): Type of visualization to generate
  - `"all"`: Generate all visualization types (default)
  - `"3d"`: Only 3D network visualizations
  - `"interactive"`: Only interactive Plotly dashboards
  - `"dashboard"`: Only dashboard interfaces
  - `"d2"` or `"diagrams"`: Only D2 diagram generation
  - `"pipeline"`: Only pipeline D2 diagrams
  - `"statistical"`: Statistical analysis plots (distributions, correlations, histograms)
  - `"pomdp"`: POMDP-specific visualizations (transitions, policies, beliefs)
  - `"network"`: Network analysis visualizations (metrics, centrality, connection strength)

#### Interactive Features

- `interactive` (bool): Enable interactive features (default: `True`)
  - When `True`: Allows interactive/dashboard branches for matching `viz_type` values
  - When `False`: Skips interactive/dashboard branches

#### Export Formats

- `export_formats` (List[str]): Formats to export (default: `["html", "json"]`)
  - Supported by Step 9 core outputs: `["html", "json", "png"]`
  - D2 diagrams support additional formats when the D2 CLI is installed

No model data is a warning-only outcome, not artifact success. `viz_type` and
`interactive` gate output creation; interactive dashboards are generated only
when an interactive type is requested and `interactive=True`.

#### D2 Configuration

- `d2_layout_engine` (str): Layout engine for D2 diagrams (default: `"dagre"`)
  - Options: `"dagre"` (fast), `"elk"` (quality), `"tala"` (advanced)
- `d2_theme` (str): Theme for D2 diagrams (default: `"default"`)
  - Options: `"default"`, `"dark"`, `"light"`, `"professional"`

No additional public performance-tuning flags are documented for this module.
Generate a narrower `viz_type` or use `interactive=False` to reduce work.

---

## Dependencies

### Required Dependencies

- `matplotlib` - Basic plotting
- `numpy` - Numerical operations

### Optional Dependencies

- `plotly` - Interactive visualizations (recovery: static plots)
- `seaborn` - Enhanced statistical plots (recovery: matplotlib)
- **`d2` CLI** - D2 diagram compilation through the shared run_subprocess_envelope (structured classification, wall-clock force-kill timeout; recovery: skip D2 diagrams, log warning)

---

## Usage Examples

### Basic Usage

```python
from gnn.advanced_visualization.processor import process_advanced_viz

success = process_advanced_viz(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/9_advanced_viz_output"),
    logger=logger,
    viz_type="all",
)
```

### Interactive Dashboard

```python
success = process_advanced_viz(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/9_advanced_viz_output"),
    logger=logger,
    viz_type="dashboard",
    interactive=True,
    export_formats=["html", "json"],
)
```

### D2 Diagram Generation (NEW)

```python
# Generate only D2 diagrams
success = process_advanced_viz(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/9_advanced_viz_output"),
    logger=logger,
    viz_type="d2",  # or "diagrams" or "pipeline"
)

# Programmatic D2 usage
from gnn.advanced_visualization.d2_visualizer import D2Visualizer

visualizer = D2Visualizer(logger=logger)
if visualizer.d2_available:
    # Generate all diagrams for a model
    results = visualizer.generate_all_diagrams_for_model(
        model_data, output_dir, formats=["svg", "png"]
    )
```

---

## Output Specification

### Output Products

- `{model}_3d_visualization.html` - 3D interactive plot
- `{model}_dashboard.html` - Interactive dashboard
- `{model}_statistical_analysis.png` - Statistical plots
- `{model}_visualization_data.json` - Underlying data
- `d2_diagrams/{model}/` - **D2 diagram files (.d2, .svg, .png)**
- `d2_diagrams/pipeline/` - **Pipeline architecture D2 diagrams**
- `advanced_viz_summary.json` - Processing summary

### Output Directory Structure

```
output/9_advanced_viz_output/
├── model_name_3d_visualization.html
├── model_name_dashboard.html
├── model_name_statistical_analysis.png
├── model_name_visualization_data.json
├── d2_diagrams/
│   ├── model_name/
│   │   ├── model_name_structure.d2
│   │   ├── model_name_structure.svg
│   │   ├── model_name_structure.png
│   │   ├── model_name_pomdp.d2
│   │   ├── model_name_pomdp.svg
│   │   └── model_name_pomdp.png
│   └── pipeline/
│       ├── gnn_pipeline_flow.d2
│       ├── gnn_pipeline_flow.svg
│       ├── framework_integration.d2
│       ├── framework_integration.svg
│       ├── active_inference_concepts.d2
│       └── active_inference_concepts.svg
└── advanced_viz_summary.json
```

---

## Performance Characteristics

### Measurement Policy

- This document is not the source of fixed runtime or memory numbers.
- Measure current performance from a fresh local or CI run when making a performance claim.
- Use narrower `viz_type` values or `interactive=False` when a run should avoid optional dashboard work.
- Treat optional dependency skips and no-data outcomes through the documented warning-code path rather than as artifact success.

---

## Error Handling

### Graceful Degradation

- **No Plotly**: Generate static/matplotlib artifacts where supported
- **No D2 CLI**: Skip D2-specific diagram rendering and report the optional dependency state
- **Large Models**: Prefer narrower `viz_type` runs and recorded warnings
- **Parsing Failures**: Return structured error information
- **Missing Dependencies**: Use available libraries with fallbacks

### Robust Error Recovery

- **Data Loading**: Multiple recovery paths for finding GNN models
- **Visualization Generation**: Individual method error isolation
- **File I/O**: Safe file operations with proper cleanup
- **Memory Management**: Proper resource cleanup and monitoring


## Integration Points

### Pipeline Integration

- **Input**: Receives processed GNN models from Step 3 (gnn processing)
- **Output**: Generates visualizations consumed by Step 20 (website generation) and Step 23 (report generation)
- **Dependencies**: Requires GNN parsing results from `3_gnn.py` output

### Module Dependencies

- **gnn/**: Reads parsed GNN model data and structure
- **visualization/**: Complements basic visualization with advanced features
- **export/**: Uses export formats for visualization data serialization

### External Integration

- **D2 CLI**: Integrates with D2 diagramming tool for professional diagrams
- **Plotly**: Optional integration for interactive visualizations

### Data Flow

```
3_gnn.py (GNN parsing)
  ↓
9_advanced_viz.py (Advanced visualization)
  ↓
  ├→ 20_website.py (HTML integration)
  ├→ 23_report.py (Report generation)
  └→ output/9_advanced_viz_output/ (Standalone visualizations)
```

---

## Testing

### Test Files

- `tests/advanced_visualization/test_advanced_visualization_overall.py`
- `tests/advanced_visualization/test_advanced_visualization_shared.py`
- `tests/advanced_visualization/test_advanced_visualization_statistical.py`
- `tests/advanced_visualization/test_advanced_visualization_interactive.py`
- `tests/advanced_visualization/test_advanced_visualization_html_generator.py`
- `tests/advanced_visualization/test_advanced_visualization_public_api.py`
- `tests/advanced_visualization/test_advanced_visualization_composability.py` (refactor: `record_attempt`, `_conn_endpoints`, palette/layout constants)
- `tests/advanced_visualization/test_advanced_visualization_public_api_refactor.py` (refactor: `VIZ_TYPE_CHOICES`, `probe_capabilities`, MCP `generate_d2` honoring, dashboard-timestamp regression)

### Test Coverage

Measure on demand:

```bash
uv run --extra dev python -m pytest tests/advanced_visualization/ \
    --cov=src/gnn/advanced_visualization --cov-report=term-missing
```


## Composability Helpers

The refactor deduplicated cross-file logic into a small set of shared helpers
in `_shared.py` and the package root:

- `record_attempt(results, attempt, *, optional_message_filter=None)` — pure
  aggregate bookkeeping for `AdvancedVisualizationAttempt` → `AdvancedVisualizationResults`
  (success/failed/skipped counts, output_files/errors/warnings). The
  `optional_message_filter` suppresses warning entries whose message mentions an
  optional dependency marker (e.g. `"D2 CLI"`) so optional CLI absence is not
  surfaced as a hard warning. Used by `process_advanced_viz` for all three
  attempt-accounting sites (previously triplicated).
- `_conn_endpoints(conn_info) -> (source_variables, target_variables)` — normalizes
  scalar `{"source", "target"}` and new `{"source_variables", "target_variables"}`
  connection formats in one call. Used by all three connection-expansion sites in
  `network_viz.py` (previously triplicated).
- `VAR_TYPE_COLORS` / `VAR_TYPE_UNKNOWN_COLOR` — the canonical var-type → hex
  color palette used by the 3D visualization (previously inlined twice).
- `FORCE_LAYOUT_SEED` / `LAYOUT_SEED` / `LAYOUT_SPAN` / `LAYOUT_ITERATIONS` / `LAYOUT_STEP`
  — named constants for the force-directed 3D layout (seed=42, span=10, iterations=50,
  step=0.01). The layout now uses `np.random.default_rng(42)` instead of mutating the
  process-global RNG via `np.random.seed(42)`.
- `VIZ_TYPE_CHOICES` (package root) — the canonical tuple of `viz_type` values
  accepted by `process_advanced_viz`; the `src/gnn/9_advanced_viz.py` orchestrator imports
  this instead of hand-maintaining a duplicate list. Scope note: this is true at
  import/config level only — at runtime the enhanced parser
  (`utils.ArgumentParser.ARGUMENT_DEFINITIONS["viz_type"]`) enforces its own copy of
  the choices, and the orchestrator's `choices=list(VIZ_TYPE_CHOICES)` only feeds the
  recovery/fallback parser. A test pins the two lists equal; unify in `utils` if the
  duplication ever drifts.
- `_MatrixVisualizer` — lazy factory from `_shared`; calling it returns a
  `MatrixVisualizer` or `None` (ImportError swallowed), so all four guard sites
  (network_viz ×2, statistical_viz, interactive_viz) check the RESULT
  (`mv = _MatrixVisualizer(); if mv is None: skipped/failed`) rather than the
  never-None proxy object. A missing `visualization.matrix_visualizer` surfaces as
  the documented "MatrixVisualizer not available" skip, not a raw ImportError.
- `probe_capabilities()` (package root) — a live runtime probe of the actual
  environment (`d2` CLI on PATH, `plotly`/`seaborn`/`matplotlib`/`numpy`/`networkx`
  importability), distinct from the static `FEATURES` map. Used by the
  `check_visualization_capabilities` MCP tool so its docstring is honest.
- `D2_COMPILE_TIMEOUT_S` / `D2_MISSING_MESSAGE` / `VALID_D2_FORMATS`
  (`d2_visualizer.py`) — named constants for the D2 compile timeout, the
  missing-CLI message, and the validated output-format set (`svg`/`png`/`pdf`);
  d2 compiles route through the shared subprocess envelope
  (`gnn.execute.subprocess_envelope`), imported lazily at the compile call site
  so the `gnn.execute` package init never rides on the
  `advanced_visualization` import chain, and a hung compile is force-killed at
  `D2_COMPILE_TIMEOUT_S` and classified (`TimeoutExpired`).
- `_theme.py` — shared CSS constants for the two HTML emitters
  (`BASE_CSS`, `FONT_STACK`, `BODY_GRADIENT`, `HEADER_H2_CSS`,
  `PARAMETER_NAME_CSS`, `STAT_LABEL_CSS`). Contains only rules that are
  byte-identical (after whitespace normalization) across `dashboard.py` and
  `html_generator.py`; every rule with cosmetic differences stays inline in
  its emitter. Parity proven against the pre-refactor (git HEAD) output.
- `AdvancedVisualizer(logger=None, extractor=None)` — constructor injection
  for the data extractor; any object with
  `extract_from_content(content) -> dict` is accepted (test doubles included).
  When omitted, a real `VisualizationDataExtractor` is built lazily per run;
  if unavailable, `generate_visualizations` degrades to the recovery path.
  `VIS_PROCESSOR_AVAILABLE` is retained for existing importers.
- `process_gnn_file_with_d2(..., parsed_json_dir=None)` — optional keyword
  overriding the cwd-dependent `output/3_gnn_output` parsed-JSON lookup;
  `None` preserves the historical behavior.
- `_create_network_graph` resolves real node indices via a name→index map
  over `blocks`; both extractor formats and scalar connections
  resolve, unresolvable pairs are silently skipped, positions stay seeded.

## Dashboard Footer Timestamp (Fixed)

`dashboard.py` previously shipped `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
as **literal text** in the footer because the closing HTML chunk was a plain
`"""` string (no `f` prefix) and `datetime` was not imported. The broad
`except Exception` in `generate_dashboard` swallowed the resulting `NameError`,
so dashboard generation silently returned `None` 100% of the time. The fix
imports `datetime` and prefixes the footer chunk with `f`. Regression test:
`TestDashboardTimestampRenders.test_footer_contains_rendered_timestamp` in
`test_advanced_visualization_public_api_refactor.py` (lines 116-117).

### Test Categories

- Unit: module imports, instantiation, basic API surface
- Integration: data extraction, end-to-end visualization generation
- Error handling: missing dependencies, malformed content, degraded paths

---
## MCP Integration

### Tools Registered

- `process_advanced_visualization` - Run Step 9 advanced visualization processing for a target directory
- `check_visualization_capabilities` - Report optional dependency and feature availability
- `list_d2_visualization_types` - List D2 diagram categories and D2 requirements
- `get_advanced_visualization_module_info` - Return module metadata, feature flags, and tool inventory

```python
def process_advanced_visualization_mcp(
    target_directory: str,
    output_directory: str,
    verbose: bool = False,
    generate_d2: bool = True,
) -> Dict[str, Any]:
    """Process advanced visualization for GNN files.

    ``generate_d2=False`` routes to ``viz_type="network"`` (non-D2); ``True``
    runs all viz types. ``verbose`` is accepted for schema stability but no
    longer passed through to ``process_advanced_viz`` (which has no
    ``verbose`` parameter — it was previously swallowed into ``**kwargs``).
```

### MCP File Location

- `src/gnn/advanced_visualization/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: D2 diagram generation fails

**Symptom**: D2 diagrams not generated or errors during generation  
**Cause**: Missing D2 CLI tool or invalid diagram syntax  
**Solution**:

- Install D2: `brew install d2` (macOS) or download from [d2lang.com](https://d2lang.com)
- Verify D2 installation: `d2 --version`
- Check diagram syntax in generated D2 files
- Use `--verbose` flag for detailed error messages

#### Issue 2: Interactive visualizations not displaying

**Symptom**: HTML files generated but visualizations not interactive  
**Cause**: Missing Plotly JavaScript or browser compatibility  
**Solution**:

- Ensure Plotly is installed: `uv pip install plotly`
- Open HTML files in modern browser (Chrome, Firefox, Safari)
- Check browser console for JavaScript errors

#### Issue 3: 3D visualizations fail to render

**Symptom**: 3D visualization generation errors  
**Cause**: Missing 3D plotting dependencies or insufficient resources  
**Solution**:

- Install required dependencies: `uv pip install plotly numpy`
- Reduce model complexity for 3D rendering
- Use 2D recovery visualizations

### Performance Issues

#### Slow Dashboard Generation

**Symptoms**: Dashboard generation takes longer than expected  
**Diagnosis**:

```bash
# Enable verbose logging
python src/gnn/9_advanced_viz.py --target-dir input/ --verbose
```

**Solutions**:

- Generate specific visualization types instead of "all"
- Disable interactive features if not needed
- Process files individually instead of batch

---

## Version History

### Current Version: 3.6.0
Tracks the package version (`pyproject.toml`). Documented-surface changes since
the 3.2.0 marker (2026-09-04): D2 compiles route through the shared subprocess
envelope (b97d0ebb6, 2026-09-19) and seaborn/scipy.stats imports became lazy
across the viz graph (55661abc0, 2026-09-21).

**Features**:

- 3D network visualization
- Interactive Plotly dashboards
- D2 diagram generation
- Statistical analysis plots
- POMDP-specific visualizations
- Network analysis visualizations

**Known Issues**:

- None currently

### Roadmap

- **Next Version**: Enhanced D2 diagram features
- **Future**: Explicit live or streaming contracts only after implementation and tests exist

---

## References

### Related Documentation

- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [Visualization Module](../visualization/AGENTS.md)
- [D2 Documentation](../../../docs/d2/)

### External Resources

- [D2 Language](https://d2lang.com)
- [Plotly Documentation](https://plotly.com/python/)
- [Active Inference Ontology](https://activeinference.org)

---

**Last Updated**: 2026-09-22
**Maintainer**: GNN Pipeline Team
**Status**: Maintained
**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: Thin Orchestrator Pattern


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
