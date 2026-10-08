# Step 9: Advanced Visualization

## Architectural Mapping

**Orchestrator**: `src/gnn/9_advanced_viz.py` (51 lines)
**Implementation Layer**: `src/gnn/advanced_visualization/`

## Module Description

This module provides comprehensive advanced visualization capabilities for GNN models, including interactive dashboards, 3D visualizations, and sophisticated data analysis visualizations.


```
src/gnn/advanced_visualization/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
├── dashboard.py                   # Dashboard generation system
├── data_extractor.py              # Data extraction and processing
├── html_generator.py              # HTML visualization generation
└── visualizer.py                  # Main visualization orchestrator
```

## Agent Identity & Capabilities

# Advanced Visualization Module - Agent Scaffolding

## Module Overview

**Purpose**: Advanced visualization, interactive plots, 3D visualizations, and dashboard generation for GNN models

**Pipeline Step**: Step 9: Advanced visualization (9_advanced_viz.py)

**Category**: Advanced Visualization / Interactive Analysis

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-01-21

---

## Core Functionality

### Primary Responsibilities

1. Generate interactive 3D visualizations
2. Create dynamic dashboard interfaces
3. Produce advanced statistical plots
4. Generate interactive HTML visualizations
5. Provide multi-dimensional data exploration
6. Generate professional D2 (Declarative Diagramming) diagrams

### Key Capabilities

- 3D network topology visualization
- Interactive Plotly dashboards
- Time-series animation
- Multi-panel comparative analysis
- HTML-based interactive reports
- **D2 diagram generation for GNN models and pipeline architecture**

---

## API Reference

### Public Functions

#### `process_advanced_viz(target_dir, output_dir, logger=None, viz_type="all", interactive=True, export_formats=None, **kwargs) -> bool | int`

**Description**: Main advanced visualization processing function called by orchestrator (9_advanced_viz.py)

**Parameters**:

- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for visualizations
- `logger` (Logger): Logger instance
- `viz_type` (str): Visualization type ("all", "3d", "interactive", "dashboard", "d2", "diagrams", "pipeline", "statistical", "pomdp", "network", default: "all")
- `interactive` (bool): Enable interactive features (default: True)
- `export_formats` (List[str]): Export formats ["html", "json", "png"], default: ["html", "json"]
- `**kwargs**: Additional options

**Returns**: `True`/`1` if artifacts were produced, `2` for warning-only/no-data recovery, `False` for hard failures

**Example**:

```python
from advanced_visualization.processor import process_advanced_viz

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
- **Real-Time Updates**: Dynamic visualization updates based on model data
- **Export Support**: HTML and PNG export formats
- **Responsive Design**: Adapts to different screen sizes

### Interactive Dashboard

- Real-time model exploration (when plotly available)
- Parameter adjustment interface
- Multi-view synchronized displays
- HTML-based interactive reports

### D2 Diagram Generation (NEW)

- **GNN Model Structure**: Visualize state space components, connections, and Active Inference ontology
- **POMDP Diagrams**: Generative model components (A, B, C, D, E matrices) and inference processes
- **Pipeline Architecture**: Complete 25-step pipeline flow with data dependencies
- **Framework Integration**: Mapping of GNN models to PyMDP, RxInfer.jl, ActiveInference.jl, DisCoPy, JAX
- **Active Inference Concepts**: Free Energy Principle, perception-action loops, belief updating
- **Multiple Output Formats**: SVG, PNG, PDF with professional themes
- **Layout Engines**: Dagre (fast), ELK (quality), TALA (advanced)

See [D2_README.md](../../../src/gnn/advanced_visualization/D2_README.md) for comprehensive D2 integration documentation.

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
  - When `True`: Generates Plotly-based interactive visualizations
  - When `False`: Generates static matplotlib visualizations

#### Export Formats

- `export_formats` (List[str]): Formats to export (default: `["html", "json"]`)
  - Supported: `["html", "json", "png", "svg", "pdf"]`
  - D2 diagrams support: `["svg", "png", "pdf"]`

#### D2 Configuration

- `d2_layout_engine` (str): Layout engine for D2 diagrams (default: `"dagre"`)
  - Options: `"dagre"` (fast), `"elk"` (quality), `"tala"` (advanced)
- `d2_theme` (str): Theme for D2 diagrams (default: `"default"`)
  - Options: `"default"`, `"dark"`, `"light"`, `"professional"`

#### Performance Tuning

- `max_nodes` (int): Maximum nodes for 3D visualization (default: `1000`)
- `simplify_large_models` (bool): Simplify large models automatically (default: `True`)
- `enable_animations` (bool): Enable animated visualizations (default: `False`)

---

## Dependencies

### Required Dependencies

- `matplotlib` - Basic plotting
- `numpy` - Numerical operations

### Optional Dependencies

- `plotly` - Interactive visualizations (recovery: static plots)
- `seaborn` - Enhanced statistical plots (recovery: matplotlib)
- `bokeh` - Interactive dashboards (recovery: HTML report)
- **`d2` CLI** - D2 diagram compilation (recovery: skip D2 diagrams, log warning)

---

## Usage Examples

### Basic Usage

```python
from advanced_visualization.processor import process_advanced_viz

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
from advanced_visualization.d2_visualizer import D2Visualizer

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

- `{model}_3d_visualization.png` - 3D plot (interactive form is `{model}_dashboard.html`)
- `{model}_dashboard.html` - Interactive dashboard
- `{model}_statistical_analysis.png` - Statistical plots
- `d2_diagrams/{model}/` - **D2 diagram files (.d2, .svg, .png)**
- `d2_diagrams/pipeline/` - **Pipeline architecture D2 diagrams**
- `advanced_viz_summary.json` - Processing summary

### Output Directory Structure

```
output/9_advanced_viz_output/
├── model_name_3d_visualization.png
├── model_name_dashboard.html
├── model_name_statistical_analysis.png
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

### Expected Performance

- **Fast Path**: ~1-2s for basic visualizations (3D, statistical)
- **Slow Path**: ~2-5s for comprehensive analysis with multiple formats
- **Memory**: ~50-100MB for large models

---

## Error Handling

### Graceful Degradation

- **No Plotly**: Generate matplotlib-based 3D visualizations ✅
- **No Bokeh**: Create static HTML reports ✅
- **Large Models**: Simplify visualization, provide warnings ✅
- **Parsing Failures**: Return structured error information ✅
- **Missing Dependencies**: Use available libraries with fallbacks ✅

### Robust Error Recovery

- **Data Loading**: Multiple recovery paths for finding GNN models
- **Visualization Generation**: Individual method error isolation
- **File I/O**: Safe file operations with proper cleanup
- **Memory Management**: Proper resource cleanup and monitoring

---

## Recent Improvements

### Comprehensive Module Enhancement ✅

**Major Fixes Applied**:

1. **Data Loading**: Fixed GNN model discovery and loading from processing results
2. **Visualization Implementation**: Uses real matplotlib-based visualizations
3. **Import Structure**: Corrected module imports and dependencies
4. **Error Handling**: Added comprehensive error handling and recovery mechanisms
5. **Test Coverage**: Created 17 comprehensive tests covering all functionality

**Key Improvements**:

- ✅ Real 3D scatter plots with variable type color coding
- ✅ Statistical analysis with pie charts, bar charts, and model metrics
- ✅ Proper data extraction with graceful error handling
- ✅ HTML dashboard generation with interactive components
- ✅ Performance optimization with matplotlib backend configuration

### Latest Major Enhancement (October 13, 2025)

**Expanded from 2 to 8 comprehensive visualization types:**

1. **3D Visualization** - Network topology in 3D space with semantic positioning and real connections
2. **Statistical Analysis** - POMDP-specific statistical analysis with real data and metrics
3. **State Transitions** - Conceptual state transition diagrams with real POMDP relationships
4. **Belief Evolution** - Belief state evolution over time, free energy landscape, policy confidence
5. **Policy Visualization** - Policy distribution, expected free energy analysis, policy convergence
6. **Matrix Correlations** - Matrix size comparison, correlation heatmaps, matrix type distribution
7. **Timeline Visualization** - POMDP model development timeline, computational complexity evolution
8. **State Space Analysis** - Comprehensive state space connectivity and manifold analysis
9. **Belief Flow Visualization** - Information flow diagrams and belief update process visualization

---

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
- **Bokeh**: Optional integration for advanced dashboards

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

- `tests/advanced_visualization/test_advanced_visualization_overall.py` ✅
- `tests/api/test_comprehensive_api.py` (integration tests)

### Test Coverage

- Measure: `uv run --extra dev python -m pytest tests/advanced_visualization/ --cov=gnn.advanced_visualization --cov-report=term-missing` (do not treat fixed percentages or per-run test tallies in this doc as canonical).

---

## MCP Integration

### Tools Registered

- `process_advanced_visualization` - Process advanced visualizations
- `check_visualization_capabilities` - Check visualization capabilities
- `list_d2_visualization_types` - List D2 visualization types
- `get_advanced_visualization_module_info` - Get module info

### Tool Endpoints

```python
@mcp_tool("process_advanced_visualization")
def process_advanced_visualization_mcp(
    target_directory: str, output_directory: str, viz_type: str = "all"
) -> Dict[str, Any]:
    """Process advanced visualizations"""
    # Implementation
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
uv run python src/gnn/9_advanced_viz.py --target-dir input/ --verbose
```

**Solutions**:

- Generate specific visualization types instead of "all"
- Disable interactive features if not needed
- Process files individually instead of batch

---

## Version History

### Current module status

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
- **Future**: Real-time visualization streaming

---

## References

### Related Documentation

- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [Visualization Module](../../../src/gnn/advanced_visualization/../visualization/AGENTS.md)
- [D2 Documentation](../../../docs/d2/)

### External Resources

- [D2 Language](https://d2lang.com)
- [Plotly Documentation](https://plotly.com/python/)
- [Active Inference Ontology](https://activeinference.org)

---

**Last Updated**: 2026-01-21
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern


---
## Documentation
- **[README](../../../src/gnn/advanced_visualization/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/advanced_visualization/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/advanced_visualization/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/advanced_visualization/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/advanced_visualization](../../../src/gnn/advanced_visualization)
