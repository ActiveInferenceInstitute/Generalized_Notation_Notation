# Step 8: Visualization

## Architectural Mapping

**Orchestrator**: `src/gnn/8_visualization.py` (55 lines)
**Implementation Layer**: `src/gnn/visualization/`

## Module Description

This module provides comprehensive visualization capabilities for GNN models, including graph visualization, matrix visualization, and interactive plotting.


```
src/gnn/visualization/
├── __init__.py                 # Public exports (MatrixVisualizer, process_visualization, …)
├── processor.py                # Facade: core + parse + plotting re-exports
├── core/                       # process.py, parsed_model.py (JSON-first loader); [README](../../../src/gnn/visualization/core/README.md)
├── parse/                      # markdown.py, gnn_file_parser.py (GNNParser); [README](../../../src/gnn/visualization/parse/README.md)
├── plotting/                   # utils.py (Agg, save_plot_safely); [README](../../../src/gnn/visualization/plotting/README.md)
├── graph/                      # network_visualizations.py, bipartite.py; [README](../../../src/gnn/visualization/graph/README.md)
├── matrix/                     # visualizer.py, extract.py, compat.py; [README](../../../src/gnn/visualization/matrix/README.md)
├── analysis/                   # combined_analysis.py; [README](../../../src/gnn/visualization/analysis/README.md)
├── ontology/                   # visualizer.py; [README](../../../src/gnn/visualization/ontology/README.md)

## Agent Identity & Capabilities

# Visualization Module - Agent Scaffolding

## Module Overview

**Purpose**: Graph and matrix visualization generation for GNN models

**Pipeline Step**: Step 8: Visualization (8_visualization.py)

**Category**: Visualization / Graph Analysis

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-04-15

---

## Core Functionality

### Primary Responsibilities
1. Generate graph visualizations from GNN models
2. Create matrix heatmaps and plots
3. Visualize model structure and connections
4. Generate network topology diagrams
5. Provide visualization data for advanced analysis

### Key Capabilities
- Network graph generation and layout
- Matrix visualization and heatmap creation
- Interactive visualization support
- Multiple output formats (PNG, SVG, HTML)
- Model structure visualization

---

## API Reference

### Public Functions

#### `process_visualization(target_dir, output_dir, verbose=False, **kwargs) -> bool`
**Description**: Main visualization processing function called by orchestrator ([8_visualization.py](../../../src/gnn/visualization/../8_visualization.py)). Implementation: [core/process.py](../../../src/gnn/visualization/core/process.py).

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for visualizations
- `verbose` (bool): Enable verbose logging (default: False)
- `**kwargs`: Additional visualization options

**Returns**: `True` if at least one artifact was generated

**Data loading**: [core/parsed_model.py](../../../src/gnn/visualization/core/parsed_model.py) `load_visualization_model` prefers `{model}_parsed.json` from step 3; when structured JSON is unavailable, [parse/markdown.py](../../../src/gnn/visualization/parse/markdown.py) `parse_gnn_content` provides the explicit raw-Markdown parser path.

**Example**:
```python
from visualization import process_visualization

success = process_visualization(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/8_visualization_output"),
    verbose=True
)
```

#### `generate_graph_visualization(graph_data, output_dir=None) -> List[str]`
**Description**: Module-level helper; delegates to [`GNNVisualizer`](../../../src/gnn/visualization/visualizer.py).

**Parameters**:
- `graph_data`: Graph data dictionary
- `output_dir`: Optional output directory

**Returns**: List of generated visualization file paths

#### `generate_matrix_visualization(matrix_data, output_dir=None) -> List[str]`
**Description**: Module-level helper; delegates to [`GNNVisualizer`](../../../src/gnn/visualization/visualizer.py).

**Parameters**:
- `matrix_data`: Matrix data dictionary
- `output_dir`: Optional output directory

**Returns**: List of generated visualization file paths

#### `GNNVisualizer.create_network_diagram(graph_data) -> Dict[str, Any]`
**Description**: Instance method on [`GNNVisualizer`](../../../src/gnn/visualization/visualizer.py), not a package-level function. Use `GNNVisualizer(...).create_network_diagram(graph_data)`.

**Returns**: Dictionary with visualization metadata / paths

---

## Dependencies

### Required Dependencies
- `matplotlib` - Plotting and visualization
- `networkx` - Network graph algorithms
- `numpy` - Numerical computations

### Optional Dependencies
- `plotly` - Interactive visualizations
- `graphviz` - Graph layout and rendering

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

---

## Configuration

### Visualization Settings
```python
VISUALIZATION_CONFIG = {
    "output_format": "png",
    "dpi": 300,
    "figsize": (10, 8),
    "colormap": "viridis",
    "layout_algorithm": "spring",
}
```

### Graph Settings
```python
GRAPH_CONFIG = {
    "node_size": 100,
    "edge_width": 1,
    "node_color": "lightblue",
    "edge_color": "gray",
    "layout": "force_directed",
}
```

---

## Usage Examples

### Basic Visualization
```python
from visualization import process_visualization

success = process_visualization(
    target_dir="input/gnn_files", output_dir="output/8_visualization_output"
)
```

### Graph Visualization
```python
from visualization import generate_graph_visualization

files = generate_graph_visualization(graph_data)
for file_path in files:
    print(f"Generated: {file_path}")
```

### Matrix Visualization
```python
from visualization import generate_matrix_visualization

files = generate_matrix_visualization(matrix_data)
for file_path in files:
    print(f"Generated: {file_path}")
```

---

## Output Specification

### Output Products
- `{model}_network_graph.png` — Network layout (directed vs undirected edges, ontology labels)
- `{model}_network_stats.json` — Counts, `gnn_edge_orientation`, optional `network_properties`
- `{model}_variable_parameter_bipartite.png` — Variables vs parameter tensors (name matches)
- `{model}_*_heatmap.png` / `*_tensor.png` / `*_analysis.png` — Matrix / POMDP outputs
- `{model}_combined_analysis.png`, `{model}_generative_model.png`, standalone panels
- `{model}_viz_manifest.json` — Artifact paths, `_viz_meta` (JSON vs markdown source), counts
- `{model}_viz_source_note.txt` — When step-3 JSON is older than source `.md`
- `visualization_summary.json` — Run-level summary (all models)

### Output Directory Structure
```
output/8_visualization_output/
├── visualization_summary.json
└── {model}/
    ├── {model}_network_graph.png
    ├── {model}_network_stats.json
    ├── {model}_viz_manifest.json
    ├── {model}_combined_analysis.png
    └── …
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~2-5 seconds per model
- **Memory**: ~50-150MB
- **Status**: ✅ Production Ready

### Expected Performance
- **Graph Generation**: 1-3 seconds
- **Matrix Visualization**: 1-2 seconds
- **Structure Analysis**: 2-4 seconds
- **Combined Visualization**: 3-6 seconds

---

## Error Handling

### Visualization Errors
1. **Graph Layout**: Graph layout algorithm failures
2. **Matrix Size**: Matrix too large for visualization
3. **File I/O**: Visualization file writing failures
4. **Dependency**: Missing visualization dependencies

### Recovery Strategies
- **Layout Recovery**: Use simpler layout algorithms
- **Matrix Sampling**: Sample large matrices
- **Format Recovery**: Try alternative output formats
- **Dependency Skip**: Skip advanced visualizations

---

## Integration Points

### Orchestrated By
- **Script**: `8_visualization.py` (Step 8)
- **Function**: `process_visualization()` ([core/process.py](../../../src/gnn/visualization/core/process.py))

### Imports From
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

### Imported By
- `advanced_visualization` - Advanced visualization module
- `tests.test_visualization_*` - Visualization tests

### Data Flow
```
GNN Files → Graph Extraction → Layout Calculation → Visualization Generation → Output Files
```

---

## Testing

### Test Files
- `tests/visualization/test_visualization_matrices.py` - Matrix visualization tests
- `tests/visualization/test_visualization_comprehensive.py` - Comprehensive real-data tests
- `tests/visualization/test_visualization_overall.py` - Module-level tests
- `tests/visualization/test_visualization_ontology.py` - Ontology visualization tests
- `tests/visualization/test_visualization_artifacts.py` - Artifact / manifest tests

### Test Coverage
- **Measurement**: `uv run --extra dev python -m pytest tests/visualization/ --cov=gnn.visualization --cov-report=term-missing` (do not treat a fixed percentage in this file as canonical).

### Key Test Scenarios
1. Graph visualization with various layouts
2. Matrix heatmap generation
3. Model structure visualization
4. Error handling and recovery
5. Matplotlib backend configuration
6. Headless environment support
7. Progress tracking validation

---

## MCP Integration

Registration lives in [`mcp.py`](../../../src/gnn/visualization/mcp.py) via `register_tools(mcp_instance)` (GNN MCP server `register_tool` API).

### Tools registered (names match server tool IDs)

| Tool name | Python handler | Purpose |
|-----------|----------------|---------|
| `process_visualization` | `process_visualization_mcp` | Run full step-8 batch for a directory |
| `get_visualization_options` | `get_visualization_options_mcp` | Return `get_visualization_options()` dict |
| `list_visualization_artifacts` | `list_visualization_artifacts_mcp` | List PNG/SVG/HTML/PDF under an output dir |
| `get_visualization_module_info` | `get_visualization_module_info_mcp` | Return `get_module_info()` metadata |

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Matplotlib Backend Warnings
**Symptom**: Warnings about matplotlib backend or "no DISPLAY" errors

**Solution**:
- ✅ **Automatic Fix**: The module now automatically detects headless environments and configures the `Agg` backend
- Environment variable: Set `MPLBACKEND=Agg` before running
- Manual fix: Add to your script:
  ```python
  import matplotlib

  matplotlib.use("Agg")
  ```

**Prevention**: Run in environments with display support or ensure `Agg` backend is used

#### 2. Missing Visualization Dependencies
**Symptom**: ImportError for matplotlib, networkx, or numpy

**Solution**:
```bash
# Using UV (recommended)
uv pip install matplotlib>=3.5.0 networkx>=2.8.0 numpy>=1.21.0

# Or install all dependencies via pyproject.toml
uv sync
```

**Alternative**: Install visualization optional group:
```bash
uv sync
```

#### 3. Large Model Visualization Failures
**Symptom**: Visualization fails or hangs with large models (>100 nodes)

**Solution**:
- ✅ **Automatic**: Module samples large models automatically
- Manual override: Set sampling parameters in config
- Alternative: Visualize model subsets

**Prevention**: No dedicated CLI flag exists for this; sampling for large models is automatic within `src/gnn/visualization/`, not user-configurable via `src/gnn/8_visualization.py` arguments

#### 4. Memory Issues During Visualization
**Symptom**: Out of memory errors or system slowdown

**Solution**:
- Reduce visualization DPI: Set `DPI=150` (default: 300)
- Process files individually instead of batch
- Increase system memory or use sampling

**Prevention**: Monitor memory usage with `--verbose` flag

#### 5. No Visualizations Generated
**Symptom**: Step completes successfully but no images created

**Diagnostic**:
```bash
# Check if GNN processing (step 3) completed successfully
ls output/3_gnn_output/

# Run visualization with verbose logging
uv run python src/gnn/8_visualization.py --verbose --target-dir input/gnn_files --output-dir output
```

**Common Causes**:
- GNN processing (step 3) not run first
- Empty or invalid GNN files
- Missing parsed model files

**Solution**:
```bash
# Run complete pipeline in order
uv run python src/gnn/main.py --only-steps "3,8" --verbose
```

#### 6. Visualization Quality Issues
**Symptom**: Blurry or pixelated visualizations

**Solution**:
- Increase DPI in configuration (default: 300)
- Use vector formats (SVG) instead of PNG
- Adjust figure size in config

**Configuration**:
```python
VISUALIZATION_CONFIG = {
    "dpi": 600,  # Higher quality
    "format": "svg",  # Vector format
    "figsize": (12, 10),  # Larger canvas
}
```

#### 7. Progress Tracking Not Visible
**Symptom**: No progress updates during long-running visualizations

**Solution**:
```bash
# Enable verbose mode for detailed progress
uv run python src/gnn/8_visualization.py --verbose --target-dir input/gnn_files
```

**Features**:
- ✅ File-by-file progress indicators: `[1/5]`, `[2/5]`, etc.
- ✅ Visualization type completion: Matrix ✅, Network ✅, Combined ✅
- ✅ Detailed step logging with emoji indicators 📊

### Performance Optimization

#### Fast Visualization Tips
1. **Use appropriate DPI**: 150 for preview, 300 for publication
2. **Sample large models**: Automatic sampling for >100 nodes
3. **Parallel processing**: Process multiple files independently
4. **Cache results**: Reuse visualizations when possible

#### Resource Management
- **Memory**: ~50-150MB per model (typical)
- **CPU**: 1-2 cores per visualization process
- **Disk**: ~1-5MB per visualization set
- **Time**: 1-5 seconds per model (typical)

### Best Practices

1. **Always run GNN processing (step 3) first**:
   ```bash
   uv run python src/gnn/3_gnn.py --target-dir input/gnn_files
   uv run python src/gnn/8_visualization.py --target-dir input/gnn_files
   ```

2. **Use verbose mode for debugging**:
   ```bash
   uv run python src/gnn/8_visualization.py --verbose
   ```

3. **Check output directory structure**:
   ```
   output/8_visualization_output/
   ├── model_name/
   │   ├── matrix_analysis.png
   │   ├── matrix_statistics.png
   │   └── model_name_combined_analysis.png
   └── visualization_results.json
   ```

4. **Monitor for warnings**:
   - Backend configuration warnings
   - Dependency availability warnings
   - Sampling notifications for large models

---

## Version History

### Current module status

**Features**:
- Graph visualization generation
- Matrix heatmap creation
- Network topology diagrams
- Model structure visualization
- Automatic headless environment detection
- Progress tracking with visual indicators

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Interactive visualizations (plotly/HTML where optional deps exist)
- **Future**: Streaming or incremental updates for large models

---

## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [Advanced Visualization](../../../src/gnn/visualization/../advanced_visualization/AGENTS.md)
- [GNN Visualization Guide](../../../docs/gnn/integration/gnn_visualization.md)

### External Resources
- [Matplotlib Documentation](https://matplotlib.org/)
- [NetworkX Documentation](https://networkx.org/)
- [Plotly Documentation](https://plotly.com/python/)

---

**Last Updated**: 2026-04-15
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern

---
## Documentation
- **[README](../../../src/gnn/visualization/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/visualization/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/visualization/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/visualization/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/visualization](../../../src/gnn/visualization)
