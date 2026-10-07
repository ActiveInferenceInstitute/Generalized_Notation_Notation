# GNN Visualization Guide

**Version**: v3.2.0 Engine (Bundle v2.0.0)  
**Last Updated**: 2026-04-15  
**Status**: ✅ Production Ready  
**Modules**: 38+ · **Pipeline steps**: 25 · **Renderers**: registry-defined backends (see [../implementations/README.md](../implementations/README.md)) · **Tests**: see [../../../README.md](../../../README.md)

This document provides comprehensive guidance on visualization capabilities for Generalized Notation Notation (GNN) models, including graph visualization, matrix visualization, and interactive plotting techniques.

## Overview

Visualization is a critical component of the GNN ecosystem, enabling researchers and developers to understand, analyze, and communicate Active Inference models through graphical representations. The GNN visualization system provides multiple visualization modalities:

- **Graph Visualization**: Network topology and structure representation
- **Matrix Visualization**: Matrix heatmaps and structure analysis
- **Interactive Plotting**: Dynamic, explorable visualizations
- **Advanced Visualization**: Interactive dashboards and 3D visualizations

## Visualization Pipeline Integration

GNN visualization is integrated into the 25-step processing pipeline:

**Step 8: Visualization** (`8_visualization.py`)

- Basic graph and matrix visualization
- Static visualization generation
- See: [src/gnn/visualization/AGENTS.md](../../../src/gnn/visualization/AGENTS.md)

**Step 9: Advanced Visualization** (`9_advanced_viz.py`)

- Interactive dashboards
- 3D visualizations
- Real-time visualization updates
- See: [src/gnn/advanced_visualization/AGENTS.md](../../../src/gnn/advanced_visualization/AGENTS.md)

**Quick Start:**

```bash
# Generate visualizations
uv run python src/gnn/main.py --only-steps "3,8,9" --target-dir input/gnn_files --verbose

# Step 8 has no type-selection flag — it generates the full set.
# Step 9 does: use --viz-type to pick one advanced visualization.
uv run python src/gnn/8_visualization.py --target-dir input/gnn_files --output-dir output --verbose
uv run python src/gnn/9_advanced_viz.py --target-dir input/gnn_files --output-dir output --viz-type network
```

## Graph Visualization

Graph visualization represents GNN models as networks, showing relationships between model components.

### Graph Structure

GNN models can be visualized as graphs with:

- **Nodes**: Representing states, observations, or model components
- **Edges**: Representing transitions, connections, or dependencies
- **Attributes**: Node and edge attributes (probabilities, weights, labels)

### Graph Layout Algorithms

The visualization system supports multiple layout algorithms:

```mermaid
graph TD
    GNN[GNN Model] --> PARSE[Parse Model]
    PARSE --> GRAPH[Extract Graph]
    GRAPH --> LAYOUT[Layout Algorithm]
    
    LAYOUT --> FORCE[Force-Directed]
    LAYOUT --> HIER[Hierarchical]
    LAYOUT --> CIRC[Circular]
    LAYOUT --> SPRING[Spring]
    
    FORCE --> RENDER[Render Graph]
    HIER --> RENDER
    CIRC --> RENDER
    SPRING --> RENDER
    
    RENDER --> OUTPUT[Output Format]
```

**Layout Types:**

- **Force-Directed**: Natural, organic layouts using force simulation
- **Hierarchical**: Tree-like structures for hierarchical models
- **Circular**: Circular arrangements for symmetric structures
- **Spring**: Spring-based layouts for balanced positioning

### Graph Visualization Example

```python
from visualization.processor import process_visualization
from pathlib import Path

# Generate graph visualizations for all GNN files
success = process_visualization(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/8_visualization_output"),
    verbose=True,
)
# Generates: network graphs (PNG), matrix heatmaps (PNG), statistics reports (JSON)
```

## Matrix Visualization

Matrix visualization provides heatmap representations of GNN matrices (A, B, C, D matrices).

### Matrix Types

GNN models contain several key matrices:

- **A Matrix**: Observation likelihood (observation × state)
- **B Matrix**: Transition probabilities (state × state)
- **C Matrix**: Prior preferences (state)
- **D Matrix**: Prior beliefs (state)

### Matrix Heatmap Features

Matrix visualizations include:

- **Color-Coded Values**: Visual representation of matrix values
- **Scale and Legend**: Value range indication
- **Annotations**: Labels and value annotations
- **Comparison**: Multi-matrix comparison views

### Matrix Visualization Example

```python
from visualization.processor import process_single_gnn_file
from pathlib import Path

# Process visualization for a single GNN file
generated_files = process_single_gnn_file(
    gnn_file=Path("input/gnn_files/model.md"),
    results_dir=Path("output/8_visualization_output/model/"),
    verbose=True,
)
# generated_files: list of PNG/HTML paths for network graphs and matrix heatmaps
```

## Interactive Plotting

Interactive plotting enables dynamic exploration of GNN models with zoom, pan, and hover capabilities.

### Interactive Features

- **Zoom and Pan**: Navigate large visualizations
- **Hover Information**: Display detailed information on hover
- **Click Interactions**: Interactive element selection
- **Dynamic Updates**: Real-time visualization updates
- **Export Functionality**: Export interactive visualizations

### Interactive Plotting Example

```python
from visualization.network_visualizations import generate_network_visualizations
from visualization.processor import parse_gnn_content
from pathlib import Path

# Parse GNN content and generate network graph visualizations
with open("input/gnn_files/discrete/actinf_pomdp_agent.md") as f:
    content = f.read()
parsed = parse_gnn_content(content)
files = generate_network_visualizations(
    parsed_data=parsed, output_dir=Path("output/network_viz/"), model_name="my_model"
)
# Generates force-directed network graph with node types and edge styles
```

## Advanced Visualization

Advanced visualization provides comprehensive dashboards and 3D visualizations.

### Interactive Dashboards

Interactive dashboards combine multiple visualization types:

- **Comprehensive Analysis**: Multiple visualization views
- **Navigation Controls**: Dashboard navigation
- **Filtering Options**: Data filtering and selection
- **Real-Time Updates**: Dynamic dashboard updates

### 3D Visualization

3D visualization enables spatial representation of models:

- **3D Graph Visualization**: Three-dimensional network visualization
- **3D Matrix Visualization**: Three-dimensional matrix representation
- **Spatial Analysis**: Spatial model analysis
- **Interactive 3D Navigation**: Interactive 3D exploration

### Advanced Visualization Example

```python
from pathlib import Path

from advanced_visualization import process_advanced_viz

# Generate advanced visualizations
success = process_advanced_viz(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/9_advanced_viz_output"),
    viz_type="all",  # or: 3d, interactive, dashboard, d2, diagrams,
    #     pipeline, statistical, pomdp, network
    interactive=True,
    export_formats=None,  # defaults chosen per viz_type
)
```

The same selection is available from the pipeline as `--viz-type` on step 9.

## Analysis Visualization (Step 16)

Post-simulation analysis (Step 16) provides a suite of visualization functions in the `src/gnn/analysis/` module (primarily `visualizations.py`, re-exported from `post_simulation.py`) that generate framework-specific and cross-framework visualizations from execution results.

### Per-Framework Visualization Functions

| Function | Description | Output |
|----------|-------------|--------|
| `plot_belief_evolution(beliefs, output_path, title, true_states)` | Plot belief distributions over time, optionally overlaid with true states | PNG |
| `animate_belief_evolution(beliefs, output_path, title)` | Create animated GIF of belief state evolution | GIF |
| `generate_belief_heatmaps(beliefs, output_path, title)` | Heatmap visualization of belief state evolution | PNG |
| `generate_action_analysis(actions, output_path, title)` | Action selection histograms and temporal patterns | PNG |
| `generate_free_energy_plots(free_energy, output_path, title)` | Free energy dynamics with moving average and convergence analysis | PNG |
| `generate_observation_analysis(observations, output_path, title)` | Observation patterns, distribution, and autocorrelation | PNG |

### Cross-Framework Visualization Functions

| Function | Description | Output |
|----------|-------------|--------|
| `visualize_all_framework_outputs(execution_dir, output_dir, logger)` | Generate complete visualization suite for all frameworks | Multiple PNGs |
| `generate_unified_framework_dashboard(framework_data, output_dir, model_name)` | Multi-panel comparison dashboard: beliefs, actions, free energy, observations across all frameworks | PNG |
| `generate_cross_framework_comparison(framework_data, output_path)` | Side-by-side comparison of key metrics across frameworks | PNG |

### Analysis Visualization Example

```python
from analysis.post_simulation import (
    generate_belief_heatmaps,
    generate_free_energy_plots,
    generate_unified_framework_dashboard,
)
from pathlib import Path

# Per-framework visualization
generate_belief_heatmaps(
    beliefs=[[0.5, 0.3, 0.2], [0.6, 0.2, 0.2], [0.8, 0.1, 0.1]],
    output_path=Path("output/16_analysis_output/pymdp/belief_heatmap.png"),
    title="PyMDP Belief Evolution",
)

generate_free_energy_plots(
    free_energy=[10.5, 8.3, 6.1, 4.2, 3.8, 3.5],
    output_path=Path("output/16_analysis_output/pymdp/free_energy.png"),
    title="PyMDP Free Energy Dynamics",
)

# Cross-framework unified dashboard
framework_data = {
    "pymdp": {"beliefs": [...], "actions": [...], "free_energy": [...]},
    "activeinference_jl": {"beliefs": [...], "actions": [...], "free_energy": [...]},
}
generate_unified_framework_dashboard(
    framework_data=framework_data,
    output_dir=Path("output/16_analysis_output/cross_framework/"),
    model_name="My POMDP Model",
)
```

## Output Formats

Visualizations can be exported in multiple formats:

- **HTML**: Interactive web-based visualizations
- **PNG**: Static image format
- **SVG**: Scalable vector graphics
- **PDF**: Document format
- **Interactive HTML**: Full-featured interactive dashboards

## Integration with Pipeline

Visualization results are integrated throughout the pipeline:

1. **Core Processing** (Steps 0-9): Visualization data extraction and generation
2. **Simulation** (Steps 10-16): Execution result visualization
3. **Integration** (Steps 17-24): Visualization integration into outputs
   - Website generation (Step 20) includes visualizations
   - Report generation (Step 23) includes visualizations

## Best Practices

### Visualization Selection

- **Graph Visualization**: Use for model structure and relationships
- **Matrix Visualization**: Use for probability distributions and transitions
- **Interactive Plotting**: Use for exploration and analysis
- **Advanced Visualization**: Use for comprehensive model analysis

### Performance Considerations

- **Large Models**: Use hierarchical layouts for large models
- **Batch Processing**: Process multiple models efficiently
- **Caching**: Cache visualization results for repeated access
- **Optimization**: Optimize visualization generation for performance

## Related Documentation

- **[Visualization Module](../README.md)**: Implementation details
- **[Advanced Visualization Module](../README.md)**: Advanced visualization implementation
- **[Visualization Documentation](../README.md)**: Visualization documentation overview
- **[GNN Tools](../operations/gnn_tools.md)**: Complete GNN tools reference

## See Also

- **[Pipeline Documentation](../operations/gnn_tools.md)**: Complete pipeline guide
- **[GNN Overview](../gnn_overview.md)**: Core GNN concepts
- **[Framework Integration](framework_integration_guide.md)**: Framework-specific visualization

---

**Status**: ✅ Production Ready  
**Last Updated**: 2026-04-15  
**Version**: v3.2.0 Engine (Bundle v2.0.0)
