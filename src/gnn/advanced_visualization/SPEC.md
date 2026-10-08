# Advanced Visualization Module Specification

## Overview
Advanced visualization generation including 3D, interactive dashboards, and D2 diagrams.

## Components

### Core
- `processor.py` - Main processor (573 lines); `process_advanced_viz` entry point
- `visualizer.py` - Visualization generation (`AdvancedVisualizer`)
- `dashboard.py` - Dashboard generation (`DashboardGenerator`)

### D2 Integration
- `d2_visualizer.py` - D2 diagram generation (`D2Visualizer`)
- `D2_README.md` - D2 documentation

### Support
- `data_extractor.py` - Data extraction utilities (`VisualizationDataExtractor`)
- `html_generator.py` - HTML report generation

## Key Exports
```python
from gnn.advanced_visualization import process_advanced_viz
```

## Saved POMDP Transition Axes

`process_advanced_viz(..., viz_type="pomdp")` consumes canonical B tensors as
`B[next_state, previous_state, action]`. Each action panel displays
`B[:, :, action]`, with previous state on the horizontal axis and next state
on the vertical axis. Passive two-dimensional B matrices use the same state
axis labels. These plots retain the supplied values and do not reorder or
rebuild the model or change policy metadata behavior.


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
