# D2 Visualization Integration for GNN Pipeline

## Overview

This module provides comprehensive D2 (Declarative Diagramming) integration for the GNN Pipeline's advanced visualization capabilities. D2 is a modern diagramming language that transforms text into professional diagrams, making it ideal for visualizing Active Inference models, pipeline architecture, and framework integrations.

## Features

### Core D2 Visualization Capabilities

1. **GNN Model Structure Diagrams**
   - Visualize state space components
   - Show variable relationships and connections
   - Display Active Inference ontology annotations
   - Support for POMDP-specific structures

2. **POMDP/Active Inference Diagrams**
   - Generative model visualization (A, B, C, D, E matrices)
   - Inference process flow (state inference, policy selection, action sampling)
   - Belief update cycles
   - Data flow between components

3. **Pipeline Architecture Diagrams**
   - Complete 25-step pipeline flow
   - Core processing modules (steps 0-9)
   - Simulation & analysis modules (steps 10-16)
   - Integration & output modules (steps 17-23)

4. **Framework Integration Mapping**
   - PyMDP integration
   - RxInfer.jl integration
   - ActiveInference.jl integration
   - DisCoPy integration
   - JAX integration

5. **Active Inference Conceptual Diagrams**
   - Free Energy Principle illustration
   - Agent-environment interaction
   - Perception-action loops
   - Belief updating processes

## Installation

### Prerequisites

**D2 CLI Installation:**

The D2 CLI must be installed on your system for diagram compilation. Install via:

```bash
# macOS
brew install d2

# Linux/macOS (install script)
curl -fsSL https://d2lang.com/install.sh | sh -s --

# Or download from https://github.com/terrastruct/d2/releases
```

**Verify Installation:**

```bash
d2 version
```

### Python Dependencies

All Python dependencies are included in the main GNN pipeline. Install using UV:

```bash
# Install core dependencies
uv sync

# Or install with visualization extras
uv sync
```

## Usage

### Basic D2 Visualization

Generate D2 diagrams as part of the advanced visualization step:

```bash
# Generate all visualizations including D2 diagrams
python src/gnn/9_advanced_viz.py --target-dir input/gnn_files --viz-type all

# Generate only D2 diagrams
python src/gnn/9_advanced_viz.py --target-dir input/gnn_files --viz-type d2

# Generate only pipeline diagrams
python src/gnn/9_advanced_viz.py --target-dir input/gnn_files --viz-type pipeline
```

### Programmatic Usage

```python
from gnn.advanced_visualization.d2_visualizer import D2Visualizer
from pathlib import Path

# Initialize visualizer
visualizer = D2Visualizer()

# Check if D2 CLI is available
if visualizer.d2_available:
    # Load GNN model data
    model_data = {
        "model_name": "Active Inference POMDP Agent",
        "state_space": {
            "A": {"dimensions": [3, 3], "type": "float"},
            "B": {"dimensions": [3, 3, 3], "type": "float"},
            # ... more state space variables
        },
        "connections": [
            {"source": "s", "target": "A", "type": "->"},
            # ... more connections
        ],
        "actinf_annotations": {
            "A": "LikelihoodMatrix",
            "B": "TransitionMatrix",
            # ... more annotations
        },
    }

    # Generate model structure diagram
    structure_spec = visualizer.generate_model_structure_diagram(model_data)

    # Generate POMDP diagram
    pomdp_spec = visualizer.generate_pomdp_diagram(model_data)

    # Compile diagrams to SVG and PNG
    output_dir = Path("output/d2_diagrams")
    result = visualizer.compile_d2_diagram(
        structure_spec, output_dir, formats=["svg", "png"]
    )

    if result.success:
        print(f"Generated {len(result.output_files)} diagram files")
        for file in result.output_files:
            print(f"  - {file}")
```

### Generate All Diagrams for a Model

```python
from gnn.advanced_visualization.d2_visualizer import D2Visualizer
from pathlib import Path

visualizer = D2Visualizer()
output_dir = Path("output/d2_diagrams")

# Generate all applicable diagrams for a model
results = visualizer.generate_all_diagrams_for_model(
    model_data, output_dir, formats=["svg", "png", "pdf"]
)

for result in results:
    if result.success:
        print(f"✓ {result.diagram_name}: {len(result.output_files)} files")
    else:
        print(f"✗ {result.diagram_name}: {result.error_message}")
```

### Generate Pipeline Diagrams

```python
from gnn.advanced_visualization.d2_visualizer import D2Visualizer
from pathlib import Path

visualizer = D2Visualizer()
output_dir = Path("output/d2_diagrams/pipeline")

# Generate pipeline flow diagram
flow_spec = visualizer.generate_pipeline_flow_diagram(include_frameworks=True)
flow_result = visualizer.compile_d2_diagram(flow_spec, output_dir)

# Generate framework integration diagram
framework_spec = visualizer.generate_framework_mapping_diagram()
framework_result = visualizer.compile_d2_diagram(framework_spec, output_dir)

# Generate Active Inference concepts diagram
concepts_spec = visualizer.generate_active_inference_concepts_diagram()
concepts_result = visualizer.compile_d2_diagram(concepts_spec, output_dir)
```

## Output Structure

When D2 diagrams are generated, they are organized in the following structure:

```
output/9_advanced_viz_output/
├── d2_diagrams/
│   ├── model_name/
│   │   ├── model_name_structure.d2        # D2 source
│   │   ├── model_name_structure.svg       # SVG output
│   │   ├── model_name_structure.png       # PNG output
│   │   ├── model_name_pomdp.d2           # POMDP diagram source
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

## D2 Diagram Customization

### Layout Engines

D2 supports multiple layout engines, each optimized for different diagram types:

- **dagre** (default): Fast, directed graph layout
- **elk**: Superior layout quality for complex diagrams
- **tala**: Terrastruct's proprietary engine (requires subscription)

```python
spec = D2DiagramSpec(
    name="custom_diagram",
    description="Custom diagram",
    d2_content=d2_content,
    layout_engine="elk",  # Use ELK for complex layouts
    theme=1,
    dark_theme=100,
)
```

### Themes

D2 includes professionally designed themes:

```python
spec = D2DiagramSpec(
    name="themed_diagram",
    description="Themed diagram",
    d2_content=d2_content,
    theme=1,  # Theme ID (see d2 list-themes)
    dark_theme=100,  # Dark theme ID for adaptive mode
    sketch_mode=False,  # Enable sketch mode for hand-drawn look
)
```

List available themes:

```bash
d2 list-themes
```

### Output Formats

Supported output formats:

- **SVG**: Scalable vector graphics (default, best for web)
- **PNG**: Raster image (good for presentations)
- **PDF**: Vector PDF (good for documents)

```python
result = visualizer.compile_d2_diagram(spec, output_dir, formats=["svg", "png", "pdf"])
```

## API Reference

### D2Visualizer

Main class for D2 diagram generation.

**Methods:**

- `generate_model_structure_diagram(model_data, output_name=None)` → `D2DiagramSpec`
  - Generate diagram for GNN model structure
  
- `generate_pomdp_diagram(model_data, output_name=None)` → `D2DiagramSpec`
  - Generate POMDP/Active Inference structure diagram
  
- `generate_pipeline_flow_diagram(include_frameworks=True)` → `D2DiagramSpec`
  - Generate GNN pipeline architecture diagram
  
- `generate_framework_mapping_diagram(frameworks=None)` → `D2DiagramSpec`
  - Generate framework integration mapping diagram
  
- `generate_active_inference_concepts_diagram()` → `D2DiagramSpec`
  - Generate Active Inference conceptual diagram
  
- `compile_d2_diagram(spec, output_dir, formats=None)` → `D2GenerationResult`
  - Compile D2 diagram to output formats (each compile routes through the
    shared subprocess envelope: wall-clock force-kill timeout, classified
    timeout / exit / never-started failures)
  
- `generate_all_diagrams_for_model(model_data, output_dir, formats=None)` → `List[D2GenerationResult]`
  - Generate all applicable diagrams for a model

### D2DiagramSpec

Dataclass representing a D2 diagram specification.

**Fields:**

- `name: str` - Diagram name (used for output filenames)
- `description: str` - Human-readable description
- `d2_content: str` - D2 diagram source code
- `output_formats: List[str]` - Output formats (default: ["svg"])
- `layout_engine: str` - Layout engine (dagre, elk, tala)
- `theme: int` - Theme ID
- `dark_theme: Optional[int]` - Dark theme ID
- `sketch_mode: bool` - Enable sketch mode
- `pad: int` - Padding around diagram
- `metadata: Dict[str, Any]` - Additional metadata

### D2GenerationResult

Dataclass representing D2 diagram generation result.

**Fields:**

- `success: bool` - Whether generation succeeded
- `diagram_name: str` - Name of the diagram
- `d2_file: Optional[Path]` - Path to D2 source file
- `output_files: List[Path]` - List of generated output files
- `compilation_time: float` - Compilation time in seconds
- `error_message: Optional[str]` - Error message if failed
- `warnings: List[str]` - List of warnings

## Error Handling

The D2 visualizer implements comprehensive error handling with graceful degradation:

### D2 CLI Not Available

If D2 CLI is not installed, the visualizer will:

1. Log a warning message
2. Skip D2 diagram generation
3. Continue with other visualizations
4. Mark D2 attempts as "skipped" in results

```python
visualizer = D2Visualizer()

if not visualizer.d2_available:
    print("D2 CLI not available. Install from https://d2lang.com")
    # Other visualizations will still work
```

### Compilation Errors

If D2 compilation fails for a specific format:

1. Error is logged with details
2. Other formats are still attempted
3. Partial success is possible (e.g., SVG succeeds but PNG fails)

```python
result = visualizer.compile_d2_diagram(spec, output_dir, formats=["svg", "png"])

if result.success:
    print(f"Generated {len(result.output_files)} of {len(formats)} requested formats")
    
for warning in result.warnings:
    print(f"Warning: {warning}")
```

## Testing

Comprehensive tests are provided in `tests/visualization/test_d2_visualizer.py`:

```bash
# Run D2 visualizer tests
uv run --extra dev python -m pytest tests/visualization/test_d2_visualizer.py -v

# Run with coverage
uv run --extra dev python -m pytest tests/visualization/test_d2_visualizer.py --cov=src/gnn/advanced_visualization/d2_visualizer
```

**Test Coverage:**

- D2Visualizer initialization and CLI checking
- Model structure diagram generation
- POMDP diagram generation
- Pipeline diagram generation
- Framework mapping diagram generation
- Active Inference conceptual diagrams
- Diagram compilation
- Helper methods (name sanitization, shape mapping, etc.)
- Error handling and recovery mechanisms
- End-to-end workflows

## Performance

### Measurement Policy

- This guide does not define fixed D2 runtime or memory targets.
- Measure diagram generation and compilation time from a fresh run when performance matters.
- Treat D2 compilation as optional: when the D2 CLI is absent, Step 9 records the skip instead of silently claiming compiled artifacts.

### Optimization Tips

1. **Use SVG format by default**: Fastest compilation, smallest file size
2. **Batch compilations**: Generate all diagrams in one call
3. **Choose appropriate layout engine**: Dagre is fastest, ELK for quality
4. **Retain source diagrams**: D2 files can be version controlled and recompiled

## Integration with GNN Pipeline

D2 visualization is seamlessly integrated into the GNN pipeline:

### Pipeline Step 9

D2 diagram generation is part of step 9 (Advanced Visualization):

```bash
# Run full pipeline with D2 diagrams
python src/gnn/main.py --target-dir input/gnn_files

# Run only advanced visualization step
python src/gnn/main.py --only-steps "9" --target-dir input/gnn_files
```

### Step Dependencies

D2 visualization depends on:

- **Step 3**: GNN parsing (provides model data)
- **Step 5**: Type checking (validates model structure)

D2 diagrams are used by:

- **Step 20**: Website generation (embeds D2 diagrams)
- **Step 23**: Report generation (includes D2 diagrams)

## Best Practices

1. **Always generate D2 source files**: Keep .d2 files for version control and recompilation
2. **Use appropriate themes**: Match themes to documentation style
3. **Choose the right layout engine**: ELK for complex models, Dagre for simple ones
4. **Generate multiple formats**: SVG for web, PNG for presentations, PDF for documents
5. **Document custom diagrams**: Add descriptions and metadata to D2DiagramSpec
6. **Test with sample models first**: Verify D2 CLI is working before production runs

## Troubleshooting

### "D2 CLI not available" Warning

**Problem**: D2 CLI is not installed or not in PATH

**Solution**:

```bash
# Install D2 CLI
curl -fsSL https://d2lang.com/install.sh | sh -s --

# Verify installation
d2 version
```

### Compilation Timeout

**Problem**: D2 compilation takes too long (>30 seconds)

**Solution**:

- Use simpler layout engine (dagre instead of elk)
- Reduce model complexity
- Check system resources

### Layout Issues

**Problem**: Diagram layout doesn't look good

**Solution**:

- Try different layout engines (elk, dagre, tala)
- Adjust padding with `pad` parameter
- Customize D2 content with explicit positioning

### Missing Diagrams

**Problem**: No D2 diagrams generated

**Solution**:

1. Check D2 CLI availability: `d2 version`
2. Verify GNN models were parsed (step 3)
3. Check logs for specific errors
4. Run with `--viz-type d2` to focus on D2 only

## References

### D2 Documentation

- [Official D2 Documentation](https://d2lang.com)
- [D2 Language Tour](https://d2lang.com/tour/)
- [D2 GitHub Repository](https://github.com/terrastruct/d2)

### GNN Pipeline Documentation

- [Advanced Visualization Module](AGENTS.md)
- [Pipeline Architecture](../../../ARCHITECTURE.md)
- [D2 Integration Guide](../../../docs/d2/gnn_d2.md)

---

**Last Updated**: 2026-09-24  
**Module Version**: 1.0.0  
**Status**: Maintained
