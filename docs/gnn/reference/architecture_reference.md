# GNN Architecture Reference

**Version**: v3.2.0 Engine (Bundle v2.0.0)  
**Last Updated**: 2026-04-15  
**Status**: Maintained
**Scope**: Pipeline architecture and extension patterns. See [framework implementations](../implementations/README.md) for current backend coverage.

**GNN Architecture Team**  
**Version**: 2.0.0  
**Status**: Maintained
**Last Updated**: 2026-04-15  

Implementation details of the thin orchestrator pattern and cross-module integration.

For complete pipeline documentation:

- **[src/gnn/AGENTS.md](../../../src/gnn/AGENTS.md)**: Master agent scaffolding and module registry
- **[src/gnn/README.md](../../../src/gnn/README.md)**: Pipeline architecture and safety patterns
- **[src/gnn/main.py](../../../src/gnn/main.py)**: Pipeline orchestrator script

## Complete 25-Step Pipeline Mapping

The GNN pipeline consists of exactly 25 steps (0-24), each following the thin orchestrator pattern:

**Core Processing (Steps 0-9)**

- `0_template.py` → `src/gnn/template/` - Pipeline initialization
- `1_setup.py` → `src/gnn/setup/` - Environment and dependency setup
- `2_tests.py` → `tests/` - Test suite execution
- `3_gnn.py` → `src/gnn/` - GNN parsing and multi-format processing
- `4_model_registry.py` → `src/gnn/model_registry/` - Model versioning
- `5_type_checker.py` → `src/gnn/type_checker/` - Type validation
- `6_validation.py` → `src/gnn/validation/` - Consistency checking
- `7_export.py` → `src/gnn/export/` - Multi-format export
- `8_visualization.py` → `src/gnn/visualization/` - Graph visualization
- `9_advanced_viz.py` → `src/gnn/advanced_visualization/` - Advanced plots

**Simulation & Analysis (Steps 10-16)**

- `10_ontology.py` → `src/gnn/ontology/` - Ontology processing
- `11_render.py` → `src/gnn/render/` - Code generation
- `12_execute.py` → `src/gnn/execute/` - Simulation execution
- `13_llm.py` → `src/gnn/llm/` - LLM analysis
- `14_ml_integration.py` → `src/gnn/ml_integration/` - ML integration
- `15_audio.py` → `src/gnn/audio/` - Audio generation
- `16_analysis.py` → `src/gnn/analysis/` - Statistical analysis

**Integration & Output (Steps 17-24)**

- `17_integration.py` → `src/gnn/integration/` - System integration
- `18_security.py` → `src/gnn/security/` - Security validation
- `19_research.py` → `src/gnn/research/` - Research tools
- `20_website.py` → `src/gnn/website/` - Website generation
- `21_mcp.py` → `src/gnn/mcp/` - MCP processing
- `22_gui.py` → `src/gnn/gui/` - GUI interface
- `23_report.py` → `src/gnn/report/` - Report generation
- `24_intelligent_analysis.py` → `src/gnn/intelligent_analysis/` - AI-enhanced analysis

For module-specific documentation, see each `src/[module]/AGENTS.md` file.

## Thin Orchestrator Pattern (Actual Implementation)

### Pattern Definition

Each numbered pipeline step (0-24) follows this structure:

- **Thin orchestrator script** handles argument parsing, logging, output management
- **Module directory** contains actual implementation logic
- **Cross-references** between steps via standardized JSON outputs

### Pattern Implementation Example

#### Step 8: Visualization (src/gnn/8_visualization.py)

```python
#!/usr/bin/env python3
"""Step 8: Visualization Processing (Thin Orchestrator)"""

import sys
from pathlib import Path

# Thin orchestrator: delegates to module
from gnn.utils.pipeline_orchestration.pipeline_template import create_standardized_pipeline_script
from visualization import process_visualization  # ← Core implementation

run_script = create_standardized_pipeline_script(
    "8_visualization.py",
    process_visualization,  # ← Delegates to module
    "Matrix and network visualization processing",
)


def main() -> int:
    return run_script()  # ← Pure orchestration
```

#### Modular Implementation Layer

Example: `src/gnn/visualization/__init__.py`

```python
# Exposes core functionality with safe imports
try:
    from .visualizer import GNNVisualizer, generate_graph_visualization
    from .matrix_visualizer import MatrixVisualizer, process_matrix_visualization
    from .processor import (
        generate_matrix_visualizations,
        generate_network_visualizations,
    )
except Exception:
    # Alternative implementations for missing dependencies
    class GNNVisualizer:
        def __init__(self, *args, **kwargs):
            self.available = False

        def generate(self, *a, **k):
            return False
```

## Cross-Module Data Flow (Actual Files)

### Step 3 → Steps 6/7/8/9: Parsed Data Transfer

```
Producer: src/gnn/processing/multi_format_processor.py
  Writes: output/3_gnn_output/gnn_processing_results.json
          output/3_gnn_output/<model>/<model>_parsed.json

Consumers (one per line):
  src/gnn/validation/workflow.py               # loads the manifest, then each parsed_model_file
  src/gnn/export/processor.py                  # loads the manifest and each parsed_model_file
  src/gnn/visualization/core/parsed_model.py   # per-model parsed JSON; markdown re-parse fallback
  src/gnn/advanced_visualization/processor.py  # manifest + parsed JSONs
```

Step 5 (type checker), Step 10 (ontology), and Step 11 (render) re-parse the target-dir
GNN files directly (type_checker/checking/core.py `_discover_gnn_files`;
ontology/processor.py; render/processor/). The previously documented claims that
5_type_checker.py reads gnn_processing_results.json and that 8_visualization.py reads
5_type_checker_output are not true of the current code and have been removed.

### Step 11 → Step 12: Generated Code Execution

```
src/gnn/11_render.py → output/11_render_output/
├── <model>/ rendered framework scripts      (per-model output)
└── render_processing_summary.json           (render manifest)

src/gnn/12_execute.py:
├── Defaults its target dir to output/11_render_output
└── Resolves scripts via the render manifest (src/gnn/execute/detection.py)
```

### Step 12/16 → Step 20 and Steps 23/24: Aggregation

- src/gnn/website/generator.py loads Step 16 analysis JSONs, the Step 12 execution summary, and Step 8/9 visualization assets.
- src/gnn/report/analyzer.py reads output/00_pipeline_summary/pipeline_execution_summary.json and censuses all 25 step output dirs (viz catalog from steps 8/9/11/12/20).
- src/gnn/intelligent_analysis/processor.py reads only pipeline_execution_summary.json.

## Module Structure Analysis (Real Locations)

### Parsing Modules (src/gnn/)

```
Core parsing functionality distribution:

src/gnn/
├── multi_format_processor.py          # Main processor (called by 3_gnn.py)
├── schema_validator.py                 # Regex patterns (line 58-63)
│   ├── SECTION_PATTERN
│   ├── VARIABLE_PATTERN  
│   ├── CONNECTION_PATTERN
│   └── PARAMETER_PATTERN
├── parser.py                          # GNNParsingSystem (line 72)
│   ├── _detect_format() (line 107)
│   └── _basic_parser() (line 120)
└── parsers/                           # Format-specific parsers
    ├── markdown_parser.py             # MarkdownGNNParser 
    ├── python_parser.py               # PythonGNNParser (line 25)
    ├── lean_parser.py                 # LeanGNNParser
    └── unified_parser.py              # UnifiedGNNParser
```

### Type Analysis Modules (src/gnn/type_checker/)

```
Type analysis implementation:

src/gnn/type_checker/
├── analysis_utils.py                  # Standalone analysis helpers
│   ├── analyze_variable_types() (line 13)
│   ├── analyze_connections() (line 78)
│   └── estimate_computational_complexity() (line 131)
├── checking/
│   ├── core.py                        # GNNTypeChecker (line 111)
│   │   ├── check_file() (line 118)
│   │   ├── validate_gnn_files() (line 199)
│   │   └── _analyze_types() (line 320)
│   ├── dimensions.py                  # Dimension consistency checks
│   └── rules.py                       # Rule definitions
├── estimation/                        # GNNResourceEstimator lives here
└── processor.py                       # Thin re-export facade (17 lines)
```

### Visualization Modules (src/gnn/visualization/)

```
Visualization implementation hierarchy:

src/gnn/visualization/
├── __init__.py                        # Safe imports
├── visualizer.py                      # GNNVisualizer (line 61)
├── matrix_visualizer.py               # Thin re-export facade (15 lines)
├── processor.py                       # Thin re-export facade (34 lines)
├── matrix/
│   ├── visualizer.py                  # MatrixVisualizer (line 171)
│   │   └── generate_matrix_visualizations() (line 1649)
│   └── compat.py                      # parse_matrix_data() (line 18)
└── graph/
    └── network_visualizations.py      # generate_network_visualizations() (line 82)
```

`matrix_visualizer.py` and `processor.py` are re-export facades kept for
import stability; the implementations moved into the `matrix/` and `graph/`
subpackages.

## Framework Integration Points (Implementation Details)

### PyMDP Integration

**Location:** `src/gnn/render/pymdp/`
**Template Variables:**

```python
# Matrix extraction from parsed GNN
A = extract_matrix(gnn_data, "A")  # Likelihood
B = extract_matrix(gnn_data, "B")  # Transitions
C = extract_vector(gnn_data, "C")  # Preferences
D = extract_vector(gnn_data, "D")  # Prior
E = extract_vector(gnn_data, "E")  # Habit

# PyMDP agent construction
agent = pymdp.Agent(A=A, B=B, C=C, D=D, E=E)
```

### RxInfer.jl Integration  

**Location:** `src/gnn/render/rxinfer/`
**Model Template:**

```julia
@model function gnn_model()
    # Extract GNN parameters
    A ~ MatrixDirichlet({{gnn_A_prior}})
    B ~ MatrixDirichlet({{gnn_B_prior}})
    
    # State evolution  
    s[1] ~ Categorical({{gnn_D}})
    for t in 2:T
        s[t] ~ Categorical(B[:, s[t-1], u[t-1]])
    end
    
    # Observation model
    for t in 1:T  
        o[t] ~ Categorical(A[:, s[t]])
    end
end
```

### DisCoPy Integration

**Location:** `src/gnn/render/discopy/`  
**Category Theory Mapping:**

```python
# GNN connections → DisCoPy morphisms
connections = gnn_data["connections"]
diagram = Id(X)  # Identity on objects

for conn in connections:
    if conn["type"] == "directed":
        # f: A → B becomes morphism
        source_obj = Object(conn["source"])
        target_obj = Object(conn["target"])
        morphism = Arrow(
            source_obj, target_obj, name=f"{conn['source']}_to_{conn['target']}"
        )
        diagram = diagram >> morphism
```

## Pipeline Orchestration Details

### Argument Flow (src/gnn/main.py → steps)

```python
# src/gnn/main.py → src/gnn/pipeline/execution.py
def execute_pipeline_step(script_name: str, args: PipelineArguments, logger):
    cmd = build_step_command_args(  # src/gnn/utils/arguments/arg_parsing.py:991
        script_name.replace(".py", ""),
        args,  # target_dir, output_dir, verbose, ...
        python_executable,
        script_path,
    )
```

Each step is invoked as a subprocess and receives the common flags
(`--target-dir`, `--output-dir`, `--verbose`) plus any step-specific arguments.

### Standardized I/O and State Management

Step scripts do not wire up logging or output paths by hand. They are built by
`create_standardized_pipeline_script` (`src/gnn/utils/pipeline_orchestration/pipeline_template.py`), which
supplies the logger and the resolved output directory:

```python
from gnn.utils.pipeline_orchestration.pipeline_template import create_standardized_pipeline_script

run = create_standardized_pipeline_script(
    "3_gnn.py",
    process_gnn_multi_format,
    "GNN file discovery and multi-format parsing",
)
```

Output directories are resolved by
`get_output_dir_for_script` (`src/gnn/pipeline/config.py:136`), which maps each
script to a numbered sibling directory:

```text
output/
├── 3_gnn_output/
├── 5_type_checker_output/
├── 8_visualization_output/
└── 11_render_output/
```

### Error Handling Pattern

The template wrapper catches exceptions from the module function, logs them,
and converts the outcome into the pipeline exit-code contract
(`0` success, `1` error, `2` success with warnings):

```python
try:
    success = module_processing_function(target_dir, output_dir, logger, **kwargs)
except Exception as e:
    logger.error(f"Critical error in processing: {e}")
    return 1
return 0 if success else 1
```

## Dependency Resolution (Actual Implementation)

### Safe Import Pattern (src/gnn/visualization/**init**.py:15-47)

```python
# Pattern used across modules for optional dependencies:
try:
    from .matrix_visualizer import MatrixVisualizer, process_matrix_visualization
except Exception:
    MatrixVisualizer = None
    process_matrix_visualization = None

# Allows graceful degradation when dependencies missing
if MatrixVisualizer is None:
    logger.warning("Matrix visualization unavailable - matplotlib missing")
    return create_alternative_html_report()
```

### MCP Integration Pattern

Each module includes `mcp.py` with tool registration:

```python
# Example: src/gnn/visualization/mcp.py
@server.tool()
def visualize_gnn_model(content: str, output_path: str) -> dict:
    """Generate visualization for GNN model content."""
    return process_visualization(target_dir=Path(content), output_dir=Path(output_path))
```

This architecture enables modular development, safe-to-fail operation, and framework interoperability while maintaining clear separation between orchestration and implementation logic.
