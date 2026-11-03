# D2 Integration with GNN Pipeline: Comprehensive Technical Overview

## Executive Summary

This document provides a comprehensive technical overview of how D2 (Declarative Diagramming) can be integrated with the Generalized Notation Notation (GNN) Pipeline to create professional visualizations for Active Inference model processing. The GNN Pipeline is a 25-step system that transforms GNN plaintext specifications into executable simulations, and D2 offers powerful declarative diagramming capabilities that complement this workflow by providing clear, professional visualizations of pipeline architecture, model structures, and data flows.

## GNN Pipeline Overview

The GNN Pipeline processes Active Inference generative models through a structured 25-step workflow:

### Core Processing Modules (Steps 0-9)
- **Step 3**: GNN parsing and multi-format serialization (22 formats)
- **Step 5**: Type checking and resource estimation
- **Step 6**: Advanced validation and consistency checking
- **Step 7**: Multi-format export (JSON, XML, GraphML, GEXF, Pickle)
- **Step 8**: Graph and matrix visualization generation

### Simulation & Analysis Modules (Steps 10-16)
- **Step 11**: Code generation for PyMDP, RxInfer.jl, ActiveInference.jl, DisCoPy, JAX
- **Step 12**: Execute rendered simulation scripts with result capture
- **Step 13**: LLM-enhanced analysis and interpretation

## D2 Applications for GNN Visualization

### 1. Pipeline Architecture Diagrams

D2 can create comprehensive visualizations of the GNN pipeline's modular architecture:

```d2
GNN Pipeline Architecture: {
  direction: down
  grid-rows: 3
  grid-columns: 1

  Core Processing: {
    direction: right
    template -> setup -> tests -> gnn -> model_registry -> type_checker -> validation -> export -> visualization -> advanced_viz
  }

  Simulation Analysis: {
    direction: right
    ontology -> render -> execute -> llm -> ml_integration -> audio -> analysis
  }

  Integration Output: {
    direction: right
    integration -> security -> research -> website -> mcp -> gui -> report
  }

  Core Processing -> Simulation Analysis -> Integration Output
}

Data Flow: {
  direction: right
  input_gnn_files -> gnn_parser -> multi_format_serializer -> downstream_steps
  downstream_steps -> framework_renderers -> simulation_executors -> analysis_tools
}
```

**Benefits for Pipeline Documentation:**
- Clear visualization of the 25-step workflow
- Easy identification of data dependencies
- Professional presentation for technical documentation
- Interactive SVG output with clickable elements

### 2. GNN Model Structure Visualization

D2 can visualize the internal structure of GNN specifications and their Active Inference components:

```d2
Active Inference POMDP Agent: {
  direction: down

  State Space: {
    direction: right
    hidden_states: Hidden State Factor {
      shape: cylinder
      label: "s[3] - Location states"
    }

    observations: Observation Modality {
      shape: circle
      label: "o[3] - State observations"
    }

    actions: Control Actions {
      shape: square
      label: "u[3] - Discrete actions"
    }
  }

  Generative Model: {
    direction: right

    A_matrix: Likelihood Matrix {
      shape: hexagon
      label: "A[3,3] - P(o|s)"
    }

    B_matrix: Transition Matrix {
      shape: hexagon
      label: "B[3,3,3] - P(s'|s,u)"
    }

    C_vector: Preferences {
      shape: diamond
      label: "C[3] - Log preferences"
    }

    D_vector: Prior States {
      shape: diamond
      label: "D[3] - Initial state prior"
    }

    E_vector: Habits {
      shape: diamond
      label: "E[3] - Policy prior"
    }
  }

  Inference: {
    direction: right

    state_inference: State Inference {
      label: "infer_states()"
    }

    policy_inference: Policy Selection {
      label: "infer_policies()"
    }

    action_selection: Action Sampling {
      label: "sample_action()"
    }
  }

  State Space -> Generative Model -> Inference: "Model Definition"
  Inference -> State Space: "Belief Updates"
}
```

### 3. Framework Integration Mapping

Visualize how GNN models are translated to different simulation frameworks:

```d2
Framework Integration: {
  direction: down

  gnn_model: GNN Specification {
    shape: document
  }

  render_step: Code Generation {
    direction: right

    pymdp: PyMDP {
      shape: rectangle
      label: "Python Active Inference"
    }

    rxinfer: RxInfer.jl {
      shape: rectangle
      label: "Julia Reactive Inference"
    }

    activeinference: ActiveInference.jl {
      shape: rectangle
      label: "Julia Active Inference"
    }

    discopy: DisCoPy {
      shape: rectangle
      label: "Python Categorical Diagrams"
    }

    jax: JAX {
      shape: rectangle
      label: "Python HPC Simulation"
    }
  }

  execution_step: Simulation Execution {
    direction: right
    pymdp_exec -> rxinfer_exec -> activeinference_exec -> discopy_exec -> jax_exec
  }

  gnn_model -> render_step: "GNN → Code Generation"
  render_step -> execution_step: "Generated Code → Execution"
}
```

### 4. Module Dependency Diagrams

Show the complex interdependencies between pipeline modules:

```d2
Module Dependencies: {
  direction: down

  core: Core Dependencies {
    direction: right
    utils -> pipeline -> config
  }

  processing: Processing Chain {
    direction: right
    gnn -> type_checker -> validation -> export -> visualization
    gnn -> render -> execute
    render -> llm
  }

  infrastructure: Infrastructure {
    direction: down
    setup -> tests
    security -> integration
  }

  core -> processing: "Foundation"
  infrastructure -> processing: "Support"
}
```

### 5. Data Flow and Pipeline Execution

Create detailed flow diagrams showing data transformation through the pipeline:

```d2
Pipeline Data Flow: {
  direction: down

  input: Input Stage {
    gnn_files: "*.md GNN Files"
    config: "config.yaml"
  }

  processing: Processing Stages {
    direction: right

    parse: GNN Parsing {
      label: "Step 3: Parse & Serialize\n22 formats"
    }

    validate: Validation {
      label: "Steps 5-6: Type check &\nconsistency validation"
    }

    export: Export {
      label: "Step 7: Multi-format export\nJSON, XML, GraphML..."
    }

    visualize: Visualization {
      label: "Steps 8-9: Graph & matrix\nvisualization"
    }
  }

  generation: Code Generation {
    direction: right

    render: Framework Rendering {
      label: "Step 11: Generate code for\nPyMDP, RxInfer.jl, etc."
    }

    execute: Simulation Execution {
      label: "Step 12: Run simulations\nwith result capture"
    }
  }

  analysis: Analysis & Output {
    direction: right
    llm: LLM Analysis -> ml_integration: ML Integration -> audio: Audio Sonification -> report: Final Report
  }

  input -> processing -> generation -> analysis

  # Step 11 re-parses input specs; step 3 parsed artifacts feed steps 6/7/8/9 (d2 comments start with #)
  generation.render -> generation.execute: "Generated code"
  generation.execute -> analysis: "Simulation results"
}
```

### 6. Active Inference Concept Diagrams

Create educational diagrams for Active Inference concepts represented in GNN:

```d2
Active Inference Free Energy Principle: {
  direction: down

  agent: Cognitive Agent {
    shape: person
  }

  world: External World {
    shape: cloud
  }

  generative_model: Generative Model {
    direction: right

    prior: Prior Beliefs {
      label: "P(s,π)"
    }

    likelihood: Likelihood {
      label: "P(o|s)"
    }

    preferences: Preferences {
      label: "P(π)"
    }
  }

  inference: Inference Process {
    direction: right

    perception: State Inference {
      label: "Minimize VFE\nVariational Free Energy"
    }

    action: Policy Selection {
      label: "Minimize EFE\nExpected Free Energy"
    }
  }

  agent -> generative_model: "Internal model"
  world -> generative_model: "Observations"
  generative_model -> inference: "Model-based inference"
  inference -> agent: "Action selection"
  inference -> generative_model: "Belief updates"
}
```

### 7. Validation and Type Checking Rules

Visualize the validation rules and type checking constraints:

```d2
GNN Validation Rules: {
  direction: down

  syntax: Syntax Validation {
    direction: right
    structure: "Required sections present"
    delimiters: "Proper GNN delimiters"
    formatting: "Markdown formatting"
  }

  semantic: Semantic Validation {
    direction: right
    dimensions: "Matrix/vector dimensions"
    types: "Data type consistency"
    ontology: "Active Inference concepts"
  }

  consistency: Cross-format Consistency {
    direction: right
    round_trip: "Parse → Serialize → Parse"
    equivalence: "Format equivalence"
    validation: "Mathematical constraints"
  }

  syntax -> semantic -> consistency: "Validation Hierarchy"
}
```

### 8. Export Format Relationships

Show relationships between different export formats and their purposes:

```d2
Export Format Ecosystem: {
  direction: down

  gnn_source: GNN Source {
    shape: document
  }

  serialization: Serialization Formats {
    direction: right

    programming: Programming Languages {
      label: "Scala, Lean, Coq, Python,\nHaskell, Isabelle..."
    }

    data_exchange: Data Exchange {
      label: "JSON, XML, YAML"
    }

    graph_theory: Graph Formats {
      label: "GraphML, GEXF"
    }

    python_specific: Python Formats {
      label: "Pickle, Protocol Buffers"
    }
  }

  visualization: Visualization Formats {
    direction: right
    graphs: "Network graphs"
    matrices: "Adjacency matrices"
    diagrams: "Structural diagrams"
  }

  gnn_source -> serialization: "Multi-format export"
  serialization -> visualization: "Visual analysis"
}
```

## D2 Integration Strategies

### 1. Pipeline-Generated D2 Files

The pipeline generates D2 diagrams in the advanced-visualization step (step 9):

```bash
# Generate D2 diagrams for GNN models
uv run python src/gnn/9_advanced_viz.py --target-dir input/gnn_files --viz-type d2 --verbose

# D2 sources and rendered images land under
output/9_advanced_viz_output/d2_diagrams/
```

### 2. Watch Mode for Interactive Development

Use D2's watch mode during GNN model development:

```bash
# Watch GNN files and auto-regenerate diagrams
d2 --watch gnn_pipeline_overview.d2

# Edit GNN specifications and see diagram updates in real-time
```

### 3. Theme Integration with Pipeline

Apply D2 themes that match the GNN pipeline's visual identity:

```d2
vars: {
  d2-config: {
    theme-id: 1  # Professional theme matching pipeline docs
    pad: 20
    dark-theme: 100
  }
}
```

### 4. Programmatic D2 Generation

Use D2's programmatic API (d2oracle) to generate diagrams from GNN processing:

```go
// Generate D2 diagrams programmatically from GNN models
func generateGNNVisualization(gnnModel *GNNModel) string {
    // Create D2 graph representing the GNN structure
    graph := d2oracle.Create(emptyGraph, nil, gnnModel.Name)
    
    // Add nodes for each GNN component
    for _, component := range gnnModel.Components {
        d2oracle.Set(graph, nil, component.Path, nil, &component.Label)
    }
    
    // Add connections based on GNN relationships
    for _, connection := range gnnModel.Connections {
        d2oracle.Create(graph, nil, connection.Path)
    }
    
    return graph
}
```

## D2 Configuration for GNN Pipeline

### Layout Engine Selection

Choose appropriate layout engines for different diagram types:

```bash
# For pipeline architecture: use ELK for complex hierarchies
d2 --layout=elk pipeline_architecture.d2 output.svg

# For model structures: use TALA for software architecture
d2 --layout=tala gnn_model_structure.d2 output.svg

# For flow diagrams: use Dagre for directed graphs
d2 --layout=dagre data_flow.d2 output.svg
```

### Custom Themes for Professional Output

Create themes that align with scientific and technical documentation standards:

```d2
vars: {
  theme-overrides: {
    B1: "#ffffff"   # Background
    B2: "#f8f9fa"   # Secondary background
    AA2: "#212529"  # Primary text
    AA4: "#6c757d"  # Secondary text
  }
  
  dark-theme-overrides: {
    B1: "#1a1a1a"   # Dark background
    B2: "#2d2d2d"   # Dark secondary
    AA2: "#ffffff"  # Light text
    AA4: "#adb5bd"  # Light secondary text
  }
}
```

## Advanced D2 Features for GNN

### 1. Sequence Diagrams for Inference Processes

Use D2's sequence diagram support to visualize Active Inference inference loops:

```d2
Inference Process: {
  shape: sequence_diagram

  agent: Cognitive Agent
  world: Environment
  model: Generative Model
  inference: Inference Engine

  agent -> world: Take action u
  world -> agent: Receive observation o
  agent -> inference: Provide observation o
  inference -> model: Query generative model
  model -> inference: Return likelihood P(o|s)
  inference -> agent: Update beliefs about s
  agent -> inference: Request policy selection
  inference -> agent: Return policy π
  agent -> agent: Sample action from π
}
```

### 2. SQL Table Diagrams for Model Parameters

Represent GNN model parameters as database schemas:

```d2
GNN Model Schema: {
  shape: sql_table

  model_metadata: {
    shape: sql_table
    id: int {constraint: primary_key}
    name: varchar(255)
    version: varchar(50)
    description: text
  }

  state_space: {
    shape: sql_table
    id: int {constraint: primary_key}
    model_id: int {constraint: foreign_key}
    factor_name: varchar(100)
    dimensions: int
    type: varchar(50)
  }

  matrices: {
    shape: sql_table
    id: int {constraint: primary_key}
    model_id: int {constraint: foreign_key}
    matrix_name: varchar(50)
    dimensions: varchar(50)
    data_type: varchar(20)
  }

  model_metadata.id <-> state_space.model_id
  model_metadata.id <-> matrices.model_id
}
```

### 3. UML Class Diagrams for GNN Components

Create UML diagrams showing the class hierarchy of GNN processing components:

```d2
GNN Processing Classes: {
  direction: down

  base: GNNProcessor {
    shape: class

    +process_file(file_path: Path) -> GNNModel
    +validate_model(model: GNNModel) -> bool
    #parse_section(section: str) -> dict
  }

  parsers: GNNParsers {
    shape: class

    +parse_markdown(content: str) -> GNNModel
    +parse_state_space(section: str) -> StateSpace
    +parse_connections(section: str) -> List[Connection]
  }

  serializers: GNNSerializers {
    shape: class

    +serialize_json(model: GNNModel) -> str
    +serialize_xml(model: GNNModel) -> str
    +serialize_graphml(model: GNNModel) -> str
  }

  validators: GNNValidators {
    shape: class

    +validate_syntax(model: GNNModel) -> ValidationResult
    +validate_semantics(model: GNNModel) -> ValidationResult
    +check_consistency(model: GNNModel) -> ValidationResult
  }

  base <-> parsers: "Uses"
  base <-> serializers: "Uses"
  base <-> validators: "Uses"
}
```

### 4. Grid Diagrams for Framework Comparisons

Compare simulation frameworks in a structured grid layout:

```d2
Framework Comparison: {
  grid-columns: 4
  grid-rows: 5

  framework: Framework
  language: Language
  performance: Performance
  features: Key Features

  pymdp: PyMDP
  pymdp.language: Python
  pymdp.performance: Fast
  pymdp.features: "Full Active Inference\nPython ecosystem"

  rxinfer: RxInfer.jl
  rxinfer.language: Julia
  rxinfer.performance: Very Fast
  rxinfer.features: "Reactive inference\nProbabilistic programming"

  activeinference: ActiveInference.jl
  activeinference.language: Julia
  activeinference.performance: Fast
  activeinference.features: "Complete Active Inference\nHierarchical models"

  discopy: DisCoPy
  discopy.language: Python
  discopy.performance: Moderate
  discopy.features: "Categorical diagrams\nCompositional reasoning"

  jax: JAX
  jax.language: Python
  jax.performance: Very Fast
  jax.features: "GPU acceleration\nAutomatic differentiation"
}
```

## Integration with GNN Pipeline Steps

### Step 9: Advanced Visualization Integration

D2 diagram generation runs in pipeline step 9 (`src/gnn/9_advanced_viz.py`). `src/gnn/advanced_visualization/processor.py` routes `--viz-type d2`, `diagrams`, or `pipeline` (and `all`) to the D2 generators:

```bash
uv run python src/gnn/9_advanced_viz.py --target-dir input/gnn_files --viz-type d2 --verbose

# Output layout
output/9_advanced_viz_output/
└── d2_diagrams/
    ├── <model>/                 # <model>_structure.d2, <model>_pomdp.d2 (+ .svg/.png)
    └── pipeline/                # gnn_pipeline_flow.d2, framework_integration.d2 (+ .svg/.png)
```

Programmatic entry points: `D2Visualizer.generate_all_diagrams_for_model(model_data, output_dir)` for the per-model diagrams, and `process_gnn_file_with_d2(gnn_file, output_dir)`, which loads the Step 3 parsed JSON (`<model>/<model>_parsed.json`, default directory `output/3_gnn_output`) or parses the GNN file directly, then generates all applicable diagrams.

### D2 Renderer Capability Summary

`D2Visualizer` (`src/gnn/advanced_visualization/d2_visualizer.py`) builds five diagram types from GNN model data and compiles them with the `d2` CLI:

- **Model structure** — state-space variables (shapes derived from Active Inference annotations and variable dimensions), model connections with typed arrows, and an ontology-mapping block.
- **POMDP structure** — generative-model matrices with dimensions and an inference-engine block; generated only for POMDP-like models (POMDP variables or Active Inference annotations present).
- **Pipeline flow** — input, core processing, code-generation, and analysis stages, with optional framework execution stages (`include_frameworks=True`).
- **Framework mapping** — GNN specification → code generation → simulation execution for PyMDP, RxInfer.jl, ActiveInference.jl, DisCoPy, and JAX.
- **Active Inference concepts** — a fixed free-energy-principle diagram.

Each diagram is written as a `<name>.d2` source file and compiled to `svg`, `png`, or `pdf` (unrecognized formats are dropped; `svg`/`png` are the fallback) using `d2 --layout=elk --theme=1 --pad=20`, plus optional `--dark-theme` and `--sketch` flags. Compilation runs through the shared subprocess envelope with a 30-second timeout; per-format failures surface as warnings. The `d2` CLI must be on `PATH`.

## Performance and Scalability

### D2 Compilation Performance

```d2
D2 Performance Characteristics: {
  direction: right

  small_diagrams: "Simple model structures" {
    label: "< 1 second"
  }

  medium_diagrams: "Pipeline architecture" {
    label: "1-3 seconds"
  }

  large_diagrams: "Complex frameworks" {
    label: "3-10 seconds"
  }

  huge_diagrams: "Full ecosystem maps" {
    label: "10-30 seconds"
  }
}
```

### Memory Usage Optimization

D2's efficient compilation pipeline ensures minimal memory footprint for GNN diagram generation.

## Best Practices for GNN-D2 Integration

### 1. Diagram Organization

```bash
# Recommended directory structure for D2 files
docs/diagrams/
├── pipeline/
│   ├── architecture.d2
│   ├── data_flow.d2
│   └── module_dependencies.d2
├── models/
│   ├── active_inference_concepts.d2
│   ├── pomdp_structure.d2
│   └── framework_mapping.d2
└── validation/
    ├── type_checking_rules.d2
    └── consistency_checks.d2
```

### 2. Naming Conventions

- Use descriptive, hierarchical names: `pipeline.architecture.overview.d2`
- Include version numbers for model-specific diagrams: `actinf_pomdp_agent.v1.structure.d2`
- Use consistent prefixes: `gnn_`, `pipeline_`, `framework_`

### 3. Theme Consistency

Maintain visual consistency across all GNN-related diagrams:

```d2
# Standard theme configuration for GNN diagrams
vars: {
  d2-config: {
    theme-id: 1
    layout: elk
    pad: 20
  }
  
  theme-overrides: {
    B1: "#ffffff"
    B2: "#f8f9fa"
    AA2: "#212529"
    AA4: "#6c757d"
  }
}
```

### 4. Documentation Integration

Include D2 diagrams in AGENTS.md documentation files:

```markdown
## Pipeline Architecture

Below is a visual representation of the GNN pipeline architecture:

> [!NOTE]
> Pipeline Architecture diagram rendered with the D2 CLI.
> `diagrams/pipeline/architecture.svg`


```d2
# D2 source code for the above diagram
GNN Pipeline Architecture: {
  direction: down
  # ... diagram definition ...
}
```
```

## Conclusion

D2 provides powerful declarative diagramming capabilities that perfectly complement the GNN Pipeline's scientific computing workflow. By integrating D2 diagram generation into the pipeline's visualization steps, researchers and developers can create professional, interactive diagrams that enhance understanding of:

- Complex pipeline architectures and data flows
- GNN model structures and Active Inference concepts
- Framework integrations and simulation mappings
- Validation rules and type checking constraints
- Module dependencies and system relationships

The combination of GNN's formal model specifications with D2's professional diagramming capabilities creates a comprehensive ecosystem for Active Inference research, enabling both rigorous mathematical modeling and clear visual communication of complex concepts.

---

## References

### D2 Documentation
- [D2 Official Documentation](https://d2lang.com)
- [D2 Tour and Examples](https://d2lang.com/tour/)
- [D2 Layout Engines](https://d2lang.com/tour/layouts/)
- [D2 Themes](https://d2lang.com/tour/themes/)
- [D2 API Reference](https://d2lang.com/tour/api/)

### GNN Pipeline Documentation
- [GNN Pipeline README](../../README.md)
- [Pipeline Architecture](../../ARCHITECTURE.md)
- [GNN Syntax Guide](../gnn/reference/gnn_syntax.md)
- [.agent_rules](../../.agent_rules)
- [Pipeline Execution Summary](#)

### Active Inference Resources
- [Active Inference Wikipedia](https://en.wikipedia.org/wiki/Active_inference)
- [Free Energy Principle](https://www.sciencedirect.com/science/article/pii/S0022249615000759)

---

**D2 Version**: 0.6.0  
**GNN Pipeline Version**: [pyproject.toml](../../pyproject.toml) (canonical)  
**Integration Status**: ✅ Implemented (`d2` CLI required on `PATH`)
