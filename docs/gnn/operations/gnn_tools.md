# GNN Tools and Resources

**Version**: v3.2.0 Engine (Bundle v2.0.0)  
**Last Updated**: 2026-04-15  
**Status**: Maintained  

This document provides information about tools, libraries, and resources for working with Generalized Notation Notation (GNN).

## GNN Tool Ecosystem

The GNN ecosystem includes various tools for different aspects of working with GNN files:

```mermaid
graph TD
    GNN[GNN Files] --> PARSER[Parsing Tools]
    GNN --> VIZ[Visualization Tools]
    GNN --> CONVERT[Conversion Tools]
    GNN --> VALID[Validation Tools]
    GNN --> EDIT[Editing Tools]
    
    PARSER --> PYPARSER[Python Parser]
    PARSER --> JSPARSER[JavaScript Parser]
    
    VIZ --> GRAPHVIZ[Static Visualizer]
    VIZ --> INTERACT[Interactive Visualizer]
    
    CONVERT --> PYCODE[Python Code Generator]
    CONVERT --> JULIACODE[Julia Code Generator]
    
    VALID --> SYNTAX[Syntax Checker]
    VALID --> SEMANT[Semantic Validator]
    
    EDIT --> IDE[IDE Plugins]
    EDIT --> WEBAPP[Web Editor]
    
    classDef core fill:#f96,stroke:#333,stroke-width:2px;
    classDef category fill:#9cf,stroke:#333,stroke-width:1px;
    classDef tool fill:#cfc,stroke:#333,stroke-width:1px;
    
    class GNN core;
    class PARSER,VIZ,CONVERT,VALID,EDIT category;
    class PYPARSER,JSPARSER,GRAPHVIZ,INTERACT,PYCODE,JULIACODE,LATEXGEN,SYNTAX,SEMANT,IDE,WEBAPP tool;
```

## Parsing Tools

### GNN Parser (Repository CLI)

The repository provides parsing through the `gnn` CLI and Step 3 orchestrator:

```bash
# Parse a single file with the CLI (positional file argument)
uv run gnn parse input/gnn_files/discrete/actinf_pomdp_agent.md --format json

# Parse via step-3 script (directory-oriented, used by the pipeline)
uv run python src/gnn/3_gnn.py --target-dir input/gnn_files --output-dir output --verbose
```

### GNN Parser API Reference

| Command | Description | Output |
|----------|-------------|--------|
| `uv run gnn parse <file> [--format json\|yaml\|summary]` | Parse and serialize a single GNN model | stdout (JSON/YAML/summary) |
| `uv run gnn validate <file> [--strict]` | Validate GNN syntax and structure | validation report + exit status |
| `uv run python src/gnn/main.py --only-steps "3,5"` | Parse then type-check (all files in `--target-dir`) | `output/3_gnn_output/`, `output/5_type_checker_output/` |

## Visualization Tools

### GNN Graph Visualizer

The GNN Graph Visualizer creates graphical representations of GNN models:

```mermaid
graph TD
    subgraph Visualization Process
        GNN[GNN File] --> PARSE[Parse Model]
        PARSE --> LAYOUT[Compute Layout]
        LAYOUT --> RENDER[Render Graph]
        RENDER --> OUTPUT[Generate Output]
    end
    
    subgraph Output Formats
        OUTPUT --> SVG[SVG]
        OUTPUT --> PNG[PNG]
        OUTPUT --> HTML[Interactive HTML]
        OUTPUT --> PDF[PDF]
    end
    
    classDef process fill:#9cf,stroke:#333,stroke-width:1px;
    classDef format fill:#cfc,stroke:#333,stroke-width:1px;
    
    class GNN,PARSE,LAYOUT,RENDER,OUTPUT process;
    class SVG,PNG,HTML,PDF format;
```

#### Command-Line Interface

```bash
# Generate a dependency graph for a single file through the CLI
uv run gnn graph input/gnn_files/discrete/actinf_pomdp_agent.md --format mermaid

# Or run visualization step directly (directory-oriented, used by the pipeline)
uv run python src/gnn/main.py --only-steps "8,9" --target-dir input/gnn_files --verbose
```

#### Python API

```python
from visualization.processor import process_visualization
from pathlib import Path

# Run visualization on parsed GNN output
process_visualization(
    target_dir=Path("input/gnn_files"), output_dir=Path("output"), verbose=True
)
```

## Conversion Tools

### GNN to Python Converter

The pipeline generates executable code from GNN files via Step 11 (Render):

```bash
# Generate code for all supported frameworks
uv run python src/gnn/11_render.py --target-dir input/gnn_files --output-dir output --verbose

# Generate for a specific framework
uv run python src/gnn/12_execute.py --frameworks "pymdp,jax" --verbose
```

### GNN Export Formats

The export step (Step 7) serializes GNN models to several structured
formats (JSON, XML, GraphML, GEXF, pickle, plaintext summary, plaintext
DSL — see `src/gnn/export/formatters.py` and `list_export_formats` for the
live list). There is currently no LaTeX export target in `src/gnn/export/`.

```bash
# Export GNN models to all supported formats
uv run python src/gnn/7_export.py --target-dir input/gnn_files --output-dir output --verbose
```

## Validation Tools

### GNN Validator

The pipeline validates GNN files via Step 5 (Type Checker) and Step 6 (Validation):

```bash
# Type checking and validation
uv run python src/gnn/5_type_checker.py --target-dir input/gnn_files --output-dir output --strict
uv run python src/gnn/6_validation.py --target-dir input/gnn_files --output-dir output --verbose
```

#### Common Validation Checks

- Syntax compliance with GNN specification
- Variable references consistency
- Dimensionality compatibility in equations
- Proper connection definitions
- Ontology mapping validity

## Editing Tools

### GNN Web Editor

The GNN Web Editor provides a browser-based interface for creating and editing GNN models:

```mermaid
graph TD
    UI[Web Interface] --> EDIT[Edit Model]
    UI --> VIZ[Visualize]
    UI --> SIM[Simulate]
    UI --> SHARE[Share]
    
    EDIT --> SYNTAX[Syntax Highlighting]
    EDIT --> COMPLETE[Auto-Complete]
    EDIT --> CHECK[Real-time Validation]
    
    VIZ --> GRAPH[Connection Graph]
    VIZ --> EQ[Equation Rendering]
    
    SIM --> RUN[Run Simulation]
    SIM --> PLOT[Plot Results]
    
    SHARE --> EXPORT[Export File]
    SHARE --> URL[Share URL]
    
    classDef main fill:#f96,stroke:#333,stroke-width:2px;
    classDef feature fill:#9cf,stroke:#333,stroke-width:1px;
    
    class UI main;
    class EDIT,VIZ,SIM,SHARE,SYNTAX,COMPLETE,CHECK,GRAPH,EQ,RUN,PLOT,EXPORT,URL feature;
```

Build and edit models visually with the [GNN GUI tooling](../../gui_oxdraw/README.md).

### IDE Plugins

GNN support is available for various IDEs:

- **VS Code Extension**: Syntax highlighting, validation, and visualization
- **Jupyter Notebook Extension**: Interactive GNN editing and visualization
- **PyCharm Plugin**: GNN integration with Python development

## Integration with Active Inference Frameworks

GNN tools integrate with several Active Inference frameworks:

Step 11 renders registered backends and Step 12 executes those with installed runtimes (bnlearn via `src/gnn/execute/bnlearn/`, ngc-learn via `src/gnn/execute/ngclearn/`, skipping until their runtimes are installed). `uv run gnn doctor` reports structured readiness diagnoses; cpomdp remains experimental and explicitly selected.

```mermaid
graph LR
    GNN[GNN Tools] --> RENDER[Step 11 Render]
    RENDER --> PYMDP[PyMDP]
    RENDER --> RXINFER[RxInfer.jl]
    RENDER --> AIFJL[ActiveInference.jl]
    RENDER --> JAX[JAX]
    RENDER --> DISCOPY[DisCoPy]
    RENDER --> BNLEARN[bnlearn]
    RENDER --> PYTORCH[PyTorch]
    RENDER --> NUMPYRO[NumPyro]
    RENDER --> STAN[Stan]
    RENDER --> NGCLEARN[ngc-learn]

    classDef tool fill:#f96,stroke:#333,stroke-width:2px;
    classDef framework fill:#9cf,stroke:#333,stroke-width:1px;

    class GNN,RENDER tool;
    class PYMDP,RXINFER,AIFJL,JAX,DISCOPY,BNLEARN,PYTORCH,NUMPYRO,STAN,NGCLEARN framework;
```

| Language | Backends |
|---|---|
| Python | PyMDP, JAX, DisCoPy, bnlearn, PyTorch, NumPyro, ngc-learn |
| Julia | RxInfer.jl, ActiveInference.jl |
| Stan | Stan |

PyMDP requires the `inferactively-pymdp` package — **not** the unrelated PyPI package named `pymdp`. Step 12 detects the wrong variant and reports it explicitly.

### Integration Examples

#### PyMDP Integration

```bash
# Render and execute via PyMDP backend
uv run python src/gnn/11_render.py --target-dir input/gnn_files --output-dir output --verbose
uv run python src/gnn/12_execute.py --frameworks "pymdp" --verbose
```

See [PyMDP implementation guide](../implementations/pymdp.md) for details on the generated code structure.

## Model Repository

The GNN Model Repository contains a collection of pre-built GNN models:

```mermaid
mindmap
  root((GNN Repository))
    Basic Models
      Static Perception
      Dynamic Perception
      Policy Selection
    Cognitive Models
      Working Memory
      Decision Making
      Attention
    Neuroscience Models
      Predictive Coding
      Neural Coordination
      Oscillations
    Embodied Models
      Sensorimotor Integration
      Navigation
      Tool Use
```

Browse example models under [input/gnn_files/](../../../input/gnn_files/) and the [example gallery](../tutorials/gnn_examples_doc.md).

## Installation and Setup

### Command Line Tools

Basic usage:

```bash
# Validate a single GNN file
uv run gnn validate input/gnn_files/discrete/actinf_pomdp_agent.md

# Parse a single GNN file
uv run gnn parse input/gnn_files/discrete/actinf_pomdp_agent.md --format json

# Visualize (dependency graph) a single GNN model
uv run gnn graph input/gnn_files/discrete/actinf_pomdp_agent.md --format mermaid

# Run the full pipeline (directory-oriented)
uv run python src/gnn/main.py --target-dir input/gnn_files --verbose
```

## Documentation and Resources

### Official Documentation

- [GNN Specification](../reference/gnn_syntax.md)
- [API Reference](../../api/README.md)
- [Tutorials](../tutorials/quickstart_tutorial.md)

### Learning Resources

- [GNN Introduction](../gnn_overview.md)
- [Active Inference Institute YouTube](https://www.youtube.com/@ActiveInference)
- [Example Gallery](../tutorials/gnn_examples_doc.md)

### Community

- [GitHub Repository](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation)
- [GitHub Discussions](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/discussions)
- [Issue Tracker](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues)

## Step 4 GNN type checker

Tutorial “step 4” lines up with the **type checker** in [Validation Tools](#validation-tools) and the **`5_type_checker.py`** orchestrator in [Complete Pipeline Stages (25 Steps)](#complete-pipeline-stages-25-steps).

## GNN type checker and resource estimator

See [Validation Tools](#validation-tools) and [resource metrics](resource_metrics.md).

## GNN visualization

See [Visualization Tools](#visualization-tools) and pipeline Steps 8–9 in [Complete Pipeline Stages (25 Steps)](#complete-pipeline-stages-25-steps).

## Future Tool Development

Planned tools and features:

```mermaid
timeline
    title GNN Tools Roadmap
    section 2023
        Q3 : Basic Parser : Syntax Validator
        Q4 : Python Code Generator : Command Line Tools
    section 2024
        Q1 : Interactive Visualizer : VS Code Extension
        Q2 : Julia and PyTorch Generators : Web Editor
        Q3 : Model Repository : Template Gallery
        Q4 : Interactive Simulator : Jupyter Integration
    section 2025
        Q1 : AI-Assisted Model Creation : Community Extensions
        Q2 : Model Comparison Tools : Cross-Framework Converter
```

## References

1. Smékal, J., & Friedman, D. A. (2023). Generalized Notation Notation for Active Inference Models. Active Inference Institute. <https://doi.org/10.5281/zenodo.7803328>
2. Active Inference Institute: [Generalized Notation Notation (GNN) GitHub Repository](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation)
3. Smith, R., Friston, K.J., & Whyte, C.J. (2022). A step-by-step tutorial on active inference and its application to empirical data. Journal of Mathematical Psychology, 107, 102632.

## GNN Processing Pipeline (`src/gnn/main.py`)

The GNN project includes a comprehensive 25-step pipeline orchestrated by `src/gnn/main.py`. This main pipeline script is designed to process GNN files through the complete workflow from parsing to execution, visualization, and report generation.

For detailed architecture information, see:

- **[src/gnn/AGENTS.md](../../../src/gnn/AGENTS.md)**: Complete module registry and architectural patterns
- **[src/gnn/README.md](../../../src/gnn/README.md)**: Pipeline safety and reliability documentation
- **[Quickstart Tutorial](../tutorials/quickstart_tutorial.md)**: Step-by-step tutorial for using the pipeline

### Overview

The `src/gnn/main.py` script orchestrates 25 numbered Python scripts (steps 0-24) located in the `src/` directory. Each script corresponds to a specific processing stage following the **thin orchestrator pattern** - delegating core functionality to modular implementations in their associated directories.

The pipeline is designed to be flexible, allowing users to run the entire sequence, skip certain steps, or run only specific steps. It manages configurations like target directories for GNN files and output directories for generated artifacts.

### Running the Pipeline

To use the pipeline, navigate to the root directory of the GeneralizedNotationNotation project and execute:

```bash
uv run python src/gnn/main.py [options]
```

### Command-Line Options

The `src/gnn/main.py` script accepts several command-line arguments:

- `--target-dir DIR`: Specifies the primary target directory for GNN files. Default: `input/gnn_files`
- `--output-dir DIR`: Defines the base directory where all output files will be saved. Default: `output/`
- `--recursive`: Enables recursive processing of directories
- `--skip-steps LIST`: Comma-separated list of step numbers to exclude. Example: `"3,5,7"` or `"3_gnn,5_type_checker"`
- `--only-steps LIST`: Comma-separated list of step numbers to run exclusively. Example: `"3,5,8,11,12"`
- `--verbose`: Enables detailed logging output
- `--strict`: Activates strict type checking mode (for step 5)
- `--estimate-resources`: Enables computational resource estimation (for step 5)

View all options: `python src/gnn/main.py --help`

### Complete Pipeline Stages (25 Steps)

The GNN processing pipeline consists of exactly 25 steps (0-24), executed in order:

0. **`0_template.py` (Template Initialization)** → `src/gnn/template/`
    - Purpose: Pipeline template and initialization

1. **`1_setup.py` (Environment Setup)** → `src/gnn/setup/`
    - Purpose: Environment setup, virtual environment management, dependency installation

2. **`2_tests.py` (Test Suite)** → `tests/`
    - Purpose: Comprehensive test suite execution

3. **`3_gnn.py` (GNN Core Processing)** → `src/gnn/`
    - Purpose: GNN file discovery, multi-format parsing, and validation
    - See: [src/gnn/AGENTS.md](../../../src/gnn/AGENTS.md)

4. **`4_model_registry.py` (Model Registry)** → `src/gnn/model_registry/`
    - Purpose: Model registry management and versioning

5. **`5_type_checker.py` (Type Checking)** → `src/gnn/type_checker/`
    - Purpose: GNN syntax validation and resource estimation
    - Supports `--strict` and `--estimate-resources` flags

6. **`6_validation.py` (Validation)** → `src/gnn/validation/`
    - Purpose: Advanced validation and consistency checking

7. **`7_export.py` (Export)** → `src/gnn/export/`
    - Purpose: Multi-format export (JSON, XML, GraphML, GEXF, Pickle)

8. **`8_visualization.py` (Visualization)** → `src/gnn/visualization/`
    - Purpose: Graph and matrix visualization generation
    - See: [src/gnn/visualization/AGENTS.md](../../../src/gnn/visualization/AGENTS.md)

9. **`9_advanced_viz.py` (Advanced Visualization)** → `src/gnn/advanced_visualization/`
    - Purpose: Advanced visualization and interactive plots

10. **`10_ontology.py` (Ontology Processing)** → `src/gnn/ontology/`
    - Purpose: Active Inference Ontology processing and validation
    - See: [src/gnn/ontology/AGENTS.md](../../../src/gnn/ontology/AGENTS.md)

11. **`11_render.py` (Code Rendering)** → `src/gnn/render/`
    - Purpose: Code generation for PyMDP, RxInfer, ActiveInference.jl, DisCoPy, JAX
    - See: [src/gnn/render/AGENTS.md](../../../src/gnn/render/AGENTS.md)

12. **`12_execute.py` (Execution)** → `src/gnn/execute/`
    - Purpose: Execute rendered simulation scripts with result capture
    - See: [src/gnn/execute/AGENTS.md](../../../src/gnn/execute/AGENTS.md)

13. **`13_llm.py` (LLM Integration)** → `src/gnn/llm/`
    - Purpose: LLM-enhanced analysis, model interpretation, and AI assistance
    - See: [src/gnn/llm/AGENTS.md](../../../src/gnn/llm/AGENTS.md)

14. **`14_ml_integration.py` (ML Integration)** → `src/gnn/ml_integration/`
    - Purpose: Machine learning integration and model training

15. **`15_audio.py` (Audio Generation)** → `src/gnn/audio/`
    - Purpose: Audio generation (SAPF, Pedalboard, and other backends)

16. **`16_analysis.py` (Analysis)** → `src/gnn/analysis/`
    - Purpose: Advanced analysis and statistical processing

17. **`17_integration.py` (System Integration)** → `src/gnn/integration/`
    - Purpose: System integration and cross-module coordination

18. **`18_security.py` (Security)** → `src/gnn/security/`
    - Purpose: Security validation and access control

19. **`19_research.py` (Research Tools)** → `src/gnn/research/`
    - Purpose: Research tools and experimental features

20. **`20_website.py` (Website Generation)** → `src/gnn/website/`
    - Purpose: Static HTML website generation from pipeline artifacts

21. **`21_mcp.py` (Model Context Protocol)** → `src/gnn/mcp/`
    - Purpose: Model Context Protocol processing and tool registration

22. **`22_gui.py` (GUI)** → `src/gnn/gui/`
    - Purpose: Interactive GUI for constructing/editing GNN models

23. **`23_report.py` (Report Generation)** → `src/gnn/report/`
    - Purpose: Comprehensive analysis report generation

24. **`24_intelligent_analysis.py` (Intelligent Analysis)** → `src/gnn/intelligent_analysis/`
    - Purpose: AI-enhanced analysis of GNN models and pipeline outputs

### MCP Tools (Step 21)

The MCP step exposes pipeline modules as callable tools through the central MCP registry. Use the audit commands for the current module/tool count rather than copying a stale snapshot into prose:

```bash
# Validate the MCP tool contract
uv run --extra dev python -m pytest tests/mcp/test_mcp_audit.py -q

# Generate a live inventory report
uv run python src/gnn/mcp/validate_tools.py

# Or via pytest (full suite totals: repository README.md)
uv run --extra dev python -m pytest tests/mcp/test_mcp_audit.py -v
```

Key tool groups:

| Domain | Example Tools |
|--------|---------------|
| audio | `process_audio`, `check_audio_backends`, `get_audio_generation_options` |
| execute | `execute_gnn_model`, `execute_pymdp_simulation`, `check_execute_dependencies` |
| llm | `analyze_gnn_with_llm`, `generate_llm_documentation`, `get_llm_providers` |
| gui | `process_gui`, `list_available_guis`, `oxdraw.convert_to_mermaid` |
| validation | `validate_gnn_file`, `check_schema_compliance`, `get_validation_report` |
| render | `render_gnn_to_format`, `list_render_frameworks` |
| gnn | `parse_gnn_content`, `validate_gnn_content` |
| extract | `extract_pomdp` |
| multimodel | `generate_dependency_graph` |
| template | `template.pull` |

For the complete tool inventory, see **[modules/21_mcp.md](../modules/21_mcp.md)**.

#### GNN Parser and Type Checker (Steps 3 & 5)

- **Pipeline Steps:** `3_gnn.py` and `5_type_checker.py`
- **Functionality:** Parse GNN files, validate syntax, check type consistency, and estimate computational resources
- **Invocation:**

    ```bash
    # Parse and type-check GNN files
    uv run python src/gnn/main.py --only-steps "3,5" --target-dir input/gnn_files
    
    # With resource estimation
    uv run python src/gnn/main.py --only-steps "3,5" --estimate-resources --target-dir input/gnn_files
    ```

- **Outputs:** `output/3_gnn_output/` and `output/5_type_checker_output/`

#### GNN Visualization (Steps 8 & 9)

- **Pipeline Steps:** `8_visualization.py` and `9_advanced_viz.py`
- **Functionality:** Generate graphical representations, factor graphs, and interactive visualizations
- **Invocation:**

    ```bash
    # Generate visualizations
    uv run python src/gnn/main.py --only-steps "8,9" --target-dir input/gnn_files
    ```

- **Outputs:** `output/8_visualization_output/` and `output/9_advanced_viz_output/`

#### Code Generation and Execution (Steps 11 & 12)

- **Pipeline Steps:** `11_render.py` and `12_execute.py`
- **Functionality:** Generate executable code for multiple frameworks and run simulations
- **Invocation:**

    ```bash
    # Generate and execute code
    uv run python src/gnn/main.py --only-steps "11,12" --target-dir input/gnn_files
    
    # Execute specific frameworks
    uv run python src/gnn/12_execute.py --frameworks "pymdp,jax" --verbose
    ```

- **Outputs:** `output/11_render_output/` and `output/12_execute_output/`

#### Post-Simulation Analysis (Step 16)

- **Pipeline Step:** `16_analysis.py`
- **Functionality:** Statistical analysis, post-simulation analysis (free energy, policy convergence, state distributions), Active Inference metric computation (entropy, KL divergence, VFE, EFE, information gain), cross-framework comparison, and visualization generation
- **Invocation:**

    ```bash
    # Run full render → execute → analyze pipeline
    uv run python src/gnn/main.py --only-steps "3,11,12,16" --target-dir input/gnn_files --verbose

    # Run analysis on existing execution results
    uv run python src/gnn/main.py --only-steps "16" --verbose
    ```

- **Outputs:** `output/16_analysis_output/` — includes `analysis_results.json`, `analysis_summary.md`, per-model post-simulation analysis, framework-specific visualizations, and cross-framework comparison dashboards
- **Key Capabilities:**
  - Per-GNN-file statistical analysis and complexity metrics
  - Post-simulation analysis consuming Step 12 execution results
  - Active Inference metrics: Shannon entropy, KL divergence, variational free energy, expected free energy, information gain
  - Framework-specific visualization generation (PyMDP, RxInfer, ActiveInference.jl, JAX, DisCoPy)
  - Cross-framework comparison reports and unified dashboards

### Pipeline Examples

```bash
# Run full pipeline
uv run python src/gnn/main.py --target-dir input/gnn_files --verbose

# Run specific steps only
uv run python src/gnn/main.py --only-steps "3,5,7,8,11,12" --verbose

# Skip certain steps
uv run python src/gnn/main.py --skip-steps "15,16" --verbose

# Run with specific framework execution
uv run python src/gnn/main.py --only-steps "11,12"
uv run python src/gnn/12_execute.py --frameworks "lite" --verbose  # Python-only: PyMDP, JAX, DisCoPy, bnlearn
```

For more detailed information on each module, see **[src/gnn/AGENTS.md](../../../src/gnn/AGENTS.md)**.

---

## Additional maintained CLI options

These flags are registered by the canonical CLI or numbered step parsers.
Use the relevant command's help to see its exact exposure and defaults.

| Command or step | Options and purpose |
| --- | --- |
| `gnn run`, `gnn gui`, `gnn models` | `-t` aliases the input target directory. `-o` selects output on commands that expose it. `-v` enables verbosity. |
| `gnn render` | `-f` selects a framework and `-o` selects the output file. |
| `gnn models list` | `-q` filters models by ontology concept substring. |
| `gnn reproduce` | `--history-dir` selects the directory holding the run history index. |
| `gnn pull` | `--overwrite` permits replacement of the selected destination model. |
| Step 3 | `--serialize-preset full` serializes all formats; `minimal` emits Markdown, JSON and Python. `--enable-round-trip` enables serializer/parser checks; `--enable-cross-format` enables consistency checks. |
| Step 7 | `--export-formats` selects the export formats exposed by this step parser. |
| Step 2 | `--include-performance` includes performance test categories. |
| Step 12 | `--timesteps` sets simulation length; `--simulation-params` supplies a JSON parameter object. |
| Step 15 | `--sonification` and `--no-sonification` enable or disable model sonification. |
| Step 22 | `--launch-editor` launches the selected oxdraw editor. |
| Boolean parser controls | `--no-recursive`, `--no-verbose`, `--no-interactive`, `--no-estimate-resources` and `--no-execution-summary-detail` disable their corresponding positive options where exposed. |
| Diagnostic testing | `--simulate-error` deliberately exercises the error path. |
