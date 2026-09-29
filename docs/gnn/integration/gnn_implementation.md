# GNN Implementation Guide

**Version**: v3.2.0 Engine (Bundle v2.0.0)  
**Last Updated**: 2026-04-14  
**Status**: ✅ Production Ready  
**Modules**: 38+ · **Pipeline steps**: 25 · **Renderers**: 11 backends (see [../implementations/README.md](../implementations/README.md)) · **Tests**: see [../../../README.md](../../../README.md)

This document provides guidance on how to implement GNN models in computational environments, with practical code examples and workflows.

## Pipeline-Based Implementation

GNN models are implemented through the processing pipeline's code generation and execution steps:

**Code Generation (Step 11)**

- **`src/gnn/11_render.py`** → Generate executable code for multiple frameworks
  - See: **[src/gnn/render/AGENTS.md](../../../src/gnn/render/AGENTS.md)** for rendering details
  - Supports eleven backends: PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy,
    PyTorch, NumPyro, Stan, bnlearn, ngclearn, cpomdp

**Execution (Step 12)**

- **`src/gnn/12_execute.py`** → Execute rendered simulation scripts
  - See: **[src/gnn/execute/AGENTS.md](../../../src/gnn/execute/AGENTS.md)** for execution details

**Quick Start:**

```bash
# Generate and execute implementations
uv run python src/gnn/main.py --only-steps "3,11,12" --target-dir input/gnn_files --verbose

# Execute specific frameworks
uv run python src/gnn/12_execute.py --frameworks "pymdp,jax" --verbose
```

For complete pipeline documentation, see **[src/gnn/AGENTS.md](../../../src/gnn/AGENTS.md)**.

---

## Step 11 — Render Module Deep Dive

The render module (`src/gnn/render/`) transforms GNN specifications into executable code for maintained simulation frameworks. It follows a POMDP-aware processing pipeline.

### Processing Pipeline

```mermaid
flowchart TD
    GNN[GNN Files<br>*.md, *.json, *.yaml] --> EXTRACT[POMDP Extraction<br>extract_pomdp_from_file]
    EXTRACT -->|POMDP spec| VALIDATE[validate_pomdp_for_rendering<br>factors, modalities, controls]
    EXTRACT -->|Non-POMDP spec| BASIC[_process_single_gnn_file_basic<br>generic render path]
    VALIDATE --> NORMALIZE[normalize_matrices<br>A/B/C/D provenance]
    NORMALIZE --> RENDER[POMDPRenderProcessor<br>process_pomdp_for_all_frameworks]
    RENDER --> PYMDP[PyMDP .py]
    RENDER --> RXINFER[RxInfer .jl]
    RENDER --> ACTINF[ActiveInference.jl .jl]
    RENDER --> JAX[JAX .py]
    RENDER --> DISCOPY[DisCoPy .py]
    RENDER --> TORCH[PyTorch .py]
    RENDER --> NUMPYRO[NumPyro .py]
    RENDER --> STAN[Stan .stan]
    RENDER --> BNLEARN[bnlearn .py]
    BASIC --> PYMDP
    BASIC --> RXINFER
    BASIC --> ACTINF
    BASIC --> DISCOPY
```

### POMDP-Aware Processing

The core function `process_render()` in `src/gnn/render/processor/pipeline.py` implements:

1. **POMDP Extraction** — Extracts state factors, observation modalities, control factors, matrices (A, B, C, D/E), and matrix provenance from GNN files via `gnn.extract.pomdp_extractor.extract_pomdp_from_file()`
2. **Validation** — `validate_pomdp_for_rendering()` checks that required dimensions and matrices are present for the requested framework
3. **Matrix Normalization** — `normalize_matrices()` ensures:
   - **A matrix** (observation model): columns sum to 1 over the observation dimension
   - **B matrix** (transition model): columns sum to 1 over the next-state dimension, per action
   - Handles 2D arrays, 3D arrays, and list-of-arrays (factorial) structures
   - Zero-sum columns are replaced with uniform distributions

### `render_gnn_spec()` API

```python
from render.processor import render_gnn_spec

success, message, warnings = render_gnn_spec(
    gnn_spec={"model_name": "my_model", ...},
    target="pymdp",           # pymdp | rxinfer | activeinference_jl | discopy
                              # | discopy_combined | bnlearn | jax | jax_pomdp
    output_directory="output/",
    options={}                # Optional additional options
)
```

`RENDER_CLI_TARGETS` in `src/gnn/render/render.py` also still accepts `rxinfer_toml`, but that
target is retired: `render_gnn_spec()` returns `(False, <message>, [])` for it and
produces no artifacts. Use `rxinfer`, which routes to the canonical renderer.

### Available Renderers

| Renderer | Language | Extension | Supported Features |
|----------|----------|-----------|--------------------|
| **PyMDP** | Python | `.py` | POMDP, MDP, Belief State Updates, Active Inference |
| **RxInfer.jl** | Julia | `.jl` | Message Passing, Probabilistic Programming, Bayesian Inference |
| **ActiveInference.jl** | Julia | `.jl` | Free Energy Minimization, Active Inference, POMDP |
| **JAX** | Python | `.py` | GPU Acceleration, Automatic Differentiation, JIT Compilation |
| **DisCoPy** | Python | `.py` | Categorical Diagrams, String Diagrams, Compositional Models |
| **PyTorch** | Python | `.py` | Tensor POMDP; `torch` ships in the `torch` extra (`uv sync --extra torch`; GHSA-rrmf-rvhw-rf47 resolved in torch 2.13.0) |
| **NumPyro** | Python | `.py` | NUTS/MCMC posterior inference |
| **Stan** | Stan | `.stan` + `_stan.py` driver | HMM forward-algorithm (Dirichlet A_est, NUTS/MAP) and Kalman marginal-likelihood programs; executed by Step 12 through cmdstanpy (`src/gnn/execute/stan/`) |
| **bnlearn** | Python | `.py` | Bayesian network structure and inference |

The registry backing this table is `FRAMEWORK_REGISTRY` in
`src/gnn/render/framework_registry.py`.

#### RxInfer renders by model kind

RxInfer is the one backend that does not emit a single model shape. Its renderer calls
`detect_model_kind()` (`src/gnn/render/pomdp_contract.py`) and dispatches to a per-kind
strategy in `src/gnn/render/rxinfer/model_strategies.py` — `FLAT`, `FACTORED`,
`HIERARCHICAL`, `MULTI_AGENT`, `CONTINUOUS`, or `LEARNING`. Detection is structural
(matrix key patterns, declared counts, `F`/`H`/`Q`/`R` and `dirichlet_[A-E]` keys), never
free-text. See
[framework_integration_guide.md](framework_integration_guide.md#rxinferjl-pipeline-details)
for the strategy table.

### Output Structure

```
output/11_render_output/
├── {model_name}/
│   ├── pymdp/                  # PyMDP Python simulations
│   ├── rxinfer/                # RxInfer.jl Julia simulations
│   ├── activeinference_jl/     # ActiveInference.jl Julia simulations
│   ├── jax/                    # JAX Python simulations
│   ├── discopy/                # DisCoPy categorical diagrams
│   ├── pytorch/                # PyTorch Python simulations
│   ├── numpyro/                # NumPyro Python simulations
│   ├── stan/                   # Stan programs (.stan)
│   ├── bnlearn/                # bnlearn Bayesian networks
│   └── processing_summary.json # Per-model render summary
├── render_processing_summary.json
└── README.md                   # Auto-generated overview
```

---

## Step 12 — Execute Module Deep Dive

The execute module (`src/gnn/execute/`) runs the scripts generated by the render step, collecting their outputs and extracting simulation data.

### Execution Flow

```mermaid
flowchart TD
    START[process_execute] --> PARSE[parse_frameworks_parameter<br>all, lite, or comma-separated]
    PARSE --> FIND[find_executable_scripts<br>*.py and *.jl in render output]
    FIND --> LOOP[For each script]
    LOOP --> CHECK[Check executor availability<br>python --version / julia --version]
    CHECK --> DEP[Framework dependency check<br>e.g. import pymdp]
    DEP --> RUN[subprocess.run<br>timeout=3600s default, cwd=script_dir]
    RUN -->|Success| EXTRACT[_extract_simulation_data<br>parse stdout/stderr]
    RUN -->|Failure| CLASSIFY[Classify error<br>DependencyError, SyntaxError, RuntimeError]
    EXTRACT --> COLLECT[collect_execution_outputs<br>visualizations, simulation_data, traces]
    COLLECT --> SAVE[Save JSON + log per script]
    CLASSIFY --> SAVE
    SAVE --> REPORT[generate_execution_report]
```

### Framework Filtering

The `--frameworks` flag supports:

| Value | Frameworks Included |
|-------|---------------------|
| `all` (default) | pymdp, jax, discopy, rxinfer, activeinference_jl, pytorch, numpyro, stan, bnlearn |
| `lite` | pymdp, jax, discopy, bnlearn |
| Custom | Comma-separated, e.g. `"pymdp,jax"` |

The authoritative list is `parse_frameworks_parameter()` in `src/gnn/execute/detection.py` (re-exported through the `src/gnn/execute/processor/` facade).
Stan scripts are the rendered `<stem>_stan.py` cmdstanpy drivers under `stan/`; they
are skipped (not failed) when cmdstanpy or a CmdStan toolchain is absent
(`uv sync --extra stan`).

### Script Discovery

`find_executable_scripts()` searches `output/11_render_output/` recursively for `.py` and `.jl` files, skipping test files, `__init__.py`, and README files. Each script is tagged with its framework based on its parent directory path.

### Subprocess Execution

`execute_single_script()` runs each script:

- **Working directory**: set to the script's parent directory
- **Timeout**: 3600 seconds per script by default (configurable via `--timeout`)
- **Environment**: `PYTHONPATH` is extended for PyMDP scripts
- **Julia**: preflight `using JSON, Distributions, StatsBase`, plus `RxInfer` when
  RxInfer is requested and `ActiveInference` when ActiveInference.jl is requested.
  `JULIA_PROJECT` defaults to the committed environment for the framework being run
  (`src/gnn/execute/rxinfer/` or `src/gnn/execute/activeinference_jl/`); an explicitly set
  `JULIA_PROJECT` still wins.

### Data Collection

After successful execution, `collect_execution_outputs()` gathers framework-specific artifacts:

- **PyMDP**: `output/pymdp_simulations/` — `.png`, `.svg`, `.json`, `.pkl`
- **DisCoPy**: `discopy_diagrams/` — `.png`, `.svg`, `.json`
- **ActiveInference.jl**: `activeinference_outputs_*/visualizations/` and `simulation_data/`
- **RxInfer**: `rxinfer_outputs_*/` — `.json`, `.csv`, `.png`
- **JAX**: `jax_outputs_*/` — `.json`, `.npy`, `.png`

### Output Structure

```
output/12_execute_output/
├── {model_name}/
│   └── {framework}/
│       └── execution_logs/
│           ├── {script}_results.json    # Structured results
│           └── {script}_execution.log   # Human-readable log
├── individual_outputs/
│   └── {framework}/
│       └── {script}_output.txt
├── execution_summary.json               # Overall results
└── execution_report.md                  # Human-readable report
```

---

## Step 16 — Analysis Module Deep Dive

The analysis module (`src/gnn/analysis/`) provides statistical analysis, post-simulation analysis with Active Inference metrics, cross-framework comparison, and visualization generation.

### Analysis Pipeline

```mermaid
flowchart TD
    START[process_analysis] --> STATS[Per-GNN-file analysis]
    STATS --> SA[perform_statistical_analysis]
    STATS --> CM[calculate_complexity_metrics]
    STATS --> PB[run_performance_benchmarks]
    STATS --> MV[generate_matrix_visualizations]
    
    START --> POST[Post-Simulation Analysis]
    POST --> EXEC[Load 12_execute_output/<br>execution_summary.json]
    EXEC --> VIZ[visualize_simulation_results]
    EXEC --> PSA[analyze_execution_results<br>per model, per framework]
    PSA --> AGG[aggregate_simulation_results<br>+ generate_summary_statistics]
    
    START --> FW[Framework-Specific Viz]
    FW --> PYMDP_VIZ[PyMDP analyzer]
    FW --> ACTINF_VIZ[ActiveInference.jl analyzer]
    FW --> DISCOPY_VIZ[DisCoPy analyzer]
    FW --> JAX_VIZ[JAX analyzer]
    FW --> RXINFER_VIZ[RxInfer analyzer]
    
    START --> CROSS[Cross-Framework Analysis]
    CROSS --> AFO[analyze_framework_outputs]
    CROSS --> REPORT[generate_framework_comparison_report]
    CROSS --> XCVIZ[visualize_cross_framework_metrics]
    CROSS --> DASH[generate_unified_framework_dashboard]
```

### Post-Simulation Analysis

The `analysis/` module (split across `post_simulation.py`, `trace_analysis.py`, `math_utils.py`, `framework_extractors.py`, and `visualizations.py`) provides framework-agnostic analysis of simulation results:

| Function | Purpose |
|----------|---------|
| `analyze_simulation_traces()` | Analyze state/observation/action trajectories |
| `analyze_free_energy()` | Free energy dynamics, convergence rate, final value |
| `analyze_policy_convergence()` | Policy evolution and convergence detection |
| `analyze_state_distributions()` | Belief state distribution analysis over time |
| `compare_framework_results()` | Cross-framework comparison of identical models |

### Active Inference Metrics

The module computes the following Active Inference-specific statistical measures:

| Metric | Function | Description |
|--------|----------|-------------|
| **Shannon Entropy** | `compute_shannon_entropy(distribution)` | Entropy of a probability distribution (nats) |
| **KL Divergence** | `compute_kl_divergence(p, q)` | D_KL(P \|\| Q) between two distributions |
| **Variational Free Energy** | `compute_variational_free_energy(obs, beliefs, A, prior)` | F = E_q[ln q(s)] − E_q[ln p(o,s)] |
| **Expected Free Energy** | `compute_expected_free_energy(beliefs, A, B, C, policy, horizon)` | G = epistemic value + pragmatic value |
| **Information Gain** | `compute_information_gain(prior, posterior)` | D_KL(posterior \|\| prior) |

These are assembled by `analyze_active_inference_metrics()` which takes complete simulation trajectories (beliefs, free energy, actions) and returns a comprehensive metrics dictionary.

### Framework-Specific Data Extraction

Each framework has a dedicated extraction function in `analysis/framework_extractors.py` (re-exported from `post_simulation.py`):

- `extract_pymdp_data()` — beliefs, actions, free energy, observations from PyMDP output
- `extract_rxinfer_data()` — posterior distributions from RxInfer JSON/CSV output  
- `extract_activeinference_jl_data()` — full Active Inference fields from `simulation_results.csv`
- `extract_jax_data()` — JAX simulation output extraction
- `extract_discopy_data()` — diagram execution counts and categorical outputs

### Output Structure

```
output/16_analysis_output/
├── analysis_results.json               # Complete analysis results
├── analysis_summary.md                 # Human-readable summary
├── {model}_post_simulation_analysis.json
├── pymdp/                              # PyMDP-specific visualizations
├── activeinference_jl/                 # ActiveInference.jl visualizations
├── discopy/                            # DisCoPy visualizations
├── jax/                                # JAX visualizations
├── rxinfer/                            # RxInfer visualizations
└── cross_framework/                    # Cross-framework comparisons
    └── unified_dashboard/              # Step 23 interactive D3.js dashboard targets
```

#### The RxInfer analysis submodule

`src/gnn/analysis/rxinfer/` carries capabilities beyond the generic per-framework
extractors, and they are what produce most of the richer Step 16 artifacts:

| Module | What it provides |
|---|---|
| `analyzer.py` | Convergence diagnostics, strategy-validation summaries, and `compute_per_factor_beliefs()` — the un-flattening of joint posteriors into per-factor (per-agent) marginals |
| `gif_animator.py` / `animator.py` | One animated GIF per model, each with a `.manifest.json` reproducibility sidecar |
| `dashboard.py` | The dashboard with category and state-size filters plus compare mode |
| `cross_framework.py` | `run_cross_framework_comparison()`, the RxInfer-anchored comparison entry point |

---

## Implementation Workflow

The process of implementing a GNN model follows these general steps:

```mermaid
graph TD
    GNN[GNN Specification] --> PARSE[Parse GNN File]
    PARSE --> VARS[Initialize Variables]
    VARS --> STRUCT[Set Up Model Structure]
    STRUCT --> IMPL[Implement Equations]
    IMPL --> TEST[Test and Validate]
    TEST --> REFINE[Refine Model]
    
    classDef start fill:#f96,stroke:#333,stroke-width:2px;
    classDef process fill:#9cf,stroke:#333,stroke-width:1px;
    classDef end fill:#cfc,stroke:#333,stroke-width:1px;
    
    class GNN start;
    class PARSE,VARS,STRUCT,IMPL process;
    class TEST,REFINE end;
```

## Prerequisites

Before implementing a GNN model, you should have:

1. A complete GNN specification file
2. Understanding of the model's purpose and functionality
3. Programming environment with appropriate libraries
4. Testing data or simulation environment

## Implementation Approaches

GNN models can be implemented in various programming environments:

```mermaid
mindmap
  root((GNN Implementation))
    Python
      NumPy/SciPy
      PyTorch
      TensorFlow
      JAX
    Julia
      Native Arrays
      Flux.jl
    MATLAB
      Native Arrays
      Deep Learning Toolbox
    R
      Base R
      Torch for R
    JavaScript
      TensorFlow.js
      Brain.js
```

## Step 1: Parsing GNN Files

The first step is to parse the GNN file to extract model structure and parameters.

### Python Example

```python
import re


def parse_gnn_file(file_path):
    """Parse a GNN file and extract sections."""
    with open(file_path, "r") as f:
        content = f.read()

    # Regular expressions to extract sections
    version_match = re.search(r"## GNN (v\d+(?:\.\d+)*)", content)
    model_name_match = re.search(r"# ([^\n]+)", content)

    # Extract state space block
    state_space_block = re.search(
        r"## State space block\n(.*?)(?=##)", content, re.DOTALL
    )

    # Extract connections
    connections_block = re.search(r"## Connections\n(.*?)(?=##)", content, re.DOTALL)

    # Extract equations
    equations_block = re.search(r"## Equations\n(.*?)(?=##)", content, re.DOTALL)

    # Return structured data
    return {
        "version": version_match.group(1) if version_match else None,
        "model_name": model_name_match.group(1) if model_name_match else None,
        "state_space": state_space_block.group(1).strip()
        if state_space_block
        else None,
        "connections": connections_block.group(1).strip()
        if connections_block
        else None,
        "equations": equations_block.group(1).strip() if equations_block else None,
        # Add other sections as needed
    }


# Usage example
model_data = parse_gnn_file("path/to/model.gnn")
print(f"Model: {model_data['model_name']}")
print(f"Version: {model_data['version']}")
```

## Step 2: Setting Up Variables

After parsing, you need to initialize the variables with appropriate dimensions.

### Python with NumPy Example

```python
import numpy as np
import re


def initialize_variables(state_space_text):
    """Initialize variables based on the state space block."""
    variables = {}

    # Split into lines and process each variable definition
    for line in state_space_text.split("\n"):
        if not line.strip() or line.strip().startswith("#"):
            continue

        # Match variable name and dimensions
        var_match = re.match(r"(\w+)(?:_\w+)?\s*\[([^\]]+)\]", line)
        if var_match:
            var_name = var_match.group(1)
            dimensions = var_match.group(2)

            # Parse dimensions (handling both numeric and symbolic dimensions)
            dims = []
            for dim in dimensions.split(","):
                dim = dim.strip()
                if re.match(r"^\d+$", dim):  # Numeric dimension
                    dims.append(int(dim))
                elif "type=" in dim:  # Type specification, ignore for now
                    continue
                else:  # Symbolic dimension (e.g., len(π))
                    # Use an explicit default value when the data source is absent
                    dims.append(2)  # Default to 2 for symbolic dimensions

            # Create the variable with zeros
            if len(dims) == 1:
                variables[var_name] = np.zeros(dims[0])
            elif len(dims) == 2:
                variables[var_name] = np.zeros((dims[0], dims[1]))
            else:
                variables[var_name] = np.zeros(tuple(dims))

    return variables


# Usage example
variables = initialize_variables(model_data["state_space"])
print("Initialized variables:")
for name, var in variables.items():
    print(f"{name}: shape {var.shape}")
```

## Step 3: Setting Up Model Structure

Next, establish the connections between variables based on the model's graphical structure.

### Python Structure Example

```python
def parse_connections(connections_text):
    """Parse connections between variables."""
    connections = {"directed": [], "undirected": []}

    for line in connections_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Check for directed connections (A>B)
        if ">" in line:
            source, target = line.split(">")
            connections["directed"].append((source.strip(), target.strip()))
        # Check for undirected connections (A-B)
        elif "-" in line:
            var1, var2 = line.split("-")
            connections["undirected"].append((var1.strip(), var2.strip()))

    return connections


# Usage example
connections = parse_connections(model_data["connections"])
print("Directed connections:", connections["directed"])
print("Undirected connections:", connections["undirected"])
```

## Step 4: Implementing Equations

The most challenging part is translating GNN equations into code. Here's a template for implementing some common equations.

### PyTorch Implementation Example

```python
import torch
import torch.nn.functional as F


class GNNModel:
    def __init__(self, variables, connections):
        """Initialize the model with variables and connections."""
        self.variables = variables
        self.connections = connections

        # Convert NumPy arrays to PyTorch tensors
        for key, value in self.variables.items():
            if isinstance(value, np.ndarray):
                self.variables[key] = torch.tensor(value, dtype=torch.float32)

    def softmax(self, x):
        """Softmax function."""
        return F.softmax(x, dim=0)

    def static_perception_step(self):
        """Implement a simple static perception step."""
        # Example equation: s = softmax(ln(D) + ln(A^T * o))
        D = self.variables.get("D", torch.ones(1))
        A = self.variables.get("A", torch.eye(2))
        o = self.variables.get("o", torch.ones(2, 1))

        # Compute the equation
        log_D = torch.log(D + 1e-10)  # Add small value to avoid log(0)
        A_transpose = torch.transpose(A, 0, 1)
        A_transpose_o = torch.matmul(A_transpose, o)
        log_A_transpose_o = torch.log(A_transpose_o + 1e-10)

        # Apply softmax
        s = self.softmax(log_D + log_A_transpose_o)

        # Update the state variable
        self.variables["s"] = s
        return s

    def dynamic_perception_step(self, time_step):
        """Implement a dynamic perception step."""
        # Simplified implementation - would need to be expanded for full model
        pass


# Usage example
model = GNNModel(variables, connections)
state = model.static_perception_step()
print("Updated state:", state)
```

## Step 5: Testing and Validation

Thoroughly test your GNN implementation with known inputs and expected outputs.

- **Unit tests**: Test individual functions and components.
- **Integration tests**: Test the entire model workflow.
- **Validation against theory**: Ensure model behavior aligns with Active Inference principles.

## Step 6: Model Refinement

Iteratively refine the model based on test results and further analysis.

- **Parameter tuning**: Adjust model parameters for desired behavior.
- **Structural changes**: Modify connections or equations if needed.
- **Optimization**: Improve computational efficiency.

## Organizing GNN Processing Tools: A Practical Approach

When developing a comprehensive environment for working with GNN, it's beneficial to structure your tools and processing steps in a modular and manageable way. The GNN project itself (which this documentation is part of) provides a practical example of such an organization within its `src/` directory.

**Key Principles Demonstrated:**

1. **Central Orchestrator (`src/gnn/main.py`):**
    - A main script (e.g., `src/gnn/main.py`) acts as the primary entry point for all GNN processing tasks.
    - It handles common functionalities like command-line argument parsing (for selecting GNN files, output directories, specific tasks to run, verbosity, etc.), loading configurations, and initializing shared resources.
    - This central script then calls individual modules or sub-scripts that perform specific stages of the processing.

2. **Modular, Numbered Scripts for Pipeline Stages:**
    - Specific tasks (e.g., core GNN parsing, setup, testing, type checking, export, visualization, ontology mapping) are implemented in separate Python scripts.
    - These scripts are often numbered (e.g., `1_setup.py`, `3_gnn.py`, `5_type_checker.py`) to indicate a logical sequence of execution or dependency.
    - Each script typically focuses on one primary function and can be developed and tested independently.
    - The orchestrator (`main.py`) dynamically discovers these numbered scripts and executes them in order, passing necessary arguments and context.

3. **Dedicated Subdirectories for Modules:**
    - Larger, more complex processing stages (like type checking or visualization) can have their own subdirectories (e.g., `src/gnn/type_checker/`, `src/gnn/visualization/`) containing their specific logic, helper files, and potentially their own internal main scripts if they can also be run standalone for debugging or specific use cases.

**Benefits of this Approach:**

- **Clarity and Organization:** The processing pipeline is clearly defined, and each step has a dedicated module.
- **Modularity:** Individual components can be updated or replaced without affecting the entire system.
- **Scalability:** New processing steps can be easily added by creating a new numbered script.
- **Flexibility:** The orchestrator can allow users to run the full pipeline, skip steps, or run only specific steps, as demonstrated by the `src/gnn/main.py` script's `--skip-steps` and `--only-steps` options.
- **Reusability:** Core logic within each module can potentially be reused in other contexts.

This structured approach, as seen in the GNN project's `src/` directory and `main.py`, is a recommended pattern for implementing a robust and maintainable GNN processing workflow.

## Advanced Topics

## Implementation Examples for Different Models

Let's look at implementations for the different complexity levels.

### 1. Static Perception Model

```mermaid
flowchart TD
    D[Prior D] --> |"ln(D)"| ADD1["+"]
    A[Matrix A] --> |"Transpose"| AT["A^T"]
    O[Observation o] --> AT
    AT --> |"A^T * o"| LOG2["ln(A^T * o)"]
    LOG2 --> ADD1
    ADD1 --> SM["softmax()"]
    SM --> S[State s]
    
    classDef input fill:#f96,stroke:#333,stroke-width:1px;
    classDef compute fill:#9cf,stroke:#333,stroke-width:1px;
    classDef output fill:#cfc,stroke:#333,stroke-width:1px;
    
    class D,A,O input;
    class ADD1,AT,LOG2,SM compute;
    class S output;
```

Python implementation for a static perception model:

```python
def static_perception(D, A, o):
    """
    Implements static perception: s = softmax(ln(D) + ln(A^T * o))
    
    Args:
        D: Prior (numpy array, shape [n, 1])
        A: Recognition matrix (numpy array, shape [m, n])
        o: Observation (numpy array, shape [m, 1])
    
    Returns:
        s: Inferred state (numpy array, shape [n, 1])
    """
    # Ensure numerical stability
    epsilon = 1e-10
    
    # Calculate A^T * o
    A_transpose = A.T
    A_transpose_o = np.dot(A_transpose, o)
    
    # Calculate ln(D) + ln(A^T * o)
    log_D = np.log(D + epsilon)
    log_A_transpose_o = np.log(A_transpose_o + epsilon)
    
    # Apply softmax
    input_to_softmax = log_D + log_A_transpose_o
    exp_input = np.exp(input_to_softmax - np.max(input_to_softmax))
    s = exp_input / np.sum(exp_input)
    
    return s
```

### 2. Dynamic Perception Model

For dynamic models, we need to track state over time.

```python
def dynamic_perception(D, A, B, o_sequence, num_timesteps):
    """
    Implements dynamic perception model.

    Args:
        D: Prior (numpy array, shape [n, 1])
        A: Recognition matrix (numpy array, shape [m, n])
        B: Transition matrix (numpy array, shape [n, n])
        o_sequence: Sequence of observations (numpy array, shape [m, T])
        num_timesteps: Number of time steps to simulate

    Returns:
        s_sequence: Sequence of inferred states (numpy array, shape [n, T])
    """
    n = D.shape[0]
    T = o_sequence.shape[1]
    s_sequence = np.zeros((n, T))

    # Initialize with static perception for the first step
    s_sequence[:, 0:1] = static_perception(D, A, o_sequence[:, 0:1])

    # Process remaining time steps
    for t in range(1, T):
        # Prediction step (forward message passing)
        s_predicted = np.dot(B, s_sequence[:, t - 1 : t])

        # Update step (backward message passing)
        o_t = o_sequence[:, t : t + 1]
        A_transpose = A.T
        A_transpose_o = np.dot(A_transpose, o_t)

        # Combine prediction and likelihood
        epsilon = 1e-10
        log_prediction = np.log(s_predicted + epsilon)
        log_likelihood = np.log(A_transpose_o + epsilon)

        # Apply softmax
        input_to_softmax = log_prediction + log_likelihood
        exp_input = np.exp(input_to_softmax - np.max(input_to_softmax))
        s_sequence[:, t : t + 1] = exp_input / np.sum(exp_input)

    return s_sequence
```

## Implementation with Different Frameworks

Implementations will vary based on the framework used. Here are examples for different platforms:

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActiveInferenceModel(nn.Module):
    def __init__(self, state_dim, obs_dim):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # Initialize learnable parameters
        self.A = nn.Parameter(torch.randn(obs_dim, state_dim))
        self.B = nn.Parameter(torch.randn(state_dim, state_dim))
        self.D = nn.Parameter(torch.ones(state_dim, 1))

        # Normalize parameters
        self._normalize_parameters()

    def _normalize_parameters(self):
        """Ensure matrices are proper probability distributions."""
        # A: columns sum to 1
        self.A.data = F.softmax(self.A.data, dim=0)
        # B: columns sum to 1
        self.B.data = F.softmax(self.B.data, dim=0)
        # D: sums to 1
        self.D.data = F.softmax(self.D.data, dim=0)

    def forward(self, o_sequence):
        """Forward pass through the model."""
        batch_size, seq_len, _ = o_sequence.shape
        s_sequence = torch.zeros(
            batch_size, seq_len, self.state_dim, device=o_sequence.device
        )

        # Process each time step
        for t in range(seq_len):
            if t == 0:
                # Initial state from prior
                s_t = self.D.expand(batch_size, -1, -1)
            else:
                # Predict from previous state
                s_prev = s_sequence[:, t - 1].unsqueeze(-1)
                s_predicted = torch.bmm(self.B.expand(batch_size, -1, -1), s_prev)

                # Update with new observation
                o_t = o_sequence[:, t].unsqueeze(-1)
                A_transpose = self.A.transpose(0, 1).expand(batch_size, -1, -1)
                likelihood = torch.bmm(A_transpose, o_t)

                # Combine prediction and likelihood
                s_t = F.softmax(
                    torch.log(s_predicted + 1e-10) + torch.log(likelihood + 1e-10),
                    dim=1,
                )

            # Store state
            s_sequence[:, t] = s_t.squeeze(-1)

        return s_sequence
```

### Julia Implementation

```julia
using Flux

struct ActiveInferenceModel
    A::Matrix{Float32}  # Recognition matrix
    B::Matrix{Float32}  # Transition matrix
    D::Vector{Float32}  # Prior
end

function ActiveInferenceModel(state_dim::Int, obs_dim::Int)
    A = softmax(randn(Float32, obs_dim, state_dim), dims=1)
    B = softmax(randn(Float32, state_dim, state_dim), dims=1)
    D = softmax(ones(Float32, state_dim))
    return ActiveInferenceModel(A, B, D)
end

function (m::ActiveInferenceModel)(o_sequence::Array{Float32, 3})
    batch_size, seq_len, obs_dim = size(o_sequence)
    state_dim = size(m.D, 1)
    s_sequence = zeros(Float32, batch_size, seq_len, state_dim)
    
    for t in 1:seq_len
        if t == 1
            # Initial state from prior
            s_t = repeat(m.D, 1, batch_size)
        else
            # Predict from previous state
            s_prev = reshape(s_sequence[:, t-1, :], state_dim, batch_size)
            s_predicted = m.B * s_prev
            
            # Update with new observation
            o_t = reshape(o_sequence[:, t, :], obs_dim, batch_size)
            likelihood = m.A' * o_t
            
            # Combine prediction and likelihood
            s_t = softmax(log.(s_predicted .+ 1f-10) .+ log.(likelihood .+ 1f-10), dims=1)
        end
        
        # Store state
        s_sequence[:, t, :] = s_t'
    end
    
    return s_sequence
end
```

## Visualizing Model Behavior

It's often useful to visualize your model's behavior:

```python
import matplotlib.pyplot as plt


def plot_state_evolution(s_sequence, true_states=None):
    """Plot the evolution of state beliefs over time."""
    time_steps = s_sequence.shape[1]
    state_dim = s_sequence.shape[0]

    plt.figure(figsize=(12, 6))

    # Plot each state dimension
    for i in range(state_dim):
        plt.plot(
            range(time_steps), s_sequence[i, :], label=f"State {i + 1}", linewidth=2
        )

    # If true states are provided, plot them as dashed lines
    if true_states is not None:
        for i in range(state_dim):
            plt.plot(
                range(time_steps),
                true_states[i, :],
                "--",
                label=f"True State {i + 1}",
                linewidth=1,
            )

    plt.xlabel("Time Step")
    plt.ylabel("State Belief / Value")
    plt.title("Evolution of State Beliefs Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# Example usage
plot_state_evolution(s_sequence, true_states)
```

## Building a GNN Parser and Renderer

A complete GNN system might include components for parsing, validation, and rendering:

```mermaid
graph TD
    GF[GNN File] --> |Read| PARSER[GNN Parser]
    PARSER --> |Validate| CHECKER[Syntax Checker]
    PARSER --> |Extract| MODEL[Model Structure]
    
    MODEL --> |Generate| CODE[Code Generator]
    MODEL --> |Visualize| GRAPH[Graph Renderer]
    MODEL --> |Document| DOCS[Documentation Generator]
    
    CODE --> |Python| PY[Python Impl.]
    CODE --> |Julia| JL[Julia Impl.]
    CODE --> |Other| OT[Other Lang.]
    
    GRAPH --> |Static| SVG[SVG Diagram]
    GRAPH --> |Interactive| HTML[HTML Visualization]
    
    classDef input fill:#f96,stroke:#333,stroke-width:1px;
    classDef process fill:#9cf,stroke:#333,stroke-width:1px;
    classDef output fill:#cfc,stroke:#333,stroke-width:1px;
    
    class GF input;
    class PARSER,CHECKER,MODEL,CODE,GRAPH,DOCS process;
    class PY,JL,OT,SVG,HTML output;
```

## Best Practices

When implementing GNN models:

1. **Start with simplified models** - Begin with static models before moving to dynamic ones
2. **Test each component separately** - Verify matrix operations, probability calculations, etc.
3. **Monitor numerical stability** - Use small epsilon values to avoid log(0) errors
4. **Validate against analytic solutions** - When possible, compare with known results
5. **Use vectorized operations** - For efficiency, especially with larger models
6. **Document assumptions** - Be clear about interpretations of the model's variables
7. **Implement gradual complexity** - Add features one at a time
8. **Version control your code** - Track changes as you implement and refine

## Common Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| Numerical instability | Add small epsilon to logarithms, use log-sum-exp trick for softmax |
| Dimensionality mismatches | Carefully track dimensions of all variables, use assertions |
| Performance bottlenecks | Use vectorized operations, consider GPU acceleration |
| Convergence issues | Implement gradient clipping, adjust learning rates |
| Memory limitations | Use mini-batches, reconsider model size |
| Debugging equations | Break complex formulas into small, testable components |

## Integration with Active Inference Frameworks

Several existing frameworks can be used with GNN models:

```mermaid
graph LR
    GNN[GNN Model] --> PYMDP[pymdp]
    GNN --> PYAI[pyaixi]
    GNN --> FEP[FEP_Active_Inference]
    GNN --> INFER[inferactively]
    GNN --> CUSTOM[Custom Implementation]
    
    classDef gnn fill:#f96,stroke:#333,stroke-width:2px;
    classDef framework fill:#9cf,stroke:#333,stroke-width:1px;
    
    class GNN gnn;
    class PYMDP,PYAI,FEP,INFER,CUSTOM framework;
```

## Variable design

Name and shape state/observation variables consistently with [GNN Syntax](../reference/gnn_syntax.md#variable-naming-conventions) before wiring renderers.

## Parameter specification

Encode priors and constraints in `InitialParameterization` and matrix blocks as described in [Step 3: Setting Up Model Structure](#step-3-setting-up-model-structure).

## Optimization

Performance tuning spans pipeline configuration (Steps 11–12) and downstream numerics in each framework; see [Performance](../../performance/README.md).

## Multi-agent coordination

Compose multiple agents or factors in GNN structure and validate before render; see [GNN Multi-Agent](../advanced/gnn_multiagent.md).

## References

1. Smith, R., Friston, K.J., & Whyte, C.J. (2022). A step-by-step tutorial on active inference and its application to empirical data. Journal of Mathematical Psychology, 107, 102632.
2. Smékal, J., & Friedman, D. A. (2023). Generalized Notation Notation for Active Inference Models. Active Inference Institute. <https://doi.org/10.5281/zenodo.7803328>
3. Hesp, C., Tschantz, A., Millidge, B., Ramstead, M., Friston, K., & Smith, R. (2021). Sophisticated Inference. Neural Computation, 33(3), 713-762.
4. PyTorch Documentation: <https://pytorch.org/docs/stable/index.html>
5. NumPy Documentation: <https://numpy.org/docs/stable/>
