# Step 11: Render

## Architectural Mapping

**Orchestrator**: `src/gnn/11_render.py` (79 lines)
**Implementation Layer**: `src/gnn/render/`

## Module Description

This module provides **POMDP-aware code generation** for GNN models. It translates parsed GNN/POMDP specifications into executable simulation code for ten frameworks: PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, NumPyro and Stan are always rendered; PyTorch, bnlearn and ngclearn are also rendered but their runtimes are gated on optional installs (`supports_execution` in `src/gnn/render/framework_registry.py`). Continuous (linear-Gaussian) models render on JAX, NumPyro, PyTorch, Stan, RxInfer.jl and ngc-learn; PyMDP, ActiveInference.jl, DisCoPy and bnlearn report render status `unsupported` for them.


- **POMDP state space extraction**: extracts Active Inference matrices (A, B, C, D, E) and dimensions from GNN specs.
- **Modular injection**: injects extracted state spaces into framework-specific renderers with compatibility checks.
- **Implementation-specific outputs**: organizes code under per-model/per-framework subfolders.
- **Structured summaries**: writes `render_processing_summary.json` and overview README content under the output directory.


```mermaid
graph TD
    GNN[GNN File] --> Extract[POMDP Extraction]
    Extract --> Check{Framework<br/>Compatible?}
    Check -->|Yes| Render[Code Generation]
    Check -->|No| Skip[Skip Framework]
    Render --> Output[Generated Scripts]
```

## Agent Identity & Capabilities

# Render Module - Agent Scaffolding

## Module Overview

**Purpose**: Code generation for simulation frameworks from parsed GNN/POMDP specifications.

**Pipeline Step**: Step 11: Code rendering (11_render.py)

**Category**: Code Generation / Simulation Framework Integration

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-08-07

---

## Core Functionality

### Primary Responsibilities
1. Generate simulation code for multiple frameworks
2. Convert GNN specifications to executable implementations
3. Support framework-specific optimizations
4. Provide framework compatibility validation
5. Generate documentation and usage examples

### Key Capabilities
- Multi-framework code generation (see Supported Frameworks)
- POMDP-aware rendering via `POMDPRenderProcessor` (per-model/per-framework output folders)
- Framework compatibility checks and matrix normalization before rendering
- Structured summaries written to `render_processing_summary.json`

### Supported Frameworks

#### PyMDP (Python)
- **Purpose**: Active Inference simulation in Python
- **Features**: Full PyMDP agent implementation
- **Output**: Complete Python simulation scripts
- **Optimization**: Matrix optimization, memory efficiency

#### RxInfer.jl (Julia)
- **Purpose**: Probabilistic programming and inference
- **Features**: Genuine `@model` + `infer()` variational message passing with `free_energy = true`; per-`ModelKind` strategy dispatch (see below)
- **Output**: Julia scripts via `src/gnn/render/rxinfer/rxinfer_renderer.py`
- **Optimization**: Variational constraints, efficient inference

##### ModelKind strategy dispatch

`detect_model_kind` (`src/gnn/render/pomdp_contract.py`) classifies each spec **structurally** — from the `GNNSection` value, per-level / per-agent matrix key patterns, explicit `nr_agents` / `num_factors`, `F`/`H`/`Q`/`R` keys, and `dirichlet_[A-E]` keys. There is no free-text scanning, and a non-mapping `InitialParameterization` raises `ValueError` rather than being guessed at.

Composed specs — a linear-Gaussian block declared alongside multi-agent (or
hierarchical / learning / factored) structure — classify under several kinds
at once: `detect_model_kinds` (same module) returns the full kind set, e.g.
`{CONTINUOUS, MULTI_AGENT}` for `input/gnn_files/continuous/multi_agent_lgssm.md`,
while `detect_model_kind` remains the max-precedence single winner. A
composed set is never rendered as the winner family alone: the render step
reports every framework `unsupported-composition` (counted under
`unsupported_framework_renderings`, never a failure) rather than silently
dropping the other family.

The canonical renderer (`rxinfer_renderer.py`) then dispatches to a per-kind strategy in `src/gnn/render/rxinfer/model_strategies.py`. Each strategy emits a genuine Julia script running `infer()`; each raises `ValueError` naming the missing parameterization when a spec reaches it without the matrices its `@model` requires.

| ModelKind | Strategy | Generated model |
|---|---|---|
| FLAT | `FlatStrategy` | `pomdp_model` — batch smoothing by default, or per-timestep filtering when `inference_mode: online` is declared in ModelParameters (or passed as a render option) |
| HIERARCHICAL | `HierarchicalStrategy` | `hierarchical_pomdp_model` for two-level exemplars (context latent coupled into the fast prior via `A_level2`, mean-field constraints + initialization); 3+ declared levels render as the documented joint composition |
| FACTORED | `FactoredStrategy` | `factored_pomdp_model` — native mean-field two-factor model with multi-parent likelihood (`DiscreteTransition(s1, A_m0, s2)`), per-factor posteriors |
| CONTINUOUS | `ContinuousStrategy` | `continuous_pomdp_model` — linear-Gaussian state space built from `F`/`H`/`Q`/`R` + Gaussian prior in InitialParameterization; beliefs are posterior means, VFE validation is sign-agnostic |
| LEARNING | `LearningStrategy` | `learning_pomdp_model` — likelihood matrix `A` learned as a latent `DirichletCollection` from `dirichlet_A` pseudo-counts, jointly with states; reports learned-A mean and prior/posterior distance to the true A |
| MULTI_AGENT | `MultiAgentStrategy` | joint composition with the true kind stamped in `runtime_metadata`; per-agent marginals are recovered downstream from the `state_factors` echo |

**Online mode (FLAT).** Batch is the default: offline smoothing over the full observation sequence, with expected free energy and the policy computed post-hoc from the smoothed posteriors. Under `inference_mode: online` the script instead calls `infer()` per timestep on the observation prefix, and the *filtered* posterior drives EFE plus habit-prior action selection.

**Action selection.** Generated scripts select actions by `softmax(log E − γ·EFE)`, where `E` is the habit/policy prior.

**Conventions baked into generated scripts.** `B` is ordered `(next_state, previous_state, action)` — the scripts embed `const B_TENSOR_ORDER = "next_state_previous_state_action"`. In the results payload, `true_states[t]` is the state that *emitted* observation `t`, so it is timing-aligned with `beliefs[t]`.

Continuous exemplars (`input/gnn_files/continuous/`) are linear-Gaussian models — `F`/`H`/`Q`/`R`, `prior_mean`/`prior_cov`, optional closed-loop `goal_mean`/`control_gain` — with no discrete stand-in. `detect_model_kind` routes them as `CONTINUOUS`: RxInfer.jl (native LGSSM strategy), JAX, NumPyro, PyTorch, ngc-learn (shared Kalman-filter generator in `render/continuous_script.py`) and Stan (Kalman marginal-likelihood program) render and execute them; PyMDP, ActiveInference.jl, DisCoPy and bnlearn return the render status `unsupported` (counted separately from failures, never executed). Continuous results echo `state_factors` and `observation_modalities` as empty because a linear-Gaussian model has no categorical factors. The one composed exception is `multi_agent_lgssm.md`, which additionally declares `nr_agents: 2`: `detect_model_kinds` returns `{CONTINUOUS, MULTI_AGENT}` and every framework reports the `unsupported-composition` status (refused, never rendered as one family) until per-agent continuous rendering lands.

#### ActiveInference.jl (Julia)
- **Purpose**: Active Inference framework implementation
- **Features**: Complete Active Inference agent
- **Output**: Julia simulation scripts
- **Optimization**: Hierarchical processing, temporal dynamics

#### DisCoPy (Python)
- **Purpose**: Categorical diagrams for compositional models
- **Features**: String diagram generation
- **Output**: Python DisCoPy diagrams
- **Optimization**: Categorical composition, type checking

#### JAX (Python)
- **Purpose**: High-performance numerical computing
- **Features**: JIT compilation, automatic differentiation
- **Output**: JAX-optimized simulation code
- **Optimization**: GPU acceleration, vectorization

#### PyTorch (Python)
- **Purpose**: Neural integration backend
- **Output**: Python scripts under `pytorch/` when the renderer is available

#### NumPyro (Python)
- **Purpose**: Probabilistic programming backend
- **Output**: Python scripts under `numpyro/` (always rendered; NumPyro is a core dependency)

#### Stan (Stan)
- **Purpose**: Probabilistic programming backend
- **Output**: `stan/<model_name>_stan.stan` plus the cmdstanpy driver `stan/<model_name>_stan.py` (always rendered; HMM forward-algorithm program for discrete models, Kalman marginal-likelihood program for continuous models)

#### bnlearn (Python)
- **Purpose**: Bayesian network structure and parameter learning, exact inference, causal discovery
- **Output**: Python scripts under `bnlearn/` (`<model>_bnlearn.py`), generated by `render.bnlearn.bnlearn_renderer.generate_bnlearn_code`
- **Notes**: Registered in `render.framework_registry` as POMDP-compatible with no required matrices; `render.health.check_renderers()` reports it alongside the other nine categorical backends; Step 12 executes its programs via `src/gnn/execute/bnlearn/`

#### ngc-learn (Python)
- **Purpose**: Predictive-processing neural simulation backend (continuous linear-Gaussian models only)
- **Features**: Kalman-filter LGSSM scripts generated by the shared `render.continuous_script` generator (numerics byte-identical to the JAX backend); codegen-only — the emitted script imports `ngclearn` behind an import-or-exit guard, never the renderer
- **Output**: Python scripts under `ngclearn/` (`<model>_ngclearn.py`)
- **Notes**: Registry entry is `continuous_only=True` — discrete POMDPs are first-class `unsupported` with the message `discrete POMDP: ngclearn supports continuous linear-Gaussian models only`; Step 12 executes its scripts via `src/gnn/execute/ngclearn/`, skipping with an install hint until `uv sync --extra ngclearn` (py3.12 marker-gated)

---

## API Reference

### Public Functions

#### `process_render(target_dir: Path, output_dir: Path, verbose: bool = False, frameworks=None, strict_validation: bool = True, **kwargs) -> bool`
**Description**: Main rendering processing function called by orchestrator (11_render.py). Processes GNN files and generates code for multiple simulation frameworks.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to process
- `output_dir` (Path): Output directory for rendered files
- `verbose` (bool): Enable verbose logging (default: False)
- `frameworks` (Optional[List[str]]): Restrict to a subset of frameworks (default: all available)
- `strict_validation` (bool): Passed to POMDP extraction (`True` by default)
- `**kwargs`: Additional options forwarded to framework renderers (e.g. `timesteps`, `simulation_params`)

**Returns**: `bool` - True if processing succeeded, False otherwise

**Example**:
```python
from gnn.render import process_render
from pathlib import Path

success = process_render(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/11_render_output"),
    verbose=True,
    frameworks=["pymdp", "rxinfer"],
    strict_validation=True,
)
```

#### `render_gnn_spec(gnn_spec: Dict[str, Any], target: str, output_directory: Union[str, Path], options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`
**Description**: Render a GNN specification dictionary to a target framework.

**Parameters**:
- `gnn_spec` (Dict[str, Any]): Parsed GNN specification dictionary
- `target` (str): Target framework ("pymdp", "rxinfer", "activeinference_jl", "jax", "discopy", "bnlearn", "pytorch", "numpyro", "stan")
- `output_directory` (Union[str, Path]): Output directory for generated code
- `options` (Optional[Dict[str, Any]]): Framework-specific options (default: None)

**Returns**: `Tuple[bool, str, List[str]]` - Tuple containing:
- `success` (bool): Whether rendering succeeded
- `message` (str): Status message
- `generated_files` (List[str]): List of generated file paths

**Location**: `src/gnn/render/processor/rendering.py`

### Canonical POMDP renderers

For POMDP targets, Step 11 and `render_gnn_spec(...)` route to the same validated renderers:

- `render_gnn_to_pymdp(...)`
- `render_gnn_to_rxinfer(...)`
- `render_gnn_to_activeinference_jl(...)`

The shared contract is `canonical_pomdp_v1`, with B stored as `(next_state, previous_state, action)`.

#### `get_module_info() -> Dict[str, Any]`
**Description**: Get information about the render module capabilities.

**Returns**: `Dict[str, Any]` - Dictionary with module information containing:
- `name` (str): Module name
- `version` (str): Module version
- `description` (str): Module description
- `supported_targets` (List[str]): List of supported target frameworks
- `available_targets` (List[str]): List of currently available targets
- `features` (List[str]): List of available features
- `supported_formats` (List[str]): List of supported output formats
- `processing_modes` (List[str]): List of available processing modes

**Location**: `src/gnn/render/processor/metadata.py`

#### `get_available_renderers() -> Dict[str, Dict[str, Any]]`
**Description**: Get information about available renderers for each framework.

**Returns**: `Dict[str, Dict[str, Any]]` - Dictionary mapping framework names to renderer information:
- Each framework entry contains:
  - `name` (str): Framework name
  - `description` (str): Framework description
  - `language` (str): Target language
  - `file_extension` (str): Output file extension
  - `supported_features` (List[str]): List of supported features
  - `function` (str): Function name for rendering
  - `output_format` (str): Output format type
  - `pomdp_compatible` (bool): Whether POMDP-aware processing is supported

**Location**: `src/gnn/render/processor/metadata.py`

#### `validate_pomdp_for_rendering(pomdp_space: Any) -> Tuple[bool, List[str]]`
**Description**: Validate POMDP state space structure for rendering compatibility.

**Parameters**:
- `pomdp_space` (Any): POMDP state space object to validate

**Returns**: `Tuple[bool, List[str]]` - Tuple containing:
- `is_valid` (bool): Whether POMDP structure is valid
- `errors` (List[str]): List of validation error messages

**Location**: `src/gnn/render/processor/parsing.py`

#### `normalize_matrices(pomdp_space: Any, logger) -> Any`
**Description**: Normalize POMDP matrices for consistent rendering.

**Parameters**:
- `pomdp_space` (Any): POMDP state space object
- `logger`: Logger instance for logging

**Returns**: `Any` - Normalized POMDP state space object

**Location**: `src/gnn/render/processor/parsing.py`

---

## Dependencies

### Required Dependencies
- `numpy` - Numerical computations
- `pathlib` - Path manipulation
- `typing` - Type hints

### Framework-Specific Dependencies
- **PyMDP**: `pymdp` package
- **RxInfer.jl**: Julia with the RxInfer.jl package (committed `Project.toml` + `Manifest.toml` under `src/gnn/execute/rxinfer/` pin RxInfer 5.5.0)
- **ActiveInference.jl**: Julia with ActiveInference.jl package
- **DisCoPy**: `discopy` package
- **JAX**: `jax`, `jaxlib` packages

### Internal Dependencies
- `gnn.parsers` - GNN parsing and validation
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

---

## Configuration

### Framework Configuration
```python
PYMDP_CONFIG = {
    "inference_algorithm": "VMP",
    "learning_rate": 0.1,
    "num_iterations": 100,
    "convergence_threshold": 1e-6,
}

RXINFER_CONFIG = {
    "inference_engine": "rxinfer_jl",
    "optimization": "auto",
    "constraints": "default",
}
```

Configuration is primarily controlled by the Step 11 orchestrator (`src/gnn/11_render.py`) and forwarded parameters to `process_render(...)`. Avoid documenting configuration keys that are not backed by code.

---

## Usage Examples

### Basic Framework Rendering
```python
from gnn.render import render_gnn_spec

# Render GNN to PyMDP
success, message, files = render_gnn_spec(
    gnn_spec=model_data,
    target="pymdp",
    output_directory="output/11_render_output",
    options={"include_examples": True},
)
```

### Multi-Framework Rendering
```python
from gnn.render import render_gnn_spec

for framework in ["pymdp", "rxinfer", "activeinference_jl"]:
    success, message, files = render_gnn_spec(
        model_data, framework, "output/11_render_output"
    )
    assert success, message
```

### Custom Framework Options
```python
# Framework-specific options
options = {
    "pymdp": {"inference_algorithm": "VMP", "num_iterations": 200},
    "rxinfer": {"constraints": "custom", "optimization": "performance"},
}
```

---

## Output Specification

### Output Products
- Framework artifacts are written under per-model/per-framework subfolders (POMDP-aware mode), typically one primary script per framework:
  - `pymdp/<model_name>_pymdp.py`
  - `rxinfer/<model_name>_rxinfer.jl`
  - `activeinference_jl/<model_name>_activeinference.jl`
  - `jax/<model_name>_jax.py`
  - `discopy/<model_name>_discopy.py`
  - `numpyro/<model_name>_numpyro.py`
  - `stan/<model_name>_stan.stan` + `stan/<model_name>_stan.py` (cmdstanpy driver)
  - `pytorch/<model_name>_pytorch.py` and `bnlearn/<model_name>_bnlearn.py` (rendered; runtimes gated on manual installs)
- `render_processing_summary.json` - Processing summary

### Output Directory Structure
```
output/11_render_output/
├── render_processing_summary.json
└── [model_stem]/
    ├── pymdp/
    ├── rxinfer/
    ├── activeinference_jl/
    ├── jax/
    ├── discopy/
    ├── bnlearn/
    ├── pytorch/        # if available
    ├── numpyro/        # if available
    └── stan/           # if available
```

---

## Performance Characteristics

Performance is tracked by the pipeline execution summaries and render summary JSON output. Avoid hard-coding numeric claims in docs unless they are generated from current benchmark outputs.

---

## Error Handling

### Generation Failures
1. **Syntax Errors**: Invalid GNN specification
2. **Framework Errors**: Framework-specific generation issues
3. **Dependency Errors**: Missing framework packages
4. **Configuration Errors**: Invalid framework options

### Recovery Strategies
- **Framework Recovery**: Try alternative frameworks
- **Template Simplification**: Use simpler templates
- **Partial Generation**: Generate what is possible
- **Error Documentation**: Provide detailed error reports

---

## Integration Points

### Orchestrated By
- **Script**: `11_render.py` (Step 11)
- **Function**: `process_render()`

### Imports From
- `gnn.parsers` - GNN parsing and validation
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

### Imported By
- `tests.test_render_*` - Render tests
- `execute.executor` - Execution framework integration

### Data Flow
```
GNN Parsing → Model Validation → Framework Selection → Code Generation → Framework-Specific Optimization → Output Generation
```

---

## Testing

### Test Files
- `tests/render/test_render_integration.py` - Integration tests
- `tests/render/test_render_overall.py` - Overall functionality tests
- `tests/render/test_render_performance.py` - Performance tests

### Test Coverage
- Measure: `uv run --extra dev python -m pytest tests/render/ --cov=gnn.render --cov-report=term-missing` (do not treat fixed percentages in this doc as canonical).

### Key Test Scenarios
1. Multi-framework code generation
2. Framework-specific optimizations
3. Error handling and recovery
4. Performance benchmarking
5. Integration with execution step

---

## MCP Integration

### Tools Registered
- `process_render` - Run Step 11 over a directory of GNN files
- `list_render_frameworks` - List registered render targets and availability
- `render_gnn_to_format` - Render one GNN file to one framework
- `get_render_module_info` - Module metadata

### Tool Endpoints
```python
@mcp_tool("render.generate_pymdp")
def generate_pymdp_tool(model_data, options=None):
    """Generate PyMDP simulation code"""
    # Implementation
```

### MCP File Location
- `src/gnn/render/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Framework-specific rendering fails
**Symptom**: Code generation fails for specific framework  
**Cause**: Missing framework dependencies or invalid GNN model structure  
**Solution**: 
- Check framework dependencies are installed
- Verify GNN model has required sections for framework
- Use `--verbose` flag for detailed error messages
- Check framework-specific requirements in documentation

#### Issue 2: POMDP validation errors
**Symptom**: POMDP-aware rendering reports validation errors  
**Cause**: GNN model missing POMDP-required components or invalid structure  
**Solution**:
- Ensure GNN model has complete state space, observations, actions
- Verify connections follow POMDP structure (s->o, s->s, a->s)
- `strict_validation` is a `process_render()` Python API parameter (default `True`), not a CLI flag; call `process_render(..., strict_validation=False)` directly for lenient POMDP validation. The CLI does expose `--strict-framework-success`, which instead controls whether a failed framework render fails the whole step
- Review POMDP requirements in documentation

#### Issue 3: Generated code doesn't execute
**Symptom**: Rendered code has syntax errors or import failures  
**Cause**: Framework version mismatch or template issues  
**Solution**:
- Verify framework versions match requirements
- Check generated code for syntax errors
- Review framework-specific documentation
- Report template issues if systematic

---

## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [PyMDP Integration](../../../docs/pymdp/)
- [RxInfer Integration](../../../docs/rxinfer/)
- [ActiveInference.jl Integration](../../../docs/activeinference_jl/)
- [DisCoPy Integration](../../../docs/discopy/)

### External Resources
- [PyMDP Framework](https://github.com/infer-actively/pymdp)
- [RxInfer.jl](https://github.com/biaslab/RxInfer.jl)
- [ActiveInference.jl](https://github.com/ComputationalPsychiatry/ActiveInference.jl)
- [DisCoPy](https://github.com/oxford-quantum-group/discopy)
- [JAX Documentation](https://jax.readthedocs.io/)

---

**Maintainer**: GNN Pipeline Team

---
## Documentation
- **[README](../../../src/gnn/render/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/render/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/render/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/render/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/render](../../../src/gnn/render)
