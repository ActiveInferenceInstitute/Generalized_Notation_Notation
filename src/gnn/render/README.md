# POMDP-Aware Render Module

This module provides **POMDP-aware code generation** for GNN models. It translates parsed GNN/POMDP specifications into executable simulation code for multiple frameworks including PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, Stan, and bnlearn.

## Key Features

- **POMDP state space extraction**: extracts Active Inference matrices (A, B, C, D, E) and dimensions from GNN specs.
- **Canonical POMDP contract**: normalizes `A/B/C/D/E`, provenance, labels, and B tensor order before framework rendering.
- **Modular injection**: injects extracted state spaces into framework-specific renderers with validation checks.
- **Implementation-specific outputs**: organizes code under per-model/per-framework subfolders.
- **Structured summaries**: writes `render_processing_summary.json` and overview README content under the output directory.

## Public admission and discovery

Python, CLI, REST and MCP rendering validate backend options through
`admission.py` before code generation. Only consumed option keys are accepted;
booleans do not stand in for integers, numeric strings are rejected, and limits
must be finite and in range. Configured `render.backend_options` merge per key
with explicit options; explicit values win and unrelated configured keys remain.
Scientific/model-kind and allocation constraints remain with the native adapter.
Unsupported Gaussian/categorical or composed model semantics return explicit
refusals rather than substituting another model or uncertainty family.

Step 11 framework choices derive from `framework_registry.py`; `render_gnn_spec`
also preserves the distinct `jax_pomdp` and `discopy_combined` generator targets.
Direct DisCoPy `matrix_permutations` require the Step 11 interface. Renderer
registration permits code generation and does not prove an installed native
runtime; MCP discovery labels that scope and leaves native readiness unknown.
MCP `render_gnn_to_format` renders exactly one requested framework and lists only
the primary current artifact verified by its receipt, excluding inherited files.

Migration from 4.0: replace coerced values with their declared types, remove
previously ignored option keys, and choose an explicit nonempty framework list.
Omitted framework selection retains the maintained default preset.

## POMDP Processing Pipeline

<!-- 5 of 10 renderers shown; the full renderer set lives in framework_registry.py -->
```mermaid
graph TD
    GNN[GNN File] --> Extract[POMDP Extraction]
    Extract --> Check{Framework<br/>Compatible?}
    
    Check -->|Yes| Inject[Modular Injection]
    Check -->|No| Error[Validation Error]
    
    Inject --> PyMDP[PyMDP Renderer]
    Inject --> RxInfer[RxInfer.jl Renderer]
    Inject --> ActInf[ActiveInference.jl Renderer]
    Inject --> JAX[JAX Renderer]
    Inject --> DisCoPy[DisCoPy Renderer]
    
    PyMDP --> Code1[Python Code]
    RxInfer --> Code2[Julia Code]
    ActInf --> Code3[Julia Code]
    JAX --> Code4[Python Code]
    DisCoPy --> Code5[Python Code]
```

### Framework Rendering Architecture

<!-- 5 of 10 renderers shown; the full renderer set lives in framework_registry.py -->

```mermaid
graph TB
    subgraph "Input Processing"
        GNNFile[GNN Specification]
        POMDPExtract[POMDP State Space Extraction]
        Validate[POMDP Validation]
    end
    
    subgraph "Framework Renderers"
        PyMDPRenderer[PyMDP Renderer]
        RxInferRenderer[RxInfer.jl Renderer]
        ActInfRenderer[ActiveInference.jl Renderer]
        JAXRenderer[JAX Renderer]
        DisCoPyRenderer[DisCoPy Renderer]
    end
    
    subgraph "Code Generation"
        PyMDPCode[Python Simulation Code]
        RxInferCode[Julia Code]
        ActInfCode[Julia Simulation Code]
        JAXCode[JAX Optimized Code]
        DisCoPyCode[DisCoPy Diagram Code]
    end
    
    subgraph "Output Organization"
        PyMDPOut[pymdp_gen/]
        RxInferOut[rxinfer/]
        ActInfOut[activeinference_jl/]
        JAXOut[jax/]
        DisCoPyOut[discopy/]
    end
    
    GNNFile --> POMDPExtract
    POMDPExtract --> Validate
    Validate --> PyMDPRenderer
    Validate --> RxInferRenderer
    Validate --> ActInfRenderer
    Validate --> JAXRenderer
    Validate --> DisCoPyRenderer
    
    PyMDPRenderer --> PyMDPCode
    RxInferRenderer --> RxInferCode
    ActInfRenderer --> ActInfCode
    JAXRenderer --> JAXCode
    DisCoPyRenderer --> DisCoPyCode
    
    PyMDPCode --> PyMDPOut
    RxInferCode --> RxInferOut
    ActInfCode --> ActInfOut
    JAXCode --> JAXOut
    DisCoPyCode --> DisCoPyOut
```

### Module Integration Flow

```mermaid
flowchart LR
    subgraph "Pipeline Step 11"
        Step11[11_render.py Orchestrator]
    end
    
    subgraph "Render Module"
        Processor[processor.py]
        POMDPProc[pomdp_processor.py]
        Contract[pomdp_contract.py]
        Generators[generators.py]
    end
    
    subgraph "Framework Renderers"
        PyMDP[pymdp/]
        RxInfer[rxinfer/]
        ActInf[activeinference_jl/]
        JAX[jax/]
        DisCoPy[discopy/]
    end
    
    subgraph "Downstream Step"
        Step12[Step 12: Execute]
    end
    
    Step11 --> Processor
    Processor --> POMDPProc
    POMDPProc --> Contract
    Processor --> Generators
    
    POMDPProc --> PyMDP
    POMDPProc --> RxInfer
    POMDPProc --> ActInf
    POMDPProc --> JAX
    POMDPProc --> DisCoPy
    
    PyMDP -->|Generated Code| Step12
    RxInfer -->|Generated Code| Step12
    ActInf -->|Generated Code| Step12
    JAX -->|Generated Code| Step12
    DisCoPy -->|Generated Code| Step12
```

## Module Structure

```
src/gnn/render/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
├── mcp.py                         # Model Context Protocol integration
├── framework_registry.py          # Canonical framework inventory + availability
├── contracts.py                   # Framework output contract checks for rendered code
├── render.py                      # Standalone CLI renderer entry point
├── health.py                      # Renderer importability health check
├── processor.py                   # Main render processor (Step 11 entry)
├── pomdp_contract.py              # Canonical POMDP matrix contract
├── pomdp_processor.py             # POMDP state space injection into renderers
├── pomdp_math.py                  # Factored-POMDP math helpers
├── matrix_utils.py                # Shared matrix normalization/validation
├── multi_agent_common.py          # Shared multi-agent detection helpers
├── continuous_common.py           # Shared linear-Gaussian spec contract
├── continuous_script.py           # Continuous (LGSSM) script generation
├── naming.py                      # Shared output naming + atomic writes
├── spec_matrices.py               # Shared discrete matrix extraction/literals
├── generators.py                  # Standalone generators (bnlearn, discopy, pre-package pymdp)
├── pymdp_template.py              # PyMDP template definitions
├── visualization_suite.py         # Data export + visualization suite
├── pymdp/                         # PyMDP code generation
│   ├── __init__.py               # PyMDP module initialization
│   ├── pymdp_renderer.py         # PyMDP renderer
│   ├── pymdp_templates.py        # PyMDP code templates
│   └── pymdp_utils.py            # PyMDP utilities
├── rxinfer/                       # RxInfer.jl code generation
│   ├── __init__.py               # RxInfer module initialization
│   ├── rxinfer_renderer.py       # Canonical renderer — genuine @model + infer()
│   ├── gnn_parser.py             # GNN parser for RxInfer
│   └── model_contracts.py        # Matrix parsers, topology config, fail-closed validation
├── activeinference_jl/            # ActiveInference.jl code generation
│   ├── __init__.py               # ActiveInference.jl module initialization
│   └── activeinference_renderer.py # ActiveInference.jl renderer
├── jax/                           # JAX code generation
│   ├── __init__.py               # JAX module initialization
│   ├── jax_renderer.py           # JAX renderer
│   └── templates/                 # JAX code templates
│       ├── __init__.py           # Templates initialization
│       ├── combined_template.py   # Combined JAX template
│       ├── general_template.py    # General JAX template
│       └── pomdp_template.py     # POMDP JAX template
├── pytorch/                       # PyTorch code generation
│   └── pytorch_renderer.py       # PyTorch renderer
├── numpyro/                       # NumPyro code generation
│   └── numpyro_renderer.py       # NumPyro renderer
├── stan/                          # Stan code generation
│   └── stan_renderer.py          # Stan renderer (program + cmdstanpy driver)
└── discopy/                       # DisCoPy code generation
    ├── __init__.py               # DisCoPy module initialization
    ├── discopy_renderer.py       # DisCoPy renderer
    ├── translator.py              # GNN to DisCoPy translator
```

## Module exports

`gnn.render.__all__` (from `src/gnn/render/__init__.py`):

- Core: `process_render`, `render_gnn_spec`, `get_available_renderers`, `get_module_info`
- Generators: `generate_pymdp_code`, `generate_rxinfer_code`, `generate_activeinference_jl_code`, `generate_discopy_code`
- Per-framework renders: `render_gnn_to_pymdp`, `render_gnn_to_rxinfer`, `render_gnn_to_discopy`, `render_gnn_to_activeinference_jl`, `render_gnn_to_pytorch`, `render_gnn_to_numpyro`, `render_stan`
- Renderer classes: `PyMDPRenderer`, `JAXRenderer`
- POMDP processing: `POMDPRenderProcessor`, `process_pomdp_for_frameworks`
- Utilities: `get_supported_frameworks`, `validate_render`

## Core Components

### Step 11 entrypoint (`processor.py`)

The pipeline calls:

- `render.process_render(target_dir, output_dir, verbose=False, frameworks=None, strict_validation=True, strict_framework_success=False, **kwargs) -> bool | int`

For POMDP targets, `render_gnn_spec(..., "pymdp" | "rxinfer" | "activeinference_jl", ...)` uses the same canonical renderers as Step 11. The shared contract is `canonical_pomdp_v1`, with B stored as `(next_state, previous_state, action)`.

In POMDP-aware mode, `process_render` delegates per-model/per-framework rendering to `POMDPRenderProcessor` in `pomdp_processor.py`.
Explicit framework selections such as `"pymdp"` or `"pymdp,jax"` require every requested renderer to succeed for every input file. Use `strict_framework_success=True` to apply that same policy to `"all"` or `"lite"` runs.

### Framework backends

Backend-specific renderers live under:

- `src/gnn/render/pymdp/`
- `src/gnn/render/rxinfer/`
- `src/gnn/render/activeinference_jl/`
- `src/gnn/render/jax/`
- `src/gnn/render/discopy/`
- additional maintained backends: `src/gnn/render/pytorch/`, `src/gnn/render/numpyro/`, `src/gnn/render/stan/` (runnable HMM / LGSSM programs plus cmdstanpy drivers), and generator-backed `bnlearn` (executed by `src/gnn/execute/bnlearn/`; skips without its runtime)
- `src/gnn/render/ngclearn/` — codegen-only 4th Python backend of the shared continuous generator (`render/continuous_script.py`); run-time extra `ngclearn`, py3.12 marker-gated

### Model kinds

`render.pomdp_contract.detect_model_kind` classifies each spec. Discrete kinds
(flat, factored, hierarchical, multi-agent, learning) are canonicalised to
`A/B/C/D[/E]` before rendering. `CONTINUOUS` specs (`F/H/Q/R`,
`prior_mean/prior_cov`, optional `goal_mean/control_gain`) bypass
canonicalisation: `render/continuous_common.py` validates the block and
`render/continuous_script.py` generates the JAX / NumPyro / PyTorch / ngc-learn Kalman-filter
scripts; Stan and RxInfer.jl have their own continuous programs. Frameworks
whose registry entry has `supports_continuous: false` (PyMDP,
ActiveInference.jl, DisCoPy, bnlearn) return `{"unsupported": true, "status":
"unsupported"}` for continuous models — counted under
`unsupported_framework_renderings`, never as failures, never executed.

`STRUCTURAL` specs (issue #111) declare boundary structure only — a
Markov-blanket wrapper with neither discrete `A/B/C/D[/E]` nor continuous
`F/H/Q/R` parameterization. They are render-only / informational: every
framework reports `{"unsupported": true, "status": "unsupported"}` with the
`structural-spec: no renderable form` reason, counted under
`unsupported_framework_renderings` — never rendered, never a failure. In
`render_gnn_spec`, targets that canonicalise to a discrete parameterization
(`pymdp`, `rxinfer`, `activeinference_jl`, `pytorch`, `numpyro`,
`jax`/`jax_pomdp`) reject a wrapper with the same `structural-spec` message
instead of attempting a discrete render; graph-backed targets (`bnlearn`,
`stan`, `discopy`) render the declared structure legitimately and are not
gated.

Composed specs declare more than one render family — e.g. a linear-Gaussian
`F/H/Q/R` block declared alongside `nr_agents > 1`.
`render.pomdp_contract.detect_model_kinds` returns the full composed kind set
(`{CONTINUOUS, MULTI_AGENT}` for that spec) while `detect_model_kind` keeps
returning the max-precedence single winner for per-kind dispatch. No backend
renders a composition whole: the render step reports every framework
`{"unsupported": true, "status": "unsupported"}` with the
`unsupported-composition:` reason — the same unsupported accounting
structural wrappers get, counted under `unsupported_framework_renderings`
and never a failure — rather than silently rendering the winner family with
the other dropped. The one exception is the composed `{FACTORED, CONTINUOUS}`
set: it renders on JAX as per-factor linear-Gaussian blocks
(`render.continuous_script`), while every other composition stays refused.
The composed exemplar is
`input/gnn_files/continuous/multi_agent_lgssm.md`.

The supported framework inventory is defined in `src/gnn/render/framework_registry.py` and consumed by `health.py`, `__init__.py`, `processor.py`, `pomdp_processor.py`, and `mcp.py`.

## Usage Examples

### Basic Code Generation

```python
from pathlib import Path
from gnn.render import process_render

success = process_render(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/11_render_output"),
    verbose=True,
    frameworks="pymdp",
)
print("ok" if success else "render completed with issues")
```

### Framework-Specific Rendering

```python
from pathlib import Path
from gnn.render import render_gnn_spec

success, msg, artifacts = render_gnn_spec(
    gnn_spec=parsed_spec,
    target="pymdp",
    output_directory=Path("output/11_render_output/single/pymdp"),
)
print(success, msg, artifacts)
```

### Multi-Framework Generation

```python
from pathlib import Path
from gnn.render import process_render

process_render(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/11_render_output"),
    frameworks=["pymdp", "jax", "discopy"],
    strict_validation=True,
)
```

## Rendering Pipeline

Rendering is driven by:

- extraction + normalization (`gnn.extract.pomdp_extractor`, `render.processor.normalize_matrices`)
- per-framework rendering (`render.pomdp_processor.POMDPRenderProcessor`)
- structured summaries (`render_processing_summary.json` and per-framework README files)

## Integration with Pipeline

### Pipeline Step 11: Code Rendering

`src/gnn/11_render.py` is a thin orchestrator that calls `render.process_render(...)`.

### Output Structure

The POMDP-aware render system creates **implementation-specific output subfolders** for organized code generation:

```
output/11_render_output/
├── [model_name]/                          # Model-specific directory
│   ├── pymdp/                            # PyMDP implementation
│   │   ├── [model_name]_pymdp.py         # Generated simulation script
│   │   ├── README.md                     # Framework-specific documentation
│   │   └── processing_summary.json       # Processing details
│   ├── rxinfer/                          # RxInfer.jl implementation  
│   │   ├── [model_name]_rxinfer.jl       # Generated Julia script
│   │   ├── README.md                     # Framework-specific documentation
│   │   └── processing_summary.json       # Processing details
│   ├── activeinference_jl/               # ActiveInference.jl implementation
│   │   ├── [model_name]_activeinference.jl # Generated Julia script
│   │   ├── README.md                     # Framework-specific documentation
│   │   └── processing_summary.json       # Processing details
│   ├── jax/                              # JAX implementation
│   │   ├── [model_name]_jax.py           # Generated JAX script
│   │   ├── README.md                     # Framework-specific documentation
│   │   └── processing_summary.json       # Processing details
│   ├── discopy/                          # DisCoPy implementation
│   │   ├── [model_name]_discopy.py       # Generated diagram script
│   │   ├── README.md                     # Framework-specific documentation
│   │   └── processing_summary.json       # Processing details
│   └── processing_summary.json           # Overall model processing summary
├── README.md                             # Overall processing documentation
└── render_processing_summary.json        # Complete processing results
```

## POMDP Processing Features

### POMDP State Space Extraction (`gnn.extract.pomdp_extractor`)

The system automatically extracts Active Inference structures from GNN specifications:

**Extracted Components:**

- **A Matrix**: Likelihood mapping P(o|s) - observations given states
- **B Matrix**: Transition dynamics P(s'|s,a) - next states given current states and actions
- **C Vector**: Preferences over observations (log-probabilities)
- **D Vector**: Prior beliefs over initial hidden states
- **E Vector**: Policy priors (habits) over actions

**State Space Variables:**

- Hidden states, observations, actions with dimensions and types
- Connections and relationships between variables
- Ontology mappings to Active Inference concepts

**Example POMDP Extraction:**

```python
from gnn.extract.pomdp_extractor import extract_pomdp_from_file

# Extract POMDP from GNN file
pomdp_space = extract_pomdp_from_file("input/gnn_files/discrete/actinf_pomdp_agent.md")

print(f"Model: {pomdp_space.model_name}")
print(f"States: {pomdp_space.num_states}")
print(f"Observations: {pomdp_space.num_observations}")
print(f"Actions: {pomdp_space.num_actions}")
print(f"State factors: {pomdp_space.state_factors}")
print(f"Observation modalities: {pomdp_space.observation_modalities}")
print(f"Matrix provenance: {pomdp_space.matrix_provenance}")
```

### Modular Injection System (`pomdp_processor.py`)

The processor validates POMDP compatibility and injects state spaces into framework renderers:

**Compatibility Validation:**

- Checks required matrices are present for each framework
- Validates matrix dimensions and consistency
- Warns about framework limitations (e.g., multi-modality support)

**Framework-Specific Processing:**

```python
from gnn.render.pomdp_processor import process_pomdp_for_frameworks

# Process for all frameworks
results = process_pomdp_for_frameworks(
    pomdp_space=pomdp_space,
    output_dir="output/11_render_output/",
    frameworks=["pymdp", "activeinference_jl", "rxinfer"],
    strict_validation=True,
)

# Results include success/failure for each framework
for framework, result in results["framework_results"].items():
    status = "✅" if result["success"] else "❌"
    print(f"{status} {framework}: {result['message']}")
```

### Structured Documentation Generation

Each framework rendering includes:

- **Model Information**: Extracted from GNN annotations
- **POMDP Dimensions**: States, observations, actions
- **Active Inference Matrices**: Available matrices with dimensions
- **Generated Files**: List of created simulation scripts
- **Usage Instructions**: Framework-specific execution guidance
- **Warnings**: Any compatibility or processing issues

## Framework Features

### PyMDP Framework

- **Purpose**: Python-based Active Inference simulation
- **Strengths**: Easy to use, comprehensive documentation
- **Use Cases**: Research, prototyping, education
- **Integration**: NumPy, SciPy, Matplotlib

### RxInfer.jl Framework

- **Purpose**: Julia-based probabilistic programming
- **Strengths**: High performance, type safety
- **Use Cases**: Advanced research, high-performance computing
- **Integration**: Julia ecosystem, ReactiveMP

### ActiveInference.jl Framework

- **Purpose**: Julia-based Active Inference implementation
- **Strengths**: Native Julia, high performance
- **Use Cases**: Research, advanced simulations
- **Integration**: Julia ecosystem, optimization

### JAX Framework

- **Purpose**: High-performance numerical computing
- **Strengths**: GPU acceleration, automatic differentiation
- **Use Cases**: Research, custom algorithms
- **Integration**: NumPy, TensorFlow, optimization

### DisCoPy Framework

- **Purpose**: Categorical quantum computing
- **Strengths**: Diagrammatic reasoning, quantum algorithms
- **Use Cases**: Quantum computing, categorical methods
- **Integration**: TensorFlow, quantum libraries

## Configuration Options

### Rendering Settings

```python
# Rendering configuration
config = {
    "default_framework": "pymdp",  # Default rendering framework
    "template_enabled": True,  # Enable template-based rendering
    "validation_enabled": True,  # Enable code validation
    "documentation_enabled": True,  # Enable documentation generation
    "testing_enabled": True,  # Enable test generation
    "optimization_enabled": True,  # Enable code optimization
}
```

### Framework-Specific Settings

Per-framework renderer options (versions, feature toggles, output paths) are declared in `src/gnn/render/framework_registry.py`; consult that registry for the authoritative settings instead of hardcoding them here.

## Error Handling

### Rendering Failures

```python
# Handle rendering failures gracefully
try:
    results = render_gnn_model(content, framework, output_dir)
except RenderingError as e:
    logger.error(f"Rendering failed: {e}")
    # Provide recovery rendering or error reporting
```

### Framework Issues

```python
# Handle framework-specific issues
try:
    renderer = select_renderer(framework)
    results = renderer.render_model(content, output_dir)
except FrameworkError as e:
    logger.warning(f"Framework failed: {e}")
    # Fall back to alternative framework
```

### Template Issues

```python
# Handle template issues
try:
    template = load_template(template_name)
    results = render_with_template(content, template)
except TemplateError as e:
    logger.error(f"Template failed: {e}")
    # Use default template or error reporting
```

## Performance Optimization

### Code Generation Optimization

- **Template Caching**: Cache rendered templates
- **Parallel Generation**: Parallel code generation
- **Incremental Generation**: Incremental code updates
- **Optimized Algorithms**: Optimize generation algorithms

### Framework Optimization

- **Framework Selection**: Optimize framework selection
- **Code Optimization**: Optimize generated code
- **Memory Management**: Optimize memory usage
- **Performance Monitoring**: Monitor generation performance

### Validation Optimization

- **Validation Caching**: Cache validation results
- **Parallel Validation**: Parallel code validation
- **Incremental Validation**: Incremental validation updates
- **Optimized Validation**: Optimize validation algorithms

## Testing and Validation

### Unit Tests

```python
# Test individual rendering functions
def test_pymdp_rendering():
    results = render_gnn_model(test_content, "pymdp", test_dir)
    assert "main_script" in results
    assert "config_file" in results
    assert results["main_script"].exists()
```

### Integration Tests

```python
# Test complete rendering pipeline
def test_rendering_pipeline():
    success = process_render(test_dir, output_dir)
    assert success
    # Verify rendering outputs
    rendering_files = list(output_dir.glob("**/*"))
    assert len(rendering_files) > 0
```

### Framework Tests

```python
# Test different frameworks
def test_framework_rendering():
    frameworks = ["pymdp", "rxinfer", "jax"]
    for framework in frameworks:
        results = render_gnn_model(test_content, framework, test_dir)
        assert results["success"]
```

## Dependencies

### Required Dependencies

- Standard library + `numpy` (used in render processors and matrix normalization)

This module uses Python string templates and code-generation helpers; it does not rely on Jinja2.

### Optional Dependencies

- **black**: Code formatting for Python
- **julia**: Julia language support
- **jax**: JAX framework support
- **discopy**: DisCoPy framework support

## Performance

Performance and success metrics are tracked by the pipeline summaries and per-step logs. Avoid hard-coded numeric claims in docs unless sourced from current benchmark outputs.

## Troubleshooting

### Common Issues

#### 1. Framework Compatibility

```
Error: Framework not supported for model type
Solution: Use compatible framework or adjust model structure
```

#### 2. Template Issues

```
Error: Template rendering failed - invalid syntax
Solution: Check template syntax or use default template
```

#### 3. Code Generation Issues

```
Error: Code generation failed - invalid model structure
Solution: Validate model structure or use simpler framework
```

#### 4. Performance Issues

```
Error: Code generation taking too long
Solution: Optimize model complexity or use faster framework
```

### Debug Mode

```python
from pathlib import Path
from gnn.render import process_render

process_render(Path("input/gnn_files"), Path("output/11_render_output"), verbose=True)
```

## Future Enhancements

### Planned Features

- **Additional Frameworks**: Support for more simulation frameworks
- **Custom Templates**: User-defined code templates
- **Real-time Rendering**: Live code generation during development
- **Advanced Optimization**: Advanced code optimization techniques

### Performance Improvements

- **Parallel Generation**: Multi-core code generation
- **Incremental Updates**: Incremental code updates
- **Advanced Caching**: Advanced caching strategies
- **Machine Learning**: ML-based code generation

## Summary

The Render module provides POMDP-aware code generation for multiple frameworks and writes structured outputs intended for execution (Step 12) and downstream analysis/reporting steps.

## License and Citation

This module is part of the GeneralizedNotationNotation project. See the main repository for license and citation information.

## References

- Project overview: ../../../README.md
- Comprehensive docs: ../../../DOCS.md
- Architecture guide: ../../../ARCHITECTURE.md
- Pipeline details: ../../../docs/pipeline/README.md


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API


### Current render receipts

`render_processing_summary.json` replaces the current input scope on each
invocation and recomputes counts from its file records. Source and generated
artifact identities contain resolved paths and SHA-256 content digests.
`receipt_identity` records the run ID and configuration digest. Prior JSON
receipts are archived under `history/render-<digest>.json` before atomic
publication of the replacement.

Pass the same `run_id` keyword or `GNN_RUN_ID` environment value to all folder
invocations belonging to one run. Only same-run, same-configuration records
whose source and artifact bytes still match can be carried forward. Standalone
calls without an explicit run ID get a fresh ID. Retrying a scope replaces its
records, including removed inputs; history never contributes to current counts.
The framework registry exposes `supports_execution`; bnlearn's is true since
`src/gnn/execute/bnlearn/` (Step 12 runs its programs through the shared
script path, skipping with an install hint when the `bnlearn` extra is
absent).


`render_gnn_spec` accepts structured parser mappings and
`GNNInternalRepresentation` objects as well as canonical POMDP mappings.
Singleton-row or singleton-column `C`, `D`, and `E` vectors are flattened before
canonical dimension inference; this preserves the two-state/two-policy shape
of the accepted FepLean symmetric Boolean bridge fixture.

### Public continuous-model dispatch

`render_gnn_spec` routes continuous linear-Gaussian models to the JAX,
NumPyro, PyTorch, RxInfer, and Stan continuous renderers before categorical
A/B/C/D validation. Unsupported continuous targets return a failed result.
Stan returns both the program and Python driver; RxInfer returns the script
and any execution metadata. The public regression suite exercises all five
routes and a three-step JAX prediction/update with an independent scalar
posterior recurrence. This numerical test is separate from formal evidence.
