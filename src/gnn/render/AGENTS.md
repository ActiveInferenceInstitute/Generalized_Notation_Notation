# Render Module - Agent Scaffolding

## Module Overview

**Purpose**: Code generation for simulation frameworks from parsed GNN/POMDP specifications.

**Pipeline Step**: Step 11: Code rendering (11_render.py)

**Category**: Code Generation / Simulation Framework Integration

**Status**: ✅ Production Ready

**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-24

## File Structure

```
pomdp_processor/
├── __init__.py            # Package facade: module docstring, logger, re-exports, POMDPRenderProcessor assembly, public functions
├── _routes.py             # RendererRoute dataclass and RENDERER_ROUTES dispatch table
├── _flow.py               # _ProcessFlowMixin: __init__ and per-framework processing flow
├── _validation.py         # _CompatibilityValidationMixin: POMDP-renderer compatibility validation
├── _canonical.py          # _CanonicalSpecMixin: canonical POMDP spec composition helpers
├── _spec_generation.py    # _SpecGenerationMixin: GNN spec builders per space model kind
├── _renderer_dispatch.py  # _RendererDispatchMixin: _invoke_renderer and _call_*_renderer methods
├── _support.py            # Cross-mixin typing contract (annotation-only declarations, no runtime members)
└── _documentation.py      # _DocumentationMixin: per-framework README generation

processor/
├── __init__.py            # Package facade: module docstring, numpy/sys.path bootstrap, logger, registry re-exports, moved defs
├── parsing.py             # Parsing helpers: output stems, node/spec mapping, parse-summary rehydration, vector flattening, validation, matrix normalization
├── receipts.py            # Receipt persistence: _load_prior_render_summary, _render_file_identity, _atomic_render_json, _write_render_receipt, parse_frameworks_selection
├── pipeline.py            # process_render, _process_single_gnn_file_basic, _create_overview_documentation
├── rendering.py           # _render_continuous_target, render_gnn_spec
└── metadata.py            # get_module_info, get_available_renderers
```

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
- **Recursive exemplar discovery**: GNN spec files are discovered recursively with `rglob`, so nested exemplar folders under `input/gnn_files/**` (e.g. `discrete/`, `continuous/`, `basics/`, `hierarchical/`) are all rendered. Model-source files are filtered by `gnn.processing.discovery.is_model_source_path`; the maintained corpus index is `input/gnn_files/INDEX.md`. Current per-framework render/execute outcomes live in the run's `render_processing_summary.json` and `execution_summary.json`, not in this file.

### Supported Frameworks

#### PyMDP (Python)
- **Purpose**: Active Inference simulation in Python
- **Features**: Full PyMDP agent implementation
- **Output**: Complete Python simulation scripts
- **Optimization**: Matrix optimization, memory efficiency
- **Continuous models**: `unsupported` (discrete POMDPs only)

#### RxInfer.jl (Julia)
- **Purpose**: Genuine `@model` + `infer()` variational message-passing inference
- **Features**: Real `@model` with `Categorical` / `DiscreteTransition` nodes, `infer()` with `free_energy=true`, real VFE traces
- **Output**: Julia scripts with embedded `@model` definitions (the former TOML target is not supported)
- **Optimization**: Committed `Project.toml` (RxInfer 5.5.0), `--project=` for reproducible execution, EFE/policy via custom Active Inference logic
- **Continuous models**: native (LGSSM programs)

#### ActiveInference.jl (Julia)
- **Purpose**: Active Inference framework implementation
- **Features**: Complete Active Inference agent
- **Output**: Julia simulation scripts
- **Optimization**: Hierarchical processing, temporal dynamics
- **Continuous models**: `unsupported` (discrete POMDPs only)

#### DisCoPy (Python)
- **Purpose**: Categorical diagrams for compositional models
- **Features**: String diagram generation
- **Output**: Python DisCoPy diagrams
- **Optimization**: Categorical composition, type checking
- **Continuous models**: `unsupported` (no linear-Gaussian diagram semantics)

#### JAX (Python)
- **Purpose**: High-performance numerical computing
- **Features**: JIT compilation, automatic differentiation
- **Output**: JAX-optimized simulation code
- **Optimization**: GPU acceleration, vectorization
- **Continuous models**: native (linear-Gaussian programs)

#### PyTorch (Python)
- **Purpose**: Neural integration backend
- **Output**: Python scripts under `pytorch/` when the renderer is available
- **Continuous models**: native (dependency intentionally unlocked; see `framework_registry.py`)

#### NumPyro (Python)
- **Purpose**: Probabilistic programming backend
- **Output**: Python scripts under `numpyro/` when the renderer is available
- **Continuous models**: native (linear-Gaussian programs)

#### Stan (Stan)
- **Purpose**: Probabilistic programming model backend
- **Output**: Stan models plus cmdstanpy drivers under `stan/` when requested
- **Continuous models**: native (LGSSM programs)

#### BNLearn (Python)
**Module**: `src/gnn/render/bnlearn/bnlearn_renderer.py` (re-exported via `gnn.render.bnlearn`).

- **Purpose**: Bayesian network / causal model backend
- **Output**: Python scripts under `bnlearn/` when requested
- **Continuous models**: `unsupported`; executed by Step 12 via `execute/bnlearn/` (skips with an install hint until `uv sync --extra bnlearn` provides the Python package, or R + the R `bnlearn` package for `.R` scripts)

#### ngc-learn (Python)
- **Purpose**: Predictive-processing neural simulation backend
- **Output**: `<model>_ngclearn.py` scripts under `ngclearn/` when requested
- **Continuous models**: native — LGSSM via the shared `render/continuous_script.py` generator (Kalman numerics byte-identical to the JAX backend); codegen-only — the emitted script imports `ngclearn`, never the renderer; executed by Step 12 via `execute/ngclearn/`, skipping with an install hint until `uv sync --extra ngclearn` (py3.12 marker-gated extra)

#### cpomdp (Python)
**Module**: `src/gnn/render/cpomdp/cpomdp_renderer.py`

- **Purpose**: Continuous active inference (Kalman filter plus expected-free-energy policy search)
- **Output**: `<model>_cpomdp.py` scripts under `cpomdp/` when requested
- **Continuous models**: native, with its own template (`render/cpomdp/script_template.py`). Discrete models are `unsupported` (`continuous_only: true`). The emitted script imports `cpomdp`, never the renderer. Step 12 skips it with an install hint until `uv sync --extra cpomdp`

The canonical framework inventory lives in `framework_registry.py`; update that
registry before changing public framework lists, MCP enums, or processor configs.

---

## API Reference

### Public Functions

#### `process_render(target_dir: Path, output_dir: Path, verbose: bool = False, frameworks: Optional[List[str] | str] = None, strict_validation: bool = True, strict_framework_success: bool = False, **kwargs) -> Union[bool, int]`
**Description**: Main rendering processing function called by orchestrator (11_render.py). Processes GNN files and generates code for multiple simulation frameworks.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to process
- `output_dir` (Path): Output directory for rendered files
- `verbose` (bool): Enable verbose logging (default: False)
- `frameworks` (Optional[List[str] | str]): Restrict rendering to a subset of frameworks. Accepts `None`/`"all"` for all configured frameworks, `"lite"` for `pymdp`, `jax`, `discopy`, and `bnlearn`, or a comma-separated string such as `"pymdp,jax"`.
- `strict_validation` (bool): Passed to POMDP extraction (`True` by default)
- `strict_framework_success` (bool): When `True`, every requested framework render must succeed for every input file. Explicit framework requests such as `"pymdp"` or `"pymdp,jax"` are strict even when this flag is omitted; `"all"` and `"lite"` keep aggregate pipeline policy unless the flag is set.
- `**kwargs`: Additional options forwarded to framework renderers (e.g. `timesteps`, `simulation_params`)

**Returns**: `Union[bool, int]` - `True` when the render policy succeeds, `False` when it fails, or `2` when no input files are found.

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
    strict_framework_success=True,
)
```

#### `render_gnn_spec(gnn_spec: Dict[str, Any], target: str, output_directory: Union[str, Path], options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`
**Description**: Render a GNN specification dictionary to a target framework.

**Parameters**:
- `gnn_spec` (Dict[str, Any]): Parsed GNN specification dictionary
- `target` (str): Target framework ("pymdp", "rxinfer", "activeinference_jl", "jax", "discopy")
- `output_directory` (Union[str, Path]): Output directory for generated code
- `options` (Optional[Dict[str, Any]]): Framework-specific options (default: None)

**Returns**: `Tuple[bool, str, List[str]]` - Tuple containing:
- `success` (bool): Whether rendering succeeded
- `message` (str): Status message
- `generated_files` (List[str]): List of generated file paths

**Location**: `src/gnn/render/processor/rendering.py`

#### `parse_frameworks_selection(frameworks: Union[str, List[str], None]) -> Tuple[Optional[List[str]], bool]`
**Description**: Normalize the `frameworks` selection used by `process_render` and the Step 11 CLI. Pure function; resolves `None`/`"all"` to `None` (all registered frameworks), `"lite"` to the registry preset (`get_lite_frameworks()`), and comma-separated strings to a stripped name list.

**Returns**: `(frameworks, explicit_request)` — `explicit_request` is `True` when the caller pinned a specific framework set, which Step 11 treats as a strict-success policy.

**Location**: `src/gnn/render/processor/receipts.py`

### Shared helpers

- `render.naming.safe_output_stem(value, fallback="model")` — filesystem-safe output stem (single source of truth for the `render/processor/` and `render/pomdp_processor/` packages).
- `render.naming.atomic_write_text(path, content)` — temp-file + `os.replace` atomic artifact write.
- `render.spec_matrices.extract_abcd_matrices(gnn_spec)` — shared discrete A/B/C/D extraction with the `stateSpace.parameters` → `initialparameterization` → `parameters` fallback chain, neutral defaults, and column normalization (used by the PyTorch and NumPyro renderers).
- `render.spec_matrices.format_array_literal(arr, *, prefix, suffix="", indent=4)` — language-neutral array-literal formatter behind `_format_tensor` / `_format_jnp_array`.

### Canonical POMDP render helpers

For POMDP targets, `render_gnn_spec(...)` and Step 11 both route through the same validated renderers:

- `render_gnn_to_pymdp(...)`
- `render_gnn_to_rxinfer(...)`
- `render_gnn_to_activeinference_jl(...)`

The shared contract is `canonical_pomdp_v1`; B is stored as `(next_state, previous_state, action)`.

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

### `render_gnn_to_jax` and friends (JAX generator)

**Module**: `src/gnn/render/jax/jax_renderer.py` (re-exported via `gnn.render.jax`). There is no `generate_jax_code` function in `generators.py` or anywhere in the codebase; the JAX entrypoints live in the `jax` submodule.

**Signatures** (all four share this shape):
```python
def render_gnn_to_jax(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]
```

- `render_gnn_to_jax` — general JAX model (routes continuous specs to the LGSSM generator and Kronecker-factorized specs to `render_gnn_to_jax_factorized`)
- `render_gnn_to_jax_factorized(gnn_spec, output_path, options=None)` — sparse mean-field script for per-factor `A_fN`/`B_fN` specs
- `render_gnn_to_jax_pomdp(gnn_spec, output_path, options=None)` — JAX POMDP solver
- `render_gnn_to_jax_combined(gnn_spec, output_path, options=None)` — combined hierarchical/multi-agent variant

**Parameters**:
- `gnn_spec` (`Dict[str, Any]`): Parsed GNN specification
- `output_path` (`Path`): Exact path of the `.py` file to write
- `options` (`Optional[Dict[str, Any]]`): Opaque options bag forwarded to the generator

**Returns**: `(success, message, generated_files)` — success flag, human-readable message, and list of written file paths (empty list on failure).

---

## Dependencies

### Required Dependencies
- `numpy` - Numerical computations
- `pathlib` - Path manipulation
- `typing` - Type hints

### Framework-Specific Dependencies
- **PyMDP**: `pymdp` package
- **RxInfer.jl**: Julia with RxInfer.jl 5.5.0 (committed `Project.toml` under `src/gnn/execute/rxinfer/`)
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
    "inference_engine": "RxInfer variational message-passing (@model + infer(), free_energy=true)",
    "optimization": "auto",
    "constraints": "default",
}
```

Configuration is primarily controlled by the Step 11 orchestrator (`src/gnn/11_render.py`) and forwarded parameters to `process_render(...)`. Avoid documenting configuration keys that are not backed by code.

Step 11 CLI options relevant to framework scoping:

```bash
uv run python src/gnn/11_render.py \
    --target-dir input/gnn_files \
    --output-dir output \
    --frameworks pymdp \
    --strict-framework-success
```

Explicit `--frameworks` selections are strict by default. `--strict-framework-success` applies the same all-requested-frameworks policy to broader `"all"` or `"lite"` runs.

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

### Scoped Pipeline Rendering
```bash
uv run python src/gnn/main.py \
    --only-steps 11 \
    --target-dir input/gnn_files/pymdp_scaling_study \
    --output-dir output/pymdp_scaling_pipeline \
    --frameworks pymdp \
    --strict-framework-success
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
  - optional/requested backends: `pytorch/`, `numpyro/`, `stan/`, `bnlearn/`
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
    ├── pytorch/        # if requested and available
    ├── numpyro/        # if requested and available
    ├── stan/           # if requested and available
    └── bnlearn/        # if requested and available
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
- **Framework Scope**: Use `--frameworks <name>` to isolate a backend while debugging.
- **Strict Policy**: Use explicit `--frameworks` selections for focused strict runs; add `--strict-framework-success` when `"all"` or `"lite"` should also fail on any framework render failure.
- **Partial Generation**: Omit strict framework success when exploratory runs should keep usable artifacts.
- **Error Documentation**: Review `render_processing_summary.json`, including `failed_framework_renderings`.

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
- `tests/render/test_render_contracts.py` - Shared-helper + policy contract tests (naming, spec_matrices, lite preset, `parse_frameworks_selection`, `_render_succeeded`, `validate_render`, MCP single-spec tool)
- `tests/render/test_render_cli_targets.py` - CLI target dispatch guard
- `tests/render/test_framework_availability.py` - Registry availability contract
- `tests/render/test_render_integration.py` / `test_render_overall.py` / `test_render_performance.py` - Broad module behavior
- `tests/render/test_jax_renderer.py` / `test_render_stan.py` / `test_continuous_renderers.py` - Framework-specific behavior

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest tests/test_render*.py \
    --cov=src/gnn/render --cov-report=term-missing
```

### Key Test Scenarios
1. Multi-framework code generation
2. Framework-specific optimizations
3. Error handling and recovery
4. Performance benchmarking
5. Integration with execution step

---

## MCP Integration

### Tools Registered
- `render.process_render` - Render every GNN file in a directory to all supported frameworks
- `render.list_render_frameworks` - List supported framework names and availability
- `render.render_gnn_to_format` - Render one GNN file (runs the Step 11 directory flow; `framework` is a hint, not a filter)
- `render.render_spec_to_format` - Render one GNN file to exactly one framework via `render_gnn_spec`
- `render.get_render_module_info` - Module metadata: supported frameworks and input/output formats

### Tool Endpoints
Tool functions live in `src/gnn/render/mcp.py` and are registered in `register_tools(mcp_instance)`; see that file for the exact JSON schemas.

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
- Use `--strict-validation=False` for lenient validation
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
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
