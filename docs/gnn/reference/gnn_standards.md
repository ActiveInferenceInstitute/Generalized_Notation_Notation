# GNN Standards

**Version**: v2.0.0 documentation bundle\
**Last Updated**: 2026-04-15  
**Status**: Maintained
**Scope**: GNN standards and authoring conventions. See [framework implementations](../implementations/README.md) for current backend coverage.

## Pipeline Processing Standards

The GNN pipeline follows strict architectural patterns and standards:

- **Thin Orchestrator Pattern**: All 25 pipeline steps delegate to modular implementations
  - See: **[src/gnn/README.md](../../../src/gnn/README.md)** for thin orchestrator pattern details
- **Module Architecture**: Each module follows consistent structure with public APIs
  - See: **[src/gnn/AGENTS.md](../../../src/gnn/AGENTS.md)** for complete module registry
- **Testing Standards**: No substitutions, real data validation, >90% test coverage
  - See: **[docs/gnn/operations/REPO_COHERENCE_CHECK.md](../operations/REPO_COHERENCE_CHECK.md)** for quality standards

**Architecture Documentation:**

- [architecture_reference.md](architecture_reference.md): Implementation patterns and data flow
- [src/gnn/README.md](../../../src/gnn/README.md): Pipeline safety and reliability

---

## Coding Standards

### Python Style

- **Type Hints**: All public functions must have complete type annotations
- **Docstrings**: Google-style docstrings for all public classes and functions
- **Imports**: Use relative imports within modules, absolute imports for cross-module references
- **Logging**: Use `logging.getLogger(__name__)` with structured log messages via `gnn.utils.pipeline_orchestration.pipeline_template`

### Module Structure

Each pipeline module follows this layout:

```text
src/<module_name>/
├── __init__.py          # Public API exports
├── processor.py         # Core processing logic
├── mcp.py               # MCP tool definitions
└── AGENTS.md            # Module documentation
```

Tests live centrally in `tests/`, not inside the module directory.
`processor.py` is the convention rather than a hard rule — `setup/`, `tests/`,
and `validation/` keep their logic in `__init__.py`, `model_registry/` uses
`registry.py`, and `website/` uses `renderer.py` + `generator.py`.

### Naming Conventions

- **Pipeline scripts**: `src/{step}_{name}.py` (e.g., `src/gnn/3_gnn.py`)
- **Modules**: lowercase with underscores (e.g., `type_checker`, `advanced_visualization`)
- **Test files**: `test_{module_name}.py` or `test_{feature}.py`
- **Output directories**: `output/{step}_{name}_output/` (e.g., `output/3_gnn_output/`)

---

## Testing Standards

### Real-Implementation Policy

All tests must use real implementations and representative data.

```python
# ❌ PROHIBITED
@patch("module.real_function")
def test_with_substitution(patched_fn):
    patched_fn.return_value = simulated_result


# ✅ REQUIRED
def test_with_real_data():
    result = real_function(real_input)
    assert result.is_valid
```

### Test Requirements

- Minimum 90% code coverage per module
- Tests must validate real pipeline outputs against known-good baselines
- Integration tests run the full pipeline with sample GNN files from `input/gnn_files/`
- Tests are run via: `uv run --extra dev python -m pytest tests/ -v` (from repository root)

---

## GNN File Structure Standards

- **GNN Files**: Markdown-based (.md) with specific sections, in this order:
  - `GNNSection`: Short, space-free model-class identifier (e.g. `ActInfPOMDP`)
  - `GNNVersionAndFlags`: Version specification and processing flags
  - `ModelName`: Descriptive model identifier
  - `ModelAnnotation`: Free-text explanation of model purpose and features
  - `StateSpaceBlock`: Variable definitions with dimensions/types (s_fX[dims,type])
  - `Connections`: Directed/undirected edges showing dependencies (>, -, ->)
  - `InitialParameterization`: Starting values, matrices (A, B, C, D), priors
  - `Equations`: LaTeX-rendered mathematical relationships
  - `Time`: Temporal settings (Dynamic/Static, DiscreteTime, ModelTimeHorizon)
  - `ActInfOntologyAnnotation`: Mapping to Active Inference Ontology terms
  - `Footer` and `Signature`: Provenance information

## GNN Syntax and Punctuation

- **Variables**: Use underscore for subscripts (X_2), caret for superscripts (X^Y)
- **Dimensions**: Square brackets for array dimensions [2,3] = 2x3 matrix
- **Causality**: `>` for directed edges (X>Y), `-` for undirected (X-Y)
- **Operations**: Standard math operators (+, -, *, /, |)
- **Grouping**: Parentheses (), exact values {1}, indexing/dimensions [2,3]
- **Comments**: Lines starting with `#` inside variable/connection blocks are treated as comments
- **Probability**: Conditional probability notation P(X|Y) using pipe |

---

## Documentation Standards

### File Metadata

Every documentation file in `docs/gnn/` must include this header:

```markdown
# [Title]

**Version**: v2.0.0 documentation bundle\
**Last Updated**: 2026-04-15  
**Status**: Maintained
**Scope**: GNN standards and authoring conventions. See [framework implementations](../implementations/README.md) for current backend coverage.
```

### Cross-Referencing

- Link to relevant module AGENTS.md files for implementation details
- Link to related documentation files within `docs/gnn/`
- Use relative paths for all internal links

### Content Principles

- **Show Not Tell**: Provide working code examples and real outputs
- **Evidence-Based**: Cite specific metrics and measurable results
- **Understated**: Concrete descriptions over promotional language
