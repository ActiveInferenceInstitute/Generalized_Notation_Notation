# Validation Module - Agent Scaffolding

## Module Overview

**Purpose**: Advanced validation and consistency checking for GNN models and pipeline components

**Pipeline Step**: Step 6: Validation (src/gnn/6_validation.py)

**Category**: Validation / Quality Assurance

**Status**: Production Ready

**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-16

---

## Module Structure

```
src/gnn/validation/
├── __init__.py             # Module exports and thin process_validation facade
├── consistency_checker.py  # Consistency checking (naming, style, structure, references)
├── semantic_validator.py   # Semantic validation (structure, state space, connections, math)
├── orientation.py          # B-tensor orientation diagnostic (canonical vs textbook POMDP)
├── simple.py               # Dependency-free basic validator: marker/extension sniffing
├── performance_profiler.py # Performance profiling (complexity, memory, parallelization)
├── structure.py            # Shared helpers (content extraction, cycle detection, score clamping)
├── workflow.py             # Step-6 directory workflow (stage runner, receipts, accumulation)
└── mcp.py                  # Model Context Protocol integration

---

## Core Functionality

### Primary Responsibilities
1. Advanced validation and consistency checking
2. Model structure and semantic validation
3. Performance profiling and optimization
4. Cross-format consistency verification
5. Quality assurance and compliance checking
6. Restricted source-bound numerical witness generation and raw-artifact replay

`witness_source.py` owns the independent restricted source reader, finite path
reference and joint-Gaussian reference. `semantic_witness.py` owns v2 receipt
construction/replay. It must never stamp expected metadata onto a native result,
trust cached reference arrays or infer native execution/Lean proof from matching
numbers. Scientific bounds and explicit evidence limits live in the
[E4 contract](../../../docs/development/remaining_roadmap_contracts.md#e4-replayable-semantic-evidence-v2).
`tests/helpers/numerical_witness.py` remains a separate test-only oracle.

`witness_composition_source.py` independently admits the E4-IG1 passive
Gaussian product contract. `witness_composition.py` compares complete native
JAX child results with joint block-diagonal prefix conditioning, using the
additive v3 receipt. Cross-agent zero covariance is a reference assumption,
not measured native covariance; control, coupling and smoothing remain refused.
See the [bounded composition contract](../../../docs/development/e4_independent_gaussian_composition.md).

### Key Capabilities
- Comprehensive model validation
- Semantic consistency checking
- Performance profiling and analysis
- Cross-format validation
- Quality metrics and compliance

---

## API Reference

### Public Functions

#### `process_validation(target_dir, output_dir, verbose=False, **kwargs) -> bool`
**Description**: Main validation processing function called by orchestrator (src/gnn/6_validation.py)

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to validate
- `output_dir` (Path): Output directory for validation results
- `verbose` (bool): Enable verbose logging (default: False)
- `**kwargs`: Additional validation options:
  - `validation_level` (str): Semantic validation depth (`basic`, `standard`, `strict`, `research`; default `standard`)
  - `strict` (bool): Shorthand raising `validation_level` to `strict` (wired to the orchestrator's `--strict` flag)
  - `run_id` (str): Stable identity for intentional cross-manifest accumulation (otherwise manifest run ID or timestamp)
  - `transpose_b` (bool): Opt-in canonical B-tensor transposition for the
    orientation stage — textbook (row-stochastic) transition tensors are
    transposed in memory and recorded in the receipt (default warnings-only;
    source files are never modified; receipt context records the flag)
  - `logger`, `recursive`, `profile` are accepted for the standardized pipeline-script contract and do not alter behavior

**Returns**: `True` only for a nonempty current pass with every file successful; historical successes cannot mask current failure.

**Example**:
```python
from gnn.validation import process_validation

success = process_validation(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/6_validation_output"),
    verbose=True,
)

```

#### `process_semantic_validation(model_data) -> Dict[str, Any]`
**Description**: Perform semantic validation on model data. Returns `{file_path, file_name, valid, errors, warnings, semantic_score}` (`status: error` dict on failure).

**Parameters**:
- `model_data` (Dict[str, Any]): Parsed GNN model data

**Returns**: Dictionary with semantic validation results

#### `profile_performance(model_data) -> Dict[str, Any]`
**Description**: Profile model performance characteristics. Returns `{file_path, file_name, metrics, warnings, performance_score}` (`status: error` dict on failure).

**Parameters**:
- `model_data` (str | Path | Dict[str, Any]): GNN content string, file path, or parsed model data

#### `check_consistency(model_data) -> Dict[str, Any]`
**Description**: Check consistency of model data. Returns `{file_path, file_name, consistent, warnings, checks, consistency_score}` (`status: error` dict on failure).

**Parameters**:
- `model_data` (str | Path | Dict[str, Any]): GNN file path or parsed model data

**Returns**: Dictionary with consistency results

#### `validate_content(content, validation_level="standard") -> Dict[str, Any]`
**Description**: Validate raw GNN content text without file I/O (companion to `process_semantic_validation` for in-memory content). Same receipt shape with `file_path`/`file_name` = `"unknown"`.

#### `validate_directory(target_dir, output_dir, services, ...) -> bool`
**Description**: The workflow behind `process_validation` (in `workflow.py`). Takes injected `StageServices` (semantic/performance/consistency callables, plus the optional `orientation` B-tensor diagnostic callable), so alternative pipelines can compose custom stage functions. Stage functions are bound from the package namespace by `process_validation`, keeping the test monkeypatch seam intact. `orientation` defaults to `None` — pipelines that do not inject it skip the stage.

### Diagnostic inventory (stage: `orientation`)

`check_b_orientation(model_data, *, transpose_b=False)` (module `orientation.py`) classifies each `B` tensor by its per-action slice row/column margins against 1.0 (type checker tolerance `1e-6`, shared with the Step 5 `GNN-E002` checks):

| Per-action slices | Receipt finding |
|-------------------|-----------------|
| Column-stochastic (canonical `B[next_state, previous_state, action]`) | silent; `orientation: canonical` |
| Row-stochastic only (textbook POMDP layout) | **warning** naming the tensor, state factor, and flipped slice indices, with the transpose fix; `orientation: row_stochastic` |
| Doubly stochastic (orientation-ambiguous) | informational note only; `orientation: ambiguous` |
| Neither | silent; `orientation: non_stochastic` (existing stochasticity error paths own it) |

With `transpose_b=True`, row-stochastic tensors are additionally transposed
in memory to the canonical order (the same mapping as
`extract.canonicalize_pomdp`) and the tensor record gains
`transposed/previous_orientation/canonical_after_transpose` plus a note
recording the transposition. Warnings are advisory: the stage is always
`valid` unless it errors operationally. Corpus guarantee: zero warnings on
the gold exemplars (`input/gnn_files/`).

#### `profile_performance` error contract
On failure it returns the same best-effort shape as the other stages: `{status: "error", file_path, file_name, error, metrics, warnings, performance_score: 0.0, recovery: True}`; success results carry `recovery: False`.


---

## Dependencies

### Required Dependencies
- `pathlib` - Path manipulation
- `re` - Regular expressions for parsing

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

---

## Configuration

No module-level configuration file. Semantic validation depth is set via `SemanticValidator(validation_level=...)` (`basic`, `standard`, `strict`, `research`); the orchestrator runs all three validators with their defaults.

---

## Usage Examples

### Basic Validation
```python
from gnn.validation import process_validation

success = process_validation(
    target_dir="input/gnn_files", output_dir="output/6_validation_output"
)
```

### Semantic Validation
```python
from gnn.validation import process_semantic_validation

result = process_semantic_validation("model.gnn")
print(f"Valid: {result['valid']}, Score: {result['semantic_score']:.2f}")
```

### Performance Profiling
```python
from gnn.validation import profile_performance

result = profile_performance("model.gnn")
print(f"Memory estimate: {result['metrics']['estimated_memory_mb']:.2f} MB")
print(f"Score: {result['performance_score']:.2f}")
```

---

## Output Specification

### Output Products
- `validation_results.json` - Validation results
- `validation_summary.json` - Validation summary

### Output Directory Structure
```
output/6_validation_output/
├── validation_results.json
└── validation_summary.json
```

---

## Performance Characteristics

### Expected Performance
- **Basic Validation**: < 1 second
- **Structure Validation**: 1-3 seconds
- **Performance Profiling**: 2-5 seconds
- **Consistency Checking**: 1-4 seconds

---

## Error Handling

### Validation Errors
1. **Structure Errors**: Invalid model structure
2. **Semantic Errors**: Semantic inconsistencies
3. **Performance Issues**: Performance problems
4. **Consistency Errors**: Cross-format inconsistencies

### Recovery Strategies
- **Structure Repair**: Suggest structural fixes
- **Semantic Resolution**: Provide semantic guidance
- **Performance Optimization**: Suggest performance improvements
- **Consistency Reconciliation**: Resolve format differences

---

## Integration Points

### Orchestrated By
- **Script**: `src/gnn/6_validation.py` (Step 6)
- **Function**: `process_validation()`

### Imports From
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

### Imported By
- `main.py` - Pipeline orchestration
- `tests.test_validation_*` - Validation tests

### Data Flow
```
Model Content → Structure Validation → Semantic Validation → Performance Profiling → Consistency Checking
```

---

## Testing

### Test Files
- `tests/validation/test_validation_overall.py` - Module-level validation tests
- `tests/validation/test_consistency_contract.py` - Consistency and stage-receipt regression tests
- `tests/validation/test_validation_public_api.py` - Public API surface tests
- `tests/validation/test_reliability_validation.py` - Current verdict, replay, run/config identity, and parse-failure regressions
- `tests/validation/test_workflow_contracts.py` - Step-6 workflow contract tests (kwargs, accumulation, DI, error contracts, cycle detection)
- `tests/validation/test_b_orientation.py` - B-tensor orientation diagnostic and `--transpose-b` contract tests
- `tests/gnn/test_gnn_validation.py` - GNN validation-focused tests (shared)

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest tests/test_validation*.py \
    --cov=src/gnn/validation --cov-report=term-missing
```
### Key Test Scenarios
1. Model structure validation
2. Semantic consistency checking
3. Performance profiling accuracy
4. Error handling and recovery

---

## MCP Integration

### Tools Registered
Registered by `validation.mcp.register_tools(mcp_instance)` (4 tools):
- `process_validation` - Run full validation pipeline on a directory (optional `transpose_b` flag applies the canonical B-tensor transposition recorded in the receipt)
- `validate_gnn_file` - Validate a single GNN file (structural checks plus the full semantic result under a `semantic` key)
- `get_validation_report` - Read saved validation reports from a previous run
- `check_schema_compliance` - Check a GNN model string against canonical schema requirements

---

## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
