# Step 6: Validation

## Architectural Mapping

**Orchestrator**: `src/gnn/6_validation.py` (57 lines)
**Implementation Layer**: `src/gnn/validation/`

## Module Description

This module provides comprehensive validation capabilities for GNN models, including consistency checking, semantic validation, and quality assessment.


```
src/gnn/validation/
├── AGENTS.md                      # Agent scaffolding documentation
├── README.md                      # This documentation
├── SKILL.md                       # Agent skill manifest (gnn-validation usage guide)
├── SPEC.md                        # Module specification
├── __init__.py                    # Module initialization, exports, and process_validation orchestrator
├── consistency_checker.py         # Consistency checking (naming, style, structure, references)
├── mcp.py                         # Model Context Protocol integration
├── orientation.py                 # B-tensor orientation diagnostic (canonical B[s',s,a] column-stochastic contract, transpose verification)
├── performance_profiler.py        # Performance profiling (complexity, memory, parallelization)
├── semantic_validator.py          # Semantic validation (structure, state space, connections, math)
├── simple.py                      # Simplified dependency-free validator used as a recovery path when the full validation system cannot run
├── structure.py                   # Shared model-content and graph utilities (content extraction, score clamping, Tarjan cycle detection)
└── workflow.py                    # Directory-level validation workflow behind process_validation (loads the parsed-model manifest, runs the three stages, persists validation_results.json / validation_summary.json)
```

## Agent Identity & Capabilities

# Validation Module - Agent Scaffolding

## Module Overview

**Purpose**: Advanced validation and consistency checking for GNN models and pipeline components

**Pipeline Step**: Step 6: Validation (6_validation.py)

**Category**: Validation / Quality Assurance

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-21

---

## Core Functionality

### Primary Responsibilities
1. Advanced validation and consistency checking
2. Model structure and semantic validation
3. Performance profiling and optimization
4. Cross-format consistency verification
5. Quality assurance and compliance checking

### Key Capabilities
- Comprehensive model validation
- Semantic consistency checking
- Performance profiling and analysis
- Cross-format validation
- Quality metrics and compliance

### Porting textbook POMDPs (B-tensor orientation)

Step 6 runs a default-on **B-tensor orientation diagnostic**. GNN's
canonical transition tensor order is `B[next_state, previous_state,
action]` (= pymdp 1.0.0 `B[s',s,a]`) with **column-stochastic** per-action
slices (rows = next states, columns = previous states, each column sums to
1). Textbook POMDP sources frequently write transition matrices the other
way (rows = previous state `s_t`, row-stochastic); imported verbatim, such
a file is silently read transposed — every transition probability flips.
The diagnostic warns on row-stochastic-only slices (naming the tensor,
state factor, and flipped slice indices), notes orientation-ambiguous
(doubly stochastic) tensors, and leaves non-stochastic tensors to the
existing stochasticity error paths. The opt-in `--transpose-b` flag (CLI
and `process_validation` MCP tool) validates the canonical transposition
in memory and records it per tensor in the receipt. Full convention,
severity table, and fix recipe:
[gnn_syntax.md § B-tensor orientation](../gnn_syntax.md).

---

## API Reference

### Public Functions

#### `process_validation(target_dir, output_dir, verbose=False, logger=None, **kwargs) -> bool`
**Description**: Main validation processing function called by orchestrator (6_validation.py)

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to validate
- `output_dir` (Path): Output directory for validation results
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Logger, optional): Logger instance (default: None)
- `strict` (bool): Enable strict validation mode (default: False)
- `profile` (bool): Enable performance profiling (default: False)
- `transpose_b` (bool): Opt-in canonical B-tensor transposition for the orientation stage — textbook (row-stochastic) transition tensors are transposed in memory and recorded in the receipt (default: orientation warnings only; source files are never modified)
- `**kwargs`: Additional validation options (including `validation_level` and `run_id`)

**Returns**: `True` if validation succeeded

**Example**:
```python
from gnn.validation import process_validation

success = process_validation(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/6_validation_output"),
    verbose=True,
    strict=True,
    profile=True
)
```

#### `process_semantic_validation(model_data: Dict[str, Any]) -> Dict[str, Any]`
**Description**: Perform semantic validation on model data

**Parameters**:
- `model_data` (Dict[str, Any]): Parsed GNN model data

**Returns**: Dictionary with semantic validation results

#### `profile_performance(model_data: Dict[str, Any]) -> Dict[str, Any]`
**Description**: Profile model performance characteristics

**Parameters**:
- `model_data` (Dict[str, Any]): Parsed GNN model data

**Returns**: Dictionary with performance metrics

#### `check_consistency(model_data: Dict[str, Any]) -> Dict[str, Any]`
**Description**: Check consistency of model data

**Parameters**:
- `model_data` (Dict[str, Any]): Parsed GNN model data

**Returns**: Dictionary with consistency results

#### `check_b_orientation(model_data, transpose_b=False) -> Dict[str, Any]`
**Description**: B-tensor orientation diagnostic (default-on Step 6 stage). Classifies each transition tensor by per-action slice row/column margins against the canonical contract and returns `{file_path, file_name, valid, warnings, notes, tensors, orientation_score}`.

**Parameters**:
- `model_data` (str | Path | Dict[str, Any]): GNN file path or parsed model data
- `transpose_b` (bool): Opt-in canonical transposition recorded in the receipt

**Returns**: Dictionary with orientation findings (see the B-tensor orientation section in [`gnn_syntax.md`](../gnn_syntax.md))

---

## Dependencies

### Required Dependencies
- `pathlib` - Path manipulation
- `re` - Regular expressions for parsing

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

---

## Configuration

### Validation Settings
```python
VALIDATION_CONFIG = {
    "strict_validation": False,
    "profile_performance": True,
    "check_consistency": True,
    "validate_semantics": True,
}
```

---

## Usage Examples

### Basic Validation
```python
from gnn.validation import process_validation

success = process_validation(
    target_dir="input/gnn_files", output_dir="output/6_validation_output"
)
```

### Semantic Validation and Consistency
```python
from gnn.validation import process_semantic_validation, check_consistency

semantic = process_semantic_validation("model.gnn")
consistency = check_consistency("model.gnn")
```

### Performance Profiling
```python
from gnn.validation import profile_performance

profile = profile_performance("model.gnn")
```

The exported surface is listed in `src/gnn/validation/__init__.py` (`__all__`):
`process_validation`, `check_consistency`, `profile_performance`,
`check_b_orientation`, `validate_content`, `validate_directory` plus the
`SemanticValidator`, `SimpleValidator`, `PerformanceProfiler` and
`StageServices` classes.

---

## Output Specification

### Output Products
- `validation_results.json` - Validation results
- `performance_profile.json` - Performance profiling
- `consistency_report.json` - Consistency checking
- `validation_summary.md` - Human-readable summary

### Output Directory Structure
```
output/6_validation_output/
├── validation_results.json
├── performance_profile.json
├── consistency_report.json
├── validation_summary.md
└── detailed_analysis/
    ├── structure_validation.json
    └── semantic_validation.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~1-5 seconds per model
- **Memory**: ~20-100MB
- **Status**: ✅ Production Ready

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
- **Script**: `6_validation.py` (Step 6)
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
- `tests/gnn/test_gnn_validation.py` - GNN validation-focused tests (shared)
- `tests/validation/test_b_orientation.py` - B-tensor orientation diagnostic and `--transpose-b` transposition contract tests

### Test Coverage
- Measure: `uv run --extra dev python -m pytest tests/validation/ --cov=gnn.validation --cov-report=term-missing` (do not treat fixed percentages in this doc as canonical).

### Key Test Scenarios
1. Model structure validation
2. Semantic consistency checking
3. Performance profiling accuracy
4. Error handling and recovery

---

## MCP Integration

### Tools Registered
- `process_validation` - Run Step 6 over a directory
- `validate_gnn_file` - Validate one GNN file
- `get_validation_report` - Read the latest validation results
- `check_schema_compliance` - Check a file against the GNN schema

### Tool Endpoints
```python
@mcp_tool("validation.validate_structure")
def validate_structure_tool(content):
    """Validate model structure"""
    # Implementation
```

---

---
## Documentation
- **[README](../../../src/gnn/validation/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/validation/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/validation/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/validation/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/validation](../../../src/gnn/validation)
