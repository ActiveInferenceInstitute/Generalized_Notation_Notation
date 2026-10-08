# Step 0: Template

## Architectural Mapping

**Orchestrator**: `src/gnn/0_template.py` (56 lines)
**Implementation Layer**: `src/gnn/template/`

## Module Description

This module serves as the **reference implementation** for the GNN pipeline's architectural pattern. It demonstrates the complete flow from `main.py` through numbered scripts as thin orchestrators to modular scripts in dedicated folders.


The GNN pipeline follows a **three-tier architectural pattern**:

```
main.py → Numbered Scripts (Thin Orchestrators) → Modular Scripts in Folders
```

```mermaid
flowchart TD
    Main[main.py<br/>Pipeline Orchestrator] -->| Executes| Step0[0_template.py<br/>Thin Orchestrator]
    
    subgraph "src/gnn/template/ Module"
        Init[__init__.py<br/>Public API]
        Proc[processor.py<br/>Core Logic]
    end

    Step0 --> Init
    Init --> Proc
```

## Agent Identity & Capabilities

# Template Module - Agent Scaffolding

## Module Overview

**Purpose**: Pipeline template and initialization system for the GNN processing pipeline

**Pipeline Step**: Step 0: Template initialization (0_template.py)

**Category**: Pipeline Infrastructure / Initialization

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-01-21

---

## Core Functionality

### Primary Responsibilities
1. Pipeline initialization and template generation
2. Infrastructure demonstration and pattern validation
3. Template processing and customization
4. Pipeline architecture documentation
5. Example generation and testing

### Key Capabilities
- Dynamic pipeline template generation
- Infrastructure pattern demonstration
- Template customization and validation
- Pipeline architecture documentation
- Example and test data generation

---

## API Reference

### Module exports

- `VERSION_INFO` — dict with `version`, `name`, `description`, `author` (included in `__all__`)
- `FEATURES` — capability flags for tooling and discovery

### Public Functions

#### `process_template_standardized(target_dir: Path, output_dir: Path, logger: logging.Logger, recursive: bool = False, verbose: bool = False, **kwargs) -> bool`
**Description**: Process pipeline template with standardized patterns. This is the main processing function called by the thin orchestrator.

**Parameters**:
- `target_dir` (Path): Target directory for template processing
- `output_dir` (Path): Output directory for results
- `logger` (logging.Logger): Logger instance for logging
- `recursive` (bool): Process subdirectories recursively (default: False)
- `verbose` (bool): Enable verbose logging (default: False)
- `**kwargs`: Additional processing options

**Returns**: `bool` - True if template processing succeeded, False otherwise

**Example**:
```python
from template import process_template_standardized
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
success = process_template_standardized(
    target_dir=Path("input/"),
    output_dir=Path("output/0_template_output/"),
    logger=logger,
    recursive=True,
    verbose=True,
)
```

#### `process_single_file(input_file: Path, output_dir: Path, options: Dict[str, Any] = None) -> bool`
**Description**: Process a single file using the template logic.

**Parameters**:
- `input_file` (Path): Path to input file to process
- `output_dir` (Path): Directory to save output files
- `options` (Dict[str, Any], optional): Processing options dictionary

**Returns**: `bool` - True if file processing succeeded, False otherwise

#### `validate_file(input_file: Path) -> Dict[str, Any]`
**Description**: Validate a file against template requirements.

**Parameters**:
- `input_file` (Path): Path to file to validate

**Returns**: `Dict[str, Any]` - Validation result dictionary with:
- `valid` (bool): Whether file is valid
- `errors` (List[str]): List of validation errors
- `warnings` (List[str]): List of validation warnings

#### `safe_template_execution(logger: Any, correlation_id: str) -> Any`
**Description**: Context manager wrapping a template operation with error handling and correlated logging (`src/gnn/template/processor.py`).

**Parameters**:
- `logger` (Any): Logger used for the wrapped operation
- `correlation_id` (str): Identifier attached to log lines for the operation
- `*args`: Positional arguments for function
- `**kwargs`: Keyword arguments for function

**Returns**: `Any` - Function return value, or None if execution failed

**Raises**: Logs errors but does not raise exceptions

#### `get_version_info() -> Dict[str, str]`
**Description**: Get module version and metadata information.

**Returns**: `Dict[str, str]` - Version information dictionary with:
- `version` (str): Module version string
- `name` (str): Module name
- `description` (str): Module description
- `author` (str): Module author

#### `generate_correlation_id() -> str`
**Description**: Generate unique correlation ID for pipeline tracking and request correlation.

**Returns**: `str` - Unique correlation ID string (UUID format)

**Example**:
```python
from template import generate_correlation_id

correlation_id = generate_correlation_id()
# Returns: "550e8400-e29b-41d4-a716-446655440000"
```

#### `demonstrate_utility_patterns(context: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]`
**Description**: Demonstrate utility patterns and capabilities for documentation and testing purposes.

**Parameters**:
- `context` (Dict[str, Any]): Processing context dictionary
- `logger` (logging.Logger): Logger instance for demonstration logging

**Returns**: `Dict[str, Any]` - Demonstration results dictionary with:
- `patterns_demonstrated` (List[str]): List of demonstrated patterns
- `results` (Dict[str, Any]): Results from each pattern demonstration
- `performance_metrics` (Dict[str, float]): Performance metrics

---

## Dependencies

### Required Dependencies
- `pathlib` - Path manipulation
- `uuid` - Unique ID generation
- `datetime` - Timestamp generation

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline template utilities

---

## Configuration

### Template Settings
```python
TEMPLATE_CONFIG = {
    "enable_demonstration": True,
    "generate_examples": True,
    "validate_patterns": True,
    "include_documentation": True,
}
```

---

## Usage Examples

### Basic Template Processing
```python
from template.processor import process_template_standardized

success = process_template_standardized(
    target_dir="input/", output_dir="output/0_template_output", logger=logger
)
```

### Utility Pattern Demonstration
```python
from template.processor import demonstrate_utility_patterns

results = demonstrate_utility_patterns(context, logger)
print(f"Patterns demonstrated: {len(results['demonstrations'])}")
```

### Correlation ID Generation
```python
from template.processor import generate_correlation_id

correlation_id = generate_correlation_id()
print(f"Generated ID: {correlation_id}")
```

---

## Output Specification

### Output Products
- `template_processing_summary.json` - Template processing results
- `infrastructure_demonstration.json` - Pattern demonstration results
- `template_validation_report.md` - Template validation report
- `pipeline_patterns_documentation.md` - Architecture documentation

### Output Directory Structure
```
output/0_template_output/
├── template_processing_summary.json
├── infrastructure_demonstration.json
├── template_validation_report.md
├── pipeline_patterns_documentation.md
└── examples/
    └── template_examples.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~1-3 seconds
- **Memory**: ~10-20MB
- **Status**: ✅ Production Ready

### Expected Performance
- **Template Processing**: < 1 second
- **Pattern Demonstration**: 1-2 seconds
- **Documentation Generation**: < 1 second
- **Validation**: < 1 second

---

## Error Handling

### Template Errors
1. **Template Generation**: Template creation failures
2. **Pattern Validation**: Pattern validation errors
3. **File I/O**: File operation failures
4. **Configuration**: Invalid template configuration

### Recovery Strategies
- **Template Regeneration**: Recreate templates from defaults
- **Pattern Simplification**: Use simpler patterns
- **Documentation Recovery**: Generate basic documentation
- **Error Logging**: Comprehensive error reporting

---

## Integration Points

### Orchestrated By
- **Script**: `0_template.py` (Step 0)
- **Function**: `process_template_standardized()`

### Imports From
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

### Imported By
- `main.py` - Pipeline orchestration
- `tests.test_template_*` - Template tests

### Data Flow
```
Template Input → Processing → Pattern Demonstration → Validation → Documentation → Output
```

---

## Testing

### Test Files
- `tests/template/test_template_overall.py` - Module-level tests (imports, outputs, and core behaviors)
- `tests/pipeline/test_pipeline_scripts.py` - Orchestrator-level checks that include `0_template.py`

### Test Coverage
- Measure: `uv run --extra dev python -m pytest tests/template/ --cov=gnn.template --cov-report=term-missing` (do not treat fixed percentages in this doc as canonical).

### Key Test Scenarios
1. Template processing and generation
2. Pattern demonstration and validation
3. Documentation creation
4. Error handling and recovery

---

## MCP Integration

### Tools Registered
- `template.process_file` - Process one file through the template step
- `template.process_directory` - Process a directory (optionally recursive)
- `template.get_info` - Module metadata

### Tool Endpoints
```python
@mcp_tool("template.process")
def process_template_tool(target_dir, output_dir):
    """Process pipeline template"""
    # Implementation
```

---

---
## Documentation
- **[README](../../../src/gnn/template/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/template/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/template/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/template/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/template](../../../src/gnn/template)
