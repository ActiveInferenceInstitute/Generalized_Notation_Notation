# GNN Repo Coherence Check

**Version**: v3.2.0 Engine (Bundle v2.0.0)  
**Last Updated**: 2026-04-15  
**Status**: Maintained
**Scope**: Repository coherence checks for the GNN pipeline. See [framework implementations](../implementations/README.md) for current backend coverage.

## Pipeline Architecture References

This document validates the complete GNN pipeline architecture. For current implementation:

- **[src/gnn/AGENTS.md](../../../src/gnn/AGENTS.md)**: Master agent scaffolding and complete 25-step pipeline registry
- **[src/gnn/README.md](../../../src/gnn/README.md)**: Pipeline architecture and thin orchestrator pattern
- **[src/gnn/main.py](../../../src/gnn/main.py)**: Pipeline orchestrator implementation (25 steps: 0-24)
- [Architecture Reference](../reference/architecture_reference.md): Implementation patterns and cross-module data flow
- [Technical Reference](../reference/technical_reference.md): Complete entry points and round-trip data flow

- ✅ 25 steps (0-24), each following the thin orchestrator pattern
- ✅ Every `src/` package ships `AGENTS.md` — enforced by `docs/development/docs_audit.py`, which fails on any missing file
- ℹ️ Per-run success rate, execution time, and peak memory come from the pipeline execution summary for the run in question; this document does not carry a performance snapshot

---

## Executive Summary

This document serves as a comprehensive checklist and analysis framework for validating the entire GNN (Generalized Notation Notation) codebase against established standards, patterns, and best practices. It covers all 25 pipeline steps (0-24) and 32 Python packages under `src/`, ensuring architectural compliance, code quality, documentation completeness, and integration consistency.

### High-Level Coherence Metrics

- **Total Pipeline Steps**: 25 (0_template.py through 24_intelligent_analysis.py, steps 0-24)

- **Total Modules**: 32 Python packages under `src/`
- **AGENTS.md Files**: 124 documentation files (including sub-modules) as of
  this writing — this count drifts as modules are added; re-verify with
  `find src -name AGENTS.md | wc -l` rather than trusting this hand-maintained
  snapshot
- **Architecture Pattern**: Thin orchestrator with modular implementation
- **Testing Policy**: No substitutions - real data and real code paths only (some previous tests still use substitutions - see Section 16.1)
- **Documentation Standard**: AGENTS.md + README.md for each module
- **Output Directory Pattern**: `N_[module]_output/` for all steps (enforced via `get_output_dir_for_script()`)

### Validation Scope

This mega-prompt validates:

1. Architecture compliance across all 25 pipeline steps
2. Code quality standards (type hints, docstrings, error handling)
3. Module structure consistency (directory patterns, **init**.py, MCP integration)
4. Documentation completeness (AGENTS.md, README.md, API docs)
5. Testing standards (real-implementation policy, real data, integration tests)
6. Pipeline integration (dependencies, data flow, output consistency)
7. Performance standards (execution time, memory usage, resource efficiency)
8. Security and validation patterns
9. Naming conventions consistency
10. Dependency management and module coupling

---

## 1. Architecture Compliance

### 1.1 Thin Orchestrator Pattern Validation

**Requirement**: All 25 numbered pipeline scripts must follow the thin orchestrator pattern, delegating core functionality to module implementations.

#### Validation Checklist for Each Pipeline Script (0-24)

For each script `N_[module_name].py`, verify:

- [ ] **File Length**: Script is < 150 lines (thin orchestrator requirement)
- [ ] **Import Pattern**: Imports from `utils` and `pipeline` modules, not inline implementations
- [ ] **Logging Setup**: Uses `setup_step_logging()` or `setup_main_logging()` (for main.py)
- [ ] **Argument Parsing**: Uses `EnhancedArgumentParser.parse_step_arguments()` or `create_standardized_pipeline_script()`
- [ ] **Core Logic Delegation**: All domain logic delegated to module functions (e.g., `process_render()`, `process_validation()`)
- [ ] **No Inline Implementations**: No function definitions > 20 lines in numbered scripts
- [ ] **Exit Codes**: Returns proper exit codes (0=success, 1=error, 2=success with warnings/skipped)
- [ ] **Output Directory**: Uses `get_output_dir_for_script()` for centralized output management
- [ ] **Error Handling**: Uses `log_step_error()`, `log_step_warning()` for structured error reporting
- [ ] **Visual Logging**: Uses visual logging utilities where appropriate

#### Specific Pipeline Scripts to Validate

**Step 0**: `src/gnn/0_template.py` → `src/gnn/template/`

- Verify: Uses `create_standardized_pipeline_script()` pattern
- Verify: Delegates to `process_template_standardized()`

**Step 1**: `src/gnn/1_setup.py` → `src/gnn/setup/`

- Verify: Environment setup delegation
- Verify: Dependency management patterns

**Step 2**: `src/gnn/2_tests.py` → `tests/`

- Verify: Test orchestration delegation
- Verify: Real test execution (no substitutions)

**Step 3**: `src/gnn/3_gnn.py` → `src/gnn/`

- Verify: GNN file discovery and parsing delegation
- Verify: Multi-format support delegation
- Verify: Delegates to `process_gnn_multi_format()` from `gnn.processing.multi_format_processor`

**Step 4**: `src/gnn/4_model_registry.py` → `src/gnn/model_registry/`

- Verify: Registry management delegation
- Verify: Versioning and metadata handling

**Step 5**: `src/gnn/5_type_checker.py` → `src/gnn/type_checker/`

- Verify: Type checking delegation
- Verify: Resource estimation delegation

**Step 6**: `src/gnn/6_validation.py` → `src/gnn/validation/`

- Verify: Validation logic delegation
- Verify: Consistency checking delegation

**Step 7**: `src/gnn/7_export.py` → `src/gnn/export/`

- Verify: Multi-format export delegation
- Verify: Format-specific exporters

**Step 8**: `src/gnn/8_visualization.py` → `src/gnn/visualization/`

- Verify: Visualization generation delegation
- Verify: Graph and matrix visualization
- Verify: Delegates to `process_visualization()` from `visualization` module

**Step 9**: `src/gnn/9_advanced_viz.py` → `src/gnn/advanced_visualization/`

- Verify: Advanced visualization delegation
- Verify: Interactive plot generation

**Step 10**: `src/gnn/10_ontology.py` → `src/gnn/ontology/`

- Verify: Ontology processing delegation
- Verify: Active Inference term mapping

**Step 11**: `src/gnn/11_render.py` → `src/gnn/render/`

- Verify: Code generation delegation
- Verify: Multi-framework rendering (PyMDP, RxInfer, ActiveInference.jl)

**Step 12**: `src/gnn/12_execute.py` → `src/gnn/execute/`

- Verify: Execution orchestration delegation
- Verify: Multi-environment execution

**Step 13**: `src/gnn/13_llm.py` → `src/gnn/llm/`

- Verify: LLM processing delegation
- Verify: AI-enhanced analysis

**Step 14**: `src/gnn/14_ml_integration.py` → `src/gnn/ml_integration/`

- Verify: ML integration delegation
- Verify: Model training and evaluation

**Step 15**: `src/gnn/15_audio.py` → `src/gnn/audio/`

- Verify: Audio generation delegation
- Verify: Multi-backend audio (SAPF, Pedalboard)

**Step 16**: `src/gnn/16_analysis.py` → `src/gnn/analysis/`

- Verify: Statistical analysis delegation
- Verify: Performance metric computation
- Verify: Delegates to `process_analysis()` from `analysis` module

**Step 17**: `src/gnn/17_integration.py` → `src/gnn/integration/`

- Verify: System integration delegation
- Verify: Cross-module coordination

**Step 18**: `src/gnn/18_security.py` → `src/gnn/security/`

- Verify: Security validation delegation
- Verify: Access control implementation

**Step 19**: `src/gnn/19_research.py` → `src/gnn/research/`

- Verify: Research tools delegation
- Verify: Experimental features

**Step 20**: `src/gnn/20_website.py` → `src/gnn/website/`

- Verify: Website generation delegation
- Verify: Static HTML generation

**Step 21**: `src/gnn/21_mcp.py` → `src/gnn/mcp/`

- Verify: MCP processing delegation
- Verify: Tool registration

**Step 22**: `src/gnn/22_gui.py` → `src/gnn/gui/`

- Verify: GUI generation delegation
- Verify: Interactive model construction

**Step 23**: `src/gnn/23_report.py` → `src/gnn/report/`

- Verify: Report generation delegation
- Verify: Comprehensive analysis reports

#### Anti-Patterns to Identify

**Incorrect Pattern Examples**:

```python
# ❌ WRONG: Long function definition in numbered script
def generate_matrix_heatmap(data):
    # 50+ lines of visualization code
    ...
```

**Correct Pattern Examples**:

```python
# ✅ CORRECT: Thin orchestrator delegating to module
from gnn.utils.pipeline_orchestration.pipeline_template import create_standardized_pipeline_script
from visualization import process_visualization

run_script = create_standardized_pipeline_script(
    "8_visualization.py",
    process_visualization,
    "Matrix and network visualization processing",
)


def main() -> int:
    """Main entry point."""
    return run_script()


if __name__ == "__main__":
    sys.exit(main())
```

### 1.2 Centralized Utilities Usage

**Requirement**: All pipeline scripts must use centralized utilities from `src/gnn/utils/`.

#### P1.2 Logic Validation Checklist

- [ ] **Logging Utilities**: Uses `setup_step_logging()`, `log_step_start()`, `log_step_success()`, `log_step_error()`, `log_step_warning()`
- [ ] **Argument Parsing**: Uses `EnhancedArgumentParser` or `create_standardized_pipeline_script()`
- [ ] **Path Management**: Uses `pathlib.Path` objects, not string paths
- [ ] **Output Directory**: Uses `get_output_dir_for_script()` from `pipeline.config`
- [ ] **Configuration**: Uses `get_pipeline_config()` for centralized configuration
- [ ] **Visual Logging**: Uses `gnn.utils.observability.visual_logging` for progress indicators and status messages
- [ ] **Error Recovery**: Uses `gnn.utils.errors.error_recovery` for graceful degradation
- [ ] **Resource Management**: Uses `gnn.utils.runtime_safety.resource_manager` for memory tracking

#### Import Pattern Validation

**Correct Import Pattern**:

```python
from gnn.utils.pipeline_orchestration.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
)
from gnn.utils.arguments import ArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config
```

**Incorrect Import Pattern**:

```python
# ❌ WRONG: Direct logging setup without centralized utilities
import logging

logger = logging.getLogger(__name__)
```

### 1.3 Exit Code Patterns

**Requirement**: All pipeline scripts must return standardized exit codes.

#### Exit Code Validation

- **0**: Success - step completed without errors
- **1**: Critical Error - step failed and cannot continue
- **2**: Success with Warnings - step completed but with non-critical warnings

#### P1.3 Exit Code Validation Checklist

- [ ] **Main Function**: Returns integer exit code
- [ ] **Error Handling**: Returns 1 on critical errors
- [ ] **Warning Handling**: Returns 2 on success with warnings
- [ ] **Success Handling**: Returns 0 on complete success
- [ ] **Exit Statement**: Uses `sys.exit(main())` pattern

---

## 2. Code Quality Standards

### 2.1 Type Hints Coverage

**Requirement**: All public functions must have complete type hints with generic types for containers.

#### Type Hint Validation

- [ ] **Function Signatures**: All public functions have type hints
- [ ] **Return Types**: All functions specify return types (including `None`)
- [ ] **Parameter Types**: All parameters have type annotations
- [ ] **Generic Types**: Container types use generics (e.g., `List[str]`, `Dict[str, Any]`)
- [ ] **Union Types**: Optional parameters use `Optional[T]` or `T | None`
- [ ] **Type Imports**: Proper imports from `typing` module

#### Example Correct Pattern

```python
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


def process_validation(
    target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs: Any
) -> bool:
    """Process validation for GNN models."""
    ...
```

### 2.2 Docstring Standards

**Requirement**: Every public function/class must have comprehensive docstrings with examples.

#### Docstring Validation

- [ ] **Module Docstrings**: All modules have module-level docstrings
- [ ] **Class Docstrings**: All classes have class-level docstrings
- [ ] **Function Docstrings**: All public functions have docstrings
- [ ] **Parameter Documentation**: Parameters documented in docstrings
- [ ] **Return Documentation**: Return values documented
- [ ] **Example Usage**: Examples provided for complex functions
- [ ] **Raises Documentation**: Exceptions documented where applicable

#### Docstring Format

```python
def process_validation(
    target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs: Any
) -> bool:
    """
    Process validation for GNN models.

    Args:
        target_dir: Directory containing GNN files to validate
        output_dir: Output directory for validation results
        verbose: Enable verbose logging
        **kwargs: Additional validation options

    Returns:
        True if validation succeeded, False otherwise

    Raises:
        ValueError: If target_dir does not exist
        PermissionError: If output_dir is not writable

    Example:
        >>> from pathlib import Path
        >>> result = process_validation(
        ...     Path("input/gnn_files"),
        ...     Path("output"),
        ...     verbose=True
        ... )
        >>> assert result is True
    """
    ...
```

### 2.3 Error Handling Patterns

**Requirement**: Comprehensive error handling with graceful degradation and recovery mechanisms.

#### P2.2 Docstring Validation Checklist

- [ ] **Try-Except Blocks**: Critical operations wrapped in try-except
- [ ] **Specific Exceptions**: Catch specific exceptions, not bare `except:`
- [ ] **Error Logging**: Errors logged using `log_step_error()`
- [ ] **Recovery Mechanisms**: Graceful degradation when possible
- [ ] **Error Messages**: Clear, actionable error messages
- [ ] **Resource Cleanup**: Resources cleaned up in finally blocks
- [ ] **Error Propagation**: Appropriate error propagation to callers

#### Error Handling Pattern

```python
def process_with_error_handling(target_dir: Path, output_dir: Path) -> bool:
    """Process with comprehensive error handling."""
    try:
        # Critical operation
        result = perform_operation(target_dir, output_dir)
        return result
    except FileNotFoundError as e:
        log_step_error(logger, f"File not found: {e}")
        return False
    except PermissionError as e:
        log_step_error(logger, f"Permission denied: {e}")
        return False
    except Exception as e:
        log_step_error(logger, f"Unexpected error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False
    finally:
        # Resource cleanup
        cleanup_resources()
```

```json
{
  "step": 7,
  "status": "success"
}
```

### 2.4 Resource Management

**Requirement**: Proper cleanup of resources (files, connections, memory) in all code paths.

#### P3.4 CLI Integration Validation Checklist

- [ ] **File Handles**: Files closed properly (use context managers)
- [ ] **Memory Management**: Large objects cleaned up when no longer needed
- [ ] **Connection Management**: Network connections closed properly
- [ ] **Context Managers**: Use `with` statements for resource management
- [ ] **Memory Tracking**: Use `gnn.utils.runtime_safety.resource_manager` for memory monitoring

#### Resource Management Pattern

```python
from pathlib import Path
from gnn.utils.runtime_safety.resource_manager import get_current_memory_usage


def process_with_resource_management(target_dir: Path) -> bool:
    """Process with proper resource management."""
    initial_memory = get_current_memory_usage()

    try:
        # Use context managers for file operations
        with open(target_dir / "data.json", "r") as f:
            data = json.load(f)

        # Process data
        result = process_data(data)

        return result
    finally:
        # Cleanup
        final_memory = get_current_memory_usage()
        memory_delta = final_memory - initial_memory
        if memory_delta > 100 * 1024 * 1024:  # 100MB threshold
            logger.warning(f"High memory usage: {memory_delta} bytes")
```

### 2.5 Performance Monitoring

**Requirement**: Built-in timing and memory usage tracking for all major operations.

#### P3.5 File Parsing Validation Checklist

- [ ] **Timing**: Operations timed using `time.time()` or `time.perf_counter()`
- [ ] **Memory Tracking**: Memory usage tracked using `gnn.utils.runtime_safety.resource_manager`
- [ ] **Performance Logging**: Performance metrics logged appropriately
- [ ] **Performance Reports**: Performance data included in output reports

---

## 3. Module Structure Consistency

### 3.1 Directory Structure Validation

**Requirement**: All modules follow the established directory structure pattern.

#### Standard Module Structure

```
src/[module_name]/
├── __init__.py          # Public API exports
├── AGENTS.md            # Module documentation
├── README.md            # Module README (optional but recommended)
├── processor.py         # Core processing logic
├── [module_specific].py # Module-specific functionality
├── mcp.py               # MCP tool registration (where applicable)
└── [submodules]/        # Sub-modules if needed
```

#### Validation Checklist for Each Module

- [ ] **Directory Exists**: Module directory exists in `src/`
- [ ] ****init**.py Present**: Module has `__init__.py` file
- [ ] **AGENTS.md Present**: Module has `AGENTS.md` documentation
- [ ] **README.md Present**: Module has `README.md` (recommended)
- [ ] **Core Logic Files**: Core logic in separate files (not in numbered scripts)
- [ ] **MCP Integration**: `mcp.py` present where applicable
- [ ] **Sub-modules**: Sub-modules properly organized

### 3.2 `__init__.py` Public API Exports

**Requirement**: Module `__init__.py` files must export public API functions.

#### P4.1 Logic Validation Checklist

- [ ] **Public Functions Exported**: Main processing functions exported
- [ ] **Import Pattern**: Uses `from .processor import process_module`
- [ ] **API Clarity**: Clear public API with well-named functions
- [ ] **Module Info**: Module information functions exported (e.g., `get_module_info()`)

#### Example `__init__.py` Pattern

```python
"""
[Module Name] Module

[Brief description of module purpose]
"""

from pathlib import Path
from typing import Dict, Any, Optional

from .processor import process_module, process_module_standardized, get_module_info

__all__ = ["process_module", "process_module_standardized", "get_module_info"]


def get_module_info() -> Dict[str, Any]:
    """Get module information."""
    return {
        "name": "[module_name]",
        "version": "1.0.0",
        "description": "[Module description]",
        "capabilities": ["Capability 1", "Capability 2"],
    }
```

```mermaid
graph LR
    A[Step 1] --> B[Step 2]
```

### 3.3 MCP Integration

**Requirement**: Modules that provide MCP tools must have `mcp.py` with functional tool registration.

#### P4.2 State Validation Checklist

- [ ] **mcp.py Exists**: Module has `mcp.py` file (where applicable)
- [ ] **Tool Registration**: Tools properly registered
- [ ] **Tool Implementation**: Tools have real implementations
- [ ] **Tool Documentation**: Tools documented in AGENTS.md

#### MCP Integration Pattern

```python
"""
MCP Tool Registration for [Module Name]
"""

from typing import Dict, Any, List
from mcp import Server, Tool


def register_tools(server: Server) -> None:
    """Register MCP tools for this module."""

    @server.tool()
    def process_module_tool(target_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Process [module] with MCP tool.

        Args:
            target_dir: Directory containing input files
            output_dir: Output directory for results

        Returns:
            Processing results
        """
        from pathlib import Path
        from .processor import process_module

        result = process_module(Path(target_dir), Path(output_dir))
        return result
```

### 3.4 Module Separation

**Requirement**: Clear separation between numbered scripts and module implementations.

#### P4.3 Connection Validation Checklist

- [ ] **No Core Logic in Scripts**: Numbered scripts contain no core logic
- [ ] **Delegation Pattern**: Scripts delegate to module functions
- [ ] **Module Independence**: Modules can be imported independently
- [ ] **Testability**: Modules can be tested without running pipeline scripts

---

## 4. Documentation Completeness

### 4.1 AGENTS.md Documentation

**Requirement**: Every module must have comprehensive AGENTS.md documentation.

#### AGENTS.md Structure Validation

Each AGENTS.md should include:

- [ ] **Module Overview**: Purpose and role in pipeline
- [ ] **Pipeline Step**: Which numbered script uses this module
- [ ] **Category**: Module category (e.g., Validation, Visualization)
- [ ] **Core Functionality**: Primary responsibilities
- [ ] **Key Capabilities**: List of key capabilities
- [ ] **API Reference**: Public functions documented
- [ ] **Dependencies**: Required and optional dependencies
- [ ] **Configuration**: Configuration options
- [ ] **Usage Examples**: Real examples with actual GNN files
- [ ] **Integration Points**: How module integrates with others
- [ ] **MCP Tools**: MCP tools provided (if applicable)

#### AGENTS.md Template

```markdown
# [Module Name] Module - Agent Scaffolding

## Module Overview

**Purpose**: [Module purpose]

**Pipeline Step**: Step N: [Step Name] (N_[module].py)

**Category**: [Category]

---

## Core Functionality

### Primary Responsibilities
1. [Responsibility 1]
2. [Responsibility 2]

### Key Capabilities
- [Capability 1]
- [Capability 2]

---

## API Reference

### Public Functions

#### `process_module(target_dir, output_dir, **kwargs) -> bool`
**Description**: [Function description]

**Parameters**:
- `target_dir`: [Parameter description]
- `output_dir`: [Parameter description]
- `**kwargs`: [Additional options]

**Returns**: [Return description]

---

## Dependencies

### Required Dependencies
- [Dependency 1]
- [Dependency 2]

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

---

## Configuration

[Configuration options]

---

## Usage Examples

[Real examples with actual GNN files]
```

```mermaid
graph TD
    A[GNN File] --> B[Parser]
```

### 4.2 README.md Presence

**Requirement**: Modules should have README.md files for additional documentation.

#### P4.4 Module validation checklist

- [ ] **README.md Exists**: Module has README.md (recommended)
- [ ] **Module Description**: README describes module purpose
- [ ] **Usage Instructions**: Usage instructions provided
- [ ] **Examples**: Examples included
- [ ] **Troubleshooting**: Troubleshooting guide (if applicable)

### 4.3 API Documentation Consistency

**Requirement**: API documentation must be consistent across modules.

#### P4.5 Integration validation checklist

- [ ] **Function Signatures**: Docstrings match function signatures
- [ ] **Parameter Names**: Parameter names consistent across modules
- [ ] **Return Types**: Return types documented consistently
- [ ] **Example Format**: Examples follow consistent format
- [ ] **Error Documentation**: Errors documented consistently

---

## 5. Testing Standards

### 5.1 Real-Implementation Policy Compliance

**Requirement**: All tests must execute real code paths and real methods. No substitution frameworks.

#### P5.1 Performance validation checklist

- [ ] **No unittest.patch**: No `patch` imports or usage
- [ ] **No Monkeypatching**: No monkeypatching of functions or classes
- [ ] **Real Code Paths**: Tests execute actual code paths
- [ ] **Real Data**: Tests use real, representative data
- [ ] **Real Dependencies**: Tests use real dependencies (skip if unavailable)
- [ ] **File-Based Assertions**: Tests assert on real file outputs

#### Known Issues

**Current Status**: All strict real-implementation policies are enforced. Previous tests using `patch` have been refactored or removed.

**Policy**: Tests must execute real code paths and real methods. No substitution frameworks are allowed.

#### Anti-Pattern Detection

**Incorrect Pattern**:

```python
# ❌ WRONG: Using substitutions
from unittest.patch import patch  # Note: do not use


@patch("module.process_function")
def test_processing(patched_process):
    patched_process.return_value = True
    ...
```

**Correct Pattern**:

```python
# ✅ CORRECT: Real code execution
def test_processing():
    from pathlib import Path
    from module import process_function

    result = process_function(Path("test_data/input"), Path("test_data/output"))
    assert result is True
    assert (Path("test_data/output") / "result.json").exists()
```

### 5.2 Real Data Processing

**Requirement**: Tests must use real, representative data.

#### P5.2 Scalability validation checklist

- [ ] **Real GNN Files**: Tests use actual GNN files from test data
- [ ] **Real Formats**: Tests cover all supported formats (Markdown, JSON, YAML, etc.)
- [ ] **Real Outputs**: Tests validate real output files
- [ ] **Real API Calls**: Tests make real API calls (skip if unavailable)
- [ ] **Real File I/O**: Tests perform real file operations

### 5.3 Integration Test Coverage

**Requirement**: End-to-end tests must validate complete pipeline execution with real inputs/outputs.

#### P5.3 Resource validation checklist

- [ ] **Pipeline Tests**: Tests for full pipeline execution
- [ ] **Step Integration**: Tests for step-to-step integration
- [ ] **Cross-Module Tests**: Tests for cross-module functionality
- [ ] **Output Validation**: Tests validate actual output artifacts
- [ ] **Error Scenarios**: Tests for real error conditions

#### Integration Test Pattern

```python
def test_pipeline_integration():
    """Test full pipeline execution with real data."""
    from pathlib import Path
    import subprocess

    # Run actual pipeline script
    result = subprocess.run(
        ["python", "src/gnn/main.py", "--target-dir", "test_data/gnn_files"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0

    # Validate actual outputs using standardized output directory names
    output_dir = Path("output")
    assert (output_dir / "3_gnn_output" / "parsed_models.json").exists()
    assert (output_dir / "7_export_output" / "exports").exists()

    # Verify output directory naming follows convention
    assert output_dir.name == "output"
    assert (output_dir / "3_gnn_output").exists()
    assert (output_dir / "7_export_output").exists()
```

### 5.4 Performance and Regression Tests

**Requirement**: Include timing assertions for critical paths, memory usage validation.

#### P6.1 Script execution validation checklist

- [ ] **Timing Assertions**: Critical paths have timing assertions
- [ ] **Memory Validation**: Memory usage validated
- [ ] **Performance Baselines**: Performance baselines established
- [ ] **Regression Detection**: Tests detect performance regressions

---

## 6. Pipeline Integration

### 6.1 Step Dependencies

**Requirement**: Pipeline steps must have correct dependency relationships.

#### P6.2 Resource usage validation checklist

- [ ] **Dependency Declaration**: Dependencies declared in configuration
- [ ] **Dependency Validation**: Dependencies validated before step execution
- [ ] **Circular Dependency Check**: No circular dependencies
- [ ] **Missing Dependency Detection**: Missing dependencies detected

#### Expected Dependency Graph

```
0_template → (no dependencies)
1_setup → (no dependencies)
2_tests → 1_setup
3_gnn → 1_setup
4_model_registry → 3_gnn
5_type_checker → 3_gnn
6_validation → 3_gnn, 5_type_checker
7_export → 3_gnn
8_visualization → 3_gnn, 7_export
9_advanced_viz → 3_gnn, 8_visualization
10_ontology → 3_gnn
11_render → 3_gnn, 10_ontology
12_execute → 11_render
13_llm → 3_gnn
14_ml_integration → 3_gnn, 12_execute
15_audio → 3_gnn
16_analysis → 3_gnn, 12_execute
17_integration → (multiple dependencies)
18_security → 3_gnn
19_research → 3_gnn
20_website → (multiple dependencies)
21_mcp → 3_gnn
22_gui → 3_gnn
23_report → (multiple dependencies)
```

### 6.2 Data Flow Validation

**Requirement**: Data flows correctly between pipeline steps.

#### P7.1 Error recovery validation checklist

- [ ] **Input Validation**: Steps validate inputs from previous steps
- [ ] **Output Format**: Steps produce expected output formats
- [ ] **Data Transformation**: Data transformations are correct
- [ ] **Semantic Preservation**: Semantic information preserved across steps

### 6.3 Output Consistency

**Requirement**: Output formats and structures must be consistent.

#### P7.2 Logging coherence validation checklist

- [ ] **Output Directory Structure**: Output directories follow conventions
- [ ] **Output File Naming**: Output files follow naming conventions
- [ ] **Output Format**: Output formats are consistent
- [ ] **Metadata**: Output metadata is consistent

#### Output Directory Conventions

**Enforcement**: Output directory names are enforced via `get_output_dir_for_script()` in `src/config.py`. This function maps script names to standardized output directories.

**Standard Pattern**: `N_[module]_output/` where N is the step number and [module] is the module name.

```
output/
├── 0_template_output/
├── 1_setup_output/
├── 2_tests_output/
├── 3_gnn_output/
├── 4_model_registry_output/
├── 5_type_checker_output/
├── 6_validation_output/
├── 7_export_output/
├── 8_visualization_output/
├── 9_advanced_viz_output/
├── 10_ontology_output/
├── 11_render_output/
├── 12_execute_output/
├── 13_llm_output/
├── 14_ml_integration_output/
├── 15_audio_output/
├── 16_analysis_output/
├── 17_integration_output/
├── 18_security_output/
├── 19_research_output/
├── 20_website_output/
├── 21_mcp_output/
├── 22_gui_output/
└── 23_report_output/
```

**Validation**: All pipeline scripts should use `get_output_dir_for_script()` from `pipeline.config` to ensure consistent output directory naming.

---

## 7. Performance Standards

### 7.1 Execution Time

**Requirement**: Full pipeline execution must complete within 30 minutes for standard workloads.

#### P8.1 Package structure validation checklist

- [ ] **Pipeline Timing**: Full pipeline execution time < 30 minutes
- [ ] **Step Timing**: Individual step timing tracked
- [ ] **Performance Logging**: Performance metrics logged
- [ ] **Performance Reports**: Performance data in output reports

### 7.2 Memory Usage

**Requirement**: Peak memory usage must not exceed 2GB for standard workloads.

#### P8.2 Dependency validation checklist

- [ ] **Memory Tracking**: Memory usage tracked per step
- [ ] **Memory Limits**: Memory usage within 2GB limit
- [ ] **Memory Cleanup**: Memory cleaned up between steps
- [ ] **Memory Reports**: Memory usage in performance reports

### 7.3 Success and Error Rates

**Requirement**:

- Success rate > 99% for all pipeline steps
- Critical failure rate < 1%

#### P8.3 Configuration validation checklist

- [ ] **Success Rate Tracking**: Success rates tracked per step
- [ ] **Error Rate Tracking**: Error rates tracked per step
- [ ] **Error Recovery**: Error recovery mechanisms in place
- [ ] **Error Reporting**: Error rates in performance reports

---

## 8. Security and Validation

### 8.1 Input Validation

**Requirement**: All inputs must be validated and sanitized.

#### P9.1 Documentation parity validation checklist

- [ ] **Path Validation**: File paths validated for security
- [ ] **Input Sanitization**: User inputs sanitized
- [ ] **Type Validation**: Input types validated
- [ ] **Range Validation**: Input ranges validated

### 8.2 Error Recovery

**Requirement**: Comprehensive error recovery mechanisms.

#### P9.2 Cross-reference validation checklist

- [ ] **Recovery Strategies**: Error recovery strategies defined
- [ ] **Graceful Degradation**: Graceful degradation implemented
- [ ] **Retry Logic**: Retry logic for transient errors
- [ ] **Error Reporting**: Clear error reporting with recovery suggestions

### 8.3 Security Patterns

**Requirement**: Security patterns implemented where applicable.

#### P9.3 Metadata validation checklist

- [ ] **Access Control**: Access control where applicable
- [ ] **Secure Configuration**: Configuration managed securely
- [ ] **Threat Detection**: Threat detection patterns (if applicable)
- [ ] **Security Validation**: Security validation in security module

---

## 9. Naming Conventions

### 9.1 Module Naming

**Requirement**: Consistent naming across modules, functions, and classes.

#### P10.1 Access control validation checklist

- [ ] **Module Names**: Module names follow `snake_case`
- [ ] **Function Names**: Function names follow `snake_case`
- [ ] **Class Names**: Class names follow `PascalCase`
- [ ] **Constant Names**: Constants follow `UPPER_SNAKE_CASE`

### 9.2 File Naming

**Requirement**: File names follow conventions.

#### P10.2 Secret management validation checklist

- [ ] **Pipeline Scripts**: Numbered scripts follow `N_[module].py` pattern
- [ ] **Module Files**: Module files follow `snake_case.py` pattern
- [ ] **Test Files**: Test files follow `test_[module].py` pattern

### 9.3 Output Naming

**Requirement**: Output files and directories follow naming conventions.

#### P10.3 Vulnerability scanning validation checklist

- [ ] **Output Directories**: Follow `N_[module]_output/` pattern
- [ ] **Output Files**: Output files have descriptive names
- [ ] **File Extensions**: File extensions match content types

---

## 10. Dependency Management

### 10.1 Import Patterns

**Requirement**: Consistent import patterns across the codebase.

#### Validation Checklist

- [ ] **Standard Library First**: Standard library imports first
- [ ] **Third-Party Second**: Third-party imports second
- [ ] **Local Imports Last**: Local imports last
- [ ] **Import Grouping**: Imports grouped appropriately
- [ ] **Absolute Imports**: Use absolute imports from `src/`

#### Import Pattern

```python
# Standard library
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

# Third-party
import numpy as np
import pandas as pd

# Local imports
from gnn.utils.pipeline_orchestration.pipeline_template import setup_step_logging
from pipeline.config import get_output_dir_for_script
from module.processor import process_module
```

### 10.2 Module Coupling

**Requirement**: Modules should minimize dependencies on other modules.

#### P12.1 Internal consistency validation checklist

- [ ] **Low Coupling**: Modules have minimal dependencies
- [ ] **Dependency Injection**: Dependency injection used where appropriate
- [ ] **Interface Segregation**: Large interfaces broken into smaller ones
- [ ] **Circular Dependency Check**: No circular dependencies

### 10.3 Dependency Validation

**Requirement**: Dependencies must be properly declared and validated.

#### P12.2 External consistency validation checklist

- [ ] **Dependency Declaration**: Dependencies declared in `pyproject.toml`
- [ ] **Dependency Validation**: Dependencies validated at runtime
- [ ] **Graceful Degradation**: Graceful degradation when dependencies unavailable
- [ ] **Optional Dependencies**: Optional dependencies handled appropriately

---

## 11. Improvement Recommendations Framework

### 11.1 Current Status Assessment

For each validation area, assess:

- **Current Status**: What exists and what's missing
- **Compliance Score**: Percentage of compliance (0-100%)
- **Critical Issues**: Must-fix items (blocking issues)
- **Improvement Opportunities**: Nice-to-have enhancements
- **Action Items**: Specific tasks with file paths and line numbers

### 11.2 Compliance Scoring

#### Scoring Methodology

- **100%**: All validation criteria met
- **90-99%**: Minor improvements needed
- **80-89%**: Some improvements needed
- **70-79%**: Significant improvements needed
- **<70%**: Major refactoring required

#### Scoring Categories

1. **Architecture Compliance**: Thin orchestrator pattern, centralized utilities
2. **Code Quality**: Type hints, docstrings, error handling
3. **Module Structure**: Directory structure, `__init__.py`, MCP integration
4. **Documentation**: AGENTS.md, README.md, API documentation
5. **Testing**: Real-implementation policy, real data, integration tests
6. **Pipeline Integration**: Dependencies, data flow, output consistency
7. **Performance**: Execution time, memory usage, success rates
8. **Security**: Input validation, error recovery, security patterns
9. **Naming Conventions**: Consistent naming across codebase
10. **Dependency Management**: Import patterns, module coupling

### 11.3 Action Item Format

Each action item should include:

- **File Path**: Specific file to modify
- **Line Numbers**: Specific lines (if applicable)
- **Issue Description**: What needs to be fixed
- **Recommended Fix**: How to fix it
- **Priority**: Critical, High, Medium, Low

#### Action Item Example

```markdown
### Action Item: Add Type Hints to process_validation

- **File**: `src/gnn/validation/processor.py`
- **Lines**: 45-60
- **Issue**: Function `process_validation` missing type hints
- **Recommended Fix**: Add type hints for all parameters and return type
- **Priority**: High

```python
# Current (missing type hints)
def process_validation(target_dir, output_dir, verbose=False, **kwargs):
    ...

# Recommended (with type hints)
def process_validation(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs: Any
) -> bool:
    ...
```

---

## 12. Usage Instructions

### 12.1 Manual Review Process

1. **Start with Executive Summary**: Review high-level coherence metrics
2. **Architecture Compliance**: Validate thin orchestrator pattern for all 25 steps
3. **Code Quality**: Check type hints, docstrings, error handling
4. **Module Structure**: Verify directory structure and `__init__.py` patterns
5. **Documentation**: Validate AGENTS.md and README.md completeness
6. **Testing**: Verify real-implementation policy and real data usage
7. **Integration**: Check pipeline dependencies and data flow
8. **Performance**: Review execution time and memory usage
9. **Security**: Validate input validation and error recovery
10. **Naming**: Check naming conventions consistency

### 12.2 Automated Validation Integration

#### Using Pipeline Validation Script

```bash
# Run the pipeline runtime validator (E2E via subprocess)
python -m gnn.pipeline.pipeline_runtime_validator

# Run preflight checks
python -m cli preflight
```

#### Using Test Suite

```bash
# Run all tests
uv run python src/gnn/2_tests.py --comprehensive
```

```bash
uv run --extra dev python -m pytest tests/gnn -q
```

# Run specific test module

uv run --extra dev python -m pytest tests/validation/test_validation_overall.py -v

```

### 12.3 Generating Coherence Reports

1. **Run Validation Scripts**: Execute pipeline validation and test suite
2. **Review Output**: Review validation reports and test results
3. **Assess Compliance**: Score each validation area
4. **Create Action Items**: Generate specific action items with priorities
5. **Track Improvements**: Track improvements over time

### 12.4 AI-Assisted Review

This mega-prompt can be used with AI assistants to:

1. **Systematic Review**: Review entire codebase systematically
2. **Pattern Detection**: Detect anti-patterns and inconsistencies
3. **Compliance Scoring**: Calculate compliance scores for each area
4. **Action Item Generation**: Generate specific, actionable improvement items
5. **Documentation Updates**: Update documentation based on findings

#### AI Review Prompt Template

```

Using the REPO_COHERENCE_CHECK.md mega-prompt, review the following:

1. Architecture compliance for [specific module/step]
2. Code quality standards (type hints, docstrings, error handling)
3. Module structure consistency
4. Documentation completeness
5. Testing standards compliance

Provide:

- Compliance scores for each area
- Specific issues found with file paths and line numbers
- Recommended fixes with code examples
- Priority levels for each action item

```

---

## 13. Integration with Existing Validation Tools

### 13.1 Pipeline Validation Module

**Location**: `src/gnn/pipeline/pipeline_runtime_validator.py`

**Capabilities**:

- Validates module imports and centralized utilities usage
- Checks configuration consistency
- Validates output structure
- Checks argument consistency
- Detects dependency cycles
- Validates output naming conventions
- Checks performance tracking coverage

**Usage**:

```bash
python -m gnn.pipeline.pipeline_runtime_validator
```

### 13.2 Validation Module

**Location**: `src/gnn/validation/`

**Capabilities**:

- Model consistency checking
- Semantic validation
- Quality assessment
- Cross-format validation

### 13.3 Test Infrastructure

**Location**: `tests/`

**Capabilities**:

- Comprehensive test suite
- Integration tests
- Performance tests
- Regression tests

---

## 14. Examples and Anti-Patterns

### 14.1 Correct Patterns

#### Thin Orchestrator Pattern

```python
#!/usr/bin/env python3
"""
Step N: [Module Name] (Thin Orchestrator)
"""

from gnn.utils.pipeline_orchestration.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
    create_standardized_pipeline_script,
)
from module import process_module

run_script = create_standardized_pipeline_script(
    "N_module.py", process_module, "Module processing description"
)


def main() -> int:
    """Main entry point."""
    return run_script()


if __name__ == "__main__":
    sys.exit(main())
```

#### Module Implementation Pattern

```python
"""
Module Processor

Core processing logic for [module name].
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging


def process_module(
    target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs: Any
) -> bool:
    """
    Process [module] functionality.

    Args:
        target_dir: Input directory
        output_dir: Output directory
        verbose: Enable verbose logging
        **kwargs: Additional options

    Returns:
        True if processing succeeded
    """
    logger = logging.getLogger(__name__)

    try:
        # Core processing logic here
        result = perform_processing(target_dir, output_dir, **kwargs)
        return result
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False
```

### 14.2 Anti-Patterns to Avoid

#### Thick Orchestrator (Incorrect)

```python
# ❌ WRONG: Long implementation in numbered script
def generate_visualization(data):
    # 100+ lines of visualization code
    import matplotlib.pyplot as plt
    import numpy as np

    # ... extensive implementation ...
    return result
```

#### Missing Type Hints (Incorrect)

```python
# ❌ WRONG: Missing type hints
def process_validation(target_dir, output_dir, verbose=False): ...
```

#### Using Substitutions in Tests (Incorrect)

```python
# ❌ WRONG: Using substitutions
from unittest.patch import patch  # Note: do not use


@patch("module.process_function")
def test_processing(patched_process):
    patched_process.return_value = True
    ...
```

---

## 15. Conclusion

This mega-prompt provides a comprehensive framework for validating repo-wide coherence across all 25 pipeline steps and 32 Python packages under `src/`. Use it systematically to:

1. **Identify Issues**: Find architectural, code quality, and documentation issues
2. **Score Compliance**: Calculate compliance scores for each validation area
3. **Generate Action Items**: Create specific, actionable improvement tasks
4. **Track Progress**: Monitor improvements over time
5. **Maintain Standards**: Ensure ongoing compliance with established patterns

### Next Steps

1. Run automated validation scripts
2. Perform manual review using this checklist
3. Generate compliance scores for each area
4. Create prioritized action items
5. Implement improvements systematically
6. Re-validate after changes

---

**Status**: Maintained
**Version**: v3.2.0 Engine (Bundle v2.0.0)
**Coverage**: all 25 pipeline steps plus the live module and `AGENTS.md`
inventories; derive changing counts from `src/gnn/AGENTS.md` and `rg --files`

---

## 16. Known Issues and Action Items

### P16.1 Testing policy validation checklist

**Issue**: Some test files violate the real-implementation policy by using substitutions.

**Affected Files**:

- `tests/utils/test_pipeline_warnings_fix.py`
- `tests/pipeline/test_pipeline_recovery.py`
- `tests/pipeline/test_pipeline_error_scenarios.py`
- `tests/test_fast_suite.py`
- `tests/visualization/test_d2_visualizer.py`
- `tests/advanced_visualization/test_advanced_visualization_overall.py`
- `tests/conftest.py`

**Priority**: High
**Action**: Refactor these tests to use real code paths and real data. Tests may skip when external dependencies are unavailable, but must never replace dependencies with substitutions.

### 16.2 Validation Priorities

When using this mega-prompt for code review, prioritize:

1. **Critical**: Architecture compliance (thin orchestrator pattern)
2. **High**: Code quality (type hints, docstrings, error handling)
3. **High**: Testing standards (real-implementation policy compliance)
4. **Medium**: Documentation completeness (AGENTS.md, README.md)
5. **Medium**: Module structure consistency
6. **Low**: Performance optimization opportunities
7. **Low**: Naming convention improvements

### 16.3 Quick Validation Commands

```bash
# Check for patch usage
grep -r "import patch" tests/

# Check pipeline script lengths
find src -name "[0-9]_*.py" -exec wc -l {} \; | sort -n

# Check for missing type hints in public functions
# (requires type checking tool like mypy or pyright)

# Validate output directory structure
ls -la output/ | grep "_output"

# Check AGENTS.md coverage
find src -name "AGENTS.md" | wc -l
```
