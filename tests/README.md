# Test Infrastructure

Test suite for the GNN Processing Pipeline. Tests mirror the
`src/gnn/` package layout: `tests/<module>/test_*.py` contains tests for
`src/gnn/<module>/`. Cross-module and infrastructure tests live at the top
level of `tests/`.

## Quick Start

### Run full test suite via the pipeline orchestrator

```bash
python src/gnn/2_tests.py --fast-only --verbose       # quick validation
python src/gnn/2_tests.py --comprehensive --verbose   # full suite
```

### Run tests for a single module

```bash
uv run --extra dev python -m pytest tests/gnn/ -v
uv run --extra dev python -m pytest tests/render/ -v
uv run --extra dev python -m pytest tests/execute/ -v
```

### Run by marker

```bash
uv run --extra dev python -m pytest -m unit          # unit tests only
uv run --extra dev python -m pytest -m integration   # integration tests only
uv run --extra dev python -m pytest -m fast          # fast tests only
uv run --extra dev python -m pytest -m slow          # slow/performance tests only
```

Opt-in markers: `pipeline` and `mcp` tests are excluded from CI and the
default suite (`-m "not pipeline and not mcp"`); run them explicitly when
working on those surfaces (see also `just bench` for pipeline performance).

## Test Statistics

- **Live file inventory**: `rg --files tests -g 'test_*.py'`
- **Live collection**: run the command-of-record collect pass with the two
  local Ollama integration files ignored
- **Pass/skip receipt**: the root [`README.md`](../../README.md) owns the latest
  dated full-run evidence; this module guide deliberately does not copy it
- **Fast-test duration**: 1-3 minutes
- **Full-suite duration**: varies by optional backend availability; latest
  command-of-record run completed in 12:09

## Directory Layout

```
tests/
├── conftest.py            # pytest fixtures + marker registration
├── categories.py          # category definitions for the modular runner
├── runner.py              # Step-2 orchestrator (TestRunner class)
├── mcp.py                 # MCP tool registrations for test execution
├── __init__.py            # public API surface (performance_tracker etc.)
├── helpers/               # shared test helpers (render_recovery)
├── infrastructure/        # TestRunner backend (ResourceMonitor, TestConfig)
├── test_data/             # on-disk fixtures consumed by tests
├── tests/                 # shared-plumbing regression tests (intentional)
│
├── <module>/test_*.py     # per-module tests mirroring src/gnn/<module>/
│   (46 maintained first-level subdirectories; 43 contain direct test files)
│
└── test_*.py              # cross-cutting / meta-tests at root
    (coverage assessments, environment probes, runner self-tests,
    test_core_modules, test_fast_suite, etc.)
```

### Mirror coverage

| Directory | Exempt rationale |
|-----------|------------------|
| `doc` | static Markdown subtree; only code is a `__version__` re-export and a no-op `register_tools` MCP stub — `tests/docs/` audits repo `docs/` tooling, not this package |
| `documentation` | `__init__`-only package, `.md` references ship as data files |
| `formal_specs` | `__init__`-only package, payload consumed by `tests/test_grammar_spec_payloads.py` |
| `grammars` | `__init__`-only package, `.bnf`/`.ebnf` payloads exercised by `tests/test_grammar_spec_payloads.py` |
| `gnn_examples` | example-data directory (single `.md` exemplar), no importable surface |
| `manuscript` | public surface exercised by `tests/main/test_manuscript_variables*.py` + root `tests/test_manuscript_*.py` |
| `schema` | exercised by `tests/gnn/test_gnn_schema.py` (`GNNParseError`, `parse_state_space`) |
| `schemas` | section_contract pinned by `tests/gnn/test_section_contract.py`, remaining artifacts non-Python |

New mirrors exercise only the pinned entry points (markdown parse path, registry presence, frontmatter utilities; discovery, lightweight/parse/structure-check surfaces) — the 23-format parser registry and five-phase `GNNProcessor` orchestration remain beyond the minimal mirrors.

### What lives where

| Location              | Contains                                         |
|-----------------------|--------------------------------------------------|
| `tests/<module>/`     | Tests for a single `src/gnn/<module>/` package   |
| `tests/pipeline/`     | Cross-module integration tests (main orchestrator, render→execute→analyze chains, step11↔step12 handshakes) |
| `tests/utils/`        | Tests for `src/gnn/utils/` helpers + shared contracts (exit codes, validation schemas, framework availability) |
| `tests/` (root)       | Environment probes, coverage audits, test-runner self-tests, cross-module smoke tests |

## Architecture

The test infrastructure follows the **thin orchestrator pattern**, where `2_tests.py` acts as a thin wrapper that delegates all core functionality to the `tests/` module.

### Component Overview

**2_tests.py** (Thin Orchestrator):

- Handles command-line argument parsing
- Sets up logging and visual output
- Manages output directory creation
- Delegates to `tests.run_tests()` from `tests/__init__.py`
- Returns standardized exit codes (0=success, 1=error, 2=success with warnings/skipped)

**runner.py** (Core Implementation):

- Contains all test execution logic
- Provides multiple execution modes: fast, comprehensive, reliable
- Implements `ModularTestRunner` for category-based execution
- Handles resource monitoring, timeouts, and error recovery
- Generates comprehensive test reports

**`gnn.utils.testing` package** (Shared Utilities):

- Provides test fixtures and helper functions (`testing/` concern package: constants, environment, fixtures, reports, runner)
- Defines test categories, stages, markers, and configuration
- Provides test data creation utilities
- Used by both test files and the runner (the old top-level `testing_utils.py` facade was removed when the SC-38 window closed)

**conftest.py** (Pytest Configuration):

- Defines pytest fixtures available to all tests
- Configures pytest markers
- Handles test environment setup/teardown
- Provides shared test utilities

### Execution Flow

```mermaid
flowchart TD
    A["2_tests.py<br/>(CLI Entry Point)"] -->|"Parse arguments"| B["Setup logging"]
    B -->|"Create output dir"| C["tests.run_tests()"]
    C -->|"Route by mode"| D{"Test Mode"}
    D -->|"fast_only=True"| E["run_fast_pipeline_tests()"]
    D -->|"comprehensive=True"| F["run_comprehensive_tests()"]
    D -->|"recovery"| G["run_fast_reliable_tests()"]
    
    E --> H["ModularTestRunner"]
    F --> H
    G --> H
    
    H -->|"Execute categories"| I["pytest subprocess"]
    I -->|"Collect tests"| J["Test discovery"]
    I -->|"Run tests"| K["Test execution"]
    K -->|"Collect results"| L["Result parsing"]
    L -->|"Generate reports"| M["JSON/Markdown reports"]
    
    style A fill:#e1f5ff
    style C fill:#fff4e1
    style H fill:#e8f5e9
    style I fill:#f3e5f5
    style M fill:#fce4ec
```

## Test File Structure

### Module-Based Naming Convention

All test files follow the pattern:

- `test_MODULENAME_overall.py` - Comprehensive module coverage
- `test_MODULENAME_area1.py` - Specific module areas
- `test_MODULENAME_area2.py` - Additional specialized areas

### Current Test Files

#### Core Module Tests

- `gnn/test_gnn_overall.py` - Comprehensive GNN module testing
- `gnn/test_gnn_parsing.py` - GNN parsing and discovery tests
- `gnn/test_gnn_validation.py` - GNN validation and consistency tests
- `gnn/test_gnn_processing.py` - GNN processing and serialization tests

#### Render Module Tests

- `render/test_render_overall.py` - Comprehensive render module testing
- `render/test_render_integration.py` - Render integration tests
- `render/test_render_performance.py` - Render performance tests

#### MCP Module Tests

- `mcp/test_mcp_overall.py` - Comprehensive MCP module testing
- `mcp/test_mcp_tools.py` - MCP tool execution tests
- `mcp/test_mcp_performance.py` - MCP performance tests

#### Audio Module Tests

- `audio/test_audio_overall.py` - Comprehensive audio module testing
- `audio/test_audio_sapf.py` - SAPF audio generation tests
- `audio/test_audio_generation.py` - Audio generation tests
- `audio/test_audio_integration.py` - Audio integration tests

#### Visualization Module Tests

- `visualization/test_visualization_overall.py` - Comprehensive visualization module testing
- `visualization/test_visualization_matrices.py` - Matrix visualization tests
- `visualization/test_visualization_ontology.py` - Ontology visualization tests

#### Pipeline Module Tests

- `pipeline/test_pipeline_integration.py` - Pipeline integration tests
- `pipeline/test_pipeline_orchestration.py` - Pipeline orchestration tests
- `pipeline/test_pipeline_performance.py` - Pipeline performance tests
- `pipeline/test_pipeline_recovery.py` - Pipeline recovery tests
- `pipeline/test_pipeline_scripts.py` - Pipeline script tests
- `pipeline/test_pipeline_infrastructure.py` - Pipeline infrastructure tests
- `pipeline/test_pipeline_functionality.py` - Pipeline functionality tests

#### Export Module Tests

- `export/test_export_overall.py` - Comprehensive export module testing

#### Execute Module Tests

- `execute/test_execute_overall.py` - Comprehensive execute module testing

#### LLM Module Tests

- `llm/test_llm_overall.py` - Comprehensive LLM module testing

#### Ontology Module Tests

- `ontology/test_ontology_overall.py` - Comprehensive ontology module testing

#### Website Module Tests

- `website/test_website_overall.py` - Comprehensive website module testing

#### Report Module Tests

- `report/test_report_overall.py` - Comprehensive report module testing
- `report/test_report_generation.py` - Report generation tests
- `report/test_report_integration.py` - Report integration tests
- `report/test_report_formats.py` - Report format tests

#### Environment Module Tests

- `infrastructure/test_environment_overall.py` - Comprehensive environment module testing
- `infrastructure/test_environment_dependencies.py` - Environment dependency tests
- `infrastructure/test_environment_integration.py` - Environment integration tests
- `infrastructure/test_environment_python.py` - Python environment tests
- `infrastructure/test_environment_system.py` - System environment tests

#### Comprehensive Tests

Mirrors the `comprehensive` entry in `categories.py`:

- `api/test_comprehensive_api.py` - Comprehensive API testing
- `pipeline/test_main_orchestrator.py` - Main orchestrator tests
- `test_core_modules.py` - Core module integration tests (root)
- `infrastructure/test_coverage_assessment.py` - Coverage assessment tests
- `infrastructure/test_coverage_overall.py` - Coverage tests
- `test_fast_suite.py` - Fast test suite (root)
- `test_performance_overall.py` - Performance tests (root)
- `test_runner_helper.py` - Runner helper CLI tests (root)
- `test_unit_overall.py` - Unit tests (root)
- `utils/test_new_utils.py` - Utils module tests

## Test Runner Configuration

The test runner (`test_runner_modular.py`) executes the categories defined in
`categories.py`. Category `files` entries are paths relative to `tests/`
(module subdirectories); entries matching nothing are skipped with a warning,
and `missing_category_files()` reports any stale entry contract-wide:

```python
MODULAR_TEST_CATEGORIES = {
    "gnn": {
        "name": "GNN Module Tests",
        "description": "GNN processing and validation tests",
        "files": [
            "gnn/test_gnn_overall.py",
            "gnn/test_gnn_parsing.py",
            "gnn/test_gnn_processing.py",
            "gnn/test_gnn_validation.py",
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True,
    },
    # ... 24 categories total; the table is pinned by
    # tests/tests/test_categories_contract.py (missing_category_files() == {})
}
```

## Test Execution

```mermaid
graph TD
    Runner[Test Runner] --> Config[Configuration]
    Runner --> Disc[Test Discovery]
    
    Disc --> Filters{Filtering}
    Filters -->|Category| Unit[Unit Tests]
    Filters -->|Marker| Integ[Integration Tests]
    Filters -->|Pattern| Perf[Performance Tests]
    
    Unit --> Exec[Test Execution]
    Integ --> Exec
    Perf --> Exec
    
    Exec --> Results[Result Collection]
    Results --> Report[Test Report]
    Results --> Metrics[Metrics Analysis]
```

### Running Tests

#### Fast Tests (Default - Pipeline Mode)

```bash
# Run fast tests only (default for pipeline)
python src/gnn/2_tests.py --fast-only --verbose

# Or simply (fast-only is default)
python src/gnn/2_tests.py --verbose
```

#### Comprehensive Tests

```bash
# Run all tests including slow and performance tests
python src/gnn/2_tests.py --comprehensive --verbose
```

#### Direct Test Runner

```bash
# Run fast test suite directly
python tests/run_fast_tests.py
```

### Environment Variables

The test runner supports several environment variables for configuration:

#### `SKIP_TESTS_IN_PIPELINE`

Skip all tests during pipeline execution (for faster pipeline runs).

```bash
export SKIP_TESTS_IN_PIPELINE=1
python src/gnn/main.py  # Tests will be skipped
```

#### `FAST_TESTS_TIMEOUT`

Override the default timeout for fast tests (default: 600 seconds = 10 minutes).

```bash
export FAST_TESTS_TIMEOUT=300  # 5 minutes
python src/gnn/2_tests.py --fast-only
```

#### Usage Examples

```bash
# Skip tests in pipeline for faster execution
SKIP_TESTS_IN_PIPELINE=1 python src/gnn/main.py

# Run fast tests with custom timeout
FAST_TESTS_TIMEOUT=180 python src/gnn/2_tests.py --fast-only

# Run comprehensive tests with verbose output
python src/gnn/2_tests.py --comprehensive --verbose
```

## Test Utilities

### Shared Test Utilities (`src/gnn/utils/testing/`)

- `TEST_CATEGORIES` - Test category definitions
- `TEST_STAGES` - Test execution stages
- `TEST_CONFIG` - Test configuration
- `is_safe_mode()` - Safe mode detection
- `setup_test_environment()` - Test environment setup
- `create_sample_gnn_content()` - Sample GNN content creation
- `performance_tracker()` - Performance tracking decorator
- `get_memory_usage()` - Memory usage monitoring
- `assert_file_exists()` - File existence assertions
- Report generation functions

### Test Fixtures (`conftest.py`)

- `project_root` - Project root directory
- `src_dir` - Source directory
- `test_dir` - Test directory
- `safe_filesystem` - Safe filesystem operations
- `sample_gnn_files` - Sample GNN files
- `isolated_temp_dir` - Isolated temporary directory
- `comprehensive_test_data` - Comprehensive test data

## Test Markers

### Available Markers

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests
- `@pytest.mark.fast` - Fast tests
- `@pytest.mark.performance` - Performance tests (also auto-applied to `slow` tests by `conftest.py`)

Registered in `pytest.ini` (single source of truth); `--strict-markers` rejects unregistered markers (there is no `safe_to_fail` marker).

### Running Tests by Marker

```bash
# Run only unit tests
uv run --extra dev python -m pytest -m unit

# Run only integration tests
uv run --extra dev python -m pytest -m integration

# Run fast tests only
uv run --extra dev python -m pytest -m fast

# Exclude slow tests
uv run --extra dev python -m pytest -m "not slow"
```

## Test Categories

### Core Module Tests

- **GNN Module**: Processing, validation, parsing, integration
- **Render Module**: Code generation, multiple targets, performance
- **MCP Module**: Model Context Protocol, tools, transport, integration
- **Audio Module**: SAPF, generation, integration
- **Visualization Module**: Graphs, matrices, ontology, interactive

### Infrastructure Module Tests

- **Pipeline Module**: Orchestration, steps, configuration, performance, recovery
- **Export Module**: Multi-format export (JSON, XML, GraphML, GEXF, Pickle)
- **Execute Module**: Execution and simulation
- **LLM Module**: LLM integration and analysis
- **Ontology Module**: Ontology processing and validation
- **Website Module**: Website generation
- **Report Module**: Report generation and formatting
- **Environment Module**: Environment setup and validation

### Specialized Module Tests

- **Type Checker Module**: Type checking and validation
- **Validation Module**: Validation and consistency
- **Model Registry Module**: Model registry and versioning
- **Analysis Module**: Analysis and statistical
- **Integration Module**: System integration
- **Security Module**: Security validation
- **Research Module**: Research tools
- **ML Integration Module**: Machine learning integration
- **Advanced Visualization Module**: Advanced visualization

### Comprehensive Tests

- **Comprehensive API**: Complete API testing
- **Core Modules**: Core module integration
- **Fast Suite**: Fast execution tests
- **Main Orchestrator**: Main orchestrator functionality
- **Coverage**: Code coverage tests
- **Performance**: Performance and benchmarking
- **Unit**: Basic unit tests

## Test Execution Features

### Resource Monitoring

- Memory usage tracking
- CPU usage monitoring
- Timeout handling
- Resource limits

### Parallel Execution

- Category-based parallel execution
- Configurable parallelization
- Resource-aware scheduling

### Error Handling

- Graceful failure handling
- Error reporting and logging
- Recovery mechanisms
- Safe-to-fail test execution

### Reporting

- Comprehensive test reports
- Performance metrics
- Coverage analysis
- Error summaries

## Test Organization Patterns

### Naming Conventions

Test files follow a consistent naming pattern:

- `test_MODULENAME_overall.py` - Comprehensive module tests (required for each module)
- `test_MODULENAME_area.py` - Specific area tests (e.g., `test_gnn_parsing.py`, `test_gnn_validation.py`)
- `test_MODULENAME_integration.py` - Integration tests for the module
- `test_MODULENAME_performance.py` - Performance tests for the module

### Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.fast  # Quick tests (< 1 second)
def test_quick_functionality():
    pass


@pytest.mark.slow  # Slow tests (> 10 seconds)
def test_complex_scenario():
    pass


@pytest.mark.integration  # Integration tests
def test_module_integration():
    pass


@pytest.mark.performance  # Performance/resource-usage tests (auto-tagged on slow tests)
def test_performance_scenario():
    pass
```

### Adding New Test Categories

To add a new test category to `MODULAR_TEST_CATEGORIES` in `categories.py`:

```python
MODULAR_TEST_CATEGORIES["new_module"] = {
    "name": "New Module Tests",
    "description": "Tests for the new module functionality",
    "files": ["test_new_module_overall.py", "test_new_module_integration.py"],
    "markers": ["new_module"],  # Optional pytest markers to filter
    "timeout_seconds": 120,  # Maximum execution time
    "max_failures": 8,  # Stop after N failures
    "parallel": True,  # Allow parallel execution
}
```

### Creating New Test Files

Example test file structure:

```python
# tests/new_module/test_new_module_overall.py
"""Comprehensive tests for the new module."""

import pytest
from pathlib import Path
from gnn.utils.testing import create_sample_gnn_content, assert_file_exists


@pytest.mark.fast
def test_new_module_basic():
    """Test basic functionality."""
    # Test implementation using real methods
    result = process_module(data)
    assert result is not None


@pytest.mark.slow
def test_new_module_complex():
    """Test complex scenarios."""
    # Test implementation
    pass


@pytest.mark.integration
def test_new_module_integration():
    """Test integration with other modules."""
    # Test implementation
    pass
```

## Best Practices

### Test Organization

1. **Module-Based Structure**: Each module has its own test files
2. **Comprehensive Coverage**: Each module has an `_overall.py` test file
3. **Specialized Testing**: Additional test files for specific areas
4. **Integration Testing**: Cross-module integration tests

### Test Writing

1. **Real Implementations**: Execute real methods and code paths.
2. **Import Error Handling**: Wrap imports in try/except blocks; skip if optional deps missing.
3. **Comprehensive Assertions**: Test both success and failure cases against real artifacts.
4. **Performance Monitoring**: Use performance tracking for slow operations

### Test Execution

1. **Category-Based**: Run tests by module category
2. **Parallel Execution**: Use parallel execution for faster results
3. **Resource Monitoring**: Monitor resource usage during execution
4. **Error Recovery**: Handle errors gracefully with recovery mechanisms

## Troubleshooting

### Common Issues and Solutions

#### Issue: Tests Timeout

**Symptoms**: Tests fail with timeout errors
**Solutions**:

- Increase timeout: `export FAST_TESTS_TIMEOUT=900` (15 minutes)
- Run fast tests only: `python src/gnn/2_tests.py --fast-only`
- Skip tests in pipeline: `export SKIP_TESTS_IN_PIPELINE=1`

#### Issue: Collection Errors

**Symptoms**: `ERROR collecting` messages, import errors
**Solutions**:

- Check for missing dependencies: `uv sync` for runtime dependencies or `uv sync --extra dev` for pytest and developer tools
- Verify Python path includes `src/` directory
- Check for syntax errors in test files
- Review error messages for specific import failures

#### Issue: Tests Fail to Run

**Symptoms**: No tests collected, exit code 5
**Solutions**:

- Verify test files follow naming convention: `test_*.py`
- Check that test functions are named with `test_` prefix
- Ensure test files are in the `tests/` directory
- Check pytest through the dev extra: `uv run --extra dev python -m pytest --version`

#### Issue: Memory Errors

**Symptoms**: Out of memory errors during test execution
**Solutions**:

- Run tests sequentially instead of parallel
- Reduce number of tests: use `--fast-only` flag
- Increase system memory or use swap space
- Check for memory leaks in test code

#### Issue: Slow Test Execution

**Symptoms**: Tests take too long to complete
**Solutions**:

- Use `--fast-only` flag to skip slow tests
- Mark slow tests with `@pytest.mark.slow` and exclude: `pytest -m "not slow"`
- Run specific test categories instead of all tests
- Use parallel execution (if not already enabled)

#### Issue: Import Errors in Tests

**Symptoms**: `ImportError` or `ModuleNotFoundError` in test files
**Solutions**:

- Ensure `src/` is in Python path
- Check that modules are properly installed
- Verify relative imports are correct
- Use `sys.path.insert(0, str(SRC_DIR))` if needed

### Getting Help

If issues persist:

1. Check test output files in `output/2_tests_output/`
2. Review `pytest_stdout.txt` / `pytest_stderr.txt` for detailed runner output
3. Check `test_execution_report.json` for execution summary
4. Verify environment variables are set correctly
5. Ensure all dependencies are installed

## Current Status

### Test Coverage

- **Live inventory** is derived from `rg --files` and pytest collection rather
  than duplicated here
- **Latest full-suite evidence** is maintained in the root
  [`README.md`](../../README.md)
- **Comprehensive module coverage** for all major modules
- **Specialized test areas** for specific functionality
- **Integration tests** for cross-module functionality
- **Performance tests** for benchmarking and regression detection
- **Error recovery tests** for resilience validation

### Test Execution Statistics

- **Fast Tests**: ~1-3 minutes (default for pipeline)
- **Comprehensive Tests**: runtime depends on optional backends and selected markers
- **Test Categories**: managed by `tests/categories.py`
- **Test Markers**: registered in `pytest.ini` and `pyproject.toml`
- **Latest Execution**: use current full-suite output for pass, skip, and duration counts

### Test Quality Metrics

- **No Simulated Usage**: All tests use real implementations per testing policy
- **Real Data**: All tests use real, representative data
- **Error Handling**: Comprehensive error scenario testing
- **Documentation**: Complete test documentation in AGENTS.md and README.md

### Test Infrastructure

- **Modular test runner** with category-based execution
- **Resource monitoring** and timeout handling
- **Parallel execution** support
- **Comprehensive reporting** and error handling

### Module Coverage

Module coverage mirrors the maintained source tree. Use `rg --files tests -g 'test_*.py'` for the current file inventory and `uv run --extra dev python -m pytest tests/ --collect-only -q` for the current collected-test count.

## Test Execution Results

Run the command below for the current result. The root
[`README.md`](../../README.md) is the authority for the latest dated receipt.

```bash
uv run --extra dev python -m pytest tests/ -q --tb=no -rsx --ignore=tests/llm/test_llm_ollama.py --ignore=tests/llm/test_llm_ollama_integration.py
```

## Future Enhancements

### Planned Improvements

1. **Coverage Analysis**: Enhanced code coverage tracking and reporting
2. **Performance Benchmarking**: Automated performance regression detection
3. **CI/CD Integration**: Automated test execution in CI/CD pipelines
4. **Test Data Management**: Centralized test data fixtures
5. **Visual Test Reports**: Enhanced HTML reporting with visualizations

### Module Expansion

- Additional tests for new modules as they're added
- Enhanced integration tests for cross-module workflows
- Expanded performance tests for scalability validation
- Additional error recovery scenarios

This test infrastructure provides a solid foundation for comprehensive testing of the GNN Processing Pipeline, with modular organization, parallel execution, and comprehensive coverage of all major components.

## References

- Project overview: ../../README.md
- Comprehensive docs: ../../DOCS.md
- Architecture guide: ../../ARCHITECTURE.md
- Pipeline details: ../../docs/pipeline/README.md


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
