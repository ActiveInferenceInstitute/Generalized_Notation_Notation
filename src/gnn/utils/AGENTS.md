# Utils Module - Agent Scaffolding

## Module Overview

**Purpose**: Shared utilities and helper functions for the GNN processing pipeline

**Pipeline Step**: Infrastructure module (not a numbered step)

**Category**: Utility Functions / Infrastructure Support

**Status**: Production Ready

**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-24

---

## Core Functionality

### Primary Responsibilities
1. Pipeline orchestration and coordination utilities
2. Logging and diagnostic utilities
3. Configuration and argument parsing utilities
4. Resource management and monitoring utilities
5. Error handling and recovery utilities
6. Performance tracking and optimization utilities

### Key Capabilities
- Centralized logging and diagnostic system
- Argument parsing and configuration management
- Resource monitoring and performance tracking
- Error handling and recovery mechanisms
- Pipeline orchestration and coordination
- Utility functions for common operations
- `runtime_safety/jax_stack_validation.py`: single probe for JAX, Optax, Flax, and pymdp 1.x (Step 1 setup, `validate_uv_setup`, CI, pytest)

---

## API Reference

### Logging Functions

#### `setup_step_logging(step_name: str, verbose: bool = False) -> logging.Logger`
**Description**: Set up standardized logging for a pipeline step with correlation ID tracking

**Parameters**:
- `step_name` (str): Name of the pipeline step (e.g., "3_gnn")
- `verbose` (bool): Enable verbose logging (default: False)

**Returns**: `logging.Logger` - Configured logger instance with correlation ID

**Example**:
```python
from gnn.utils import setup_step_logging

logger = setup_step_logging("3_gnn", verbose=True)
```

#### `setup_main_logging(log_dir: Optional[Path] = None, verbose: bool = False, log_format: str = "human") -> logging.Logger`
**Description**: Set up logging for the main pipeline orchestrator. `log_dir` is the directory for log files; `log_format` selects console output style (`"human"` or `"json"`). Initializes `PipelineLogger`, sets verbosity, establishes the `"main"` correlation context, and returns the configured `GNN_Pipeline` logger.

**Location**: `src/gnn/utils/logging/logging_utils.py` (re-exported by the `utils/logging_utils.py` facade).

#### `log_step_start(logger, message)` / `log_step_success(logger, message)` / `log_step_error(logger, message)` / `log_step_warning(logger, message)`
**Description**: Step lifecycle logging helpers (`utils/logging_utils.py`). `log_step_start` returns a correlation-aware context; the others log structured lifecycle events. `observability/structured_logging.py` additionally exposes `log_step_start(logger, step_name, **context)` and friends with richer metadata when a step needs it.

#### `get_performance_summary() -> Dict[str, Any]`
**Description**: Get summary of performance metrics across all tracked operations

**Returns**: `Dict[str, Any]` - Performance summary with timing, memory, and resource usage

#### `setup_correlation_context(correlation_id: Optional[str] = None, step_name: Optional[str] = None)`
**Description**: Set up correlation context for logging (delegates to `observability/structured_logging.py`)

**Parameters**:
- `correlation_id` (Optional[str]): Existing correlation ID or None to generate new
- `step_name` (Optional[str]): Name of the pipeline step

### Argument Parsing Functions

> **Module layout**: argument handling lives in single-responsibility modules under
> `arguments/` — `arg_definitions` (shared `STEP_ARGUMENTS`/constants),
> `arg_parsing` (the `ArgumentParser`), `path_conversion` (`validate_and_convert_paths`),
> `pipeline_arguments` (command building), and `step_config` (`StepConfiguration`).
> `gnn.utils.runtime_safety.safe_eval.safe_literal_eval` provides bounded,
> DoS-resistant literal evaluation for untrusted GNN parameter strings (RED_TEAM V-03).

#### `ArgumentParser.parse_step_arguments(step_name: str) -> argparse.Namespace`
**Description**: Parse arguments for a specific pipeline step with recovery support

**Parameters**:
- `step_name` (str): Name of the pipeline step

**Returns**: `argparse.Namespace` - Parsed arguments with standard pipeline options

**Standard Arguments**:
- `--target-dir`: Target directory for input files
- `--output-dir`: Output directory for results
- `--verbose`: Enable verbose logging
- `--recursive`: Recursively process directories

#### `build_step_command_args(step_name: str, pipeline_args: PipelineArguments, python_executable: str, script_path: Path) -> List[str]`
**Description**: Build validated command-line arguments for a pipeline step

**Parameters**:
- `step_name` (str): Name of the pipeline step (e.g., "1_gnn")
- `pipeline_args` (PipelineArguments): Main pipeline arguments
- `python_executable` (str): Path to the Python executable
- `script_path` (Path): Path to the step script

**Raises**: `ValueError` if the step configuration is invalid

**Returns**: `List[str]` - Command-line argument list

#### `gnn.utils.arguments.arg_parsing.audit_step_contracts(python_executable: Optional[str] = None, script_dir: Optional[Path] = None) -> List[Dict[str, Any]]`
**Description**: Audit registered step contracts for drift between `STEP_ARGUMENTS`, `StepConfiguration`, parser defaults, and command-builder propagation.

**Returns**: `List[Dict[str, Any]]` - Contract audit issues; each entry describes a per-step mismatch (empty list means no drift)

**Contract**: `StepConfiguration` is the shared source for step defaults and critical-step metadata. Exit codes are canonical across numbered scripts and the main orchestrator: `0=success`, `1=error`, `2=success with warnings/skipped`.

#### `validate_and_convert_paths(args, logger) -> argparse.Namespace`
**Description**: Validate and convert string paths to Path objects

**Parameters**:
- `args`: Parsed pipeline arguments
- `logger` (logging.Logger): Logger for validation messages

### Pipeline Utilities

The pipeline utility functions live in their canonical home, `gnn.pipeline`
(not in this facade):

#### `get_output_dir_for_script(script_name: str, base_output_dir: Optional[Path] = None) -> Path`
**Location**: `gnn.pipeline.config` (re-exported by `gnn.pipeline`)
**Description**: Get standardized output directory for a pipeline script

**Parameters**:
- `script_name` (str): Name of the script (e.g., "3_gnn.py")
- `base_output_dir` (Optional[Path]): Base output directory (default: Path("output"))

**Returns**: `Path` - Output directory path (e.g., "output/3_gnn_output/")

### Resource Management Functions

#### `get_current_memory_usage() -> float`
**Description**: Get current process memory usage

**Returns**: `float` - Memory usage in megabytes (MB)

#### `get_memory_usage() -> float`
**Description**: Canonical MB-scale process-memory probe (alias of `get_current_memory_usage`).

### Error Recovery Functions

#### `ErrorRecoveryManager(logger=None).handle_error(context: ErrorContext) -> bool`
**Description**: Handle an error through the registered recovery strategies (`errors/error_recovery.py`). Errors are constructed as `ErrorContext` objects (operation, severity, message, error_code, details).

#### `format_and_log_error(error_code: str, operation: str, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, details: Optional[Dict[str, Any]] = None, suggestions: Optional[List[str]] = None, exception: Optional[Exception] = None) -> ErrorContext`
**Description**: Build an `ErrorContext`, run it through the shared `ErrorRecoveryManager`, and return the context (`src/gnn/utils/errors/error_recovery.py`).

### Configuration Functions

#### `load_config(config_path: Optional[Path] = None) -> GNNPipelineConfig`
**Description**: Load pipeline configuration (defaults when no path given)

#### `get_config_value(config: dict, key: str) -> Any` / `set_config_value(config: dict, key: str, value: Any) -> Any`
**Description**: Get/set a configuration value by key (dot-notation supported) on the passed config dict

### Dependency Management Functions

#### `validate_pipeline_dependencies(step_names: Optional[List[str]] = None, logger=None, python_path=None) -> bool`
**Description**: Validate dependencies for specific pipeline steps

#### `check_optional_dependencies() -> dict`
**Description**: Return a status summary of all optional dependencies (`{'optional_dependencies': {...}, 'missing_optional': [...]}`)

#### `install_missing_dependencies() -> dict`
**Description**: Attempt to install missing Python dependencies via pip; returns `{'installed': [...], 'failed': [...], 'skipped': [...]}`

### Performance Tracking Functions

#### `PerformanceTracker.track_operation(operation: str, metadata: Optional[Dict[str, Any]] = None)`
**Description**: Context manager tracking operation timing; usage: `with tracker.track_operation("name", {...}):`

#### `track_operation_standalone(operation: str, metadata: Optional[Dict[str, Any]] = None) -> ContextManager[None]`
**Description**: `@contextmanager` that measures the duration of a `with` block and records it on the global `performance_tracker` via `record_timing(operation, duration, metadata)`. Usage: `with track_operation_standalone("name", {...}):`. Yields `None` (`src/gnn/utils/observability/performance_tracking.py`).


## Composability Notes

### Shared single-source helpers

Each helper below is the only implementation of its concern; import it from the path shown:

- **Writable-directory probe**: `gnn.utils.config_io.io_utils.verify_directory_writable(directory, probe_name=".write_probe") -> None` is the single create-rename-cleanup probe. `gnn.utils.pipeline_orchestration.pipeline_validator.check_pipeline_readiness` calls it and keeps its own error messaging.
- **Canonical memory probe**: `gnn.utils.runtime_safety.resource_manager.get_memory_usage` (alias of `get_current_memory_usage`).
- **Step-argument fallback defaults**: `gnn.utils.arguments.arg_parsing.fallback_default_for(arg_name)` backed by the `_FALLBACK_DEFAULTS` mapping supplies the defaults for `ArgumentParser.parse_step_arguments` and `ArgumentParser.create_default_namespace`; `create_default_namespace` matches the registered contract, including `advanced_stats` (`False`) and `simulation_params` (`"{}"`).
- **Injectable project root**: `StepConfiguration.validate_step_args(step_name, args, project_root=None)` accepts an explicit project root for missing-input-path repair; when omitted, the caller-frame heuristic applies.
- **`with_resource_limits`**: exceptions raised by the wrapped body always propagate; limit violations are only raised when the body completed normally, so a limit error never masks a body failure.
- **Environment redaction**: `gnn.utils.mcp.SENSITIVE_ENV_KEY_MARKERS`, `is_sensitive_env_key(key) -> bool`, and `redact_environment() -> dict[str, str]` centralize secret filtering used by `get_environment_info` (markers include `credential`, `passwd`, `auth`).
- **Monitor alert bands**: `PipelineMonitor.health_thresholds["duration_variance"]` defines `"critical": 3.0` (above 3x baseline); the degraded-band warning fires between 2x and 3x baseline.

---

## Dependencies

### Required Dependencies
- `pathlib` - Path manipulation
- `logging` - Logging functionality
- `argparse` - Argument parsing
- `typing` - Type hints

### Optional Dependencies
- `psutil` - System resource monitoring
- `numpy` - Numerical computations

### Internal Dependencies
- None (base infrastructure module)

---

## Configuration

### Logging and Performance
No module-level config dicts. Verbosity is set through the `verbose` flag on `setup_step_logging` / `setup_main_logging`; performance tracking is invoked programmatically via `PerformanceTracker`.

---

## Usage Examples

### Step Logging Setup
```python
from gnn.utils.logging_utils import setup_step_logging

logger = setup_step_logging("3_gnn.py", verbose=True)
logger.info("Starting GNN processing")
```

### Output Directory Management
```python
from gnn.pipeline import get_output_dir_for_script

output_dir = get_output_dir_for_script("3_gnn.py", Path("output"))
print(f"GNN output directory: {output_dir}")
```

### Pipeline Script Creation
```python
from gnn.utils.pipeline_orchestration.pipeline_template import create_standardized_pipeline_script

run_script = create_standardized_pipeline_script(
    "3_gnn.py", process_gnn_files, "GNN file processing"
)

# Execute the script
exit_code = run_script()
```

### Memory Monitoring
```python
from gnn.utils.runtime_safety.resource_manager import get_current_memory_usage

memory_before = get_current_memory_usage()
# ... do some work ...
memory_after = get_current_memory_usage()
print(f"Memory delta: {memory_after - memory_before} MB")
```

---

## Output Specification

### Output Products
- Console and (when configured) file logs emitted by the structured logging facade
- In-memory performance metrics retrievable via `get_performance_summary()`

There is no fixed `output/logs/` or `output/performance/` directory tree owned by this module; log destinations follow the logging setup for each entry point.

---

## Performance Characteristics

### Expected Performance
- **Logging**: < 1ms per log entry
- **Path Operations**: < 1ms per operation
- **Memory Monitoring**: < 5ms per check
- **Configuration**: < 10ms per operation


## Error Handling

### Utility Errors
1. **Configuration Errors**: Invalid configuration parameters
2. **Path Errors**: Invalid or inaccessible paths
3. **Logging Errors**: Logging system failures
4. **Resource Errors**: Resource monitoring failures

### Recovery Strategies
- **Configuration Repair**: Use default values
- **Path Resolution**: Resolve relative paths
- **Logging Recovery**: Use basic logging
- **Resource Monitoring**: Continue without monitoring

---

## Integration Points

### Orchestrated By
- All pipeline scripts and modules

### Imports From
- None (base infrastructure module)

### Imported By
- All pipeline scripts (0_template.py through 24_intelligent_analysis.py)
- All pipeline modules

### Data Flow
```
Configuration → Logging Setup → Resource Monitoring → Error Handling → Performance Tracking
```

---

## Testing

### Test Files
- `tests/utils/test_utils_core.py` - Core utils tests
- `tests/utils/test_new_utils.py` - Additional utils tests
- `tests/utils/test_shared_helpers.py` - Shared-helper behavior tests (probe, memory aliases, fallback defaults, redaction, alert bands)

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest tests/utils/ \
    --cov=src/gnn/utils --cov-report=term-missing
```
### Key Test Scenarios
1. Logging and diagnostic utilities
2. Configuration and argument parsing
3. Resource management and monitoring
4. Error handling and recovery

---

## MCP Integration

### Tools Registered
Registered by the `gnn.utils.mcp` package (`src/gnn/utils/mcp/`):
- `get_system_info` - Platform, Python, and memory information
- `get_environment_info` - Environment and dependency overview
- `get_logging_info` - Logging configuration
- `validate_dependencies` - Dependency validation

There is no `utils.get_performance_metrics` tool.

---

## Troubleshooting

### Common Issues

#### Issue 1: Logging not working
**Symptom**: No log output or logs in wrong location  
**Cause**: Logging configuration incorrect or permissions issues  
**Solution**: 
- Verify log directory exists and is writable
- Check logging level configuration
- Use `--verbose` flag for detailed logging
- Review logging configuration in pipeline config

#### Issue 2: Argument parsing errors
**Symptom**: Script fails with argument parsing errors  
**Cause**: Argument definition mismatch or missing required arguments  
**Solution**:
- Verify argument definitions match script usage
- Check required arguments are provided
- Review argument parser configuration
- Use `--help` flag to see expected arguments

---

## Version History

### Current Version: [pyproject.toml](../../../pyproject.toml) (canonical)

**Features**:
- Centralized logging system
- Argument parsing utilities
- Resource monitoring
- Performance tracking
- Error handling utilities

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced performance monitoring
- **Future**: Real-time resource tracking

---

## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [Pipeline Module](../pipeline/AGENTS.md)

### External Resources
- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)

---

**Last Updated**: 2026-09-24
**Maintainer**: GNN Pipeline Team
**Status**: Production Ready
**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API

Step 7 accepts `--formats` and `--geo-infer-options-file`; argument definitions,
step configuration and `PipelineArguments` preserve these through orchestration.

[`file_workers.py`](file_workers.py) owns bounded ordered file jobs. Explicit
counts select spawned processes unless threads are explicitly requested.
Workers compute and return data; parent callers own publication and selection
identity. Step subprocess supervision remains authoritative for pipeline cleanup.
