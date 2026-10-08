# Step 1: Setup

## Architectural Mapping

**Orchestrator**: `src/gnn/1_setup.py` (116 lines)
**Implementation Layer**: `src/gnn/setup/`

## Module Description

This module provides comprehensive environment setup and dependency management capabilities for the GNN pipeline using **UV** (modern Python package manager), including UV environment management, package installation, and system configuration.


```
src/gnn/setup/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
├── constants.py                   # Shared constants
├── dependency_setup.py            # JAX test install, Julia env, optional groups, setup_complete_environment
├── mcp.py                         # Model Context Protocol integration
├── package_names.py               # Package name mapping
├── setup.py                       # perform_full_setup entry point
├── utils.py                       # Setup utilities
├── uv_management.py               # UV environment: setup_uv_environment, install_uv_dependencies, check_system_requirements
├── uv_package_ops.py              # add/remove/update/lock UV dependencies
└── validator.py                   # validate_system, environment and UV status
```


```mermaid
graph TD
    Pipeline[1_setup.py] --> Orchestrator[setup_orchestrator]
    Orchestrator --> EnvSetup[setup_uv_environment]
    Orchestrator --> SysCheck[check_system_requirements]
    EnvSetup --> UVSync[uv sync]
```

## Agent Identity & Capabilities

# Setup Module - Agent Scaffolding

## Module Overview

**Purpose**: Environment setup, dependency management, and system configuration for the GNN processing pipeline

**Pipeline Step**: Step 1: Environment setup (1_setup.py)

**Category**: Environment Management / Dependency Installation

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-01-21

---

## Core Functionality

### Primary Responsibilities
1. Virtual environment creation and management
2. Dependency installation and validation via native UV commands
3. System requirement verification
4. UV (Python package manager) integration
5. Environment configuration and optimization

### Key Capabilities
- Automated virtual environment setup using UV
- Comprehensive dependency management via `uv sync`
- System requirement validation
- UV environment optimization
- Dependency conflict resolution via `uv.lock`
- Environment health monitoring
- Native UV dependency operations (`add`, `remove`, `sync`, `lock`)

---

## API Reference

### Public Functions

#### `setup_orchestrator(target_dir, output_dir, logger, **kwargs) -> bool`
**Description**: Main setup orchestrator function called by orchestrator (1_setup.py). This function coordinates setup operations based on provided arguments.

**Parameters**:
- `target_dir` (Path): Target directory (used for context)
- `output_dir` (Path): Output directory for setup logs
- `logger` (Logger): Logger instance
- `**kwargs`: Additional setup options:
  - `verbose` (bool): Enable verbose output
  - `recreate_venv` (bool): Recreate virtual environment
  - `dev` (bool): If true, `uv sync --extra dev` (pytest stack, linters, etc.)
  - `install_all_extras` (bool): If true, `uv sync --all-extras` (every optional group in pyproject)
  - `setup_core_only` (bool): If true, skips the post-install JAX/Optax/Flax/pymdp functional probe (`gnn.utils.runtime_safety.jax_stack_validation`; Step 12 backends remain core deps)
  - `install_optional` (bool): Install optional dependencies
  - `optional_groups` (str): Comma-separated list of optional groups

**Returns**: `True` if setup succeeded

**Note**: Default path runs `uv sync` for core dependencies, which include Step 12 core backends (JAX, NumPyro, DisCoPy), LLM clients, and interactive visualization (pandas, plotly, seaborn, h5py). PyTorch (via the resolved `torch>=2.13.0` extra) and bnlearn (`uv sync --extra bnlearn`) remain supported optional installs. `SETUP_DEFAULT_PIPELINE_EXTRAS` is empty by default. Step 22 (GUI) needs Gradio: `uv sync --extra gui` (otherwise the GUI step runs headless and emits recovery artifacts only).

#### `setup_uv_environment(verbose=False, recreate=False, dev=False, extras=None, install_all_extras=False, skip_jax_test=False, output_dir=None) -> bool`
**Description**: Set up UV virtual environment with dependencies using native UV sync

**Parameters**:
- `verbose`: Enable verbose output
- `recreate`: Recreate existing environment
- `dev`: Install development optional group (`--extra dev`)
- `extras`: Additional package groups to install (each as `--extra`)
- `install_all_extras`: If true, `uv sync --all-extras` (takes precedence over `dev`)
- `skip_jax_test`: Skip the JAX stack probe (same probe as ``gnn.utils.runtime_safety.jax_stack_validation.verify_jax_pymdp_stack``)
- `output_dir`: Output directory for setup logs

**Returns**: `True` if setup succeeded

#### `setup_complete_environment(verbose=False, recreate=False, install_optional=False, optional_groups=None, output_dir=None) -> bool`
**Description**: Complete environment setup with optional dependencies (system requirements check, UV environment creation, core dependency install, then optional dependency groups). `setup_orchestrator` calls this instead of `setup_uv_environment` whenever `install_optional` or `optional_groups` is requested.

**Parameters**:
- `verbose`: Enable verbose logging
- `recreate`: Recreate environment if it exists
- `install_optional`: Install optional dependencies
- `optional_groups`: Specific optional groups to install (`None` installs all groups)
- `output_dir`: Output directory for setup logs

**Returns**: `True` if setup succeeded

#### `install_uv_dependencies(verbose=False, dev=False, extras=None, install_all_extras=False) -> bool`
**Description**: Install UV dependencies using `uv sync` from pyproject.toml

**Parameters**:
- `verbose`: Enable verbose output
- `dev`: If true and `install_all_extras` is false, append `--extra dev`
- `extras`: Additional package groups (`--extra` each)
- `install_all_extras`: If true, append `--all-extras` (ignores `dev` for sync flags)

**Returns**: `True` if installation succeeded

#### `add_uv_dependency(package: str, dev: bool = False, verbose: bool = False) -> bool`
**Description**: Add a dependency using `uv add` command

**Parameters**:
- `package`: Package name with optional version specifier
- `dev`: Add as development dependency
- `verbose`: Enable verbose logging

**Returns**: `True` if successful

#### `remove_uv_dependency(package: str, verbose: bool = False) -> bool`
**Description**: Remove a dependency using `uv remove` command

**Parameters**:
- `package`: Package name to remove
- `verbose`: Enable verbose logging

**Returns**: `True` if successful

#### `update_uv_dependencies(verbose: bool = False, upgrade: bool = False) -> bool`
**Description**: Update dependencies using `uv sync` command

**Parameters**:
- `verbose`: Enable verbose logging
- `upgrade`: Upgrade dependencies to latest compatible versions

**Returns**: `True` if successful

#### `lock_uv_dependencies(verbose: bool = False) -> bool`
**Description**: Update lock file using `uv lock` command

**Parameters**:
- `verbose`: Enable verbose logging

**Returns**: `True` if successful

#### `check_system_requirements(verbose=False) -> bool`
**Description**: Check system requirements for GNN pipeline

**Parameters**:
- `verbose`: Enable verbose output

**Returns**: `True` if requirements are met

---

## Dependencies

### Required Dependencies
- `uv` - Python package manager (required, native commands used)
- `python` - Python interpreter (>=3.11, <3.15 per `pyproject.toml`; native acceptance depends on the selected environment)
- `pyproject.toml` - Project dependencies configuration

### Optional Dependencies
- None (UV handles all dependency management)

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

---

## Configuration

### Environment Settings
```python
UV_CONFIG = {
    "python_version": "3.11",
    "environment_name": "gnn-pipeline",
    "dependency_source": "pyproject.toml",  # Primary source
    "lock_file": "uv.lock",  # Dependency lock file
    "use_native_uv": True,  # Use native UV commands
    "dev_dependencies": True,
    "test_dependencies": True,
}
```

### System Requirements
```python
SYSTEM_REQUIREMENTS = {
    "python_version_min": "3.11",
    "memory_min_gb": 4,
    "disk_space_min_gb": 2,
    "cpu_cores_min": 2,
    "uv_required": True,
}
```

---

## Usage Examples

### Basic Environment Setup
```python
from setup import setup_uv_environment

success = setup_uv_environment(verbose=True, dev=True, extras=["audio", "gui", "ml-ai"])
```

### Add New Dependency
```python
from setup import add_uv_dependency

# Add production dependency
success = add_uv_dependency("requests>=2.28.0", dev=False, verbose=True)

# Add development dependency
success = add_uv_dependency("pytest>=7.0.0", dev=True, verbose=True)
```

### Remove Dependency
```python
from setup import remove_uv_dependency

success = remove_uv_dependency("old-package", verbose=True)
```

### Update Dependencies
```python
from setup import update_uv_dependencies

# Sync with lock file
success = update_uv_dependencies(verbose=True, upgrade=False)

# Upgrade to latest compatible versions
success = update_uv_dependencies(verbose=True, upgrade=True)
```

### Lock Dependencies
```python
from setup import lock_uv_dependencies

# Update uv.lock file
success = lock_uv_dependencies(verbose=True)
```

### System Requirements Check
```python
from setup import check_system_requirements

requirements_met = check_system_requirements(verbose=True)
if requirements_met:
    print("System requirements satisfied")
else:
    print("System requirements not met")
```

---

## Output Specification

### Output Products
- `setup_summary.json` - Setup completion summary
- `environment_info.json` - Environment information
- `dependency_status.json` - Dependency installation status
- `setup_log.txt` - Detailed setup log
- `uv.lock` - Dependency lock file (updated)

### Output Directory Structure
```
output/1_setup_output/
├── setup_summary.json
├── environment_info.json
├── dependency_status.json
├── setup_log.txt
└── environment_details/
    ├── python_version.txt
    └── package_list.txt
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~2-5 minutes for full setup
- **Memory**: ~50-100MB during installation
- **Status**: ✅ Production Ready

### Expected Performance
- **Environment Creation**: 30-60 seconds
- **Dependency Installation**: 1-3 minutes (via `uv sync`)
- **System Validation**: < 30 seconds
- **Health Check**: < 10 seconds
- **Dependency Lock**: < 10 seconds

---

## Error Handling

### Setup Errors
1. **Environment Creation**: Virtual environment creation failures
2. **Dependency Installation**: Package installation errors via `uv sync`
3. **System Requirements**: Insufficient system resources or missing UV
4. **Network Issues**: Package download failures
5. **Permission Errors**: Insufficient file permissions
6. **Lock File Conflicts**: Lock file corruption or conflicts

### Recovery Strategies
- **Retry Logic**: Automatic retry for transient failures
- **Lock File Regeneration**: Regenerate uv.lock if corrupted
- **Graceful Degradation**: Continue with available packages
- **Manual Instructions**: Provide manual installation guidance
- **Environment Recreate**: Option to recreate environment from scratch

---

## Integration Points

### Orchestrated By
- **Script**: `1_setup.py` (Step 1)
- **Function**: `setup_uv_environment()`

### Imports From
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

### Imported By
- `main.py` - Pipeline orchestration
- `tests.test_setup_*` - Setup tests

### Data Flow
```
System Check → UV Environment Creation → UV Sync (pyproject.toml → uv.lock) → Validation → Health Report
```

---

## Testing

### Test Files
- `tests/setup/test_setup_overall.py` - Module-level setup tests
- `tests/test_uv_environment.py` - UV environment behavior tests
- `tests/test_environment_overall.py` - Environment-related integration checks

### Test Coverage
- Measure: `uv run --extra dev python -m pytest tests/setup/ --cov=gnn.setup --cov-report=term-missing` (do not treat fixed percentages in this doc as canonical).

### Key Test Scenarios
1. Environment creation and setup
2. Dependency installation via UV sync
3. System requirement verification
4. Native UV command operations
5. Error handling and recovery

---

## MCP Integration

### Tools Registered
- `ensure_directory_exists` - Create a directory if missing
- `find_project_gnn_files` - Locate GNN files in the project
- `get_standard_output_paths` - Standard output paths for the pipeline
- `check_uv_project_status` - Check UV project status
- `get_uv_environment_info` - UV environment information
- `setup_uv_project_structure` - Create the UV project structure
- `install_uv_dependency` - Add a dependency via UV
- `sync_uv_dependencies` - Sync dependencies via UV

### Tool Endpoints
```python
@mcp_tool("check_uv_project_status")
def check_environment_tool():
    """Check system environment for GNN pipeline"""
    # Implementation


@mcp_tool("setup.add_dependency")
def add_dependency_tool(package: str, dev: bool = False):
    """Add a dependency using UV add"""
    # Implementation
```

---

---
## Documentation
- **[README](../../../src/gnn/setup/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/setup/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/setup/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/setup/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/setup](../../../src/gnn/setup)
