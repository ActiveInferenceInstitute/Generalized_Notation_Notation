# Setup Module - Agent Scaffolding

## Module Overview

**Purpose**: Environment setup, dependency management, and system configuration for the GNN processing pipeline

**Pipeline Step**: Step 1: Environment setup (1_setup.py)

**Category**: Environment Management / Dependency Installation

**Status**: Production Ready

**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-24

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

### Orchestrator Entry Point

The thin orchestrator `src/gnn/1_setup.py` defines:

```python
setup_orchestrator(target_dir, output_dir, logger, **kwargs) -> bool
```

It wraps `setup_uv_environment` (default path) or `setup_complete_environment` (when
optional groups are requested). `**kwargs` accepts:

- `verbose` (bool) — verbose logging
- `recreate_venv` (bool) — recreate `.venv`
- `dev` (bool) — `uv sync --extra dev`
- `install_all_extras` (bool) — `uv sync --all-extras` (supersedes `dev`)
- `setup_core_only` (bool) — skip the JAX/Optax/Flax/pymdp probe
- `install_optional` (bool) — install optional groups
- `optional_groups` (str) — comma-separated group names (see `OPTIONAL_GROUPS`)

Default path: `uv sync` for core deps. Step 12 core backends (JAX, NumPyro,
DisCoPy), interactive visualization (pandas, plotly, seaborn, h5py), and LLM
clients are in `[project.dependencies]` and therefore installed without any
`--extra`. PyTorch and bnlearn are still supported by render/execute code paths,
through the declared `torch` and `bnlearn` extras. Patched Torch is resolved
in the frozen lock. `SETUP_DEFAULT_PIPELINE_EXTRAS` is empty by default because core
dependencies already cover the default pipeline runtime.
Step 22 (GUI) additionally needs `uv sync --extra gui` to pull Gradio.

### Module-level Public Functions

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
- `python` - Python interpreter (>=3.12)
- `pyproject.toml` - Project dependencies configuration

### Optional Dependencies
- None (UV handles all dependency management)

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

---

## Configuration

### Environment Settings

Defined in `setup/constants.py`:

```python
VENV_DIR = ".venv"                 # Managed virtual environment location
MIN_PYTHON_VERSION = (3, 12)       # Shared floor: setup, preflight, execute validator
SETUP_DEFAULT_PIPELINE_EXTRAS: tuple[str, ...] = ()  # No extras needed by default
```

### Optional Groups

`OPTIONAL_GROUPS` in `constants.py` derives installable extras from installed
`Provides-Extra` metadata, with a source-manifest fallback. Consult
`pyproject.toml` for the complete registry, including `torch`, `bnlearn`,
`stan`, `cpomdp`, `thrml`, `ngclearn` and `geo-infer`. The default PEP 735
`dev` tooling group and the CmdStanPy `dev` extra are separate surfaces.

---

## Usage Examples

All setup helpers are exported from the package root. Prefer `from gnn.setup import …`.

```python
from gnn.setup import (
    setup_uv_environment,
    add_uv_dependency,
    remove_uv_dependency,
    update_uv_dependencies,
    lock_uv_dependencies,
    check_system_requirements,
)

setup_uv_environment(verbose=True, dev=True, extras=["audio", "gui", "ml-ai"])

add_uv_dependency("requests>=2.28.0", dev=False, verbose=True)
add_uv_dependency("pytest>=7.0.0", dev=True, verbose=True)
remove_uv_dependency("old-package", verbose=True)
update_uv_dependencies(verbose=True, upgrade=False)
update_uv_dependencies(verbose=True, upgrade=True)
lock_uv_dependencies(verbose=True)

if not check_system_requirements(verbose=True):
    raise SystemExit("System requirements not met")
```

---

## Output Specification

### Output Products
- `environment_setup_summary.json` - Setup completion summary, timings, and probe results (written by `setup.uv_management`)
- `installed_packages.json` - Installed package inventory (written under the step's `setup_artifacts/` subdirectory)
- `uv.lock` - Dependency lock file (updated)

### Output Directory Structure
```
output/1_setup_output/
├── environment_setup_summary.json
└── installed_packages.json
```

---

## Performance Characteristics

Full setup is dominated by `uv sync` download/install time; no performance figures are pinned here.

---

## Error Handling

Setup functions log failures and return `False` on: missing or outdated Python, missing `uv`, failed venv creation, and `uv sync`/`uv add`/`uv remove`/`uv lock` command errors (network, permission, lock issues).

`recreate=True` (or `recreate_venv` for the orchestrator) rebuilds `.venv` from scratch; `lock_uv_dependencies` regenerates `uv.lock`. There is no automatic retry logic.

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

Measure on demand — this file does not pin a number:

```bash
uv run --extra dev python -m pytest tests/setup tests/test_uv_environment.py \
    --cov=src/gnn/setup --cov-report=term-missing
```

### Key Test Scenarios
1. Environment creation and setup
2. Dependency installation via UV sync
3. System requirement verification
4. Native UV command operations
5. Error handling and recovery

---

## MCP Integration
### Tools Registered

`setup/mcp.py` `register_tools` registers these tools (no `setup.` prefix):

- `ensure_directory_exists` - Create a directory if missing
- `find_project_gnn_files` - Find GNN `.md` files in a directory
- `get_standard_output_paths` - Get/create standard step output subdirectories
- `check_uv_project_status` - Check pyproject.toml, uv.lock, and venv status
- `get_uv_environment_info` - Current UV environment paths and status
- `setup_uv_project_structure` - Set up a new UV project structure
- `install_uv_dependency` - Install a package with optional extras
- `sync_uv_dependencies` - Sync dependencies from pyproject.toml

---

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
