# Setup Module (Step 1)

Environment setup and dependency management for the GNN pipeline. Delegates all package
operations to [uv](https://docs.astral.sh/uv/) and reads dependencies from
`pyproject.toml` / `uv.lock`.

The thin orchestrator [`src/gnn/1_setup.py`](../1_setup.py) calls functions exported here.

`OPTIONAL_GROUPS` derives selectable names from installed `Provides-Extra`
metadata and preserves maintained descriptions. Source-only use reads the
checkout's actual `project.optional-dependencies` when distribution metadata is
absent; malformed installed metadata fails discovery rather than falling back.
No optional scientific runtime is imported to discover groups. Python module
metadata/options and CLI, REST and MCP preflight payloads expose this inventory,
including groups added by the packaging owner. Discovery does not establish
native readiness or perform an installation.

## Module Layout

```
src/gnn/setup/
├── __init__.py           # Public API (see __all__)
├── constants.py          # Paths, minimum Python version, OPTIONAL_GROUPS table
├── setup.py              # `perform_full_setup` — three-phase orchestration
├── uv_management.py      # Env creation, uv sync, probes, health checks
├── uv_package_ops.py     # `uv add` / `uv remove` / `uv sync` / `uv lock`
├── dependency_setup.py   # JAX stack probe, optional group install, project init
├── utils.py              # Directory helpers, module info, output paths
├── validator.py          # System / environment / uv status reports
├── mcp.py                # MCP tool registrations
├── AGENTS.md             # Agentic overview (API signatures)
├── SPEC.md               # Component breakdown
└── SKILL.md              # Capability card for LLM consumers
```

## Setup Workflow

```mermaid
flowchart LR
    CLI[1_setup.py] --> Orch[setup_orchestrator]
    Orch -->|default| Sync[setup_uv_environment]
    Orch -->|--install-optional| Full[setup_complete_environment]
    Sync --> Phase1[check_system_requirements]
    Phase1 --> Phase2[create_uv_environment]
    Phase2 --> Phase3[install_uv_dependencies]
    Phase3 --> Probe[run_jax_stack_probe_subprocess]
    Full --> Sync
    Full --> Extras[install_optional_package_group*]
```

## Public API

The sections below list what is exported via `src/gnn/setup/__init__.py`. Every signature below
matches the implementation. Prefer importing from the package root (`from setup import …`)
rather than submodules.

### Environment setup

| Function | Defined in | Summary |
|----------|-----------|---------|
| `setup_uv_environment(verbose=False, recreate=False, dev=False, extras=None, install_all_extras=False, skip_jax_test=False, output_dir=None)` | `uv_management` | End-to-end env creation + `uv sync` + optional JAX probe |
| `install_uv_dependencies(verbose=False, dev=False, extras=None, install_all_extras=False)` | `uv_management` | `uv sync` with selected `--extra`/`--all-extras` flags |
| `validate_uv_setup()` | `uv_management` | Structured dict of venv/python/uv status |
| `cleanup_uv_setup(verbose=False)` | `uv_management` | Best-effort removal of `.venv` |
| `get_uv_setup_info()` | `uv_management` | Current uv version, lock hash, platform metadata |
| `check_system_requirements(verbose=False) -> bool` | `uv_management` | Minimum Python, memory, disk checks |
| `check_uv_availability() -> bool` | `uv_management` | Whether `uv` is on `PATH` |
| `check_environment_health()` | `uv_management` | Aggregate venv health report |
| `log_system_info(verbose=False)` | `uv_management` | Emits platform and interpreter info |
| `get_installed_package_versions()` | `uv_management` | Dict of installed package → version |

### Native uv operations (`uv_package_ops`)

```python
add_uv_dependency(package: str, dev: bool = False, verbose: bool = False) -> bool
remove_uv_dependency(package: str, verbose: bool = False) -> bool
update_uv_dependencies(verbose: bool = False, upgrade: bool = False) -> bool
lock_uv_dependencies(verbose: bool = False) -> bool
```

### Optional dependency groups (`dependency_setup`)

```python
install_optional_dependencies(verbose: bool = False) -> bool
install_optional_package_group(group_name: str, verbose: bool = False) -> bool
install_all_optional_packages(verbose: bool = False) -> bool
setup_complete_environment(verbose=False, recreate=False, install_optional=False,
                           optional_groups: list[str] | None = None,
                           output_dir: Path | None = None) -> bool
setup_gnn_project(project_name: str, base_dir: Path, verbose: bool = False) -> bool
create_project_structure(project_root: Path, verbose: bool = False) -> bool
install_jax_and_test(verbose: bool = False) -> bool
```

Valid group names are defined in `OPTIONAL_GROUPS` (`constants.py`):
`dev`, `api`, `ml-ai`, `audio`, `gui`, `graphs`, `research`, `scaling`, `all`.

Step 12 core backends (JAX, NumPyro, DisCoPy), LLM clients, and interactive
visualization packages are already in `[project.dependencies]`, so `uv sync`
installs them without any extra. PyTorch and bnlearn remain supported by the
renderer/executor code paths, but are manual installs while the current Torch
advisory has no patched release.

### Validators (`validator`)

```python
validate_system() -> dict        # {"success": bool, "message"|"error": str}
get_environment_info() -> dict   # uv + package versions
get_uv_status() -> dict          # uv availability and setup info
```

### Helpers (`utils`)

```python
ensure_directory(path: Path) -> Path
find_gnn_files(directory: Path, recursive: bool = True) -> list[Path]
get_output_paths(base_output_dir: Path) -> dict[str, Path]
get_module_info() -> dict
get_setup_options() -> dict
setup_environment(verbose=False, **kwargs) -> bool     # thin wrapper around uv_management
install_dependencies(verbose=False, **kwargs) -> bool  # thin wrapper around uv_management
```

### Orchestration

`perform_full_setup(verbose=False, recreate_venv=False, dev=False, extras=None,
skip_jax_test=False) -> int` in `setup.py` runs the three-phase flow
(requirements → env → dependencies → optional JAX probe) and returns a shell exit code.
The thin orchestrator `src/gnn/1_setup.py` defines `setup_orchestrator(target_dir, output_dir,
logger, **kwargs)` which wraps `perform_full_setup` / `setup_complete_environment`.

### Package metadata

- `__version__` — currently `"1.6.0"`
- `FEATURES` — dict of capability flags (see `__init__.py`)
- `OPTIONAL_GROUPS` — name → description map used by Step 1 CLI help
- `EnvironmentManager`, `VirtualEnvironment` — thin classes kept for setup tests
- `validate_environment()`, `check_python_version()` — small helpers used by tests

## Usage

### From the command line

```bash
# Default: uv sync core dependencies (Step 12 backends already included).
python src/gnn/1_setup.py --target-dir input/gnn_files --output-dir output --verbose

# Skip JAX/Optax/Flax/pymdp self-test
python src/gnn/1_setup.py --setup-core-only --verbose

# Development tools (pytest, ruff, mypy, …)
python src/gnn/1_setup.py --dev --verbose

# Every optional group
python src/gnn/1_setup.py --install-all-extras --verbose

# Specific groups beyond the core dependency set
python src/gnn/1_setup.py --install-optional --optional-groups "gui,audio" --verbose
```

### Programmatic

```python
from gnn.setup import setup_uv_environment, install_optional_package_group

setup_uv_environment(verbose=True, dev=True)
install_optional_package_group("gui", verbose=True)
```

### `uv` directly

```bash
uv sync                     # Core (JAX, NumPyro, DisCoPy, pymdp, …; PyTorch is a manual install)
uv sync --extra dev         # Add dev tools
uv sync --extra gui
uv sync --all-extras        # Everything
```

## Output

Step 1 writes artifacts under `output/1_setup_output/`:

```
output/1_setup_output/
├── environment_setup_summary.json  # High-level setup result, timings, and probes
└── installed_packages.json         # Installed packages and versions
```

The root `uv.lock` file is updated by `uv` itself when any sync operation runs.

## Error Handling

- Missing `uv` → `check_uv_availability()` returns `False`; callers log a clear error and
  point at <https://docs.astral.sh/uv/>.
- `uv sync` failure → surfaced via `subprocess.CompletedProcess.returncode`; logs include
  captured stdout/stderr and return `False`.
- JAX probe failure → a single `uv sync` retry with `SETUP_DEFAULT_PIPELINE_EXTRAS`, then
  a final warning; Step 1 still returns success so the pipeline continues.
- Lock file conflicts → regenerate with `uv lock` (`lock_uv_dependencies`).

## Testing

```bash
uv run --extra dev python -m pytest tests/setup/test_setup_overall.py -v
uv run --extra dev python -m pytest tests/test_uv_environment.py -v
uv run --extra dev python -m pytest tests/test_environment_overall.py -v
```

The tests cover: environment creation, `uv sync` invocation, optional-group installation,
system requirement validation, and the JAX stack probe.

## MCP Integration

`mcp.py` registers the following tools with the GNN MCP server:

- `ensure_directory_exists`
- `find_project_gnn_files`
- `get_standard_output_paths`
- `check_uv_project_status`
- `get_uv_environment_info`
- `setup_uv_project_structure`
- `install_uv_dependency`
- `sync_uv_dependencies`

## References

- [`src/gnn/1_setup.py`](../1_setup.py) — thin orchestrator
- [`pyproject.toml`](../../../pyproject.toml) — dependency declarations and extras
- [`SETUP_GUIDE.md`](../../../SETUP_GUIDE.md) — end-to-end install walkthrough
- [`docs/gnn/modules/01_setup.md`](../../../docs/gnn/modules/01_setup.md) — module documentation

---

## Documentation

- **[README](README.md)** — this file
- **[AGENTS](AGENTS.md)** — agent-facing overview
- **[SPEC](SPEC.md)** — component breakdown
- **[SKILL](SKILL.md)** — capability card

Explicit backend groups include `torch`, `stan`, `bnlearn`, `ngclearn`, and
`cpomdp`. The Stan extra supplies cmdstanpy; CmdStan requires the separate
opt-in installation described in [Stan installation](../../../docs/gnn/implementations/stan.md#installation).
Readiness diagnostics distinguish a missing Python module from a missing
toolchain, unsupported interpreter/version, failed probe, and probe timeout.
