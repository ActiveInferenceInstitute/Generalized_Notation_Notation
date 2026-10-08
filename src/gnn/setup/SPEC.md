# Setup Module Specification

## Purpose

Own all environment setup for the GNN pipeline (Step 1) using [uv](https://docs.astral.sh/uv/)
as the exclusive package manager. Everything here should be a thin wrapper around `uv`
subcommands plus the project's `pyproject.toml` / `uv.lock`.

## Components

| File | LOC | Role |
|------|----:|------|
| `__init__.py` | 170 | Public API surface and `__all__` re-exports |
| `constants.py` | 97 | Paths (`PROJECT_ROOT`, `VENV_PATH`, `LOCK_PATH`), `MIN_PYTHON_VERSION`, live packaged `OPTIONAL_GROUPS`, `SETUP_DEFAULT_PIPELINE_EXTRAS` |
| `setup.py` | 168 | `perform_full_setup` three-phase orchestrator (system → env → deps) |
| `uv_management.py` | 834 | `setup_uv_environment`, `install_uv_dependencies`, `check_*`, probes, reporting |
| `uv_package_ops.py` | 192 | `add_uv_dependency`, `remove_uv_dependency`, `update_uv_dependencies`, `lock_uv_dependencies` |
| `dependency_setup.py` | 448 | JAX stack probe, optional group installers, Julia, project structure |
| `utils.py` | 189 | `ensure_directory`, `find_gnn_files`, `get_output_paths`, `get_module_info`, `setup_environment`, `install_dependencies` |
| `validator.py` | 84 | `validate_system`, `get_environment_info`, `get_uv_status` |
| `mcp.py` | 354 | MCP tool registrations (setup utilities) |

Line counts are recomputed from the source (`wc -l src/gnn/setup/*.py`). Update when the
implementation changes so this specification stays accurate.

## Responsibilities

1. Detect `uv` availability and install (or point the user at) it.
2. Create or recreate `.venv` via `uv venv` / `uv sync`.
3. Read dependency declarations from `pyproject.toml` and resolve them against
   `uv.lock`.
4. Install optional groups on request (`--extra …`) and surface the set of names via
   `OPTIONAL_GROUPS`.
5. Probe the JAX/Optax/Flax/pymdp stack used by Step 12 (runs in a subprocess so a
   crash in the stack cannot crash Step 1).
6. Report system, environment, and uv status via `validator.py` helpers.

## Key Exports

```python
from gnn.setup import (
    # Env
    setup_uv_environment,
    install_uv_dependencies,
    validate_uv_setup,
    check_system_requirements,
    check_uv_availability,
    # Native uv ops
    add_uv_dependency,
    remove_uv_dependency,
    update_uv_dependencies,
    lock_uv_dependencies,
    # Optional groups
    install_optional_dependencies,
    install_optional_package_group,
    install_all_optional_packages,
    setup_complete_environment,
    # Validators
    validate_system,
    get_environment_info,
    get_uv_status,
    # Helpers
    ensure_directory,
    find_gnn_files,
    get_output_paths,
    OPTIONAL_GROUPS,
    FEATURES,
)
```

## Invariants

- No direct `pip` calls. Anything installed in `.venv` is installed via `uv`.
- No silent fallback from `uv` to `pip`; if `uv` is missing, callers log and return
  `False`.
- Step 1 is allowed to emit warnings but must return a usable environment to downstream
  steps whenever at least `uv sync` succeeded.
- JAX probe failures do **not** fail Step 1 — they log and continue so later steps can
  proceed in the default "core-deps-only" configuration.

## Interfaces

- **Called by**: [`src/gnn/1_setup.py`](../1_setup.py) via `setup_orchestrator`.
- **Imported by**: tests in `tests/setup/`, `tests/test_uv_environment.py`,
  `tests/test_environment_overall.py`, and any other module that needs environment
  introspection (for example dependency-aware fallbacks).
- **External tools**: `uv`, `python`, the interpreter under `.venv`.

## Output

Step 1 writes to `output/1_setup_output/`:

- `environment_setup_summary.json` — setup result, timings, and probe outcomes
- `installed_packages.json` — installed package inventory

`uv.lock` at the repository root is updated by `uv` directly.

---

## Documentation

- **[README](README.md)** — Usage and API
- **[AGENTS](AGENTS.md)** — Agent-facing overview
- **[SPEC](SPEC.md)** — This file
- **[SKILL](SKILL.md)** — Capability card
