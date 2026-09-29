# ngc-learn Runner — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

## Purpose

Step 12 backend that discovers and runs rendered ngc-learn simulation scripts (continuous linear-Gaussian models) via subprocess, with dependency checking, syntax pre-validation, log persistence, and wall-clock timing. ngc-learn is the py3.12 marker-gated `ngclearn` extra; an absent runtime is a skip, never a failure.

## Architecture

```
execute/ngclearn/
├── __init__.py           # Package exports
├── AGENTS.md             # Sub-module agent scaffolding
├── README.md             # Usage and feature notes
├── ngclearn_runner.py    # Availability probe, script discovery, subprocess execution
└── SPEC.md               # This specification
```

## Contract

| Element | Behaviour |
|---|---|
| Availability probe | `is_ngclearn_available()` → `False` without the extra; imports `jax`, `ngclearn`, and `ngcsimlib` (the import is the check) |
| Runner | `run_ngclearn_scripts(...)` is fail-closed: `False` when the probe fails, `True` when no ngclearn scripts are found, and `failure_count == 0` after execution |
| Executor spec row | `result_key` `ngclearn_executions`, `operation_name` `execute_ngclearn_scripts`; absent runtime records status `SKIPPED` with message `ngc-learn framework not installed (optional dependency - install with: uv sync --extra ngclearn)` and `total_failures` 0 |
| Env routing | `NGCLEARN_OUTPUT_DIR` → results land in `<model>/ngclearn/simulation_data/simulation_results.json` |
| Script-path pre-flight | On Python < 3.12 the extra cannot install: reason `Dependency not installable: ngclearn requires Python >= 3.12 (interpreter is 3.11)`, `skip_category` `python_too_old`; on Python >= 3.12 without the extra: reason `Dependency not installed: ngclearn`, `skip_category` `missing_module`, hint `uv sync --extra ngclearn` |