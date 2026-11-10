# PyTorch Execution Sub-module

## Overview

Discovers and runs PyTorch-generated POMDP scripts via subprocess. Includes syntax validation, dependency checking, log persistence, and execution timing.

## Architecture

```
pytorch/
├── __init__.py            # Package exports
└── pytorch_runner.py      # PyTorch script discovery and execution (239 lines)
```

## Key Functions

- **`run_pytorch_scripts(render_dir, output_dir)`** — Discovers and executes all PyTorch scripts in the render output directory.
- **Dependency validation** — Checks for `torch` availability before execution.
- **Syntax checking** — Pre-validates Python syntax before subprocess execution.
- **Log persistence** — Captures stdout/stderr to log files alongside results.
- **Timing** — Records wall-clock execution time for each script.

## Dependencies

- `torch` (runtime, checked before execution)
- `torchlens` (optional; the child script imports it only when
  `GNN_TORCHLENS_CAPTURE` is set, and records `unavailable` when it is missing)

## Rules

- The runner must not import `torchlens`; capture lives in the emitted
  script so the parent stays free of heavy imports.
- Capture outcomes are reported by the child in `simulation_results.json`;
  the runner does not reinterpret them.

## Parent Module

See [execute/AGENTS.md](../AGENTS.md) for the overall execution architecture.

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)
