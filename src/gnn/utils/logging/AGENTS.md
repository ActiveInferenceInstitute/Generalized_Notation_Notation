# Logging Sub-module

## Overview

Centralized, production-grade logging infrastructure for the GNN pipeline. Provides correlation-aware tracing across all 25 steps, visual progress indicators, and multi-format structured output (JSON-L + Text).

## Architecture

Follows the **Thin Orchestrator** pattern where individual steps delegate to this centralized system to ensure consistent formatting and deduplication.

```
logging/
├── __init__.py          # Public API exports
├── logging_utils.py     # Handler lifecycle, logging events and timing
├── formatters.py        # Correlation context and text/JSON record formatting
├── visual.py            # Terminal formatting and in-memory progress
├── AGENTS.md            # Agent capabilities and status
├── README.md            # Developer usage guide
└── SPEC.md              # Technical specification
```

## Hardening Achievements (v1.6.0)

- **Remediated Duplication**: Fixed a handler collision between root and step-level handlers that caused duplicate terminal output.
- **Unified Tracing**: Established a single-source-of-truth for correlation IDs, ensuring logs from subprocesses are correctly tagged and formatted.
- **Logging Facade**: `gnn.utils.logging_utils` retains the supported public imports;
  internal formatter and progress names are also re-exported from
  `logging/logging_utils.py`. The shared correlation context is owned by
  `formatters.py`; importing presentation owners does not configure handlers.

## Key Capabilities

- **`setup_step_logging`** — Comprehensive configuration with auto-detection of terminal capabilities.
- **`log_step_*` Suite** — Semantic logging with status markers and progress tracking.
- **`timed_operation`** — Context manager for automatic $O(N)$ performance instrumentation.
- **Structured JSON-L** — Concurrent machine-readable logging for downstream analysis.
- **`logging.TRACE` (numeric 5)** — Registered for high-volume parser diagnostics; enable root/handlers at TRACE when debugging parse traces without flooding default `--verbose` output.

## Features

- **Correlation Tracing**: `[ID:STEP]` tags on every line for cross-module debugging.
- **Rich Aesthetics**: Vibrant terminal output with glassmorphism-compatible visual logging.
- **Performance Aware**: Integrated memory and duration tracking in every log line.
- **Resilient Initialization**: Defensive handler management prevents double-logging even with redundant calls.

## Documentation
- [Technical Specification](SPEC.md)
- [Developer Guide](README.md)

---
**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)
**Status**: Production Hardened  
**Pipeline Integration**: Steps 0-24
