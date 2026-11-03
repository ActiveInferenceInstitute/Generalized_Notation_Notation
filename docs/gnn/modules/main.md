# `main.py` — Pipeline Orchestrator

## Overview

The main pipeline orchestrator executes all 25 steps (0–24) in sequence with comprehensive monitoring, dependency resolution, and structured logging. This is the entry point for running the complete GNN processing pipeline.

## Usage

```bash
# Full pipeline
uv run python src/gnn/main.py --target-dir input/gnn_files --verbose

# Specific steps only (auto-resolves dependencies)
uv run python src/gnn/main.py --only-steps "0,1,2,3" --verbose

# Skip certain steps
uv run python src/gnn/main.py --skip-steps "15,16" --verbose
```

## Features

### Step Dependency Resolution

When using `--only-steps`, dependencies are automatically included:

| Step | Dependencies |
|:----:|:------------:|
| 11 (Render) | 3 (GNN) |
| 12 (Execute) | 3, 11 |
| 16 (Analysis) | 3, 7 |
| 23 (Report) | 8, 13 |
| 24 (Intelligent Analysis) | all prior steps (0–23) |

### Execution Infrastructure

- **Subprocess isolation**: Each step runs as a separate Python process
- **Streaming output**: Real-time stdout/stderr via `execute_command_streaming()`
- **Step timeouts**: Configurable per-step timeouts via `pipeline/step_timeouts.py`
- **Memory tracking**: Per-step and peak memory usage via `get_current_memory_usage()`
- **Prerequisite validation**: `validate_step_prerequisites()` checks before each step

### Logging and Progress

- **Structured JSON logging**: Via `PipelineLogger.enable_json_logging()` when available
- **Visual logging**: Emoji-prefixed progress, banners, and step summaries
- **Correlation IDs**: UUID-based tracking across log entries
- **Log rotation**: Automatic log file rotation in `output/00_pipeline_logs/`

### Pipeline Summary

Saved to `output/00_pipeline_summary/pipeline_execution_summary.json` with:

- Per-step timing, exit codes, memory usage, and status
- Overall status: `SUCCESS`, `SUCCESS_WITH_WARNINGS`, `PARTIAL_SUCCESS`, or `FAILED`
- Per-step warning evidence: `warning_reason_source` (`step_output_scan`, `step_declaration`, `exit_code_only`, or `none`) with bounded `warning_reasons`, `warning_reason_count`, and `warning_reasons_truncated` — see [pipeline_warnings.md](../../troubleshooting/pipeline_warnings.md)
- Environment info (Python version, platform, CPU count)
- Safe warning pattern filtering (matplotlib backend, optional dependencies)

## Source

- **Script**: [src/gnn/main.py](../../../src/gnn/main.py)
- **Shape**: `main()` is a compact coordinator; argument/config preparation, step execution recording, final summary writing, and failure handling live in named helpers.
- **Key functions**: `main()`, `execute_pipeline_step()`, `parse_step_list()`, `get_environment_info()`, `validate_pipeline_summary()`
