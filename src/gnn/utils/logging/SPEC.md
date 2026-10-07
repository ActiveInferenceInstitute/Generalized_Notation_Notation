# GNN Pipeline Logging Specification (SPEC.md)

## Overview
The GNN Pipeline uses a centralized, structured logging system designed for high observability, correlation-based tracing across 25 processing steps, and rich visual feedback in the terminal.

## Architecture

`logging_utils.py` owns handler configuration, rotation, event emission and
timing. `formatters.py` owns record formatting and its one shared thread-local
correlation context. `visual.py` owns terminal presentation and progress state;
it depends on formatters, never on handler configuration. Existing public names
and signatures remain available through their prior import surfaces.

Text formatters decorate a copy of the log record. Multiple text handlers
therefore emit metadata once, while JSON handlers retain the original message
and its separately structured fields regardless of handler order. Correlation
context remains isolated by thread and shared by every formatter.

### 1. Thin Orchestrator Pattern
Logging is initialized in the numbered step scripts (e.g., `3_gnn.py`) via `setup_step_logging()`. This ensures that logs from the step orchestrator and the underlying module implementations are unified.

### 2. Correlation IDs
Every log record carries a `correlation_id` and a `step_name`.
- **Format**: `[ID:STEP]` (e.g., `[04ae8b91:3_gnn]`)
- **Persistence**: Correlation IDs are stored in thread-local storage (`threading.local()`) and automatically injected into log records by the `CorrelationFormatter`.
- **Tracing**: This allows tracing a single execution context through multiple modules and layers.

## Log Formats

### 1. Visual Text (Human-Readable)
- **Destination**: `stdout`
- **Format**: `%(asctime)s [%(correlation_id)s:%(step_name)s] %(name)s - %(levelname)s - %(message)s`
- **Features**: Color coding, emojis (🚀, ✅, ⚠️, ❌), and performance metrics in square brackets (e.g., `[⏱️ 1.2s | 🧠 150MB]`).

### 2. Structured JSON (Machine-Readable)
- **Destination**: `output/00_pipeline_logs/pipeline.jsonl`
- **Format**: JSON Lines (one JSON object per line).
- **Fields**: `timestamp`, `level`, `logger`, `message`, `correlation_id`, `step_name`, `data` (structured metadata), `performance`.

## Log Levels
- **DEBUG**: Verbose information for troubleshooting.
- **INFO**: Standard operational messages.
- **STEP (25)**: Custom level for marking major pipeline milestones.
- **WARNING**: Recoverable issues or potential problems.
- **ERROR**: Serious issues that might affect step output.
- **CRITICAL**: System-wide failures.

## Performance Tracking
Integrated with `gnn.utils.observability.performance_tracking`.
- Automatically logs memory usage (RSS) and duration for tracked operations.
- Appends `[⏱️ duration | 🧠 memory]` to visual logs when available.

## Handler Management
To prevent duplication:
- `PipelineLogger.initialize()` clears existing handlers on the root logger.
- Only one `StreamHandler` and one `FileHandler` are permitted on the root logger.
- Propagation is used for child loggers instead of attaching per-logger handlers.
