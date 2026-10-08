# GNN Pipeline Logging

Modular, structured logging system for Active Inference model processing.

Import from `gnn.utils.logging_utils` — the single public entry point. The
implementation module `gnn.utils.logging.logging_utils` is internal.

Handler lifecycle and event emission remain in that implementation module.
Record formatting and shared correlation context live in
[`formatters.py`](formatters.py), while terminal presentation and progress
tracking live in [`visual.py`](visual.py). Neither presentation owner initializes
logging handlers; the existing facade still exports the same public classes.

## Quick Start

### For Pipeline Modules
Initialize logging at the start of your module:

```python
from gnn.utils.logging_utils import setup_step_logging

logger = setup_step_logging("my_step_name", verbose=True)
logger.info("Starting processing")
```

### Logging with Structured Data
Attach metadata to your logs for better analysis:

```python
from gnn.utils.logging_utils import PipelineLogger

PipelineLogger.log_structured(
    logger, logging.INFO, "Processed file", file_name="model.md", size_bytes=1024
)
```

### Timing Operations
Use the context manager to automatically log duration and memory usage:

```python
from gnn.utils.logging_utils import PipelineLogger

with PipelineLogger.timed_operation("Parsing GNN", logger):
    # Perform expensive work
    parse_gnn()
```

## Features
- **Correlation-Aware**: Every log line is tagged with a correlation ID for easy tracing.
- **Visual Feedback**: Real-time progress bars and color-coded status in the terminal.
- **Structured Output**: Concurrent JSON-L logging for automated analysis.
- **Performance Monitoring**: Automatic memory and duration tracking for all major operations.

## Output Locations
- **Terminal**: Real-time human-readable logs.
- **`output/00_pipeline_logs/pipeline.log`**: Full text logs.
- **`output/00_pipeline_logs/pipeline.jsonl`**: Machine-readable structured logs.

## Best Practices
1. **Use Emojis**: Use standard pipeline emojis for consistency:
   - 🚀 `log_step_start`
   - ✅ `log_step_success`
   - ⚠️ `log_step_warning`
   - ❌ `log_step_error`
2. **Avoid Print**: Never use `print()`. Always use the `logger`.
3. **Propagate**: Do not add handlers to your local loggers. Rely on the root handlers provided by the pipeline.
