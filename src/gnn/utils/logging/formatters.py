"""Record formatting and shared thread-local correlation for pipeline logging.

Formatting never configures handlers or starts pipeline instrumentation.
"""

import json
import logging
import threading
from datetime import datetime
from typing import Any

_correlation_context = threading.local()


class CorrelationFormatter(logging.Formatter):
    """Formatter that includes correlation IDs for tracing across pipeline steps."""

    def format(self, record: Any) -> Any:
        # Add correlation ID to log record
        """Provide format behavior."""
        correlation_id = getattr(_correlation_context, "correlation_id", "MAIN")
        step_name = getattr(_correlation_context, "step_name", "pipeline")

        # Create enhanced record with correlation info
        record.correlation_id = correlation_id
        record.step_name = step_name

        return super().format(record)


class StructuredFormatter(CorrelationFormatter):
    """Formatter that handles structured logging data."""

    def format(self, record: Any) -> Any:
        # Extract structured data if present
        """Provide format behavior."""
        if hasattr(record, "structured_data"):
            structured_data = record.structured_data

            # Add structured data to the log message
            if isinstance(structured_data, dict) and structured_data:
                structured_str = " | ".join(
                    [
                        f"{k}={v}"
                        for k, v in structured_data.items()
                        if k != "event_type"
                    ]
                )
                if structured_str:
                    record.msg = f"{record.msg} [{structured_str}]"

        # Continue with correlation formatting
        # Continue with correlation formatting
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON lines for structured logging."""

    def format(self, record: Any) -> Any:
        """Format the log record as a valid JSON object."""
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(
                record,
                "correlation_id",
                getattr(_correlation_context, "correlation_id", "MAIN"),
            ),
            "step_name": getattr(
                record,
                "step_name",
                getattr(_correlation_context, "step_name", "pipeline"),
            ),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.threadName,
        }

        # Add structured data if present
        if hasattr(record, "structured_data"):
            log_entry["data"] = record.structured_data

        # Add performance context if present
        if hasattr(record, "performance_context"):
            log_entry["performance"] = record.performance_context

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)
