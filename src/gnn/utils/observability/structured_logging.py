#!/usr/bin/env python3
"""
GNN Pipeline Structured Logging Framework

This module provides comprehensive structured logging capabilities for the GNN processing
pipeline with correlation IDs, performance tracking, and multi-format output support.
"""

import json
import logging
import os
import platform
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# ``psutil`` is process telemetry, not a logging prerequisite: it is imported
# lazily by ``_process_handle`` so that importing this module (and therefore
# ``gnn.ontology`` and the other packages that log through it) succeeds in a
# minimal closure without ``psutil``. When it is installed the recorded
# telemetry is unchanged; when it is absent performance tracking degrades to
# the same non-fatal debug path as any other probe failure (issue #400).


def _process_handle() -> Any:
    """Return ``psutil.Process()`` for this process, importing ``psutil`` lazily.

    Raises ``ImportError`` when ``psutil`` is unavailable; callers treat that
    like any other telemetry probe failure.
    """
    import psutil

    return psutil.Process()


class LogLevel(Enum):
    """Structured log levels with numeric values."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogFormat(Enum):
    """Supported log output formats."""

    JSON = "json"
    TEXT = "text"
    STRUCTURED = "structured"


@dataclass
class LogContext:
    """Context information for structured logging."""

    correlation_id: str
    step_name: Optional[str] = None
    operation: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )
    version: str = "1.1.3"
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""

    start_time: float
    end_time: Optional[float] = None
    memory_start_mb: float = 0.0
    memory_end_mb: Optional[float] = None
    cpu_percent_start: float = 0.0
    cpu_percent_end: Optional[float] = None
    operation_count: int = 0
    error_count: int = 0

    @property
    def duration_seconds(self) -> float:
        """Calculate operation duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def memory_delta_mb(self) -> float:
        """Calculate memory usage delta."""
        if self.memory_end_mb is not None:
            return self.memory_end_mb - self.memory_start_mb
        return 0.0


class StructuredLogger:
    """Enhanced logger with structured logging and performance tracking."""

    def __init__(self, name: str, correlation_id: Optional[str] = None) -> None:
        """Initialize the instance."""
        self.name = name
        self.correlation_id = correlation_id or self._generate_correlation_id()
        self.base_logger = logging.getLogger(name)
        self.context = LogContext(correlation_id=self.correlation_id)
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        self.log_format = LogFormat.STRUCTURED
        self._setup_base_logger()

    def _generate_correlation_id(self) -> str:
        """Generate a unique correlation ID."""
        return f"gnn_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    def _setup_base_logger(self) -> Any:
        """Setup the base logger with appropriate formatting."""
        if not self.base_logger.handlers:
            handler = logging.StreamHandler()
            formatter = StructuredFormatter(self.correlation_id)
            handler.setFormatter(formatter)
            self.base_logger.addHandler(handler)
            self.base_logger.setLevel(logging.DEBUG)

    def set_context(self, **kwargs: Any) -> Any:
        """Update the logging context."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)

    def log(self, level: LogLevel, message: str, **extra_data: Any) -> Any:
        """Log a message with structured data."""
        log_data: dict[Any, Any] = {
            "correlation_id": self.correlation_id,
            "step_name": self.context.step_name,
            "operation": self.context.operation,
            "timestamp": time.time(),
            "level": level.name,
            "message": message,
            "logger": self.name,
            "thread": threading.current_thread().name,
            "process": os.getpid(),
            **extra_data,
        }

        # Add performance metrics if tracking an operation
        if (
            self.context.operation
            and self.context.operation in self.performance_metrics
        ):
            metrics = self.performance_metrics[self.context.operation]
            log_data["performance"] = {
                "duration_seconds": metrics.duration_seconds,
                "memory_delta_mb": metrics.memory_delta_mb,
                "operation_count": metrics.operation_count,
                "error_count": metrics.error_count,
            }

        # Choose log method based on level
        log_method = getattr(self.base_logger, level.name.lower())

        if self.log_format == LogFormat.JSON:
            log_method(json.dumps(log_data))
        else:
            # For text/structured format, log the message and add structured data
            log_method(f"[{self.correlation_id}] {message}")
            if extra_data:
                self.base_logger.debug(
                    f"[{self.correlation_id}] Extra data: {extra_data}"
                )

    def debug(self, message: str, **extra_data: Any) -> Any:
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **extra_data)

    def info(self, message: str, **extra_data: Any) -> Any:
        """Log info message."""
        self.log(LogLevel.INFO, message, **extra_data)

    def warning(self, message: str, **extra_data: Any) -> Any:
        """Log warning message."""
        self.log(LogLevel.WARNING, message, **extra_data)

    def error(self, message: str, **extra_data: Any) -> Any:
        """Log error message."""
        self.log(LogLevel.ERROR, message, **extra_data)

    def critical(self, message: str, **extra_data: Any) -> Any:
        """Log critical message."""
        self.log(LogLevel.CRITICAL, message, **extra_data)

    @contextmanager
    def operation_context(
        self, operation_name: str, track_performance: bool = True
    ) -> Any:
        """Context manager for operation tracking."""
        old_operation = self.context.operation
        self.context.operation = operation_name

        if track_performance:
            self._start_performance_tracking(operation_name)

        try:
            yield
        finally:
            if track_performance:
                self._end_performance_tracking(operation_name)

            self.context.operation = old_operation

    def _start_performance_tracking(self, operation_name: str) -> Any:
        """Start tracking performance metrics for an operation."""
        try:
            process = _process_handle()
            memory_info = process.memory_info()

            metrics = PerformanceMetrics(
                start_time=time.time(),
                memory_start_mb=memory_info.rss / 1024 / 1024,
                cpu_percent_start=process.cpu_percent(),
            )

            self.performance_metrics[operation_name] = metrics

        except Exception as e:
            # If performance tracking fails, continue without it
            self.base_logger.debug(
                f"Performance tracking start failed (non-fatal): {e}"
            )

    def _end_performance_tracking(self, operation_name: str) -> Any:
        """End tracking performance metrics for an operation."""
        if operation_name not in self.performance_metrics:
            return

        try:
            metrics = self.performance_metrics[operation_name]
            process = _process_handle()
            memory_info = process.memory_info()

            metrics.end_time = time.time()
            metrics.memory_end_mb = memory_info.rss / 1024 / 1024
            metrics.cpu_percent_end = process.cpu_percent()

            # Log performance summary
            self.info(
                f"Operation '{operation_name}' completed",
                duration_seconds=round(metrics.duration_seconds, 3),
                memory_delta_mb=round(metrics.memory_delta_mb, 2),
            )

        except Exception as e:
            # If performance tracking fails, continue without it
            self.base_logger.debug(f"Performance tracking end failed (non-fatal): {e}")

    def log_pipeline_step(
        self,
        step_name: str,
        status: str,
        duration: (float) | None = None,
        **extra_data: Any,
    ) -> Any:
        """Log pipeline step completion with structured data."""
        self.set_context(step_name=step_name)

        log_data: dict[Any, Any] = {
            "step_name": step_name,
            "status": status,
            "event_type": "pipeline_step",
            **extra_data,
        }

        if duration is not None:
            log_data["duration_seconds"] = round(duration, 3)

        if status == "SUCCESS":
            self.info(f"Pipeline step '{step_name}' completed successfully", **log_data)
        elif status == "SUCCESS_WITH_WARNINGS":
            self.warning(
                f"Pipeline step '{step_name}' completed with warnings", **log_data
            )
        elif status == "FAILED":
            self.error(f"Pipeline step '{step_name}' failed", **log_data)
        else:
            self.info(
                f"Pipeline step '{step_name}' completed with status: {status}",
                **log_data,
            )

    def log_error(
        self, error: Exception, step_name: Optional[str] = None, **context: Any
    ) -> Any:
        """Log an error with full context and traceback."""
        import traceback

        if step_name:
            self.set_context(step_name=step_name)

        error_data: dict[Any, Any] = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "module": getattr(error, "__module__", "unknown"),
            "event_type": "error",
            **context,
        }

        self.error(f"Error in {step_name or 'unknown'}: {error}", **error_data)

    def create_child_logger(self, child_name: str) -> "StructuredLogger":
        """Create a child logger with the same correlation ID."""
        child_logger = StructuredLogger(
            f"{self.name}.{child_name}", self.correlation_id
        )
        child_logger.context = self.context
        return child_logger


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging output."""

    def __init__(self, correlation_id: str) -> None:
        """Initialize the instance."""
        super().__init__()
        self.correlation_id = correlation_id

    def format(self, record: Any) -> Any:
        # Add correlation ID to all log records
        """Provide format behavior."""
        if not hasattr(record, "correlation_id"):
            record.correlation_id = self.correlation_id

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).isoformat()

        # Create structured message
        if hasattr(record, "correlation_id"):
            message = f"[{record.correlation_id}] {timestamp} [{record.levelname}] {record.name}: {record.getMessage()}"
        else:
            message = (
                f"{timestamp} [{record.levelname}] {record.name}: {record.getMessage()}"
            )

        # Add any extra data from the record
        if hasattr(record, "extra_data") and record.extra_data:
            message += f" | {record.extra_data}"

        return message


class LogAggregator:
    """Aggregate logs from multiple sources for analysis."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logs: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}

    def add_log(self, log_data: Dict[str, Any]) -> Any:
        """Add a log entry to the aggregator."""
        self.logs.append(log_data)

        # Update metrics
        if log_data.get("event_type") == "pipeline_step":
            step_name = str(log_data.get("step_name") or "unknown")
            status = log_data.get("status")
            duration = log_data.get("duration_seconds", 0)

            if step_name not in self.metrics:
                self.metrics[step_name] = {
                    "count": 0,
                    "successes": 0,
                    "failures": 0,
                    "total_duration": 0,
                }

            self.metrics[step_name]["count"] += 1
            self.metrics[step_name]["total_duration"] += duration

            if status in ["SUCCESS", "SUCCESS_WITH_WARNINGS"]:
                self.metrics[step_name]["successes"] += 1
            else:
                self.metrics[step_name]["failures"] += 1

    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of aggregated logs."""
        summary: dict[str, Any] = {
            "total_logs": len(self.logs),
            "step_metrics": self.metrics,
            "timestamp": time.time(),
        }

        # Calculate success rates
        for _, metrics in self.metrics.items():
            if metrics["count"] > 0:
                metrics["success_rate"] = metrics["successes"] / metrics["count"]
                metrics["average_duration"] = (
                    metrics["total_duration"] / metrics["count"]
                )

        return summary

    def save_summary(self, output_dir: Path) -> Path:
        """Save the aggregated summary to a file."""
        summary_file = output_dir / f"log_summary_{int(time.time())}.json"
        summary = self.generate_summary()

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_file


# Global instances for pipeline-wide usage
_log_aggregator = LogAggregator()
_correlation_context = threading.local()


def get_pipeline_logger(name: str) -> StructuredLogger:
    """Get a structured logger for pipeline use."""
    correlation_id = getattr(_correlation_context, "correlation_id", None)
    return StructuredLogger(name, correlation_id)


def set_correlation_context(correlation_id: str) -> Any:
    """Set the correlation ID for the current thread."""
    _correlation_context.correlation_id = correlation_id


def get_system_info() -> Dict[str, str]:
    """Get system information for logging context."""
    try:
        return {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "cpu_count": str(os.cpu_count()),
            "hostname": platform.node(),
        }
    except Exception:
        return {"error": "Could not retrieve system info"}


# Convenience functions for common logging patterns
def log_pipeline_start(logger: Any, pipeline_name: str, **context: Any) -> Any:
    """Log pipeline start event - compatible with both StructuredLogger and standard Logger."""
    if isinstance(logger, StructuredLogger):
        # StructuredLogger - can handle extra data
        logger.info(
            f"Starting pipeline: {pipeline_name}",
            event_type="pipeline_start",
            pipeline_name=pipeline_name,
            system_info=get_system_info(),
            **context,
        )
    else:
        # Standard logger - just log the message (no extra kwargs)
        logger.info(f"🚀 Starting pipeline: {pipeline_name}")


def log_pipeline_complete(
    logger: Any, pipeline_name: str, status: str, total_duration: float, **context: Any
) -> Any:
    """Log pipeline completion event - compatible with both StructuredLogger and standard Logger."""
    if isinstance(logger, StructuredLogger):
        # StructuredLogger - can handle extra data
        logger.info(
            f"Pipeline {pipeline_name} completed with status: {status}",
            event_type="pipeline_complete",
            pipeline_name=pipeline_name,
            status=status,
            total_duration_seconds=round(total_duration, 3),
            **context,
        )
    else:
        # Standard logger - just log the message (no extra kwargs)
        logger.info(
            f"🏁 Pipeline {pipeline_name} completed with status: {status} in {total_duration:.2f}s"
        )


def log_step_start(logger: Any, step_name: str, **context: Any) -> Any:
    """Log step start event - compatible with both StructuredLogger and standard Logger."""
    if isinstance(logger, StructuredLogger):
        # StructuredLogger - can handle extra data
        logger.info(
            f"Starting step: {step_name}",
            event_type="step_start",
            step_name=step_name,
            **context,
        )
    else:
        # Standard logger - just log the message (no extra kwargs)
        logger.info(f"🚀 Starting step: {step_name}")


def log_step_success(logger: Any, message: str, **context: Any) -> Any:
    """Log step success event - compatible with both StructuredLogger and standard Logger."""
    if isinstance(logger, StructuredLogger):
        # StructuredLogger - can handle extra data
        logger.info(message, event_type="step_success", **context)
    else:
        # Standard logger - just log the message (no extra kwargs)
        logger.info(f"✅ {message}")


def log_step_error(logger: Any, message: str, **context: Any) -> Any:
    """Log step error event - compatible with both StructuredLogger and standard Logger."""
    if isinstance(logger, StructuredLogger):
        # StructuredLogger - can handle extra data
        logger.error(message, event_type="step_error", **context)
    else:
        # Standard logger - just log the message (no extra kwargs)
        logger.error(f"❌ {message}")


def log_step_warning(logger: Any, message: str, **context: Any) -> Any:
    """Log step warning event - compatible with both StructuredLogger and standard Logger."""
    if isinstance(logger, StructuredLogger):
        # StructuredLogger - can handle extra data
        logger.warning(message, event_type="step_warning", **context)
    else:
        # Standard logger - just log the message (no extra kwargs)
        logger.warning(f"⚠️ {message}")
