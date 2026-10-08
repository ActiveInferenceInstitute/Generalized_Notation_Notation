#!/usr/bin/env python3
"""
Internal implementation module for GNN pipeline logging.

Provides coherent, correlation-based logging across all pipeline steps
with enhanced visual formatting and centralized configuration.

Do not import this module directly: ``gnn.utils.logging_utils`` is the
single public entry point and re-exports this module's public surface.

"""

import gzip
import logging
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, cast

# Import performance tracking from the observability concern package.
from gnn.utils.observability.performance_tracking import performance_tracker

from .formatters import (
    CorrelationFormatter as CorrelationFormatter,
)
from .formatters import (
    JSONFormatter as JSONFormatter,
)
from .formatters import (
    StructuredFormatter as StructuredFormatter,
)
from .formatters import (
    _correlation_context as _correlation_context,
)
from .visual import (
    PipelineProgressTracker as PipelineProgressTracker,
)
from .visual import (
    VisualFormatter as VisualFormatter,
)
from .visual import (
    VisualLoggingEnhancer as VisualLoggingEnhancer,
)


class BasicPipelineLogger:
    """Basic centralized logger for the GNN pipeline with correlation support."""

    _loggers: Dict[str, logging.Logger] = {}
    _initialized = False
    _log_file_handler: Optional[logging.FileHandler] = None

    @classmethod
    def initialize(
        cls,
        log_dir: Optional[Path] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
    ) -> None:
        """Initialize the centralized logging system."""
        if cls._initialized:
            return

        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Clear existing handlers to avoid duplicates
        root_logger.handlers.clear()

        # Create correlation-aware formatter
        console_formatter = CorrelationFormatter(
            "%(asctime)s [%(correlation_id)s:%(step_name)s] %(name)s - %(levelname)s - %(message)s"
        )
        file_formatter = CorrelationFormatter(
            "%(asctime)s [%(correlation_id)s:%(step_name)s] %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # File handler (if log directory provided)
        if log_dir:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "pipeline.log"
                cls._log_file_handler = logging.FileHandler(log_file, mode="w")
                cls._log_file_handler.setLevel(file_level)
                cls._log_file_handler.setFormatter(file_formatter)
                root_logger.addHandler(cls._log_file_handler)
            except OSError as e:
                console_handler.emit(
                    logging.LogRecord(
                        name="PipelineLogger",
                        level=logging.ERROR,
                        pathname="",
                        lineno=0,
                        msg=f"Failed to setup file logging in {log_dir} ({type(e).__name__}: {e})",
                        args=(),
                        exc_info=None,
                    )
                )

        # Silence noisy third-party libraries
        for noisy_lib in ["PIL", "matplotlib", "urllib3", "requests"]:
            logging.getLogger(noisy_lib).setLevel(logging.WARNING)

        cls._initialized = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger with the given name, ensuring it's properly configured."""
        if not cls._initialized:
            cls.initialize()

        if name not in cls._loggers:
            logger = logging.getLogger(name)
            # Ensure it inherits from root configuration
            logger.propagate = True
            cls._loggers[name] = logger

        return cls._loggers[name]

    @classmethod
    def set_correlation_context(
        cls, step_name: str, correlation_id: Optional[str] = None
    ) -> str:
        """Set correlation context for current thread."""
        if correlation_id is None:
            # Try to inherit from existing context if available
            correlation_id = getattr(
                _correlation_context, "correlation_id", str(uuid.uuid4())[:8]
            )

        _correlation_context.correlation_id = correlation_id
        _correlation_context.step_name = step_name

        return correlation_id

    @classmethod
    def clear_correlation_context(cls) -> None:
        """Clear correlation context for current thread."""
        if hasattr(_correlation_context, "correlation_id"):
            delattr(_correlation_context, "correlation_id")
        if hasattr(_correlation_context, "step_name"):
            delattr(_correlation_context, "step_name")

    @classmethod
    def set_verbosity(cls, verbose: bool) -> None:
        """Update console log level based on verbosity."""
        level = logging.DEBUG if verbose else logging.INFO
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if (
                isinstance(handler, logging.StreamHandler)
                and handler.stream == sys.stdout
            ):
                handler.setLevel(level)
                break


def setup_main_logging(
    log_dir: Optional[Path] = None, verbose: bool = False, log_format: str = "human"
) -> logging.Logger:
    """
    Setup logging for the main pipeline orchestrator.

    Args:
        log_dir: Directory for log files
        verbose: Whether to enable verbose logging
        log_format: format for console output ('human' or 'json')

    Returns:
        Configured main logger
    """
    PipelineLogger.initialize(log_dir=log_dir, log_format=log_format)
    PipelineLogger.set_verbosity(verbose)

    correlation_id = PipelineLogger.set_correlation_context("main")
    logger = PipelineLogger.get_logger("GNN_Pipeline")

    logger.info(f"GNN Pipeline logging initialized [correlation_id: {correlation_id}]")

    return logger


# Standalone logging setup for scripts not part of the main pipeline
def setup_standalone_logging(
    level: int = logging.INFO,
    logger_name: str = "GNN_Pipeline",
    output_dir: Optional[Path] = None,
    log_filename: str = "pipeline.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Setup standalone logging with file and console handlers."""
    log_dir = output_dir / "logs" if output_dir else None
    PipelineLogger.initialize(
        log_dir=log_dir, console_level=console_level, file_level=file_level
    )
    return PipelineLogger.get_logger(logger_name)


def silence_noisy_modules_in_console() -> Any:
    """Reduce console output from verbose third-party modules."""
    for module in ["PIL", "matplotlib", "urllib3", "requests"]:
        logging.getLogger(module).setLevel(logging.WARNING)


def set_verbose_mode(verbose: bool) -> Any:
    """Set verbose mode for console output."""
    PipelineLogger.set_verbosity(verbose)


def log_section_header(
    logger: logging.Logger, title: str, char: str = "=", length: int = 80
) -> Any:
    """Log a formatted section header."""
    border = char * length
    padded_title = f" {title} ".center(length, char)

    logger.info("")
    logger.info(border)
    logger.info(padded_title)
    logger.info(border)
    logger.info("")


# Enhanced logging functionality
_logging_any = cast(Any, logging)
_logging_any.TRACE = 5  # Finer than DEBUG for high-volume parse/trace lines
logging.addLevelName(_logging_any.TRACE, "TRACE")

_logging_any.STEP = 25  # Custom log level between INFO and WARNING
logging.addLevelName(_logging_any.STEP, "STEP")


def step(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Log a step-level message."""
    if self.isEnabledFor(_logging_any.STEP):
        self._log(_logging_any.STEP, message, args, **kwargs)


cast(Any, logging.Logger).step = step


def rotate_logs(log_dir: Path, max_files: int = 5, compress: bool = True) -> Any:
    """
    Rotate log files in the specified directory.

    Args:
        log_dir: Directory containing log files
        max_files: Maximum number of rotated files to keep (including current)
        compress: Whether to compress rotated files
    """
    try:
        log_file = log_dir / "pipeline.log"
        if not log_file.exists():
            return

        # Rotate existing logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_name = f"pipeline_{timestamp}.log"
        rotated_path = log_dir / rotated_name

        # Rename current log
        shutil.move(str(log_file), str(rotated_path))

        # Compress if requested
        if compress:
            with open(rotated_path, "rb") as f_in:
                compressed_path = rotated_path.with_suffix(".log.gz")
                with gzip.open(compressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            rotated_path.unlink()  # Remove uncompressed file

        # Cleanup old logs
        pattern = "pipeline_*.log.gz" if compress else "pipeline_*.log"
        log_files = sorted(log_dir.glob(pattern), key=lambda x: x.stat().st_mtime)

        # Keep only max_files (subtract 1 for the new current log that will be created)
        to_delete_count = len(log_files) - (max_files - 1)
        if to_delete_count > 0:
            for i in range(to_delete_count):
                try:
                    log_files[i].unlink()
                except OSError as e:
                    logging.getLogger(__name__).debug(
                        "Could not remove old rotated log %s: %s",
                        log_files[i],
                        e,
                    )

    except Exception as e:
        # Don't fail pipeline on log rotation issues
        print(f"Warning: Log rotation failed: {e}")


class PipelineLogger(BasicPipelineLogger):
    """Pipeline logger with structured logging support."""

    @classmethod
    def initialize(
        cls,
        log_dir: Optional[Path] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        enable_structured: bool = True,
        log_format: str = "human",
        force: bool = False,
    ) -> None:
        """Initialize logging with structured data support."""
        if cls._initialized and not force:
            return

        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Clear existing handlers to avoid duplicates, but only if we're initializing
        # the root logger for the first time or forcing a reset.
        if not cls._initialized or force:
            root_logger.handlers.clear()

        # Choose formatter based on structured logging preference
        console_formatter: logging.Formatter
        file_formatter: logging.Formatter
        if enable_structured:
            if log_format == "json":
                console_formatter = JSONFormatter()
            else:
                console_formatter = VisualFormatter(
                    "%(asctime)s [%(correlation_id)s:%(step_name)s] %(name)s - %(levelname)s - %(message)s",
                    include_performance=True,
                )
            file_formatter = StructuredFormatter(
                "%(asctime)s [%(correlation_id)s:%(step_name)s] %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            )
        else:
            if log_format == "json":
                console_formatter = JSONFormatter()
            else:
                console_formatter = CorrelationFormatter(
                    "%(asctime)s [%(correlation_id)s:%(step_name)s] %(name)s - %(levelname)s - %(message)s"
                )
            file_formatter = CorrelationFormatter(
                "%(asctime)s [%(correlation_id)s:%(step_name)s] %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            )

        # Add console handler if not already present
        has_console = any(
            isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
            for h in root_logger.handlers
        )
        if not has_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

        # File handler
        if log_dir:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "pipeline.log"

                # Check for existing file handler to same path
                has_file = any(
                    isinstance(h, logging.FileHandler)
                    and h.baseFilename == str(log_file.absolute())
                    for h in root_logger.handlers
                )

                if not has_file:
                    cls._log_file_handler = logging.FileHandler(log_file, mode="w")
                    cls._log_file_handler.setLevel(file_level)
                    cls._log_file_handler.setFormatter(file_formatter)
                    root_logger.addHandler(cls._log_file_handler)
            except OSError as e:
                # If console handler exists, emit error
                record = logging.LogRecord(
                    name="PipelineLogger",
                    level=logging.ERROR,
                    pathname="",
                    lineno=0,
                    msg=f"Failed to setup file logging in {log_dir} ({type(e).__name__}: {e})",
                    args=(),
                    exc_info=None,
                )
                root_logger.handle(record)

        # Silence noisy libraries
        for noisy_lib in [
            "PIL",
            "matplotlib",
            "urllib3",
            "requests",
            "werkzeug",
            "ray",
        ]:
            logging.getLogger(noisy_lib).setLevel(logging.WARNING)

        cls._initialized = True

    @classmethod
    def log_structured(
        cls, logger: logging.Logger, level: int, message: str, **structured_data: Any
    ) -> Any:
        """Log a message with structured data."""
        record = logging.LogRecord(
            name=logger.name,
            level=level,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        record.structured_data = structured_data
        logger.handle(record)

    @classmethod
    def enable_json_logging(cls, log_dir: Path, level: int = logging.DEBUG) -> Any:
        """Enable dedicated JSON-formatted log file."""
        if not cls._initialized:
            # Recovery to initialize first if not ready
            cls.initialize(log_dir=log_dir)
            return

        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            json_log_file = log_dir / "pipeline.jsonl"

            # Create handler
            json_handler = logging.FileHandler(json_log_file, mode="a")
            json_handler.setLevel(level)
            json_handler.setFormatter(JSONFormatter())

            logging.getLogger().addHandler(json_handler)
        except OSError as e:
            print(
                f"Failed to enable JSON logging in {log_dir} ({type(e).__name__}: {e})"
            )

    @classmethod
    @contextmanager
    def timed_operation(
        cls,
        operation_name: str,
        logger: Optional[logging.Logger] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Context manager for timing operations with structured logging."""
        if logger is None:
            logger = cls.get_logger("performance")

        start_time = time.time()
        cls.log_structured(
            logger,
            logging.INFO,
            f"🚀 Starting {operation_name}",
            event_type="operation_start",
            operation=operation_name,
            **(metadata or {}),
        )

        try:
            with performance_tracker.track_operation(operation_name, metadata):
                yield

            duration = time.time() - start_time
            cls.log_structured(
                logger,
                logging.INFO,
                f"✅ Completed {operation_name}",
                event_type="operation_complete",
                operation=operation_name,
                duration_seconds=round(duration, 3),
                **(metadata or {}),
            )

        except Exception as e:
            duration = time.time() - start_time
            cls.log_structured(
                logger,
                logging.ERROR,
                f"❌ Failed {operation_name}: {e}",
                event_type="operation_error",
                operation=operation_name,
                duration_seconds=round(duration, 3),
                error=str(e),
                **(metadata or {}),
            )
            raise


def setup_step_logging(
    step_name: str,
    verbose: bool = False,
    enable_structured: bool = True,
    log_format: str = "human",
) -> logging.Logger:
    """
    Setup logging for a pipeline step with structured data support.

    Args:
        step_name: Name of the pipeline step
        verbose: Whether to enable verbose logging
        enable_structured: Whether to enable structured logging
        log_format: 'human' or 'json' Output format

    Returns:
        Configured logger for the step
    """
    # Initialize enhanced pipeline logger
    PipelineLogger.initialize(
        enable_structured=enable_structured, log_format=log_format
    )

    # Set correlation context
    correlation_id = PipelineLogger.set_correlation_context(
        step_name.replace(".py", "")
    )

    # Configure verbosity
    PipelineLogger.set_verbosity(verbose)

    # Get logger instance
    logger = PipelineLogger.get_logger(step_name)

    # Add step-specific attributes
    logger_any = cast(Any, logger)
    logger_any.step_name = step_name
    logger_any.correlation_id = correlation_id

    return logger


# Global progress tracker
_global_progress_tracker = None


def set_global_progress_tracker(tracker: Any) -> None:
    """Set the global progress tracker used by log_step_* functions."""
    global _global_progress_tracker
    _global_progress_tracker = tracker


# Enhanced logging functions with visual improvements
def log_step_start(
    logger_or_step_name: Any,
    message: Optional[str] = None,
    step_number: Optional[int] = None,
    total_steps: Optional[int] = None,
    **metadata: Any,
) -> Any:
    """Enhanced step start logging with visual progress indicators."""
    if isinstance(logger_or_step_name, str):
        step_name = logger_or_step_name
        message = message or f"Starting {step_name}"
        logger = PipelineLogger.get_logger(step_name)
    else:
        logger = logger_or_step_name
        message = message or "Starting step"

    # Add progress tracking if step numbers provided
    global _global_progress_tracker
    if step_number and total_steps:
        if not _global_progress_tracker:
            _global_progress_tracker = PipelineProgressTracker(total_steps)

        progress_header = _global_progress_tracker.start_step(
            step_number,
            step_name
            if isinstance(logger_or_step_name, str)
            else f"Step {step_number}",
        )
        message = f"{progress_header}\n    {message}"

    PipelineLogger.log_structured(
        logger,
        logging.INFO,
        message,
        event_type="step_start",
        step_number=step_number,
        total_steps=total_steps,
        **metadata,
    )


def log_step_success(
    logger_or_step_name: Any,
    message: Optional[str] = None,
    step_number: Optional[int] = None,
    duration: Optional[float] = None,
    **metadata: Any,
) -> Any:
    """Enhanced step success logging with visual indicators."""
    if isinstance(logger_or_step_name, str):
        step_name = logger_or_step_name
        message = message or f"{step_name} completed successfully"
        logger = PipelineLogger.get_logger(step_name)
    else:
        logger = logger_or_step_name
        message = message or "Step completed successfully"

    # Add completion tracking - only if step hasn't been completed yet
    global _global_progress_tracker
    if step_number and _global_progress_tracker:
        # Check if step was already completed to avoid duplicate completion messages
        step_status = _global_progress_tracker.step_status.get(step_number)
        if step_status != "RUNNING" and step_status is not None:
            # Step already completed, just log the message without completion summary
            logger.debug(
                "Step %s already completed with status %s",
                step_number,
                step_status,
            )
        else:
            # Step not yet completed, complete it and add summary
            completion_summary = _global_progress_tracker.complete_step(
                step_number, "SUCCESS", duration
            )
            message = f"{completion_summary}\n    {message}"

    # Add performance context to log record
    performance_context: dict[Any, Any] = {}
    if duration:
        performance_context["duration"] = duration
    if "memory_mb" in metadata:
        performance_context["memory_mb"] = metadata["memory_mb"]

    record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=f"✅ {message}",
        args=(),
        exc_info=None,
    )
    record.structured_data = {
        **metadata,
        "event_type": "step_success",
        "step_number": step_number,
    }
    record.performance_context = performance_context
    logger.handle(record)


def log_step_warning(
    logger_or_step_name: Any,
    message: Optional[str] = None,
    step_number: Optional[int] = None,
    **metadata: Any,
) -> Any:
    """Enhanced step warning logging with visual indicators."""
    if isinstance(logger_or_step_name, str):
        step_name = logger_or_step_name
        message = message or f"Warning in {step_name}"
        logger = PipelineLogger.get_logger(step_name)
    else:
        logger = logger_or_step_name
        message = message or "Step warning"

    # Add completion tracking for warnings
    global _global_progress_tracker
    if step_number and _global_progress_tracker:
        completion_summary = _global_progress_tracker.complete_step(
            step_number, "SUCCESS_WITH_WARNINGS"
        )
        message = f"{completion_summary}\n    {message}"

    PipelineLogger.log_structured(
        logger,
        logging.WARNING,
        f"⚠️ {message}",
        event_type="step_warning",
        step_number=step_number,
        **metadata,
    )


def log_step_error(
    logger_or_step_name: Any,
    message: Optional[str] = None,
    step_number: Optional[int] = None,
    **metadata: Any,
) -> Any:
    """Enhanced step error logging with visual indicators."""
    if isinstance(logger_or_step_name, str):
        step_name = logger_or_step_name
        message = message or f"Error in {step_name}"
        logger = PipelineLogger.get_logger(step_name)
    else:
        logger = logger_or_step_name
        message = message or "Step error"

    # Add completion tracking for errors
    global _global_progress_tracker
    if step_number and _global_progress_tracker:
        completion_summary = _global_progress_tracker.complete_step(
            step_number, "FAILED"
        )
        message = f"{completion_summary}\n    {message}"

    # Extract event_type if provided, otherwise use default
    event_type = metadata.pop("event_type", "step_error") if metadata else "step_error"

    PipelineLogger.log_structured(
        logger,
        logging.ERROR,
        f"❌ {message}",
        event_type=event_type,
        step_number=step_number,
        **metadata,
    )
    # Return a small dict to satisfy tests expecting non-None
    return {"status": "ERROR", "message": message}


def log_pipeline_summary(logger: logging.Logger, summary_data: Dict[str, Any]) -> Any:
    """Enhanced pipeline summary logging with sophisticated visual formatting."""

    # Extract summary statistics - use performance_summary for accurate counts
    steps = summary_data.get("steps", [])
    perf_summary = summary_data.get("performance_summary", {})

    # Use performance_summary counts which are correctly calculated in main.py
    total_steps = perf_summary.get("total_steps", len(steps))
    successful_steps = perf_summary.get("successful_steps", 0)
    warning_count = perf_summary.get("warnings", 0)
    failed_steps = perf_summary.get("failed_steps", 0)

    # Calculate pure successes vs successes with warnings from step statuses
    pure_successes = len([s for s in steps if s.get("status") == "SUCCESS"])
    successes_with_warnings = len(
        [s for s in steps if s.get("status") == "SUCCESS_WITH_WARNINGS"]
    )
    _failures = len(
        [
            s
            for s in steps
            if "FAILED" in s.get("status", "") or "ERROR" in s.get("status", "")
        ]
    )  # noqa: F841 - computed for summary diagnostics

    # Use total_duration from summary if available, otherwise calculate
    total_duration = summary_data.get("total_duration_seconds", 0)
    if not total_duration:
        total_duration = sum(
            s.get("duration_seconds", 0) for s in steps if s.get("duration_seconds")
        )

    # Determine overall status from summary_data
    overall_status = summary_data.get("overall_status", "UNKNOWN")
    if overall_status == "FAILED":
        status_color = "RED"
        status_icon = "❌"
    elif overall_status in ("SUCCESS_WITH_WARNINGS", "PARTIAL_SUCCESS"):
        status_color = "YELLOW"
        status_icon = "⚠️"
    else:
        status_color = "GREEN"
        status_icon = "✅"

    # Create sophisticated visual summary box
    box_width = 85
    title = f"{status_icon} PIPELINE EXECUTION SUMMARY - {overall_status} {status_icon}"

    # Unicode box drawing characters
    top_border = "╔" + "═" * (box_width - 2) + "╗"
    bottom_border = "╚" + "═" * (box_width - 2) + "╝"
    middle_border = "╠" + "═" * (box_width - 2) + "╣"

    # Create title line with proper centering
    title_padding = (box_width - 2 - len(title)) // 2
    title_line = f"║{' ' * title_padding}{VisualLoggingEnhancer.colorize(title, status_color, True)}{' ' * (box_width - 2 - title_padding - len(title))}║"

    # Create content lines with enhanced formatting
    content_lines: list[Any] = []

    # Basic statistics
    stats_line = f"║ Total Steps: {VisualLoggingEnhancer.colorize(str(total_steps), 'WHITE', True)}"
    content_lines.append(stats_line.ljust(box_width - 1) + "║")

    # Success statistics with colors - show breakdown of pure successes and successes with warnings
    success_text = f"✅ Successful: {VisualLoggingEnhancer.colorize(str(successful_steps), 'GREEN', True)}"
    if successes_with_warnings > 0:
        success_breakdown = (
            f" ({pure_successes} pure, {successes_with_warnings} with warnings)"
        )
        success_text += success_breakdown
    warning_text = f"⚠️ Warnings: {VisualLoggingEnhancer.colorize(str(warning_count), 'YELLOW', True)}"
    failure_text = (
        f"❌ Failed: {VisualLoggingEnhancer.colorize(str(failed_steps), 'RED', True)}"
    )

    # Calculate line lengths accounting for ANSI color codes
    success_display_len = len(f"✅ Successful: {successful_steps}")
    if successes_with_warnings > 0:
        success_display_len += len(
            f" ({pure_successes} pure, {successes_with_warnings} with warnings)"
        )
    warning_display_len = len(f"⚠️ Warnings: {warning_count}")
    failure_display_len = len(f"❌ Failed: {failed_steps}")

    success_line = f"║ {success_text}"
    content_lines.append(
        success_line.ljust(box_width - 1 + (len(success_text) - success_display_len))
        + "║"
    )

    warning_line = f"║ {warning_text}"
    content_lines.append(
        warning_line.ljust(box_width - 1 + (len(warning_text) - warning_display_len))
        + "║"
    )

    failure_line = f"║ {failure_text}"
    content_lines.append(
        failure_line.ljust(box_width - 1 + (len(failure_text) - failure_display_len))
        + "║"
    )

    # Duration with enhanced formatting
    duration_text = (
        f"⏱️ Total Time: {VisualLoggingEnhancer.format_duration(total_duration)}"
    )
    duration_display_len = len("⏱️ Total Time: ") + len(
        VisualLoggingEnhancer.format_duration(total_duration)
        .replace("\033[", "")
        .split("m")[-1]
        .replace("\033[0m", "")
    )
    duration_line = f"║ {duration_text}"
    content_lines.append(
        duration_line.ljust(box_width - 1 + (len(duration_text) - duration_display_len))
        + "║"
    )

    # Add performance insights if available
    if total_duration > 0:
        avg_step_time = total_duration / total_steps
        performance_line = f"║ Average Step Time: {VisualLoggingEnhancer.format_duration(avg_step_time)}"
        content_lines.append(performance_line.ljust(box_width - 1) + "║")

    # Success rate calculation - use successful_steps which includes both pure successes and successes with warnings
    success_rate = (successful_steps / total_steps * 100) if total_steps > 0 else 0
    success_rate_color = (
        "GREEN" if success_rate >= 90 else "YELLOW" if success_rate >= 70 else "RED"
    )
    rate_text = f"📊 Success Rate: {VisualLoggingEnhancer.colorize(f'{success_rate:.1f}%', success_rate_color, True)}"
    rate_display_len = len(f"📊 Success Rate: {success_rate:.1f}%")
    rate_line = f"║ {rate_text}"
    content_lines.append(
        rate_line.ljust(box_width - 1 + (len(rate_text) - rate_display_len)) + "║"
    )

    # Log the sophisticated formatted summary
    logger.info("")
    logger.info(top_border)
    logger.info(title_line)
    logger.info(middle_border)
    for line in content_lines:
        logger.info(line)
    logger.info(bottom_border)
    logger.info("")

    # Add step-by-step breakdown for failures/warnings with detailed warning messages
    if failed_steps > 0 or successes_with_warnings > 0:
        logger.info(
            "🔍 "
            + VisualLoggingEnhancer.colorize("DETAILED STEP ANALYSIS:", "CYAN", True)
        )
        for step in steps:
            status = step.get("status", "UNKNOWN")
            step_name = step.get("script_name", "Unknown")
            _step_description = step.get("description", step_name)  # noqa: F841 - available for enhanced logging

            if "FAILED" in status or "ERROR" in status or "WARNING" in status:
                duration = step.get("duration_seconds", 0)
                duration_str = (
                    f" ({VisualLoggingEnhancer.format_duration(duration)})"
                    if duration
                    else ""
                )

                if "FAILED" in status or "ERROR" in status:
                    icon = "❌"
                    color = "RED"
                else:
                    icon = "⚠️"
                    color = "YELLOW"

                # Extract warning/error messages from step output
                warning_messages: list[Any] = []
                stdout = step.get("stdout", "")
                stderr = step.get("stderr", "")
                combined_output = f"{stdout}\n{stderr}"

                # Look for warning patterns in output
                import re

                warning_patterns: list[Any] = [
                    r"WARNING[:\s]+([^\n]+)",
                    r"⚠️[:\s]+([^\n]+)",
                ]

                for pattern in warning_patterns:
                    matches = re.findall(pattern, combined_output, re.IGNORECASE)
                    warning_messages.extend(
                        matches[:3]
                    )  # Limit to first 3 warnings per step

                # Same log line must not appear twice (e.g. overlapping patterns)
                warning_messages = list(
                    dict.fromkeys(m.strip() for m in warning_messages if m.strip())
                )

                # Also check dependency_warnings
                dep_warnings = step.get("dependency_warnings", [])
                if dep_warnings:
                    warning_messages.extend(
                        dep_warnings[:2]
                    )  # Limit to first 2 dependency warnings

                # Display step with warnings
                logger.info(
                    f"  {icon} {VisualLoggingEnhancer.colorize(step_name, color)}: {status}{duration_str}"
                )
                if warning_messages:
                    for msg in warning_messages[:3]:  # Show max 3 warning messages
                        clean_msg = msg.strip()[:100]  # Limit message length
                        logger.info(
                            f"    └─ {VisualLoggingEnhancer.colorize(clean_msg, 'YELLOW')}"
                        )
        logger.info("")


def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary for the current pipeline run."""
    return cast("dict[str, Any]", performance_tracker.get_summary())


def reset_progress_tracker() -> Any:
    """Reset the global progress tracker for a new pipeline run."""
    global _global_progress_tracker
    _global_progress_tracker = None


def get_progress_summary() -> str:
    """Get a summary of current progress."""
    return "Progress tracking not available in this context"


def setup_correlation_context(
    step_name: str, correlation_id: Optional[str] = None
) -> str:
    """
    Setup correlation context for a pipeline step.

    Args:
        step_name: Name of the pipeline step
        correlation_id: Optional correlation ID (generated if not provided)

    Returns:
        The correlation ID that was set
    """
    return PipelineLogger.set_correlation_context(step_name, correlation_id)


# Module-level function for setting correlation context
# Some modules import set_correlation_context directly from this module.
# This is a thin wrapper that delegates to PipelineLogger.set_correlation_context.
def set_correlation_context(
    step_name: str, correlation_id: Optional[str] = None
) -> str:
    """Set correlation context for logging. Delegates to PipelineLogger.set_correlation_context."""
    return PipelineLogger.set_correlation_context(step_name, correlation_id)
