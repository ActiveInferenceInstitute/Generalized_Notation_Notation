"""Terminal presentation and in-memory progress for pipeline logging.

This owner depends on record formatting, never on logger configuration.
"""

import os
import shutil
import sys
import time
from typing import Any, Dict, Optional

from .formatters import StructuredFormatter


class VisualLoggingEnhancer:
    """Enhanced visual formatting for pipeline logging with progress tracking."""

    # Color codes for terminal output
    COLORS: dict[str, Any] = {
        "RESET": "\033[0m",
        "BOLD": "\033[1m",
        "DIM": "\033[2m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "RED": "\033[91m",
        "BLUE": "\033[94m",
        "MAGENTA": "\033[95m",
        "CYAN": "\033[96m",
        "WHITE": "\033[97m",
        "BG_GREEN": "\033[102m",
        "BG_YELLOW": "\033[103m",
        "BG_RED": "\033[101m",
    }

    # Progress indicators
    PROGRESS_CHARS: list[Any] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    @classmethod
    def supports_color(cls) -> bool:
        """Check if terminal supports color output."""
        if not sys.stdout.isatty():
            return False

        # Check environment variables
        if os.getenv("NO_COLOR"):
            return False
        if os.getenv("FORCE_COLOR"):
            return True

        # Check terminal type
        term = os.getenv("TERM", "").lower()
        if "color" in term or term in ["xterm", "xterm-256color", "screen", "tmux"]:
            return True

        # Check if we have a known terminal with color support
        return shutil.which("tput") is not None

    @classmethod
    def colorize(cls, text: str, color: str, bold: bool = False) -> str:
        """Apply color formatting to text if terminal supports it."""
        if not cls.supports_color():
            return text

        color_code = cls.COLORS.get(color.upper(), "")
        bold_code = cls.COLORS["BOLD"] if bold else ""
        reset_code = cls.COLORS["RESET"]

        return f"{color_code}{bold_code}{text}{reset_code}"

    @classmethod
    def format_step_header(
        cls, step_num: int, total_steps: int, step_name: str, status: str = "RUNNING"
    ) -> str:
        """Create a formatted step header with progress bar."""
        progress = step_num / total_steps
        bar_length = 30
        filled_length = int(bar_length * progress)

        # Create progress bar
        bar = "█" * filled_length + "▒" * (bar_length - filled_length)
        percentage = int(progress * 100)

        # Color code based on status
        if status == "SUCCESS":
            status_colored = cls.colorize(status, "GREEN", True)
            bar_colored = cls.colorize(bar, "GREEN")
        elif status == "FAILED":
            status_colored = cls.colorize(status, "RED", True)
            bar_colored = cls.colorize(bar, "RED")
        elif status == "WARNING":
            status_colored = cls.colorize(status, "YELLOW", True)
            bar_colored = cls.colorize(bar, "YELLOW")
        else:  # RUNNING
            status_colored = cls.colorize(status, "CYAN", True)
            bar_colored = cls.colorize(bar, "CYAN")

        step_info = cls.colorize(f"Step {step_num}/{total_steps}", "WHITE", True)
        step_name_colored = cls.colorize(step_name, "MAGENTA")
        percentage_colored = cls.colorize(f"({percentage}%)", "DIM")

        return f"┌─ {step_info}: {step_name_colored} {status_colored}\n└─ Progress: [{bar_colored}] {percentage_colored}"

    @classmethod
    def format_duration(cls, duration_seconds: float) -> str:
        """Format duration with appropriate units and color coding."""
        if duration_seconds < 1:
            duration_str = f"{duration_seconds * 1000:.0f}ms"
            color = "GREEN"
        elif duration_seconds < 10:
            duration_str = f"{duration_seconds:.2f}s"
            color = "GREEN"
        elif duration_seconds < 60:
            duration_str = f"{duration_seconds:.1f}s"
            color = "YELLOW"
        elif duration_seconds < 300:
            minutes = int(duration_seconds // 60)
            seconds = duration_seconds % 60
            duration_str = f"{minutes}m{seconds:.0f}s"
            color = "YELLOW"
        else:
            minutes = int(duration_seconds // 60)
            hours = minutes // 60
            minutes = minutes % 60
            if hours > 0:
                duration_str = f"{hours}h{minutes}m"
            else:
                duration_str = f"{minutes}m"
            color = "RED"

        return cls.colorize(duration_str, color)

    @classmethod
    def format_memory_usage(cls, memory_mb: float) -> str:
        """Format memory usage with appropriate units and color coding."""
        if memory_mb < 100:
            memory_str = f"{memory_mb:.1f}MB"
            color = "GREEN"
        elif memory_mb < 1000:
            memory_str = f"{memory_mb:.0f}MB"
            color = "YELLOW"
        else:
            memory_gb = memory_mb / 1024
            memory_str = f"{memory_gb:.1f}GB"
            color = "RED"

        return cls.colorize(memory_str, color)


class PipelineProgressTracker:
    """Track pipeline progress across steps with visual indicators."""

    def __init__(self, total_steps: int) -> None:
        """Initialize the instance."""
        self.total_steps = total_steps
        self.current_step = 0
        self.step_status: Dict[int, str] = {}
        self.step_durations: Dict[int, float] = {}
        self.start_time = time.time()

    def start_step(self, step_num: int, step_name: str) -> str:
        """Mark step as started and return formatted header."""
        self.current_step = step_num
        self.step_status[step_num] = "RUNNING"

        return VisualLoggingEnhancer.format_step_header(
            step_num, self.total_steps, step_name, "RUNNING"
        )

    def complete_step(
        self, step_num: int, status: str, duration: Optional[float] = None
    ) -> str:
        """Mark step as completed and return summary."""
        self.step_status[step_num] = status
        if duration:
            self.step_durations[step_num] = duration

        # Calculate completion stats
        completed = len([s for s in self.step_status.values() if s != "RUNNING"])
        success_count = len([s for s in self.step_status.values() if "SUCCESS" in s])

        duration_str = (
            f" in {VisualLoggingEnhancer.format_duration(duration)}" if duration else ""
        )

        if status == "SUCCESS":
            icon = "✅"
            color = "GREEN"
        elif "WARNING" in status:
            icon = "⚠️"
            color = "YELLOW"
        else:
            icon = "❌"
            color = "RED"

        completion_text = VisualLoggingEnhancer.colorize(
            f"{icon} Step {step_num} completed with {status}{duration_str}", color, True
        )

        progress_text = VisualLoggingEnhancer.colorize(
            f"Progress: {completed}/{self.total_steps} steps ({success_count} successful)",
            "CYAN",
        )

        return f"{completion_text}\n{progress_text}"

    def get_overall_progress(self) -> str:
        """Get overall pipeline progress summary."""
        completed = len([s for s in self.step_status.values() if s != "RUNNING"])
        success_count = len([s for s in self.step_status.values() if "SUCCESS" in s])
        warning_count = len([s for s in self.step_status.values() if "WARNING" in s])
        failed_count = len(
            [s for s in self.step_status.values() if "FAILED" in s or "ERROR" in s]
        )

        elapsed = time.time() - self.start_time

        progress_bar = VisualLoggingEnhancer.format_step_header(
            completed,
            self.total_steps,
            "Overall Progress",
            "SUCCESS"
            if completed == self.total_steps and failed_count == 0
            else "RUNNING",
        )

        stats: list[Any] = [
            f"✅ Success: {success_count}",
            f"⚠️ Warnings: {warning_count}",
            f"❌ Failed: {failed_count}",
            f"⏱️ Elapsed: {VisualLoggingEnhancer.format_duration(elapsed)}",
        ]

        return f"{progress_bar}\nStats: {' | '.join(stats)}"


class VisualFormatter(StructuredFormatter):
    """Formatter with visual improvements and performance context."""

    def __init__(
        self,
        format_string: Any,
        include_performance: Any = True,
        use_colors: Any = True,
    ) -> None:
        """Initialize the instance."""
        super().__init__(format_string)
        self.include_performance = include_performance
        self.use_colors = use_colors and VisualLoggingEnhancer.supports_color()

    def format(self, record: Any) -> Any:
        # Get base formatted message
        """Provide format behavior."""
        formatted = super().format(record)

        # Add color coding for log levels and emojis
        if self.use_colors:
            if "🚀" in formatted:
                formatted = formatted.replace(
                    "🚀", VisualLoggingEnhancer.colorize("🚀", "BLUE", True)
                )
            elif "✅" in formatted:
                formatted = formatted.replace(
                    "✅", VisualLoggingEnhancer.colorize("✅", "GREEN", True)
                )
            elif "⚠️" in formatted:
                formatted = formatted.replace(
                    "⚠️", VisualLoggingEnhancer.colorize("⚠️", "YELLOW", True)
                )
            elif "❌" in formatted:
                formatted = formatted.replace(
                    "❌", VisualLoggingEnhancer.colorize("❌", "RED", True)
                )

        # Add performance context if available
        if self.include_performance and hasattr(record, "performance_context"):
            perf_data = record.performance_context
            perf_parts: list[Any] = []

            if "duration" in perf_data:
                duration_str = VisualLoggingEnhancer.format_duration(
                    perf_data["duration"]
                )
                perf_parts.append(f"⏱️ {duration_str}")

            if "memory_mb" in perf_data:
                memory_str = VisualLoggingEnhancer.format_memory_usage(
                    perf_data["memory_mb"]
                )
                perf_parts.append(f"🧠 {memory_str}")

            if perf_parts:
                formatted += f" [{' | '.join(perf_parts)}]"

        return formatted
