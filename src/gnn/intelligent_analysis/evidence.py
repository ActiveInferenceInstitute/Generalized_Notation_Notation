"""Pure pipeline evidence analysis, classifications, and recommendations."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple

from gnn.pipeline.schemas import StepStatus

# Constrained type for step flag severity.
FlagType = Literal["none", "yellow", "red", "green"]

# Flag thresholds for per-step analysis. Module-level constants so tests and
# callers can inspect them instead of hardcoding magic numbers.
SLOW_THRESHOLD_SECONDS: float = 60.0
VERY_SLOW_THRESHOLD_SECONDS: float = 120.0
HIGH_MEMORY_THRESHOLD_MB: float = 500.0
CRITICAL_MEMORY_THRESHOLD_MB: float = 1000.0
AVG_MEMORY_MULTIPLIER: float = 3.0
ABOVE_AVG_DURATION_MULTIPLIER: float = 3.0


@dataclass
class StepAnalysis:
    """Detailed analysis of a single pipeline step."""

    step_number: int
    script_name: str
    description: str
    status: StepStatus
    duration_seconds: float
    memory_mb: float
    exit_code: int
    flags: List[str] = field(default_factory=list)
    flag_type: FlagType = "none"
    summary: str = ""
    stdout_snippet: str = ""
    stderr_snippet: str = ""

    def to_dict(self, *, include_snippets: bool = False) -> Dict[str, Any]:
        """Serialize to the JSON payload shape used by ``analysis_data.json``.

        Args:
            include_snippets: Also include ``stdout_snippet`` and
                ``stderr_snippet`` (omitted from the default payload for
                compactness)

        Returns:
            Dictionary representation of this step analysis
        """
        data: dict[str, Any] = {
            "step_number": self.step_number,
            "script_name": self.script_name,
            "description": self.description,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "memory_mb": self.memory_mb,
            "exit_code": self.exit_code,
            "flags": list(self.flags),
            "flag_type": self.flag_type,
            "summary": self.summary,
        }
        if include_snippets:
            data["stdout_snippet"] = self.stdout_snippet
            data["stderr_snippet"] = self.stderr_snippet
        return data


def analyze_pipeline_summary(summary_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze pipeline summary data to extract key insights.

    Args:
        summary_data: Pipeline execution summary dictionary

    Returns:
        Dictionary containing analysis results
    """
    analysis: dict[str, Any] = {
        "overall_status": summary_data.get("overall_status", "UNKNOWN"),
        "total_duration": summary_data.get("total_duration_seconds", 0),
        "step_count": len(summary_data.get("steps", [])),
        "failures": [],
        "warnings": [],
        "performance_metrics": {},
        "health_score": 0.0,
    }

    steps = summary_data.get("steps", [])
    performance = summary_data.get("performance_summary", {})

    # Extract failures
    for step in steps:
        status = str(step.get("status", "UNKNOWN")).upper()
        if status == "FAILED":
            analysis["failures"].append(
                {
                    "step": step.get("script_name"),
                    "error": step.get("stderr", "")[-1000:]
                    if step.get("stderr")
                    else "No error captured",
                    "duration": step.get("duration_seconds", 0),
                    "exit_code": step.get("exit_code", -1),
                }
            )
        elif "WARNING" in status:
            analysis["warnings"].append(
                {
                    "step": step.get("script_name"),
                    "message": step.get("stdout", "")[-500:]
                    if step.get("stdout")
                    else "No output captured",
                }
            )

    # Calculate performance metrics
    analysis["performance_metrics"] = {
        "peak_memory_mb": performance.get("peak_memory_mb", 0),
        "successful_steps": performance.get("successful_steps", 0),
        "failed_steps": performance.get("failed_steps", 0),
        "warning_count": performance.get("warnings", 0),
    }

    # Calculate health score (0-100)
    total_steps = analysis["step_count"]
    if total_steps > 0:
        successful_steps = sum(
            str(step.get("status", "")).upper().startswith("SUCCESS") for step in steps
        )
        success_ratio = successful_steps / total_steps
        warning_penalty = min(len(analysis["warnings"]) * 0.05, 0.2)
        analysis["health_score"] = max(
            0, min(100, (success_ratio - warning_penalty) * 100)
        )

    return analysis


def _classify_step_flags(
    step: Dict[str, Any],
    avg_duration: float,
    avg_memory: float,
) -> Tuple[List[str], FlagType]:
    """Classify one pipeline step into flags and a flag severity.

    Pure function: no I/O, no module state. Thresholds are the module-level
    ``*_THRESHOLD_*`` constants so callers and tests can inspect them.

    Args:
        step: One step record from the pipeline execution summary
        avg_duration: Mean duration across all steps (0 when unknown)
        avg_memory: Mean peak memory across all steps (0 when unknown)

    Returns:
        Tuple of (flag description strings, overall flag severity)
    """
    duration = step.get("duration_seconds", 0)
    memory = step.get("peak_memory_mb", 0)
    status = step.get("status", "UNKNOWN")
    exit_code = step.get("exit_code", 0)

    flags: list[str] = []
    flag_type: FlagType = "none"

    # Determine flags and severity
    if status == "FAILED" or exit_code != 0:
        flag_type = "red"
        flags.append(f"FAILED with exit code {exit_code}")
        if step.get("stderr"):
            flags.append("Error output captured")

    # Performance flags
    if duration > VERY_SLOW_THRESHOLD_SECONDS:
        flags.append(
            f"Very slow: {duration:.1f}s (>{VERY_SLOW_THRESHOLD_SECONDS}s threshold)"
        )
        if flag_type != "red":
            flag_type = "yellow"
    elif duration > SLOW_THRESHOLD_SECONDS:
        flags.append(f"Slow: {duration:.1f}s (>{SLOW_THRESHOLD_SECONDS}s threshold)")
        if flag_type != "red":
            flag_type = "yellow"
    elif avg_duration > 0 and duration > avg_duration * ABOVE_AVG_DURATION_MULTIPLIER:
        flags.append(
            f"Significantly above average: {duration:.1f}s ({duration / avg_duration:.1f}x avg)"
        )
        if flag_type != "red":
            flag_type = "yellow"

    # Memory flags (relative to average, then absolute thresholds)
    if avg_memory > 0 and memory > avg_memory * AVG_MEMORY_MULTIPLIER:
        flags.append(
            f"Memory above average: {memory:.0f}MB ({memory / avg_memory:.1f}x avg)"
        )
        if flag_type not in ("red", "yellow"):
            flag_type = "yellow"

    if memory > CRITICAL_MEMORY_THRESHOLD_MB:
        flags.append(
            f"Critical memory: {memory:.0f}MB (>{CRITICAL_MEMORY_THRESHOLD_MB}MB)"
        )
        if flag_type != "red":
            flag_type = "yellow"
    elif memory > HIGH_MEMORY_THRESHOLD_MB:
        flags.append(f"High memory: {memory:.0f}MB (>{HIGH_MEMORY_THRESHOLD_MB}MB)")
        if flag_type != "red":
            flag_type = "yellow"

    # Warning in status
    if "WARNING" in status:
        flags.append("Step completed with warnings")
        if flag_type == "none":
            flag_type = "yellow"

    # Retry flags
    retry_count = step.get("retry_count", 0)
    if retry_count > 0:
        flags.append(f"Required {retry_count} retries")
        if flag_type == "none":
            flag_type = "yellow"

    # Dependency warnings
    dep_warnings = step.get("dependency_warnings", [])
    if dep_warnings:
        flags.append(f"{len(dep_warnings)} dependency warning(s)")
        if flag_type == "none":
            flag_type = "yellow"

    return flags, flag_type


def _step_result_summary(status: str, duration: float, flag_count: int) -> str:
    """Render the one-line summary for a step analysis."""
    if status == "SUCCESS" and flag_count == 0:
        return f"Completed successfully in {duration:.2f}s"
    if status == "SUCCESS":
        return f"Completed with {flag_count} flag(s) in {duration:.2f}s"
    if status == "FAILED":
        return f"FAILED after {duration:.2f}s"
    return f"Status: {status} ({duration:.2f}s)"


def _group_by_flag_type(
    step_analyses: List[StepAnalysis],
) -> Dict[str, List[StepAnalysis]]:
    """Group step analyses into red/yellow/green buckets."""
    grouped: dict[str, list[StepAnalysis]] = {"red": [], "yellow": [], "green": []}
    for analysis in step_analyses:
        if analysis.flag_type == "red":
            grouped["red"].append(analysis)
        elif analysis.flag_type == "yellow":
            grouped["yellow"].append(analysis)
        else:
            grouped["green"].append(analysis)
    return grouped


def analyze_individual_steps(
    summary_data: Dict[str, Any],
) -> Tuple[List[StepAnalysis], Dict[str, List[StepAnalysis]]]:
    """
    Perform detailed analysis of each pipeline step.

    Args:
        summary_data: Pipeline execution summary dictionary

    Returns:
        Tuple of (list of StepAnalysis objects, dict of flags by type)
    """
    steps = summary_data.get("steps", [])

    durations = [
        s.get("duration_seconds", 0) for s in steps if s.get("duration_seconds")
    ]
    avg_duration = sum(durations) / len(durations) if durations else 0

    memories = [s.get("peak_memory_mb", 0) for s in steps if s.get("peak_memory_mb")]
    avg_memory = sum(memories) / len(memories) if memories else 0

    step_analyses: list[StepAnalysis] = []
    for step in steps:
        duration = step.get("duration_seconds", 0)
        memory = step.get("peak_memory_mb", 0)
        status = step.get("status", "UNKNOWN")
        flags, flag_type = _classify_step_flags(step, avg_duration, avg_memory)
        stdout = step.get("stdout", "")
        stderr = step.get("stderr", "")
        step_analyses.append(
            StepAnalysis(
                step_number=step.get("step_number", 0),
                script_name=step.get("script_name", "unknown"),
                description=step.get("description", ""),
                status=status,
                duration_seconds=duration,
                memory_mb=memory,
                exit_code=step.get("exit_code", 0),
                flags=flags,
                flag_type=flag_type,
                summary=_step_result_summary(status, duration, len(flags)),
                stdout_snippet=_extract_meaningful_snippet(stdout),
                stderr_snippet=_extract_meaningful_snippet(stderr) if stderr else "",
            )
        )

    return step_analyses, _group_by_flag_type(step_analyses)


def _extract_meaningful_snippet(
    output: str, max_lines: int = 5, max_chars: int = 500
) -> str:
    """Extract meaningful snippet from step output."""
    if not output:
        return ""

    lines = output.strip().split("\n")

    # Look for important patterns
    important_patterns: list[str] = [
        "ERROR",
        "WARN",
        "FAIL",
        "SUCCESS",
        "Generated",
        "Processed",
        "Completed",
    ]
    important_lines: list[str] = []

    for line in lines:
        if any(pattern in line.upper() for pattern in important_patterns):
            important_lines.append(line.strip())

    if important_lines:
        snippet = "\n".join(important_lines[:max_lines])
    else:
        # Take last few lines if no important patterns found
        snippet = "\n".join(lines[-max_lines:])

    return snippet[:max_chars]


def identify_bottlenecks(
    summary_data: Dict[str, Any], threshold_seconds: float = 60.0
) -> List[Dict[str, Any]]:
    """
    Identify performance bottlenecks in pipeline execution.

    Args:
        summary_data: Pipeline execution summary dictionary
        threshold_seconds: Duration threshold for flagging slow steps

    Returns:
        List of bottleneck descriptions
    """
    bottlenecks: list[dict[str, Any]] = []
    steps = summary_data.get("steps", [])

    # Calculate average duration
    durations = [
        s.get("duration_seconds", 0) for s in steps if s.get("duration_seconds")
    ]
    avg_duration = sum(durations) / len(durations) if durations else 0

    for step in steps:
        duration = step.get("duration_seconds", 0)
        if duration > threshold_seconds or duration > avg_duration * 2:
            bottlenecks.append(
                {
                    "step": step.get("script_name"),
                    "duration_seconds": duration,
                    "threshold_exceeded": duration > threshold_seconds,
                    "above_average_ratio": duration / avg_duration
                    if avg_duration > 0
                    else 0,
                    "memory_mb": step.get("peak_memory_mb", 0),
                }
            )

    # Sort by duration descending
    bottlenecks.sort(key=lambda x: x["duration_seconds"], reverse=True)
    return bottlenecks


def extract_failure_context(summary_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract detailed context about failures for root cause analysis.

    Args:
        summary_data: Pipeline execution summary dictionary

    Returns:
        List of failure contexts with diagnostic information
    """
    failures: list[dict[str, Any]] = []
    steps = summary_data.get("steps", [])

    for i, step in enumerate(steps):
        if step.get("status") == "FAILED":
            # Get preceding step for context
            preceding_step = steps[i - 1] if i > 0 else None

            failure_context: dict[str, Any] = {
                "step_number": step.get("step_number"),
                "step_name": step.get("script_name"),
                "description": step.get("description"),
                "exit_code": step.get("exit_code", -1),
                "error_output": step.get("stderr", "")[-2000:]
                if step.get("stderr")
                else None,
                "stdout_tail": step.get("stdout", "")[-1000:]
                if step.get("stdout")
                else None,
                "duration": step.get("duration_seconds"),
                "memory_at_failure": step.get("peak_memory_mb"),
                "preceding_step": {
                    "name": preceding_step.get("script_name")
                    if preceding_step
                    else None,
                    "status": preceding_step.get("status") if preceding_step else None,
                }
                if preceding_step
                else None,
                "dependency_warnings": step.get("dependency_warnings", []),
                "prerequisite_check_passed": step.get("prerequisite_check", True),
            }
            failures.append(failure_context)

    return failures


def generate_recommendations(
    analysis: Dict[str, Any],
    bottlenecks: List[Dict[str, Any]],
    flags_by_type: Dict[str, List[StepAnalysis]],
) -> List[str]:
    """
    Generate actionable recommendations based on analysis.

    Args:
        analysis: Analysis results dictionary
        bottlenecks: List of identified bottlenecks
        flags_by_type: Dictionary of steps grouped by flag type

    Returns:
        List of recommendation strings
    """
    recommendations: list[str] = []

    # Red flag recommendations (critical)
    red_flags = flags_by_type.get("red", [])
    if red_flags:
        recommendations.append(
            f"🔴 **CRITICAL**: {len(red_flags)} step(s) have red flags requiring immediate attention."
        )
        for step in red_flags[:3]:  # Top 3
            recommendations.append(
                f"   ↳ **{step.script_name}**: {', '.join(step.flags[:2])}"
            )

    # Yellow flag recommendations (warnings)
    yellow_flags = flags_by_type.get("yellow", [])
    if yellow_flags:
        recommendations.append(
            f"🟡 **WARNINGS**: {len(yellow_flags)} step(s) have yellow flags that should be reviewed."
        )
        for step in yellow_flags[:3]:  # Top 3
            recommendations.append(
                f"   ↳ **{step.script_name}**: {', '.join(step.flags[:2])}"
            )

    # Performance-based recommendations
    if bottlenecks:
        slowest = bottlenecks[0]
        recommendations.append(
            f"⚡ **Performance**: Slowest step is **{slowest['step']}** ({slowest['duration_seconds']:.1f}s). "
            "Consider parallelization or caching."
        )

    # Memory-based recommendations
    peak_memory = analysis["performance_metrics"].get("peak_memory_mb", 0)
    if peak_memory > 2048:
        recommendations.append(
            f"💾 **Memory**: Peak usage {peak_memory:.0f}MB exceeds 2GB. "
            "Consider memory optimization."
        )
    elif peak_memory > 1024:
        recommendations.append(
            f"💾 **Memory**: Peak usage {peak_memory:.0f}MB is elevated. "
            "Monitor for resource constraints."
        )

    # Health score recommendations
    health_score = analysis["health_score"]
    if health_score == 100:
        recommendations.append(
            "✅ **Health**: Pipeline is healthy (100/100). All systems nominal."
        )
    elif health_score >= 90:
        recommendations.append(
            f"✅ **Health**: Pipeline health is good ({health_score:.0f}/100)."
        )
    elif health_score >= 70:
        recommendations.append(
            f"⚠️ **Health**: Pipeline health needs attention ({health_score:.0f}/100)."
        )
    else:
        recommendations.append(
            f"🔴 **Health**: Pipeline health is critical ({health_score:.0f}/100). "
            "Address failures before production use."
        )

    return recommendations
