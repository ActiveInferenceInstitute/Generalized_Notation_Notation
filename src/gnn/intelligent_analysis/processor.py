#!/usr/bin/env python3
"""
Intelligent Analysis processor module for GNN pipeline analysis.

This module provides the main processing logic for intelligent pipeline analysis,
including LLM-powered insights and executive report generation with per-step
breakdowns, yellow/red flag detection, and actionable recommendations.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gnn.utils.observability.structured_logging import (
    log_step_error,
    log_step_start,
    log_step_success,
)

from .analyzer import pick_evidence_timestamp
from .evidence import (
    ABOVE_AVG_DURATION_MULTIPLIER as ABOVE_AVG_DURATION_MULTIPLIER,
)
from .evidence import (
    AVG_MEMORY_MULTIPLIER as AVG_MEMORY_MULTIPLIER,
)
from .evidence import (
    CRITICAL_MEMORY_THRESHOLD_MB as CRITICAL_MEMORY_THRESHOLD_MB,
)
from .evidence import (
    HIGH_MEMORY_THRESHOLD_MB as HIGH_MEMORY_THRESHOLD_MB,
)
from .evidence import (
    SLOW_THRESHOLD_SECONDS as SLOW_THRESHOLD_SECONDS,
)
from .evidence import (
    VERY_SLOW_THRESHOLD_SECONDS as VERY_SLOW_THRESHOLD_SECONDS,
)
from .evidence import (
    FlagType as FlagType,
)
from .evidence import (
    StepAnalysis as StepAnalysis,
)
from .evidence import (
    _classify_step_flags as _classify_step_flags,
)
from .evidence import (
    _extract_meaningful_snippet as _extract_meaningful_snippet,
)
from .evidence import (
    _group_by_flag_type as _group_by_flag_type,
)
from .evidence import (
    _step_result_summary as _step_result_summary,
)
from .evidence import (
    analyze_individual_steps as analyze_individual_steps,
)
from .evidence import (
    analyze_pipeline_summary as analyze_pipeline_summary,
)
from .evidence import (
    extract_failure_context as extract_failure_context,
)
from .evidence import (
    generate_recommendations as generate_recommendations,
)
from .evidence import (
    identify_bottlenecks as identify_bottlenecks,
)


def resolve_pipeline_summary_path(output_dir: Path) -> Path:
    """Locate the pipeline execution summary for ``output_dir``.

    Prefers ``output_dir/00_pipeline_summary/pipeline_execution_summary.json``
    and falls back to the same path under ``output_dir.parent`` (the layout
    used when step 24 writes into a per-step subdirectory).

    Args:
        output_dir: Pipeline output root directory

    Returns:
        Path to the summary JSON (which may not exist yet)
    """
    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    if context is not None:
        summary_path = (
            Path(context.output_root) / "00_pipeline_summary" / "current_summary.json"
        )
        if not summary_path.is_file():
            raise ValueError("Current-run pipeline summary snapshot is unavailable")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("run_id") != context.run_id:
            raise ValueError("Pipeline summary snapshot belongs to another run")
        if summary.get("snapshot_path"):
            snapshot_path = Path(summary["snapshot_path"])
            if not snapshot_path.resolve().is_relative_to(
                Path(context.output_root).resolve()
            ):
                raise ValueError("Pipeline snapshot escapes its output root")
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            if snapshot.get("run_id") != context.run_id:
                raise ValueError("Pipeline snapshot belongs to another run")
            return snapshot_path
        return summary_path
    summary_path = (
        output_dir / "00_pipeline_summary" / "pipeline_execution_summary.json"
    )
    if not summary_path.exists():
        summary_path = (
            output_dir.parent
            / "00_pipeline_summary"
            / "pipeline_execution_summary.json"
        )
    return summary_path


def resolve_analysis_output_dir(output_dir: Path) -> Path:
    """Resolve the directory that step-24 artifacts are written to.

    When ``output_dir`` is already the step-24 output directory, artifacts go
    directly into it; otherwise a ``24_intelligent_analysis_output``
    subdirectory is used.

    Args:
        output_dir: Pipeline output root directory

    Returns:
        The resolved artifact directory (not created)
    """
    if output_dir.name == "24_intelligent_analysis_output":
        return output_dir
    return output_dir / "24_intelligent_analysis_output"


def _is_complete_llm_report(content: str) -> bool:
    """Require the complete requested report topology before labeling LLM output."""
    required_headings = (
        "### Executive Summary",
        "### Red Flags (Critical Issues)",
        "### Yellow Flags (Warnings)",
        "### Root Cause Analysis",
        "### Optimization Opportunities",
        "### Action Items",
    )
    return bool(content.strip()) and all(
        heading in content for heading in required_headings
    )


async def _run_llm_analysis(
    context: Dict[str, Any],
    step_analyses: List[StepAnalysis],
    flags_by_type: Dict[str, List[StepAnalysis]],
    logger: logging.Logger,
    analysis_model: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Tuple[str, str]:
    """
    Run LLM-powered analysis on pipeline context.

    Args:
        context: Analysis context dictionary
        step_analyses: List of per-step analysis objects
        flags_by_type: Steps grouped by flag type
        logger: Logger instance
        analysis_model: Optional model tag (CLI ``--analysis-model``); else ``OLLAMA_MODEL``, else ``llm.defaults.DEFAULT_OLLAMA_MODEL``.
        cache_dir: Optional directory for the content-addressed LLM response cache (LLMCache). None disables caching.

    Returns:
        Analysis markdown and its source (``llm`` or ``rule_based``).
    """
    try:
        from gnn.llm.llm_processor import initialize_global_processor
        from gnn.llm.providers.base_provider import LLMMessage

        processor = await initialize_global_processor()
    except Exception as e:
        logger.warning(f"Failed to initialize LLM processor: {e}")
        return (
            _generate_rule_based_summary(context, step_analyses, flags_by_type),
            "rule_based",
        )

    if not processor:
        return (
            _generate_rule_based_summary(context, step_analyses, flags_by_type),
            "rule_based",
        )

    cache: Any = None
    if cache_dir is not None:
        try:
            from gnn.llm.cache import LLMCache

            cache = LLMCache(cache_dir=cache_dir)
        except Exception as e:  # pragma: no cover - cache is best-effort
            logger.debug(f"LLM analysis cache unavailable: {e}")

    # Construct comprehensive prompt
    status_emoji = (
        "✅"
        if context["overall_status"] == "SUCCESS"
        else "❌"
        if context["overall_status"] == "FAILED"
        else "⚠️"
    )
    duration = context.get("total_duration", 0)
    peak_memory = context.get("performance_metrics", {}).get("peak_memory_mb", 0)

    # Build step summaries for LLM
    step_summaries: list[str] = []
    for sa in step_analyses:
        flag_indicator = (
            "🔴"
            if sa.flag_type == "red"
            else "🟡"
            if sa.flag_type == "yellow"
            else "✅"
        )
        step_summaries.append(
            f"{flag_indicator} **{sa.script_name}** ({sa.description}): {sa.summary}"
            + (f" | Flags: {', '.join(sa.flags)}" if sa.flags else "")
        )

    red_count = len(flags_by_type.get("red", []))
    yellow_count = len(flags_by_type.get("yellow", []))

    prompt = f"""You are an expert DevOps analyst reviewing a pipeline execution report. Provide a comprehensive but concise analysis.

## Pipeline Overview
- **Status**: {status_emoji} {context["overall_status"]}
- **Duration**: {duration:.2f}s
- **Peak Memory**: {peak_memory:.2f} MB
- **Health Score**: {context.get("health_score", 0):.1f}/100
- **Total Steps**: {len(step_analyses)}
- **Red Flags**: {red_count}
- **Yellow Flags**: {yellow_count}

## Per-Step Results
{chr(10).join(step_summaries)}

## Failures
{json.dumps(context.get("failures", []), indent=2) if context.get("failures") else "None"}

## Warnings
{json.dumps(context.get("warnings", []), indent=2) if context.get("warnings") else "None"}

---

IMPORTANT STRONGLY ENFORCED RULES:
1. You are writing a BUSINESS REPORT in pure Markdown prose.
2. Output ONLY the requested headings and their content. No greeting, no scripts, no meta-commentary.
3. Use the exact headings listed below and fill them in with analytical paragraphs and bullet points.

Please provide analysis in EXACTLY this format:

### Executive Summary
[2-3 sentences summarizing the overall pipeline health and key findings]

### Red Flags (Critical Issues)
[List any critical issues that need immediate attention, or "None" if all clear]

### Yellow Flags (Warnings)
[List concerning patterns or performance issues, or "None" if all clear]

### Root Cause Analysis
[If failures exist, explain likely causes. If successful, note any concerning patterns]

### Optimization Opportunities
[Specific, actionable suggestions for improving pipeline performance]

### Action Items
[Prioritized bullet list of what the user should do next]
"""

    try:
        from gnn.llm.defaults import DEFAULT_OLLAMA_MODEL

        model_name = analysis_model or os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
        messages: list[LLMMessage] = [LLMMessage(role="user", content=prompt)]

        cache_content = json.dumps(context, sort_keys=True)
        if cache is not None:
            cached = cache.get(cache_content, model_name, prompt)
            if cached is not None:
                logger.info("⚡ Cache HIT for step-24 LLM analysis")
                return cached, "llm"

        response = await processor.get_response(
            messages=messages, model_name=model_name, max_tokens=2500
        )
        content = response.content.strip()

        # Reject code and incomplete prose rather than publishing it as a
        # complete AI analysis.
        if (
            "```python" in content.lower()
            or "def " in content
            or not _is_complete_llm_report(content)
        ):
            logger.warning(
                "LLM output did not satisfy the report contract; using rule-based summary."
            )
            return (
                _generate_rule_based_summary(context, step_analyses, flags_by_type),
                "rule_based",
            )

        if cache is not None:
            cache.put(cache_content, model_name, prompt, content)

        return content, "llm"
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        return (
            _generate_rule_based_summary(context, step_analyses, flags_by_type),
            "rule_based",
        )


def _generate_rule_based_summary(
    context: Dict[str, Any],
    step_analyses: List[StepAnalysis],
    flags_by_type: Dict[str, List[StepAnalysis]],
) -> str:
    """Generate a rule-based summary when LLM is unavailable."""

    red_flags = flags_by_type.get("red", [])
    yellow_flags = flags_by_type.get("yellow", [])

    parts: list[str] = []

    # Executive Summary
    parts.append("### Executive Summary\n")
    if context["overall_status"] == "SUCCESS" and not red_flags:
        parts.append(
            f"Pipeline completed successfully with a health score of {context['health_score']:.0f}/100. "
        )
        if yellow_flags:
            parts.append(
                f"There are {len(yellow_flags)} yellow flag(s) to review for optimization opportunities."
            )
        else:
            parts.append("All systems nominal with no flags raised.")
    elif context["overall_status"] == "SUCCESS" and red_flags:
        parts.append(
            f"Pipeline completed but {len(red_flags)} critical issue(s) detected. Review required."
        )
    else:
        parts.append(
            f"Pipeline {context['overall_status']} with {len(red_flags)} critical and {len(yellow_flags)} warning flags."
        )
    parts.append("\n\n")

    # Red Flags
    parts.append("### Red Flags (Critical Issues)\n")
    if red_flags:
        for step in red_flags:
            parts.append(f"- **{step.script_name}**: {', '.join(step.flags)}\n")
    else:
        parts.append("None - No critical issues detected.\n")
    parts.append("\n")

    # Yellow Flags
    parts.append("### Yellow Flags (Warnings)\n")
    if yellow_flags:
        for step in yellow_flags[:5]:  # Top 5
            parts.append(f"- **{step.script_name}**: {', '.join(step.flags)}\n")
        if len(yellow_flags) > 5:
            parts.append(f"- ...and {len(yellow_flags) - 5} more\n")
    else:
        parts.append("None - No warnings detected.\n")
    parts.append("\n")

    # Root Cause Analysis
    parts.append("### Root Cause Analysis\n")
    failures = context.get("failures", [])
    if failures:
        for failure in failures[:5]:
            step = failure.get("step", "unknown step")
            error = str(failure.get("error", "No error captured")).strip()
            parts.append(f"- **{step}**: {error[:300]}\n")
    else:
        parts.append("No failed steps were recorded in the execution receipt.\n")
    parts.append("\n")

    # Optimization Opportunities
    parts.append("### Optimization Opportunities\n")
    slow_steps = [step for step in step_analyses if step.duration_seconds > 60.0]
    if slow_steps:
        for step in slow_steps[:5]:
            parts.append(
                f"- Profile **{step.script_name}**, recorded at "
                f"{step.duration_seconds:.2f}s, before selecting an optimization.\n"
            )
    else:
        parts.append("No step exceeded the 60-second review threshold.\n")
    parts.append("\n")

    # Action Items
    parts.append("### Action Items\n")
    if red_flags:
        parts.append("1. **Immediate**: Address red flag issues before proceeding\n")
    if yellow_flags:
        parts.append(
            f"{'2' if red_flags else '1'}. **Review**: Investigate yellow flag warnings\n"
        )
    if context.get("performance_metrics", {}).get("peak_memory_mb", 0) > 1024:
        parts.append(
            f"{'3' if red_flags and yellow_flags else '2' if red_flags or yellow_flags else '1'}. **Monitor**: Track memory usage in production\n"
        )
    if not red_flags and not yellow_flags:
        parts.append("1. Continue normal operations - pipeline is healthy\n")
        parts.append("2. Consider setting up monitoring for regression detection\n")

    return "".join(parts)


@dataclass
class FullAnalysisResult:
    """Complete rule-based analysis of one pipeline execution.

    Groups every analysis artifact that ``process_intelligent_analysis``
    computes so programmatic callers get the whole picture from one pure
    call (no I/O, no LLM).
    """

    analysis: Dict[str, Any]
    step_analyses: List[StepAnalysis]
    flags_by_type: Dict[str, List[StepAnalysis]]
    bottlenecks: List[Dict[str, Any]]
    failures: List[Dict[str, Any]]
    recommendations: List[str]
    recovery_plan: List[str]
    red_count: int = field(init=False)
    yellow_count: int = field(init=False)
    green_count: int = field(init=False)

    def __post_init__(self) -> None:
        """Derive flag counts from the grouped steps."""
        self.red_count = len(self.flags_by_type.get("red", []))
        self.yellow_count = len(self.flags_by_type.get("yellow", []))
        self.green_count = len(self.flags_by_type.get("green", []))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the ``analysis_data.json`` payload shape.

        Returns:
            Dictionary with flag counts and every analysis component; the
            ``analysis`` and ``failures`` entries carry the report's
            evidence timestamp under the same keys the file contract uses.
        """
        return {
            "analysis": self.analysis,
            "step_analyses": [sa.to_dict() for sa in self.step_analyses],
            "flags_summary": {
                "red_count": self.red_count,
                "yellow_count": self.yellow_count,
                "green_count": self.green_count,
            },
            "bottlenecks": self.bottlenecks,
            "failures": self.failures,
            "recommendations": self.recommendations,
            "recovery_plan": self.recovery_plan,
        }


def compute_full_analysis(
    summary_data: Dict[str, Any],
    bottleneck_threshold: float = 60.0,
) -> FullAnalysisResult:
    """Run the complete rule-based analysis over one pipeline summary.

    Pure function: no filesystem access, no logging, no LLM calls. This is
    the deterministic core of ``process_intelligent_analysis`` steps 2-6;
    callers who need the analysis without report-writing side effects
    should prefer this entry point.

    Args:
        summary_data: Pipeline execution summary dictionary
        bottleneck_threshold: Duration threshold in seconds for
            bottleneck detection (matches the CLI ``--bottleneck-threshold``)

    Returns:
        A :class:`FullAnalysisResult` with every analysis component
    """
    analysis = analyze_pipeline_summary(summary_data)
    step_analyses, flags_by_type = analyze_individual_steps(summary_data)
    bottlenecks = identify_bottlenecks(
        summary_data, threshold_seconds=bottleneck_threshold
    )
    failures = extract_failure_context(summary_data)
    recommendations = generate_recommendations(analysis, bottlenecks, flags_by_type)
    recovery_plan = generate_recovery_plan(analysis)
    return FullAnalysisResult(
        analysis=analysis,
        step_analyses=step_analyses,
        flags_by_type=flags_by_type,
        bottlenecks=bottlenecks,
        failures=failures,
        recommendations=recommendations,
        recovery_plan=recovery_plan,
    )


def generate_recovery_plan(context: Dict[str, Any]) -> List[str]:
    """Generate a heuristic-based programmatic recovery plan for failed pipeline components."""
    plan: list[str] = []
    health = context.get("health_score", 100)

    if health >= 100:
        return plan

    failures = context.get("failures", [])
    for failure in failures:
        script = (
            failure.get("script_name")
            or failure.get("step")
            or failure.get("step_name")
            or "unknown step"
        )
        error_msg = str(
            failure.get("error") or failure.get("error_output") or ""
        ).lower()
        # Heuristics based on observed failure modes
        if "timeout" in error_msg:
            plan.append(
                f"Review the recorded timeout for `{script}` and its documented "
                "timeout configuration before retrying."
            )
        elif "memory" in error_msg or "allocation" in error_msg:
            plan.append(
                f"Profile the recorded allocation failure in `{script}` and reduce "
                "the verified workload or memory use before retrying."
            )
        elif "llm" in script.lower() or "connection" in error_msg:
            plan.append(
                f"Verify the configured provider for `{script}` is reachable, then "
                "retry using only documented arguments."
            )
        else:
            plan.append(
                f"Inspect the captured output for `{script}` and validate its "
                "prerequisites before retrying."
            )

    if health < 70 and len(failures) > 2:
        plan.insert(
            0,
            "CRITICAL: Multiple failures were recorded. Start with the first "
            "failure because later failures may be consequential.",
        )

    return plan


def generate_executive_report(
    analysis: Dict[str, Any],
    bottlenecks: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    recommendations: List[str],
    step_analyses: List[StepAnalysis],
    flags_by_type: Dict[str, List[StepAnalysis]],
    llm_analysis: Optional[str] = None,
    summary_data: Optional[Dict[str, Any]] = None,
    analysis_source: str = "rule_based",
) -> str:
    """
    Generate a comprehensive executive report with per-step analysis.

    Args:
        analysis: Analysis results
        bottlenecks: Identified bottlenecks
        failures: Failure contexts
        recommendations: Generated recommendations
        step_analyses: Per-step analysis objects
        flags_by_type: Steps grouped by flag type
        llm_analysis: Optional LLM-generated analysis
        summary_data: Original pipeline summary data

    Returns:
        Markdown formatted executive report
    """
    report_parts: list[str] = []

    # Header
    status = analysis["overall_status"]
    status_emoji = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⚠️"

    red_count = len(flags_by_type.get("red", []))
    yellow_count = len(flags_by_type.get("yellow", []))
    green_count = len(flags_by_type.get("green", []))

    report_parts.append("# Pipeline Intelligent Analysis Report\n")
    report_parts.append(
        f"**Evidence As Of**: {pick_evidence_timestamp(summary_data or {})}\n"
    )
    report_parts.append(f"**Status**: {status_emoji} {status}\n")
    report_parts.append(f"**Health Score**: {analysis['health_score']:.1f}/100\n")
    report_parts.append("")

    # Quick Stats Box
    report_parts.append("## Quick Overview\n")
    report_parts.append("| Metric | Value |")
    report_parts.append("|--------|-------|")
    report_parts.append(f"| Total Steps | {analysis['step_count']} |")
    report_parts.append(f"| Duration | {analysis['total_duration']:.2f}s |")
    report_parts.append(
        f"| Peak Memory | {analysis['performance_metrics'].get('peak_memory_mb', 0):.1f} MB |"
    )
    report_parts.append(f"| 🔴 Red Flags | {red_count} |")
    report_parts.append(f"| 🟡 Yellow Flags | {yellow_count} |")
    report_parts.append(f"| ✅ Green (Clean) | {green_count} |")
    report_parts.append("")

    # AI-Powered Analysis (if available)
    if llm_analysis and analysis_source == "llm":
        report_parts.append("## AI-Powered Analysis\n")
        report_parts.append(llm_analysis)
        report_parts.append("")
    elif llm_analysis:
        # Show rule-based analysis
        report_parts.append("## Analysis Summary\n")
        report_parts.append(llm_analysis)
        report_parts.append("")

    # Red Flags Section
    red_flags = flags_by_type.get("red", [])
    if red_flags:
        report_parts.append("## 🔴 Red Flags (Critical)\n")
        for step in red_flags:
            report_parts.append(f"### {step.script_name}\n")
            report_parts.append(f"- **Status**: {step.status}")
            report_parts.append(f"- **Exit Code**: {step.exit_code}")
            report_parts.append(f"- **Duration**: {step.duration_seconds:.2f}s")
            report_parts.append(f"- **Issues**: {', '.join(step.flags)}")
            if step.stderr_snippet:
                report_parts.append(
                    f"\n**Error Output**:\n```\n{step.stderr_snippet}\n```"
                )
            report_parts.append("")

    # Yellow Flags Section
    yellow_flags = flags_by_type.get("yellow", [])
    if yellow_flags:
        report_parts.append("## 🟡 Yellow Flags (Warnings)\n")
        report_parts.append("| Step | Duration | Memory | Issues |")
        report_parts.append("|------|----------|--------|--------|")
        for step in yellow_flags:
            issues = "; ".join(step.flags[:2]) if step.flags else "N/A"
            report_parts.append(
                f"| {step.script_name} | {step.duration_seconds:.1f}s | "
                f"{step.memory_mb:.0f}MB | {issues} |"
            )
        report_parts.append("")

    # Per-Step Breakdown
    report_parts.append("## Per-Step Execution Details\n")
    report_parts.append("| # | Step | Status | Duration | Memory | Flags |")
    report_parts.append("|---|------|--------|----------|--------|-------|")

    for step in step_analyses:
        flag_emoji = (
            "🔴"
            if step.flag_type == "red"
            else "🟡"
            if step.flag_type == "yellow"
            else "✅"
        )
        status_display = f"{flag_emoji} {step.status}"
        flags_display = len(step.flags) if step.flags else "-"
        report_parts.append(
            f"| {step.step_number} | {step.script_name} | {status_display} | "
            f"{step.duration_seconds:.2f}s | {step.memory_mb:.0f}MB | {flags_display} |"
        )
    report_parts.append("")

    # Detailed Step Output (for flagged steps only)
    flagged_steps = [s for s in step_analyses if s.flag_type != "none"]
    if flagged_steps:
        report_parts.append("## Detailed Step Output (Flagged Steps)\n")
        for step in flagged_steps:
            report_parts.append(f"### {step.script_name}\n")
            report_parts.append(f"**{step.description}**\n")
            report_parts.append(f"- Status: {step.status}")
            report_parts.append(f"- Duration: {step.duration_seconds:.2f}s")
            report_parts.append(f"- Memory: {step.memory_mb:.0f}MB")
            if step.flags:
                report_parts.append(f"- Flags: {', '.join(step.flags)}")
            if step.stdout_snippet:
                report_parts.append(
                    f"\n**Output Snippet**:\n```\n{step.stdout_snippet}\n```"
                )
            if step.stderr_snippet:
                report_parts.append(
                    f"\n**Error Output**:\n```\n{step.stderr_snippet}\n```"
                )
            report_parts.append("")

    # Performance Bottlenecks
    if bottlenecks:
        report_parts.append("## Performance Bottlenecks\n")
        report_parts.append("| Step | Duration (s) | Memory (MB) | Above Avg Ratio |")
        report_parts.append("|------|-------------|-------------|-----------------|")
        for bn in bottlenecks[:5]:  # Top 5
            report_parts.append(
                f"| {bn['step']} | {bn['duration_seconds']:.1f} | "
                f"{bn.get('memory_mb', 0):.0f} | {bn['above_average_ratio']:.1f}x |"
            )
        report_parts.append("")

    # Recommendations Section
    report_parts.append("## Recommendations\n")
    for rec in recommendations:
        report_parts.append(f"- {rec}")
    report_parts.append("")

    # Autonomous Execution Recovery Section
    if summary_data:
        recovery_plan = generate_recovery_plan(analysis)
        if recovery_plan:
            report_parts.append("## 🛡️ Execution Recovery Plan\n")
            report_parts.append(
                "The following evidence-based recovery checks are recommended:\n"
            )
            for plan_item in recovery_plan:
                report_parts.append(f"- {plan_item}")
            report_parts.append("")

    # Pipeline Arguments (for context)
    if summary_data and summary_data.get("arguments"):
        args = summary_data["arguments"]
        report_parts.append("## Pipeline Configuration\n")
        report_parts.append("```json")
        # Show relevant config only
        relevant_args = {
            k: v
            for k, v in args.items()
            if k
            in [
                "target_dir",
                "output_dir",
                "verbose",
                "strict",
                "only_steps",
                "skip_steps",
                "frameworks",
            ]
        }
        report_parts.append(json.dumps(relevant_args, indent=2))
        report_parts.append("```")
        report_parts.append("")

    return "\n".join(report_parts)


def process_intelligent_analysis(
    target_dir: Path, output_dir: Path, logger: logging.Logger, **kwargs: Any
) -> bool:
    """
    Perform intelligent analysis of the pipeline execution.

    Args:
        target_dir: Directory containing input files (not used directly here)
        output_dir: Output directory for generated artifacts
        logger: Logger instance
        **kwargs: Additional arguments

    Returns:
        True if analysis succeeded, False if analysis itself failed
    """
    log_step_start(logger, "Intelligent Pipeline Analysis")

    # 1. Locate Pipeline Summary
    # Note: When running as part of the pipeline, the summary file is written
    # AFTER all steps complete. Since this step (24) runs before the pipeline
    # finishes, the summary from the CURRENT run won't exist yet.
    # We look for the most recent PREVIOUS run's summary instead, or gracefully
    # handle its absence.
    summary_path = resolve_pipeline_summary_path(output_dir)
    if not summary_path.exists():
        # Brief wait in case the file is being written concurrently
        logger.info("Pipeline summary not found yet, waiting briefly...")
        time.sleep(2)

    if not summary_path.exists():
        # Generate a partial report noting the summary was unavailable.
        logger.warning(
            f"Pipeline summary not found at {summary_path} — this is expected when "
            "running as part of the pipeline (summary is written after all steps complete). "
            "Generating partial analysis from available outputs."
        )
        analysis_output_dir = resolve_analysis_output_dir(output_dir)
        analysis_output_dir.mkdir(parents=True, exist_ok=True)

        partial_report = (
            "# Pipeline Intelligent Analysis Report\n\n"
            "**Evidence As Of**: unavailable\n\n"
            "## Note\n\n"
            "The pipeline execution summary was not available at analysis time. "
            "This typically occurs because this step runs before the pipeline "
            "writes its final summary. Run this analysis again after a completed "
            "pipeline execution summary is available.\n"
        )
        report_path = analysis_output_dir / "intelligent_analysis_report.md"
        with open(report_path, "w") as f:
            f.write(partial_report)
        logger.info(f"Partial report saved to {report_path}")
        log_step_success(
            logger, "Intelligent analysis completed (partial — summary unavailable)"
        )
        return True

    try:
        with open(summary_path, "r") as f:
            summary_data = json.load(f)
    except Exception as e:
        log_step_error(logger, f"Failed to load pipeline summary: {e}")
        return False
    evidence_timestamp = pick_evidence_timestamp(summary_data)
    evidence_timestamp_source = (
        "pipeline_execution_summary"
        if evidence_timestamp != "unavailable"
        else "unavailable"
    )

    # 2-6. Full rule-based analysis (single deterministic core)
    logger.info("Analyzing pipeline execution data...")
    full = compute_full_analysis(
        summary_data,
        bottleneck_threshold=float(kwargs.get("bottleneck_threshold", 60.0)),
    )
    analysis = full.analysis

    logger.info(
        f"Analysis complete: Status={analysis['overall_status']}, "
        f"Failures={len(analysis['failures'])}, Health={analysis['health_score']:.1f}"
    )

    logger.info("Performing per-step analysis...")
    step_analyses = full.step_analyses
    flags_by_type = full.flags_by_type
    bottlenecks = full.bottlenecks
    failures = full.failures
    recommendations = full.recommendations
    recovery_plan = full.recovery_plan
    red_count = full.red_count
    yellow_count = full.yellow_count
    logger.info(f"Step analysis: {red_count} red flags, {yellow_count} yellow flags")
    if bottlenecks:
        logger.info(f"Identified {len(bottlenecks)} performance bottlenecks")

    # 7. Run LLM Analysis
    llm_analysis = None
    analysis_source = "rule_based"
    if kwargs.get("skip_llm"):
        logger.info("LLM analysis skipped by configuration; using rule-based summary")
        llm_analysis = _generate_rule_based_summary(
            analysis, step_analyses, flags_by_type
        )
    else:
        try:
            llm_analysis, analysis_source = asyncio.run(
                _run_llm_analysis(
                    analysis,
                    step_analyses,
                    flags_by_type,
                    logger,
                    analysis_model=kwargs.get("analysis_model"),
                    cache_dir=resolve_analysis_output_dir(output_dir) / ".cache",
                )
            )
            logger.info("Analysis summary completed using %s evidence", analysis_source)
        except Exception as e:
            logger.warning(f"LLM analysis skipped: {e}")
            llm_analysis = _generate_rule_based_summary(
                analysis, step_analyses, flags_by_type
            )

    # 8. Generate Executive Report
    report_content = generate_executive_report(
        analysis,
        bottlenecks,
        failures,
        recommendations,
        step_analyses,
        flags_by_type,
        llm_analysis,
        summary_data,
        analysis_source=analysis_source,
    )

    # 9. Save Outputs
    analysis_output_dir = resolve_analysis_output_dir(output_dir)
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    # Save markdown report
    report_path = analysis_output_dir / "intelligent_analysis_report.md"
    try:
        with open(report_path, "w") as f:
            f.write(report_content)
        logger.info(f"Analysis report saved to {report_path}")
    except Exception as e:
        log_step_error(logger, f"Failed to save report: {e}")
        return False

    # Save JSON analysis data
    analysis_data_path = analysis_output_dir / "analysis_data.json"
    try:
        step_analyses_dict = [sa.to_dict() for sa in step_analyses]

        with open(analysis_data_path, "w") as f:
            json.dump(
                {
                    "run_id": summary_data.get("run_id"),
                    "summary_source": str(summary_path),
                    "summary_source_sha256": hashlib.sha256(
                        summary_path.read_bytes()
                    ).hexdigest(),
                    "model_selection": summary_data.get("model_selection", []),
                    "timestamp": evidence_timestamp,
                    "timestamp_source": evidence_timestamp_source,
                    "analysis": analysis,
                    "step_analyses": step_analyses_dict,
                    "flags_summary": {
                        "red_count": red_count,
                        "yellow_count": yellow_count,
                        "green_count": len(flags_by_type.get("green", [])),
                    },
                    "bottlenecks": bottlenecks,
                    "failures": failures,
                    "recommendations": recommendations,
                    "recovery_plan": recovery_plan,
                    "analysis_source": analysis_source,
                    "llm_analysis_available": analysis_source == "llm",
                },
                f,
                indent=2,
            )
        logger.info(f"Analysis data saved to {analysis_data_path}")
    except Exception as e:
        logger.warning(f"Failed to save analysis data: {e}")

    # Save summary
    summary_output_path = analysis_output_dir / "intelligent_analysis_summary.json"
    try:
        with open(summary_output_path, "w") as f:
            json.dump(
                {
                    "timestamp": evidence_timestamp,
                    "timestamp_source": evidence_timestamp_source,
                    "analysis_source": analysis_source,
                    "overall_status": analysis["overall_status"],
                    "health_score": analysis["health_score"],
                    "failure_count": len(failures),
                    "red_flag_count": red_count,
                    "yellow_flag_count": yellow_count,
                    "bottleneck_count": len(bottlenecks),
                    "recommendation_count": len(recommendations),
                    "recovery_plan_length": len(recovery_plan),
                    "recovery_plan": recovery_plan,
                    "report_file": str(report_path),
                    "data_file": str(analysis_data_path),
                },
                f,
                indent=2,
            )
    except Exception as e:
        logger.warning(f"Failed to save summary: {e}")

    log_step_success(logger, "Intelligent analysis completed successfully")
    return True
