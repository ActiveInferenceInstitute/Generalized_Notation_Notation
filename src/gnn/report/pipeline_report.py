#!/usr/bin/env python3
"""
Pipeline Report Generator — Auto-generates PIPELINE_REPORT.md after each run.

Reads pipeline execution summary and per-step outputs to produce a structured
Markdown report covering: overview, step status, timing, artifacts, and errors.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

from gnn.pipeline.artifact_ownership import current_artifact_files

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineReportContext:
    """Resolved inputs used by each report section."""

    output_dir: Path
    summary_path: Path
    summary: Dict[str, Any]
    step_dirs: List[Path]
    step_stats: Dict[str, Dict[str, Any]]
    mode: Literal["final", "preliminary"]


def generate_pipeline_report(
    output_dir: Path,
    summary_path: Optional[Path] = None,
    *,
    mode: Literal["final", "preliminary"] = "final",
) -> str:
    """
    Generate a comprehensive Markdown report from pipeline results.

    Args:
        output_dir: Root output directory containing per-step subdirectories.
        summary_path: Path to pipeline_execution_summary.json.
                      Defaults to output_dir/pipeline_execution_summary.json.
        mode: Use "preliminary" while step 23 is running, or "final" after the
              completed pipeline summary has been written.

    Returns:
        Markdown string for PIPELINE_REPORT.md.
    """
    context = _build_report_context(output_dir, summary_path, mode)

    # Build report sections
    sections: list[Any] = [
        _section_header(context.summary),
        _section_step_status(
            context.summary,
            context.step_stats,
            mode=context.mode,
        ),
        _section_timing(context.summary),
        _section_artifacts(context.step_dirs),
        _section_statistics(context.output_dir),
        _section_errors(context.summary, context.step_stats),
    ]

    report = "\n\n---\n\n".join(s for s in sections if s)
    logger.info(f"📄 Pipeline report generated ({len(report)} chars)")
    return report


def _build_report_context(
    output_dir: Path,
    summary_path: Optional[Path],
    mode: Literal["final", "preliminary"],
) -> PipelineReportContext:
    """Resolve report inputs once so section renderers stay side-effect free."""
    resolved_output_dir = Path(output_dir)
    resolved_summary_path = (
        summary_path
        or resolved_output_dir
        / "00_pipeline_summary"
        / "pipeline_execution_summary.json"
    )
    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    if context is not None and mode == "preliminary":
        resolved_summary_path = (
            Path(context.output_root) / "00_pipeline_summary" / "current_summary.json"
        )
    summary = _load_summary(resolved_summary_path)
    if context is not None and summary.get("run_id") != context.run_id:
        raise ValueError("Pipeline report summary belongs to a different run")
    step_dirs = _discover_step_dirs(resolved_output_dir)
    return PipelineReportContext(
        output_dir=resolved_output_dir,
        summary_path=resolved_summary_path,
        summary=summary,
        step_dirs=step_dirs,
        step_stats=_collect_step_stats(step_dirs),
        mode=mode,
    )


def _load_summary(path: Path) -> Dict[str, Any]:
    """Load pipeline_execution_summary.json, returning empty dict on failure."""
    # Check the provided path first, then fall back to 00_pipeline_summary subdir
    candidates: list[Any] = [path]
    if path.parent.name != "00_pipeline_summary":
        candidates.append(path.parent / "00_pipeline_summary" / path.name)

    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate) as f:
                    return cast("dict[str, Any]", json.load(f))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load summary from {candidate}: {e}")
    return {}


def _discover_step_dirs(output_dir: Path) -> List[Path]:
    """Find all step output directories (e.g. 3_gnn_output, 7_export_output)."""
    dirs = sorted(
        d
        for d in output_dir.iterdir()
        if d.is_dir() and d.name[0].isdigit() and "_output" in d.name
    )
    return dirs


def _collect_step_stats(step_dirs: List[Path]) -> Dict[str, Dict[str, Any]]:
    """Collect basic stats from each step output directory."""
    stats: dict[Any, Any] = {}
    for d in step_dirs:
        file_count = len(current_artifact_files(d))
        total_size = sum(f.stat().st_size for f in current_artifact_files(d))
        # Look for a results JSON
        results_json = None
        for candidate in [
            d / "results.json",
            d / f"{d.name.replace('_output', '')}_results.json",
        ]:
            if candidate.exists():
                results_json = candidate
                break
        # Also check commonly named files
        if not results_json:
            jsons = list(d.glob("*results*.json"))
            if jsons:
                results_json = jsons[0]

        stats[d.name] = {
            "file_count": file_count,
            "total_size_kb": round(total_size / 1024, 1),
            "results_json": results_json,
        }
    return stats


def _section_header(summary: Dict[str, Any]) -> str:
    """Generate the report header section."""
    timestamp = (
        summary.get("end_time")
        or summary.get("start_time")
        or summary.get("timestamp")
        or "unavailable"
    )
    total_duration = summary.get(
        "total_duration_seconds", summary.get("total_duration", "N/A")
    )
    if isinstance(total_duration, (int, float)):
        total_duration = f"{total_duration:.1f}"

    # Support both boolean 'success' and string 'overall_status' keys.
    overall_status = summary.get("overall_status", None)
    success = summary.get("success", None)
    if overall_status:
        status_text = str(overall_status).upper()
        is_success = status_text == "SUCCESS"
        is_warning = "WARN" in status_text or status_text == "PARTIAL_SUCCESS"
        is_failed = status_text in ("FAILED", "FAILURE")
    elif success is not None:
        is_success = success is True
        is_warning = False
        is_failed = success is False
        status_text = "SUCCESS" if is_success else "FAILED"
    else:
        is_success = False
        is_warning = False
        is_failed = False
        status_text = "UNKNOWN"

    status_emoji = (
        "🟢" if is_success else "🟡" if is_warning else "🔴" if is_failed else "⚪"
    )

    return f"""# Pipeline Execution Report

{status_emoji} **Status**: {status_text}
📅 **Evidence As Of**: {timestamp}
⏱️ **Total Duration**: {total_duration}s
🔧 **Generated by**: `src/report/pipeline_report.py`"""


def _section_step_status(
    summary: Dict[str, Any],
    step_stats: Dict[str, Dict[str, Any]],
    *,
    mode: Literal["final", "preliminary"] = "final",
) -> str:
    """Generate step status table."""
    lines: list[Any] = ["## Step Status"]

    steps = summary.get("steps", [])
    if steps:
        lines.append("")
        lines.append("| # | Step | Status | Duration (s) | Files | Size (KB) |")
        lines.append("|---|------|--------|-------------|-------|-----------|")
        for step in steps:
            name = step.get("description", step.get("name", "unknown"))
            status = step.get("status", "?")
            duration = step.get("duration_seconds", "—")
            if isinstance(duration, float):
                duration = f"{duration:.1f}"
            # Derive output dir name from script_name (e.g. "3_gnn.py" → "3_gnn_output")
            output_dir_name = step.get("output_dir", "")
            if not output_dir_name:
                script_name = step.get("script_name", "")
                if script_name:
                    output_dir_name = script_name.replace(".py", "_output")
            st = step_stats.get(output_dir_name, {})
            files = st.get("file_count", "—")
            size = st.get("total_size_kb", "—")
            step_num = step.get("step_number", step.get("step_num", "?"))
            emoji = (
                "✅"
                if status in ("SUCCESS", "success")
                else "⚠️"
                if "warn" in str(status).lower()
                else "❌"
            )
            lines.append(
                f"| {step_num} | {name} | {emoji} {status} | {duration} | {files} | {size} |"
            )

        if mode == "preliminary":
            # Step 23 can only see a partial summary while it is running. Add
            # explicit in-flight rows there, but never in the final report.
            recorded_scripts = {s.get("script_name", "") for s in steps}
            self_referencing_steps: list[Any] = [
                ("23_report.py", "Comprehensive analysis report generation"),
                (
                    "24_intelligent_analysis.py",
                    "AI-powered pipeline analysis and executive reports",
                ),
            ]
            for script_name, desc in self_referencing_steps:
                if script_name not in recorded_scripts:
                    output_dir_name = script_name.replace(".py", "_output")
                    st = step_stats.get(output_dir_name, {})
                    files = st.get("file_count", "—")
                    size = st.get("total_size_kb", "—")
                    step_num = int(script_name.split("_")[0]) + 1
                    lines.append(
                        f"| {step_num} | {desc} | 🔄 IN PROGRESS | — | {files} | {size} |"
                    )

    elif step_stats:
        lines.append("")
        lines.append("| Step Output | Files | Size (KB) |")
        lines.append("|-------------|-------|-----------|")
        for dir_name, st in sorted(step_stats.items()):
            lines.append(
                f"| `{dir_name}` | {st['file_count']} | {st['total_size_kb']} |"
            )
    else:
        lines.append("\n*No step results available.*")

    return "\n".join(lines)


def _section_timing(summary: Dict[str, Any]) -> str:
    """Generate timing breakdown section."""
    steps = summary.get("steps", [])
    if not steps:
        return ""

    total = summary.get("total_duration_seconds", summary.get("total_duration", 0))
    lines: list[Any] = ["## Timing Breakdown"]

    # Sort by duration descending
    sorted_steps = sorted(
        steps, key=lambda s: s.get("duration_seconds", 0), reverse=True
    )

    lines.append("")
    lines.append("| Step | Duration (s) | % of Total |")
    lines.append("|------|-------------|-----------|")
    for step in sorted_steps:
        dur = step.get("duration_seconds", 0)
        pct = (dur / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        name = step.get("description", step.get("name", "?"))
        lines.append(f"| {name} | {dur:.1f} | {bar} {pct:.0f}% |")

    return "\n".join(lines)


def _section_artifacts(step_dirs: List[Path]) -> str:
    """Generate artifact inventory section."""
    lines: list[Any] = ["## Artifact Inventory"]
    lines.append("")

    if not step_dirs:
        lines.append("*No artifacts discovered.*")
        return "\n".join(lines)

    total_files = 0
    total_size = 0
    for d in step_dirs:
        files = current_artifact_files(d)
        file_items = [f for f in files if f.is_file()]
        count = len(file_items)
        size = sum(f.stat().st_size for f in file_items)
        total_files += count
        total_size += size

        # Show notable files (JSON results, reports)
        notable = [
            f.name for f in file_items if f.suffix in (".json", ".md", ".html", ".svg")
        ][:5]
        notable_str = ", ".join(f"`{n}`" for n in notable) if notable else "—"
        lines.append(
            f"- **`{d.name}`**: {count} files ({size / 1024:.1f} KB) — {notable_str}"
        )

    lines.append(f"\n**Total**: {total_files} files ({total_size / 1024:.1f} KB)")
    return "\n".join(lines)


def _section_statistics(output_dir: Path) -> str:
    """Pull model/parse statistics from GNN step outputs."""
    lines: list[Any] = ["## Pipeline Statistics"]
    stats: dict[Any, Any] = {}

    # GNN parse results
    gnn_results = output_dir / "3_gnn_output" / "gnn_results.json"
    if gnn_results.exists():
        try:
            with open(gnn_results) as f:
                data = json.load(f)
            stats["Files parsed"] = data.get("total_files", "?")
            stats["Models discovered"] = data.get(
                "total_models", data.get("files_parsed", "?")
            )
            stats["Sections extracted"] = sum(
                len(fi.get("sections", []))
                for fi in data.get("file_results", data.get("results", []))
                if isinstance(fi, dict)
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Could not load GNN results from %s: %s", gnn_results, e)

    # Export results
    export_dir = output_dir / "7_export_output"
    if export_dir.exists():
        export_files = current_artifact_files(export_dir)
        stats["Export artifacts"] = len([f for f in export_files if f.is_file()])

    # LLM cache stats
    cache_dir = output_dir / ".cache"
    if cache_dir.exists():
        stats["LLM cache entries"] = len(list(cache_dir.glob("*.json")))

    if not stats:
        return ""

    lines.append("")
    for k, v in stats.items():
        lines.append(f"- **{k}**: {v}")

    return "\n".join(lines)


def _section_errors(
    summary: Dict[str, Any],
    step_stats: Dict[str, Dict[str, Any]],
) -> str:
    """Generate error summary section."""
    errors = list(summary.get("errors", []))
    if summary.get("error"):
        errors.append({"step": "Pipeline", "error": summary["error"]})
    for step in summary.get("steps", []):
        if step.get("status") in ("FAILED", "TIMEOUT"):
            errors.append(
                {
                    "step": step.get("script_name", "?"),
                    "error": step.get("error")
                    or step.get("stderr")
                    or step.get("stop_reason")
                    or "Step failed without an error message",
                }
            )
    auth_errors = summary.get("auth_errors", [])

    if not errors and not auth_errors:
        return "## Errors\n\n✅ No errors recorded."

    lines: list[Any] = ["## Errors"]
    lines.append("")

    for err in errors:
        if isinstance(err, dict):
            lines.append(
                f"- ❌ **{err.get('step', '?')}**: {err.get('error', str(err))}"
            )
        else:
            lines.append(f"- ❌ {err}")

    for ae in auth_errors:
        if isinstance(ae, dict):
            lines.append(
                f"- 🔑 Auth failure: **{ae.get('provider', '?')}** — {ae.get('error', '')[:100]}"
            )

    return "\n".join(lines)
