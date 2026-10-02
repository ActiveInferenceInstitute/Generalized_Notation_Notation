"""Summary/report/perf wiring for the GNN pipeline runner.

Extracted verbatim from ``gnn/main.py`` (SCOPE M-01 band slice, wave 3):
summary/report/performance helpers, ``execute_pipeline_step``, and the
``parse_step_list`` / ``get_environment_info`` / ``validate_pipeline_summary``
tail helpers. ``gnn.main`` re-exports these names for backward compatibility
(``gnn.pipeline.execution`` calls ``main.execute_pipeline_step``).
"""

import json
import logging
import os
import re
import sys
import time
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from gnn.utils.arguments.pipeline_arguments import PipelineArguments
from gnn.utils.arguments.step_config import StepConfiguration
from gnn.utils.errors.error_handling import (
    is_critical_pipeline_step,
)
from gnn.utils.errors.error_handling import (
    pipeline_exit_code as _shared_pipeline_exit_code,
)
from gnn.utils.errors.error_handling import (
    status_from_exit_code as _shared_status_from_exit_code,
)
from gnn.utils.logging_utils import (
    PipelineProgressTracker,
    log_pipeline_summary,
    log_step_error,
)
from gnn.utils.observability.visual_logging import (
    print_completion_summary,
    print_step_summary,
)
from gnn.utils.pipeline_orchestration.pipeline_validator import (
    validate_step_prerequisites,
)
from gnn.utils.runtime_safety.resource_manager import get_current_memory_usage

_module_logger = logging.getLogger(__name__)

STRUCTURED_LOGGING_AVAILABLE = True


def _late_main(name: str) -> Any:
    """Resolve ``name`` through ``gnn.main`` at call time.

    These call sites historically resolved module globals in ``gnn.main``;
    tests monkeypatch ``gnn.main.<name>`` to intercept them. The function-level
    import avoids the ``gnn.main`` <-> ``summary_wiring`` import cycle.
    """
    import gnn.main

    return getattr(gnn.main, name)


def _derive_critical_scripts() -> set[str]:
    """Return script filenames marked critical by StepConfiguration metadata."""
    return {
        f"{step_name}.py"
        for step_name, config in StepConfiguration.STEP_CONFIGS.items()
        if config.get("critical", is_critical_pipeline_step(step_name))
    }


# Metadata-driven cache used by the orchestrator hot path. The critical-step
# contract lives in StepConfiguration and utils.errors.error_handling; do not maintain
# a second hand-written critical-step list here.
CRITICAL_SCRIPTS: set[str] = _derive_critical_scripts()

SAFE_WARNING_PATTERNS: tuple[str, ...] = (
    r"matplotlib.*?backend",
    r"using agg backend",
    r"no display",
    r"pymdp.*?not available",
    r"optional.*?dependency",
    r"plotly.*?not available",
    r"numpy.*?not available",
    r"seaborn.*?not available",
    r"bokeh.*?not available",
    r"d2.*?not available",
    r"d2 cli.*?not available",
    r"d2 visualizer.*?not available",
    r"d2 cli.*?install",
    r"interactive.*?limited",
    r"numeric.*?limited",
    r"warnings: 0",
    r"optional test tooling not installed",
)

WARNING_PATTERN = re.compile(r"(WARNING|⚠️|warn)", re.IGNORECASE)
SAFE_WARNING_PATTERN = re.compile(
    "|".join(f"({pattern})" for pattern in SAFE_WARNING_PATTERNS),
    re.IGNORECASE,
)


def step_number_from_script_name(script_name: str) -> int:
    """Return the numeric step encoded in an ``N_*.py`` script filename.

    Args:
        script_name: Script filename such as ``"11_render.py"``.

    Returns:
        The leading step number, or ``-1`` for non-numbered names.
    """
    match = re.match(r"^(\d+)_", script_name)
    return int(match.group(1)) if match else -1


def _read_input_config(config_path: Path) -> dict[str, Any]:
    """Load and parse a pipeline YAML config file.

    Args:
        config_path: Path to the YAML config (typically ``input/config.yaml``).

    Returns:
        Parsed config mapping; ``{}`` when the file is absent.

    Raises:
        Exception: Propagates YAML/import errors to the caller, which owns
            the failure-reporting policy (warning vs. debug).
    """
    if not config_path.exists():
        return {}

    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _log_pipeline_step_start(
    actual_step_number: int,
    script_name: str,
    description: str,
    total_steps: int,
    progress_tracker: Optional[PipelineProgressTracker],
    logger: logging.Logger,
) -> None:
    """Log step start through structured logging when available."""
    if STRUCTURED_LOGGING_AVAILABLE and progress_tracker:
        progress_tracker.start_step(actual_step_number, description)
        from gnn.utils.logging_utils import (
            log_step_start as structured_log_step_start,
        )

        structured_log_step_start(
            logger,
            f"Starting {description}",
            step_number=actual_step_number,
            total_steps=total_steps,
            script_name=script_name,
        )
    else:
        logger.info(f"🔄 Executing step {actual_step_number}: {description}")


def _annotate_step_result(
    step_result: dict[str, Any],
    actual_step_number: int,
    script_name: str,
    description: str,
    step_start_datetime: datetime,
    step_end_datetime: datetime,
    step_duration: float,
) -> None:
    """Attach timing and standard metadata to a raw step result."""
    step_result.update(
        {
            "step_number": actual_step_number,
            "script_name": script_name,
            "description": description,
            "start_time": step_start_datetime.isoformat(),
            "end_time": step_end_datetime.isoformat(),
            "duration_seconds": step_duration,
            "exit_code": step_result.get("exit_code", 0),
            "retry_count": step_result.get("retry_count", 0),
            "prerequisite_check": step_result.get("prerequisite_check", True),
            "dependency_warnings": step_result.get("dependency_warnings", []),
            "recoverable": step_result.get("recoverable", False),
            "memory_usage_mb": step_result.get("memory_usage_mb", 0.0),
            "peak_memory_mb": step_result.get("peak_memory_mb", 0.0),
            "memory_delta_mb": step_result.get("memory_delta_mb", 0.0),
        }
    )


def _step_has_actionable_warning(step_result: dict[str, Any]) -> bool:
    """Return true when step output contains warnings not on the safe-noise list."""
    combined_output = (
        f"{step_result.get('stdout', '')}\n{step_result.get('stderr', '')}"
    )
    warning_lines = [
        line for line in combined_output.splitlines() if WARNING_PATTERN.search(line)
    ]
    if not warning_lines:
        return False

    return any(not SAFE_WARNING_PATTERN.search(line) for line in warning_lines)


def _update_performance_summary(
    pipeline_summary: dict[str, Any],
    step_result: dict[str, Any],
    script_name: str,
    has_warning: bool,
    total_steps: int,
) -> None:
    """Update aggregate counts and peak memory after a step finishes."""
    perf_summary = pipeline_summary["performance_summary"]
    complete = {
        step.get("script_name")
        for step in pipeline_summary.get("steps", [])
        if step.get("status") in ("SUCCESS", "SUCCESS_WITH_WARNINGS", "SKIPPED")
    }
    pipeline_summary["unfinished_steps"] = [
        script
        for script in pipeline_summary.get("planned_steps", [])
        if script not in complete
    ]
    if step_result["status"] in ("SUCCESS", "SUCCESS_WITH_WARNINGS", "SKIPPED"):
        perf_summary["successful_steps"] += 1
    elif step_result["status"] == "FAILED":
        perf_summary["failed_steps"] += 1
        if script_name in CRITICAL_SCRIPTS and step_result.get("exit_code", 0) != 0:
            perf_summary["critical_failures"] += 1

    if has_warning or step_result.get("status") == "SUCCESS_WITH_WARNINGS":
        perf_summary["warnings"] += 1

    step_memory = step_result.get("memory_usage_mb", 0.0)
    step_peak_memory = step_result.get("peak_memory_mb", 0.0)
    perf_summary["peak_memory_mb"] = max(
        step_memory,
        step_peak_memory,
        perf_summary["peak_memory_mb"],
    )
    perf_summary["total_steps"] = total_steps


def _log_pipeline_step_completion(
    actual_step_number: int,
    description: str,
    step_result: dict[str, Any],
    step_duration: float,
    progress_tracker: Optional[PipelineProgressTracker],
    logger: logging.Logger,
) -> None:
    """Print and log the completion state for one pipeline step."""
    status_for_logging = step_result["status"]
    step_stats: dict[str, Any] = {
        "Status": status_for_logging,
        "Duration": f"{step_duration:.2f}s",
        "Memory": f"{step_result.get('peak_memory_mb', 0):.1f}MB",
        "Exit Code": step_result.get("exit_code", 0),
    }
    print_step_summary(
        actual_step_number,
        description,
        status_for_logging,
        step_duration,
        step_stats,
    )

    if STRUCTURED_LOGGING_AVAILABLE and progress_tracker:
        if "WARNING" in status_for_logging:
            from gnn.utils.logging_utils import (
                log_step_warning as structured_log_step_warning,
            )

            structured_log_step_warning(
                logger,
                f"{description} completed with warnings",
                step_number=actual_step_number,
                duration=step_duration,
                status=status_for_logging,
            )
        elif status_for_logging.startswith("SUCCESS"):
            from gnn.utils.logging_utils import (
                log_step_success as structured_log_step_success,
            )

            structured_log_step_success(
                logger,
                f"{description} completed",
                step_number=actual_step_number,
                duration=step_duration,
                status=status_for_logging,
            )
        else:
            from gnn.utils.logging_utils import (
                log_step_error as structured_log_step_error,
            )

            structured_log_step_error(
                logger,
                f"{description} failed",
                step_number=actual_step_number,
                duration=step_duration,
                status=status_for_logging,
            )
        return

    if status_for_logging == "PARTIAL_SUCCESS" or (
        isinstance(status_for_logging, str) and "WARNING" in status_for_logging
    ):
        logger.warning(
            f"⚠️ Step {actual_step_number} completed with warnings in {step_duration:.2f}s"
        )
    elif str(status_for_logging).startswith("SUCCESS"):
        logger.info(
            f"✅ Step {actual_step_number} completed successfully in {step_duration:.2f}s"
        )
    else:
        logger.error(
            f"❌ Step {actual_step_number} failed with status: {status_for_logging}"
        )


def _record_step_result(
    step_result: dict[str, Any],
    summary_step_number: int,
    script_name: str,
    description: str,
    step_start_datetime: datetime,
    step_end_datetime: datetime,
    step_duration: float,
    pipeline_summary: dict[str, Any],
    total_steps: int,
    progress_tracker: Optional[PipelineProgressTracker],
    logger: logging.Logger,
) -> None:
    """Annotate, warning-upgrade, record, and log one finished pipeline step.

    Shared tail of the serial and parallel execution paths so both record
    identical summary entries.
    """
    _annotate_step_result(
        step_result,
        summary_step_number,
        script_name,
        description,
        step_start_datetime,
        step_end_datetime,
        step_duration,
    )

    has_warning = _step_has_actionable_warning(step_result)
    if step_result["status"] == "SUCCESS" and has_warning:
        step_result["status"] = "SUCCESS_WITH_WARNINGS"

    # Receipt: record which execution mode produced this step entry. The
    # consolidated executor stamps "consolidated" on its own receipts.
    step_result.setdefault("execution_mode", "subprocess")

    pipeline_summary["steps"].append(step_result)
    _update_performance_summary(
        pipeline_summary,
        step_result,
        script_name,
        has_warning,
        total_steps,
    )
    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    if context is not None:
        from gnn.pipeline._io import atomic_write_text

        atomic_write_text(
            Path(context.output_root) / "00_pipeline_summary" / "current_summary.json",
            json.dumps(pipeline_summary, indent=2, default=str),
        )
    _log_pipeline_step_completion(
        summary_step_number,
        description,
        step_result,
        step_duration,
        progress_tracker,
        logger,
    )


def _consolidated_step_selected(
    script_name: str,
    args: PipelineArguments,
    pipeline_summary: dict[str, Any],
) -> bool:
    """Decide the execution mode for one serial step under --consolidated-steps.

    Strictly opt-in: when the flag is off, or the executor declines the step
    (not in the consolidated whitelist, or testing-matrix folder dispatch is
    active), the canonical subprocess path runs unchanged.
    """
    if os.environ.get("GNN_RUN_CONTEXT_FILE"):
        # Threads cannot enforce filesystem quiescence at a hard step deadline.
        # Budgeted top-level invocations keep a killable process boundary.
        return False
    if not getattr(args, "consolidated_steps", False):
        return False
    from gnn.pipeline.step_executor import can_execute_in_process

    return can_execute_in_process(
        script_name,
        args,
        pipeline_config=pipeline_summary.get("identity_config", {}).get("input_config"),
    )


def _finalize_pipeline_summary(pipeline_summary: dict[str, Any]) -> None:
    """Set end time, duration, and final overall status."""
    end_time_dt = datetime.now()
    pipeline_summary["end_time"] = end_time_dt.isoformat()
    pipeline_summary["total_duration_seconds"] = _elapsed_run_duration(
        pipeline_summary, end_time_dt
    )

    perf_summary = pipeline_summary["performance_summary"]
    timeout_steps = [
        step
        for step in pipeline_summary.get("steps", [])
        if step.get("stop_reason") in ("total_timeout", "step_timeout")
    ]
    if timeout_steps:
        pipeline_summary["overall_status"] = "FAILED"
        pipeline_summary["stop_reason"] = timeout_steps[0]["stop_reason"]
    elif perf_summary["critical_failures"] > 0:
        pipeline_summary["overall_status"] = "FAILED"
    elif perf_summary["failed_steps"] > 0:
        total_steps = perf_summary["total_steps"]
        failed_ratio = perf_summary["failed_steps"] / total_steps if total_steps else 0

        if failed_ratio > 0.5:
            pipeline_summary["overall_status"] = "FAILED"
        else:
            pipeline_summary["overall_status"] = "PARTIAL_SUCCESS"
    elif perf_summary.get("warnings", 0) > 0 or any(
        step.get("status") == "SUCCESS_WITH_WARNINGS"
        for step in pipeline_summary.get("steps", [])
    ):
        pipeline_summary["overall_status"] = "SUCCESS_WITH_WARNINGS"
    else:
        pipeline_summary["overall_status"] = "SUCCESS"


def _pipeline_exit_code(overall_status: str) -> int:
    """Map final pipeline status to the public process exit contract."""
    return _shared_pipeline_exit_code(overall_status)


def _elapsed_run_duration(summary: dict[str, Any], end_time: datetime) -> float:
    """Measure live elapsed time independently of host wall clock adjustments."""
    if "start_monotonic" in summary:
        return time.monotonic() - float(summary["start_monotonic"])
    return (end_time - datetime.fromisoformat(summary["start_time"])).total_seconds()


def _status_from_step_exit_code(
    exit_code: int, dependency_warnings: Optional[list[Any]] = None
) -> str:
    """Map a child step exit code to the pipeline step status contract."""
    return _shared_status_from_exit_code(exit_code, dependency_warnings)


def _pipeline_summary_path(output_dir: Path) -> Path:
    """Return the canonical pipeline execution summary path."""
    return output_dir / "00_pipeline_summary" / "pipeline_execution_summary.json"


def _write_performance_dashboard(
    summary_path: Path,
    pipeline_summary: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Render the D3 performance dashboard when the template is available."""
    try:
        template_path = Path(__file__).parent / "performance_dashboard.template.html"
        if template_path.exists():
            template_content = template_path.read_text()
            json_payload = json.dumps(pipeline_summary)
            final_html = template_content.replace("{SUMMARY_JSON}", json_payload)
            db_path = summary_path.parent / "performance_dashboard.html"
            from gnn.pipeline._io import atomic_write_text

            atomic_write_text(db_path, final_html)
            logger.info(f"📊 D3 Performance Dashboard rendered to: {db_path}")
    except Exception as e:
        if os.environ.get("GNN_RUN_CONTEXT_FILE"):
            raise
        logger.warning(f"Could not render performance dashboard: {e}")


def _write_final_pipeline_report(
    output_dir: Path,
    summary_path: Path,
    logger: logging.Logger,
) -> None:
    """Regenerate the final pipeline report from the final summary."""
    try:
        from gnn.pipeline._io import atomic_write_text
        from gnn.report.pipeline_report import generate_pipeline_report

        report_path = output_dir / "PIPELINE_REPORT.md"
        atomic_write_text(
            report_path,
            generate_pipeline_report(
                output_dir,
                summary_path=summary_path,
                mode="final",
            ),
        )
        logger.info(f"📄 Final pipeline report written to: {report_path}")
    except Exception as e:
        if os.environ.get("GNN_RUN_CONTEXT_FILE"):
            raise
        logger.warning(f"Could not render final pipeline report: {e}")


def _log_pipeline_summary_counts(
    pipeline_summary: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Log a compact success/failure count after saving the final summary."""
    steps = pipeline_summary["steps"]
    successful = sum(
        1 for step in steps if step["status"] in ("SUCCESS", "SUCCESS_WITH_WARNINGS")
    )
    failed = sum(1 for step in steps if step["status"] == "FAILED")
    logger.info(f"Summary: {successful}/{len(steps)} steps successful, {failed} failed")


def _save_minimal_pipeline_summary(
    summary_path: Path,
    pipeline_summary: dict[str, Any],
    error: Exception,
    logger: logging.Logger,
) -> None:
    """Persist a reduced summary when full summary rendering fails."""
    try:
        minimal_summary: dict[str, Any] = {
            "run_id": pipeline_summary.get("run_id", os.environ.get("GNN_RUN_ID")),
            "run_hash": pipeline_summary.get("run_hash"),
            "run_hash_schema": pipeline_summary.get("run_hash_schema"),
            "identity_config": pipeline_summary.get("identity_config"),
            "start_time": pipeline_summary.get("start_time"),
            "end_time": datetime.now().isoformat(),
            "overall_status": "FAILED",
            "error": str(error),
            "arguments": pipeline_summary.get("arguments", {}),
            "steps_count": len(pipeline_summary.get("steps", [])),
            "performance_summary": pipeline_summary.get("performance_summary", {}),
            "steps": pipeline_summary.get("steps", []),
        }
        from gnn.pipeline._io import atomic_write_text

        atomic_write_text(
            summary_path, json.dumps(minimal_summary, indent=4, default=str)
        )
        logger.info("Minimal summary saved as recovery")
    except Exception as fallback_error:
        logger.error(f"Failed to save even minimal summary: {fallback_error}")


def _write_pipeline_summary_outputs(
    args: PipelineArguments,
    config_pipeline_settings: dict[Any, Any],
    pipeline_summary: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Validate, save, index, and render all final pipeline summary artifacts."""
    from gnn.pipeline.hasher import RUN_HASH_SCHEMA, index_run, verify_indexed_run

    run_config = {
        "args": args.to_dict(),
        "pipeline": config_pipeline_settings,
        "identity_config": pipeline_summary.get("identity_config"),
        "run_hash_schema": pipeline_summary.get("run_hash_schema"),
    }
    summary_path = _pipeline_summary_path(args.output_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving pipeline summary to: {summary_path}")

    try:
        if pipeline_summary.get("run_hash_schema") == RUN_HASH_SCHEMA:
            problems = verify_indexed_run(
                {
                    "run_hash": pipeline_summary.get("run_hash"),
                    "file_hashes": pipeline_summary.get("file_hashes"),
                    "config": run_config,
                }
            )
            if problems:
                raise ValueError(
                    "Run identity changed before publication: " + "; ".join(problems)
                )
        _late_main("validate_pipeline_summary")(pipeline_summary, logger)
        from gnn.pipeline._io import atomic_write_text

        atomic_write_text(
            summary_path, json.dumps(pipeline_summary, indent=4, default=str)
        )
        atomic_write_text(
            summary_path.parent / "current_summary.json",
            json.dumps(pipeline_summary, indent=4, default=str),
        )
        logger.info("Pipeline summary saved successfully")

        run_hash_value = pipeline_summary.get("run_hash")
        if isinstance(run_hash_value, str) and run_hash_value:
            index_run(
                run_hash=run_hash_value,
                summary_path=summary_path,
                config=run_config,
                file_hashes=pipeline_summary.get("file_hashes"),
            )

        _late_main("_write_performance_dashboard")(
            summary_path, pipeline_summary, logger
        )
        _late_main("_write_final_pipeline_report")(
            args.output_dir, summary_path, logger
        )
        _log_pipeline_summary_counts(pipeline_summary, logger)
    except Exception as e:
        logger.error(f"Failed to save pipeline summary: {e}")
        pipeline_summary["overall_status"] = "FAILED"
        pipeline_summary["steps_count"] = len(pipeline_summary.get("steps", []))
        _late_main("_save_minimal_pipeline_summary")(
            summary_path, pipeline_summary, e, logger
        )
        if os.environ.get("GNN_RUN_CONTEXT_FILE"):
            raise


def _print_pipeline_completion(
    pipeline_summary: dict[str, Any],
    progress_tracker: Optional[PipelineProgressTracker],
    logger: logging.Logger,
) -> None:
    """Print the final visual completion summary and structured log summary."""
    total_duration = float(pipeline_summary["total_duration_seconds"] or 0.0)
    perf_summary = pipeline_summary["performance_summary"]
    completion_stats: dict[str, Any] = {
        "Total Steps": len(pipeline_summary.get("steps", [])),
        "Successful": perf_summary["successful_steps"],
        "Failed": perf_summary["failed_steps"],
        "Warnings": perf_summary["warnings"],
        "Peak Memory": f"{perf_summary['peak_memory_mb']:.1f}MB",
        "Duration": f"{total_duration:.1f}s",
    }

    success = pipeline_summary["overall_status"] == "SUCCESS"
    print_completion_summary(success, total_duration, completion_stats)

    if STRUCTURED_LOGGING_AVAILABLE:
        if progress_tracker:
            logger.info(progress_tracker.get_overall_progress())
        log_pipeline_summary(logger, pipeline_summary)
    elif success:
        logger.info(f"🎯 Pipeline completed successfully in {total_duration:.2f}s")
    else:
        logger.info(f"⚠️ Pipeline completed with issues in {total_duration:.2f}s")


def _handle_pipeline_failure(
    error: Exception,
    args: PipelineArguments,
    pipeline_summary: dict[str, Any],
    logger: logging.Logger,
) -> int:
    """Invalidate all terminal claim surfaces after an invocation failure."""
    end_time_dt = datetime.now()
    pipeline_summary["end_time"] = end_time_dt.isoformat()
    pipeline_summary["overall_status"] = "FAILED"
    pipeline_summary["error"] = str(error)
    complete = {
        step.get("script_name")
        for step in pipeline_summary.get("steps", [])
        if step.get("status") in ("SUCCESS", "SUCCESS_WITH_WARNINGS", "SKIPPED")
    }
    pipeline_summary["unfinished_steps"] = [
        script
        for script in pipeline_summary.get("planned_steps", [])
        if script not in complete
    ]
    if isinstance(error, TimeoutError):
        pipeline_summary["stop_reason"] = "total_timeout"
    pipeline_summary["evidence_integrity"] = {"status": "failed", "error": str(error)}
    pipeline_summary["total_duration_seconds"] = _elapsed_run_duration(
        pipeline_summary, end_time_dt
    )

    summary_path = _pipeline_summary_path(args.output_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    from gnn.pipeline._io import atomic_write_text

    try:
        from gnn.pipeline.finalization import record_failed_evidence

        record_failed_evidence(args.output_dir, pipeline_summary)
    except Exception as recovery_error:
        pipeline_summary.setdefault("finalization_errors", []).append(
            f"Failed evidence recovery: {recovery_error}"
        )
        logger.error("Failed evidence recovery: %s", recovery_error)
    atomic_write_text(summary_path, json.dumps(pipeline_summary, indent=4, default=str))
    atomic_write_text(
        summary_path.parent / "current_summary.json",
        json.dumps(pipeline_summary, indent=4, default=str),
    )
    try:
        from gnn.pipeline.hasher import index_run

        run_hash = pipeline_summary.get("run_hash")
        if isinstance(run_hash, str) and run_hash:
            index_run(
                run_hash,
                summary_path,
                config={
                    "args": args.to_dict(),
                    "identity_config": pipeline_summary.get("identity_config"),
                    "run_hash_schema": pipeline_summary.get("run_hash_schema"),
                },
                file_hashes=pipeline_summary.get("file_hashes"),
            )
    except Exception as index_error:
        logger.error("Could not index failed run: %s", index_error)
    try:
        _late_main("_write_performance_dashboard")(
            summary_path, pipeline_summary, logger
        )
    except Exception as dashboard_error:
        import html

        logger.error("Could not render failed dashboard: %s", dashboard_error)
        atomic_write_text(
            summary_path.parent / "performance_dashboard.html",
            "<!doctype html><title>Pipeline FAILED</title><h1>FAILED</h1><pre>"
            + html.escape(json.dumps(pipeline_summary, indent=2, default=str))
            + "</pre>",
        )
    try:
        _late_main("_write_final_pipeline_report")(
            args.output_dir, summary_path, logger
        )
    except Exception as report_error:
        logger.error("Could not render failed report: %s", report_error)
        atomic_write_text(
            args.output_dir / "PIPELINE_REPORT.md",
            f"# Pipeline Report\n\n**Status**: FAILED\n\n"
            f"**Run ID**: {pipeline_summary.get('run_id')}\n\n"
            f"{error}\n\nReport rendering failed: {report_error}\n",
        )

    log_step_error(logger, f"Pipeline failed: {str(error)}")
    return 1


def _fail_pipeline_startup(error: Exception) -> int:
    """Report a failure raised before the run context existed.

    Args:
        error: The startup exception (for example an invalid step selection).

    Returns:
        The pipeline failure exit code (1).
    """
    log_step_error(_module_logger, f"Pipeline failed during startup: {error}")
    return 1


def execute_pipeline_step(
    script_name: str,
    args: PipelineArguments,
    logger: Any,
    *,
    run_id: Optional[str] = None,
    pipeline_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a single pipeline step with comprehensive monitoring."""
    # Initialize performance tracking
    start_memory = get_current_memory_usage()
    peak_memory = start_memory

    step_result: dict[str, Any] = {
        "status": "UNKNOWN",
        "stdout": "",
        "stderr": "",
        "memory_usage_mb": 0.0,
        "peak_memory_mb": 0.0,
        "memory_delta_mb": 0.0,
        "exit_code": -1,
        "retry_count": 0,
        "prerequisite_check": True,
        "dependency_warnings": [],
    }

    try:
        # Load config.yaml once — used for both skip_steps (prereq validation) and testing_matrix
        config_skip_steps: list[Any] = []
        testing_matrix: dict[Any, Any] = {}
        try:
            _full_cfg = (
                pipeline_config
                if pipeline_config is not None
                else _late_main("_read_input_config")(Path("input/config.yaml"))
            )
            config_skip_steps = _full_cfg.get("pipeline", {}).get("skip_steps", [])
            testing_matrix = _full_cfg.get("testing_matrix", {})
        except Exception as e:
            logger.debug(f"Could not parse input/config.yaml: {e}")

        # Validate step prerequisites
        prereq_result = validate_step_prerequisites(
            script_name, args, logger, skip_steps=config_skip_steps
        )
        step_result["prerequisite_check"] = prereq_result["passed"]
        step_result["dependency_warnings"] = prereq_result.get("warnings", [])

        # Log prerequisite warnings if any
        if prereq_result.get("warnings"):
            for warning in prereq_result["warnings"]:
                logger.warning(f"Prerequisite warning for {script_name}: {warning}")

        # Get script path
        script_path = Path(__file__).parent.parent / script_name

        # Get virtual environment Python path
        project_root = Path(__file__).parent.parent.parent.parent
        venv_python = project_root / ".venv" / "bin" / "python"

        # Use virtual environment Python if available, otherwise fall back to system Python
        python_executable = str(venv_python) if venv_python.exists() else sys.executable

        # Extract step number
        step_num = step_number_from_script_name(script_name)

        matrix_enabled = testing_matrix.get("enabled", False)
        target_folders: list[Any] = []
        from gnn.pipeline.step_registry import step_for_name

        step_metadata = step_for_name(script_name)
        matrix_folder_scope = (
            step_metadata is None
            or step_metadata.execution_scope not in {"run", "corpus"}
        )

        # Check global_steps: if a global step (0, 1, 2) is disabled, skip it entirely
        if matrix_enabled:
            global_steps = testing_matrix.get("global_steps", {})
            script_stem = script_name.replace(".py", "")
            if script_stem in global_steps and not global_steps[script_stem]:
                logger.info(
                    f"⏭️ Skipping {script_name}: disabled in testing_matrix.global_steps"
                )
                step_result["status"] = "SKIPPED"
                step_result["exit_code"] = 0
                step_result["stdout"] = (
                    f"Skipped by global_steps config (testing_matrix.global_steps.{script_stem}: false)\n"
                )
                return step_result

        # Only apply folder-matrix logic if enabled and the step is a processing step (>= 3)
        if matrix_enabled and step_num >= 3 and matrix_folder_scope:
            base_target_dir = args.target_dir
            if base_target_dir.exists() and base_target_dir.is_dir():
                folders_config = testing_matrix.get("folders", {})
                default_steps = testing_matrix.get("default_steps", [])

                # Check all subdirectories in the base target directory
                for item in sorted(base_target_dir.iterdir()):
                    if item.is_dir() and item.name != "archived_gnn_files":
                        # Determine allowed steps for this folder
                        allowed_steps = folders_config.get(item.name, default_steps)
                        if step_num in allowed_steps:
                            target_folders.append(item)

        from gnn.pipeline.step_timeouts import get_step_timeout
        from gnn.utils.arguments.arg_parsing import build_step_command_args
        from gnn.utils.pipeline_orchestration.execution_utils import (
            execute_command_streaming,
        )

        # Prepare environment
        _env = os.environ.copy()
        if run_id is not None:
            _env["GNN_RUN_ID"] = run_id
        _env.setdefault("PYTHONUNBUFFERED", "1")
        comprehensive_requested = any("--comprehensive" in str(arg) for arg in sys.argv)
        step_timeout_seconds: float = get_step_timeout(
            script_name, comprehensive=comprehensive_requested
        )
        from gnn.pipeline.run_context import current_run_context

        context = current_run_context()
        if (
            matrix_enabled
            and step_num >= 3
            and matrix_folder_scope
            and not target_folders
        ):
            from gnn.parsers.common import get_supported_gnn_extensions
            from gnn.pipeline.run_context import selected_model_sources

            root_sources = selected_model_sources(
                args.target_dir,
                step_num,
                recursive=False,
                extensions=get_supported_gnn_extensions(),
            )
            if not root_sources:
                return {
                    **step_result,
                    "status": "SKIPPED",
                    "exit_code": 0,
                    "skip_reason": "no_selected_models",
                    "stdout": "No selected model sources; no work dispatched",
                }
        if context is not None:
            if step_num == 13 and not os.environ.get("GNN_STEP_TIMEOUT_13"):
                configured_llm_timeout = context.input_config.get("llm", {}).get(
                    "timeout_seconds"
                )
                step_timeout_seconds = (
                    float(configured_llm_timeout)
                    if configured_llm_timeout is not None
                    else 600 * max(1, len(context.selected_models(13)))
                )
            step_timeout_seconds = context.bounded_timeout(step_timeout_seconds)
            if step_timeout_seconds <= 0:
                return {
                    **step_result,
                    "status": "FAILED",
                    "exit_code": -1,
                    "stop_reason": "total_timeout",
                    "stderr": "Pipeline total deadline expired",
                }

        logical_step_deadline = time.monotonic() + step_timeout_seconds
        result: dict[str, Any]

        if matrix_enabled and step_num >= 3 and matrix_folder_scope and target_folders:
            # MATRIX MODE: We found specific folders to run for this step
            if args.verbose:
                logger.info(
                    f"Testing matrix enabled. Running step {step_num} on {len(target_folders)} specific folders: {[f.name for f in target_folders]}"
                )

            combined_stdout = ""
            combined_stderr = ""
            worst_exit_code = 0
            folder_results = []

            for folder in target_folders:
                remaining = logical_step_deadline - time.monotonic()
                if remaining <= 0:
                    result = {
                        "status": "TIMEOUT",
                        "exit_code": -1,
                        "stderr": "Logical matrix step deadline expired",
                        "force_killed": False,
                    }
                    worst_exit_code = -1
                    break
                if args.verbose:
                    logger.info(f"  -> Executing for folder: {folder.name}")

                # Creating a modified args copy to point to the specific subfolder
                folder_args = copy(args)
                folder_args.target_dir = folder

                cmd = build_step_command_args(
                    script_name.replace(".py", ""),
                    folder_args,
                    python_executable,
                    script_path,
                )

                # We do not use print_stdout=True for every subfolder iteration to avoid extreme terminal spam,
                # but we will print if requested verbose globally
                if args.verbose:
                    logger.info(f"  -> CMD ACTUALLY IS: {' '.join(cmd)}")

                result = execute_command_streaming(
                    cmd,
                    cwd=project_root,
                    env=_env,
                    timeout=remaining,
                    print_stdout=args.verbose,
                    print_stderr=True,
                    capture_output=True,
                )

                combined_stdout += (
                    f"\n--- Output for {folder.name} ---\n{result.get('stdout', '')}\n"
                )
                if result.get("stderr"):
                    combined_stderr += f"\n--- Stderr for {folder.name} ---\n{result.get('stderr', '')}\n"

                exit_code = result.get("exit_code", -1)
                folder_results.append(
                    {
                        "folder": str(folder),
                        "status": result.get("status"),
                        "exit_code": exit_code,
                        **{
                            key: result[key]
                            for key in (
                                "containment",
                                "cleanup_verified",
                                "streams_drained",
                                "force_killed",
                            )
                            if key in result
                        },
                    }
                )
                if exit_code != 0:
                    if exit_code == 2 and worst_exit_code == 0:
                        worst_exit_code = 2
                    elif exit_code != 2:
                        worst_exit_code = exit_code
                    logger.warning(
                        f"  -> Folder {folder.name} execution returned code {exit_code}"
                    )
                if result.get("status") == "TIMEOUT":
                    break

            end_memory = get_current_memory_usage()
            peak_memory = max(peak_memory, end_memory)

            step_result["stdout"] = combined_stdout
            step_result["stderr"] = combined_stderr
            step_result["exit_code"] = worst_exit_code
            step_result["memory_usage_mb"] = end_memory
            step_result["peak_memory_mb"] = peak_memory
            step_result["memory_delta_mb"] = end_memory - start_memory
            step_result["folder_results"] = folder_results
            step_result["unfinished_folders"] = [
                str(folder) for folder in target_folders[len(folder_results) :]
            ]

        else:
            # STANDARD MODE
            cmd = build_step_command_args(
                script_name.replace(".py", ""), args, python_executable, script_path
            )

            if args.verbose:
                logger.info(f"Executing command: {' '.join(cmd)}")

            result = execute_command_streaming(
                cmd,
                cwd=project_root,
                env=_env,
                timeout=step_timeout_seconds,
                print_stdout=True,
                print_stderr=True,
                capture_output=True,
            )

            end_memory = get_current_memory_usage()
            peak_memory = max(peak_memory, end_memory)

            step_result["stdout"] = result.get("stdout", "")
            step_result["stderr"] = result.get("stderr", "")
            step_result["exit_code"] = result.get("exit_code", -1)
            step_result["memory_usage_mb"] = end_memory
            step_result["peak_memory_mb"] = peak_memory
            step_result["memory_delta_mb"] = end_memory - start_memory

            if args.verbose:
                logger.info(
                    "Command completed with exit code: " + str(step_result["exit_code"])
                )

        # Determine status
        for key in ("containment", "cleanup_verified", "streams_drained"):
            if key in result:
                step_result[key] = result[key]
        if result.get("status") == "TIMEOUT":
            step_result["stop_reason"] = result.get("stop_reason") or (
                "total_timeout"
                if context and context.remaining_seconds() == 0
                else "step_timeout"
            )
            step_result["force_killed"] = result.get("force_killed", False)
        step_result["status"] = _status_from_step_exit_code(
            step_result["exit_code"], step_result["dependency_warnings"]
        )
        if step_result["status"] == "FAILED":
            # Respect the child process exit code to avoid masking failures
            # Log detailed failure information
            logger.error(
                f"Step {script_name} failed with exit code {step_result['exit_code']}"
            )
            if step_result["stderr"]:
                logger.error(
                    f"Error output: {step_result['stderr'][:500]}..."
                )  # Limit to first 500 chars
        elif step_result["exit_code"] == 2:
            logger.warning(f"Step {script_name} completed with warnings")

        # Steps determine their own output directories via get_output_dir_for_script

        # Add recovery status to result
        if not step_result.get("recoverable"):
            step_result["recoverable"] = False

        return step_result

    except Exception as e:
        logger.error(f"Exception in execute_pipeline_step for {script_name}: {e}")
        step_result["status"] = "FAILED"
        step_result["exit_code"] = -1
        step_result["stderr"] = str(e)
        return step_result


def parse_step_list(step_input: Any) -> List[int]:
    """Parse step input (string or list) into list of integers."""
    if step_input is None:
        return []
    if isinstance(step_input, list):
        return [int(s) for s in step_input if str(s).isdigit() or isinstance(s, int)]
    if isinstance(step_input, str):
        try:
            return [int(s.strip()) for s in step_input.split(",") if s.strip()]
        except ValueError:
            _module_logger.debug(
                "Could not parse step list '%s' as comma-separated integers", step_input
            )
            return []
    return []


def get_environment_info() -> Dict[str, Any]:
    """Get environment information."""
    info: dict[str, Any] = {
        "python_version": sys.version,
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "working_directory": str(Path.cwd()),
        "user": os.getenv("USER", "unknown"),
    }

    try:
        import psutil

        info["memory_total_gb"] = f"{psutil.virtual_memory().total / 1024**3:.1f}"
        info["disk_free_gb"] = f"{psutil.disk_usage('/').free / 1024**3:.1f}"
    except ImportError:
        info["memory_total_gb"] = "unavailable (psutil not installed)"
        info["disk_free_gb"] = "unavailable (psutil not installed)"

    return info


def validate_pipeline_summary(summary: dict, logger: Any) -> None:
    """
    Validate pipeline summary structure and data integrity.

    Args:
        summary: Pipeline summary dictionary to validate
        logger: Logger instance for validation messages
    """
    required_fields: list[Any] = [
        "start_time",
        "arguments",
        "steps",
        "end_time",
        "overall_status",
        "total_duration_seconds",
        "environment_info",
        "performance_summary",
    ]

    # Check required fields
    for field in required_fields:
        if field not in summary:
            logger.warning(f"Pipeline summary missing required field: {field}")
        elif summary[field] is None and field not in ["end_time"]:
            logger.warning(f"Pipeline summary field '{field}' is None")

    # Validate steps structure
    steps = summary.get("steps", [])
    if not isinstance(steps, list):
        logger.error("Pipeline summary 'steps' should be a list")
        return

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            logger.error(f"Step {i} should be a dictionary")
            continue

        step_required: list[Any] = [
            "status",
            "step_number",
            "script_name",
            "description",
        ]
        for field in step_required:
            if field not in step:
                logger.warning(f"Step {i} missing required field: {field}")

        # Validate step status values
        if "status" in step:
            valid_statuses: list[Any] = [
                "SUCCESS",
                "SUCCESS_WITH_WARNINGS",
                "PARTIAL_SUCCESS",
                "FAILED",
                "TIMEOUT",
                "SKIPPED",
            ]
            if step["status"] not in valid_statuses:
                logger.warning(f"Step {i} has invalid status: {step['status']}")

        # Validate numeric fields
        numeric_fields: list[Any] = [
            "step_number",
            "exit_code",
            "retry_count",
            "duration_seconds",
            "memory_usage_mb",
            "peak_memory_mb",
            "memory_delta_mb",
        ]
        for field in numeric_fields:
            if field in step and not isinstance(step[field], (int, float)):
                logger.warning(
                    f"Step {i} field '{field}' should be numeric, got {type(step[field])}"
                )

    # Validate performance summary
    perf = summary.get("performance_summary", {})
    if not isinstance(perf, dict):
        logger.error("Performance summary should be a dictionary")
        return

    # Validate numeric fields
    numeric_fields = [
        "peak_memory_mb",
        "total_steps",
        "failed_steps",
        "critical_failures",
        "successful_steps",
        "warnings",
    ]
    for field in numeric_fields:
        if field in perf:
            if not isinstance(perf[field], (int, float)):
                logger.warning(
                    f"Performance summary field '{field}' should be numeric, got {type(perf[field])}"
                )

    # Validate overall status
    if "overall_status" in summary:
        valid_statuses = [
            "SUCCESS",
            "SUCCESS_WITH_WARNINGS",
            "PARTIAL_SUCCESS",
            "FAILED",
        ]
        if summary["overall_status"] not in valid_statuses:
            logger.warning(f"Invalid overall status: {summary['overall_status']}")

    # Validate timing consistency
    if (
        "start_time" in summary
        and "end_time" in summary
        and "total_duration_seconds" in summary
    ):
        try:
            start_dt = datetime.fromisoformat(summary["start_time"])
            end_dt = datetime.fromisoformat(summary["end_time"])
            calculated_duration = (end_dt - start_dt).total_seconds()
            reported_duration = summary["total_duration_seconds"]

            # Allow for small timing differences due to processing
            if abs(calculated_duration - reported_duration) > 1.0:
                logger.warning(
                    f"Timing inconsistency: calculated {calculated_duration:.2f}s vs reported {reported_duration:.2f}s"
                )
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not validate timing: {e}")


# All pipeline utility functions have been moved to appropriate utils modules:
# - validate_step_prerequisites, validate_pipeline_step_sequence → utils/pipeline_orchestration/pipeline_validator.py
# - get_current_memory_usage → utils/runtime_safety/resource_manager.py
# - attempt_step_recovery and recovery functions → utils/errors/error_recovery.py
# - generate_pipeline_health_report → utils/pipeline_orchestration/pipeline_monitor.py
