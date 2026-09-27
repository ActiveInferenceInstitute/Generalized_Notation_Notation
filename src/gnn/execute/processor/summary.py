#!/usr/bin/env python3
"""Step 12 aggregate summary assembly, classification, and persistence."""

import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from gnn.execute.metadata import (
    _atomic_execution_json,
    _merge_prior_execution_summary,
    _slim_execution_detail,
    generate_execution_report,
)
from gnn.execute.types import ExecutionOutcome

def _init_execution_summary(
    target_dir: Path,
    output_dir: Path,
    execution_benchmark_repeats: int,
    execution_summary_detail: bool,
) -> Dict[str, Any]:
    """Build the empty Step 12 execution summary envelope."""
    return {
        "timestamp": datetime.now().isoformat(),
        "target_directory": str(target_dir),
        "output_directory": str(output_dir),
        "total_scripts_found": 0,
        "successful_executions": 0,
        "failed_executions": 0,
        "skipped_executions": 0,
        "execution_details": [],
        "framework_status": {},
        "execution_mode": "local",
        "execution_workers": 1,
        "backend": None,
        "execution_benchmark_repeats": execution_benchmark_repeats,
        "execution_summary_detail": execution_summary_detail,
        "dispatch_max_retries": 0,
        "success": False,
        "status": "pending",
        "exit_code": None,
    }


def _update_framework_status(
    execution_results: Dict[str, Any], details: List[Dict[str, Any]]
) -> None:
    """Fold per-script results into aggregate counters and per-framework status."""
    for exec_result in details:
        execution_results["execution_details"].append(exec_result)

        # Update framework status
        framework = exec_result.get("framework", "unknown")
        if framework not in execution_results["framework_status"]:
            execution_results["framework_status"][framework] = {
                "status": "unknown",
                "executions": 0,
                "successful": 0,
                "failed": 0,
                "skipped": 0,
            }

        framework_summary = execution_results["framework_status"][framework]
        framework_summary["executions"] += 1

        if exec_result.get("skipped"):
            execution_results["skipped_executions"] = (
                execution_results.get("skipped_executions", 0) + 1
            )
            framework_summary["skipped"] += 1
            if "error" in exec_result:
                framework_summary["error"] = exec_result["error"]
        elif exec_result["success"]:
            execution_results["successful_executions"] += 1
            framework_summary["successful"] += 1
        else:
            execution_results["failed_executions"] += 1
            framework_summary["failed"] += 1
            if "error" in exec_result:
                framework_summary["error"] = exec_result["error"]

    for framework_summary in execution_results["framework_status"].values():
        if framework_summary["failed"]:
            framework_summary["status"] = "failed"
        elif framework_summary["successful"] and framework_summary["skipped"]:
            framework_summary["status"] = "success_with_skips"
        elif framework_summary["successful"]:
            framework_summary["status"] = "success"
        else:
            framework_summary["status"] = "skipped"


def _classify_execute_outcome(
    *,
    total_found: int,
    successful: int,
    failed: int,
    skipped: int,
    render_failures: List[Dict[str, str]],
    missing_render_scripts: List[str],
    missing_render_summary: Optional[str],
    strict_requested_frameworks: bool,
    unsupported_receipts: Optional[List[Dict[str, str]]] = None,
) -> ExecutionOutcome:
    """Classify a finished Step 12 run into its outcome contract.

    Pure function of the run counters: the durable summary and the API result
    are derived from the same classification, so they can never disagree.
    """
    unsupported_receipts = list(unsupported_receipts or [])
    attempted = total_found - skipped
    if missing_render_summary:
        outcome: Union[bool, int] = False
        status = "failed"
        reason = "required_render_summary_missing"
    elif strict_requested_frameworks and render_failures:
        outcome = False
        status = "failed"
        reason = "requested_framework_render_failure"
    elif missing_render_scripts:
        outcome = False
        status = "failed"
        reason = "rendered_script_missing"
    elif total_found == 0:
        outcome = False if strict_requested_frameworks else 2
        status = "failed" if strict_requested_frameworks else "skipped"
        if unsupported_receipts and not render_failures:
            reason = "unsupported_render_refusals"
        else:
            reason = "no_executable_scripts"
    elif strict_requested_frameworks and (failed > 0 or skipped > 0):
        outcome = False
        status = "failed"
        reason = "requested_framework_execution_incomplete"
    elif failed > 0:
        outcome = False
        status = "failed"
        reason = "script_execution_failure"
    elif skipped > 0:
        outcome = True
        status = "success_with_skips"
        reason = "optional_dependencies_unavailable"
    elif render_failures:
        outcome = True
        status = "success_with_render_failures"
        reason = "best_effort_render_subset_executed"
    else:
        outcome = True
        status = "success"
        reason = "all_scripts_succeeded"

    exit_code = 0 if outcome is True else outcome if outcome == 2 else 1
    return ExecutionOutcome(
        outcome=outcome,
        status=status,
        reason=reason,
        exit_code=exit_code,
        attempted=attempted,
    )


def _write_execution_summaries(
    results_dir: Path,
    execution_results: Dict[str, Any],
    execution_summary_detail: bool,
    logger: Any,
) -> None:
    """Persist the slim aggregate (+ optional detail) and regenerate the report.

    The slim aggregate is what lands on disk; full per-script payloads are
    restored on ``execution_results`` afterwards for in-memory consumers.
    """
    summaries_dir = results_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    results_file = summaries_dir / "execution_summary.json"

    # The pipeline invokes this step once per top-level input folder, all
    # writing the same summary file. Carry the earlier folders' script
    # results forward so the durable summary covers every executed script
    # (mirrors the Step 11 render-summary merge).
    _merge_prior_execution_summary(execution_results, results_file, logger)

    full_details_snapshot = copy.deepcopy(execution_results["execution_details"])
    execution_results["execution_details"] = [
        _slim_execution_detail(d) for d in full_details_snapshot
    ]
    execution_results["execution_summary_format"] = "slim_v1"

    try:
        if execution_summary_detail:
            detail_path = summaries_dir / "execution_summary_detail.json"
            detail_payload = dict(execution_results)
            detail_payload["execution_details"] = full_details_snapshot
            detail_payload["execution_summary_format"] = "detail_v1"
            _atomic_execution_json(detail_path, detail_payload)
        _atomic_execution_json(results_file, execution_results)
        generate_execution_report(execution_results, results_dir, logger)
    finally:
        execution_results["execution_details"] = full_details_snapshot

