#!/usr/bin/env python3
"""Shared pipeline execution plumbing for the API surfaces.

Both FastAPI surfaces — ``api.app`` (run/runs) and ``api.server``
(process/jobs/tools) — delegate real execution to the same orchestrator,
``src/gnn/main.py``, and read the same canonical summary
(``<output_dir>/00_pipeline_summary/pipeline_execution_summary.json``).

This module is the single home for that shared contract so the two surfaces
cannot drift:

- :func:`build_pipeline_command` is a pure argv builder for one real
  orchestrator invocation (no filesystem access, no side effects).
- :func:`read_pipeline_summary` reads the canonical run summary tolerantly.
- :func:`normalize_summary_steps` normalizes raw summary entries into the
  small typed :class:`StepOutcome` record both surfaces consume.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from gnn.pipeline.admission import validate_boolean, validate_steps
from gnn.pipeline.step_registry import STEPS, get_llm_steps
from gnn.utils.pipeline_orchestration.pipeline_step_dependencies import (
    resolve_step_dependencies,
)

logger = logging.getLogger(__name__)

# Canonical step surface, derived once from the pipeline step registry so no
# API surface hardcodes step counts or the LLM-step set (single source of
# truth; adding step 25 or renumbering LLM updates the API automatically).
PIPELINE_STEP_COUNT: int = len(STEPS)
VALID_STEP_NUMBERS: frozenset[int] = frozenset(
    int(s.script_stem.partition("_")[0]) for s in STEPS
)
MAX_PIPELINE_STEP: int = max(VALID_STEP_NUMBERS)
LLM_STEP_NUMBERS: frozenset[int] = frozenset(
    int(s.script_stem.partition("_")[0]) for s in get_llm_steps()
)

#: Location of the orchestrator script relative to the repository root.
MAIN_SCRIPT = Path("src/gnn") / "main.py"

#: Location of the canonical run summary relative to an output directory.
PIPELINE_SUMMARY = Path("00_pipeline_summary") / "pipeline_execution_summary.json"


@dataclass(frozen=True)
class StepOutcome:
    """Normalized view of one step entry in the canonical pipeline summary."""

    script_name: str
    step_num: int
    status: str
    duration_seconds: float


def build_pipeline_command(
    target_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    only_steps: Optional[List[int]] = None,
    skip_steps: Optional[List[int]] = None,
    verbose: bool = False,
    strict: bool = False,
    parallel: bool = False,
    consolidated_steps: bool = False,
    repo_root: Optional[Path] = None,
) -> List[str]:
    """Return the argv for one real ``src/gnn/main.py`` orchestrator invocation.

    Pure function: no filesystem access and no side effects. ``only_steps``
    and ``skip_steps`` are emitted sorted and comma-joined so identical
    selections always produce identical argv.

    Args:
        target_dir: Directory containing GNN files (passed verbatim).
        output_dir: Directory for pipeline outputs (passed verbatim).
        only_steps: Specific steps to run (``None`` = all).
        skip_steps: Steps to skip.
        verbose: Enable orchestrator verbose output.
        strict: Treat orchestrator warnings as errors.
        repo_root: Explicit checkout script override. By default module execution
            preserves the API workspace cwd for configuration and relative paths.

    Returns:
        The full argv list, starting with the current interpreter.
    """
    only_steps = validate_steps(only_steps, field_name="steps", allow_empty=False)
    skip_steps = validate_steps(skip_steps, field_name="skip_steps")
    validate_boolean(verbose, field_name="verbose")
    validate_boolean(strict, field_name="strict")
    validate_boolean(parallel, field_name="parallel")
    validate_boolean(consolidated_steps, field_name="consolidated_steps")
    if set(only_steps or ()) & set(skip_steps or ()):
        raise ValueError("Pipeline steps cannot be both requested and skipped")
    command: List[str] = (
        [sys.executable, str(repo_root.resolve() / MAIN_SCRIPT)]
        if repo_root is not None
        else [sys.executable, "-P", "-m", "gnn.main"]
    )
    command += [
        "--target-dir",
        str(target_dir),
        "--output-dir",
        str(output_dir),
    ]
    if only_steps:
        command += ["--only-steps", ",".join(str(step) for step in sorted(only_steps))]
    if skip_steps:
        command += ["--skip-steps", ",".join(str(step) for step in sorted(skip_steps))]
    if verbose:
        command.append("--verbose")
    if strict:
        command.append("--strict")
    if parallel:
        command.append("--parallel")
    if consolidated_steps:
        command.append("--consolidated-steps")
    return command


def planned_step_numbers(
    only_steps: Optional[List[int]], skip_steps: Optional[List[int]] = None
) -> List[int]:
    """Resolve the same prerequisites and skip set as the owned orchestrator."""
    requested = validate_steps(only_steps, field_name="steps", allow_empty=False)
    skipped = validate_steps(skip_steps, field_name="skip_steps")
    selected = (
        resolve_step_dependencies(requested)
        if requested is not None
        else sorted(VALID_STEP_NUMBERS)
    )
    return [number for number in selected if number not in set(skipped or ())]


def read_pipeline_summary(
    output_dir: Union[str, Path],
    *,
    not_before_ns: Optional[int] = None,
    expected_run_id: Optional[str] = None,
) -> Optional[List[Any]]:
    """Read the canonical pipeline execution summary, tolerantly.

    Returns the raw ``steps`` list, or ``None`` when the summary is missing,
    unreadable, structurally wrong, too old, or from a different requested
    invocation — callers treat ``None`` as "no
    summary is available" and must keep prior state untouched.
    """
    summary_path = Path(output_dir) / PIPELINE_SUMMARY
    if not summary_path.is_file():
        logger.warning("Pipeline summary not found: %s", summary_path)
        return None
    try:
        if (
            not_before_ns is not None
            and summary_path.stat().st_mtime_ns < not_before_ns
        ):
            logger.warning(
                "Ignoring summary from an earlier invocation: %s", summary_path
            )
            return None
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read pipeline summary %s: %s", summary_path, exc)
        return None
    if not isinstance(summary, dict):
        logger.warning("Pipeline summary has non-object root: %s", summary_path)
        return None
    if expected_run_id is not None and summary.get("run_id") != expected_run_id:
        logger.warning("Ignoring summary from a different invocation: %s", summary_path)
        return None
    steps = summary.get("steps", [])
    if not isinstance(steps, list):
        logger.warning("Pipeline summary has non-list steps: %s", summary_path)
        return None
    return steps


def pipeline_exit_succeeded(exit_code: Optional[int], *, strict: bool = False) -> bool:
    """Accept rc0 and non-strict rc2; missing and other exit codes fail."""
    return exit_code == 0 or (exit_code == 2 and not strict)


def normalize_summary_steps(steps: List[Any]) -> List[StepOutcome]:
    """Normalize raw summary step entries into typed outcomes.

    Tolerates the malformed shapes the summary can contain: non-dict entries
    are skipped, a missing ``step_num`` falls back to the digit prefix of
    ``script_name`` (or the entry index), and unparsable durations become
    ``0.0``.
    """
    outcomes: List[StepOutcome] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        script_name = str(step.get("script_name") or step.get("name") or "step")
        raw_step_num: Any = step.get("step_num")
        if raw_step_num is None:
            prefix = script_name.split("_", 1)[0]
            raw_step_num = prefix if prefix.isdigit() else index
        try:
            step_num = int(raw_step_num)
        except (TypeError, ValueError):
            step_num = index
        status = str(step.get("status", "UNKNOWN"))
        try:
            duration = float(step.get("duration_seconds", 0.0) or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        outcomes.append(
            StepOutcome(
                script_name=script_name,
                step_num=step_num,
                status=status,
                duration_seconds=duration,
            )
        )
    return outcomes


def summarize_step_progress(outcomes: List[StepOutcome]) -> Dict[str, List[int]]:
    """Classify outcomes into ``steps_completed`` / ``steps_failed`` numbers.

    Steps reporting a ``SUCCESS*`` status count as completed, ``FAILED`` as
    failed; every other status (e.g. ``SKIPPED``) is intentionally left
    uncounted so callers can distinguish "did not run" from "ran badly".
    """
    return {
        "steps_completed": [
            outcome.step_num
            for outcome in outcomes
            if outcome.status.startswith("SUCCESS")
        ],
        "steps_failed": [
            outcome.step_num for outcome in outcomes if outcome.status == "FAILED"
        ],
    }
