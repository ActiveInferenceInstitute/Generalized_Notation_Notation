#!/usr/bin/env python3
"""Live wiring of run sessions + durable streams into the main.py composition.

Bridges :mod:`pipeline.run_session` (resumable run-session manifests) and
:mod:`pipeline.run_manifest` (durable stream manifests + replayable execution
traces) onto the live 25-step pipeline composed in :mod:`gnn.main`. Every
public function here is pure orchestration over an already-supplied
:class:`RunSession`: it NEVER executes a pipeline step, runs a container, or
calls a cluster. It only creates/updates/checkpoints session units and, at
close time, emits and re-verifies run manifests from artifacts that the
already-completed run wrote to disk.

An invocation with a ``RunContext`` requires verified session checkpoints:
wiring errors propagate to the common failure finalizer. Standalone existing
callers retain advisory guards. Terminal pipeline evidence is staged and
verified by :mod:`pipeline.finalization` before success publication.

Checkpoint location is canonical: ``<output_dir>/00_pipeline_summary/run_session.json``
(next to ``pipeline_execution_summary.json``). Manifest emission at close
writes ``<output_dir>/v3_run_manifest/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

from gnn.pipeline import run_session as rs
from gnn.pipeline.config import get_output_dir_for_script
from gnn.pipeline.run_session import RunSession, UnitStatus, WorkUnit

if TYPE_CHECKING:
    from gnn.utils.arguments.pipeline_arguments import PipelineArguments

__all__ = [
    "close_run_session",
    "close_run_session_guarded",
    "mark_units_running",
    "mark_units_running_guarded",
    "open_run_session",
    "open_run_session_guarded",
    "record_step_result",
    "record_step_result_guarded",
    "run_session_path",
]


def run_session_path(output_dir: Union[str, Path]) -> Path:
    """Return the canonical run-session checkpoint path for a run output dir.

    Args:
        output_dir: Base output directory of the pipeline run.

    Returns:
        ``<output_dir>/00_pipeline_summary/run_session.json``.
    """
    return Path(output_dir) / "00_pipeline_summary" / "run_session.json"


def open_run_session(
    args: "PipelineArguments",
    steps_to_execute: List[Any],
    pipeline_summary: Dict[str, Any],
) -> RunSession:
    """Open a fresh run session covering exactly the selected pipeline steps.

    Builds one PENDING :class:`WorkUnit` per selected step, ids the session
    from the run identity (``run_id`` when present, else a ``run-<hash12>``
    fallback derived from ``run_hash``), and checkpoints the session
    immediately so a crash before the first step still leaves a resumable
    manifest on disk.

    Args:
        args: Pipeline arguments (only ``output_dir`` is read).
        steps_to_execute: Selected ``(script_name, description)`` tuples, in
            execution order.
        pipeline_summary: The live pipeline summary dict (read for ``run_id``
            and ``run_hash``).

    Returns:
        The opened, checkpointed session.
    """
    units = [
        WorkUnit(
            unit_id=script_name,
            steps=[int(script_name.split("_", 1)[0])],
            status=UnitStatus.PENDING,
        )
        for script_name, _description in steps_to_execute
    ]
    session_id = str(
        pipeline_summary.get("run_id")
        or f"run-{str(pipeline_summary.get('run_hash'))[:12]}"
    )
    session = rs.start_session(session_id, units, created_by="gnn.main")
    rs.checkpoint(session, run_session_path(args.output_dir))
    return session


def mark_units_running(
    session: RunSession,
    script_names: Iterable[str],
    output_dir: Union[str, Path],
) -> RunSession:
    """Mark the given units RUNNING and checkpoint the session.

    Args:
        session: The session to update.
        script_names: Unit ids (script names) transitioning to RUNNING.
        output_dir: Base output directory of the pipeline run.

    Returns:
        The updated session (input unchanged; :func:`run_session.mark` copies).

    Raises:
        KeyError: If any unit id is not present in the session.
    """
    updated = session
    for name in script_names:
        updated = rs.mark(updated, name, UnitStatus.RUNNING)
    rs.checkpoint(updated, run_session_path(output_dir))
    return updated


def record_step_result(
    session: RunSession,
    script_name: str,
    step_result: Dict[str, Any],
    output_dir: Union[str, Path],
) -> RunSession:
    """Fold one finished step's result into the session and checkpoint.

    Status mapping follows the pipeline step-status contract: ``SUCCESS`` and
    ``SUCCESS_WITH_WARNINGS`` map to ``DONE``; ``SKIPPED`` maps to
    ``SKIPPED``; anything else maps to ``FAILED`` with the step's error (or a
    fallback naming the unexpected status). Artifact provenance is honest:
    the unit's conventional step output dir is referenced only when that
    directory actually exists under ``output_dir``.

    Call this AFTER ``main._record_step_result`` so the warning-upgrade
    (``SUCCESS`` -> ``SUCCESS_WITH_WARNINGS``) is already applied to
    ``step_result``.

    Args:
        session: The session to update.
        script_name: Unit id (script name) of the finished step.
        step_result: The recorded step result dict (needs at least ``status``).
        output_dir: Base output directory of the pipeline run.

    Returns:
        The updated, checkpointed session.

    Raises:
        KeyError: If ``script_name`` is not present in the session.
    """
    status_raw = str(step_result.get("status", ""))
    step_dir = get_output_dir_for_script(script_name, Path(output_dir))
    artifact_refs = [step_dir.name] if step_dir.is_dir() else []

    if status_raw in ("SUCCESS", "SUCCESS_WITH_WARNINGS"):
        updated = rs.mark(
            session, script_name, UnitStatus.DONE, artifact_refs=artifact_refs
        )
    elif status_raw == "SKIPPED":
        updated = rs.mark(
            session, script_name, UnitStatus.SKIPPED, artifact_refs=artifact_refs
        )
    else:
        error = str(step_result.get("error") or f"step status {status_raw!r}")
        updated = rs.mark(
            session,
            script_name,
            UnitStatus.FAILED,
            artifact_refs=artifact_refs,
            error=error,
        )
    unit = next(unit for unit in updated.units if unit.unit_id == script_name)
    unit.input_identity = {
        "run_id": step_result.get("run_id", session.session_id),
        "model_ids": step_result.get("selected_model_ids", []),
    }
    unit.artifact_hashes = {
        record["path"]: record["sha256"] for record in step_result.get("artifacts", [])
    }
    rs.checkpoint(updated, run_session_path(output_dir))
    return updated


def close_run_session(
    session: RunSession,
    output_dir: Union[str, Path],
    logger: logging.Logger,
) -> RunSession:
    """Emit + verify durable run manifests, then checkpoint the final session.

    Manifest emission is best-effort: any exception (missing summary, bad
    artifacts, import failure of the numpy-adjacent machinery) downgrades to
    a logged warning and never raises — the run's exit code is decided
    elsewhere. Verification problems are logged individually as warnings.
    The session is always re-checkpointed (idempotent final state) before
    returning.

    Args:
        session: The session to close.
        output_dir: Base output directory of the completed run.
        logger: Logger for the receipt/warning lines.

    Returns:
        The unchanged session (checkpointed to disk).
    """
    from gnn.pipeline.run_context import current_run_context

    if current_run_context() is not None:
        # Top-level orchestration owns mandatory manifest verification and the
        # terminal verdict. Closing a session must not emit it a second time.
        rs.checkpoint(session, run_session_path(output_dir))
        return session
    try:
        from gnn.pipeline.run_manifest import emit_run_manifests, verify_run_manifests

        emission = emit_run_manifests(output_dir)
        problems = verify_run_manifests(emission["manifest_dir"], output_dir)
        for problem in problems:
            logger.warning("Run manifest problem: %s", problem)
        logger.info(
            "Run manifests emitted: streams=%d trace_events=%d integrity_ok=%s dir=%s",
            emission["stream_count"],
            emission["trace_event_count"],
            emission["trace_integrity_ok"],
            emission["manifest_dir"],
        )
    except Exception as e:
        logger.warning(f"Run manifest emission skipped (continuing): {e}")
    finally:
        rs.checkpoint(session, run_session_path(output_dir))
    return session


def open_run_session_guarded(
    args: "PipelineArguments",
    steps_to_execute: List[Any],
    pipeline_summary: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[RunSession]:
    """Open a required invocation checkpoint, or an advisory standalone session."""
    try:
        session = open_run_session(args, steps_to_execute, pipeline_summary)
        logger.info(
            "Run session opened: %s (%d unit(s))",
            session.session_id,
            len(session.units),
        )
        return session
    except Exception as open_err:
        from gnn.pipeline.run_context import current_run_context

        if current_run_context() is not None:
            raise
        logger.warning(f"Run session open failed (continuing): {open_err}")
        return None


def mark_units_running_guarded(
    session: Optional[RunSession],
    script_names: Iterable[str],
    args: "PipelineArguments",
    logger: logging.Logger,
) -> Optional[RunSession]:
    """Mark units RUNNING; invocation errors propagate, standalone errors warn."""
    if session is None:
        return None
    try:
        return mark_units_running(session, script_names, args.output_dir)
    except Exception as e:
        from gnn.pipeline.run_context import current_run_context

        if current_run_context() is not None:
            raise
        logger.warning(f"Run session update failed (continuing): {e}")
        return session


def record_step_result_guarded(
    session: Optional[RunSession],
    script_name: str,
    step_result: Dict[str, Any],
    args: "PipelineArguments",
    logger: logging.Logger,
) -> Optional[RunSession]:
    """Bind a step receipt; invocation errors propagate, standalone errors warn."""
    if session is None:
        return None
    try:
        return record_step_result(session, script_name, step_result, args.output_dir)
    except Exception as e:
        from gnn.pipeline.run_context import current_run_context

        if current_run_context() is not None:
            raise
        logger.warning(f"Run session update failed (continuing): {e}")
        return session


def close_run_session_guarded(
    session: Optional[RunSession],
    args: "PipelineArguments",
    logger: logging.Logger,
) -> Optional[RunSession]:
    """Close a session; invocation errors propagate, standalone errors warn."""
    if session is None:
        return None
    try:
        return close_run_session(session, args.output_dir, logger)
    except Exception as close_err:
        from gnn.pipeline.run_context import current_run_context

        if current_run_context() is not None:
            raise
        logger.warning(f"Run session close failed (continuing): {close_err}")
        return session
