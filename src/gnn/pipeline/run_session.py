#!/usr/bin/env python3
"""Resumable run-session management for long-running pipeline acceptance runs.

Pure data layer for orchestrating extended model-family acceptance runs across
many work units. Provides resumable run manifests, status inspection, and
cancellation-safe cleanup. This module NEVER executes containers, calls
docker/kubectl, opens sockets, or mutates any live infrastructure — it only
generates, validates, serializes, replays, and cleans up its own local manifest
and artifact files under an explicitly provided working directory.

All checkpoint writes are atomic (write to a temp file in the same directory,
then ``os.replace``) so an interrupted write never corrupts an existing
manifest. No wall-clock values participate in the run hash.
"""

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, Field


class UnitStatus(str, Enum):
    """Lifecycle status of a single work unit within a run session."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkUnit(BaseModel):
    """A single resumable unit of work within a run session."""

    unit_id: str
    steps: List[int] = Field(default_factory=list)
    status: UnitStatus = UnitStatus.PENDING
    artifact_refs: List[str] = Field(default_factory=list)
    error: str = ""
    input_identity: Dict[str, Any] = Field(default_factory=dict)
    artifact_hashes: Dict[str, str] = Field(default_factory=dict)


class RunSession(BaseModel):
    """A resumable manifest of work units for a long-running acceptance run."""

    session_id: str
    created_by: str = ""
    run_hash: str = ""
    units: List[WorkUnit] = Field(default_factory=list)
    schema_version: str = "1.0"
    final_status: Optional[str] = None
    evidence_integrity: Dict[str, Any] = Field(default_factory=dict)


def _compute_session_hash(unit_ids: List[str], hash_length: int = 12) -> str:
    """Compute a deterministic content hash from sorted unit ids.

    Args:
        unit_ids: The unit ids to hash (order-independent).
        hash_length: Number of leading hex characters to return.

    Returns:
        A deterministic hex string derived from the sorted unit ids.
    """
    hasher = hashlib.sha256()
    for uid in sorted(unit_ids):
        hasher.update(uid.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()[:hash_length]


def start_session(
    session_id: str,
    units: Sequence[Union[str, WorkUnit]],
    created_by: str = "",
) -> RunSession:
    """Create a fresh run session from unit ids or WorkUnit instances.

    Args:
        session_id: Stable identifier for this session.
        units: A list of unit ids (str) and/or WorkUnit instances.
        created_by: Optional provenance label for who/what created the session.

    Returns:
        A new RunSession with a deterministic run_hash over the sorted unit ids.

    Raises:
        ValueError: if any unit_id is duplicated. ``mark`` updates the first match
            only, so a duplicate id would leave a second unit permanently
            unreachable (it could never complete, and resume would loop forever).
    """
    work_units: List[WorkUnit] = []
    for u in units:
        if isinstance(u, WorkUnit):
            work_units.append(u.model_copy(deep=True))
        else:
            work_units.append(WorkUnit(unit_id=u))

    unit_ids = [wu.unit_id for wu in work_units]
    duplicates = sorted({uid for uid in unit_ids if unit_ids.count(uid) > 1})
    if duplicates:
        raise ValueError(f"duplicate unit_id(s) not allowed in a session: {duplicates}")
    run_hash = _compute_session_hash(unit_ids)
    return RunSession(
        session_id=session_id,
        created_by=created_by,
        run_hash=run_hash,
        units=work_units,
    )


def mark(
    session: RunSession,
    unit_id: str,
    status: UnitStatus,
    artifact_refs: Optional[List[str]] = None,
    error: str = "",
) -> RunSession:
    """Return a new session with the given unit updated (non-destructive copy).

    Args:
        session: The session to update.
        unit_id: The unit to mark.
        status: New status for the unit.
        artifact_refs: If provided, replaces the unit's artifact_refs.
        error: Optional error message to attach.

    Returns:
        A new RunSession with the targeted unit updated; the input is unchanged.

    Raises:
        KeyError: If unit_id is not present in the session.
    """
    updated = session.model_copy(deep=True)
    for wu in updated.units:
        if wu.unit_id == unit_id:
            wu.status = status
            if artifact_refs is not None:
                wu.artifact_refs = list(artifact_refs)
            wu.error = error
            return updated
    raise KeyError(f"unit_id not found in session: {unit_id!r}")


def checkpoint(session: RunSession, path: Union[str, Path]) -> Path:
    """Atomically write the session JSON to ``path``.

    Writes through the shared :mod:`pipeline._io` atomic-write helper (unique
    temp file in the same directory, then ``os.replace``), so an interrupted
    write never corrupts an existing manifest.

    Args:
        session: The session to persist.
        path: Destination path for the manifest JSON.

    Returns:
        The resolved destination Path.
    """
    from gnn.pipeline._io import atomic_write_text

    dest = Path(path)
    atomic_write_text(dest, session.model_dump_json(indent=2))
    return dest


def load_session(path: Union[str, Path]) -> RunSession:
    """Load a run session from a checkpoint file.

    Args:
        path: Path to a manifest written by ``checkpoint``.

    Returns:
        The deserialized RunSession.
    """
    text = Path(path).read_text(encoding="utf-8")
    return RunSession.model_validate_json(text)


def remaining_units(session: RunSession) -> List[str]:
    """Return unit ids that still need work (PENDING, RUNNING, or FAILED).

    Args:
        session: The session to inspect.

    Returns:
        Ordered list of unit ids whose status is PENDING, RUNNING, or FAILED.
    """
    return [
        wu.unit_id
        for wu in session.units
        if wu.status in (UnitStatus.PENDING, UnitStatus.RUNNING, UnitStatus.FAILED)
    ]


def resume_plan(session: RunSession) -> List[str]:
    """Return the resume targets for a session.

    Documents resume intent: a resumed run should process exactly the units
    returned here (those not yet DONE/SKIPPED, i.e. PENDING, RUNNING, or FAILED).

    Args:
        session: The session to inspect.

    Returns:
        The list of unit ids to (re)process on resume.
    """
    return remaining_units(session)


def status_report(session: RunSession) -> dict:
    """Summarize session progress.

    Args:
        session: The session to inspect.

    Returns:
        A dict with keys: total, by_status (per-UnitStatus counts), completed
        (count of DONE units), percent_complete (0-100 float), and done (bool).
    """
    total = len(session.units)
    by_status = {status.value: 0 for status in UnitStatus}
    for wu in session.units:
        by_status[wu.status.value] += 1

    completed = by_status[UnitStatus.DONE.value]
    percent_complete = (completed / total * 100.0) if total else 0.0
    return {
        "total": total,
        "by_status": by_status,
        "completed": completed,
        "percent_complete": percent_complete,
        "done": total > 0
        and completed == total
        and (
            session.created_by != "gnn.main"
            or session.final_status in ("SUCCESS", "SUCCESS_WITH_WARNINGS")
        ),
        "final_status": session.final_status,
        "evidence_integrity": session.evidence_integrity,
    }


def _safe_resolve_under(workdir: Path, ref: str) -> Optional[Path]:
    """Resolve ``ref`` under ``workdir``, returning None if it escapes.

    Args:
        workdir: The directory artifacts must stay within.
        ref: A (possibly relative or malicious) artifact reference.

    Returns:
        The resolved Path if it lies inside workdir, otherwise None.
    """
    base = workdir.resolve()
    candidate = (base / ref).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


def cancel_safe_cleanup(session: RunSession, workdir: Union[str, Path]) -> List[str]:
    """Delete artifacts of non-DONE units under ``workdir``; preserve DONE.

    For every unit NOT in DONE status, delete its artifact_refs files under
    ``workdir`` if present. Path references are resolved safely and any ref that
    would escape ``workdir`` is ignored (never deleted). Idempotent: a second
    call removes nothing more and does not error.

    Args:
        session: The session whose non-DONE artifacts should be cleaned up.
        workdir: The directory artifacts are confined to.

    Returns:
        The list of removed file paths (as strings), in deletion order.
    """
    base = Path(workdir)
    removed: List[str] = []
    if not base.exists():
        return removed

    protected = {
        resolved
        for unit in session.units
        if unit.status == UnitStatus.DONE
        for ref in unit.artifact_refs
        if (resolved := _safe_resolve_under(base, ref)) is not None
    }
    for wu in session.units:
        if wu.status == UnitStatus.DONE:
            continue
        for ref in wu.artifact_refs:
            resolved = _safe_resolve_under(base, ref)
            if resolved is None or any(
                resolved == keep or resolved.is_relative_to(keep) for keep in protected
            ):
                continue
            if resolved.is_file():
                resolved.unlink()
                removed.append(str(resolved))
    return removed
