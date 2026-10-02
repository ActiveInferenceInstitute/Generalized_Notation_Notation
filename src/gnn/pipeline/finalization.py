"""Stage and verify terminal run evidence before publishing success claims."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from gnn.pipeline._io import atomic_write_bytes, atomic_write_text
from gnn.pipeline.run_session import (
    RunSession,
    UnitStatus,
    checkpoint,
    load_session,
    start_session,
)
from gnn.pipeline.run_session_wiring import run_session_path


def _unit_status(record: dict[str, Any]) -> UnitStatus:
    status = record.get("status")
    if status in ("SUCCESS", "SUCCESS_WITH_WARNINGS"):
        return UnitStatus.DONE
    return UnitStatus.SKIPPED if status == "SKIPPED" else UnitStatus.FAILED


def _verified_session(
    session: RunSession | None, output_dir: Path, summary: dict[str, Any]
) -> RunSession:
    """Bind the checkpoint to the actual selected units and byte receipts."""
    if session is None or session.session_id != summary.get("run_id"):
        raise ValueError(
            "Current-run session is unavailable or has a different identity"
        )
    if load_session(run_session_path(output_dir)).model_dump() != session.model_dump():
        raise ValueError("Current-run session checkpoint changed before finalization")
    expected_steps = summary.get("planned_steps", [])
    if [unit.unit_id for unit in session.units] != expected_steps:
        raise ValueError("Current-run session does not cover the planned steps")
    records = {record["script_name"]: record for record in summary.get("steps", [])}
    if len(records) != len(summary.get("steps", [])) or set(records) != set(
        expected_steps
    ):
        raise ValueError("Current-run session has missing or duplicate step receipts")
    for unit in session.units:
        record = records[unit.unit_id]
        if unit.status != _unit_status(record):
            raise ValueError(
                f"Session status differs from step receipt: {unit.unit_id}"
            )
        expected_identity = {
            "run_id": summary["run_id"],
            "model_ids": record.get("selected_model_ids", []),
        }
        if unit.input_identity != expected_identity:
            raise ValueError(f"Session source selection differs: {unit.unit_id}")
        expected_hashes = {
            artifact["path"]: artifact["sha256"]
            for artifact in record.get("artifacts", [])
        }
        if unit.artifact_hashes != expected_hashes:
            raise ValueError(f"Session artifact bindings differ: {unit.unit_id}")
    terminal = session.model_copy(deep=True)
    terminal.final_status = summary["overall_status"]
    terminal.evidence_integrity = dict(summary["evidence_integrity"])
    return terminal


def _replace_manifest_directory(output_dir: Path, staged: Path) -> None:
    """Preserve the previous manifest tree before promoting the verified tree."""
    destination = output_dir / "v3_run_manifest"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            raise ValueError("Manifest directory cannot be a symlink")
        archive = (
            output_dir
            / "00_pipeline_summary"
            / ".history"
            / f"manifests-{uuid.uuid4().hex}"
        )
        if not archive.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError("Manifest history directory escapes its output root")
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(destination, archive)
    os.replace(staged, destination)


def finalize_run_evidence(
    output_dir: Path, summary: dict[str, Any], session: RunSession | None
) -> None:
    """Verify a staged summary/session/manifest bundle, then promote evidence.

    The caller publishes the final summary and derived reports only after this
    returns. The output lease remains held throughout promotion. A later failure
    must invalidate every terminal claim using ``record_failed_evidence``.
    """
    from gnn.pipeline.artifact_ownership import verify_owned_artifacts
    from gnn.pipeline.run_context import current_run_context
    from gnn.pipeline.run_manifest import emit_run_manifests, verify_run_manifests

    context = current_run_context()
    if context is None:
        raise ValueError("Terminal pipeline evidence requires an invocation context")
    if summary.get("run_id") != context.run_id:
        raise ValueError("Terminal summary belongs to a different invocation")
    if output_dir.resolve() != Path(context.output_root).resolve():
        raise ValueError("Terminal evidence output differs from invocation output root")
    context.raise_if_expired()
    staged = (
        output_dir
        / "00_pipeline_summary"
        / "finalization"
        / context.run_id
        / uuid.uuid4().hex
    )
    if not staged.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError("Finalization staging directory escapes its output root")
    staged.mkdir(parents=True)
    terminal = _verified_session(session, output_dir, summary)
    session_candidate = staged / "run_session.json"
    checkpoint(terminal, session_candidate)
    if load_session(session_candidate).model_dump() != terminal.model_dump():
        raise ValueError("Staged session did not preserve the verified checkpoint")
    summary["run_session_sha256"] = hashlib.sha256(
        session_candidate.read_bytes()
    ).hexdigest()
    summary_candidate = staged / "pipeline_execution_summary.json"
    atomic_write_text(summary_candidate, json.dumps(summary, indent=4, default=str))
    emission = emit_run_manifests(
        output_dir,
        manifest_out=staged / "v3_run_manifest",
        summary_path=summary_candidate,
    )
    problems = verify_run_manifests(
        emission["manifest_dir"], output_dir, summary_path=summary_candidate
    )
    if problems or not emission["trace_integrity_ok"]:
        raise ValueError(
            f"Durable current-run manifest verification failed: {problems}"
        )
    context.verify_sources()
    verify_owned_artifacts(output_dir, summary)
    if (
        hashlib.sha256(session_candidate.read_bytes()).hexdigest()
        != summary["run_session_sha256"]
    ):
        raise ValueError("Staged session changed during manifest verification")
    context.raise_if_expired()
    _replace_manifest_directory(output_dir, Path(emission["manifest_dir"]))
    os.replace(session_candidate, run_session_path(output_dir))


def record_failed_evidence(output_dir: Path, summary: dict[str, Any]) -> None:
    """Persist an explicitly failed session and invalidate unaccepted manifests."""
    session_path = run_session_path(output_dir)
    try:
        session = load_session(session_path)
        if session.session_id != summary["run_id"]:
            raise ValueError("Historical session")
    except (OSError, ValueError):
        session = start_session(
            summary["run_id"], summary.get("planned_steps", []), created_by="gnn.main"
        )
        records = {record["script_name"]: record for record in summary.get("steps", [])}
        for unit in session.units:
            if unit.unit_id in records:
                unit.status = _unit_status(records[unit.unit_id])
    session.final_status = "FAILED"
    session.evidence_integrity = dict(summary["evidence_integrity"])
    checkpoint(session, session_path)
    summary["run_session_sha256"] = hashlib.sha256(
        session_path.read_bytes()
    ).hexdigest()
    staged = (
        output_dir
        / "00_pipeline_summary"
        / "finalization"
        / f"failed-{uuid.uuid4().hex}"
    )
    if not staged.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError("Failed evidence staging directory escapes its output root")
    staged.mkdir(parents=True)
    index = {
        "schema_version": "3.2",
        "verification_status": "FAILED",
        "provenance": {
            "run_id": summary["run_id"],
            "run_hash": summary.get("run_hash"),
            "overall_status": "FAILED",
        },
        "error": summary["error"],
        "trace_integrity_ok": False,
        "manifests": [],
        "stream_count": 0,
        "binary_artifacts": [],
        "binary_count": 0,
    }
    atomic_write_text(staged / "index.json", json.dumps(index, indent=2))
    # Failure invalidation must work even when a history symlink was the cause
    # of rejection. Preserve a safe old index beside this recovery candidate,
    # then replace only the canonical verdict anchor inside the owned root.
    destination = output_dir / "v3_run_manifest"
    if destination.is_symlink() or not destination.resolve().is_relative_to(
        output_dir.resolve()
    ):
        raise ValueError("Failed manifest directory escapes its output root")
    destination.mkdir(parents=True, exist_ok=True)
    old_index = destination / "index.json"
    if (
        old_index.is_file()
        and not old_index.is_symlink()
        and old_index.stat().st_nlink == 1
    ):
        atomic_write_bytes(staged / "previous_index.json", old_index.read_bytes())
    atomic_write_text(old_index, json.dumps(index, indent=2))
