"""Admit an explicit pipeline resume and recover its verified progress."""

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from gnn.pipeline.artifact_ownership import verify_owned_artifacts
from gnn.pipeline.config import get_output_dir_for_script
from gnn.pipeline.run_context import RunContext
from gnn.pipeline.run_session import RunSession, UnitStatus, WorkUnit, load_session
from gnn.pipeline.run_session_wiring import run_session_path
from gnn.utils.pipeline_orchestration.pipeline_step_dependencies import (
    PIPELINE_OPTIONAL_PRODUCERS,
    PIPELINE_STEP_DEPENDENCIES,
)


@dataclass(frozen=True)
class ResumeCheckpoint:
    summary: dict[str, Any]
    session: RunSession

    @property
    def run_id(self) -> str:
        return self.session.session_id


def load_resume_checkpoint(output_dir: Path) -> ResumeCheckpoint:
    """Read the live summary and session while holding the output lease."""
    checkpoint_path = run_session_path(output_dir)
    summary_path = checkpoint_path.with_name("current_summary.json")
    try:
        session = load_session(checkpoint_path)
        summary = json.loads(summary_path.read_bytes())
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read pipeline resume checkpoint: {error}") from error
    if (
        not isinstance(summary, dict)
        or summary.get("artifact_inventory_contract") != "current-run-v1"
        or session.created_by != "gnn.main"
        or session.schema_version != "1.0"
        or summary.get("run_id") != session.session_id
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session.session_id)
    ):
        raise ValueError("Pipeline resume checkpoint has incompatible run identity")
    return ResumeCheckpoint(summary, session)


def resume_runtime_identity() -> str:
    """Fingerprint GNN code and the interpreter's installed distribution versions."""
    package_dir = Path(__file__).resolve().parent.parent
    sources = {}
    for path in sorted(package_dir.rglob("*.py")):
        with path.open("rb") as stream:
            sources[path.relative_to(package_dir).as_posix()] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    distributions = sorted(
        (distribution.metadata.get("Name", ""), distribution.version)
        for distribution in metadata.distributions()
    )
    identity = json.dumps(
        [sys.version, sources, distributions], sort_keys=True
    ).encode()
    return hashlib.sha256(identity).hexdigest()


def restore_pipeline_progress(
    checkpoint: ResumeCheckpoint,
    summary: dict[str, Any],
    context: RunContext,
    attempt_id: str,
) -> RunSession:
    """Restore only completed work whose source, runtime and dependency receipts agree."""
    previous = checkpoint.summary
    identity_keys = (
        "run_hash_schema",
        "identity_config",
        "file_hashes",
        "planned_steps",
        "source_inventory",
    )
    for key in identity_keys:
        before = json.dumps(previous.get(key), sort_keys=True, default=str)
        after = json.dumps(summary[key], sort_keys=True, default=str)
        if before != after:
            raise ValueError(f"Pipeline resume input or plan changed: {key}")

    planned = summary["planned_steps"]
    units = checkpoint.session.units
    if [unit.unit_id for unit in units] != planned:
        raise ValueError("Pipeline resume session does not match the current step plan")
    rows = previous.get("steps")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Pipeline resume checkpoint has invalid step records")
    records = {row.get("script_name"): row for row in rows}
    if len(records) != len(rows) or not set(records).issubset(planned):
        raise ValueError(
            "Pipeline resume checkpoint has duplicate or unknown step records"
        )

    reusable = set()
    if previous.get("resume_runtime_sha256") == summary["resume_runtime_sha256"]:
        for unit in units:
            record = records.get(unit.unit_id)
            if record is not None and _step_receipt_matches(unit, record, context):
                reusable.add(unit.unit_id)
    reusable = _dependency_closed(planned, reusable)

    from gnn.pipeline.summary_wiring import (
        _step_has_actionable_warning,
        _update_performance_summary,
    )

    for name in planned:
        if name in reusable:
            record = dict(records[name], reused_from_checkpoint=True)
            summary["steps"].append(record)
            _update_performance_summary(
                summary,
                record,
                name,
                _step_has_actionable_warning(record),
                len(planned),
            )
    summary["resume"] = {
        "attempt_id": attempt_id,
        "reused_steps": [name for name in planned if name in reusable],
        "scheduled_steps": [name for name in planned if name not in reusable],
    }
    return checkpoint.session.model_copy(
        update={
            "units": [
                unit
                if unit.unit_id in reusable
                else WorkUnit(unit_id=unit.unit_id, steps=unit.steps)
                for unit in units
            ],
            "final_status": None,
            "evidence_integrity": {},
        },
        deep=True,
    )


def _step_receipt_matches(
    unit: WorkUnit, record: dict[str, Any], context: RunContext
) -> bool:
    step_number = int(unit.unit_id.partition("_")[0])
    model_ids = [model.model_id for model in context.selected_models(step_number)]
    if (
        unit.status != UnitStatus.DONE
        or unit.steps != [step_number]
        or record.get("status") not in ("SUCCESS", "SUCCESS_WITH_WARNINGS")
        or record.get("run_id") != context.run_id
        or record.get("selected_model_ids") != model_ids
        or unit.input_identity != {"run_id": context.run_id, "model_ids": model_ids}
    ):
        return False
    try:
        artifacts = record["artifacts"]
        by_path = {artifact["path"]: artifact["sha256"] for artifact in artifacts}
        output_dir = Path(context.output_root)
        step_dir = get_output_dir_for_script(unit.unit_id, output_dir).relative_to(
            output_dir
        )
        if (
            len(by_path) != len(artifacts)
            or by_path != unit.artifact_hashes
            or any(not Path(path).is_relative_to(step_dir) for path in by_path)
        ):
            return False
        verify_owned_artifacts(output_dir, {"steps": [record]})
    except (OSError, ValueError, TypeError, KeyError):
        return False
    return True


def _dependency_closed(planned: list[str], reusable: set[str]) -> set[str]:
    by_number = {int(name.partition("_")[0]): name for name in planned}
    while True:
        waiting = {
            name
            for number, name in by_number.items()
            if name in reusable
            and any(
                by_number[dependency] not in reusable
                for dependency in (
                    *PIPELINE_STEP_DEPENDENCIES.get(number, ()),
                    *PIPELINE_OPTIONAL_PRODUCERS.get(number, ()),
                )
                if dependency in by_number
            )
        }
        if not waiting:
            return reusable
        reusable -= waiting
