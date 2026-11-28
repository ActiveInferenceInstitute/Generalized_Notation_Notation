#!/usr/bin/env python3
"""Resumable, session-wrapped model-family acceptance orchestration.

Wraps the existing manifest-driven model-family acceptance runner
(:mod:`pipeline.model_family_acceptance`) in a resumable :class:`RunSession`
(:mod:`pipeline.run_session`) so an extended multi-family acceptance run is
checkpointed after every family and can be resumed after a crash.

This module is pure orchestration over the existing acceptance runner. It
NEVER executes the pipeline, runs containers, or calls clusters: it delegates
to ``run_model_family_acceptance`` (which itself only stages on-disk fixtures
and invokes an injectable ``runner`` callable), records per-family ledgers and
artifact directories, and checkpoints a session manifest to disk after each
family so an interrupted run is resumable.

The session work units are the family names. Each family is processed in its
own single-family acceptance run; the unit is marked ``DONE`` on a passing
ledger and ``FAILED`` on an exception or a failed ledger. Checkpoint writes are
atomic (handled by :func:`pipeline.run_session.checkpoint`).

verify_session_artifacts joins a session checkpoint's recorded per-unit
artifact inventories against the on-disk artifacts (Perf#6): recorded files
must exist with matching digests and the directory inventory must be complete.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from gnn import __version__
from gnn.pipeline.hasher import runtime_config_identity
from gnn.pipeline.model_family_acceptance import (
    DEFAULT_EVIDENCE_STEPS,
    DEFAULT_FAMILY_TIMEOUT_SECONDS,
    ModelFamily,
    Runner,
    load_model_family_manifest,
    run_model_family_acceptance,
    select_model_families,
)
from gnn.pipeline.run_session import (
    RunSession,
    UnitStatus,
    WorkUnit,
    checkpoint,
    load_session,
    mark,
    start_session,
    status_report,
)


def _selected_family_names(
    manifest_path: Path, family_names: Optional[Iterable[str]]
) -> List[str]:
    """Return ordered family names to process, filtered by ``family_names``.

    Args:
        manifest_path: Path to the model-family manifest.
        family_names: Optional subset of family names to include; ``None`` or
            empty means every family declared in the manifest.

    Returns:
        Family names in manifest order, restricted to the requested subset.

    Raises:
        KeyError: If any requested family name is absent from the manifest.
    """
    families = load_model_family_manifest(manifest_path)
    return [family.name for family in select_model_families(families, family_names)]


def _file_digest(path: Path) -> str:
    """Hash artifact bytes without loading large execution outputs into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _family_identity(
    family: ModelFamily, manifest_path: Path, output_dir: Path, strict: bool
) -> Dict[str, Any]:
    """Bind a family to the actual source, policy, and destination of this call."""
    return {
        "schema": "acceptance-family-v3",
        "manifest_path": str(manifest_path.resolve()),
        "source_directory": str(family.target_dir.resolve()),
        "description": family.description,
        "representative_files": list(family.representative_files),
        "sources": {
            name: _file_digest(family.target_dir / name)
            for name in sorted(family.representative_files)
        },
        "frameworks": family.frameworks,
        "acceptance_profile": family.acceptance_profile,
        "strict": strict,
        "steps": DEFAULT_EVIDENCE_STEPS,
        "timeout_seconds": DEFAULT_FAMILY_TIMEOUT_SECONDS,
        "runtime_config": runtime_config_identity(),
        "gnn_version": __version__,
        "output_directory": str(output_dir.resolve()),
    }


def _artifact_hashes(directory: Path, exclude_path: Optional[Path]) -> Dict[str, str]:
    """Snapshot the complete output inventory, excluding ``exclude_path``.

    ``exclude_path`` (typically the session checkpoint itself) is compared by
    resolved path; ``None`` excludes nothing.
    """
    exclude = exclude_path.resolve() if exclude_path is not None else None
    root = directory.resolve()
    hashes: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(
                f"Acceptance artifact escapes its family directory: {path}"
            )
        if resolved != exclude:
            hashes[path.relative_to(directory).as_posix()] = _file_digest(path)
    return hashes


def _acceptance_session_hash(session: RunSession) -> str:
    identities = {unit.unit_id: unit.input_identity for unit in session.units}
    return hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()[
        :12
    ]


def _load_acceptance_session(path: Path) -> RunSession:
    session = load_session(path)
    if session.created_by != "session_acceptance" or session.schema_version != "1.0":
        raise ValueError(
            "Checkpoint is not a supported model-family acceptance session"
        )
    if len({unit.unit_id for unit in session.units}) != len(session.units):
        raise ValueError("Acceptance checkpoint has duplicate family units")
    if any(
        unit.input_identity for unit in session.units
    ) and session.run_hash != _acceptance_session_hash(session):
        raise ValueError(
            "Acceptance checkpoint does not match its recorded family identities"
        )
    return session


def _can_reuse_family(
    unit: WorkUnit, identity: dict[str, Any], output_dir: Path, session_path: Path
) -> bool:
    if (
        unit.status != UnitStatus.DONE
        or unit.input_identity != identity
        or unit.artifact_refs != [str(output_dir)]
        or not unit.artifact_hashes
    ):
        return False
    ledger_path = output_dir / "model_family_acceptance_ledger.json"
    try:
        if unit.artifact_hashes != _artifact_hashes(output_dir, session_path):
            return False
        ledger = json.loads(ledger_path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeError):
        return False
    if not isinstance(ledger, dict) or ledger.get("status") != "passed":
        return False
    families = ledger.get("families")
    return (
        isinstance(families, list)
        and len(families) == 1
        and isinstance(families[0], dict)
        and families[0].get("name") == unit.unit_id
        and families[0].get("status") == "passed"
    )


def run_session_acceptance(
    manifest_path: Path,
    output_dir: Path,
    session_path: Path,
    *,
    family_names: Optional[Iterable[str]] = None,
    strict: bool = False,
    runner: Optional[Runner] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """Run model-family acceptance under a resumable, checkpointed session.

    Loads the requested families, starts (or resumes) a :class:`RunSession`
    over the family names, and runs single-family acceptance for each family
    that is not yet ``DONE``. The session is checkpointed to ``session_path``
    after every family so a crash mid-run is resumable.

    Args:
        manifest_path: Path to the model-family manifest.
        output_dir: Directory for per-family acceptance ledger artifacts.
        session_path: Destination for the resumable session manifest JSON.
        family_names: Optional subset of family names to process; ``None`` means
            every family in the manifest.
        strict: Forwarded to ``run_model_family_acceptance`` (raises on a failed
            ledger). A failed family is recorded as ``FAILED`` regardless.
        runner: Optional injectable runner callable forwarded to the underlying
            acceptance runner (used to avoid executing the real pipeline).
        resume: When ``True`` and ``session_path`` exists, resume that session
            instead of starting a fresh one.

    Returns:
        A dict with keys ``session`` (the final ``RunSession`` dumped to a
        dict), ``status`` (the ``status_report``), and ``ledgers`` (a mapping of
        family name to its acceptance ledger dict, for families processed this
        call).
    """
    output_dir = Path(output_dir).resolve()
    session_path = Path(session_path)

    families = select_model_families(
        load_model_family_manifest(manifest_path), family_names
    )
    selected_names = [family.name for family in families]
    for name in selected_names:
        family_output = (output_dir / name).resolve()
        if family_output == output_dir or not family_output.is_relative_to(output_dir):
            raise ValueError(f"Acceptance family output escapes its root: {name}")
    identities = {
        family.name: _family_identity(
            family, manifest_path, output_dir / family.name, strict
        )
        for family in families
    }

    session: RunSession
    if resume and session_path.exists():
        session = _load_acceptance_session(session_path)
        # A resumed session must cover every requested family. If the caller
        # requests a family the persisted session never knew about, the loop
        # below would silently skip it and report done=True having run nothing —
        # fail loudly on that mismatch instead.
        session_units = {unit.unit_id for unit in session.units}
        unknown = sorted(name for name in selected_names if name not in session_units)
        if unknown:
            raise ValueError(
                "resume request includes families not in the persisted session "
                f"{sorted(session_units)}: {unknown}"
            )
    else:
        session = start_session(
            session_path.stem or "session_acceptance",
            selected_names,
            created_by="session_acceptance",
        )

    ledgers: Dict[str, Any] = {}

    for family in families:
        name = family.name
        family_output = output_dir / name
        unit = next(unit for unit in session.units if unit.unit_id == name)
        if _can_reuse_family(unit, identities[name], family_output, session_path):
            continue
        unit.input_identity = identities[name]
        unit.artifact_hashes = {}
        session.run_hash = _acceptance_session_hash(session)
        session = mark(
            session, name, UnitStatus.RUNNING, artifact_refs=[str(family_output)]
        )
        checkpoint(session, session_path)
        try:
            ledger = run_model_family_acceptance(
                manifest_path,
                family_output,
                family_names=[name],
                strict=strict,
                runner=runner,
            )
            current_families = select_model_families(
                load_model_family_manifest(manifest_path), [name]
            )
            if (
                _family_identity(
                    current_families[0], manifest_path, family_output, strict
                )
                != identities[name]
            ):
                raise ValueError(f"Source changed during acceptance: {name}")
            artifacts = _artifact_hashes(family_output, session_path)
        except Exception as exc:  # noqa: BLE001 — record failure, keep resumable
            session = mark(
                session,
                name,
                UnitStatus.FAILED,
                artifact_refs=[str(family_output)],
                error=f"{type(exc).__name__}: {exc}",
            )
            checkpoint(session, session_path)
            raise

        ledgers[name] = ledger
        if ledger.get("status") == "passed":
            session = mark(
                session,
                name,
                UnitStatus.DONE,
                artifact_refs=[str(family_output)],
            )
        else:
            failed = ledger.get("failed_families") or [name]
            session = mark(
                session,
                name,
                UnitStatus.FAILED,
                artifact_refs=[str(family_output)],
                error=f"acceptance ledger failed: {', '.join(map(str, failed))}",
            )
        next(
            unit for unit in session.units if unit.unit_id == name
        ).artifact_hashes = artifacts
        checkpoint(session, session_path)

    return {
        "session": session.model_dump(),
        "status": status_report(session),
        "ledgers": ledgers,
    }


def _resolve_artifact_dir(unit: WorkUnit, output_dir: Path) -> Optional[Path]:
    """Resolve a unit's artifact directory from its recorded refs.

    Refs recorded by :func:`run_session_acceptance` are absolute family
    output paths; relative refs are resolved under ``output_dir``. The first
    ref that resolves to an existing directory wins.

    Args:
        unit: The work unit whose ``artifact_refs`` to resolve.
        output_dir: Base directory for relative refs.

    Returns:
        The existing artifact directory, or ``None`` when no ref resolves.
    """
    for ref in unit.artifact_refs:
        candidate = Path(ref)
        if not candidate.is_absolute():
            candidate = output_dir / candidate
        if candidate.is_dir():
            return candidate
    return None


def verify_session_artifacts(
    session: Union[str, Path, RunSession],
    output_dir: Union[str, Path],
    *,
    session_file: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Join a session checkpoint's artifact inventory against live artifacts.

    Perf#6 durability join: for every unit that RECORDS an artifact inventory
    (``artifact_hashes``), re-hash the unit's artifact directory on disk and
    compare — recorded files must exist with matching sha256 digests, and
    every file under the directory must be recorded. Detects deletion,
    tampering, and post-checkpoint additions.

    Units without a recorded inventory (e.g. step-session units that
    reference output dirs without hashing them, or ``SKIPPED`` units) are
    outside the join: they record no claim for this function to verify.

    Args:
        session: A session checkpoint path (loads it) or a :class:`RunSession`.
        output_dir: Base directory under which relative ``artifact_refs``
            resolve; absolute refs are used as-is.
        session_file: Path of the session checkpoint to exclude from the
            on-disk inventory when ``session`` is passed as an instance.
            Defaults to ``session`` itself when it is a path.

    Returns:
        A list of human-readable problems. Empty means the join re-validates.
    """
    loaded: RunSession
    exclude_path: Optional[Path]
    if isinstance(session, (str, Path)):
        loaded = load_session(session)
        exclude_path = Path(session)
    else:
        loaded = session
        exclude_path = Path(session_file) if session_file is not None else None
    output_dir = Path(output_dir)

    problems: List[str] = []
    for unit in loaded.units:
        if unit.status == UnitStatus.SKIPPED or not unit.artifact_hashes:
            continue
        directory = _resolve_artifact_dir(unit, output_dir)
        if directory is None:
            problems.append(
                f"unit {unit.unit_id!r}: artifact directory not found "
                f"(refs={unit.artifact_refs})"
            )
            continue
        try:
            live = _artifact_hashes(directory, exclude_path)
        except (OSError, ValueError) as error:
            problems.append(
                f"unit {unit.unit_id!r}: invalid artifact inventory: {error}"
            )
            continue
        for rel in sorted(set(unit.artifact_hashes) - set(live)):
            problems.append(
                f"unit {unit.unit_id!r}: recorded artifact missing on disk: {rel}"
            )
        for rel in sorted(set(live) - set(unit.artifact_hashes)):
            problems.append(
                f"unit {unit.unit_id!r}: unrecorded artifact on disk: {rel}"
            )
        for rel in sorted(set(unit.artifact_hashes) & set(live)):
            if unit.artifact_hashes[rel] != live[rel]:
                problems.append(f"unit {unit.unit_id!r}: checksum mismatch for {rel}")
    return problems
