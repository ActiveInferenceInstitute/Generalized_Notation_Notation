#!/usr/bin/env python3
"""Durable run manifest emission — turn a completed pipeline run into v3 artifacts.

This module is ADDITIVE and PURE: it reads existing on-disk run artifacts (the
``output/`` directory of a COMPLETED pipeline run) and emits durable, replayable
v3 artifacts. It NEVER executes the pipeline, runs containers, or contacts a
cluster. All file IO is local and atomic (delegated to ``durable_streams``).

For a finished run's output directory it produces:
  - one :class:`StreamManifest` per produced JSON artifact under the
    ``N_*_output/`` step directories (content-addressed, replayable);
  - one :class:`StreamManifest` per produced binary artifact (``.png``,
    ``.gif``, ``.npy``, ``.csv``) under those same step directories;
  - one :class:`ExecutionTrace` reconstructed from
    ``00_pipeline_summary/pipeline_execution_summary.json`` (one event per step
    record, in run order), or — if that summary is absent — from the sorted set
    of existing ``N_*_output/`` step directories;
  - an index JSON listing every emitted artifact.

Binary artifacts are recorded ADDITIVELY: the JSON inventory keys
(``manifests``/``stream_count``) keep their existing semantics and content,
and binary records live under the new ``binary_artifacts``/``binary_count``
index keys. Consumers must read the new keys additively and never repurpose
existing ones (see the B5-class consumer announce in the wave PR body).

Public surface:
  - emit_run_manifests(run_output_dir, *, manifest_out=None) -> dict
  - verify_run_manifests(manifest_dir, run_output_dir) -> list[str]
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from gnn.pipeline.durable_streams import (
    ExecutionTrace,
    StreamManifest,
    read_stream_manifest,
    read_trace,
    replay_trace,
    trace_integrity,
    validate_stream_manifest,
    write_stream_manifest,
    write_trace,
)

# Step output directories are named like ``3_gnn_output`` or ``11_render_output``.
_STEP_DIR_RE = re.compile(r"^(\d+)_.*_output$")

# Default subdirectory (relative to the run output dir) for emitted artifacts.
DEFAULT_MANIFEST_SUBDIR = "v3_run_manifest"

# Canonical location of the per-run execution summary within a run output dir.
_SUMMARY_REL = Path("00_pipeline_summary") / "pipeline_execution_summary.json"


def _stable_stream_id(rel_path: Path) -> str:
    """Derive a stable, filesystem-safe stream id from a relative artifact path.

    The id is a readable slug of the POSIX relative path PLUS a short hash of the
    exact path. The hash makes the id INJECTIVE: distinct artifacts whose slugs
    would otherwise collide (e.g. ``a.b.json`` and ``a-b.json`` both slugify to
    ``a_b_json``) get different ids, so no artifact's manifest can silently
    overwrite another's and leave its bytes unbound.

    Args:
        rel_path: Path of the artifact relative to the run output directory.

    Returns:
        A stable, collision-free stream identifier string.
    """
    posix = rel_path.as_posix()
    slug = re.sub(r"[^0-9A-Za-z]+", "_", posix).strip("_")
    digest = hashlib.sha256(posix.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}"


def _discover_artifacts(
    run_output_dir: Path, summary_path: Optional[Path] = None
) -> List[Path]:
    """Return sorted JSON artifact paths under the run's ``N_*_output/`` dirs.

    Only ``*.json`` files inside step output directories are collected. The
    ``00_pipeline_summary`` directory is a meta directory (it backs the trace,
    not an artifact stream) and is therefore excluded. The result is sorted by
    POSIX relative path for deterministic ordering.

    Args:
        run_output_dir: The completed run's output directory.

    Returns:
        A deterministically sorted list of absolute artifact paths.
    """
    owned = _current_artifact_inventory(run_output_dir, summary_path)
    artifacts: List[Path] = []
    for child in run_output_dir.iterdir():
        if not child.is_dir():
            continue
        if not _STEP_DIR_RE.match(child.name):
            continue
        for json_path in child.rglob("*.json"):
            if json_path.is_file():
                if (
                    owned is not None
                    and json_path.relative_to(run_output_dir).as_posix() not in owned
                ):
                    continue
                if not json_path.resolve().is_relative_to(run_output_dir.resolve()):
                    raise ValueError(
                        f"Artifact escapes run output directory: {json_path}"
                    )
                artifacts.append(json_path)
    artifacts.sort(key=lambda p: p.relative_to(run_output_dir).as_posix())
    return artifacts


#: Binary artifact suffixes recorded by :func:`emit_run_manifests` (Perf#5).
#: Extending this set is additive: new kinds appear only under the
#: ``binary_artifacts`` index key and never touch the JSON inventory.
_BINARY_EXTENSIONS: frozenset[str] = frozenset({".png", ".gif", ".npy", ".csv"})

#: Logical dtype label recorded in each binary artifact's StreamManifest.
#: ``uint8`` for raw-byte containers (png/gif/npy), ``text`` for csv.
_BINARY_DTYPE: Dict[str, str] = {
    ".png": "uint8",
    ".gif": "uint8",
    ".npy": "uint8",
    ".csv": "text",
}


def _discover_binary_artifacts(
    run_output_dir: Path, summary_path: Optional[Path] = None
) -> List[Path]:
    """Return sorted binary artifact paths under the run's ``N_*_output/`` dirs.

    Mirrors :func:`_discover_artifacts` exactly, but collects files whose
    extension is in :data:`_BINARY_EXTENSIONS` instead of ``*.json``. The
    result is sorted by POSIX relative path for deterministic ordering.

    Args:
        run_output_dir: The completed run's output directory.

    Returns:
        A deterministically sorted list of absolute binary artifact paths.
    """
    owned = _current_artifact_inventory(run_output_dir, summary_path)
    artifacts: List[Path] = []
    for child in run_output_dir.iterdir():
        if not child.is_dir():
            continue
        if not _STEP_DIR_RE.match(child.name):
            continue
        for path in child.rglob("*"):
            if path.is_file() and path.suffix.lower() in _BINARY_EXTENSIONS:
                if (
                    owned is not None
                    and path.relative_to(run_output_dir).as_posix() not in owned
                ):
                    continue
                if not path.resolve().is_relative_to(run_output_dir.resolve()):
                    raise ValueError(f"Artifact escapes run output directory: {path}")
                artifacts.append(path)
    artifacts.sort(key=lambda p: p.relative_to(run_output_dir).as_posix())
    return artifacts


def _summary_path(run_dir: Path, supplied: Optional[Path]) -> Path:
    """Resolve a staged or canonical summary within the owned output root."""
    path = supplied if supplied is not None else run_dir / _SUMMARY_REL
    if not path.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("Summary path escapes run output directory")
    return path


def _current_artifact_inventory(
    run_dir: Path, summary_path: Optional[Path] = None
) -> Optional[set[str]]:
    """Project new-contract receipts exactly; standalone manifests keep discovery."""
    path = _summary_path(run_dir, summary_path)
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("artifact_inventory_contract") != "current-run-v1":
        return None
    from gnn.pipeline.artifact_ownership import verify_owned_artifacts

    verify_owned_artifacts(run_dir, summary)
    return {
        record["path"]
        for step in _step_records(summary)
        for record in step.get("artifacts", [])
    }


def _step_records(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the ordered list of per-step records from a summary dict.

    Args:
        summary: The parsed ``pipeline_execution_summary.json`` content.

    Returns:
        The list of step record dicts (possibly empty).
    """
    if not isinstance(summary, dict):
        raise ValueError("Execution summary must be an object")
    steps = summary.get("steps", [])
    if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
        raise ValueError("Execution summary steps must be a list of objects")
    return steps


def _step_label(record: Dict[str, Any], fallback_index: int) -> str:
    """Build a human-meaningful step label from a step record.

    Prefers ``script_name``; falls back through ``description`` /
    ``step_number`` / a positional index so the label is always non-empty.

    Args:
        record: A single step record dict.
        fallback_index: Positional index used if the record carries no name.

    Returns:
        A non-empty step label.
    """
    for key in ("script_name", "step_name", "description"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    number = record.get("step_number")
    if number is not None:
        return f"step_{number}"
    return f"step_{fallback_index}"


def _build_trace_from_summary(trace_id: str, summary: Dict[str, Any]) -> ExecutionTrace:
    """Reconstruct an execution trace from a parsed run summary.

    One event is appended per step record, in run order. The event ``step`` is
    the step label, ``action`` is the recorded status, and the payload bytes are
    the canonical JSON encoding of that step record (so the checksum binds the
    full step summary).

    Args:
        trace_id: Identifier for the resulting trace.
        summary: The parsed execution summary.

    Returns:
        A populated :class:`ExecutionTrace`.
    """
    trace = ExecutionTrace(trace_id=trace_id, created_by="run_manifest")
    for index, record in enumerate(_step_records(summary)):
        label = _step_label(record, index)
        status = record.get("status")
        action = status if isinstance(status, str) and status else "UNKNOWN"
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")
        trace = trace.append_event(
            step=label,
            action=action,
            payload_bytes=payload,
            payload_ref=label,
        )
    return trace


def _build_trace_from_dirs(trace_id: str, run_output_dir: Path) -> ExecutionTrace:
    """Reconstruct a trace from the sorted set of existing step dirs.

    Used when no execution summary is present. Step directories are sorted by
    their numeric prefix, and one event is appended per directory. The payload
    bytes are the directory's relative name, so the trace digest is stable for a
    given set of step directories.

    Args:
        trace_id: Identifier for the resulting trace.
        run_output_dir: The completed run's output directory.

    Returns:
        A populated :class:`ExecutionTrace`.
    """
    step_dirs: List[Path] = []
    for child in run_output_dir.iterdir():
        if child.is_dir() and _STEP_DIR_RE.match(child.name):
            step_dirs.append(child)

    def _numeric_prefix(path: Path) -> tuple[int, str]:
        match = _STEP_DIR_RE.match(path.name)
        number = int(match.group(1)) if match else 0
        return (number, path.name)

    step_dirs.sort(key=_numeric_prefix)

    trace = ExecutionTrace(trace_id=trace_id, created_by="run_manifest")
    for child in step_dirs:
        name = child.name
        trace = trace.append_event(
            step=name,
            action="present",
            payload_bytes=name.encode("utf-8"),
            payload_ref=name,
        )
    return trace


def _run_provenance(
    run_dir: Path, summary_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Bind complete summary metadata, or explicitly declare directory-only evidence."""
    summary_path = _summary_path(run_dir, summary_path)
    if not summary_path.exists():
        return {"mode": "directories"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _step_records(summary)
    session_hash = summary.get("run_session_sha256")
    if session_hash is not None:
        session_path = summary_path.parent / "run_session.json"
        if not session_path.resolve().is_relative_to(run_dir.resolve()):
            raise ValueError("Run session escapes its output root")
        if hashlib.sha256(session_path.read_bytes()).hexdigest() != session_hash:
            raise ValueError("Run session checksum differs from the terminal summary")
        session = json.loads(session_path.read_text(encoding="utf-8"))
        if session.get("session_id") != summary.get("run_id") or session.get(
            "final_status"
        ) != summary.get("overall_status"):
            raise ValueError(
                "Run session terminal identity/status differs from summary"
            )
    canonical = json.dumps(summary, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {
        "mode": "summary",
        "summary_sha256": hashlib.sha256(canonical).hexdigest(),
        "run_id": summary.get("run_id"),
        "run_hash": summary.get("run_hash"),
        "overall_status": summary.get("overall_status"),
    }


def _contained_file(base: Path, relative: Any) -> Path:
    """Resolve an artifact reference without admitting absolute paths or escapes."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"Invalid relative artifact reference: {relative!r}")
    path = base / relative
    if ".." in Path(relative).parts or not path.resolve().is_relative_to(
        base.resolve()
    ):
        raise ValueError(f"Artifact reference escapes its base: {relative!r}")
    return path


def _write_file_stream_manifest(
    artifact: Path,
    run_dir: Path,
    manifest_dir: Path,
    dtype: str,
    written_filenames: set[str],
) -> Dict[str, str]:
    """Checksum one artifact into a StreamManifest and write its manifest file.

    Shared by the JSON and binary inventory sections: both record FILE-kind
    manifests over raw bytes. Returns the index entry for the artifact.

    Args:
        artifact: Absolute artifact path under ``run_dir``.
        run_dir: The run output directory the artifact lives in.
        manifest_dir: Destination directory for the emitted manifest.
        dtype: Logical dtype label recorded in the manifest.
        written_filenames: Shared set of already-written manifest filenames
            (defense in depth against non-injective stream ids).

    Returns:
        The index entry dict with ``stream_id`` / ``source`` /
        ``manifest_file`` keys.

    Raises:
        ValueError: If the derived manifest filename would collide with one
            already written (stream id is not injective).
    """
    rel = artifact.relative_to(run_dir)
    stream_id = _stable_stream_id(rel)
    manifest = StreamManifest.from_file(
        stream_id=stream_id,
        path=artifact,
        source=rel.as_posix(),
        dtype=dtype,
        created_by="run_manifest",
    )
    manifest_filename = f"{stream_id}.manifest.json"
    # Defense in depth: a non-injective id would let one artifact's manifest
    # silently overwrite another's, leaving its bytes unbound. Refuse to.
    if manifest_filename in written_filenames:
        raise ValueError(
            f"manifest filename collision for {rel.as_posix()!r}: {manifest_filename} "
            "already written (stream_id is not injective)"
        )
    written_filenames.add(manifest_filename)
    write_stream_manifest(manifest, manifest_dir / manifest_filename)
    return {
        "stream_id": stream_id,
        "source": rel.as_posix(),
        "manifest_file": manifest_filename,
    }


def emit_run_manifests(
    run_output_dir: Union[str, Path],
    *,
    manifest_out: Optional[Union[str, Path]] = None,
    summary_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Emit durable v3 manifests + a trace for a COMPLETED pipeline run.

    Walks ``run_output_dir`` for produced JSON artifacts and binary artifacts
    (``.png``/``.gif``/``.npy``/``.csv``, under the ``N_*_output/`` step
    dirs), builds a content-addressed :class:`StreamManifest` for each,
    reconstructs an :class:`ExecutionTrace` from the run summary (or, if
    absent, from the existing step directories), and writes everything plus
    an index JSON into ``manifest_out``.

    This function reads on-disk data only. It does not execute anything.

    The index schema is additive: JSON records keep the ``manifests`` /
    ``stream_count`` keys with their existing meaning; binary records are
    written under the new ``binary_artifacts`` / ``binary_count`` keys
    (``schema_version`` ``"3.2"``). Consumers must read the new keys
    additively and never repurpose existing ones.

    Args:
        run_output_dir: The output directory of a completed run.
        manifest_out: Destination directory for the emitted artifacts; defaults
            to ``run_output_dir/v3_run_manifest``.
        summary_path: Optional staged summary under the owned run output root.

    Returns:
        A summary dict::

            {
                "stream_count": int,
                "binary_count": int,
                "trace_event_count": int,
                "trace_integrity_ok": bool,
                "manifest_dir": str,
            }

    Raises:
        FileNotFoundError: If ``run_output_dir`` does not exist.
        NotADirectoryError: If ``run_output_dir`` is not a directory.
    """
    run_dir = Path(run_output_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run output dir does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"run output dir is not a directory: {run_dir}")

    manifest_dir = (
        Path(manifest_out)
        if manifest_out is not None
        else run_dir / DEFAULT_MANIFEST_SUBDIR
    )
    if manifest_dir.resolve().is_relative_to(run_dir.resolve()):
        relative = manifest_dir.resolve().relative_to(run_dir.resolve())
        if relative.parts and _STEP_DIR_RE.match(relative.parts[0]):
            raise ValueError(
                "Manifest output cannot be inside a step artifact directory"
            )
    provenance = _run_provenance(run_dir, summary_path)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build a StreamManifest per produced JSON artifact (deterministic order).
    artifacts = _discover_artifacts(run_dir, summary_path)
    manifest_entries: List[Dict[str, str]] = []
    written_filenames: set[str] = set()
    for artifact in artifacts:
        manifest_entries.append(
            _write_file_stream_manifest(
                artifact, run_dir, manifest_dir, "uint8", written_filenames
            )
        )

    # 1b. Build a StreamManifest per produced binary artifact — additive
    # (Perf#5): binary records live under the new index keys and never
    # touch the JSON inventory.
    binary_entries: List[Dict[str, str]] = []
    for artifact in _discover_binary_artifacts(run_dir, summary_path):
        entry = _write_file_stream_manifest(
            artifact,
            run_dir,
            manifest_dir,
            _BINARY_DTYPE.get(artifact.suffix.lower(), "uint8"),
            written_filenames,
        )
        entry["format"] = artifact.suffix.lower().lstrip(".")
        binary_entries.append(entry)

    # 2. Reconstruct the execution trace from the summary, else from step dirs.
    trace_id = f"run::{run_dir.name}"
    summary_path = _summary_path(run_dir, summary_path)
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        trace = _build_trace_from_summary(trace_id, summary)
    else:
        trace = _build_trace_from_dirs(trace_id, run_dir)

    trace_filename = "execution_trace.json"
    write_trace(trace, manifest_dir / trace_filename)
    integrity_problems = trace_integrity(trace)
    trace_integrity_ok = not integrity_problems

    # 3. Write an index JSON listing every emitted artifact.
    index = {
        "run_output_dir": str(run_dir),
        "schema_version": "3.2",
        "provenance": provenance,
        "trace_file": trace_filename,
        "trace_event_count": len(trace.events),
        "trace_integrity_ok": trace_integrity_ok,
        "stream_count": len(manifest_entries),
        "manifests": manifest_entries,
        "binary_count": len(binary_entries),
        "binary_artifacts": binary_entries,
    }
    index_text = json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False)
    _write_index(manifest_dir / "index.json", index_text)

    return {
        "stream_count": len(manifest_entries),
        "trace_event_count": len(trace.events),
        "binary_count": len(binary_entries),
        "trace_integrity_ok": trace_integrity_ok,
        "manifest_dir": str(manifest_dir),
    }


def _write_index(path: Path, text: str) -> None:
    """Atomically write the index JSON.

    Delegates to the shared :mod:`pipeline._io` atomic-write helper (the same
    recipe used by stream manifests, traces, and session checkpoints) instead
    of re-implementing mkstemp + ``os.replace`` here.
    """
    from gnn.pipeline._io import atomic_write_text

    atomic_write_text(path, text)


def _verify_index_entry(
    entry: Any,
    mdir: Path,
    run_dir: Path,
    sources: set[str],
    files: set[str],
) -> List[str]:
    """Validate one index inventory entry against its manifest file.

    Shared by the JSON and binary inventory loops. Raises on
    identity/containment defects (the caller maps those to problem strings)
    and returns the per-manifest validation problems (e.g. checksum
    mismatches) already prefixed with the stream id.

    Args:
        entry: The raw index entry (must be an object with ``source``,
            ``manifest_file``, and ``stream_id``).
        mdir: The manifest directory the entry's manifest file lives in.
        run_dir: The run output directory the source artifact lives in.
        sources: Running set of seen sources (duplicates are rejected).
        files: Running set of seen manifest filenames (duplicates rejected).

    Returns:
        The manifest's validation problems (empty when it re-validates).

    Raises:
        ValueError: On any identity, containment, or duplicate defect.
        OSError: If the manifest file cannot be read.
    """
    if not isinstance(entry, dict):
        raise ValueError("Manifest index entry must be an object")
    source = entry.get("source")
    if not isinstance(source, str):
        raise ValueError("Manifest source must be a relative path string")
    source_path = _contained_file(run_dir, source)
    filename = entry.get("manifest_file")
    if not isinstance(filename, str):
        raise ValueError("Manifest file must be a relative path string")
    manifest_path = _contained_file(mdir, filename)
    if source in sources or filename in files:
        raise ValueError("Duplicate source or manifest file in index")
    sources.add(source)
    files.add(filename)
    manifest = read_stream_manifest(manifest_path)
    expected_id = _stable_stream_id(source_path.relative_to(run_dir))
    if (
        entry.get("stream_id") != expected_id
        or manifest.stream_id != expected_id
        or manifest.source != source
    ):
        raise ValueError("Manifest source/stream identity differs from index")
    if manifest.kind.value != "FILE":
        raise ValueError("Run artifacts require file-backed manifests")
    return [
        f"{manifest.stream_id}: {problem}"
        for problem in validate_stream_manifest(manifest, run_dir)
    ]


def verify_run_manifests(
    manifest_dir: Union[str, Path],
    run_output_dir: Union[str, Path],
    *,
    summary_path: Optional[Path] = None,
) -> List[str]:
    """Re-validate every emitted manifest and the trace.

    Each emitted :class:`StreamManifest` is re-checked against its backing file
    under ``run_output_dir`` (so a tampered artifact surfaces as a checksum
    mismatch). The trace's structural integrity is re-verified AND its meaning is
    re-bound to ground truth: the trace is re-derived from the live execution
    summary (or step dirs) and its replay digest is compared to the stored trace,
    so tampering the summary (e.g. flipping a step status) or relabeling a stored
    event surfaces as a digest mismatch — not just structurally-clean but stale.

    Binary records (schema 3.2) are re-validated the same way under the
    additive ``binary_artifacts`` inventory. An older ``"3.1"`` index that
    predates binary records is still accepted, but binary artifacts present
    on disk are then reported as unrecorded rather than silently ignored.

    Args:
        manifest_dir: Directory previously written by :func:`emit_run_manifests`.
        run_output_dir: The run output directory the manifests describe.
        summary_path: Optional staged summary under the owned run output root.

    Returns:
        A list of human-readable problems. Empty means everything re-validates.
    """
    mdir = Path(manifest_dir)
    run_dir = Path(run_output_dir)
    problems: List[str] = []
    try:
        index = json.loads((mdir / "index.json").read_text(encoding="utf-8"))
        if not isinstance(index, dict) or not isinstance(index.get("manifests"), list):
            return ["index.json must contain a manifests list"]
        if index.get("verification_status") == "FAILED":
            return [
                f"Run evidence explicitly failed finalization: {index.get('error')}"
            ]
        if index.get("schema_version") not in ("3.1", "3.2"):
            problems.append(
                "Index lacks supported complete inventory/provenance schema 3.1/3.2"
            )
        if index.get("provenance") != _run_provenance(run_dir, summary_path):
            problems.append("Run provenance differs from the indexed summary")
        actual_sources = {
            path.relative_to(run_dir).as_posix()
            for path in _discover_artifacts(run_dir, summary_path)
        }
        actual_binary_sources = {
            path.relative_to(run_dir).as_posix()
            for path in _discover_binary_artifacts(run_dir, summary_path)
        }
    except (OSError, ValueError, TypeError) as exc:
        return [f"Cannot read run manifest inventory/provenance: {exc}"]

    sources: set[str] = set()
    files: set[str] = set()
    entries = index["manifests"]
    if type(index.get("stream_count")) is not int or index["stream_count"] != len(
        entries
    ):
        problems.append("stream_count does not match manifest inventory")
    for entry in entries:
        try:
            problems.extend(_verify_index_entry(entry, mdir, run_dir, sources, files))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            problems.append(f"Invalid manifest entry: {exc}")
    if sources != actual_sources:
        problems.append(
            f"Artifact inventory differs: missing={sorted(actual_sources - sources)}, "
            f"unexpected={sorted(sources - actual_sources)}"
        )

    binary_entries = index.get("binary_artifacts")
    if binary_entries is None and actual_binary_sources:
        problems.append(
            "Binary artifacts not recorded in the index: "
            f"missing={sorted(actual_binary_sources)}"
        )
    if isinstance(binary_entries, list):
        binary_sources: set[str] = set()
        binary_files: set[str] = set()
        binary_count = index.get("binary_count")
        if type(binary_count) is not int or binary_count != len(binary_entries):
            problems.append("binary_count does not match binary manifest inventory")
        for entry in binary_entries:
            try:
                problems.extend(
                    _verify_index_entry(
                        entry, mdir, run_dir, binary_sources, binary_files
                    )
                )
            except (OSError, ValueError, TypeError, KeyError) as exc:
                problems.append(f"Invalid binary manifest entry: {exc}")
        if binary_sources != actual_binary_sources:
            problems.append(
                "Binary artifact inventory differs: "
                f"missing={sorted(actual_binary_sources - binary_sources)}, "
                f"unexpected={sorted(binary_sources - actual_binary_sources)}"
            )
    elif binary_entries is not None:
        problems.append("index.json binary_artifacts must be a list when present")

    try:
        trace_path = _contained_file(mdir, index.get("trace_file"))
        trace = read_trace(trace_path)
        problems.extend(f"trace: {problem}" for problem in trace_integrity(trace))
        if index.get("trace_event_count") != len(trace.events):
            problems.append("trace_event_count does not match stored trace")
        trace_id = f"run::{run_dir.name}"
        summary_path = _summary_path(run_dir, summary_path)
        expected = (
            _build_trace_from_summary(
                trace_id, json.loads(summary_path.read_text(encoding="utf-8"))
            )
            if summary_path.is_file()
            else _build_trace_from_dirs(trace_id, run_dir)
        )
        if replay_trace(expected) != replay_trace(trace):
            problems.append(
                "trace: replay digest does not match the trace re-derived from the "
                "live run summary (summary tampered or stored trace relabeled)"
            )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        problems.append(f"Cannot verify trace: {exc}")
    return problems
