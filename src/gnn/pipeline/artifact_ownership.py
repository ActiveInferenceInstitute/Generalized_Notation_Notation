"""Current-invocation artifact ownership independent of stale output contents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """Hash bytes without allocating a whole artifact in parent memory."""
    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            if context is not None:
                context.raise_if_expired()
            digest.update(block)
    return digest.hexdigest()


def snapshot_files(directory: Path) -> dict[str, tuple[int, int, str]]:
    """Record file state without claiming files from a previous invocation."""
    if not directory.exists():
        return {}
    for path in directory.rglob("*"):
        if path.is_file() and not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Artifact escapes step output directory: {path}")
    return {
        p.relative_to(directory).as_posix(): (
            p.stat().st_mtime_ns,
            p.stat().st_size,
            file_sha256(p),
        )
        for p in directory.rglob("*")
        if p.is_file()
        and not any(
            part.startswith(".") or part == "history"
            for part in p.relative_to(directory).parts
        )
    }


def changed_artifacts(
    directory: Path, output_root: Path, before: dict[str, tuple[int, int, str]]
) -> list[dict[str, Any]]:
    """Bind files authored during this step to their exact output bytes."""
    after = snapshot_files(directory)
    return [
        {
            "path": (directory / relative).relative_to(output_root).as_posix(),
            "sha256": state[2],
        }
        for relative, state in sorted(after.items())
        if before.get(relative) != state
    ]


def verify_owned_artifacts(output_root: Path, summary: dict[str, Any]) -> None:
    """Verify every byte-bound current-run output before accepting completion."""
    for step in summary.get("steps", []):
        for record in step.get("artifacts", []):
            relative = Path(record["path"])
            path = output_root / relative
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not path.resolve().is_relative_to(output_root.resolve())
            ):
                raise ValueError("Artifact inventory escapes output root")
            if file_sha256(path) != record["sha256"]:
                raise ValueError(f"Current-run artifact changed: {relative}")


def owned_artifact_checksums(output_root: Path) -> dict[str, str | None] | None:
    """Read current-run artifact digests, or None for standalone output."""
    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    if context is None:
        return None
    path = output_root / "00_pipeline_summary" / "current_summary.json"
    if not path.is_file():
        return {}
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("run_id") != context.run_id:
        raise ValueError("Artifact inventory belongs to a different run")
    return {
        record["path"]: record.get("sha256")
        for step in summary.get("steps", [])
        for record in step.get("artifacts", [])
        if isinstance(record, dict)
    }


def owned_artifact_paths(output_root: Path) -> set[str] | None:
    """Return an explicit current-run inventory, None for standalone output."""
    checksums = owned_artifact_checksums(output_root)
    return set(checksums) if checksums is not None else None


def current_artifact_files(directory: Path, pattern: str = "*") -> list[Path]:
    """List only current-invocation files, retaining standalone discovery."""
    import fnmatch

    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    if context is None:
        return sorted(path for path in directory.rglob(pattern) if path.is_file())
    root = Path(context.output_root)
    prefix = directory.resolve().relative_to(root.resolve()).as_posix() + "/"
    return sorted(
        root / relative
        for relative in owned_artifact_paths(root) or set()
        if relative.startswith(prefix) and fnmatch.fnmatch(Path(relative).name, pattern)
    )


def execution_view(execution_dir: Path, model_stems: set[str]) -> Path:
    """Project only current-run authored execution files for selected models."""
    import shutil

    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    if context is None:
        return execution_dir
    root = Path(context.output_root)
    inventory = owned_artifact_paths(root) or set()
    fingerprint = hashlib.sha256(json.dumps(sorted(model_stems)).encode()).hexdigest()[
        :24
    ]
    view = (
        root
        / "00_pipeline_summary"
        / "execution_selection"
        / context.run_id
        / fingerprint
    )
    view.mkdir(parents=True, exist_ok=True)
    prefix = execution_dir.resolve().relative_to(root.resolve()).as_posix() + "/"
    for relative in sorted(inventory):
        context.raise_if_expired()
        if not relative.startswith(prefix):
            continue
        inside = Path(relative[len(prefix) :])
        if inside.parts[0] not in model_stems and inside.parts[0] != "summaries":
            continue
        destination = view / inside
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = root / relative
        if inside.parts[0] == "summaries" and source.suffix == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(
                payload.get("execution_details"), list
            ):
                # Pointer-following analyzers must see the same selection as
                # directory walkers. The source receipt remains immutable.
                details = [
                    entry
                    for entry in payload["execution_details"]
                    if isinstance(entry, dict)
                    and entry.get("model_name") in model_stems
                ]
                for entry in details:
                    for key in (
                        "implementation_directory",
                        "structured_result_file",
                        "output_file",
                        "native_result_file",
                    ):
                        value = entry.get(key)
                        if not isinstance(value, str):
                            continue
                        candidate = Path(value)
                        if not candidate.is_absolute():
                            candidate = Path.cwd() / candidate
                        try:
                            projected_path = candidate.resolve().relative_to(
                                execution_dir.resolve()
                            )
                        except (OSError, ValueError):
                            continue
                        if (
                            projected_path.parts
                            and projected_path.parts[0] in model_stems
                        ):
                            entry[key] = str((view / projected_path).resolve())
                payload["execution_details"] = details
                for key in ("total_scripts_found", "total_scripts"):
                    if key in payload:
                        payload[key] = len(details)
                for key, predicate in (
                    (
                        "successful_executions",
                        lambda entry: entry.get("success") and not entry.get("skipped"),
                    ),
                    (
                        "failed_executions",
                        lambda entry: (
                            not entry.get("success") and not entry.get("skipped")
                        ),
                    ),
                    ("skipped_executions", lambda entry: entry.get("skipped")),
                ):
                    if key in payload:
                        payload[key] = sum(bool(predicate(entry)) for entry in details)
                successes = sum(
                    bool(entry.get("success")) and not entry.get("skipped", False)
                    for entry in details
                )
                skips = sum(bool(entry.get("skipped")) for entry in details)
                failures = len(details) - successes - skips
                attempted = len(details) - skips
                payload.update(
                    attempted_scripts=attempted,
                    success_rate=round(100 * successes / attempted, 2)
                    if attempted
                    else 0.0,
                    success=bool(attempted) and not failures,
                    status="failed"
                    if failures
                    else "success"
                    if attempted
                    else "skipped",
                    exit_code=1 if failures else 0 if attempted else 2,
                )
                payload["framework_status"] = {}
                for framework in sorted(
                    {entry.get("framework", "unknown") for entry in details}
                ):
                    rows = [
                        entry
                        for entry in details
                        if entry.get("framework", "unknown") == framework
                    ]
                    failed = sum(
                        not entry.get("success") and not entry.get("skipped")
                        for entry in rows
                    )
                    payload["framework_status"][framework] = {
                        "executions": len(rows),
                        "successful": sum(
                            bool(entry.get("success"))
                            and not entry.get("skipped", False)
                            for entry in rows
                        ),
                        "failed": failed,
                        "skipped": sum(bool(entry.get("skipped")) for entry in rows),
                        "status": "failed"
                        if failed
                        else "skipped"
                        if all(row.get("skipped") for row in rows)
                        else "success",
                    }
                selected_ids = {
                    model.model_id
                    for model in context.models
                    if model.artifact_stem in model_stems
                }
                if isinstance(payload.get("model_selection"), list):
                    payload["model_selection"] = [
                        model
                        for model in payload["model_selection"]
                        if isinstance(model, dict)
                        and (
                            model.get("artifact_stem") in model_stems
                            or model.get("model_id") in selected_ids
                        )
                    ]
                payload["_selection_view"] = {
                    "model_stems": sorted(model_stems),
                    "source_receipt_sha256": file_sha256(source),
                    "claim": "derived analysis selection; original execution receipt is unchanged",
                }
                destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                continue
        shutil.copy2(source, destination)
    return view
