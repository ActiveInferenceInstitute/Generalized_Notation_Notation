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


def owned_artifact_paths(output_root: Path) -> set[str] | None:
    """Return an explicit current-run inventory, None for standalone output."""
    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    if context is None:
        return None
    path = output_root / "00_pipeline_summary" / "current_summary.json"
    if not path.is_file():
        return set()
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("run_id") != context.run_id:
        raise ValueError("Artifact inventory belongs to a different run")
    return {
        record["path"]
        for step in summary.get("steps", [])
        for record in step.get("artifacts", [])
        if isinstance(record, dict)
    }


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
        shutil.copy2(root / relative, destination)
    return view
