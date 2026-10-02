"""Copy current visualization bundles without breaking relative dependencies."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from gnn.pipeline.run_context import current_run_context


def selected_website_asset(path: Path) -> bool:
    """Exclude artifacts owned by models outside this website's selection."""
    context = current_run_context()
    if context is None:
        return True
    selected = {model.model_id for model in context.selected_models(20)}
    owners = [
        model
        for model in context.models
        if model.artifact_stem in path.parts
        or path.stem == model.artifact_stem
        or path.stem.startswith(model.artifact_stem + "_")
    ]
    return not owners or all(model.model_id in selected for model in owners)


def copy_interactive_bundle(
    artifact: Path,
    source_root: Path,
    assets_dir: Path,
    output_root: Path,
    inventory: set[str] | None,
) -> str:
    """Preserve an HTML document's relative CSS/JS/data/image layout.

    A separate directory per producer prevents colliding sidecars from silently
    overwriting one another. Current runs copy only owned, selected artifacts;
    standalone collection retains its complete visualization subtree.
    """
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    artifact = artifact.resolve()
    relative = artifact.relative_to(source_root)
    bundle_id = hashlib.sha256(str(source_root).encode()).hexdigest()[:16]
    destination_root = assets_dir / "interactive" / bundle_id
    context = current_run_context()
    for source in sorted(source_root.rglob("*")):
        if not source.is_file():
            continue
        resolved = source.resolve()
        if not resolved.is_relative_to(source_root):
            raise ValueError(
                "Visualization bundle source escapes its producer directory"
            )
        if (
            inventory is not None
            and resolved.relative_to(output_root).as_posix() not in inventory
        ):
            continue
        if not selected_website_asset(source):
            continue
        if context is not None:
            context.raise_if_expired()
        destination = destination_root / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    target = destination_root / relative
    if not target.is_file():
        raise ValueError("Interactive artifact is absent from the selected bundle")
    return target.relative_to(assets_dir).as_posix()
