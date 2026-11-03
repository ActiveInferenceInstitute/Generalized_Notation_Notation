"""Copy current visualization bundles without breaking relative dependencies."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _MaterializedBundle:
    """Selection state under which a producer's bundle was materialized."""

    inventory: set[str] | None
    selection: tuple[tuple[str, bool], ...] | None


_MATERIALIZED_BUNDLES: dict[tuple[Path, Path, Path], _MaterializedBundle] = {}


def _selection_fingerprint() -> tuple[tuple[str, bool], ...] | None:
    """Return the model ownership/selection mapping that filters bundle files."""
    context = current_run_context()
    if context is None:
        return None
    selected = {model.model_id for model in context.selected_models(20)}
    return tuple(
        sorted(
            (model.artifact_stem, model.model_id in selected)
            for model in context.models
        )
    )


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
    standalone collection retains its complete visualization subtree. One
    producer's subtree is copied once while its inventory and selected-model
    filters are unchanged, so a producer with many interactive documents does
    not rewrite the whole subtree for each of them.
    """
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    artifact = artifact.resolve()
    relative = artifact.relative_to(source_root)
    bundle_id = hashlib.sha256(str(source_root).encode()).hexdigest()[:16]
    destination_root = assets_dir / "interactive" / bundle_id
    target = destination_root / relative
    context = current_run_context()
    selection = _selection_fingerprint()
    materialized = _MATERIALIZED_BUNDLES.get(
        (source_root, output_root, destination_root)
    )
    if (
        materialized is not None
        and materialized.inventory is inventory
        and materialized.selection == selection
        and target.is_file()
    ):
        if context is not None:
            context.raise_if_expired()
        return target.relative_to(assets_dir).as_posix()
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
    if not target.is_file():
        raise ValueError("Interactive artifact is absent from the selected bundle")
    _MATERIALIZED_BUNDLES[(source_root, output_root, destination_root)] = (
        _MaterializedBundle(inventory=inventory, selection=selection)
    )
    return target.relative_to(assets_dir).as_posix()
