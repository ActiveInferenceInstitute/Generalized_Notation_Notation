"""Project the current parser receipt into each artifact consumer's selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gnn.pipeline.artifact_ownership import owned_artifact_paths
from gnn.pipeline.run_context import current_run_context


def selected_parse_receipt(receipt: dict[str, Any], step: int) -> dict[str, Any]:
    """Require current-run coverage and preserve the consumer's staged identities."""
    context = current_run_context()
    if context is None:
        return receipt
    if receipt.get("run_id") != context.run_id:
        raise ValueError("Parser receipt belongs to a different run")
    records = {
        record.get("model_id"): record for record in receipt.get("processed_files", [])
    }
    selected = context.selected_models(step)
    missing = [model.model_id for model in selected if model.model_id not in records]
    if missing:
        raise ValueError(
            f"Step 3 has no current-run receipt for selected models: {missing}"
        )
    root = Path(context.output_root)
    owned = owned_artifact_paths(root) or set()
    view = context.input_view(step)
    projected = []
    for model in selected:
        record = dict(records[model.model_id])
        if record.get("parsed_model_file"):
            parsed = Path(record["parsed_model_file"]).resolve()
            if (
                not parsed.is_relative_to(root.resolve())
                or parsed.relative_to(root.resolve()).as_posix() not in owned
            ):
                raise ValueError(
                    "Parsed model artifact is outside the current-run inventory"
                )
        record["producer_file_path"] = record["file_path"]
        record["file_path"] = str(
            view / f"{model.artifact_stem}{Path(model.source_path).suffix}"
        )
        projected.append(record)
    return {**receipt, "processed_files": projected}
