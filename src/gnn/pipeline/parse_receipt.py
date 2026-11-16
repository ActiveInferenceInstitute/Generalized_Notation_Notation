"""Version parsed-model artifacts and project the current parser receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypedDict

from gnn.pipeline.artifact_ownership import owned_artifact_checksums
from gnn.pipeline.run_context import current_run_context

PARSED_MODEL_SCHEMA = "gnn_parsed_model_v1"


class ParsedModelIdentity(TypedDict):
    """Identity shared by a Step 3 manifest entry and its parsed model."""

    schema: str
    run_id: str | None
    artifact_stem: str
    model_id: str | None
    source_path: str
    source_relative_path: str
    source_sha256: str


def parsed_model_identity(
    record: dict[str, Any], artifact_stem: str, run_id: str | None
) -> ParsedModelIdentity:
    """Construct the embedded identity from a producer's source record."""
    return {
        "schema": PARSED_MODEL_SCHEMA,
        "run_id": run_id,
        "artifact_stem": artifact_stem,
        "model_id": record.get("model_id"),
        "source_path": record["source_path"],
        "source_relative_path": record["source_relative_path"],
        "source_sha256": record["source_sha256"],
    }


def read_parsed_model(
    path: Path,
    record: dict[str, Any],
    *,
    schema: str | None,
    run_id: str | None,
    source_sha256: str | None,
) -> tuple[dict[str, Any], ParsedModelIdentity | None]:
    """Read unversioned JSON or verify a Step 3 handoff before admission."""
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Parsed model must be a JSON object")
    provenance = payload.get("_gnn_provenance")
    if schema is None:
        if "parsed_model_sha256" in record or (
            isinstance(provenance, dict) and "schema" in provenance
        ):
            raise ValueError("Versioned parsed model requires a versioned manifest")
        return payload, None

    if schema != PARSED_MODEL_SCHEMA:
        raise ValueError(f"Unsupported parsed-model schema: {schema!r}")
    if hashlib.sha256(raw).hexdigest() != record.get("parsed_model_sha256"):
        raise ValueError(
            "Parsed model artifact checksum does not match the Step 3 manifest"
        )
    stem = record.get("artifact_stem") or Path(record["file_path"]).stem
    expected = parsed_model_identity(record, stem, run_id)
    if provenance != expected:
        raise ValueError("Parsed model identity does not match the Step 3 manifest")
    if source_sha256 != expected["source_sha256"]:
        raise ValueError("Model source changed after Step 3 parsing")
    return payload, expected


def selected_parse_receipt(
    receipt: dict[str, Any],
    step: int,
    *,
    receipt_identity: tuple[Path, str] | None = None,
) -> dict[str, Any]:
    """Require current-run coverage and preserve the consumer's staged identities."""
    schema = receipt.get("parsed_model_schema")
    if schema is not None and schema != PARSED_MODEL_SCHEMA:
        raise ValueError(f"Unsupported parsed-model schema: {schema!r}")
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
    root = Path(context.output_root).resolve()
    owned = owned_artifact_checksums(root) or {}
    if schema is not None and receipt_identity is not None:
        path, digest = receipt_identity
        resolved = path.resolve()
        if (
            not resolved.is_relative_to(root)
            or owned.get(resolved.relative_to(root).as_posix()) != digest
        ):
            raise ValueError(
                "Step 3 manifest checksum is outside the current-run inventory"
            )
    view = context.input_view(step)
    projected = []
    for model in selected:
        record = dict(records[model.model_id])
        if record.get("parsed_model_file"):
            parsed = Path(record["parsed_model_file"]).resolve()
            if (
                not parsed.is_relative_to(root)
                or (relative := parsed.relative_to(root).as_posix()) not in owned
            ):
                raise ValueError(
                    "Parsed model artifact is outside the current-run inventory"
                )
            if schema is not None and (
                owned[relative] is None
                or owned[relative] != record.get("parsed_model_sha256")
            ):
                raise ValueError(
                    "Parsed model checksum does not match current-run inventory"
                )
        if (
            schema is not None
            and record.get("parse_success")
            and (
                record.get("source_sha256") != model.sha256
                or record.get("source_relative_path") != model.relative_path
                or record.get("source_path") != model.source_path
            )
        ):
            raise ValueError("Parser source identity does not match the selected model")
        record["producer_file_path"] = record["file_path"]
        record["file_path"] = str(
            view / f"{model.artifact_stem}{Path(model.source_path).suffix}"
        )
        projected.append(record)
    return {**receipt, "processed_files": projected}
