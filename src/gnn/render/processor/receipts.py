#!/usr/bin/env python3
"""Render receipt persistence and framework selection.

Atomic JSON receipt writing with same-run carry-forward, source and
artifact identity digests, and normalization of the ``frameworks``
selection against the canonical registry.
"""

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from gnn.frameworks import RENDER_FRAMEWORKS
from gnn.render.framework_registry import get_lite_frameworks

logger = logging.getLogger(__name__)


def _load_prior_render_summary(summary_file: Path) -> Dict[str, Any]:
    """Load a previously written render summary, or an empty dict.

    The pipeline invokes ``process_render`` once per top-level input folder,
    each writing the same ``render_processing_summary.json``. This helper lets
    a later invocation carry forward the earlier folders' ``file_results`` and
    aggregate counts so Step 12's manifest-based script discovery (V-10) sees
    every rendered script, not just the last folder's.
    """
    if not summary_file.exists():
        return {}
    try:
        data = json.loads(summary_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        logger.warning(
            "Could not read prior render summary %s; starting fresh", summary_file
        )
        return {}


def _render_file_identity(path: Path) -> Dict[str, str]:
    """Identify a source or artifact by canonical path and its current bytes."""
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _atomic_render_json(path: Path, payload: Dict[str, Any]) -> None:
    """Publish a complete JSON document with a same-directory atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _write_render_receipt(
    output_dir: Path,
    target_dir: Path,
    results: Dict[str, Any],
    configuration: Dict[str, Any],
    run_id: str,
    processing_type: str,
) -> Dict[str, Any]:
    """Replace this scope, carry only verified same-run records, and recount."""
    summary_file = output_dir / "render_processing_summary.json"
    prior = _load_prior_render_summary(summary_file)
    if prior:
        # Digest the prior receipt WITHOUT its wall-clock timestamp so an
        # otherwise-identical rerun maps to the same history archive name
        # instead of appending a new render-<digest>.json every invocation.
        digest_payload = {
            key: value for key, value in prior.items() if key != "timestamp"
        }
        digest = hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        history = output_dir / "history" / f"render-{digest}.json"
        if not history.exists():
            _atomic_render_json(history, prior)
    identity = {
        "run_id": run_id,
        "config_sha256": hashlib.sha256(
            json.dumps(configuration, sort_keys=True, default=str).encode()
        ).hexdigest(),
    }
    merged: Dict[str, Any] = {}
    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    prior_results = prior.get("file_results", {})
    if (
        context is None
        and prior.get("receipt_identity") == identity
        and isinstance(prior_results, dict)
    ):
        for source, record in prior_results.items():
            path = Path(source).resolve()
            if path.is_relative_to(target_dir.resolve()) or not isinstance(
                record, dict
            ):
                continue
            try:
                artifacts_valid = all(
                    isinstance(fr, dict)
                    and all(
                        artifact == _render_file_identity(Path(artifact["path"]))
                        for artifact in fr.get("artifact_identities", [])
                    )
                    for fr in record.get("framework_results", {}).values()
                )
                if artifacts_valid and record.get(
                    "source_identity"
                ) == _render_file_identity(path):
                    merged[str(path)] = record
            except (OSError, TypeError, KeyError, AttributeError):
                continue
    for source, record in results.items():
        path = Path(source).resolve()
        try:
            source_identity = _render_file_identity(path)
            if record.get("source_identity", source_identity) != source_identity:
                record = {
                    "overall_success": False,
                    "framework_results": {},
                    "error": "Source changed while rendering",
                }
            record["source_identity"] = source_identity
            for framework_result in record.get("framework_results", {}).values():
                if isinstance(framework_result, dict):
                    framework_result["artifact_identities"] = [
                        _render_file_identity(Path(artifact))
                        for artifact in framework_result.get("output_files", [])
                        if Path(artifact).is_file()
                    ]
        except OSError as exc:
            record = {
                "overall_success": False,
                "framework_results": {},
                "error": str(exc),
            }
        merged[str(path)] = record
    from gnn.pipeline.run_context import current_run_context, model_provenance

    context = current_run_context()
    if context is not None:
        for source, record in merged.items():
            record.update(model_provenance(Path(source), 11))
    successful = attempts = rendered = 0
    failed: List[Dict[str, str]] = []
    unsupported: List[Dict[str, str]] = []
    for source, record in merged.items():
        successful += bool(record.get("overall_success", record.get("success", False)))
        framework_results = record.get("framework_results", {})
        if not isinstance(framework_results, dict):
            continue
        for framework, result in framework_results.items():
            if not isinstance(result, dict):
                continue
            diagnostic = {
                "file": source,
                "framework": framework,
                "message": str(result.get("message", "")),
            }
            if result.get("unsupported"):
                unsupported.append(diagnostic)
            else:
                attempts += 1
                rendered += bool(result.get("success"))
                if not result.get("success"):
                    failed.append(diagnostic)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "processing_type": processing_type,
        "receipt_identity": identity,
        "target_directory": str(target_dir.resolve()),
        "configuration": configuration,
        "file_results": merged,
        "total_files": len(merged),
        "successful_files": successful,
        "failed_files": len(merged) - successful,
        "total_framework_attempts": attempts,
        "successful_framework_renderings": rendered,
        "framework_success_rate": rendered / attempts * 100 if attempts else 0,
        "failed_framework_renderings": failed,
        "unsupported_framework_renderings": unsupported,
    }
    if context is not None:
        summary["run_id"] = context.run_id
        summary["model_selection"] = [
            model.__dict__ for model in context.selected_models(11)
        ]
    _atomic_render_json(summary_file, summary)
    return summary


def parse_frameworks_selection(
    frameworks: Union[str, List[str], None],
) -> Tuple[Optional[List[str]], bool]:
    """Normalize the ``frameworks`` selection into a framework list.

    Accepts ``None`` (all frameworks), ``"all"``, the ``"lite"`` preset
    (resolved through the canonical registry), or a comma-separated string /
    list of framework names.

    Returns:
        Tuple of ``(frameworks, explicit_request)`` where *frameworks* is
        ``None`` for "every registered framework" and *explicit_request* is
        ``True`` when the caller pinned a specific framework set (which the
        caller treats as a strict-success policy).
    """
    if isinstance(frameworks, str):
        normalized = frameworks.strip().lower()
        if normalized == "all":
            return None, False
        if normalized == "lite":
            return get_lite_frameworks(), False
        frameworks = [f.strip() for f in frameworks.split(",")]
    if frameworks is not None:
        if not isinstance(frameworks, list) or not frameworks:
            raise ValueError("Explicit frameworks must be a nonempty list")
        for name in frameworks:
            if not isinstance(name, str) or name not in RENDER_FRAMEWORKS:
                raise ValueError(f"Unknown render framework: {name!r}")
        if len(frameworks) != len(set(frameworks)):
            raise ValueError("Framework selection must not contain duplicates")
    return frameworks, frameworks is not None
