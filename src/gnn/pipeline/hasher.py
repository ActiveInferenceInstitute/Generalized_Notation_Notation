#!/usr/bin/env python3
"""
Content-Addressable Run Hashing — Reproducible pipeline identification.

Provides:
  - compute_run_hash(): SHA256 of input files + config → 12-char hex prefix
  - index_run(): store run metadata in .history/index.json
  - lookup_run(): retrieve run config by hash prefix
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

logger = logging.getLogger(__name__)
RUN_HASH_SCHEMA = "gnn-run-v2"
# Repo root is parents[3] of src/gnn/pipeline/hasher.py (parents[2] is src/).
RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[3] / "input" / "config.yaml"


def runtime_config_identity() -> Dict[str, Any]:
    """Bind the live file read by child modules, which cannot all use overrides."""
    if not RUNTIME_CONFIG_PATH.exists():
        return {"present": False, "sha256": None}
    with RUNTIME_CONFIG_PATH.open("rb") as handle:
        return {
            "present": True,
            "sha256": hashlib.file_digest(handle, "sha256").hexdigest(),
        }


def effective_run_config(
    arguments: Dict[str, Any],
    pipeline_settings: Dict[str, Any],
    selected_steps: list[str],
    input_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Canonical semantic configuration; output/logging/invocation data is excluded."""
    incidental = {
        "target_dir",
        "output_dir",
        "verbose",
        "log_format",
        "run_id",
        "only_steps",
        "skip_steps",
        "pipeline_summary_file",
    }
    args = {key: value for key, value in arguments.items() if key not in incidental}
    return {
        "arguments": args,
        "pipeline": pipeline_settings,
        "selected_steps": list(selected_steps),
        "runtime_config": runtime_config_identity(),
        "input_config": input_config
        if input_config is not None
        else {"pipeline": pipeline_settings},
    }


def verify_indexed_run(entry: Dict[str, Any]) -> list[str]:
    """Reject reproduction when indexed source or effective configuration has drifted."""
    config = entry.get("config")
    if not isinstance(config, dict):
        return ["Run history configuration is malformed"]
    identity = config.get("identity_config")
    hashes = entry.get("file_hashes")
    if (
        config.get("run_hash_schema") != RUN_HASH_SCHEMA
        or not isinstance(identity, dict)
        or not isinstance(hashes, dict)
    ):
        return [
            "Unbound run record lacks verifiable identity; create a new indexed run"
        ]
    args = config.get("args")
    settings = config.get("pipeline")
    if not isinstance(args, dict) or not isinstance(settings, dict):
        return ["Run history arguments or pipeline settings are malformed"]
    try:
        effective = effective_run_config(
            args, settings, identity["selected_steps"], identity["input_config"]
        )
        if effective != identity:
            return [
                "Saved effective configuration does not match reconstructed arguments"
            ]
        target = Path(args["target_dir"])
        if not target.is_dir():
            return ["Recorded input directory is missing"]
        digest, current = compute_run_hash_with_files(target, config=effective)
        if current != hashes:
            return ["Input inventory/content differs from the indexed run"]
        if digest != entry.get("run_hash"):
            return ["Run hash differs from the indexed source/configuration identity"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return [f"Cannot verify indexed run: {exc}"]
    return []


def _compute_run_hash_impl(
    target_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    hash_length: int = 12,
) -> Tuple[str, Dict[str, str]]:
    """Core implementation — returns (run_hash, file_hashes_dict)."""
    from gnn.parsers.common import get_supported_gnn_extensions
    from gnn.processing.discovery import is_model_source_path

    target_dir = Path(target_dir)
    file_hashes: Dict[str, str] = {}
    if target_dir.exists():
        files = sorted(
            {
                path
                for suffix in set(get_supported_gnn_extensions()) | {".gnn", ".txt"}
                for path in target_dir.rglob(f"*{suffix}")
                if is_model_source_path(path)
            }
        )
        for path in files:
            with path.open("rb") as handle:
                file_hashes[path.relative_to(target_dir).as_posix()] = (
                    hashlib.file_digest(handle, "sha256").hexdigest()
                )
    payload = {"schema": RUN_HASH_SCHEMA, "files": file_hashes, "config": config or {}}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    run_hash = hashlib.sha256(encoded).hexdigest()[:hash_length]
    logger.debug("Run hash: %s (%s input files)", run_hash, len(file_hashes))
    return run_hash, file_hashes


def compute_run_hash(
    target_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    hash_length: int = 12,
) -> str:
    """Compute a content-addressable hash for a pipeline run. Returns hex string."""
    run_hash, _ = _compute_run_hash_impl(target_dir, config, hash_length)
    return run_hash


def compute_run_hash_with_files(
    target_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    hash_length: int = 12,
) -> Tuple[str, Dict[str, str]]:
    """Compute a content-addressable hash and return (hash, file_hashes_dict)."""
    return _compute_run_hash_impl(target_dir, config, hash_length)


def index_run(
    run_hash: str,
    summary_path: Path,
    history_dir: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    file_hashes: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Store run metadata in .history/index.json.

    Args:
        run_hash: The computed run hash.
        summary_path: Path to pipeline_execution_summary.json.
        history_dir: Archive dir. Defaults to summary_path.parent / ".history".
        config: Optional config dict for re-running.
        file_hashes: Optional dict of file hashes.

    Returns:
        Path to index.json.
    """
    history_dir = history_dir or summary_path.parent / ".history"
    history_dir.mkdir(parents=True, exist_ok=True)
    index_path = history_dir / "index.json"

    # Load existing index
    index: dict[str, dict[str, Any]] = {}
    if index_path.exists():
        try:
            with open(index_path) as f:
                index = json.load(f)
            if not isinstance(index, dict):
                raise ValueError("Run history must be an object")
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"Could not load history index {index_path}: {e}")

    # Add/update entry
    entry: dict[str, Any] = {
        "run_hash": run_hash,
        "summary_path": str(summary_path),
        "config": config or {},
    }
    if file_hashes is not None:
        entry["file_hashes"] = file_hashes

    index[run_hash] = entry

    # Atomic replace so a concurrent reader never observes a torn index and a
    # crashed write cannot destroy the previously indexed runs.
    from gnn.pipeline._io import atomic_write_text

    atomic_write_text(index_path, json.dumps(index, indent=2))

    logger.info(f"📇 Indexed run {run_hash} in {index_path}")
    return index_path


def lookup_run(
    run_hash_prefix: str,
    history_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    Look up a run by its hash prefix.

    Args:
        run_hash_prefix: Partial or full hash to match.
        history_dir: Directory containing index.json.

    Returns:
        Run entry dict if found, None otherwise.
    """
    index_path = history_dir / "index.json"
    if not index_path.exists():
        return None

    try:
        with open(index_path) as f:
            index = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if not isinstance(index, dict):
        return None

    # Exact match first
    if run_hash_prefix in index:
        return cast("dict[str, Any] | None", index[run_hash_prefix])

    # Prefix match
    matches = {k: v for k, v in index.items() if k.startswith(run_hash_prefix)}
    if len(matches) == 1:
        return cast("dict[str, Any] | None", next(iter(matches.values())))
    elif len(matches) > 1:
        logger.warning(
            f"Ambiguous hash prefix '{run_hash_prefix}': {len(matches)} matches"
        )

    return None
