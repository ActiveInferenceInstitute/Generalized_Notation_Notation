"""Load canonical visualization dict, preferring step-3 `{model}_parsed.json`."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from gnn.processing.discovery import recorded_artifact_stem

from ..parse.markdown import parse_gnn_content

logger = logging.getLogger(__name__)


def resolve_gnn_step3_output_dir(results_dir: Path) -> Path:
    """Directory where step 3 writes `{model}/{model}_parsed.json`."""
    from gnn.pipeline.config import get_output_dir_for_script

    parent = results_dir.parent if results_dir.name.endswith("_output") else results_dir
    return Path(get_output_dir_for_script("3_gnn.py", parent))


def load_parse_artifact_stems(results_dir: Path) -> dict[Path, str]:
    """Read Step 3's overrides for colliding standalone source paths."""
    receipt = resolve_gnn_step3_output_dir(results_dir) / "gnn_processing_results.json"
    try:
        result = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    records = result.get("processed_files") if isinstance(result, dict) else None
    if not isinstance(records, list):
        return {}
    stems = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        source = record.get("source_path")
        stem = recorded_artifact_stem(record.get("artifact_stem"))
        if isinstance(source, str) and stem is not None:
            stems[Path(source)] = stem
    return stems


def _ontology_list_to_dict(mappings: Any) -> Dict[str, str]:
    """Handle ontology list to dict for internal callers."""
    out: Dict[str, str] = {}
    if isinstance(mappings, list):
        for m in mappings:
            if isinstance(m, dict):
                vn = m.get("variable_name") or m.get("name")
                term = m.get("ontology_term") or m.get("term")
                if vn and term:
                    out[str(vn)] = str(term)
    elif isinstance(mappings, dict):
        for k, v in mappings.items():
            out[str(k)] = str(v)
    return out


def _dict_from_parsed_json(
    data: Dict[str, Any], json_path: Path, stale: bool
) -> Dict[str, Any]:
    """Handle dict from parsed json for internal callers."""
    raw_sections = data.get("raw_sections") or {}
    if not isinstance(raw_sections, dict):
        raw_sections = {}

    return {
        "sections": {
            k: [line for line in str(v).splitlines() if line.strip()]
            for k, v in raw_sections.items()
        },
        "raw_sections": raw_sections,
        "variables": list(data.get("variables") or []),
        "connections": list(data.get("connections") or []),
        "matrices": [],
        "parameters": list(data.get("parameters") or []),
        "metadata": {
            "model_name": data.get("model_name", ""),
            "annotation": data.get("annotation", ""),
            "version": data.get("version", ""),
        },
        "ontology_mappings": data.get("ontology_mappings"),
        "ontology_labels": _ontology_list_to_dict(data.get("ontology_mappings")),
        "_viz_meta": {
            "source": "parsed_json",
            "json_path": str(json_path),
            "json_stale": stale,
        },
    }


def load_visualization_model(
    gnn_file: Path,
    content: str,
    results_dir: Path,
    verbose: bool = False,
    *,
    parsed_model: Optional[Dict[str, Any]] = None,
    parsed_artifact_stem: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the visualization input dict.

    Uses `{stem}_parsed.json` from step 3 when present and at least as new as the
    source file; otherwise parses markdown. When JSON exists but is older than the
    source, still loads JSON (JSON-primary) and sets ``_viz_meta.json_stale`` and
    expects the caller to write a warning note.

    ``parsed_model`` (consolidated executor carrier) supplies the same
    step-3 JSON payload in memory; when given, the on-disk JSON is not
    re-read and staleness is still derived from the file mtimes, producing
    a dict identical to a disk load.
    """
    model_name = parsed_artifact_stem or load_parse_artifact_stems(results_dir).get(
        gnn_file.absolute(), gnn_file.stem
    )
    if Path(model_name).name != model_name or model_name in {".", ".."}:
        raise ValueError("Parsed artifact stem must be a single directory name")
    gnn_out = resolve_gnn_step3_output_dir(results_dir)
    parsed_json = gnn_out / model_name / f"{model_name}_parsed.json"

    if parsed_model is not None and isinstance(parsed_model, dict):
        # Consolidated in-memory carrier: the executor already read
        # ``{model}_parsed.json`` once; reuse those bytes. Staleness still
        # comes from the on-disk mtimes so ``_viz_meta`` stays identical to
        # a disk load.
        try:
            stale = parsed_json.stat().st_mtime < gnn_file.stat().st_mtime
        except OSError as e:
            logger.warning(
                "Could not stat %s for staleness check: %s; assuming fresh",
                parsed_json,
                e,
            )
            stale = False
        result = _dict_from_parsed_json(parsed_model, parsed_json, stale)
        if verbose and stale:
            logger.warning(
                "Parsed JSON older than %s; re-run step 3 for fresh data. Using JSON anyway.",
                gnn_file.name,
            )
        elif verbose:
            logger.info("Visualization data loaded from %s", parsed_json)
        return result

    if parsed_json.is_file():
        try:
            src_mtime = gnn_file.stat().st_mtime
            json_mtime = parsed_json.stat().st_mtime
            stale = json_mtime < src_mtime
            with open(parsed_json, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                result = _dict_from_parsed_json(data, parsed_json, stale)
                if verbose and stale:
                    logger.warning(
                        "Parsed JSON older than %s; re-run step 3 for fresh data. Using JSON anyway.",
                        gnn_file.name,
                    )
                elif verbose:
                    logger.info("Visualization data loaded from %s", parsed_json)
                return result
        except Exception as e:
            logger.warning(
                "Failed to load %s: %s; falling back to markdown", parsed_json, e
            )

    if verbose:
        logger.info("Visualization data from markdown parse (%s)", gnn_file.name)
    md = parse_gnn_content(content)
    md["_viz_meta"] = {"source": "markdown", "json_path": None, "json_stale": False}
    md.setdefault("ontology_labels", {})
    return md


def stale_json_note_text(gnn_file: Path, parsed_json: Path) -> str:
    """Provide stale json note text behavior."""
    return (
        f"Step-3 parsed JSON is older than the source GNN file.\n"
        f"Source: {gnn_file}\n"
        f"JSON:   {parsed_json}\n"
        f"Re-run: uv run python src/gnn/main.py --only-steps 3 --verbose\n"
    )


def write_stale_json_note_if_needed(
    parsed_data: Dict[str, Any], model_dir: Path, model_name: str, gnn_file: Path
) -> None:
    """Write stale json note if needed."""
    meta = parsed_data.get("_viz_meta") or {}
    if not meta.get("json_stale"):
        return
    path_str = meta.get("json_path")
    if not path_str:
        return
    note = model_dir / f"{model_name}_viz_source_note.txt"
    try:
        note.write_text(
            stale_json_note_text(gnn_file, Path(path_str)), encoding="utf-8"
        )
    except OSError as e:
        logger.debug("Could not write stale JSON note: %s", e)
