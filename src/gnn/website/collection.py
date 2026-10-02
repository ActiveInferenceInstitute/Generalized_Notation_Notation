"""Pipeline-artifact collection for the website generator.

Gathers every artifact the static site renders — GNN source files, per-step
statuses, analysis results, visualization assets, report artifacts, and the
step-21 MCP page data — into the plain dict consumed by
``gnn.website.generator.WebsiteGenerator`` and exported to callers as
``gnn.website.collect_website_data``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

from .asset_transport import copy_interactive_bundle, selected_website_asset
from .source_collection import _collect_gnn_files, _collect_parsed_models
from .steps import PIPELINE_STEPS

logger = logging.getLogger(__name__)

# Caller-supplied datasets that fully replace disk collection in pure-dict mode.
PURE_DICT_KEYS: frozenset[str] = frozenset(
    {
        "gnn_files",
        "models",
        "analysis",
        "visualizations",
        "reports",
        "mcp_tools",
        "mcp_summary",
        "pipeline_summary",
        "step_statuses",
        "processed_files",
        "gui_navigation",
    }
)


def website_data_from_dict(
    user_data: dict[str, Any],
    *,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Build the generator's data dict from a caller-supplied dict (NO disk access).

    Pure-dict mode: the caller supplies the datasets; nothing is collected
    from disk. Keys absent from ``user_data`` take the exact empty defaults
    the filesystem collectors produce (empty lists/``{}``/all-pending
    statuses, ``p_root=None`` → the pages' truthful no-root empty state).
    Keys other than the known dataset keys are preserved verbatim (e.g.
    ``search_data``). ``output_dir`` normalizes like
    ``collect_website_data`` does; ``p_root``/``output_dir`` caller values
    are kept when provided.
    """
    data: dict[str, Any] = {
        "p_root": None,
        "output_dir": Path(output_dir) if output_dir is not None else None,
        "gnn_files": [],
        "models": [],
        "analysis": [],
        "warnings": [],
        "visualizations": [],
        "reports": [],
        "mcp_tools": [],
        "mcp_summary": {},
        "pipeline_summary": {},
        "step_statuses": {step.number: "pending" for step in PIPELINE_STEPS},
        "processed_files": 0,
        "gui_navigation": False,
    }
    if user_data:
        data.update(
            {
                k: v
                for k, v in user_data.items()
                if k not in ("output_dir", "input_dir", "pipeline_output_root")
            }
        )
        root = user_data.get("pipeline_output_root")
        if root is not None:
            data["p_root"] = Path(root)
    return data


def _load_pipeline_summary(p_root: Path) -> dict[str, Any]:
    """Load ``pipeline_execution_summary.json`` (absent or malformed → ``{}``)."""
    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    candidates: tuple[Path, ...] = (
        p_root / "00_pipeline_summary" / "pipeline_execution_summary.json",
        p_root / "pipeline_execution_summary.json",
    )
    if context is not None:
        candidates = (p_root / "00_pipeline_summary" / "current_summary.json",)
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
            if context is not None and loaded.get("run_id") != context.run_id:
                continue
        except Exception as e:
            logger.debug(f"Skipped malformed pipeline summary file: {e}")
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _normalize_website_step_status(raw: Any) -> str | None:
    """Map a recorded pipeline step status to the site badge vocabulary.

    Returns ``ok``/``skip``/``error``, or ``None`` for an empty/unusable
    status (the step then stays ``pending``).
    """
    normalized = str(raw or "").strip().upper()
    if not normalized:
        return None
    if "SKIP" in normalized:
        return "skip"
    if "SUCCESS" in normalized and "PARTIAL" not in normalized:
        return "ok"
    if normalized in ("PASS", "PASSED", "OK", "COMPLETED"):
        return "ok"
    return "error"


def _step_statuses_from_summary(summary: dict[str, Any]) -> dict[int, str]:
    """Per-step site statuses derived from the pipeline summary's ``steps`` list."""
    statuses: dict[int, str] = {}
    raw_steps = summary.get("steps")
    if not isinstance(raw_steps, list):
        return statuses
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            continue
        match = re.match(r"(?P<number>\d+)_", str(raw_step.get("script_name", "")))
        if match:
            step_number = int(match.group("number"))
        elif isinstance(raw_step.get("step_number"), int):
            step_number = int(raw_step["step_number"])
        else:
            continue
        badge = _normalize_website_step_status(raw_step.get("status"))
        if badge is not None:
            statuses[step_number] = badge
    return statuses


def _collect_step_statuses(p_root: Path) -> dict[int, str]:
    """Map each step number to ``ok``/``skip``/``error``/``pending``.

    Primary source is the durable ``pipeline_execution_summary.json`` written
    by the orchestrator: a step whose output dir exists but whose recorded
    status is ``FAILED`` or ``SKIPPED`` must not be advertised as complete.
    Falls back to the numbered-output-dir heuristic only when the summary is
    absent, malformed, or carries no parseable per-step records.
    """
    pending = {step.number: "pending" for step in PIPELINE_STEPS}
    if not p_root.exists():
        return pending
    summary = _load_pipeline_summary(p_root)
    if summary:
        from_summary = _step_statuses_from_summary(summary)
        if from_summary:
            return {**pending, **from_summary}
    dir_names = [d.name for d in p_root.iterdir() if d.is_dir()]
    statuses: dict[int, str] = {}
    for step in PIPELINE_STEPS:
        found = any(
            name.startswith(f"{step.number:02d}_") or name.startswith(f"{step.number}_")
            for name in dir_names
        )
        statuses[step.number] = "ok" if found else "pending"
    return statuses


def _collect_analysis_results(p_root: Path) -> list[Any]:
    """Load Step 16 analysis JSON files (best effort, deduplicated by name)."""
    results: list[Any] = []
    seen: set[str] = set()
    for candidate in (
        p_root / "16_analysis_output" / "analysis_results",
        p_root / "16_analysis_output",
    ):
        if not candidate.exists():
            continue
        from gnn.pipeline.artifact_ownership import current_artifact_files

        for jf in current_artifact_files(candidate, "*.json"):
            identity = jf.resolve().as_posix()
            if identity in seen or not selected_website_asset(jf):
                continue
            seen.add(identity)
            try:
                loaded = json.loads(jf.read_text())
            except Exception as e:
                logger.debug(f"Skipped malformed analysis file {jf.name}: {e}")
                continue
            if not isinstance(loaded, dict):
                logger.debug(f"Skipped non-dict analysis file {jf.name}")
                continue
            results.append(loaded)

    return results


def _load_mcp_summary(p_root: Path) -> dict[str, Any]:
    """Load the Step 21 MCP processing summary (absent or malformed → ``{}``)."""
    candidate = p_root / "21_mcp_output" / "mcp_processing_summary.json"
    from gnn.pipeline.artifact_ownership import owned_artifact_paths

    inventory = owned_artifact_paths(p_root)
    if (
        inventory is not None
        and candidate.relative_to(p_root).as_posix() not in inventory
    ):
        return {}
    if not candidate.exists():
        return {}
    try:
        loaded = json.loads(candidate.read_text())
    except Exception as e:
        logger.debug(f"Skipped malformed MCP summary file: {e}")
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_registered_tools(p_root: Path) -> list[dict[str, str]]:
    """Map ``registered_tools.json`` (step 21) to the page tool-card shape."""
    candidate = p_root / "21_mcp_output" / "registered_tools.json"
    from gnn.pipeline.artifact_ownership import owned_artifact_paths

    inventory = owned_artifact_paths(p_root)
    if (
        inventory is not None
        and candidate.relative_to(p_root).as_posix() not in inventory
    ):
        return []
    if not candidate.exists():
        logger.debug("No registered_tools.json found for website (step 21 optional)")
        return []
    try:
        loaded = json.loads(candidate.read_text())
    except Exception as e:
        logger.debug(f"Skipped malformed registered tools file: {e}")
        return []
    if not isinstance(loaded, list):
        logger.debug("registered_tools.json is not a list; ignoring it")
        return []
    cards: list[dict[str, str]] = []
    for item in loaded:
        if not isinstance(item, dict):
            continue
        cards.append(
            {
                "name": item.get("name", ""),
                "module": item.get("module", ""),
                "desc": item.get("description") or item.get("desc") or "",
                "category": item.get("category", ""),
            }
        )
    return cards


def _collect_visualizations(
    viz_dirs: list[Path], assets_dir: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    """Copy PNG/HTML visualization artifacts into ``assets_dir`` and describe them.

    Identical artifact filenames from different source directories must not
    silently overwrite each other in the shared ``assets_dir``: later
    collisions are copied under a ``<source-dir>__``-prefixed (or counter-
    disambiguated) name so every artifact keeps its own gallery card.

    An artifact whose copy fails is skipped (never linked from the site):
    the caller gets ``(visualizations, warnings)`` with a message per
    skipped artifact instead of a gallery card whose ``assets/`` href
    would dangle.
    """
    visualizations: list[dict[str, Any]] = []
    warnings: list[str] = []
    used_names: set[str] = set()
    from gnn.pipeline.artifact_ownership import owned_artifact_paths

    inventory = owned_artifact_paths(assets_dir.parent.parent)
    for viz_dir in viz_dirs:
        if not viz_dir.exists():
            continue
        for pattern, artifact_type in (("*.png", "image"), ("*.html", "html")):
            for artifact in viz_dir.rglob(pattern):
                if (
                    inventory is not None
                    and artifact.relative_to(assets_dir.parent.parent).as_posix()
                    not in inventory
                ):
                    continue
                if not selected_website_asset(artifact):
                    continue
                if artifact_type == "html":
                    try:
                        bundle_path = copy_interactive_bundle(
                            artifact,
                            viz_dir,
                            assets_dir,
                            assets_dir.parent.parent,
                            inventory,
                        )
                    except (OSError, ValueError) as error:
                        warnings.append(
                            f"Skipped visualization {artifact.name} (bundle copy failed: {error})"
                        )
                        continue
                    visualizations.append(
                        {
                            **_visualization_provenance(artifact),
                            "title": artifact.stem,
                            "path": bundle_path,
                            "type": artifact_type,
                        }
                    )
                    continue
                dest_name = artifact.name
                if dest_name.lower() in used_names:
                    prefix = f"{viz_dir.name}__"
                    dest_name = f"{prefix}{artifact.name}"
                    stem, ext = os.path.splitext(dest_name)
                    counter = 2
                    while dest_name.lower() in used_names:
                        dest_name = f"{stem}_{counter}{ext}"
                        counter += 1
                dest = assets_dir / dest_name
                try:
                    shutil.copy2(artifact, dest)
                except OSError as e:
                    warnings.append(
                        f"Skipped visualization {artifact.name} (copy failed: {e})"
                    )
                    continue
                used_names.add(dest_name.lower())
                visualizations.append(
                    {
                        **_visualization_provenance(artifact),
                        "title": artifact.stem,
                        "path": dest.name,
                        "type": artifact_type,
                    }
                )
    return visualizations, warnings


def _visualization_provenance(artifact: Path) -> dict[str, str]:
    """Identify assets using the original path-bound model, not display names."""
    from gnn.pipeline.run_context import current_run_context, model_provenance

    context = current_run_context()
    if context is None:
        return {}
    provenance = model_provenance(artifact, 20)
    if provenance:
        return provenance
    matches = [
        model
        for model in context.selected_models(20)
        if artifact.stem == model.artifact_stem
        or artifact.stem.startswith(model.artifact_stem + "_")
    ]
    if len(matches) == 1:
        model = matches[0]
        return {
            "model_id": model.model_id,
            "source_path": model.source_path,
            "source_relative_path": model.relative_path,
            "source_sha256": model.sha256,
        }
    return {}


def _collect_reports(p_root: Path) -> list[dict[str, Any]]:
    """Collect capped JSON artifacts from every numbered output directory."""
    reports: list[dict[str, Any]] = []
    if not p_root.exists():
        return reports
    for entry in sorted(p_root.iterdir()):
        if not (entry.is_dir() and entry.name[0].isdigit()):
            continue
        from gnn.pipeline.artifact_ownership import current_artifact_files

        for jf in current_artifact_files(entry, "*.json")[:5]:  # cap per dir
            try:
                content = jf.read_text(encoding="utf-8", errors="replace")
                reports.append(
                    {
                        "name": jf.name,
                        "dir": entry.name,
                        "content": content[:2000],
                        "size": jf.stat().st_size,
                    }
                )
            except Exception as e:
                logger.debug(f"Skipped unreadable report file {jf.name}: {e}")
    return reports


def collect_website_data(
    pipeline_output_root: Path,
    input_dir: Path,
    assets_dir: Path,
    *,
    output_dir: Path | None = None,
    user_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate every artifact the website pages render.

    Pure with respect to the repository state except for copying
    visualization assets into ``assets_dir``. Step statuses come from the
    durable ``pipeline_execution_summary.json`` written by the orchestrator
    (numbered-output-dir heuristic only as a fallback when the summary is
    absent). The MCP page data comes from the step-21 artifacts:
    ``21_mcp_output/mcp_processing_summary.json`` for the summary and
    ``registered_tools.json`` for the tool inventory, so the site reflects
    what steps actually recorded.
    """
    p_root = Path(pipeline_output_root)
    data: dict[str, Any] = {
        "p_root": p_root,
        "output_dir": Path(output_dir) if output_dir is not None else None,
        "gnn_files": [],
        "models": [],
        "analysis": [],
        "warnings": [],
        "visualizations": [],
        "reports": [],
        "mcp_tools": [],
        "mcp_summary": {},
        "step_statuses": {},
        "processed_files": 0,
        "gui_navigation": (p_root / "22_gui_output" / "navigation.html").is_file(),
    }
    if user_data:
        data.update(
            {
                k: v
                for k, v in user_data.items()
                if k not in ("output_dir", "input_dir", "pipeline_output_root")
            }
        )

    discovered, found_source = _collect_gnn_files(p_root, input_dir)
    data["gnn_files"].extend(discovered)
    if found_source:
        data["processed_files"] = len(data["gnn_files"])
    data["models"].extend(_collect_parsed_models(data["gnn_files"]))
    data["step_statuses"] = _collect_step_statuses(p_root)
    data["analysis"].extend(_collect_analysis_results(p_root))
    viz_entries, viz_warnings = _collect_visualizations(
        [
            p_root / "08_visualization_output" / "visualization_results",
            p_root / "8_visualization_output" / "visualization_results",
            p_root / "09_advanced_viz_output",
            p_root / "9_advanced_viz_output",
        ],
        assets_dir,
    )
    data["visualizations"].extend(viz_entries)
    data["warnings"].extend(viz_warnings)
    data["reports"].extend(_collect_reports(p_root))
    data["mcp_summary"] = _load_mcp_summary(p_root)
    data["pipeline_summary"] = _load_pipeline_summary(p_root)
    data["mcp_tools"] = _load_registered_tools(p_root)
    return data
