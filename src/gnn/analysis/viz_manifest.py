#!/usr/bin/env python3
"""
GridWorld analysis manifest writing for GNN Step 16 analysis visualizations.

Extracted from ``analysis.visualizations``.
"""

import json
import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    Optional,
)

from .viz_schema import (
    _framework_from_path_or_payload,
    _is_gridworld_payload,
    _model_name_from_path,
)

logger = logging.getLogger(__name__)

# The manifest is the strict cross-framework execution proof for the public
# POMDP GridWorld run (PyMDP, RxInfer.jl, ActiveInference.jl). Ungated runner
# schemas (pytorch/numpyro) are plotted elsewhere but carry no
# ``matrix_provenance``, so they are excluded here.
GRIDWORLD_MANIFEST_SCHEMAS = frozenset(
    {
        "pymdp_simulation_v1",
        "rxinfer_simulation_v1",
        "activeinference_jl_simulation_v1",
    }
)


def _relative_or_absolute(path: Path, base: Path) -> str:
    """Handle relative or absolute for internal callers."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def write_gridworld_analysis_manifest(
    execution_dir: Path,
    analysis_dir: Path,
    allowed_frameworks: Optional[set[str]] = None,
    allowed_model_names: Optional[set[str]] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Write a manifest for current GridWorld logs, statistics, PNGs, and GIFs."""
    log = logger_instance or logger
    source_results: list[dict[str, Any]] = []
    provenances: list[Dict[str, Any]] = []

    for sim_file in sorted(execution_dir.rglob("*simulation_results.json")):
        try:
            payload = json.loads(sim_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.debug(f"Skipping unreadable simulation result {sim_file}: {e}")
            continue
        if not isinstance(payload, dict) or not _is_gridworld_payload(payload):
            continue
        if payload.get("schema_version") not in GRIDWORLD_MANIFEST_SCHEMAS:
            continue

        framework = _framework_from_path_or_payload(sim_file, payload)
        model_name = _model_name_from_path(sim_file)
        if allowed_frameworks and framework not in allowed_frameworks:
            continue
        if allowed_model_names and model_name not in allowed_model_names:
            continue

        matrix_provenance = payload.get("matrix_provenance", {})
        if isinstance(matrix_provenance, dict):
            provenances.append(matrix_provenance)
        source_results.append(
            {
                "framework": framework,
                "model_name": model_name,
                "schema_version": payload.get("schema_version"),
                "source_path": str(sim_file),
                "num_timesteps": payload.get("num_timesteps"),
                "validation": payload.get("validation", {}),
            }
        )

    if not source_results:
        return None

    current_frameworks = allowed_frameworks or {
        entry["framework"] for entry in source_results
    }
    current_models = allowed_model_names or {
        entry["model_name"] for entry in source_results
    }

    manifest_path = (
        analysis_dir / "cross_framework" / "gridworld_analysis_manifest.json"
    )

    current_cross_framework_files = {
        "confidence_comparison.png",
        "cross_framework_comparison.png",
        "efe_convergence_comparison.png",
        "framework_comparison_data.json",
        "framework_comparison_report.md",
        "framework_performance_comparison.png",
        "framework_radar.png",
        "framework_success_rates.png",
    }
    current_root_files = {
        "analysis_results.json",
        "analysis_summary.md",
        "cross_model_comparison_report.md",
    }

    def is_current_artifact(path: Path) -> bool:
        """Return whether current artifact."""
        if path == manifest_path or not path.is_file():
            return False

        try:
            rel = path.relative_to(analysis_dir)
        except ValueError:
            return False

        parts = rel.parts
        if not parts:
            return False

        name = path.name
        first = parts[0]
        if first in current_frameworks:
            return any(model_name in name for model_name in current_models)

        if first == "cross_framework":
            if "gridworld_animations" in parts:
                return name.startswith("gridworld_cross_framework_") or any(
                    model_name in name for model_name in current_models
                )
            if "unified_dashboard" in parts:
                return path.suffix == ".png"
            if name in current_cross_framework_files:
                return True
            if name.endswith("_post_simulation_analysis.json"):
                return any(
                    name == f"{model_name}_post_simulation_analysis.json"
                    for model_name in current_models
                )
            return False

        if len(parts) == 1:
            return name in current_root_files

        return False

    png_outputs = sorted(
        _relative_or_absolute(path, analysis_dir)
        for path in analysis_dir.rglob("*.png")
        if is_current_artifact(path)
    )
    gif_outputs = sorted(
        _relative_or_absolute(path, analysis_dir)
        for path in analysis_dir.rglob("*.gif")
        if is_current_artifact(path)
    )
    dashboard_outputs = sorted(
        _relative_or_absolute(path, analysis_dir)
        for path in (analysis_dir / "cross_framework" / "unified_dashboard").glob(
            "*.png"
        )
        if is_current_artifact(path)
    )
    statistics_outputs = sorted(
        _relative_or_absolute(path, analysis_dir)
        for path in analysis_dir.rglob("*")
        if path.suffix in {".json", ".md"} and is_current_artifact(path)
    )

    first_provenance = provenances[0] if provenances else {}
    matrix_provenance_equal = all(
        provenance == first_provenance for provenance in provenances
    )
    manifest = {
        "schema_version": "gridworld_analysis_manifest_v1",
        "model_names": sorted({entry["model_name"] for entry in source_results}),
        "frameworks": sorted({entry["framework"] for entry in source_results}),
        "source_results": source_results,
        "matrix_provenance_equal": matrix_provenance_equal,
        "outputs": {
            "statistics": statistics_outputs,
            "png": png_outputs,
            "gif": gif_outputs,
            "dashboard": dashboard_outputs,
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info(f"Generated GridWorld analysis manifest: {manifest_path}")
    return str(manifest_path)
