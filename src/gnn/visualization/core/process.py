"""Step-8 visualization orchestration (JSON-first model loading).

The orchestration lives here; the heavy rendering lives in the
``matrix``, ``graph`` and ``analysis`` subpackages. Per-file work is
broken into small cohesive helpers so they can be exercised in isolation:

* :func:`discover_visualization_files` — deterministic input discovery.
* :func:`load_cached_artifacts` — mtime-gated PNG cache reuse.
* :func:`sample_parsed_data` (in :mod:`visualization.core.sampling`) — pure
  downsampling of large models.
* :func:`collect_visualization_matrices` (in :mod:`visualization.matrix.extract`)
  — pure matrix collection from a parsed model dict.
* :func:`render_matrix_artifacts` — 2D/3D matrix render dispatch.
* :func:`write_viz_manifest` — per-model artifact manifest.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from gnn.processing.discovery import is_model_source_path
from gnn.utils.logging_utils import (
    log_step_error,
    log_step_start,
    log_step_success,
    log_step_warning,
)

from ..analysis.combined_analysis import (
    generate_combined_analysis,
    generate_combined_visualizations,
)
from ..core.parsed_model import (
    load_visualization_model,
    write_stale_json_note_if_needed,
)
from ..core.sampling import sample_parsed_data
from ..graph import (
    generate_network_visualizations,
    generate_variable_parameter_bipartite,
)
from ..matrix.extract import collect_visualization_matrices
from ..matrix.visualizer import MatrixVisualizer

logger = logging.getLogger(__name__)

_MATRIX_NETWORK_LIMIT = 200


def discover_visualization_files(
    target_dir: Path, recursive: bool = True
) -> List[Path]:
    """Discover visualization inputs with explicit recursive semantics."""
    if not target_dir.exists() or not target_dir.is_dir():
        return []

    matcher = target_dir.rglob if recursive else target_dir.glob
    files = [
        path
        for path in matcher("*.md")
        if path.is_file() and is_model_source_path(path)
    ]
    files.extend(path for path in matcher("*.gnn") if path.is_file())
    return sorted(set(files), key=lambda path: path.relative_to(target_dir).as_posix())


def _write_visualization_summary(
    results_dir: Path,
    gnn_files: List[Path],
    visualizations: List[str],
    warnings: List[str],
    errors: List[str],
) -> None:
    """Write the Step 8 run summary even for warning-only outcomes."""
    summary: Dict[str, Any] = {
        "processed_files": len(gnn_files),
        "total_visualizations": len(visualizations),
        "visualization_files": visualizations,
        "warnings": warnings,
        "errors": errors,
        "success": len(visualizations) > 0,
    }
    summary_file = results_dir / "visualization_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def process_visualization(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    *,
    logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> Union[bool, int]:
    """Process visualization for every GNN file in ``target_dir``.

    Returns ``True`` when at least one artifact was generated, the warning
    code ``2`` for no-input / no-artifact outcomes, and ``False`` for hard
    processing failures. Accepts an optional ``logger`` for dependency
    injection (the pipeline passes its configured step logger); when omitted
    the module-level ``"visualization"`` logger is used, preserving the
    direct-call behavior.
    """
    log = logger or logging.getLogger("visualization")

    # Optional in-memory carrier (consolidated executor): step 3's parsed
    # models forwarded by gnn.pipeline.step_executor as the ``parsed_model``
    # kwarg. Absent or malformed carriers leave the per-file disk loads
    # unchanged.
    carrier = kwargs.get("parsed_model")
    parsed_models = carrier.get("models") if isinstance(carrier, dict) else None
    if not isinstance(parsed_models, dict):
        parsed_models = None
    try:
        log_step_start(log, "Processing visualizations")

        results_dir = output_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        recursive = bool(kwargs.get("recursive", True))
        log.info("Visualization discovery recursive=%s", recursive)

        gnn_files = discover_visualization_files(target_dir, recursive=recursive)
        if not gnn_files:
            warning = (
                f"No GNN files found for visualization in {target_dir} "
                f"(recursive={recursive})"
            )
            log_step_warning(log, warning)
            _write_visualization_summary(results_dir, [], [], [warning], [])
            return 2

        all_visualizations: List[str] = []
        processing_errors: List[str] = []
        for gnn_file in gnn_files:
            try:
                all_visualizations.extend(
                    process_single_gnn_file(
                        gnn_file,
                        results_dir,
                        verbose,
                        parsed_model=(
                            parsed_models.get(gnn_file.stem) if parsed_models else None
                        ),
                    )
                )
            except Exception as e:
                message = f"Error processing {gnn_file}: {e}"
                processing_errors.append(message)
                log.warning(message)

        if len(gnn_files) > 1:
            try:
                all_visualizations.extend(
                    generate_combined_visualizations(
                        gnn_files, results_dir, verbose, parsed_models=parsed_models
                    )
                )
            except Exception as e:
                message = f"Error generating combined visualizations: {e}"
                processing_errors.append(message)
                log.warning(message)

        all_visualizations = sorted(set(all_visualizations))
        _write_visualization_summary(
            results_dir, gnn_files, all_visualizations, [], processing_errors
        )

        if all_visualizations:
            log_step_success(log, f"Generated {len(all_visualizations)} visualizations")
            if processing_errors:
                log_step_warning(
                    log,
                    f"Visualization completed with {len(processing_errors)} warning(s)",
                )
                return 2
            return True
        log_step_warning(log, "No visualizations generated")
        return 2

    except Exception as e:
        log_step_error(log, f"Visualization processing failed: {e}")
        return False


def load_cached_artifacts(model_dir: Path, source_mtime: float) -> List[str]:
    """Return cached PNG paths when fresher than ``source_mtime``, else ``[]``.

    Also removes stale cache PNGs (older than the source) so the caller can
    re-render cleanly. Non-PNG artifacts and missing directories return ``[]``.
    """
    existing = sorted(str(p) for p in model_dir.glob("*.png"))
    if not existing:
        return []
    cache_mtime = min(Path(png).stat().st_mtime for png in existing)
    if cache_mtime >= source_mtime:
        return existing
    for png_file in existing:
        try:
            Path(png_file).unlink()
        except OSError as e:
            logger.debug("Could not remove stale cache file %s: %s", png_file, e)
    return []


def render_matrix_artifacts(
    matrices: Dict[str, Any],
    model_dir: Path,
    model_name: str,
    visualizer: Any,
    verbose: bool = False,
) -> List[str]:
    """Render 2D heatmaps and 3D tensor panels for each collected matrix."""
    artifacts: List[str] = []
    for m_name, m_data in matrices.items():
        if m_data.ndim == 3:
            tensor_path = model_dir / f"{model_name}_{m_name}_tensor.png"
            if visualizer.generate_3d_tensor_visualization(
                m_name, m_data, tensor_path, tensor_type="transition"
            ):
                artifacts.append(str(tensor_path))
            html_path = model_dir / f"{model_name}_{m_name}_threejs.html"
            if visualizer.generate_threejs_tensor_explorer(m_name, m_data, html_path):
                artifacts.append(str(html_path))
            analysis_path = model_dir / f"{model_name}_{m_name}_analysis.png"
            if visualizer.generate_pomdp_transition_analysis(m_data, analysis_path):
                artifacts.append(str(analysis_path))
        else:
            heatmap_path = model_dir / f"{model_name}_{m_name}_heatmap.png"
            if visualizer.generate_matrix_heatmap(m_name, m_data, heatmap_path):
                artifacts.append(str(heatmap_path))
    if verbose and artifacts:
        logger.info(
            "Generated %s matrix visualizations for %s", len(artifacts), model_name
        )
    return artifacts


def write_viz_manifest(
    model_name: str,
    parsed_data: Dict[str, Any],
    artifacts: List[str],
    model_dir: Path,
) -> Optional[Path]:
    """Write ``{model}_viz_manifest.json``; return the path or ``None`` on failure."""
    manifest_path = model_dir / f"{model_name}_viz_manifest.json"
    try:
        manifest: Dict[str, Any] = {
            "model_name": model_name,
            "viz_meta": parsed_data.get("_viz_meta") or {},
            "artifact_count": len(artifacts),
            "artifacts": list(artifacts),
            "variable_count": len(parsed_data.get("variables") or []),
            "connection_count": len(parsed_data.get("connections") or []),
            "parameter_count": len(parsed_data.get("parameters") or []),
            "ontology_label_count": len(parsed_data.get("ontology_labels") or []),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest_path
    except (OSError, TypeError, ValueError) as e:
        logger.debug("Could not write viz manifest for %s: %s", model_name, e)
        return None


def write_sampling_note(
    model_dir: Path, model_name: str, summary: Dict[str, Any]
) -> None:
    """Write the ``{model}_sampling_note.txt`` sidecar when sampling was applied."""
    note_path = model_dir / f"{model_name}_sampling_note.txt"
    try:
        note_path.write_text(
            f"Sampling applied to {model_name}:\n"
            f"Original variables: {summary.get('original_variables', 0)}\n"
            f"Sampled variables: {summary.get('sampled_variables', 0)}\n"
            f"Original connections: {summary.get('original_connections', 0)}\n"
            f"Sampled connections: {summary.get('sampled_connections', 0)}\n",
            encoding="utf-8",
        )
    except OSError as e:
        logger.debug("Could not write sampling note for %s: %s", model_name, e)


def process_single_gnn_file(
    gnn_file: Path,
    results_dir: Path,
    verbose: bool = False,
    *,
    parsed_model: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Process a single GNN file into per-model PNG/JSON/HTML artifacts.

    ``parsed_model`` (consolidated executor carrier) supplies the step-3
    parsed payload in memory instead of re-reading ``{model}_parsed.json``.
    """
    with open(gnn_file, encoding="utf-8") as f:
        content = f.read()

    model_name = gnn_file.stem
    model_dir = results_dir / model_name
    model_dir.mkdir(exist_ok=True)

    cached = load_cached_artifacts(model_dir, gnn_file.stat().st_mtime)
    if cached:
        if verbose:
            print(f"Using cached visualizations for {model_name}")
        return cached

    parsed_data = load_visualization_model(
        gnn_file, content, results_dir, verbose, parsed_model=parsed_model
    )
    write_stale_json_note_if_needed(parsed_data, model_dir, model_name, gnn_file)
    sampled = sample_parsed_data(parsed_data)
    if sampled and verbose:
        print(f"Large dataset detected for {model_name}, applying sampling")

    visualizations: List[str] = []

    if len(parsed_data.get("variables") or []) <= _MATRIX_NETWORK_LIMIT:
        try:
            visualizations.extend(
                generate_network_visualizations(parsed_data, model_dir, model_name)
            )
        except Exception as e:
            if verbose:
                print(f"Network visualization failed for {model_name}: {e}")
    elif verbose:
        print(f"Skipping network visualizations for {model_name} - too many nodes")

    try:
        visualizations.extend(
            generate_variable_parameter_bipartite(parsed_data, model_dir, model_name)
        )
    except Exception as e:
        if verbose:
            logger.debug("Bipartite visualization skipped: %s", e)

    try:
        mv = MatrixVisualizer()
        matrices = collect_visualization_matrices(parsed_data)
        if matrices:
            visualizations.extend(
                render_matrix_artifacts(matrices, model_dir, model_name, mv, verbose)
            )
        elif verbose:
            logger.warning(
                "No matrix data found for %s - checked parameters, variables, matrices",
                model_name,
            )
    except Exception as e:
        if verbose:
            logger.exception("Matrix visualization failed for %s: %s", model_name, e)

    try:
        visualizations.extend(
            generate_combined_analysis(parsed_data, model_dir, model_name)
        )
    except Exception as e:
        if verbose:
            print(f"Combined analysis failed for {model_name}: {e}")

    if sampled and visualizations:
        write_sampling_note(
            model_dir, model_name, parsed_data.get("_sampling_applied") or {}
        )

    manifest_path = write_viz_manifest(
        model_name, parsed_data, visualizations, model_dir
    )
    if manifest_path is not None:
        visualizations.append(str(manifest_path))

    return visualizations
