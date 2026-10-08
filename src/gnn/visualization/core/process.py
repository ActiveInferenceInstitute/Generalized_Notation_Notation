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
from ..matrix.extract import collect_visualization_matrices, is_transition_tensor_name
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

    Returns ``True`` when every figure was generated, the warning code ``2``
    (success with warnings) for no-input / no-artifact outcomes and for runs
    where any figure failed while the step still completed, and ``False`` for
    hard processing failures. Accepts an optional ``logger`` for dependency
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
            figure_failures: List[str] = []
            try:
                all_visualizations.extend(
                    process_single_gnn_file(
                        gnn_file,
                        results_dir,
                        verbose,
                        parsed_model=(
                            parsed_models.get(gnn_file.stem) if parsed_models else None
                        ),
                        failures=figure_failures,
                    )
                )
            except Exception as e:
                figure_failures.append(f"Error processing {gnn_file}: {e}")
            for message in figure_failures:
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
    *,
    failures: Optional[List[str]] = None,
) -> List[str]:
    """Render 2D heatmaps and 3D tensor panels for each collected matrix.

    Every generator that reports failure (returns ``False``) is recorded in
    ``failures`` when given, so the caller can surface it in the step exit
    code instead of dropping the missing figure silently. Tensors of rank > 3
    have no renderer; they are logged as unsupported, not attempted.
    """
    artifacts: List[str] = []

    def _record(ok: bool, path: Path, what: str, m_name: str) -> None:
        if ok:
            artifacts.append(str(path))
        elif failures is not None:
            failures.append(f"{model_name}: {what} for {m_name} failed ({path.name})")

    for m_name, m_data in matrices.items():
        if m_data.ndim > 3:
            logger.info(
                "Skipping matrix figures for %s.%s: rank-%d tensor %s is "
                "unsupported (heatmaps take rank <= 2, tensor panels rank 3)",
                model_name,
                m_name,
                m_data.ndim,
                m_data.shape,
            )
        elif m_data.ndim == 3:
            tensor_type = (
                "transition" if is_transition_tensor_name(m_name) else "generic"
            )
            tensor_path = model_dir / f"{model_name}_{m_name}_tensor.png"
            _record(
                visualizer.generate_3d_tensor_visualization(
                    m_name, m_data, tensor_path, tensor_type=tensor_type
                ),
                tensor_path,
                "3D tensor figure",
                m_name,
            )
            html_path = model_dir / f"{model_name}_{m_name}_threejs.html"
            _record(
                visualizer.generate_threejs_tensor_explorer(m_name, m_data, html_path),
                html_path,
                "Three.js tensor explorer",
                m_name,
            )
            if tensor_type == "transition":
                analysis_path = model_dir / f"{model_name}_{m_name}_analysis.png"
                _record(
                    visualizer.generate_pomdp_transition_analysis(
                        m_data, analysis_path
                    ),
                    analysis_path,
                    "POMDP transition analysis",
                    m_name,
                )
        else:
            heatmap_path = model_dir / f"{model_name}_{m_name}_heatmap.png"
            _record(
                visualizer.generate_matrix_heatmap(m_name, m_data, heatmap_path),
                heatmap_path,
                "matrix heatmap",
                m_name,
            )
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
    figure_failures: Optional[List[str]] = None,
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
            "figure_failures": list(figure_failures or []),
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


def _cached_figure_failures(model_dir: Path, model_name: str) -> List[str]:
    """Return the figure failures recorded by the run that produced the cache."""
    manifest_path = model_dir / f"{model_name}_viz_manifest.json"
    try:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    failures = recorded.get("figure_failures") if isinstance(recorded, dict) else None
    return [str(f) for f in failures] if isinstance(failures, list) else []


def process_single_gnn_file(
    gnn_file: Path,
    results_dir: Path,
    verbose: bool = False,
    *,
    parsed_model: Optional[Dict[str, Any]] = None,
    failures: Optional[List[str]] = None,
) -> List[str]:
    """Process a single GNN file into per-model PNG/JSON/HTML artifacts.

    ``parsed_model`` (consolidated executor carrier) supplies the step-3
    parsed payload in memory instead of re-reading ``{model}_parsed.json``.
    ``failures``, when given, receives one message per figure that could not
    be produced, so the step can report them in its exit code. A cache whose
    manifest records figure failures is not reused; the figures are retried.
    """
    figure_failures: List[str] = []
    with open(gnn_file, encoding="utf-8") as f:
        content = f.read()

    model_name = gnn_file.stem
    model_dir = results_dir / model_name
    model_dir.mkdir(exist_ok=True)

    cached = load_cached_artifacts(model_dir, gnn_file.stat().st_mtime)
    if cached and _cached_figure_failures(model_dir, model_name):
        # The run that produced this cache lost figures: retry them rather
        # than letting a partial cache pass as a clean result.
        logger.info("Re-rendering %s: cached run recorded figure failures", model_name)
        cached = []
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
            figure_failures.append(f"{model_name}: network visualization failed: {e}")
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
                render_matrix_artifacts(
                    matrices,
                    model_dir,
                    model_name,
                    mv,
                    verbose,
                    failures=figure_failures,
                )
            )
        elif verbose:
            logger.warning(
                "No matrix data found for %s - checked parameters, variables, matrices",
                model_name,
            )
    except Exception as e:
        logger.exception("Matrix visualization failed for %s", model_name)
        figure_failures.append(f"{model_name}: matrix visualization failed: {e}")

    try:
        visualizations.extend(
            generate_combined_analysis(parsed_data, model_dir, model_name)
        )
    except Exception as e:
        figure_failures.append(f"{model_name}: combined analysis failed: {e}")

    if sampled and visualizations:
        write_sampling_note(
            model_dir, model_name, parsed_data.get("_sampling_applied") or {}
        )

    manifest_path = write_viz_manifest(
        model_name, parsed_data, visualizations, model_dir, figure_failures
    )
    if manifest_path is not None:
        visualizations.append(str(manifest_path))

    if failures is not None:
        failures.extend(figure_failures)
    return visualizations
