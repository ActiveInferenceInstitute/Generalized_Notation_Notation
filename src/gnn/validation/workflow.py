"""Directory-level validation workflow behind ``validation.process_validation``.

Loads the parsed-model manifest emitted by step 3
(``gnn_processing_results.json``), runs the three best-effort validation
stages (semantic, performance, consistency) on every successfully parsed
file, retains distinct results within a run/configuration, and persists
``validation_results.json`` / ``validation_summary.json`` receipts for
downstream steps and MCP tooling.

Stage callables are injected by the caller (``validation.process_validation``
binds them from its own module globals at call time, which keeps the
monkeypatch seam used by the test suite working), so this module stays free
of imports from the validator implementations.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

StageFn = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class StageServices:
    """Injected callables for the best-effort validation stages.

    ``validation.process_validation`` binds these from its own module globals
    at call time, so tests (or alternative pipelines) can supply patched or
    replacement stage functions without touching this module. ``orientation``
    defaults to ``None`` — set it to run the B-tensor orientation diagnostic
    as a fourth stage.
    """

    semantic: StageFn
    performance: StageFn
    consistency: StageFn
    orientation: StageFn | None = None


@dataclass(frozen=True)
class _StageSpec:
    """Static metadata for one validation stage."""

    key: str  # key under file_result["validations"] and StageServices
    label: str  # log-message label, e.g. "Semantic validation"
    noun: str  # recovery-mode fallback noun, e.g. "semantic validation"
    score_key: str  # per-stage score key, e.g. "semantic_score"
    wants_validation_level: bool = False
    wants_transpose_b: bool = False


_STAGE_SPECS: tuple[_StageSpec, ...] = (
    _StageSpec(
        key="semantic",
        label="Semantic validation",
        noun="semantic validation",
        score_key="semantic_score",
        wants_validation_level=True,
    ),
    _StageSpec(
        key="performance",
        label="Performance profiling",
        noun="performance profiling",
        score_key="performance_score",
    ),
    _StageSpec(
        key="consistency",
        label="Consistency checking",
        noun="consistency checking",
        score_key="consistency_score",
    ),
    _StageSpec(
        key="orientation",
        label="B-orientation check",
        noun="B-orientation check",
        score_key="orientation_score",
        wants_transpose_b=True,
    ),
)


def _validation_stage_failed(result: dict[str, Any]) -> bool:
    """Return whether a stage reported operational failure or semantic invalidity."""
    return (
        result.get("status") == "error"
        or result.get("recovery") is True
        or result.get("valid") is False
        or result.get("is_valid") is False
    )


def _locate_gnn_results(output_dir: Path, log: logging.Logger) -> Path | None:
    """Resolve the step-3 manifest path, or log actionable errors and return None."""
    from gnn.pipeline.config import get_output_dir_for_script

    # Look in the base output directory, not the step-specific directory
    base_output_dir = (
        output_dir.parent
        if output_dir.name.startswith(("6_validation", "7_export", "8_visualization"))
        else output_dir
    )
    gnn_output_dir = get_output_dir_for_script("3_gnn.py", base_output_dir)
    gnn_results_file = gnn_output_dir / "gnn_processing_results.json"

    if not gnn_results_file.exists():
        log.error(
            f"GNN processing results not found at {gnn_results_file}. Run step 3 first."
        )
        log.error(f"Expected file location: {gnn_results_file}")
        log.error(f"GNN output directory: {gnn_output_dir}")
        log.error(f"GNN output directory exists: {gnn_output_dir.exists()}")
        if gnn_output_dir.exists():
            log.error(f"Contents: {list(gnn_output_dir.iterdir())}")
        return None
    return gnn_results_file


def _load_or_init_results(
    output_dir: Path,
    target_dir: Path,
    log: logging.Logger,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Retain receipts only within the same run and validation configuration."""
    previous: dict[str, Any] = {}
    path = output_dir / "validation_results.json"
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("existing validation results must be a JSON object")
            if loaded.get("context") == context:
                files = loaded.get("files_validated", [])
                sources = loaded.get("source_directories", [])
                if not isinstance(files, list) or not all(
                    isinstance(f, dict) for f in files
                ):
                    raise ValueError("invalid saved file receipts")
                if not isinstance(sources, list) or not all(
                    isinstance(s, str) for s in sources
                ):
                    raise ValueError("invalid saved source directories")
                previous = loaded
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            log.warning(
                "Could not load existing validation results, starting fresh: %s", exc
            )
    sources = previous.get("source_directories", [])
    if str(target_dir) not in sources:
        sources.append(str(target_dir))
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "source_directory": str(target_dir),
        "source_directories": sources,
        "output_directory": str(output_dir),
        "context": context,
        "files_validated": previous.get("files_validated", []),
    }


def _receipt_identity(
    file_result: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Identify the source, current input bytes, parser outcome, and run/config."""
    identity: dict[str, Any] = {
        "source_path": str(Path(file_result["file_path"]).resolve()),
        "parse_success": bool(file_result.get("parse_success")),
        **context,
    }
    for key in ("file_path", "parsed_model_file"):
        path = file_result.get(key)
        try:
            identity[key + "_sha256"] = (
                hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
            )
        except OSError:
            identity[key + "_sha256"] = None
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return {
        "receipt_key": hashlib.sha256(encoded.encode()).hexdigest(),
        "input_identity": identity,
    }


def _summarize(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute totals and scores from distinct receipts, never old counters."""
    scores: dict[str, list[Any]] = {spec.key: [] for spec in _STAGE_SPECS}
    for result in files:
        for spec in _STAGE_SPECS:
            stage = result.get("validations", {}).get(spec.key, {})
            if spec.score_key in stage:
                scores[spec.key].append(stage[spec.score_key])
    successful = sum(result["success"] is True for result in files)
    summary = {
        "total_files": len(files),
        "successful_validations": successful,
        "failed_validations": len(files) - successful,
        "validation_scores": scores,
    }
    _record_average_scores(summary)
    return summary


def _run_stage(
    spec: _StageSpec,
    stage_fn: StageFn,
    model_data: dict[str, Any],
    file_validation_result: dict[str, Any],
    summary_scores: dict[str, list[Any]],
    validation_level: str,
    transpose_b: bool,
    log: logging.Logger,
) -> None:
    """Run one best-effort stage, recording receipts, errors, and scores."""
    file_name = str(file_validation_result["file_name"])
    stage_kwargs: dict[str, Any] = {}
    if spec.wants_validation_level:
        stage_kwargs["validation_level"] = validation_level
    if spec.wants_transpose_b:
        stage_kwargs["transpose_b"] = transpose_b
    try:
        result = stage_fn(model_data, **stage_kwargs)
        file_validation_result["validations"][spec.key] = result
        summary_scores[spec.key].append(result.get(spec.score_key, 0.0))
        if _validation_stage_failed(result):
            error = str(
                result.get("error")
                or "; ".join(map(str, result.get("errors", [])))
                or f"{spec.noun} reported failure"
            )
            file_validation_result["errors"].append(error)
            file_validation_result["success"] = False
            log.error(f"{spec.label} failed for {file_name}: {error}")
        else:
            log.info(f"{spec.label} completed for {file_name}")
    except Exception as e:
        log.error(f"{spec.label} failed for {file_name}: {e}")
        file_validation_result["validations"][spec.key] = {
            "status": "error",
            "error": str(e),
            "recovery": True,
        }
        file_validation_result["errors"].append(str(e))
        file_validation_result["success"] = False


def _validate_file(
    file_result: dict[str, Any],
    services: StageServices,
    validation_level: str,
    summary_scores: dict[str, list[Any]],
    transpose_b: bool,
    log: logging.Logger,
) -> dict[str, Any]:
    """Run every validation stage on one parsed-model manifest entry."""
    file_name = str(file_result["file_name"])
    log.info(f"Validating: {file_name}")

    # Load the actual parsed GNN specification
    model_load_error: str | None = None
    parsed_model_file = file_result.get("parsed_model_file")
    if parsed_model_file and Path(parsed_model_file).exists():
        try:
            with open(parsed_model_file, "r") as f:
                actual_gnn_spec = json.load(f)
            log.info(f"Loaded parsed GNN specification from {parsed_model_file}")
            model_data: dict[str, Any] = actual_gnn_spec
        except Exception as e:
            model_load_error = (
                f"Failed to load parsed GNN spec from {parsed_model_file}: {e}"
            )
            log.error(model_load_error)
            model_data = file_result
    else:
        model_load_error = (
            f"Parsed model file not found for {file_name}; using summary data"
        )
        log.warning(model_load_error)
        model_data = file_result

    file_validation_result: dict[str, Any] = {
        "file_name": file_name,
        "file_path": file_result["file_path"],
        "validations": {},
        "success": model_load_error is None,
        "errors": [model_load_error] if model_load_error else [],
    }
    if model_load_error:
        file_validation_result["input_recovery"] = {
            "status": "error",
            "error": model_load_error,
            "recovery": True,
        }

    for spec in _STAGE_SPECS:
        stage_fn = getattr(services, spec.key, None)
        if stage_fn is None:
            continue
        _run_stage(
            spec,
            stage_fn,
            model_data,
            file_validation_result,
            summary_scores,
            validation_level,
            transpose_b,
            log,
        )

    return file_validation_result


def _record_average_scores(summary: dict[str, Any]) -> None:
    """Store per-stage average scores when at least one score was recorded."""
    for spec in _STAGE_SPECS:
        scores = summary["validation_scores"][spec.key]
        if scores:
            avg_score = sum(scores) / len(scores)
            summary["validation_scores"][f"avg_{spec.key}_score"] = avg_score


def _write_receipts(output_dir: Path, validation_results: dict[str, Any]) -> None:
    """Persist the full results receipt and the summary-only receipt."""
    for name, payload in (
        ("validation_results.json", validation_results),
        ("validation_summary.json", validation_results["summary"]),
    ):
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output_dir, delete=False
            ) as handle:
                temporary = handle.name
                json.dump(payload, handle, indent=2)
            os.replace(temporary, output_dir / name)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)


def _log_summary(summary: dict[str, Any], log: logging.Logger) -> None:
    """Emit the end-of-run count and average-score log block."""
    log.info("Validation processing completed:")
    log.info(f"  Total files: {summary['total_files']}")
    log.info(f"  Successful validations: {summary['successful_validations']}")
    log.info(f"  Failed validations: {summary['failed_validations']}")

    validation_scores = summary["validation_scores"]
    if validation_scores["semantic"]:
        log.info(
            f"  Average semantic score: {validation_scores['avg_semantic_score']:.2f}"
        )

    if validation_scores["performance"]:
        log.info(
            f"  Average performance score: {validation_scores['avg_performance_score']:.2f}"
        )

    if validation_scores["consistency"]:
        log.info(
            f"  Average consistency score: {validation_scores['avg_consistency_score']:.2f}"
        )


def validate_directory(
    target_dir: Path,
    output_dir: Path,
    services: StageServices,
    verbose: bool = False,
    validation_level: str = "standard",
    log: logging.Logger | None = None,
    run_id: str | None = None,
    transpose_b: bool = False,
) -> bool:
    """Validate every parsed GNN file listed in the step-3 manifest.

    Args:
        target_dir: Source directory recorded on the receipt (informational).
        output_dir: Directory that receives the validation receipts.
        services: Injected stage callables (semantic/performance/consistency,
            plus the optional ``orientation`` B-tensor diagnostic stage).
        verbose: Enable DEBUG-level logging on the resolved logger.
        validation_level: Semantic validation depth forwarded to the semantic
            stage ("basic", "standard", "strict", "research").
        log: Logger to use; defaults to this module's logger.
        run_id: Optional stable run identity for repeated subdirectory passes.
            Defaults to the manifest run_id, then its timestamp. Manifests
            with neither are unbound records scoped to the output directory.
        transpose_b: Opt-in canonical B-tensor transposition for the
            orientation stage: textbook (row-stochastic) tensors are
            transposed in memory and the per-tensor transposition is
            recorded in the receipt. Source files are never modified.

    Returns:
        True only for a nonempty current pass with every file successful.
        Historical successes never mask a current failure or empty pass.
    """
    active_logger = log if log is not None else logger
    if verbose:
        active_logger.setLevel(logging.DEBUG)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        gnn_results_file = _locate_gnn_results(output_dir, active_logger)
        if gnn_results_file is None:
            return False

        with open(gnn_results_file, "r") as f:
            gnn_results = json.load(f)
        from gnn.pipeline.parse_receipt import selected_parse_receipt

        gnn_results = selected_parse_receipt(gnn_results, 6)

        active_logger.info(
            f"Loaded {len(gnn_results['processed_files'])} parsed GNN files"
        )

        context = {
            "receipt_version": 1,
            "run_id": run_id
            if run_id is not None
            else gnn_results.get("run_id", gnn_results.get("timestamp")),
            "validation_level": validation_level,
            "transpose_b": transpose_b,
        }
        validation_results = _load_or_init_results(
            output_dir, Path(target_dir), active_logger, context
        )
        retained = {
            str(Path(item["file_path"]).resolve()): item
            for item in validation_results["files_validated"]
        }
        current: dict[str, dict[str, Any]] = {}
        for file_result in gnn_results["processed_files"]:
            identity = _receipt_identity(file_result, context)
            if file_result.get("parse_success"):
                file_validation_result = _validate_file(
                    file_result,
                    services,
                    validation_level,
                    {spec.key: [] for spec in _STAGE_SPECS},
                    transpose_b,
                    active_logger,
                )
            else:
                file_validation_result = {
                    "file_name": file_result["file_name"],
                    "file_path": file_result["file_path"],
                    "validations": {},
                    "success": False,
                    "errors": ["Current input did not parse successfully"],
                }
            file_validation_result.update(identity)
            source = identity["input_identity"]["source_path"]
            current[source] = file_validation_result
            retained[source] = file_validation_result

        validation_results["files_validated"] = list(retained.values())
        validation_results["summary"] = _summarize(list(retained.values()))
        current_summary = _summarize(list(current.values()))
        validation_results["current_summary"] = current_summary
        _write_receipts(output_dir, validation_results)
        _log_summary(current_summary, active_logger)
        return bool(current and current_summary["failed_validations"] == 0)

    except Exception as e:
        active_logger.error(f"Validation processing failed: {e}")
        return False
