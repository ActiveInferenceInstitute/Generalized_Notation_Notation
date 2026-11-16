#!/usr/bin/env python3
"""
GNN multi-format processing module.

Implements comprehensive discovery, parsing, and multi-format serialization
for GNN specifications. This keeps the numbered step script thin.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast

from gnn.parsers.common import get_extension_for_format, get_supported_gnn_extensions
from gnn.processing.discovery import model_artifact_stems
from gnn.utils import log_step_error, log_step_start, log_step_success


def _formats_for_serialize_preset(
    preset: str, supported_formats: Sequence[Any]
) -> Tuple[List[Any], str]:
    """Return filtered formats list and normalized preset label."""
    label = (preset or "full").strip().lower()
    if label not in {"full", "minimal"}:
        label = "full"
    if label == "minimal":
        minimal_allow: set[Any] = {"markdown", "json", "python"}
        filtered = [
            f
            for f in supported_formats
            if getattr(f, "value", str(f)).lower() in minimal_allow
        ]
        return filtered, label
    return list(supported_formats), label


_PARSER_WORKER_SYSTEM: Any = None


def _parse_file_job(
    job: tuple[Path, list[Any], str | None, bool], *, system: Any = None
) -> dict[str, Any]:
    """Parse and serialize one input without writing pipeline artifacts."""
    from gnn.parsers import GNNParsingSystem

    path, formats, expected_hash, bind_standalone_source = job
    record: dict[str, Any] = {"file_name": path.name, "file_path": str(path)}
    try:
        if bind_standalone_source:
            expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        elif (
            expected_hash
            and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash
        ):
            raise ValueError("Model source changed after run planning")
        if system is None:
            global _PARSER_WORKER_SYSTEM
            if _PARSER_WORKER_SYSTEM is None:
                _PARSER_WORKER_SYSTEM = GNNParsingSystem(strict_validation=True)
            system = _PARSER_WORKER_SYSTEM
        parsed = system.parse_file(path)
        record.update(
            parse_success=parsed.success, errors=parsed.errors, warnings=parsed.warnings
        )
        if not parsed.success:
            return record
        model = parsed.model
        record.update(
            source_format=model.source_format.value if model.source_format else None,
            model_info={
                "name": model.model_name,
                "version": model.version,
                "variables_count": len(model.variables),
                "connections_count": len(model.connections),
                "parameters_count": len(model.parameters),
                "equations_count": len(model.equations),
            },
            generated_formats={},
            _parsed_payload=model.to_dict(),
            _serialized={},
        )
        for format_enum in formats:
            try:
                record["_serialized"][format_enum.value] = system.serialize(
                    model, format_enum
                )
                record["generated_formats"][format_enum.value] = {"success": True}
            except Exception as exc:
                record["generated_formats"][format_enum.value] = {
                    "success": False,
                    "error": str(exc),
                }
        if (
            expected_hash
            and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash
        ):
            raise ValueError("Model source changed during parsing")
        if bind_standalone_source:
            record["source_sha256"] = expected_hash
        return record
    except Exception as exc:
        return {
            "file_name": path.name,
            "file_path": str(path),
            "parse_success": False,
            "error": str(exc),
        }


def _publish_parse_result(
    record: dict[str, Any],
    output: Path,
    artifact_stem: str,
    formats: list[Any],
    provenance: dict[str, str],
    deadline: float | None,
    cancel_token: Any,
) -> int:
    """Publish one computed result in the parent, retaining per-format failures."""
    from gnn.utils.file_workers import check_file_work

    destination = output / artifact_stem
    check_file_work(deadline, cancel_token)
    destination.mkdir(exist_ok=True)
    parsed_file = destination / f"{artifact_stem}_parsed.json"
    payload = record.pop("_parsed_payload")
    if provenance:
        payload["_gnn_provenance"] = dict(provenance)
    parsed_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    record["parsed_model_file"] = str(parsed_file)
    serialized = record.pop("_serialized")
    generated = 0
    for format_enum in formats:
        check_file_work(deadline, cancel_token)
        out_file = (
            destination
            / f"{artifact_stem}_{format_enum.value}{get_extension_for_format(format_enum)}"
        )
        info = record["generated_formats"][format_enum.value]
        info["output_file"] = str(out_file)
        if info["success"]:
            try:
                out_file.write_text(serialized[format_enum.value], encoding="utf-8")
                info["file_size"] = out_file.stat().st_size
                generated += 1
            except TimeoutError:
                raise
            except (OSError, TypeError, ValueError) as exc:
                info.update(success=False, error=str(exc))
    return generated


def process_gnn_multi_format(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = True,
    verbose: bool = False,
    cancel_token: Any = None,
    **kwargs: Any,
) -> bool:
    """Discover, parse, and serialize GNN models to all supported formats.

    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Base output directory (step-specific directory will be created)
        logger: Logger instance
        recursive: Whether to recurse into subdirectories
        verbose: Enable verbose logs

    Returns:
        True on success, False otherwise
    """
    # Resolve step-specific output directory (lazy import avoids gnn → pipeline coupling)
    from gnn.pipeline.config import get_output_dir_for_script

    step_output_dir = get_output_dir_for_script("3_gnn.py", output_dir)
    step_output_dir.mkdir(parents=True, exist_ok=True)

    log_step_start(logger, "Processing GNN files with full multi-format generation")

    # Import the comprehensive GNN parsing system
    try:
        from gnn.parsers import GNNParsingSystem
    except Exception as e:  # pragma: no cover - defensive
        log_step_error(logger, f"Failed to import GNN parsing system: {e}")
        return False

    try:
        parsing_system = GNNParsingSystem(strict_validation=True)
        supported_formats = parsing_system.get_supported_formats()
        preset_raw = str(kwargs.get("serialize_preset") or "full")
        raw_norm = preset_raw.strip().lower()
        formats_to_emit, preset_norm = _formats_for_serialize_preset(
            preset_raw, supported_formats
        )
        if raw_norm not in {"full", "minimal"}:
            logger.warning(
                "Unknown serialize_preset %r; using full-format serialization",
                preset_raw,
            )
        available_serializers = parsing_system.get_available_serializers()

        logger.info(
            f"Initialized GNN parsing system with {len(supported_formats)} supported formats "
            f"({preset_norm} preset emits {len(formats_to_emit)} serializers)"
        )
        logger.info(f"Available serializers: {list(available_serializers.keys())}")

        # A run projects its frozen model selection; standalone calls discover normally.
        from gnn.pipeline.run_context import (
            current_run_context,
            selected_model_sources,
        )
        from gnn.utils.file_workers import (
            check_file_work,
            ordered_file_map,
            resolve_file_workers,
        )

        run_context = current_run_context()
        requested_workers, executor_mode = resolve_file_workers(
            kwargs.get("file_workers"), "process"
        )
        deadline = run_context.deadline_monotonic if run_context else None
        check_file_work(deadline, cancel_token)
        target_path = Path(target_dir)
        gnn_files = selected_model_sources(
            target_path,
            3,
            recursive=recursive,
            extensions=get_supported_gnn_extensions(),
        )

        logger.info(f"Found {len(gnn_files)} potential GNN files")

        if not gnn_files:
            logger.warning("No GNN files found for multi-format processing")
            return True

        processing_results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "source_directory": str(target_path),
            "output_directory": str(step_output_dir),
            "processing_config": {
                "recursive": recursive,
                "verbose": verbose,
                "serialize_preset": preset_norm,
                "supported_formats": [fmt.value for fmt in supported_formats],
                "serialized_formats": [fmt.value for fmt in formats_to_emit],
                "available_serializers": {
                    fmt.value: name for fmt, name in available_serializers.items()
                },
            },
            "processed_files": [],
            "summary": {
                "total_files": 0,
                "successful_parses": 0,
                "failed_parses": 0,
                "total_formats_generated": 0,
                "formats_per_file": {},
            },
        }
        if run_context is not None:
            processing_results["run_id"] = run_context.run_id
            processing_results["model_selection"] = [
                m.__dict__ for m in run_context.selected_models(3)
            ]

        provenance_by_stem = (
            {
                model.artifact_stem: {
                    "model_id": model.model_id,
                    "source_path": model.source_path,
                    "source_relative_path": model.relative_path,
                    "source_sha256": model.sha256,
                }
                for model in run_context.selected_models(3)
            }
            if run_context
            else {}
        )
        standalone_stems = (
            model_artifact_stems(target_path, gnn_files) if run_context is None else {}
        )
        standalone_collisions = {
            path for path, stem in standalone_stems.items() if stem != path.stem
        }
        worker_metadata: dict[str, Any] = {
            "requested_workers": requested_workers,
            "effective_workers": min(requested_workers, len(gnn_files)),
        }
        processing_results["worker_execution"] = worker_metadata
        jobs = (
            (
                path,
                formats_to_emit,
                provenance_by_stem.get(path.stem, {}).get("source_sha256"),
                path in standalone_collisions,
            )
            for path in gnn_files
        )
        compute = (
            _parse_file_job
            if worker_metadata["effective_workers"] > 1
            else partial(_parse_file_job, system=parsing_system)
        )
        for file_result in ordered_file_map(
            compute,
            jobs,
            workers=min(requested_workers, len(gnn_files)),
            executor=executor_mode,
            deadline=deadline,
            cancel_token=cancel_token,
            metadata=worker_metadata,
        ):
            check_file_work(deadline, cancel_token)
            file_path = Path(file_result["file_path"])
            artifact_stem = standalone_stems.get(file_path, file_path.stem)
            provenance = provenance_by_stem.get(file_path.stem, {})
            if file_path in standalone_collisions and file_result["parse_success"]:
                provenance = {
                    "source_path": str(file_path.absolute()),
                    "source_relative_path": file_path.relative_to(
                        target_path
                    ).as_posix(),
                    "source_sha256": file_result["source_sha256"],
                }
            if file_result["parse_success"]:
                try:
                    formats_generated = _publish_parse_result(
                        file_result,
                        step_output_dir,
                        artifact_stem,
                        formats_to_emit,
                        provenance,
                        deadline,
                        cancel_token,
                    )
                except TimeoutError:
                    raise
                except (OSError, TypeError, ValueError) as exc:
                    file_result = {
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "parse_success": False,
                        "error": str(exc),
                    }
                else:
                    processing_results["summary"]["formats_per_file"][
                        f"{artifact_stem}{file_path.suffix}"
                    ] = formats_generated
                    processing_results["summary"]["total_formats_generated"] += (
                        formats_generated
                    )
                    processing_results["summary"]["successful_parses"] += 1
                    logger.info(
                        "Generated %s formats for %s", formats_generated, file_path.name
                    )
            if not file_result["parse_success"]:
                processing_results["summary"]["failed_parses"] += 1
                logger.warning(
                    "Failed to process %s: %s",
                    file_path,
                    file_result.get("errors", file_result.get("error")),
                )
            if artifact_stem != file_path.stem:
                file_result["artifact_stem"] = artifact_stem
            file_result.update(provenance)
            processing_results["processed_files"].append(file_result)

        check_file_work(deadline, cancel_token)
        processing_results["summary"]["total_files"] = len(gnn_files)
        check_file_work(deadline, cancel_token)
        # Save artifacts
        results_file = step_output_dir / "gnn_processing_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(processing_results, f, indent=2, default=str)

        summary_file = step_output_dir / "gnn_processing_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(processing_results["summary"], f, indent=2)

        # Aggregate format statistics
        format_stats: Dict[str, Dict[str, int]] = {}
        for pf in processing_results["processed_files"]:
            if pf.get("parse_success"):
                for fmt, info in pf["generated_formats"].items():
                    if fmt not in format_stats:
                        format_stats[fmt] = {
                            "successful": 0,
                            "failed": 0,
                            "total_size": 0,
                        }
                    if info.get("success"):
                        format_stats[fmt]["successful"] += 1
                        format_stats[fmt]["total_size"] += int(info.get("file_size", 0))
                    else:
                        format_stats[fmt]["failed"] += 1

        stats_file = step_output_dir / "format_statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(format_stats, f, indent=2)

        success = processing_results["summary"]["successful_parses"] > 0
        if run_context is not None:
            success = processing_results["summary"]["successful_parses"] == len(
                run_context.selected_models(3)
            )
        if success:
            total_formats = processing_results["summary"]["total_formats_generated"]
            log_step_success(
                logger,
                f"Processed {processing_results['summary']['successful_parses']} files, "
                f"generated {total_formats} format instances across {len(formats_to_emit)} formats "
                f"(preset={preset_norm})",
            )
        else:
            log_step_error(
                logger,
                f"Selected parse coverage incomplete: {processing_results['summary']['successful_parses']} successful, {processing_results['summary']['failed_parses']} failed",
            )

        return cast("bool", success)

    except Exception as e:  # pragma: no cover - outer guard
        log_step_error(logger, f"GNN processing failed: {e}")
        return False
