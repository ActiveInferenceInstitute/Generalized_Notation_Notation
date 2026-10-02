#!/usr/bin/env python3
"""
GNN multi-format processing module.

Implements comprehensive discovery, parsing, and multi-format serialization
for GNN specifications. This keeps the numbered step script thin.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast

from gnn.parsers.common import get_extension_for_format, get_supported_gnn_extensions
from gnn.processing.discovery import is_model_source_path
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


def process_gnn_multi_format(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = True,
    verbose: bool = False,
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

        # Discover GNN files
        target_path = Path(target_dir)
        gnn_files: List[Path] = []
        extensions = get_supported_gnn_extensions()
        if recursive:
            for ext in extensions:
                gnn_files.extend(target_path.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                gnn_files.extend(target_path.glob(f"*{ext}"))
        gnn_files = [path for path in gnn_files if is_model_source_path(path)]

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
        from gnn.pipeline.run_context import current_run_context

        run_context = current_run_context()
        if run_context is not None:
            processing_results["run_id"] = run_context.run_id
            processing_results["model_selection"] = [
                m.__dict__ for m in run_context.selected_models(3)
            ]

        for file_path in gnn_files:
            try:
                logger.info(f"Processing: {file_path}")
                parse_result = parsing_system.parse_file(file_path)

                if not parse_result.success:
                    logger.warning(
                        f"Failed to parse {file_path}: {parse_result.errors}"
                    )
                    processing_results["processed_files"].append(
                        {
                            "file_name": file_path.name,
                            "file_path": str(file_path),
                            "parse_success": False,
                            "errors": parse_result.errors,
                            "warnings": parse_result.warnings,
                        }
                    )
                    processing_results["summary"]["failed_parses"] += 1
                    continue

                # File-specific output directory
                file_output_dir = step_output_dir / file_path.stem
                file_output_dir.mkdir(exist_ok=True)

                # Save parsed model JSON
                parsed_model_file = file_output_dir / f"{file_path.stem}_parsed.json"
                parsed_payload = parse_result.model.to_dict()
                if run_context is not None:
                    from gnn.pipeline.run_context import model_provenance

                    parsed_payload["_gnn_provenance"] = model_provenance(file_path, 3)
                with open(parsed_model_file, "w", encoding="utf-8") as f:
                    json.dump(parsed_payload, f, indent=2, default=str)

                file_result: Dict[str, Any] = {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "parse_success": True,
                    "parsed_model_file": str(parsed_model_file),
                    "source_format": (
                        parse_result.model.source_format.value
                        if parse_result.model.source_format
                        else None
                    ),
                    "model_info": {
                        "name": parse_result.model.model_name,
                        "version": parse_result.model.version,
                        "variables_count": len(parse_result.model.variables),
                        "connections_count": len(parse_result.model.connections),
                        "parameters_count": len(parse_result.model.parameters),
                        "equations_count": len(parse_result.model.equations),
                    },
                    "generated_formats": {},
                    "errors": parse_result.errors,
                    "warnings": parse_result.warnings,
                }
                if run_context is not None:
                    reference = next(
                        m
                        for m in run_context.selected_models(3)
                        if m.artifact_stem == file_path.stem
                    )
                    file_result.update(
                        model_id=reference.model_id,
                        source_path=reference.source_path,
                        source_relative_path=reference.relative_path,
                        source_sha256=reference.sha256,
                    )

                formats_generated = 0

                for format_enum in formats_to_emit:
                    ext = get_extension_for_format(format_enum)
                    try:
                        out_file = (
                            file_output_dir
                            / f"{file_path.stem}_{format_enum.value}{ext}"
                        )
                        serialized = parsing_system.serialize(
                            parse_result.model, format_enum
                        )
                        with open(out_file, "w", encoding="utf-8") as f:
                            f.write(serialized)

                        file_result["generated_formats"][format_enum.value] = {
                            "output_file": str(out_file),
                            "file_size": out_file.stat().st_size,
                            "success": True,
                        }
                        formats_generated += 1

                        if verbose:
                            logger.info(
                                f"  Generated {format_enum.value}: {out_file.stat().st_size} bytes"
                            )
                    except Exception as gen_err:  # Continue on per-format failure
                        logger.warning(
                            f"  Failed to generate {format_enum.value}: {gen_err}"
                        )
                        file_result["generated_formats"][format_enum.value] = {
                            "output_file": str(
                                file_output_dir
                                / f"{file_path.stem}_{format_enum.value}{ext}"
                            ),
                            "error": str(gen_err),
                            "success": False,
                        }

                processing_results["summary"]["formats_per_file"][file_path.name] = (
                    formats_generated
                )
                processing_results["summary"]["total_formats_generated"] += (
                    formats_generated
                )
                logger.info(
                    f"Generated {formats_generated} formats for {file_path.name}"
                )

                processing_results["processed_files"].append(file_result)
                processing_results["summary"]["successful_parses"] += 1
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                processing_results["processed_files"].append(
                    {
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "parse_success": False,
                        "error": str(e),
                    }
                )
                processing_results["summary"]["failed_parses"] += 1

        processing_results["summary"]["total_files"] = len(gnn_files)
        if run_context is not None:
            from gnn.pipeline.run_context import model_provenance

            for record in processing_results["processed_files"]:
                record.update(model_provenance(Path(record["file_path"]), 3))

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
