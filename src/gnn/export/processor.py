#!/usr/bin/env python3
"""
Export processor module for GNN Processing Pipeline.

This module provides the main export processing functionality.
"""

import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn.utils.observability.structured_logging import (
    log_step_error,
    log_step_start,
    log_step_success,
    log_step_warning,
)

# Import actual formatter implementations
from .formatters import (
    export_to_json_gnn,
    export_to_plaintext_dsl,
    export_to_plaintext_summary,
    export_to_python_pickle,
    export_to_xml_gnn,
)
from .notation_metadata import derive_geo_metadata
from .registry import DEFAULT_PIPELINE_FORMATS, get_format_spec, resolve_format_writer

# Canonical default format set for the pipeline path (``process_export``) and
# ``export_model``. ``export_gnn_model`` supports a different (text-leaning)
# set — see ``_GNN_MODEL_WRITERS``.
_DEFAULT_FORMATS: List[str] = list(DEFAULT_PIPELINE_FORMATS)

# ``export_model`` writes one file per format under fixed ``model.<ext>`` names.
_MODEL_FORMAT_FILES: Dict[str, str] = {
    "json": "model.json",
    "xml": "model.xml",
    "graphml": "model.graphml",
    "gexf": "model.gexf",
    "pickle": "model.pickle",
}

# ``export_gnn_model`` writers and their fixed output filenames. Only these
# formats are supported here (graph formats are not — the GNN-model dict
# shape does not carry the graph nodes/edges the graph exporters need).
_GNN_MODEL_WRITERS: Dict[str, Tuple[Any, str]] = {
    "json": (export_to_json_gnn, "gnn_model.json"),
    "xml": (export_to_xml_gnn, "gnn_model.xml"),
    "pickle": (export_to_python_pickle, "gnn_model.pickle"),
    "txt": (export_to_plaintext_summary, "gnn_model_summary.txt"),
    "dsl": (export_to_plaintext_dsl, "gnn_model.dsl"),
}

# Keep the five defaults stable; the strict interchange writer is an
# explicit opt-in resolved on demand in the pipeline loop.
_PIPELINE_WRITERS: Dict[str, Any] = {
    name: resolve_format_writer(name) for name in DEFAULT_PIPELINE_FORMATS
}


def _pipeline_formats_generated(
    geo_options: Any, requested_formats: Any = None
) -> Dict[str, int]:
    """Return the per-format counters, including ``geo_infer`` when opted in."""
    counters: Dict[str, int] = dict.fromkeys(DEFAULT_PIPELINE_FORMATS, 0)
    if isinstance(geo_options, dict) or (
        isinstance(requested_formats, (list, tuple, set))
        and "geo_infer" in requested_formats
    ):
        counters["geo_infer"] = 0
    return counters


def _pipeline_geo_options(geo_options: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the global ``geo_infer`` mapping for the strict writer.

    Resolves ``state_ids_path`` into the ``state_ids`` list the strict
    writer expects; ``raw_content`` is attached by the caller.
    """
    options: Dict[str, Any] = dict(geo_options)
    state_ids_path = options.pop("state_ids_path", None)
    if state_ids_path is not None:
        with open(state_ids_path, "r", encoding="utf-8") as f:
            state_ids = json.load(f)
        if not isinstance(state_ids, list):
            raise ValueError("geo_infer state_ids_path must point to a JSON array")
        options["state_ids"] = state_ids
    return options


def _read_geo_source(source: Path | None) -> str:
    """Read the bounded UTF-8 GNN source used by notation-derived metadata.

    Raises:
        ValueError: The source is not a file inside ``target_dir`` or exceeds
            the four MiB bound.
    """
    from .geo_infer import MAX_SOURCE_BYTES

    if source is None:
        raise ValueError("GNN source must be a file inside target_dir")
    with source.open("rb") as stream:
        raw = stream.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("GNN source exceeds four MiB")
    return raw.decode("utf-8")


def generate_exports(target_dir: Path, output_dir: Path, verbose: bool = False) -> bool:
    """
    Generate exports in multiple formats for GNN files.

    Args:
        target_dir: Directory containing GNN files to export
        output_dir: Directory to save exports
        verbose: Enable verbose output

    Returns:
        True if exports generated successfully, False otherwise
    """
    logger = logging.getLogger("export")

    try:
        log_step_start(logger, "Generating multi-format exports")

        # Create exports directory
        exports_dir = output_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)

        # Find GNN files
        gnn_files = list(target_dir.glob("*.md"))
        if not gnn_files:
            log_step_warning(logger, "No GNN files found for export")
            return True

        # Generate exports for each file
        export_results: dict[Any, Any] = {}
        for gnn_file in gnn_files:
            file_exports = export_single_gnn_file(gnn_file, exports_dir)
            export_results[gnn_file.name] = file_exports

        # Save export results
        results_file = exports_dir / "export_results.json"
        with open(results_file, "w") as f:
            json.dump(export_results, f, indent=2)

        # Check overall success
        all_successful = all(result["success"] for result in export_results.values())

        if all_successful:
            log_step_success(logger, "All exports generated successfully")
        else:
            failed_files = [
                name for name, result in export_results.items() if not result["success"]
            ]
            log_step_error(logger, f"Export failed for some files: {failed_files}")

        return all_successful

    except Exception as e:
        log_step_error(logger, f"Export generation failed: {e}")
        return False


def export_single_gnn_file(gnn_file: Path, exports_dir: Path) -> Dict[str, Any]:
    """
    Export a single GNN file to multiple formats.

    Args:
        gnn_file: Path to the GNN file to export
        exports_dir: Directory to save exports

    Returns:
        Dictionary with export results
    """
    try:
        # Read file content
        with open(gnn_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse GNN content
        parsed_content = parse_gnn_content(content)

        # Generate exports for the five pipeline formats, using the
        # canonical extension for each (pickle uses ``.pkl``).
        exports: dict[Any, Any] = {}
        extensions = {"pickle": "pkl"}
        for fmt in DEFAULT_PIPELINE_FORMATS:
            writer = resolve_format_writer(fmt)
            ext = extensions.get(fmt, fmt)
            output_file = exports_dir / f"{gnn_file.stem}.{ext}"
            if writer is None:
                exports[fmt] = False
            else:
                exports[fmt] = writer(parsed_content, output_file)

        return {
            "success": all(exports.values()),
            "exports": exports,
            "file_path": str(gnn_file),
        }

    except Exception as e:
        return {"success": False, "error": str(e), "file_path": str(gnn_file)}


def parse_gnn_content(content: str) -> Dict[str, Any]:
    """
    Parse GNN content into structured data.

    Args:
        content: Raw GNN file content

    Returns:
        Dictionary with parsed GNN data
    """
    try:
        from gnn import parse_gnn_file

        parsed = parse_gnn_file("inline_export_input.md", content=content)
        raw_sections = parsed.get("sections", {}) if parsed.get("success") else {}
        raw_variables = parsed.get("variables", []) if parsed.get("success") else []
        sections: object = _normalize_export_sections(raw_sections, content)
        variables = _normalize_export_variables(raw_variables, content)
        connections: list[Any] = []
        section_iterables: list[Any] = []
        if isinstance(sections, dict):
            section_iterables = [
                section_lines
                for section_name, section_lines in sections.items()
                if "connection" in str(section_name).lower()
            ]
        else:
            section_iterables = [content.splitlines()]

        for section_lines in section_iterables:
            if isinstance(section_lines, str):
                iterable = section_lines.splitlines()
            else:
                iterable = section_lines if isinstance(section_lines, list) else []
            for line in iterable:
                text = str(line).strip()
                operator = "->" if "->" in text else "→" if "→" in text else None
                if not operator:
                    continue
                source, target = text.split(operator, 1)
                connections.append({"source": source.strip(), "target": target.strip()})
        return {
            "sections": sections,
            "variables": variables,
            "connections": connections,
            "raw_content": content,
            "canonical_parse": parsed,
        }

    except Exception as e:
        return {"error": str(e), "raw_content": content}


def _normalize_export_sections(sections: Any, content: str) -> Dict[str, List[str]]:
    """Normalize canonical parser sections for compatibility export formatters."""
    if isinstance(sections, dict):
        normalized: Dict[str, List[str]] = {}
        for name, value in sections.items():
            if isinstance(value, str):
                normalized[str(name)] = value.splitlines()
            elif isinstance(value, list):
                normalized[str(name)] = [str(item) for item in value]
            else:
                normalized[str(name)] = [str(value)]
        return normalized

    parsed_sections: Dict[str, List[str]] = {}
    current_name: Optional[str] = None
    current_lines: List[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_name is not None:
                parsed_sections[current_name] = current_lines
            current_name = stripped.lstrip("#").strip() or "Untitled"
            current_lines = []
            continue
        if current_name is not None:
            current_lines.append(line)
    if current_name is not None:
        parsed_sections[current_name] = current_lines

    if isinstance(sections, list):
        for section_name in sections:
            parsed_sections.setdefault(str(section_name), [])
    return parsed_sections


def _normalize_export_variables(variables: Any, content: str) -> List[Dict[str, str]]:
    """Normalize canonical parser variables for compatibility graph/XML exporters."""
    normalized: List[Dict[str, str]] = []
    if isinstance(variables, list):
        for variable in variables:
            if isinstance(variable, dict):
                normalized.append(
                    {
                        "name": str(variable.get("name", "")),
                        "type": str(variable.get("type", variable.get("var_type", ""))),
                    }
                )
            else:
                normalized.append({"name": str(variable), "type": ""})

    types_by_name: Dict[str, str] = {}
    in_variables = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            in_variables = stripped.lstrip("#").strip().lower() == "variables"
            continue
        if not in_variables or ":" not in stripped:
            continue
        name, var_type = stripped.split(":", 1)
        types_by_name[name.strip()] = var_type.strip()

    if types_by_name:
        seen = {item["name"] for item in normalized}
        for item in normalized:
            if item["name"] in types_by_name:
                item["type"] = types_by_name[item["name"]]
        for name, var_type in types_by_name.items():
            if name not in seen:
                normalized.append({"name": name, "type": var_type})

    return normalized


def export_model(
    model_data: Dict[str, Any], output_dir: Path, formats: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Export model data to multiple formats.

    Args:
        model_data: Model data to export
        output_dir: Output directory
        formats: List of formats to export (default: all)

    Returns:
        Dictionary with export results
    """
    try:
        if formats is None:
            formats = list(_DEFAULT_FORMATS)

        results: dict[str, Any] = {
            "success": True,
            "exports": {},
            "errors": [],
            "formats": {},
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for format_type in formats:
            try:
                spec = get_format_spec(format_type)
                filename = _MODEL_FORMAT_FILES.get(format_type) or (
                    "model" + spec["extension"] if spec else None
                )
                writer = resolve_format_writer(format_type)
                if filename is None or writer is None:
                    results["errors"].append(f"Unsupported format: {format_type}")
                    continue

                output_file = output_dir / filename
                if format_type == "json":
                    # Recovery minimal JSON writer guarantees at least one
                    # success even if the formatter returns False or raises.
                    try:
                        success = writer(model_data, output_file)
                        if not success:
                            raise RuntimeError("formatter returned False")
                    except Exception as exc:
                        logging.getLogger(__name__).warning(
                            "JSON export recovery writer used for format '%s' "
                            "after formatter failure: %s",
                            format_type,
                            exc,
                        )
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(model_data, f, indent=2, ensure_ascii=False)
                        success = True
                else:
                    success = writer(model_data, output_file)

                results["exports"][format_type] = {
                    "success": success,
                    "file": str(output_file),
                }
                results["formats"][format_type] = success

                if not success:
                    results["success"] = False

            except Exception as e:
                results["errors"].append(f"Error exporting to {format_type}: {e}")
                results["success"] = False

        return results

    except Exception as e:
        return {"success": False, "error": str(e), "exports": {}, "errors": [str(e)]}


def _gnn_model_to_dict(gnn_content: str) -> Dict[str, Any]:
    """
    Convert GNN content to dictionary format.

    Args:
        gnn_content: Raw GNN content

    Returns:
        Dictionary representation of GNN model
    """
    try:
        # Parse the content
        parsed = parse_gnn_content(gnn_content)

        # Create structured model data
        model_data: dict[str, Any] = {
            "model_type": "gnn",
            "sections": parsed.get("sections", {}),
            "variables": parsed.get("variables", []),
            "connections": parsed.get("connections", []),
            "metadata": {"parsed_at": "2024-01-01T00:00:00Z", "version": "1.0.0"},
        }

        return model_data

    except Exception as e:
        return {"error": str(e), "raw_content": gnn_content}


def export_gnn_model(
    model_data: Dict[str, Any], output_dir: Path, formats: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Export GNN model to multiple formats.

    Args:
        model_data: GNN model data
        output_dir: Output directory
        formats: List of formats to export

    Returns:
        Dictionary with export results
    """
    try:
        if formats is None:
            formats = ["json", "xml", "pickle", "txt", "dsl"]

        # Normalize formats param if passed incorrectly as a single string.
        formats_raw: object = formats
        if isinstance(formats_raw, str):
            formats = [formats_raw]

        results: dict[str, Any] = {"success": True, "exports": {}, "errors": []}

        for format_type in formats:
            entry = _GNN_MODEL_WRITERS.get(format_type)
            if entry is None:
                results["errors"].append(f"Unsupported format: {format_type}")
                results["success"] = False
                continue
            writer, filename = entry
            try:
                output_file = output_dir / filename
                success = writer(model_data, output_file)
                results["exports"][format_type] = {
                    "success": success,
                    "file": str(output_file),
                }
                if not success:
                    results["success"] = False
            except Exception as e:
                results["errors"].append(f"Error exporting to {format_type}: {e}")
                results["success"] = False

        # Surface a top-level error string for failed runs so callers that
        # inspect ``error`` see one. (Previously a bogus ``"No valid formats
        # requested"`` message was appended on the success path — that is
        # removed; an all-success run now has an empty ``errors`` list.)
        if not results["success"] and "error" not in results:
            results["error"] = (
                "; ".join(results["errors"]) if results["errors"] else "Export failed"
            )
        return results

    except Exception as e:
        return {"success": False, "error": str(e), "exports": {}, "errors": [str(e)]}


def process_export(
    target_dir: Any, output_dir: Any, verbose: bool = False, **kwargs: Any
) -> bool:
    """
    Main export processing function for GNN models.

    This function orchestrates the complete export workflow including:
    - Multi-format export (JSON, XML, GraphML, GEXF, Pickle)
    - Format validation and error handling
    - Output directory management

    Args:
        target_dir: Directory containing GNN files to export
        output_dir: Output directory for export results
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options including 'formats' and
            'logger' (a ``logging.Logger`` injected by the pipeline
            template; when omitted the module logger is used). An optional
            ``geo_infer`` mapping (``step_seconds`` required; optional
            ``state_ids_path`` and ``space_kind``) opts the request into the
            strict GEO-INFER export when ``"geo_infer"`` is in ``formats``.
        geo_infer_options: Optional explicit per-model GEO metadata mapping
            (source filenames to options), set by ``process_export_cli``.
        geo_derive_metadata: When truthy, missing ``step_seconds`` (or the
            whole options mapping) is derived from explicit notation
            declarations as the last resolution fallback; explicit options
            win per field and the derivation is recorded in the artifact's
            ``provenance.metadata_derivation``.

    Returns:
        True if export succeeded, False otherwise
    """
    # Setup logging: honor an injected logger (passed by the pipeline
    # template as ``logger=...``); fall back to this module's logger.
    # The verbose flag only widens the level on the module-owned logger —
    # an injected logger's level is owned by its configurator.
    injected_logger = kwargs.pop("logger", None)
    logger = (
        injected_logger if injected_logger is not None else logging.getLogger(__name__)
    )
    if verbose and injected_logger is None:
        logger.setLevel(logging.DEBUG)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    # GEO-INFER opt-in: the strict writer's requirements (explicit options
    # per file, or a global ``geo_infer`` mapping) are enforced inside the
    # export loop so failures are recorded per format.
    requested_formats = kwargs.get("formats") or list(_DEFAULT_FORMATS)
    geo_options = kwargs.get("geo_infer")

    # Optional in-memory carrier (consolidated executor): step 3's parse
    # artifacts collected once by gnn.pipeline.step_executor and forwarded
    # as the ``parsed_model`` kwarg. An absent or malformed carrier leaves
    # the on-disk path below unchanged.
    carrier = kwargs.get("parsed_model")
    carrier_results = None
    carrier_models = None
    if isinstance(carrier, dict):
        raw_results = carrier.get("results")
        if isinstance(raw_results, dict) and isinstance(
            raw_results.get("processed_files"), list
        ):
            carrier_results = raw_results
        raw_models = carrier.get("models")
        if isinstance(raw_models, dict):
            carrier_models = raw_models

    try:
        if carrier_results is not None:
            gnn_results = carrier_results
            logger.info(
                "Loaded %d parsed GNN files (consolidated in-memory carrier)",
                len(gnn_results["processed_files"]),
            )
        else:
            # Load parsed GNN data from previous step (step 3). The consolidated
            # helper walks up out of any ``*_output`` subdir, so the old
            # name-prefix heuristic (6_/7_/8_ -> parent) is no longer needed.
            from gnn.pipeline.config import resolve_step_output_dir

            gnn_output_dir = resolve_step_output_dir("3_gnn", Path(output_dir))
            gnn_results_file = gnn_output_dir / "gnn_processing_results.json"

            if not gnn_results_file.exists():
                logger.error(
                    f"GNN processing results not found at {gnn_results_file}. Run step 3 first."
                )
                logger.error(f"Expected file location: {gnn_results_file}")
                logger.error(f"GNN output directory: {gnn_output_dir}")
                logger.error(f"GNN output directory exists: {gnn_output_dir.exists()}")
                if gnn_output_dir.exists():
                    logger.error(f"Contents: {list(gnn_output_dir.iterdir())}")
                return False

            with open(gnn_results_file, "r") as f:
                gnn_results = json.load(f)

            logger.info(
                f"Loaded {len(gnn_results['processed_files'])} parsed GNN files"
            )

        from gnn.pipeline.parse_receipt import selected_parse_receipt

        gnn_results = selected_parse_receipt(gnn_results, 7)

        # Export results
        export_results: dict[str, Any] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "source_directory": str(target_dir),
            "output_directory": str(output_dir),
            "files_exported": [],
            "summary": {
                "total_files": 0,
                "successful_exports": 0,
                "failed_exports": 0,
                "formats_generated": _pipeline_formats_generated(
                    kwargs.get("geo_infer"), requested_formats
                ),
            },
        }

        # Get requested formats
        requested_formats = kwargs.get("formats") or list(_DEFAULT_FORMATS)

        filename_counts: dict[str, int] = {}
        for entry in gnn_results["processed_files"]:
            name = entry.get("file_name", "")
            filename_counts[name] = filename_counts.get(name, 0) + 1

        # Process each file
        for file_result in gnn_results["processed_files"]:
            if not file_result["parse_success"]:
                continue

            file_name = file_result["file_name"]
            if (
                not isinstance(file_name, str)
                or not file_name.endswith(".md")
                or file_name in {".md", "..md"}
                or "/" in file_name
                or "\\" in file_name
                or "\x00" in file_name
            ):
                raise ValueError("Step 3 file_name must be a simple Markdown filename")
            logger.info(f"Exporting: {file_name}")

            # Load the actual parsed GNN specification — from the
            # consolidated in-memory carrier when one was forwarded
            # (identical bytes to the on-disk JSON; no re-read), else from
            # disk exactly as before.
            model_stem = Path(file_name).stem
            if carrier_models is not None and model_stem in carrier_models:
                model_data = carrier_models[model_stem]
                logger.info(
                    "Loaded parsed GNN specification for %s from the "
                    "consolidated in-memory carrier",
                    file_name,
                )
            else:
                parsed_model_file = file_result.get("parsed_model_file")
                if parsed_model_file and Path(parsed_model_file).exists():
                    try:
                        with open(parsed_model_file, "r") as f:
                            actual_gnn_spec = json.load(f)
                        logger.info(
                            f"Loaded parsed GNN specification from {parsed_model_file}"
                        )
                        model_data = actual_gnn_spec
                    except Exception as e:
                        logger.error(
                            f"Failed to load parsed GNN spec from {parsed_model_file}: {e}"
                        )
                        model_data = file_result
                else:
                    logger.warning(
                        f"Parsed model file not found for {file_name}, using summary data"
                    )
                    model_data = file_result

            source = None
            source_error = None
            relative_source = Path(file_name)
            if "geo_infer" in requested_formats:
                try:
                    source_root = Path(target_dir).resolve()
                    supplied_source = Path(file_result["file_path"])
                    candidates = (
                        [supplied_source.resolve()]
                        if supplied_source.is_absolute()
                        else [
                            supplied_source.resolve(),
                            (source_root / supplied_source).resolve(),
                        ]
                    )
                    source = next(
                        (
                            candidate
                            for candidate in candidates
                            if candidate.is_relative_to(source_root)
                            and candidate.is_file()
                        ),
                        None,
                    )
                    if source is None:
                        raise ValueError("GNN source must be a file inside target_dir")
                    if source.name != file_name:
                        raise ValueError(
                            "GNN source filename disagrees with Step 3 file_name"
                        )
                    relative_source = source.relative_to(source_root)
                except (TypeError, ValueError, OSError) as exc:
                    source_error = exc

            # Preserve nested source identities instead of overwriting equal basenames.
            file_output_dir = (
                output_dir / relative_source.parent / relative_source.stem
            ).resolve()
            if not file_output_dir.is_relative_to(Path(output_dir).resolve()):
                raise ValueError("Model output directory must remain inside output_dir")
            file_output_dir.mkdir(parents=True, exist_ok=True)

            file_export_result: dict[str, Any] = {
                "file_name": file_name,
                "file_path": file_result["file_path"],
                "exports": {},
                "success": True,
            }

            # Generate exports for each format
            for format_name in requested_formats:
                writer = _PIPELINE_WRITERS.get(format_name)
                if writer is None and format_name == "geo_infer":
                    writer = resolve_format_writer("geo_infer")
                if writer is None:
                    logger.warning(f"Unsupported format: {format_name}")
                    continue
                try:
                    spec = get_format_spec(format_name)
                    extension = spec["extension"].lstrip(".") if spec else format_name
                    export_file = (
                        file_output_dir
                        / f"{Path(file_name).stem}_{format_name}.{extension}"
                    )
                    if not export_file.resolve().is_relative_to(
                        Path(output_dir).resolve()
                    ):
                        raise ValueError("Export file must remain inside output_dir")
                    export_data = model_data
                    if format_name == "geo_infer":
                        if source_error is not None:
                            raise ValueError(str(source_error)) from source_error
                        per_model = kwargs.get("geo_infer_options", {})
                        options = None
                        if isinstance(per_model, dict):
                            options = per_model.get(relative_source.as_posix())
                            if options is None and filename_counts[file_name] == 1:
                                options = per_model.get(file_name)
                        if options is None and isinstance(geo_options, dict):
                            # Global CLI opt-in (--geo-* flags): a single
                            # ``geo_infer`` mapping applies to every model.
                            options = _pipeline_geo_options(geo_options)
                        derive_requested = bool(kwargs.get("geo_derive_metadata"))
                        geo_content: str | None = None
                        if isinstance(options, dict):
                            # Explicit metadata cannot claim derivation
                            # provenance; strip any forged marker first.
                            options = {
                                key: value
                                for key, value in options.items()
                                if key != "metadata_derivation"
                            }
                        elif derive_requested:
                            # Opt-in derivation is the last fallback before
                            # the explicit-options requirement.
                            geo_content = _read_geo_source(source)
                            derived = derive_geo_metadata(geo_content)
                            options = dict(derived["options"])
                            options["metadata_derivation"] = derived[
                                "metadata_derivation"
                            ]
                        if not isinstance(options, dict):
                            raise ValueError(
                                f"Explicit geo_infer_options required for {file_name}"
                            )
                        if "step_seconds" not in options and derive_requested:
                            # Explicit fields win; derivation only fills gaps.
                            geo_content = _read_geo_source(source)
                            derived = derive_geo_metadata(geo_content)
                            options = {**derived["options"], **options}
                            options["metadata_derivation"] = derived[
                                "metadata_derivation"
                            ]
                        if geo_content is None:
                            geo_content = _read_geo_source(source)
                        export_data = dict(
                            model_data,
                            raw_content=geo_content,
                            geo_infer=dict(options),
                        )
                    success = writer(export_data, export_file)

                    if success:
                        file_export_result["exports"][format_name] = {
                            "success": True,
                            "export_file": str(export_file),
                            "file_size": export_file.stat().st_size
                            if export_file.exists()
                            else 0,
                        }
                        export_results["summary"]["formats_generated"][format_name] += 1
                        logger.info(f"Generated {format_name} export for {file_name}")
                    else:
                        file_export_result["exports"][format_name] = {
                            "success": False,
                            "error": "Export function returned False",
                        }
                        file_export_result["success"] = False

                except Exception as e:
                    logger.error(
                        f"Failed to generate {format_name} export for {file_name}: {e}"
                    )
                    file_export_result["exports"][format_name] = {
                        "success": False,
                        "error": str(e),
                    }
                    file_export_result["success"] = False

            export_results["files_exported"].append(file_export_result)
            export_results["summary"]["total_files"] += 1

            if file_export_result["success"]:
                export_results["summary"]["successful_exports"] += 1
            else:
                export_results["summary"]["failed_exports"] += 1

        # Save export results
        export_results_file = output_dir / "export_results.json"
        if not export_results_file.resolve().is_relative_to(Path(output_dir).resolve()):
            raise ValueError("Export results manifest must remain inside output_dir")
        with open(export_results_file, "w") as f:
            json.dump(export_results, f, indent=2)

        # Save export summary
        export_summary_file = output_dir / "export_summary.json"
        if not export_summary_file.resolve().is_relative_to(Path(output_dir).resolve()):
            raise ValueError("Export summary manifest must remain inside output_dir")
        with open(export_summary_file, "w") as f:
            json.dump(export_results["summary"], f, indent=2)

        logger.info("Export processing completed:")
        logger.info(f"  Total files: {export_results['summary']['total_files']}")
        logger.info(
            f"  Successful exports: {export_results['summary']['successful_exports']}"
        )
        logger.info(f"  Failed exports: {export_results['summary']['failed_exports']}")
        logger.info(
            f"  Formats generated: {export_results['summary']['formats_generated']}"
        )

        success = (
            export_results["summary"]["successful_exports"] > 0
            and export_results["summary"]["failed_exports"] == 0
        )
        return cast("bool", success)

    except Exception as e:
        logger.error(f"Export processing failed: {e}")
        return False


def _check_content(path: Path) -> Tuple[bool, str]:
    """Lightweight sanity check for an exported file by extension.

    JSON must parse; XML/GraphML/GEXF must parse without DTDs or entities;
    pickle must load through the restricted GNN record loader; everything else must be non-empty UTF-8 text.
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif suffix in (".xml", ".graphml", ".gexf"):
            from defusedxml import ElementTree

            ElementTree.parse(
                path, forbid_dtd=True, forbid_entities=True, forbid_external=True
            )
        elif suffix in (".pkl", ".pickle"):
            from gnn.parsers.binary_parser import safe_pickle_load

            with path.open("rb") as stream:
                safe_pickle_load(stream)
        else:
            if not path.read_text(encoding="utf-8").strip():
                return False, "empty text file"
        return True, ""
    except Exception as e:
        return False, str(e)


def validate_export_outputs(
    output_dir: Any,
    expected_formats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate the artifacts of a completed ``process_export`` run.

    Reads ``<output_dir>/export_results.json`` and checks every export
    the manifest records as successful: the file must exist, be non-empty,
    and parse cleanly for its format (JSON loads; XML/GraphML/GEXF parse
    without DTDs or entities; pickle loads through the restricted GNN loader). When *expected_formats* is provided, models
    whose successful exports do not cover every expected format are
    reported as ``incomplete``.

    Args:
        output_dir: Directory that received ``process_export`` output.
        expected_formats: Optional list of format names that every
            exported model must include; missing ones go to
            ``incomplete``.

    Returns:
        Dict with keys ``success``, ``checked``, ``missing``,
        ``invalid``, ``incomplete``, ``files``.
    """
    out = Path(output_dir)
    manifest_path = out / "export_results.json"
    result: dict[str, Any] = {
        "success": True,
        "checked": 0,
        "missing": [],
        "invalid": [],
        "incomplete": [],
        "files": {},
    }
    if not manifest_path.exists():
        result["success"] = False
        result["missing"].append(str(manifest_path))
        return result
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        result["success"] = False
        result["invalid"].append({"file": str(manifest_path), "error": str(e)})
        return result

    expected = set(expected_formats) if expected_formats else set()

    for file_entry in manifest.get("files_exported", []):
        file_name = file_entry.get("file_name", "<unknown>")
        per_file: dict[str, Any] = {}
        covered: set[str] = set()
        for fmt, info in file_entry.get("exports", {}).items():
            if not info.get("success"):
                continue
            # ``process_export`` records ``export_file``; ``export_model``
            # records ``file`` — accept either.
            path_str = info.get("export_file") or info.get("file")
            if not path_str:
                continue
            path = Path(path_str)
            entry: dict[str, Any] = {"exists": path.exists()}
            if path.exists():
                entry["size"] = path.stat().st_size
                ok, err = _check_content(path)
                entry["valid"] = ok
                if not ok:
                    result["invalid"].append(
                        {"file": str(path), "format": fmt, "error": err}
                    )
                    result["success"] = False
                elif path.stat().st_size == 0:
                    result["invalid"].append(
                        {"file": str(path), "format": fmt, "error": "empty file"}
                    )
                    result["success"] = False
                else:
                    covered.add(fmt)
            else:
                result["missing"].append(str(path))
                result["success"] = False
            per_file[fmt] = entry
            result["checked"] += 1
        result["files"][file_name] = per_file
        if expected and not expected.issubset(covered):
            result["incomplete"].append(
                {"file": file_name, "missing_formats": sorted(expected - covered)}
            )
            result["success"] = False
    return result
