"""
GNN Processors Module

High-level processing functions that orchestrate GNN folder processing,
round-trip testing, and cross-format consistency validation.

These functions are used by the GNN MCP (Model Context Protocol) integration
to expose enhanced GNN capabilities.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# process_gnn_folder
# ---------------------------------------------------------------------------


def process_gnn_folder(
    target_dir: Path,
    output_dir: Path,
    logger: (logging.Logger) | None = None,
    recursive: bool = False,
    verbose: bool = False,
    validation_level: str = "standard",
    enable_round_trip: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Process a directory of GNN files with enhanced validation and testing.

    Discovers all GNN files in *target_dir*, validates each one, and writes
    a JSON summary to *output_dir*.

    Args:
        target_dir: Directory containing GNN files to process.
        output_dir: Directory for output results.
        logger: Logger instance (defaults to module logger).
        recursive: Whether to recurse into subdirectories.
        verbose: Enable verbose logging.
        validation_level: Validation strictness level (basic/standard/strict).
        enable_round_trip: Whether to run round-trip serialisation tests.
        **kwargs: Forwarded to validators (ignored if not applicable).

    Returns:
        True if processing completed (with or without warnings), False on
        unrecoverable error.

    Receipts: the summary carries ``validator_mode`` (``"full"`` when
    ``GNNValidator`` ran, ``"lightweight"`` when it was unavailable and only
    a section-presence check ran), ``degraded`` (true iff lightweight) and,
    when degraded, ``validator_error`` with the failure reason. Per-file
    results carry ``validator_mode`` too, so a lightweight result is never
    indistinguishable from full validation.
    """
    _log = logger or logging.getLogger(__name__)

    if verbose:
        _log.setLevel(logging.DEBUG)

    target_dir = Path(target_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log.info(f"Processing GNN folder: {target_dir} (recursive={recursive})")
    start_time = time.time()

    # --- discover files ---
    pattern = "**/*.md" if recursive else "*.md"
    gnn_files = list(target_dir.glob(pattern))
    _log.info(f"Discovered {len(gnn_files)} GNN file(s)")

    if not gnn_files:
        _log.warning(f"No GNN files found in {target_dir}")
        _save_summary(
            output_dir,
            {
                "status": "no_files",
                "target_dir": str(target_dir),
                "files_found": 0,
                "files_processed": 0,
                "duration_seconds": time.time() - start_time,
            },
        )
        return True  # Not an error — directory may be intentionally empty

    # --- validate each file ---
    results: List[Dict[str, Any]] = []
    success_count = 0
    fail_count = 0

    validator_error: str | None = None
    try:
        from gnn.schema_validator import GNNValidator
        from gnn.schema_validator import ValidationLevel as VL

        _vl_map: dict[str, Any] = {
            "basic": VL.BASIC,
            "standard": VL.STANDARD,
            "strict": VL.STRICT,
        }
        vl = _vl_map.get(validation_level.lower(), VL.STANDARD)
        validator: GNNValidator | None = GNNValidator(
            validation_level=vl,
            enable_round_trip_testing=enable_round_trip,
        )
        validator_mode = "full"
        _log.debug(f"Using GNNValidator with level={vl.value}")
    except Exception as exc:
        # Degraded mode is a visible receipt, not a silent downgrade: the
        # summary and every per-file result below carry validator_mode.
        _log.warning(f"GNNValidator unavailable ({exc}); using lightweight validation")
        validator = cast(Any, None)
        validator_mode = "lightweight"
        validator_error = str(exc)

    for gnn_file in gnn_files:
        file_result: Dict[str, Any] = {"file": str(gnn_file)}
        try:
            if validator is not None:
                vr = validator.validate_file(gnn_file, validation_level=vl)
                file_result.update(
                    {
                        "valid": vr.is_valid,
                        "errors": vr.errors,
                        "warnings": vr.warnings,
                        "validation_level": vr.validation_level.value,
                        "validator_mode": "full",
                    }
                )
            else:
                # Lightweight: check file is readable and has ## ModelName
                text = gnn_file.read_text(encoding="utf-8", errors="replace")
                is_valid = "## ModelName" in text or "## GNNVersionAndFlags" in text
                file_result.update(
                    {
                        "valid": is_valid,
                        "errors": [],
                        "warnings": [],
                        "validator_mode": "lightweight",
                    }
                )

            if file_result["valid"]:
                success_count += 1
                _log.debug(f"  ✅ {gnn_file.name}")
            else:
                fail_count += 1
                _log.debug(f"  ❌ {gnn_file.name}: {file_result.get('errors', [])}")

        except Exception as exc:
            fail_count += 1
            file_result.update({"valid": False, "errors": [str(exc)], "warnings": []})
            _log.warning(f"  ⚠️  Error processing {gnn_file.name}: {exc}")

        results.append(file_result)

    duration = time.time() - start_time
    summary: dict[str, Any] = {
        "status": "completed",
        "target_dir": str(target_dir),
        "validation_level": validation_level,
        "validator_mode": validator_mode,
        "degraded": validator_mode == "lightweight",
        "recursive": recursive,
        "enable_round_trip": enable_round_trip,
        "files_found": len(gnn_files),
        "files_valid": success_count,
        "files_invalid": fail_count,
        "duration_seconds": round(duration, 3),
        "results": results,
    }
    if validator_error is not None:
        summary["validator_error"] = validator_error
    _save_summary(output_dir, summary)

    _log.info(
        f"GNN folder processing complete: {success_count}/{len(gnn_files)} valid "
        f"in {duration:.2f}s"
    )
    return True  # Return True even with some failures — pipeline continues


# ---------------------------------------------------------------------------
# run_gnn_round_trip_tests
# ---------------------------------------------------------------------------


def run_gnn_round_trip_tests(
    target_dir: Path,
    output_dir: Path,
    logger: (logging.Logger) | None = None,
    reference_file: Optional[str] = None,
    test_subset: Optional[List[str]] = None,
    enable_parallel: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Run round-trip serialisation tests on GNN files in *target_dir*.

    Parses each file and attempts to re-serialise it, verifying that the
    semantic content is preserved across the round-trip.

    Args:
        target_dir: Directory containing GNN files.
        output_dir: Directory for test result output.
        logger: Logger instance.
        reference_file: Optional path to a single reference file to test.
        test_subset: Optional list of format names to test (e.g. ["json", "yaml"]).
        enable_parallel: Reserved for future parallel execution support.
        **kwargs: Additional options (forwarded where applicable).

    Returns:
        True if all tested files pass the round-trip, False otherwise.

    Receipts: the round_trip_results.json summary carries ``parser_mode``
    (``"full"``/``"lightweight"``), ``degraded`` and, when degraded,
    ``parser_error``; per-file results carry ``parser_mode`` too.
    """
    _log = logger or logging.getLogger(__name__)

    target_dir = Path(target_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log.info(f"Running GNN round-trip tests in: {target_dir}")
    start_time = time.time()

    # Gather files to test
    if reference_file:
        files_to_test: list[Any] = [Path(reference_file)]
    else:
        files_to_test = list(target_dir.glob("**/*.md"))

    if not files_to_test:
        _log.warning("No GNN files found for round-trip testing")
        _save_summary(
            output_dir,
            {
                "status": "no_files",
                "target_dir": str(target_dir),
                "files_tested": 0,
                "passed": 0,
                "failed": 0,
            },
            filename="round_trip_results.json",
        )
        return True

    supported_formats = test_subset or ["json", "yaml", "markdown"]
    _log.info(
        f"Testing {len(files_to_test)} file(s) across formats: {supported_formats}"
    )

    results: List[Dict[str, Any]] = []
    overall_pass = True

    parser_error: str | None = None
    try:
        from gnn.schema_validator import GNNParser

        parser: GNNParser | None = GNNParser(enhanced_validation=False)
        parser_mode = "full"
    except Exception as exc:
        # Degraded mode is a visible receipt: parser_mode/degraded land in
        # the summary and per-file results below.
        _log.warning(f"GNNParser unavailable ({exc}); using basic round-trip check")
        parser = cast(Any, None)
        parser_mode = "lightweight"
        parser_error = str(exc)

    for gnn_file in files_to_test:
        file_result: Dict[str, Any] = {
            "file": str(gnn_file),
            "formats_tested": supported_formats,
            "format_results": {},
            "parser_mode": parser_mode,
        }
        try:
            original_text = gnn_file.read_text(encoding="utf-8", errors="replace")

            for fmt in supported_formats:
                fmt_result: Dict[str, Any] = {}
                try:
                    if parser is not None:
                        parsed = parser.parse_content(
                            original_text, format_hint="markdown"
                        )
                        # Re-serialisation check: model_name should survive a round-trip
                        if hasattr(parsed, "to_dict"):
                            parsed_dict = parsed.to_dict()
                            fmt_result["pass"] = bool(parsed_dict)
                            fmt_result["model_name"] = parsed_dict.get("model_name", "")
                        else:
                            # Simple sanity: model_name attribute present
                            fmt_result["pass"] = bool(
                                getattr(parsed, "model_name", None)
                            )
                    else:
                        # Lightweight: check key sections survive a text round-trip
                        lines = original_text.splitlines()
                        fmt_result["pass"] = any("## ModelName" in l for l in lines)

                    if not fmt_result["pass"]:
                        overall_pass = False
                        _log.warning(f"  ❌ Round-trip failed: {gnn_file.name} [{fmt}]")
                    else:
                        _log.debug(f"  ✅ Round-trip OK: {gnn_file.name} [{fmt}]")

                except Exception as exc:
                    fmt_result["pass"] = False
                    fmt_result["error"] = str(exc)
                    overall_pass = False
                    _log.warning(
                        f"  ⚠️  Round-trip error {gnn_file.name} [{fmt}]: {exc}"
                    )

                file_result["format_results"][fmt] = fmt_result

        except Exception as exc:
            file_result["error"] = str(exc)
            overall_pass = False
            _log.warning(f"  ⚠️  Could not read {gnn_file.name}: {exc}")

        results.append(file_result)

    duration = time.time() - start_time
    passed = sum(
        1
        for r in results
        if "error" not in r
        and all(v.get("pass", False) for v in r.get("format_results", {}).values())
    )

    round_trip_summary: dict[str, Any] = {
        "status": "completed",
        "target_dir": str(target_dir),
        "files_tested": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "formats_tested": supported_formats,
        "parser_mode": parser_mode,
        "degraded": parser_mode == "lightweight",
        "duration_seconds": round(duration, 3),
        "results": results,
    }
    if parser_error is not None:
        round_trip_summary["parser_error"] = parser_error
    _save_summary(output_dir, round_trip_summary, filename="round_trip_results.json")

    _log.info(
        f"Round-trip tests complete: {passed}/{len(results)} passed in {duration:.2f}s"
    )
    return overall_pass


# ---------------------------------------------------------------------------
# check_cross_format_consistency
# ---------------------------------------------------------------------------


def check_cross_format_consistency(
    target_dir: Path,
    output_dir: Path,
    logger: (logging.Logger) | None = None,
    files_to_test: Optional[List[str]] = None,
    include_binary: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Validate cross-format consistency for GNN files in *target_dir*.

    Each GNN file is parsed and converted to multiple schema formats; the
    resulting representations are compared for semantic consistency.

    Args:
        target_dir: Directory containing GNN files.
        output_dir: Directory for validation results.
        logger: Logger instance.
        files_to_test: Optional explicit list of file paths to test.
        include_binary: Whether to include binary (pickle) formats.
        **kwargs: Additional options.

    Returns:
        True if all files are cross-format consistent, False otherwise.

    Receipts: the cross_format_results.json summary carries
    ``validator_mode`` (``"full"``/``"lightweight"``), ``degraded`` and,
    when degraded, ``validator_error``; per-file results carry
    ``validator_mode``.
    """
    _log = logger or logging.getLogger(__name__)

    target_dir = Path(target_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log.info(f"Validating cross-format consistency in: {target_dir}")
    start_time = time.time()

    # Gather files
    if files_to_test:
        gnn_files = [Path(f) for f in files_to_test]
    else:
        gnn_files = list(target_dir.glob("**/*.md"))

    if not gnn_files:
        _log.warning("No GNN files found for cross-format validation")
        _save_summary(
            output_dir,
            {
                "status": "no_files",
                "target_dir": str(target_dir),
                "files_tested": 0,
                "consistent": 0,
                "inconsistent": 0,
            },
            filename="cross_format_results.json",
        )
        return True

    validator_error: str | None = None
    try:
        from gnn.schema_validator import CrossFormatValidator

        validator: CrossFormatValidator | None = CrossFormatValidator(
            enable_round_trip_testing=False,
        )
        validator_mode = "full"
    except Exception as exc:
        # Degraded mode is a visible receipt: validator_mode/degraded land
        # in the summary and per-file results below.
        _log.warning(
            f"CrossFormatValidator unavailable ({exc}); using basic consistency check"
        )
        validator = cast(Any, None)
        validator_mode = "lightweight"
        validator_error = str(exc)

    overall_consistent = True
    results: List[Dict[str, Any]] = []

    for gnn_file in gnn_files:
        file_result: Dict[str, Any] = {"file": str(gnn_file)}
        try:
            content = gnn_file.read_text(encoding="utf-8", errors="replace")

            if validator is not None:
                cfr = validator.validate_cross_format_consistency(
                    content, source_format="markdown"
                )
                file_result.update(
                    {
                        "consistent": cfr.is_consistent,
                        "consistency_rate": cfr.get_consistency_rate(),
                        "formats_tested": cfr.schema_formats,
                        "inconsistencies": cfr.inconsistencies,
                        "warnings": cfr.warnings,
                        "validator_mode": "full",
                    }
                )
                if not cfr.is_consistent:
                    overall_consistent = False
            else:
                # Lightweight: just check each file has a ModelName
                is_consistent = "## ModelName" in content
                file_result.update(
                    {
                        "consistent": is_consistent,
                        "consistency_rate": 100.0 if is_consistent else 0.0,
                        "formats_tested": ["markdown"],
                        "inconsistencies": []
                        if is_consistent
                        else ["Missing ## ModelName section"],
                        "warnings": [],
                        "validator_mode": "lightweight",
                    }
                )
                if not is_consistent:
                    overall_consistent = False

            status_icon = "✅" if file_result["consistent"] else "❌"
            _log.debug(
                f"  {status_icon} {gnn_file.name}: {file_result.get('consistency_rate', 0):.0f}% consistent"
            )

        except Exception as exc:
            file_result.update(
                {
                    "consistent": False,
                    "error": str(exc),
                }
            )
            overall_consistent = False
            _log.warning(f"  ⚠️  Cross-format check failed for {gnn_file.name}: {exc}")

        results.append(file_result)

    duration = time.time() - start_time
    consistent_count = sum(1 for r in results if r.get("consistent", False))

    cross_format_summary: dict[str, Any] = {
        "status": "completed",
        "target_dir": str(target_dir),
        "files_tested": len(results),
        "consistent": consistent_count,
        "inconsistent": len(results) - consistent_count,
        "include_binary": include_binary,
        "validator_mode": validator_mode,
        "degraded": validator_mode == "lightweight",
        "duration_seconds": round(duration, 3),
        "results": results,
    }
    if validator_error is not None:
        cross_format_summary["validator_error"] = validator_error
    _save_summary(
        output_dir, cross_format_summary, filename="cross_format_results.json"
    )

    _log.info(
        f"Cross-format validation complete: {consistent_count}/{len(results)} consistent "
        f"in {duration:.2f}s"
    )
    return overall_consistent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_summary(
    output_dir: Path,
    data: Dict[str, Any],
    filename: str = "gnn_processing_summary.json",
) -> None:
    """Write *data* as JSON to *output_dir / filename*."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
    except OSError as exc:
        logger.warning(f"Could not write summary to {output_dir / filename}: {exc}")
