#!/usr/bin/env python3
"""
Round-trip testing strategy for GNN models.

This module provides the RoundTripTestStrategy class for testing
semantic preservation across format conversions.
"""

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RoundTripResult:
    """Result of a round-trip test operation."""

    success: bool
    target_format: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    test_time: float = 0.0
    source_file: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RoundTripTestStrategy:
    """
    Round-trip testing strategy for GNN models.

    Tests semantic preservation across format conversions
    and validates model consistency.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.test_subset: Optional[List[str]] = None
        self.reference_file: Optional[str] = None
        self.output_dir: Optional[Path] = None
        self.round_trip_tester: Any = None
        self._initialize_tester()

    def configure(
        self,
        test_subset: Optional[List[str]] = None,
        reference_file: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> Any:
        """Configure testing parameters."""
        self.test_subset = test_subset
        self.reference_file = reference_file
        self.output_dir = output_dir

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_tester(self) -> Any:
        """Initialize round-trip testing components."""
        try:
            from .round_trip_tester import GNNRoundTripTester

            self.round_trip_tester = GNNRoundTripTester()
            logger.debug("Round-trip tester initialized")
        except ImportError as e:
            logger.warning(f"Round-trip tester not available: {e}")
            self.round_trip_tester = None

    def test(self, files: List[Path]) -> Dict[str, Any]:
        """
        Run round-trip tests on provided files.

        Args:
            files: List of file paths to test

        Returns:
            Dictionary with test results and metadata
        """
        if not self.round_trip_tester:
            return {
                "success": False,
                "error": "Round-trip tester not available",
                "tests_run": 0,
                "files_tested": 0,
            }

        logger.info(f"Running round-trip tests on {len(files)} files")

        results: dict[str, Any] = {
            "success": True,
            "tests_run": 0,
            "files_tested": len(files),
            "file_results": {},
            "summary": {},
        }

        for file_path in files:
            try:
                file_result = self._test_file_round_trip(file_path)
                results["file_results"][str(file_path)] = file_result
                results["tests_run"] += 1

                if not file_result["success"]:
                    results["success"] = False

            except Exception as e:
                logger.error(f"Round-trip test failed for {file_path}: {e}")
                results["file_results"][str(file_path)] = {
                    "success": False,
                    "error": str(e),
                }
                results["success"] = False

        # Generate summary
        results["summary"] = self._generate_test_summary(results["file_results"])

        # Save results if output directory is configured
        if self.output_dir:
            self._save_test_results(results)

        return results

    def _test_file_round_trip(self, file_path: Path) -> Dict[str, Any]:
        """Test round-trip conversion for a single file."""
        logger.debug(f"Testing round-trip for {file_path}")

        try:
            from .round_trip_tester import GNNRoundTripTester

            tester = self.round_trip_tester
            if type(tester) is GNNRoundTripTester:
                if not file_path.exists():
                    raise FileNotFoundError(
                        f"Round-trip source file not found: {file_path}"
                    )
                # A native tester owns mutable source, timeout and artifact
                # state. Each call gets its own instance and artifact directory
                # so neither sequential nor concurrent sources can cross over.
                tester.temp_dir.mkdir(parents=True, exist_ok=True)
                directory = Path(
                    tempfile.mkdtemp(prefix="source-", dir=tester.temp_dir)
                )
                local_tester = GNNRoundTripTester(temp_dir=directory)
                local_tester.reference_file = file_path
                local_tester.supported_formats = list(tester.supported_formats)
                tester = local_tester

            # Run comprehensive round-trip tests
            report = tester.run_comprehensive_tests()
            report_source = getattr(report, "reference_file", None)

            # Extract results for this file
            file_results: list[Any] = []
            for result in report.round_trip_results:
                # Native rows belong to their report's source. Injected
                # reports can continue supplying a per-row source_file.
                result_source = getattr(result, "source_file", report_source)
                if result_source is not None and str(result_source) == str(file_path):
                    file_results.append(
                        {
                            "target_format": result.target_format.value
                            if hasattr(result.target_format, "value")
                            else str(result.target_format),
                            "success": result.success,
                            "errors": result.errors,
                            "warnings": result.warnings,
                            "test_time": result.test_time,
                        }
                    )

            success_rate = (
                sum(1 for r in file_results if r["success"])
                / max(len(file_results), 1)
                * 100
            )

            return {
                "success": success_rate >= 80.0,  # 80% success threshold
                "success_rate": success_rate,
                "format_results": file_results,
                "total_formats_tested": len(file_results),
            }

        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            logger.error("Round-trip source %s failed (%s)", file_path, reason)
            return {
                "success": False,
                "error": reason,
                "success_rate": 0.0,
                "format_results": [],
                "total_formats_tested": 0,
            }

    def _generate_test_summary(self, file_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from test results."""
        total_files = len(file_results)
        successful_files = sum(
            1 for result in file_results.values() if result.get("success", False)
        )

        format_stats: dict[Any, Any] = {}
        for result in file_results.values():
            if "format_results" in result:
                for format_result in result["format_results"]:
                    format_name = format_result["target_format"]
                    if format_name not in format_stats:
                        format_stats[format_name] = {"success": 0, "total": 0}
                    format_stats[format_name]["total"] += 1
                    if format_result["success"]:
                        format_stats[format_name]["success"] += 1

        return {
            "total_files": total_files,
            "successful_files": successful_files,
            "success_rate": (successful_files / max(total_files, 1)) * 100,
            "format_statistics": format_stats,
        }

    def _save_test_results(self, results: Dict[str, Any]) -> Any:
        """Save test results to output directory."""
        if not self.output_dir:
            return

        try:
            import json
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.output_dir / f"round_trip_test_results_{timestamp}.json"

            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Round-trip test results saved to {results_file}")

        except Exception as e:
            logger.error(f"Failed to save test results: {e}")
