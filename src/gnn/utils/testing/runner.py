"""Public test utilities delegating lazily to the monitored source-test runner.

Native, daemon and environment-mutating tests require their explicit profiles.
An installed distribution without the repository harness reports unavailable.
"""

import logging
import math
from pathlib import Path
from typing import Any

from gnn.utils.testing.constants import TEST_CONFIG, TEST_DIR, TEST_STAGES
from gnn.utils.testing.environment import (
    cleanup_test_environment,
    setup_test_environment,
)

_HARNESS_DIR = TEST_DIR / "infrastructure"


class TestRunner:
    """Adapter preserving the public dictionary result and lazy imports."""

    def __init__(self, config: Any = None) -> None:
        """Store configuration without importing the repository test harness."""
        self.config = config
        self.logger = logging.getLogger("test_runner")

    def run_tests(self, test_paths: Any, output_dir: Any) -> dict[str, Any]:
        """Run real pytest subprocesses through the canonical monitored runner."""
        try:
            from importlib import import_module

            owners = {
                "tests.infrastructure.test_config": _HARNESS_DIR / "test_config.py",
                "tests.infrastructure.test_runner": _HARNESS_DIR / "test_runner.py",
            }
            if any(not path.is_file() for path in owners.values()):
                raise ImportError("Repository test harness files are unavailable")
            for name, expected_path in owners.items():
                module = import_module(name)
                module_path = getattr(module, "__file__", None)
                if (
                    module_path is None
                    or Path(module_path).resolve() != expected_path.resolve()
                ):
                    raise ImportError(f"Repository test harness owner mismatch: {name}")
            from tests.infrastructure.test_config import TestExecutionConfig
            from tests.infrastructure.test_runner import TestRunner as SourceTestRunner

            if self.config is None or isinstance(self.config, dict):
                options: dict[str, Any] = {"parallel": False, "coverage": False}
                options.update(self.config or {})
                config = TestExecutionConfig(**options)
            elif isinstance(self.config, TestExecutionConfig):
                config = self.config
            else:
                raise ValueError(
                    "Expected test configuration dictionary or TestExecutionConfig"
                )
            if config.parallel:
                raise ValueError(
                    "This adapter requires serial test execution (parallel=False)"
                )
            if (
                isinstance(config.timeout_seconds, bool)
                or not isinstance(config.timeout_seconds, (int, float))
                or not math.isfinite(config.timeout_seconds)
                or config.timeout_seconds <= 0
            ):
                raise ValueError("timeout_seconds must be finite and positive")
            if (
                isinstance(config.max_failures, bool)
                or not isinstance(config.max_failures, int)
                or config.max_failures <= 0
            ):
                raise ValueError("max_failures must be a positive integer")
            if not isinstance(config.markers, list) or any(
                not isinstance(marker, str) or not marker.strip()
                for marker in config.markers
            ):
                raise ValueError("markers must be a list of nonempty expressions")
            selected_paths = (
                [test_paths]
                if isinstance(test_paths, (str, Path))
                else list(test_paths)
            )
            paths = [Path(path).resolve() for path in selected_paths]
            if not paths or any(not path.exists() for path in paths):
                raise ValueError("Test selection must contain existing test paths")
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            result: dict[str, Any] = (
                SourceTestRunner(config).run_tests(paths, output).to_dict()
            )
            result["success"] = bool(result["success"] and result["tests_passed"] > 0)
            if result["tests_passed"] == 0 and not result.get("error_message"):
                result["error_message"] = (
                    "No tests passed; empty or fully skipped selections are unverified"
                )
            return result
        except Exception as error:
            self.logger.error("Test execution unavailable or failed: %s", error)
            return {"success": False, "tests_run": 0, "error": str(error)}


class TestResult:
    """Public summary counters and failure context."""

    def __init__(
        self,
        success: Any = False,
        tests_run: Any = 0,
        tests_passed: Any = 0,
        tests_failed: Any = 0,
        tests_skipped: Any = 0,
        execution_time: Any = 0.0,
        error_message: Any = None,
    ) -> None:
        """Store summary counters and optional failure context."""
        self.success = success
        self.tests_run = tests_run
        self.tests_passed = tests_passed
        self.tests_failed = tests_failed
        self.tests_skipped = tests_skipped
        self.execution_time = execution_time
        self.error_message = error_message

    def to_dict(self) -> Any:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_skipped": self.tests_skipped,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
        }


class TestCategory:
    """Named test category metadata."""

    def __init__(self, name: Any = "", description: Any = "") -> None:
        """Store the category name and description."""
        self.name = name
        self.description = description

    def __str__(self) -> Any:
        """Return the compact category display name."""
        return f"TestCategory({self.name})"

    def __repr__(self) -> Any:
        """Return the developer-facing category representation."""
        return self.__str__()


class TestStage:
    """Named stage execution limits."""

    def __init__(
        self,
        name: Any = "",
        timeout: Any = 300,
        max_failures: Any = 10,
        parallel: Any = True,
        coverage: Any = False,
    ) -> None:
        """Store stage execution limits and coverage settings."""
        self.name = name
        self.timeout = timeout
        self.max_failures = max_failures
        self.parallel = parallel
        self.coverage = coverage

    def __str__(self) -> Any:
        """Return the compact stage display name."""
        return f"TestStage({self.name})"

    def __repr__(self) -> Any:
        """Return the developer-facing stage representation."""
        return self.__str__()


class CoverageTarget:
    """Named coverage threshold."""

    def __init__(self, name: Any = "", target_percentage: Any = 0.0) -> None:
        """Store the coverage target name and threshold."""
        self.name = name
        self.target_percentage = target_percentage

    def __str__(self) -> Any:
        """Return the compact coverage target display name."""
        return f"CoverageTarget({self.name}: {self.target_percentage}%)"

    def __repr__(self) -> Any:
        """Return the developer-facing coverage target representation."""
        return self.__str__()


# These source selections never provision dependencies or start a daemon.
_SOURCE_PROFILE = "not ollama and not env_heavy and not toolchain"
_CATEGORY_MARKERS = {
    "fast": "fast and not slow and not performance",
    "standard": "not fast and not slow and not performance",
    "slow": "slow",
    "performance": "performance",
    "unit": "unit",
    "integration": "integration",
    "mcp": "mcp",
}


def _run_selection(
    output_dir: Path,
    verbose: bool,
    *,
    category: str | None = None,
    stage: dict[str, Any] | None = None,
    coverage: bool = False,
) -> dict[str, Any]:
    """Execute an admitted source selection and retain its actual outcome."""
    if category is not None and category not in _CATEGORY_MARKERS:
        return {
            "success": False,
            "tests_run": 0,
            "error": f"Unknown test category: {category}",
        }
    selection = _SOURCE_PROFILE
    if category is not None:
        selection = f"({_CATEGORY_MARKERS[category]}) and ({selection})"
    limits = stage or {}
    result = TestRunner(
        {
            "parallel": limits.get("parallel", False),
            "coverage": coverage,
            "verbose": verbose,
            "markers": [selection],
            "timeout_seconds": limits.get(
                "timeout", TEST_CONFIG.get("timeout_seconds", 300)
            ),
            "max_failures": limits.get("max_failures", 10),
        }
    ).run_tests([TEST_DIR], output_dir)
    return {**result, "selection": selection, "coverage_requested": coverage}


def run_tests(target_dir: Path, output_dir: Path, verbose: bool = False) -> bool:
    """Run source tests; target_dir is the pipeline's GNN input, not test discovery."""
    return bool(_run_selection(output_dir, verbose)["success"])


def run_test_category(
    category: str, target_dir: Path, output_dir: Path, verbose: bool = False
) -> bool:
    """Run only the requested admitted category; unknown and empty selections fail."""
    return bool(_run_selection(output_dir, verbose, category=category)["success"])


def run_test_stage(
    stage: str, target_dir: Path, output_dir: Path, verbose: bool = False
) -> bool:
    """Apply the named stage's selection, timeout, failure and coverage controls."""
    if stage not in TEST_STAGES or stage not in _CATEGORY_MARKERS:
        logging.getLogger(__name__).error("Unknown test stage: %s", stage)
        return False
    settings = TEST_STAGES[stage]
    return bool(
        _run_selection(
            output_dir,
            verbose,
            category=stage,
            stage=settings,
            coverage=bool(settings.get("coverage", False)),
        )["success"]
    )


def _run_all_test_results(
    target_dir: Path, output_dir: Path, verbose: bool = False
) -> dict[str, Any]:
    """Return only current execution results, shared by Python and MCP callers."""
    from gnn.utils.testing.reports import generate_test_report

    results: dict[str, bool] = {}
    category_results: dict[str, Any] = {}
    report: dict[str, Any] = {
        "success": False,
        "selection_profile": _SOURCE_PROFILE,
        "results": results,
        "category_results": category_results,
    }
    test_output = Path(output_dir) / "test_results"
    try:
        setup_test_environment()
        categories = ["fast"]
        if not TEST_CONFIG.get("fast_only", False):
            categories.append("standard")
            # Repository slow tests are also tagged performance. Select them
            # once; performance adds the remaining performance-marked tests.
            categories.append(
                "performance"
                if TEST_CONFIG.get("include_performance", False)
                else "slow"
            )
        test_output.mkdir(parents=True, exist_ok=True)
        for category in categories:
            category_results[category] = _run_selection(
                test_output / category, verbose, category=category
            )
            results[category] = category_results[category]["success"] is True
        report["success"] = bool(results) and all(results.values())
        report["test_categories"] = list(results)
    except Exception as error:
        report.update(success=False, error=str(error))
    finally:
        try:
            cleanup_test_environment()
        except Exception as error:
            report.update(success=False, cleanup_error=str(error))
    if test_output.is_dir() and not generate_test_report(report, test_output):
        report.update(success=False, error="Could not write current test report")
    return report


def run_all_tests(target_dir: Path, output_dir: Path, verbose: bool = False) -> bool:
    """Run configured source categories and publish their observed results."""
    return bool(_run_all_test_results(target_dir, output_dir, verbose)["success"])


def run_fast_tests(target_dir: Path, output_dir: Path, verbose: bool) -> bool:
    """Run the fast source-test selection."""
    return run_test_category("fast", target_dir, output_dir, verbose)


def run_standard_tests(target_dir: Path, output_dir: Path, verbose: bool) -> bool:
    """Run source tests outside the fast, slow and performance selections."""
    return run_test_category("standard", target_dir, output_dir, verbose)


def run_slow_tests(target_dir: Path, output_dir: Path, verbose: bool) -> bool:
    """Run the slow source-test selection."""
    return run_test_category("slow", target_dir, output_dir, verbose)


def run_performance_tests(target_dir: Path, output_dir: Path, verbose: bool) -> bool:
    """Run the performance source-test selection."""
    return run_test_category("performance", target_dir, output_dir, verbose)


def run_coverage_tests(test_results_dir: Path, verbose: bool) -> bool:
    """Run source tests with coverage, failing if the coverage plugin is unavailable."""
    return bool(_run_selection(test_results_dir, verbose, coverage=True)["success"])
