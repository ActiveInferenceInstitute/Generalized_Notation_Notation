"""
Test runner mode functions.

Provides run_fast_pipeline_tests(), run_comprehensive_tests(),
and run_fast_reliable_tests() execution modes.

Extracted from runner.py for maintainability.
"""

import json
import logging
import os
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest

# Import from infrastructure
from .infrastructure import (
    _extract_collection_errors,
    _generate_error_report,
    _generate_markdown_report,
    _generate_timeout_report,
    _parse_coverage_statistics,
    _parse_test_statistics,
)
from .infrastructure.report_generator import flatten_pipeline_test_summary


def _build_fast_pipeline_command(
    has_xdist: bool,
    has_timeout: bool,
    timeout_value: str = "600",
    verbose: bool = False,
) -> list[Any]:
    """Build the marker-based fast pytest command.

    SC-40: the fast subset is marker-based (``-m fast``, defined in
    pytest.ini) and targets the tests/ tree — the pre-3.3.0 test directory
    named in the old ignores no longer exists, and slow tests are
    deselected by the marker, so no per-file ignores are needed.
    """
    cmd: list[Any] = [
        sys.executable,
        "-m",
        "pytest",
        "--tb=short",
        "--maxfail=5",
        "--durations=10",
        "-ra",
    ]

    if has_xdist:
        # Parallel execution keeps the "fast" subset under the overall timeout
        # on shared/CI machines (serial ``-m "not slow"`` exceeded 600s).
        cmd.extend(["-n", "auto"])

    if has_timeout:
        cmd.extend(["--timeout", timeout_value])

    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    cmd.extend(["-m", "fast"])
    cmd.append("tests/")
    return cmd


def run_fast_pipeline_tests(
    logger: logging.Logger, output_dir: Path, verbose: bool = False
) -> bool:
    """
    Run FAST tests for quick pipeline validation.

    This runs only fast tests (marked with the ``fast`` marker) to keep
    pipeline execution efficient.
    """
    if os.getenv("SKIP_TESTS_IN_PIPELINE"):
        logger.info("Skipping tests (SKIP_TESTS_IN_PIPELINE set)")
        logger.info("Set SKIP_TESTS_IN_PIPELINE='' or unset to run tests in pipeline")
        return True

    logger.info("Running fast test subset for quick pipeline validation")
    logger.info("To skip tests in pipeline: export SKIP_TESTS_IN_PIPELINE=1")
    logger.info("To customize timeout: export FAST_TESTS_TIMEOUT=<seconds>")

    try:
        import pytest_timeout  # type: ignore[import-untyped]

        has_timeout = True
    except ImportError:
        has_timeout = False

    try:
        import xdist  # type: ignore[import-untyped]  # noqa: F401 - presence check for parallel execution

        has_xdist = True
    except ImportError:
        has_xdist = False

    cmd: list[Any] = _build_fast_pipeline_command(
        has_xdist=has_xdist,
        has_timeout=has_timeout,
        timeout_value=os.getenv("FAST_TESTS_TIMEOUT", "600"),
        verbose=verbose,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    isolated_output_root = output_dir / "isolated_pipeline_outputs"

    project_root = Path(__file__).parent.parent

    logger.info(f"Executing fast tests: {' '.join(cmd)}")

    try:
        overall_timeout = int(os.getenv("FAST_TESTS_TIMEOUT", "600")) + 30
        env = os.environ.copy()
        env.setdefault("GNN_PIPELINE_TEST_OUTPUT_DIR", str(isolated_output_root))

        result = subprocess.run(  # nosec B603
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=overall_timeout,
            env=env,
        )

        output_file = output_dir / "pytest_reliable_output.txt"
        with open(output_file, "w") as f:
            f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")

        collection_errors = _extract_collection_errors(result.stdout, result.stderr)
        if collection_errors:
            logger.warning(
                f"Found {len(collection_errors)} collection errors (import/syntax issues)"
            )
            for err in collection_errors[:3]:
                logger.warning(f"  Collection error: {err[:200]}")

        test_stats = _parse_test_statistics(result.stdout)
        coverage_json = project_root / "coverage.json"
        coverage_stats = (
            _parse_coverage_statistics(coverage_json) if coverage_json.is_file() else {}
        )

        summary: dict[str, Any] = {
            "execution_summary": {
                "test_mode": "fast_pipeline",
                "command": " ".join(cmd),
                "exit_code": result.returncode,
                "tests_run": test_stats.get("total", 0),
                "tests_passed": test_stats.get("passed", 0),
                "tests_failed": test_stats.get("failed", 0),
                "tests_skipped": test_stats.get("skipped", 0),
                "tests_errors": test_stats.get("errors", 0),
                "collection_errors": len(collection_errors),
                "coverage": coverage_stats,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        }

        summary_file = output_dir / "test_execution_report.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        _generate_markdown_report(
            output_dir / "test_execution_report.md",
            flatten_pipeline_test_summary(summary),
        )

        success = result.returncode == 0
        total = test_stats.get("total", 0)
        passed = test_stats.get("passed", 0)
        failed = test_stats.get("failed", 0)
        skipped = test_stats.get("skipped", 0)

        if success:
            logger.info(
                f"Fast tests completed: {total} run, {passed} passed, {skipped} skipped"
            )
        else:
            logger.warning(
                f"Fast tests had failures: {total} run, {passed} passed, {failed} failed, {skipped} skipped"
            )

        return success

    except subprocess.TimeoutExpired:
        logger.error("Fast test execution timed out")
        _generate_timeout_report(output_dir, cmd, overall_timeout)
        return False
    except Exception as e:
        logger.error(f"Fast test execution failed: {e}")
        _generate_error_report(output_dir, cmd, str(e))
        return False


def run_comprehensive_tests(
    logger: logging.Logger, output_dir: Path, verbose: bool = False
) -> bool:
    """
    Run comprehensive tests including slow and performance tests.
    """
    logger.info("Running comprehensive test suite (all categories)")

    from .test_runner_modular import _ModularTestRunner

    class ComprehensiveArgs:
        def __init__(self, output_dir: Any, verbose: Any) -> None:
            self.output_dir = str(output_dir)
            self.verbose = verbose
            self.categories = None
            self.parallel = False

    args = ComprehensiveArgs(output_dir, verbose)
    runner = _ModularTestRunner(args, logger)
    results = runner.run_all_categories()

    success = results.get("overall_success", False)
    if success:
        logger.info("Comprehensive tests completed successfully")
    else:
        logger.warning("Comprehensive tests had some failures")

    return cast("bool", success)


def run_fast_reliable_tests(
    logger: logging.Logger, output_dir: Path, verbose: bool = False, timeout: int = 600
) -> bool:
    """
    Run a reliable subset of fast tests with improved error handling.

    Args:
        logger: Logger instance
        output_dir: Output directory for test results
        verbose: Enable verbose output
        timeout: Subprocess timeout in seconds (default 180s, overridable via FAST_TESTS_TIMEOUT env var)
    """
    # Allow environment variable override for timeout
    env_timeout = os.getenv("FAST_TESTS_TIMEOUT")
    if env_timeout:
        try:
            timeout = int(env_timeout)
            logger.info(f"⏱️ Using FAST_TESTS_TIMEOUT override: {timeout}s")
        except ValueError:
            logger.warning(
                f"⚠️ Invalid FAST_TESTS_TIMEOUT value '{env_timeout}', using default {timeout}s"
            )

    logger.info(f"Running reliable fast test subset (timeout: {timeout}s)")

    reliable_tests: list[str] = [
        "test_core_modules.py",
        "test_fast_suite.py",
        "pipeline/test_main_orchestrator.py",
    ]

    cmd: list[Any] = [
        sys.executable,
        "-m",
        "pytest",
        "--tb=short",
        "--maxfail=3",
        "--durations=3",
        "-v" if verbose else "-q",
    ]

    test_dir = Path(__file__).parent
    for test_file in reliable_tests:
        test_path = test_dir / test_file
        if test_path.exists():
            cmd.append(str(test_path))

    logger.info(f"Executing reliable tests: {' '.join(cmd)}")

    try:
        result = subprocess.run(  # nosec B603
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "pytest_reliable_output.txt", "w") as f:
            f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")

        success = result.returncode == 0
        if success:
            logger.info("Reliable fast tests completed successfully")
        else:
            logger.warning("Reliable fast tests had some failures")

        return success

    except subprocess.TimeoutExpired:
        logger.error("Reliable test execution timed out")
        return False
    except Exception as e:
        logger.error(f"Reliable test execution failed: {e}")
        return False


pytestmark = pytest.mark.fast


def test_fast_pipeline_command_is_marker_based_on_tests_tree() -> None:
    """SC-40: the fast subset is ``-m fast`` over tests/, with no stale paths."""
    cmd = _build_fast_pipeline_command(
        has_xdist=False, has_timeout=False, verbose=False
    )
    assert cmd[-3:] == ["-m", "fast", "tests/"]
    assert not any("src/tests" in arg for arg in cmd)


def test_comprehensive_routing_covers_the_whole_tree() -> None:
    """SC-40: the routing table is grown from the tree, so every test file
    pytest would collect is routed for comprehensive runs."""
    from .categories import discover_test_files, get_all_test_files

    assert set(discover_test_files()) <= set(get_all_test_files())
