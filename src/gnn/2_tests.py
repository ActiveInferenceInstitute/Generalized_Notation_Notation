#!/usr/bin/env python3
"""
Step 2: Test Suite Execution (Thin Orchestrator)

Orchestrates comprehensive tests for the GNN pipeline. Follows thin orchestrator pattern,
delegating all core functionality to the tests module (tests.run_tests() from runner.py).

Usage:
  uv run python src/gnn/2_tests.py --fast-only --verbose      # Fast tests (default)
  uv run python src/gnn/2_tests.py --comprehensive --verbose  # All tests

Environment Variables:
  SKIP_TESTS_IN_PIPELINE  Set to skip tests entirely
  FAST_TESTS_TIMEOUT      Override timeout (default: 600 seconds)
"""

import os
import sys

if globals().get("__package__") is None and not sys.flags.safe_path:
    sys.path.pop(0)  # Keep adjacent gnn/types from shadowing stdlib types.

from pathlib import Path
from typing import Any, cast

from gnn.utils.pipeline_orchestration.pipeline_template import (
    create_standardized_pipeline_script,
)


def _test_runner_wrapper(
    target_dir: Any, output_dir: Any, logger: Any, **kwargs: Any
) -> bool:
    """Wrapper to map standard pipeline args to run_tests."""
    if os.getenv("SKIP_TESTS_IN_PIPELINE"):
        logger.info("⏭️ SKIP_TESTS_IN_PIPELINE set - tests will be skipped")
        return True

    # Script-mode invocation (python src/gnn/2_tests.py) puts only src/gnn on
    # sys.path; the tests package lives at the repository root.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from tests import run_tests
    except Exception as e:
        logger.error(f"Test harness import failed: {e}")
        return False

    if os.getenv("FAST_TESTS_TIMEOUT"):
        logger.info(f"⏱️ Custom timeout: {os.getenv('FAST_TESTS_TIMEOUT')} seconds")

    comprehensive = kwargs.get("comprehensive", False)
    fast_only = not comprehensive and kwargs.get("fast_only", True)
    test_mode = "comprehensive" if comprehensive else "fast"
    verbose = kwargs.get("verbose", False)

    logger.info(f"🧪 Running {test_mode} test suite")
    logger.info(f"📍 Output directory: {output_dir}")

    try:
        success = run_tests(
            logger=logger,
            output_dir=output_dir,
            verbose=verbose,
            fast_only=fast_only,
            comprehensive=comprehensive,
            generate_coverage=False,
            auto_fallback=True,
        )

        if not success:
            logger.error("❌ Test execution failed")
            logger.error(
                "💡 Check that test files exist and follow pytest naming conventions (test_*.py)"
            )

        return success
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback

        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False


additional_args: dict[str, Any] = {
    "fast_only": {
        "action": "store_true",
        "default": True,
        "help": "Run only fast tests (default)",
        "flag": "--fast-only",
    },
    "comprehensive": {
        "action": "store_true",
        "help": "Run all tests (overrides fast-only)",
        "flag": "--comprehensive",
    },
}

run_script = create_standardized_pipeline_script(
    "2_tests.py",
    _test_runner_wrapper,
    "Run GNN Pipeline Tests",
    additional_arguments=additional_args,
)


def main() -> int:
    """Main entry point for the tests step."""
    return cast("int", run_script())


if __name__ == "__main__":
    raise SystemExit(main())
