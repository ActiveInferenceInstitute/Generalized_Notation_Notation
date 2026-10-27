#!/usr/bin/env python3
"""
Test Runner Helper

This script provides convenient commands for running different test configurations
in the GNN pipeline. It supports all the staging options and provides quick access
to common test scenarios.

Usage:
    python tests/test_runner_helper.py --help
    python tests/test_runner_helper.py fast
    python tests/test_runner_helper.py full
    python tests/test_runner_helper.py debug
"""

import argparse
import os
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Source imports and the repository-only test harness have separate roots.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_PROJECT_ROOT / "src"), str(_PROJECT_ROOT)]


def _default_output_dir() -> str:
    """Return the helper output root, overridable by pipeline test runners."""
    return os.getenv("GNN_PIPELINE_TEST_OUTPUT_DIR", "output")


def run_test_configuration(
    config_name: str,
    target_dir: str = "input/gnn_files",
    output_dir: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    """Run a predefined test configuration."""
    output_dir = output_dir or _default_output_dir()

    # Define test configurations
    configurations: dict[str, Any] = {
        "fast": {
            "args": ["--fast-only"],
            "description": "Run the Step 2 fast selection",
        },
        "standard": {
            "args": [],
            "description": "Run source tests outside slow and performance selections",
        },
        "full": {
            "args": ["--comprehensive"],
            "description": "Run all tests including slow integration tests",
        },
        "performance": {
            "args": [],
            "description": "Include performance benchmark tests",
        },
        "debug": {
            "args": ["--verbose"],
            "description": "Run with verbose output for debugging test issues",
        },
        "coverage": {
            "args": ["--verbose"],
            "description": "Run with detailed coverage reporting",
        },
        "minimal": {
            "args": ["--fast-only"],
            "description": "Minimal test run with no coverage for maximum speed",
        },
    }

    if config_name not in configurations:
        print(f"Unknown configuration: {config_name}")
        print("Available configurations:")
        for name, config in configurations.items():
            print(f"  {name}: {config['description']}")
        return False

    config = configurations[config_name]

    if config_name in {"standard", "performance", "coverage"}:
        selections = {
            "standard": "not slow and not performance",
            "performance": "performance",
            "coverage": "not slow and not performance",
        }
        return run_custom_tests(
            markers=selections[config_name],
            target_dir=target_dir,
            output_dir=output_dir,
            verbose=verbose,
            coverage=config_name == "coverage",
        )

    # Build command
    test_script = Path(__file__).resolve().parents[1] / "src" / "gnn" / "2_tests.py"
    cmd: list[Any] = [
        sys.executable,
        str(test_script),
        "--target-dir",
        target_dir,
        "--output-dir",
        output_dir,
    ]

    # Add configuration-specific args
    cmd.extend(config["args"])

    # Add verbose if requested
    if verbose and "--verbose" not in cmd:
        cmd.append("--verbose")

    print(f"🚀 Running test configuration: {config_name}")
    print(f"📋 Description: {config['description']}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print("-" * 60)

    start_time = time.time()

    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)  # nosec B603
        duration = time.time() - start_time

        if result.returncode == 0:
            print(
                f"✅ Test configuration '{config_name}' completed successfully in {duration:.1f}s"
            )
        else:
            print(
                f"⚠️ Test configuration '{config_name}' completed with issues in {duration:.1f}s (exit code: {result.returncode})"
            )

        return result.returncode == 0

    except KeyboardInterrupt:
        duration = time.time() - start_time
        print(f"\n🛑 Test execution interrupted after {duration:.1f}s")
        return False
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Test execution failed after {duration:.1f}s: {e}")
        return False


def list_configurations() -> None:
    """List all available test configurations."""
    configurations: dict[str, Any] = {
        "fast": "Step 2 fast selection",
        "standard": "Source tests outside slow and performance selections",
        "full": "Step 2 comprehensive selection",
        "performance": "Source performance selection",
        "debug": "Step 2 fast selection with verbose output",
        "coverage": "Source selection with coverage reporting",
        "minimal": "Step 2 fast selection without coverage",
    }

    print("Available test configurations:")
    print("=" * 50)
    for name, description in configurations.items():
        print(f"  {name:<12} - {description}")
    print()
    print("Usage examples:")
    print("  python tests/test_runner_helper.py fast")
    print("  python tests/test_runner_helper.py full --verbose")
    print("  python tests/test_runner_helper.py debug --target-dir custom/path")


def run_custom_tests(
    markers: Optional[str] = None, timeout: Optional[int] = None, **kwargs: Any
) -> bool:
    """Run source tests with the requested marker expression and total timeout."""
    from gnn.utils.testing.runner import TestRunner

    if timeout is not None and (isinstance(timeout, bool) or timeout <= 0):
        print("Custom timeout must be a positive number of seconds")
        return False
    if kwargs.get("target_dir", "input/gnn_files") != "input/gnn_files":
        print(
            "Custom pytest selections use repository tests; a GNN target override is unsupported"
        )
        return False
    profile = "not ollama and not env_heavy and not toolchain"
    selection = f"({markers}) and ({profile})" if markers else profile
    options = {
        "markers": [selection],
        "timeout_seconds": timeout or 300,
        "parallel": False,
        "coverage": bool(kwargs.get("coverage", False)),
        "verbose": bool(kwargs.get("verbose", False)),
    }
    output = Path(kwargs.get("output_dir") or _default_output_dir()) / "custom_tests"
    result = TestRunner(options).run_tests(
        [Path(__file__).resolve().parent],
        output,
    )
    return bool(result["success"])


def main() -> int:
    """Main entry point for test runner helper."""
    parser = argparse.ArgumentParser(
        description="GNN Test Runner Helper - Easy access to different test configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s fast                    # Quick validation tests
  %(prog)s standard --verbose     # Standard tests with verbose output  
  %(prog)s full                   # All tests including slow ones
  %(prog)s debug                  # Debug mode with verbose output
  %(prog)s list                   # List all available configurations
        """,
    )

    parser.add_argument(
        "configuration",
        nargs="?",
        help="Test configuration to run (fast, standard, full, performance, debug, coverage, minimal, list)",
    )
    parser.add_argument(
        "--target-dir",
        default="input/gnn_files",
        help="Directory containing GNN files to test (default: input/gnn_files)",
    )
    parser.add_argument(
        "--output-dir",
        default=_default_output_dir(),
        help="Output directory for test results (default: output)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--custom-markers", help="Custom pytest markers to use (advanced)"
    )
    parser.add_argument(
        "--custom-timeout", type=int, help="Custom timeout in seconds (advanced)"
    )

    args = parser.parse_args()

    if args.configuration == "list" or args.configuration is None:
        list_configurations()
        return 0

    # Handle custom configuration
    if args.custom_markers or args.custom_timeout:
        success = run_custom_tests(
            markers=args.custom_markers,
            timeout=args.custom_timeout,
            target_dir=args.target_dir,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
    else:
        # Handle predefined configurations
        success = run_test_configuration(
            args.configuration,
            target_dir=args.target_dir,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
