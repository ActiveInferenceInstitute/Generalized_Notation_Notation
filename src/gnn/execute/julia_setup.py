#!/usr/bin/env python3
"""
Julia Environment Setup for GNN Pipeline

This module provides automated Julia environment setup and package installation
for RxInfer.jl and ActiveInference.jl frameworks used in the execution step.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from gnn.execute.julia_env import julia_subprocess_env
from gnn.execute.subprocess_envelope import (  # nosec B404
    NEVER_STARTED,
    run_subprocess_envelope,
)

# The probe is owned by the runtime-safety leaf; this module re-exports the
# same function object so ``julia_setup.julia_executable`` stays importable.
from gnn.utils.runtime_safety.julia_probe import julia_executable

logger = logging.getLogger(__name__)


def julia_available() -> bool:
    """Whether a ``julia`` executable is available on PATH."""
    return julia_executable() is not None


def check_julia_availability() -> tuple[bool, Optional[str]]:
    """
    Check if Julia is available and return its path.

    Returns:
        Tuple of (is_available, julia_path)
    """
    julia_path = julia_executable()
    if julia_path:
        logger.info(f"✅ Julia found at: {julia_path}")
        return True, julia_path
    else:
        logger.warning("❌ Julia not found in PATH")
        return False, None


def check_julia_version(julia_path: str) -> Optional[str]:
    """
    Get Julia version.

    Args:
        julia_path: Path to Julia executable

    Returns:
        Version string or None if check fails
    """
    try:
        result = subprocess.run(  # nosec B603
            [julia_path, "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version_line = result.stdout.strip().split("\n")[0]
            logger.info(f"Julia version: {version_line}")
            return version_line
        else:
            logger.warning(f"Failed to get Julia version: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"Error checking Julia version: {e}")
        return None


def is_julia_available(min_version: tuple = (1, 6, 0)) -> bool:
    """
    Return True if Julia is installed and meets the minimum version requirement.

    Args:
        min_version: Minimum (major, minor, patch) tuple. Defaults to (1, 6, 0).
    """
    import re

    available, julia_path = check_julia_availability()
    if not available or julia_path is None:
        return False

    version_str = check_julia_version(julia_path)
    if version_str:
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_str)
        if match:
            version = tuple(int(match.group(i)) for i in (1, 2, 3))
            if version < min_version:
                logger.warning(
                    "Julia %d.%d.%d is below minimum %d.%d.%d", *version, *min_version
                )
                return False
    return True


def run_julia_setup_script(
    julia_path: str,
    setup_script: Path,
    verbose: bool = False,
    force_reinstall: bool = False,
    validate_only: bool = False,
) -> bool:
    """
    Run the Julia setup script.

    Args:
        julia_path: Path to Julia executable
        setup_script: Path to setup_environment.jl script
        verbose: Enable verbose output
        force_reinstall: Force reinstall of packages
        validate_only: Only validate, don't install

    Returns:
        True if successful
    """
    if not setup_script.exists():
        logger.error(f"Julia setup script not found: {setup_script}")
        return False

    logger.info(f"Running Julia setup script: {setup_script}")

    cmd: list[Any] = [julia_path, str(setup_script)]

    if verbose:
        cmd.append("--verbose")
    if force_reinstall:
        cmd.append("--force-reinstall")
    if validate_only:
        cmd.append("--validate-only")

    # Canonical subprocess envelope (MAJ-10): timeout / OSError / non-zero
    # exit all arrive in one structured outcome.
    outcome = run_subprocess_envelope(
        cmd,
        cwd=setup_script.parent,
        timeout=300,  # 5 minute timeout for package installation
        env=julia_subprocess_env(),
    )

    if outcome["stdout"]:
        logger.info(f"Julia setup stdout:\n{outcome['stdout']}")
    if outcome["stderr"]:
        logger.warning(f"Julia setup stderr:\n{outcome['stderr']}")

    if outcome["success"]:
        logger.info("✅ Julia setup completed successfully")
        return True
    if outcome.get("error_type") == "TimeoutExpired":
        logger.error("⏰ Julia setup timed out after 5 minutes")
    elif outcome["return_code"] != NEVER_STARTED:
        logger.error(
            f"❌ Julia setup failed with return code: {outcome['return_code']}"
        )
    else:
        logger.error(f"❌ Julia setup failed: {outcome.get('error', 'unknown error')}")
    return False


def setup_julia_environment(
    verbose: bool = False,
    force_reinstall: bool = False,
    validate_only: bool = False,
    frameworks: (List[str]) | None = None,
) -> Dict[str, Any]:
    """
    Setup Julia environment for specified frameworks.

    Args:
        verbose: Enable verbose output
        force_reinstall: Force reinstall of packages
        validate_only: Only validate, don't install
        frameworks: List of frameworks to setup (None = all)

    Returns:
        Dictionary with setup results
    """
    logger.info("🔧 Setting up Julia environment for GNN execution")

    # Check Julia availability
    julia_available, julia_path = check_julia_availability()
    if not julia_available or julia_path is None:
        return {
            "success": False,
            "julia_available": False,
            "error": "Julia not found in PATH",
            "suggestion": "Install Julia from https://julialang.org/downloads/",
        }

    # Check Julia version
    version = check_julia_version(julia_path)
    if not version:
        return {
            "success": False,
            "julia_available": True,
            "error": "Failed to determine Julia version",
            "julia_path": julia_path,
        }

    # Define framework setup scripts
    framework_scripts: dict[str, Any] = {
        "rxinfer": "src/gnn/execute/rxinfer/setup_environment.jl",
        "activeinference_jl": "src/gnn/execute/activeinference_jl/setup_environment.jl",
    }

    # Filter frameworks if specified
    if frameworks:
        framework_scripts = {
            k: v for k, v in framework_scripts.items() if k in frameworks
        }

    setup_results: dict[Any, Any] = {}
    overall_success = True

    for framework, script_path in framework_scripts.items():
        script_file = Path(script_path)
        if not script_file.exists():
            logger.warning(f"Setup script not found for {framework}: {script_file}")
            setup_results[framework] = {
                "success": False,
                "error": f"Setup script not found: {script_file}",
                "skipped": True,
            }
            continue

        logger.info(f"Setting up {framework}...")

        success = run_julia_setup_script(
            julia_path,
            script_file,
            verbose=verbose,
            force_reinstall=force_reinstall,
            validate_only=validate_only,
        )

        setup_results[framework] = {
            "success": success,
            "script_path": str(script_file),
            "julia_path": julia_path,
            "version": version,
        }

        if not success:
            overall_success = False

    return {
        "success": overall_success,
        "julia_available": True,
        "julia_path": julia_path,
        "julia_version": version,
        "frameworks_setup": setup_results,
    }


def main() -> int:
    """Main entry point for Julia setup."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup Julia environment for GNN execution"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--force-reinstall", action="store_true", help="Force reinstall of packages"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate, don't install"
    )
    parser.add_argument(
        "--frameworks",
        type=str,
        help="Comma-separated list of frameworks to setup (rxinfer,activeinference_jl)",
    )

    args = parser.parse_args()

    # Parse frameworks
    frameworks = None
    if args.frameworks:
        frameworks = [f.strip() for f in args.frameworks.split(",")]

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Run setup
    results = setup_julia_environment(
        verbose=args.verbose,
        force_reinstall=args.force_reinstall,
        validate_only=args.validate_only,
        frameworks=frameworks,
    )

    # Print results
    print(f"\n{'=' * 60}")
    print("JULIA SETUP RESULTS")
    print(f"{'=' * 60}")

    if results["julia_available"]:
        print(f"✅ Julia found: {results['julia_path']}")
        print(f"📦 Version: {results['julia_version']}")
    else:
        print("❌ Julia not found")
        print("💡 Install Julia from: https://julialang.org/downloads/")

    print("\nFramework Setup Results:")
    for framework, result in results["frameworks_setup"].items():
        if result.get("skipped"):
            print(f"⏭️  {framework}: Skipped ({result['error']})")
        elif result["success"]:
            print(f"✅ {framework}: Setup completed")
        else:
            print(f"❌ {framework}: Setup failed")

    print(f"\nOverall Success: {'✅ Yes' if results['success'] else '❌ No'}")

    # Exit with appropriate code
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
