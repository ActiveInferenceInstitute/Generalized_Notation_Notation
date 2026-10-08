"""
JAX Runner for Executing Rendered JAX POMDP Scripts

Discovers and runs JAX-generated scripts, manages device selection, logs hardware/software info, and benchmarks performance.

@Web: https://github.com/google/jax
@Web: https://optax.readthedocs.io
@Web: https://flax.readthedocs.io
@Web: https://pfjax.readthedocs.io
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Union

logger = logging.getLogger(__name__)


def initialize_jax_devices() -> List[Any]:
    """Initialize and return real JAX devices."""
    try:
        import jax

        return list(jax.devices())
    except Exception as exc:
        logger.warning("JAX devices unavailable: %s", exc)
        return []


def is_jax_available() -> bool:
    """Check JAX in a supervised child of the current Python interpreter."""
    from gnn.utils.runtime_safety.framework_availability import check_framework

    diagnosis = check_framework("jax", logger=logger)
    return diagnosis.available


def find_jax_scripts(base_dir: Union[str, Path], recursive: bool = True) -> List[Path]:
    """Find JAX scripts in the specified directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.warning(f"Directory not found: {base_path}")
        return []
    pattern = "**/*.py" if recursive else "*.py"
    return [
        f
        for f in base_path.glob(pattern)
        if "jax" in f.name.lower() or f.parent.name == "jax"
    ]


def execute_jax_script(
    script_path: Path,
    verbose: bool = False,
    device: Optional[str] = None,
    output_dir: Optional[Path] = None,
    timeout: int = 300,
) -> bool:
    """Execute a single JAX script with enhanced dependency checking, error handling, and log persistence.

    Args:
        script_path: Path to the JAX script to execute
        verbose: Enable verbose output logging
        device: JAX device to use ('cpu', 'gpu', 'tpu')
        output_dir: Directory for execution logs (stdout.txt, stderr.txt, execution_log.json)
        timeout: Execution timeout in seconds (default: 300)

    Returns:
        True if script executed successfully, False otherwise
    """
    import json as json_mod
    import time as time_mod

    if not script_path.exists():
        logger.error(f"Script file not found: {script_path}")
        return False

    logger.info(f"Executing JAX script: {script_path}")

    from gnn.execute.preconditions import script_readiness

    deadline, readiness_error = script_readiness("jax", timeout)
    if readiness_error is not None:
        logger.error(
            "JAX readiness failed (%s): %s",
            readiness_error.get("reason_code") or readiness_error.get("error_type"),
            readiness_error.get("reason") or readiness_error.get("error"),
        )
        return False

    # Validate script syntax
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
            compile(content, script_path.name, "exec")
        logger.debug(f"✅ Script syntax valid: {script_path.name}")
    except SyntaxError as e:
        logger.error(f"❌ Syntax error in {script_path.name}: {e}")
        return False

    # Delegate the subprocess envelope (run + timing + error normalization) to
    # the canonical safe executor. Function-local import: executor.py imports
    # this runner at module scope, so a module-level import would be circular.
    from gnn.execute.executor import execute_script_safely

    env = os.environ.copy()
    if device:
        env["JAX_PLATFORM_NAME"] = device
        logger.info(f"Using JAX device: {device}")

    # Generated scripts from earlier pipeline versions and the current version
    # receive the same absolute path.
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = str(output_dir.resolve())
        env["JAX_OUTPUT_DIR"] = destination
        env["GNN_OUTPUT_DIR"] = destination
        logger.debug("Set JAX_OUTPUT_DIR and GNN_OUTPUT_DIR=%s", destination)

    abs_script_path = script_path.resolve()
    envelope = execute_script_safely(
        abs_script_path,
        timeout=timeout,
        cwd=abs_script_path.parent,
        env=env,
        deadline_monotonic=deadline,
    )

    if envelope["return_code"] == -1 and "error" in envelope:
        # The script never ran (timeout / not-found / unexpected failure).
        # The pre-delegation envelope logged and returned without writing
        # per-script logs, so keep that behavior.
        logger.error(
            f"❌ Error executing script {script_path.name}: {envelope['error']}"
        )
        return False

    elapsed = envelope["duration_seconds"]
    success: bool = bool(envelope["success"])
    result_stdout = envelope["stdout"]
    result_stderr = envelope["stderr"]
    return_code = envelope["return_code"]

    if success:
        logger.info(
            f"✅ Script executed successfully: {script_path.name} ({elapsed:.1f}s)"
        )
        if verbose and result_stdout.strip():
            logger.debug(f"Output from {script_path.name}:\n{result_stdout}")
    else:
        logger.error(f"❌ Script execution failed: {script_path.name}")
        logger.error(f"Return code: {return_code}")
        if result_stderr.strip():
            logger.error(f"Error output:\n{result_stderr}")
        if result_stdout.strip():
            logger.debug(f"Standard output:\n{result_stdout}")

    # Save execution logs (E-1/J-1)
    log_dir = output_dir if output_dir else abs_script_path.parent
    try:
        log_dir.mkdir(parents=True, exist_ok=True)

        # Save stdout and stderr
        stdout_file = log_dir / "stdout.txt"
        with open(stdout_file, "w", encoding="utf-8") as f:
            f.write(result_stdout)

        stderr_file = log_dir / "stderr.txt"
        with open(stderr_file, "w", encoding="utf-8") as f:
            f.write(result_stderr)

        # Save execution log JSON
        execution_log: dict[str, Any] = {
            "script": str(abs_script_path),
            "return_code": return_code,
            "success": success,
            "elapsed_seconds": round(elapsed, 2),
            "device": device or "default",
            "timeout": timeout,
            "timestamp": time_mod.strftime("%Y-%m-%d %H:%M:%S"),
        }

        log_file = log_dir / "execution_log.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json_mod.dump(execution_log, f, indent=2)

        logger.debug(f"Execution logs saved to: {log_dir}")
    except Exception as log_err:
        logger.warning(f"Could not save execution logs: {log_err}")

    return success


def run_jax_scripts(
    rendered_simulators_dir: Union[str, Path],
    execution_output_dir: Optional[Union[str, Path]] = None,
    recursive_search: bool = True,
    verbose: bool = False,
    device: Optional[str] = None,
    timeout: Optional[int] = None,
) -> bool:
    """Find and run JAX scripts on rendered models.

    ``timeout`` optionally overrides the per-script execution timeout;
    when omitted each script runs with its historical 300 s default.
    """

    from gnn.execute.preconditions import execution_precondition

    script_timeout = timeout if timeout is not None else 300
    if execution_precondition(script_timeout, None) is not None:
        logger.error("Invalid or exhausted script execution budget")
        return False

    # Set up execution output directory
    if execution_output_dir:
        exec_output_dir = Path(execution_output_dir)
        exec_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"JAX execution outputs will be saved to: {exec_output_dir}")

    jax_dir = Path(rendered_simulators_dir) / "jax"
    logger.info(f"Looking for JAX scripts in: {jax_dir}")
    script_files = find_jax_scripts(jax_dir, recursive_search)
    if not script_files:
        logger.info("No JAX scripts found")
        return True  # Consider this success if no scripts to run
    success_count = 0
    failure_count = 0

    for script_file in script_files:
        script_output = (
            Path(execution_output_dir) / script_file.stem
            if execution_output_dir
            else None
        )
        if execute_jax_script(
            script_file, verbose, device, script_output, timeout=script_timeout
        ):
            success_count += 1
        else:
            failure_count += 1
    logger.info(
        f"JAX script execution summary: {success_count} succeeded, {failure_count} failed, {success_count + failure_count} total"
    )
    # Return True only if no failures occurred (success means zero failures)
    return failure_count == 0


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(
        description="Execute JAX scripts generated by the GNN rendering step"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="../output",
        help="Main pipeline output directory",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively search for scripts in the output directory",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "tpu"],
        default=None,
        help="Device to run JAX scripts on",
    )
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    success = run_jax_scripts(
        rendered_simulators_dir=args.output_dir,
        recursive_search=args.recursive,
        verbose=args.verbose,
        device=args.device,
    )
    sys.exit(0 if success else 1)
