#!/usr/bin/env python3
"""
NumPyro Runner for Executing Rendered NumPyro POMDP Scripts

Discovers and runs NumPyro-generated scripts via subprocess with dependency
checking, syntax validation, log persistence, and execution timing.

@Web: https://num.pyro.ai/
"""

import json as json_mod
import logging
import os
import sys
import tempfile
import time as time_mod
from pathlib import Path
from typing import Any, List, Optional, Union

logger = logging.getLogger(__name__)


def is_numpyro_available() -> bool:
    """Check NumPyro in a supervised child of the current Python interpreter."""
    from gnn.utils.runtime_safety.framework_availability import check_framework

    diagnosis = check_framework("numpyro", logger=logger)
    return diagnosis.available


def find_numpyro_scripts(
    base_dir: Union[str, Path], recursive: bool = True
) -> List[Path]:
    """Find NumPyro scripts in the specified directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.warning(f"Directory not found: {base_path}")
        return []
    pattern = "**/*.py" if recursive else "*.py"
    return [
        f
        for f in base_path.glob(pattern)
        if "numpyro" in f.name.lower() or f.parent.name == "numpyro"
    ]


def execute_numpyro_script(
    script_path: Path,
    verbose: bool = False,
    output_dir: Optional[Path] = None,
    timeout: int = 300,
) -> bool:
    """Execute a single NumPyro script with log persistence.

    Args:
        script_path: Path to the NumPyro script.
        verbose: Enable verbose output.
        output_dir: Directory for execution logs.
        timeout: Execution timeout in seconds.
    """
    if not script_path.exists():
        logger.error(f"Script file not found: {script_path}")
        return False

    logger.info(f"Executing NumPyro script: {script_path}")

    from gnn.execute.preconditions import script_readiness

    deadline, readiness_error = script_readiness("numpyro", timeout)
    if readiness_error is not None:
        logger.error(
            "NumPyro readiness failed (%s): %s",
            readiness_error.get("reason_code") or readiness_error.get("error_type"),
            readiness_error.get("reason") or readiness_error.get("error"),
        )
        return False

    # Syntax validation
    try:
        content = script_path.read_text()
        compile(content, script_path.name, "exec")
        logger.debug(f"✅ Script syntax valid: {script_path.name}")
    except SyntaxError as e:
        logger.error(f"❌ Syntax error in {script_path.name}: {e}")
        return False

    env = os.environ.copy()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        env["NUMPYRO_OUTPUT_DIR"] = str(output_dir.resolve())

    # Delegate the subprocess envelope (run + timing + error normalization) to
    # the canonical safe executor. Function-local import: executor.py imports
    # this runner at module scope, so a module-level import would be circular.
    from gnn.execute.executor import execute_script_safely

    abs_path = script_path.resolve()
    envelope = execute_script_safely(
        abs_path,
        timeout=timeout,
        cwd=abs_path.parent,
        env=env,
        deadline_monotonic=deadline,
    )

    if envelope["return_code"] == -1 and "error" in envelope:
        # The script never ran (timeout / not-found / unexpected failure).
        # The pre-delegation envelope logged and returned without writing
        # per-script logs, so keep that behavior.
        logger.error(f"❌ Error executing {script_path.name}: {envelope['error']}")
        return False

    elapsed = envelope["duration_seconds"]
    success: bool = bool(envelope["success"])
    result_stdout = envelope["stdout"]
    result_stderr = envelope["stderr"]
    return_code = envelope["return_code"]

    if success:
        logger.info(f"✅ Script executed: {script_path.name} ({elapsed:.1f}s)")
        if verbose and result_stdout.strip():
            logger.debug(f"Output:\n{result_stdout}")
    else:
        logger.error(f"❌ Script failed: {script_path.name}")
        logger.error(f"Return code: {return_code}")
        if result_stderr.strip():
            logger.error(f"Error:\n{result_stderr}")

    # Save execution logs
    log_dir = output_dir if output_dir else abs_path.parent
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout.txt"
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=stdout_path.parent, delete=False
        ) as tmp_f:
            tmp_f.write(result_stdout)
        os.replace(tmp_f.name, str(stdout_path))
        stderr_path = log_dir / "stderr.txt"
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=stderr_path.parent, delete=False
        ) as tmp_f:
            tmp_f.write(result_stderr)
        os.replace(tmp_f.name, str(stderr_path))
        execution_log: dict[str, Any] = {
            "script": str(abs_path),
            "return_code": return_code,
            "success": success,
            "elapsed_seconds": round(elapsed, 2),
            "timeout": timeout,
            "timestamp": time_mod.strftime("%Y-%m-%d %H:%M:%S"),
        }
        exec_log_path = log_dir / "execution_log.json"
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=exec_log_path.parent, delete=False
        ) as tmp_f:
            tmp_f.write(json_mod.dumps(execution_log, indent=2))
        os.replace(tmp_f.name, str(exec_log_path))
        logger.debug(f"Execution logs saved to: {log_dir}")
    except Exception as log_err:
        logger.warning(f"Could not save execution logs: {log_err}")

    return success


def run_numpyro_scripts(
    rendered_simulators_dir: Union[str, Path],
    execution_output_dir: Optional[Union[str, Path]] = None,
    recursive_search: bool = True,
    verbose: bool = False,
    timeout: Optional[int] = None,
) -> bool:
    """Find and run NumPyro scripts on rendered models.

    ``timeout`` optionally overrides the per-script execution timeout;
    when omitted each script runs with its historical 300 s default.
    """

    from gnn.execute.preconditions import execution_precondition

    script_timeout = timeout if timeout is not None else 300
    if execution_precondition(script_timeout, None) is not None:
        logger.error("Invalid or exhausted script execution budget")
        return False

    if execution_output_dir:
        exec_out = Path(execution_output_dir)
        exec_out.mkdir(parents=True, exist_ok=True)
        logger.info(f"NumPyro execution outputs → {exec_out}")

    numpyro_dir = Path(rendered_simulators_dir) / "numpyro"
    logger.info(f"Looking for NumPyro scripts in: {numpyro_dir}")
    scripts = find_numpyro_scripts(numpyro_dir, recursive_search)
    if not scripts:
        logger.info("No NumPyro scripts found")
        return True

    success_count = 0
    failure_count = 0

    for script in scripts:
        out = Path(execution_output_dir) / script.stem if execution_output_dir else None
        if execute_numpyro_script(script, verbose, out, timeout=script_timeout):
            success_count += 1
        else:
            failure_count += 1

    logger.info(
        f"NumPyro execution summary: {success_count} succeeded, "
        f"{failure_count} failed, {success_count + failure_count} total"
    )
    return failure_count == 0


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(
        description="Execute NumPyro scripts generated by GNN rendering step"
    )
    parser.add_argument("--output-dir", type=Path, default="../output")
    parser.add_argument(
        "--recursive", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    ok = run_numpyro_scripts(
        rendered_simulators_dir=args.output_dir,
        recursive_search=args.recursive,
        verbose=args.verbose,
    )
    sys.exit(0 if ok else 1)
