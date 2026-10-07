#!/usr/bin/env python3
"""
PyTorch Runner for Executing Rendered PyTorch POMDP Scripts

Discovers and runs PyTorch-generated scripts via subprocess with syntax validation,
dependency checking, log persistence, and execution timing.

@Web: https://pytorch.org/docs/stable/
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


def is_pytorch_available() -> bool:
    """Check PyTorch in a supervised child of the current Python interpreter."""
    from gnn.utils.runtime_safety.framework_availability import check_framework

    diagnosis = check_framework("pytorch", logger=logger)
    return diagnosis.available


def find_pytorch_scripts(
    base_dir: Union[str, Path], recursive: bool = True
) -> List[Path]:
    """Find PyTorch scripts in the specified directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.warning(f"Directory not found: {base_path}")
        return []
    pattern = "**/*.py" if recursive else "*.py"
    return [
        f
        for f in base_path.glob(pattern)
        if "pytorch" in f.name.lower() or f.parent.name == "pytorch"
    ]


def execute_pytorch_script(
    script_path: Path,
    verbose: bool = False,
    device: Optional[str] = None,
    output_dir: Optional[Path] = None,
    timeout: int = 300,
) -> bool:
    """Execute a single PyTorch script with log persistence.

    Args:
        script_path: Path to the PyTorch script to execute.
        verbose: Enable verbose output logging.
        device: Device hint ('cpu' or 'cuda').
        output_dir: Directory for execution logs.
        timeout: Execution timeout in seconds.
    """
    if not script_path.exists():
        logger.error(f"Script file not found: {script_path}")
        return False

    logger.info(f"Executing PyTorch script: {script_path}")

    from gnn.execute.preconditions import script_readiness

    deadline, readiness_error = script_readiness("pytorch", timeout)
    if readiness_error is not None:
        logger.error(
            "PyTorch readiness failed (%s): %s",
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
    if device:
        env["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else "0"
        logger.info(f"Using device: {device}")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        env["PYTORCH_OUTPUT_DIR"] = str(output_dir.resolve())

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
            logger.debug(f"Output:\n{result_stdout}")
    else:
        logger.error(f"❌ Script execution failed: {script_path.name}")
        logger.error(f"Return code: {return_code}")
        if result_stderr.strip():
            logger.error(f"Error output:\n{result_stderr}")

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
            "device": device or "default",
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


def run_pytorch_scripts(
    rendered_simulators_dir: Union[str, Path],
    execution_output_dir: Optional[Union[str, Path]] = None,
    recursive_search: bool = True,
    verbose: bool = False,
    device: Optional[str] = None,
    timeout: Optional[int] = None,
) -> bool:
    """Find and run PyTorch scripts on rendered models.

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
        logger.info(f"PyTorch execution outputs → {exec_out}")

    pytorch_dir = Path(rendered_simulators_dir) / "pytorch"
    logger.info(f"Looking for PyTorch scripts in: {pytorch_dir}")
    scripts = find_pytorch_scripts(pytorch_dir, recursive_search)
    if not scripts:
        logger.info("No PyTorch scripts found")
        return True

    success_count = 0
    failure_count = 0

    for script in scripts:
        out = Path(execution_output_dir) / script.stem if execution_output_dir else None
        if execute_pytorch_script(script, verbose, device, out, timeout=script_timeout):
            success_count += 1
        else:
            failure_count += 1

    logger.info(
        f"PyTorch execution summary: {success_count} succeeded, "
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
        description="Execute PyTorch scripts generated by GNN rendering step"
    )
    parser.add_argument("--output-dir", type=Path, default="../output")
    parser.add_argument(
        "--recursive", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    ok = run_pytorch_scripts(
        rendered_simulators_dir=args.output_dir,
        recursive_search=args.recursive,
        verbose=args.verbose,
        device=args.device,
    )
    sys.exit(0 if ok else 1)
