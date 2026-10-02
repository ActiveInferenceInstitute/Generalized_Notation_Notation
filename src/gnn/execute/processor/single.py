#!/usr/bin/env python3
"""Single-script execution: subprocess envelopes, environment, receipts."""

import hashlib
import json
import logging
import os
import platform
import subprocess  # nosec B404
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast
from uuid import uuid4

from gnn.execute.data_extractors import (
    collect_execution_outputs,
)
from gnn.execute.data_extractors import (
    extract_simulation_data as _extract_simulation_data,
)
from gnn.execute.data_extractors import (
    extract_simulation_data_from_files as _extract_simulation_data_from_files,
)
from gnn.execute.detection import (
    _build_script_execution_context,
    _detect_accelerator_type,
)
from gnn.execute.julia_env import (
    GKSWSTYPE_HEADLESS,
    GKSWSTYPE_VAR,
    _build_script_execution_command,
    _julia_project_for_framework,
)
from gnn.execute.metadata import _load_rxinfer_execution_metadata_from_script
from gnn.execute.processor.envelope import (
    _base_execution_envelope,
    _make_skipped_result,
)
from gnn.execute.security_gate import check_script_allowed
from gnn.execute.subprocess_envelope import (
    INTERNAL_ERROR,
    NEVER_STARTED,
    UNKNOWN_STATE,
    run_subprocess_envelope,
)
from gnn.execute.types import ExecutionFrameworkName, ScriptExecutionContext
from gnn.utils.runtime_safety.framework_availability import (
    FRAMEWORK_IMPORT_CHECK,
    FRAMEWORK_JULIA_PACKAGES,
)

logger = logging.getLogger(__name__)


def _aggregate_benchmark_samples(samples: List[float]) -> Dict[str, Any]:
    """Aggregate repeated execution durations (median + population std)."""
    import statistics

    if not samples:
        return {}
    med = float(statistics.median(samples))
    mean = float(statistics.mean(samples))
    std = float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0
    return {
        "execution_time": med,
        "execution_time_mean": mean,
        "execution_time_std": std,
        "execution_time_samples": list(samples),
    }


def _new_execution_result(context: ScriptExecutionContext) -> Dict[str, Any]:
    """Create the standard execution result envelope."""
    return _base_execution_envelope(
        script_path=str(context.script_path),
        script_name=context.script_name,
        framework=context.framework,
        model_name=context.model_name,
        executor=context.executor,
        status="failed",
        skipped=False,
    )


def _build_execution_environment(
    context: ScriptExecutionContext,
    results_dir: Path,
) -> Dict[str, str]:
    """Build environment variables for a rendered-script subprocess."""
    env = os.environ.copy()
    # Julia frameworks: default JULIA_PROJECT to the committed framework
    # environment so `using GnnRxInferModels` / `using ActiveInference`
    # resolve without an ambient env (e.g. a /tmp test env that may not
    # exist). An explicitly set JULIA_PROJECT still wins.
    julia_project = _julia_project_for_framework(context.framework)
    if julia_project is not None:
        env.setdefault("JULIA_PROJECT", str(julia_project))
    if context.executor == "julia":
        # Headless GR default (see julia_env.julia_subprocess_env): rendered
        # Julia scripts may plot via Plots.jl/GR, whose gksqt Qt window
        # hangs display-less hosts. An explicitly set GKSwstype still wins.
        env.setdefault(GKSWSTYPE_VAR, GKSWSTYPE_HEADLESS)
    if context.framework == "pymdp":
        env["PYTHONPATH"] = (
            str(context.script_path.parent) + os.pathsep + env.get("PYTHONPATH", "")
        )
        proc_path = Path(__file__).resolve()
        repo_root = proc_path.parent.parent.parent.parent
        env["GNN_PROJECT_ROOT"] = str(repo_root)
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        jax_platform = os.environ.get("GNN_JAX_PLATFORM")
        if jax_platform and str(jax_platform).strip():
            env["JAX_PLATFORM_NAME"] = str(jax_platform).strip()

    simulation_data_dir = (
        results_dir / context.model_name / context.framework / "simulation_data"
    )
    output_env_vars = {
        "jax": "GNN_OUTPUT_DIR",
        "numpyro": "NUMPYRO_OUTPUT_DIR",
        "pytorch": "PYTORCH_OUTPUT_DIR",
        "ngclearn": "NGCLEARN_OUTPUT_DIR",
        "cpomdp": "CPOMDP_OUTPUT_DIR",
        "stan": "STAN_OUTPUT_DIR",
        "bnlearn": "BNLEARN_OUTPUT_DIR",
    }
    if context.framework in output_env_vars:
        simulation_data_dir.mkdir(parents=True, exist_ok=True)
        env[output_env_vars[context.framework]] = str(simulation_data_dir)

    if context.framework == "thrml":
        implementation_dir = results_dir / context.model_name / context.framework
        implementation_dir.mkdir(parents=True, exist_ok=True)
        env["THRML_OUTPUT_DIR"] = str(implementation_dir)
        env["GNN_THRML_EXECUTION_ID"] = uuid4().hex

    return env


def _framework_for_data_helpers(framework: str) -> ExecutionFrameworkName:
    """Narrow framework names for typed data-collection helper calls."""
    return cast(ExecutionFrameworkName, framework)


_GNN_ALLOW_MISSING_DEPS = "GNN_ALLOW_MISSING_DEPS"


def _gnn_allow_missing_deps() -> bool:
    """Whether to continue past a missing PyMDP dependency (SC-34, opt-in)."""
    return os.environ.get(_GNN_ALLOW_MISSING_DEPS, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _sandbox_mode() -> str:
    """Effective sandbox mode from ``GNN_SANDBOX`` (default ``off``)."""
    from gnn.execute.sandbox import SANDBOX_MODES

    mode = os.environ.get("GNN_SANDBOX", "off").strip().lower()
    return mode if mode in SANDBOX_MODES else "off"


def _sandbox_command_prefix(mode: str) -> tuple[list[str], Optional[str]]:
    """Return ``(prefix, blocked_reason)`` for the requested sandbox mode.

    ``blocked_reason`` is non-None only when ``require`` cannot find a backend.
    """
    if mode == "off":
        return [], None
    from gnn.execute.sandbox import detect_sandbox

    spec = detect_sandbox()
    if spec is None:
        if mode == "require":
            return [], (
                "GNN_SANDBOX=require but no sandbox backend "
                "(firejail/bwrap/nsjail) is installed"
            )
        logger.warning(
            "GNN_SANDBOX=%s but no sandbox backend found; running unsandboxed",
            mode,
        )
        return [], None
    return list(spec.prefix), None


def _backend_version_from_runner_metadata(
    runner_metadata: Dict[str, Any],
) -> Optional[str]:
    """Best-effort backend version string from runner-provided metadata.

    Runners stamp their own version keys into ``execution_metadata`` (the
    ``pymdp_version`` rollout receipt, the Julia ``julia_version`` probe);
    mirror the first known key as ``backend_version`` for the structured
    receipt, falling back to any other ``*_version`` string the runner
    provided, else ``None``.
    """
    for key in ("pymdp_version", "julia_version", "rxinfer_version"):
        value = runner_metadata.get(key)
        if isinstance(value, str) and value:
            return value
    for key in sorted(runner_metadata):
        value = runner_metadata[key]
        if key.endswith("_version") and isinstance(value, str) and value:
            return value
    return None


def execute_single_script(
    script_info: Dict[str, Any],
    results_dir: Path,
    verbose: bool,
    logger: Any,
    timeout: int = 3600,
    *,
    execution_benchmark_repeats: int = 1,
) -> Dict[str, Any]:
    """
    Execute a single script using subprocess.

    Args:
        script_info: Dictionary containing script information
        results_dir: Directory to save execution results (will create implementation-specific subfolders)
        verbose: Enable verbose logging
        logger: Logger instance

    Returns:
        Dictionary with execution results
    """
    context = _build_script_execution_context(script_info)
    script_path = context.script_path
    executor = context.executor
    model_name = context.model_name
    framework = context.framework
    from gnn.execute.preconditions import execution_precondition, script_deadline

    precondition = execution_precondition(timeout, None)
    if precondition is not None:
        return {**_new_execution_result(context), **precondition}
    deadline = script_deadline(timeout)
    from gnn.execute import processor as _processor_facade

    # Probe the actual interpreter and committed environment before dispatch.
    if framework in FRAMEWORK_IMPORT_CHECK or framework in FRAMEWORK_JULIA_PACKAGES:
        dependency_status = _processor_facade._check_framework_by_name(
            framework,
            executor=executor,
            logger=logger,
            deadline_monotonic=deadline,
        )
        if not dependency_status.available:
            return _make_skipped_result(
                script_info, framework, model_name, executor, logger, dependency_status
            )

    # Prepare execution result
    exec_result = _new_execution_result(context)

    # Pre-execution security gate (RED_TEAM_REVIEW V-01/V-06): scan rendered
    # code BEFORE running it. Shared helper (SC-1) also gates the GNNExecutor
    # / MCP path and fails CLOSED when the scanner cannot be imported (SC-2a);
    # GNN_ALLOW_UNSAFE_EXEC remains the explicit operator opt-out at both sites.
    gate_verdict = check_script_allowed(script_path)
    if gate_verdict["overridden"]:
        logger.warning(
            "GNN_ALLOW_UNSAFE_EXEC set: pre-execution security gate bypassed "
            "for %s (trusted-local use only)",
            script_info["name"],
        )
    if not gate_verdict["ok"]:
        exec_result["error"] = (
            f"Pre-execution security gate blocked {script_info['name']}: "
            f"{gate_verdict['reason']}"
        )
        exec_result["error_type"] = gate_verdict.get(
            "error_type", "SecurityGateBlocked"
        )
        exec_result["security_findings"] = gate_verdict["blocked"]
        logger.error(exec_result["error"])
        return exec_result

    if framework == "rxinfer":
        exec_result["execution_metadata"] = (
            _load_rxinfer_execution_metadata_from_script(script_path)
        )

    try:
        if verbose:
            logger.info(
                f"Executing {script_info['framework']} script: {script_info['name']}"
            )

        # Check if the executor is available
        try:
            precondition = execution_precondition(
                timeout, None, deadline_monotonic=deadline
            )
            if precondition is not None:
                return {**exec_result, **precondition}
            probe_timeout = min(5.0, max(0.001, deadline - time.monotonic()))
            # For Python scripts, check if Python is available (most are Python scripts)
            if executor in ["python", "python3"]:
                subprocess.run(
                    [executor, "--version"],  # nosec B603
                    capture_output=True,
                    text=True,
                    timeout=probe_timeout,
                    check=True,
                )

                # For PyMDP, specifically check if it's importable
                if framework == "pymdp":
                    try:
                        import_check = subprocess.run(  # nosec B603
                            [executor, "-c", 'import pymdp; print("ok")'],
                            capture_output=True,
                            text=True,
                            timeout=min(5.0, max(0.001, deadline - time.monotonic())),
                        )
                        if import_check.returncode != 0:
                            logger.warning(
                                f"PyMDP package appears missing or broken: {import_check.stderr}"
                            )
                            exec_result["error"] = (
                                f"PyMDP dependency missing: {import_check.stderr}"
                            )
                            if not _gnn_allow_missing_deps():
                                # SC-34: continuation is opt-in. Default is a
                                # structured fast failure instead of a doomed run.
                                exec_result["error_type"] = "DependencyMissing"
                                exec_result["return_code"] = NEVER_STARTED
                                logger.error(
                                    f"Failing fast for {script_info['name']}: PyMDP "
                                    "unavailable (set GNN_ALLOW_MISSING_DEPS=1 to "
                                    "attempt continuation)"
                                )
                                return exec_result
                            # Opt-in continuation: might still be a local import.
                    except Exception as e:
                        logger.debug(f"Error checking PyMDP importability: {e}")

            elif framework in FRAMEWORK_JULIA_PACKAGES:
                # The shared preflight already checked this committed project.
                pass
            # For other executors, try a basic check
            else:
                subprocess.run(
                    [executor, "--version"],  # nosec B603
                    capture_output=True,
                    text=True,
                    timeout=probe_timeout,
                    check=True,
                )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ) as e:
            exec_result["error"] = (
                f"Executor '{executor}' is not available or not working: {e}"
            )
            exec_result["error_type"] = "ExecutorUnavailable"
            exec_result["return_code"] = NEVER_STARTED
            logger.warning(
                f"Executor unavailable for {script_info['name']}: {executor}"
            )
            return exec_result

        # Execute the script with improved error handling. Failure carriers use
        # ``subprocess.CompletedProcess`` so the success and failure paths share
        # one return-code/stdout/stderr shape (no per-call class needed).
        result: subprocess.CompletedProcess[str] | None = None

        K = max(1, int(execution_benchmark_repeats))
        exec_result["execution_benchmark_repeats"] = K

        durations_success: List[float] = []
        broke_early = False

        try:
            env = _processor_facade._build_execution_environment(context, results_dir)

            sandbox_mode = _sandbox_mode()
            sandbox_prefix, sandbox_blocked = _sandbox_command_prefix(sandbox_mode)
            if sandbox_blocked is not None:
                exec_result["error"] = sandbox_blocked
                exec_result["error_type"] = "SandboxUnavailable"
                logger.error(sandbox_blocked)
                return exec_result
            # SC-2b: running unsandboxed is a record, not a whisper — emit a
            # pipeline-visible receipt and carry the mode in execution metadata.
            if sandbox_mode == "off":
                logger.warning(
                    "sandbox_disabled_receipt: script %s executed WITHOUT a "
                    "sandbox (GNN_SANDBOX=off, the default). Rendered scripts "
                    "run with operator privileges; set GNN_SANDBOX=prefer/require "
                    "for isolation.",
                    script_info["name"],
                    extra={
                        "event": "sandbox_disabled_receipt",
                        "sandbox_mode": sandbox_mode,
                        "script": script_info["name"],
                    },
                )
            else:
                logger.warning(
                    "sandbox_active_receipt: script %s will run under sandbox "
                    "isolation (GNN_SANDBOX=%s)",
                    script_info["name"],
                    sandbox_mode,
                    extra={
                        "event": "sandbox_active_receipt",
                        "sandbox_mode": sandbox_mode,
                    },
                )
            exec_result["execution_metadata"] = dict(
                exec_result.get("execution_metadata") or {}
            )
            exec_result["execution_metadata"]["sandbox_mode"] = sandbox_mode
            exec_result["execution_metadata"]["sandboxed"] = bool(sandbox_prefix)
            base_command = _build_script_execution_command(context, sandbox_prefix)

            for rep in range(K):
                if framework == "thrml":
                    env["GNN_THRML_EXECUTION_ID"] = uuid4().hex
                exec_result["attempts_started"] = rep + 1
                rep_start = datetime.now()
                # Canonical subprocess envelope (MAJ-10): one structured
                # outcome for timeout / OSError / non-zero exit, with the
                # partial stdout/stderr subprocess captured before a timeout
                # preserved for debugging.
                envelope = run_subprocess_envelope(
                    base_command,
                    timeout=timeout,
                    deadline_monotonic=deadline,
                    cwd=script_path.parent,
                    env=env,
                    sandbox=False,
                )
                elapsed_rep = (datetime.now() - rep_start).total_seconds()
                # Surface the envelope's measurement keys into the per-script
                # result (previously dropped on this path): peak child RSS is
                # a max over reps; sample counts accumulate; cancel is sticky.
                rss_peak_rep = envelope.get("child_peak_rss_mb")
                if rss_peak_rep is not None:
                    prior_peak = exec_result.get("child_peak_rss_mb")
                    exec_result["child_peak_rss_mb"] = (
                        rss_peak_rep
                        if prior_peak is None
                        else max(prior_peak, rss_peak_rep)
                    )
                    exec_result["rss_sample_interval_seconds"] = envelope.get(
                        "rss_sample_interval_seconds"
                    )
                    exec_result["rss_samples_count"] = (
                        exec_result.get("rss_samples_count") or 0
                    ) + (envelope.get("rss_samples_count") or 0)
                exec_result["cancelled"] = bool(
                    exec_result.get("cancelled", False) or envelope.get("cancelled")
                )

                for receipt_key in (
                    "containment",
                    "observed_descendant_count",
                    "cleanup_verified",
                    "streams_drained",
                    "cleanup_timeout_seconds",
                    "cleanup_error",
                    "execution_error_type",
                ):
                    if receipt_key in envelope:
                        exec_result[receipt_key] = envelope[receipt_key]

                if envelope.get("error_type") == "TimeoutExpired":
                    exec_result["execution_time"] = elapsed_rep
                    exec_result["error"] = (
                        f"Script execution timed out after {timeout} seconds"
                    )
                    exec_result["error_type"] = "TimeoutExpired"
                    exec_result["return_code"] = NEVER_STARTED
                    exec_result["stdout"] = envelope["stdout"]
                    exec_result["stderr"] = envelope["stderr"]
                    logger.warning(
                        f"⏰ Script {script_info['name']} timed out after {timeout} seconds "
                        f"(rep {rep + 1}/{K})"
                    )
                    result = subprocess.CompletedProcess(
                        args=base_command,
                        returncode=NEVER_STARTED,
                        stdout=envelope["stdout"],
                        stderr=envelope["stderr"],
                    )
                    broke_early = True
                    break

                run_result = subprocess.CompletedProcess(
                    args=base_command,
                    returncode=envelope["return_code"],
                    stdout=envelope["stdout"],
                    stderr=envelope["stderr"],
                )

                if not envelope["success"]:
                    exec_result["execution_time"] = elapsed_rep
                    exec_result["return_code"] = run_result.returncode
                    exec_result["stdout"] = run_result.stdout
                    exec_result["stderr"] = run_result.stderr
                    if run_result.returncode == NEVER_STARTED:
                        exec_result["error"] = (
                            f"Script could not be started: {envelope.get('error', 'unknown error')}"
                        )
                        exec_result["error_type"] = (
                            envelope.get("error_type") or "ExecutorUnavailable"
                        )
                        logger.error(
                            f"Script {script_info['name']} could not be started: "
                            f"{envelope.get('error', 'unknown error')}"
                        )
                    else:
                        exec_result["error"] = (
                            f"Script failed with return code {run_result.returncode}"
                        )
                        if "ModuleNotFoundError" in run_result.stderr:
                            exec_result["error_type"] = "DependencyError"
                            logger.error(
                                f"Missing dependency in {script_info['name']}: "
                                f"{run_result.stderr.splitlines()[-1]}"
                            )
                        elif "SyntaxError" in run_result.stderr:
                            exec_result["error_type"] = "SyntaxError"
                            logger.error(f"Syntax error in {script_info['name']}")
                        else:
                            exec_result["error_type"] = "RuntimeError"

                    if envelope.get("error_type") == "ProcessCleanupFailure":
                        exec_result["error_type"] = "ProcessCleanupFailure"
                        exec_result["error"] = envelope.get(
                            "error", "Subprocess cleanup could not be verified"
                        )

                    logger.warning(
                        f"⚠️ Script {script_info['name']} failed with return code "
                        f"{run_result.returncode} (rep {rep + 1}/{K})"
                    )
                    if run_result.stderr:
                        logger.warning(f"Error output: {run_result.stderr[:500]}...")
                    result = run_result
                    broke_early = True
                    break

                result = run_result
                if framework == "thrml":
                    from gnn.execute.thrml import validate_native_result

                    artifact = (
                        Path(env["THRML_OUTPUT_DIR"])
                        / "simulation_data"
                        / "simulation_results.json"
                    )
                    if (
                        artifact.is_symlink()
                        or not artifact.is_file()
                        or artifact.stat().st_size > 16 * 1024 * 1024
                    ):
                        raise ValueError(
                            "THRML native result is missing, unsafe, or exceeds 16 MiB"
                        )
                    payload = json.loads(artifact.read_text(encoding="utf-8"))
                    analysis = validate_native_result(
                        payload,
                        execution_id=env["GNN_THRML_EXECUTION_ID"],
                        script_sha256=hashlib.sha256(
                            script_path.read_bytes()
                        ).hexdigest(),
                    )
                    precondition = execution_precondition(
                        timeout, None, deadline_monotonic=deadline
                    )
                    if precondition is not None:
                        raise TimeoutError(
                            "THRML deadline exhausted during result validation"
                        )
                    exec_result["scientific_analysis"] = analysis
                    exec_result["native_result_file"] = str(artifact)
                    exec_result["execution_metadata"].update(
                        payload["runtime_metadata"]
                    )

                durations_success.append(elapsed_rep)
                result = run_result
                if verbose and K > 1:
                    logger.info(
                        f"Benchmark rep {rep + 1}/{K} for {script_info['name']}: {elapsed_rep:.3f}s"
                    )

            if not broke_early and result is not None and len(durations_success) == K:
                agg = _aggregate_benchmark_samples(durations_success)
                exec_result.update(agg)
                exec_result["success"] = True
                exec_result["status"] = "success"
                exec_result["return_code"] = result.returncode
                exec_result["stdout"] = result.stdout
                exec_result["stderr"] = result.stderr
                if K == 1:
                    logger.info(f"✅ Successfully executed {script_info['name']}")
                else:
                    logger.info(
                        f"✅ Successfully executed {script_info['name']} "
                        f"({K} reps, median {exec_result['execution_time']:.3f}s)"
                    )
                if verbose and result.stdout:
                    logger.info(f"Script output: {result.stdout[:200]}...")
        except Exception as e:
            exec_result["success"] = False
            exec_result["status"] = (
                "timed_out" if isinstance(e, TimeoutError) else "failed"
            )
            exec_result["execution_time"] = exec_result.get("execution_time", 0)
            exec_result["error"] = f"Script execution failed: {e}"
            exec_result["error_type"] = type(e).__name__
            exec_result["return_code"] = (
                result.returncode if result is not None else INTERNAL_ERROR
            )
            exec_result["stdout"] = result.stdout if result is not None else ""
            exec_result["stderr"] = result.stderr if result is not None else str(e)
            if result is not None and result.returncode == 0:
                exec_result["execution_result_success"] = True
            logger.warning(f"❌ Script {script_info['name']} execution failed: {e}")
            if result is None:
                result = subprocess.CompletedProcess(
                    args=[], returncode=INTERNAL_ERROR, stdout="", stderr=str(e)
                )

        # Ensure result is defined before using it
        if result is None:
            result = subprocess.CompletedProcess(
                args=[], returncode=UNKNOWN_STATE, stdout="", stderr="Unknown error"
            )

        # Save individual script output in implementation-specific subdirectory
        # Create the implementation-specific directory structure
        impl_specific_dir = results_dir / model_name / framework / "execution_logs"
        impl_specific_dir.mkdir(parents=True, exist_ok=True)

        # Note: Framework-specific subdirectories (visualizations, simulation_data, etc.)
        # are created on-demand by collect_execution_outputs() only when actual content
        # is copied to them, avoiding empty folder creation.

        # Extract simulation data from stdout/stderr
        data_framework = _framework_for_data_helpers(framework)
        simulation_data = _extract_simulation_data(
            result.stdout, result.stderr, data_framework, logger
        )
        exec_result["simulation_data"] = simulation_data

        # Hardware / accelerator metadata
        accelerator_type = _detect_accelerator_type()

        # Save structured execution results in JSON format
        structured_result: dict[str, Any] = {
            "framework": framework,
            "model_name": model_name,
            "script_name": script_info["name"],
            "script_path": str(script_path),
            "success": exec_result["success"],
            "return_code": exec_result.get("return_code"),
            "execution_time": exec_result.get("execution_time", 0),
            "timestamp": exec_result["timestamp"],
            "simulation_data": simulation_data,
            "execution_benchmark_repeats": exec_result.get(
                "execution_benchmark_repeats", 1
            ),
            "execution_metadata": {
                "executor": executor,
                "accelerator_type": accelerator_type,
                "python_version": platform.python_version(),
                "stdout_length": len(result.stdout),
                "stderr_length": len(result.stderr),
                "output_directory": str(impl_specific_dir.parent),
                **exec_result.get("execution_metadata", {}),
                "backend_version": _backend_version_from_runner_metadata(
                    exec_result.get("execution_metadata") or {}
                ),
            },
        }
        for bench_key in (
            "execution_time_mean",
            "execution_time_std",
            "execution_time_samples",
            "child_peak_rss_mb",
            "rss_sample_interval_seconds",
            "rss_samples_count",
            "cancelled",
            "containment",
            "observed_descendant_count",
            "cleanup_verified",
            "streams_drained",
            "cleanup_timeout_seconds",
            "cleanup_error",
            "execution_error_type",
            "error_type",
            "error",
        ):
            if bench_key in exec_result:
                structured_result[bench_key] = exec_result[bench_key]

        # Save structured JSON result
        json_output_file = impl_specific_dir / f"{script_info['name']}_results.json"
        with open(json_output_file, "w") as f:
            json.dump(structured_result, f, indent=2, default=str)

        exec_result["structured_result_file"] = str(json_output_file)

        # Also save human-readable log
        output_file = impl_specific_dir / f"{script_info['name']}_execution.log"
        with open(output_file, "w") as f:
            f.write(f"Execution Results for {script_info['name']}\n")
            f.write(f"Timestamp: {exec_result['timestamp']}\n")
            f.write(f"Return Code: {result.returncode}\n")
            f.write(
                f"Benchmark repeats: {exec_result.get('execution_benchmark_repeats', 1)}\n"
            )
            f.write(
                f"Execution Time (median wall-clock): {exec_result['execution_time']:.2f} seconds\n"
            )
            if exec_result.get("execution_time_samples"):
                f.write(
                    f"Sample durations (s): {exec_result['execution_time_samples']}\n"
                )
            if exec_result.get("execution_time_std") is not None:
                f.write(
                    f"Duration mean/std (s): {exec_result.get('execution_time_mean', 0):.4f} / "
                    f"{exec_result.get('execution_time_std', 0):.4f}\n"
                )
            f.write(f"Model: {model_name}\n")
            f.write(f"Framework: {framework}\n")
            f.write(f"Output Directory: {impl_specific_dir.parent}\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)

        exec_result["output_file"] = str(output_file)
        exec_result["implementation_directory"] = str(impl_specific_dir.parent)

        # Collect execution outputs (visualizations, simulation data, traces)
        if exec_result["success"]:
            try:
                logger.info(
                    f"Collecting execution outputs for {framework} script {script_info['name']}"
                )
                collected_outputs = collect_execution_outputs(
                    script_path, impl_specific_dir.parent, data_framework, logger
                )
                exec_result["collected_outputs"] = collected_outputs

                # Update structured result with collected file paths
                structured_result["collected_outputs"] = collected_outputs

                # Re-save structured result with collected outputs
                with open(json_output_file, "w") as f:
                    json.dump(structured_result, f, indent=2, default=str)
                logger.debug("Updated results JSON with collected outputs")

                # Enhance simulation data extraction from collected files
                if collected_outputs:
                    logger.info(
                        f"Extracting simulation data from collected files for {framework}"
                    )
                    enhanced_data = _extract_simulation_data_from_files(
                        impl_specific_dir.parent, data_framework, logger
                    )
                    if enhanced_data:
                        logger.info(
                            f"Extracted {len(enhanced_data)} data fields from files"
                        )
                        simulation_data.update(enhanced_data)
                        exec_result["simulation_data"] = simulation_data
                        structured_result["simulation_data"] = simulation_data

                        # Re-save again with enhanced data
                        with open(json_output_file, "w") as f:
                            json.dump(structured_result, f, indent=2, default=str)
                        logger.debug(
                            "Updated results JSON with enhanced simulation data"
                        )
                    else:
                        logger.debug(
                            f"No additional data extracted from files for {framework}"
                        )

                if framework == "pymdp":
                    sim_dir = impl_specific_dir.parent / "simulation_data"
                    sr_candidates = list(sim_dir.glob("*simulation_results.json"))
                    if (
                        not sr_candidates
                        and (sim_dir / "simulation_results.json").exists()
                    ):
                        sr_candidates = [sim_dir / "simulation_results.json"]
                    if sr_candidates:
                        try:
                            with open(sr_candidates[0], encoding="utf-8") as sf:
                                payload = json.load(sf)
                            n_steps = payload.get("num_timesteps")
                            if n_steps is None:
                                n_steps = len(payload.get("observations", []))
                            logger.info(
                                "pymdp_execution_summary model=%s script=%s simulation_results=%s timesteps=%s",
                                model_name,
                                script_info["name"],
                                sr_candidates[0],
                                n_steps,
                            )
                        except (OSError, json.JSONDecodeError, TypeError) as ex:
                            logger.debug("pymdp_execution_summary skipped: %s", ex)

            except Exception as e:
                exec_result["execution_result_success"] = True
                exec_result["success"] = False
                exec_result["status"] = "failed"
                exec_result["error_type"] = "ResultCollectionFailure"
                exec_result["error"] = str(e)
                logger.warning(f"Failed to collect execution outputs: {e}")
                import traceback

                logger.debug(traceback.format_exc())

    except subprocess.TimeoutExpired:
        exec_result["execution_result_success"] = bool(exec_result.get("success"))
        exec_result["success"] = False
        exec_result["status"] = "timed_out"
        exec_result["error_type"] = "TimeoutExpired"
        exec_result["error"] = f"Script execution timed out ({timeout} seconds)"
        logger.error(f"Script {script_info['name']} timed out")

    except Exception as e:
        exec_result["execution_result_success"] = bool(exec_result.get("success"))
        exec_result["success"] = False
        exec_result["status"] = "failed"
        exec_result["error"] = str(e)
        exec_result["error_type"] = type(e).__name__
        logger.error(f"Error executing {script_info['name']}: {e}")

    precondition = execution_precondition(timeout, None, deadline_monotonic=deadline)
    if (
        precondition is not None
        and exec_result.get("error_type") != "ProcessCleanupFailure"
    ):
        exec_result["execution_result_success"] = bool(
            exec_result.get("execution_result_success") or exec_result.get("success")
        )
        # Preserve actual child output and containment facts from the envelope.
        exec_result.update(
            {
                key: precondition[key]
                for key in ("success", "status", "error_type", "error")
            }
        )
    if not exec_result.get("success") and exec_result.get("structured_result_file"):
        receipt = Path(exec_result["structured_result_file"])
        persisted = json.loads(receipt.read_text(encoding="utf-8"))
        persisted.update(
            {
                key: exec_result[key]
                for key in (
                    "success",
                    "status",
                    "error_type",
                    "error",
                    "execution_result_success",
                )
                if key in exec_result
            }
        )
        receipt.write_text(
            json.dumps(persisted, indent=2, default=str), encoding="utf-8"
        )
    return exec_result
