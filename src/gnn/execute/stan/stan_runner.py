#!/usr/bin/env python3
"""
Stan Runner for executing rendered Stan drivers.

Step 11's Stan renderer emits, per model, a ``.stan`` program and a sibling
``<stem>_stan.py`` cmdstanpy driver (simulate → compile → sample → results).
Step 12 executes the driver like any other Python framework script; this
module provides the dependency probe and a direct runner used by tests and
callers outside the pipeline.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from gnn.execute.preconditions import (
    execution_precondition,
    script_deadline,
    unavailable_framework_result,
)
from gnn.execute.security_gate import check_script_allowed
from gnn.execute.subprocess_envelope import (
    CancelToken,
    _resolve_timeout,
    run_subprocess_envelope,
)
from gnn.utils.runtime_safety.framework_availability import (
    DEFAULT_PROBE_TIMEOUT_SECONDS,
    FrameworkStatus,
    check_framework,
)

logger = logging.getLogger(__name__)


def is_stan_available() -> bool:
    """True when a bounded child imports cmdstanpy and finds CmdStan."""
    return _check_stan_status().available


def _check_stan_status(
    python_executable: str | None = None,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    *,
    deadline_monotonic: float | None = None,
) -> FrameworkStatus:
    """Preserve the shared interpreter/toolchain diagnosis for result receipts."""
    return check_framework(
        "stan",
        executor=python_executable,
        logger=logger,
        timeout=timeout,
        deadline_monotonic=deadline_monotonic,
    )


def _unavailable_stan_result(
    script: Path, diagnosis: FrameworkStatus
) -> dict[str, Any]:
    """Dependency absence skips; failed probes remain failed work."""
    return {
        "script": str(script),
        "framework": "stan",
        **unavailable_framework_result(diagnosis),
    }


def find_stan_scripts(render_output_dir: Union[str, Path]) -> List[Path]:
    """Return every rendered Stan driver (``*_stan.py``) under a render tree."""
    root = Path(render_output_dir)
    return sorted(p for p in root.rglob("*_stan.py") if p.parent.name == "stan")


def execute_stan_script(
    script_path: Union[str, Path],
    output_dir: Union[str, Path],
    timeout: float | None = 1800,
    python_executable: Optional[str] = None,
    cancel_token: Optional[CancelToken] = None,
    *,
    deadline_monotonic: float | None = None,
) -> Dict[str, Any]:
    """Run one Stan driver with ``STAN_OUTPUT_DIR`` set; return a result dict."""
    precondition = execution_precondition(
        timeout, cancel_token, deadline_monotonic=deadline_monotonic
    )
    if precondition is not None:
        return {"script": str(script_path), "framework": "stan", **precondition}
    resolved_timeout = _resolve_timeout(timeout)
    try:
        deadline = script_deadline(resolved_timeout, deadline_monotonic)
    except (TypeError, ValueError) as error:
        return {
            "script": str(script_path),
            "framework": "stan",
            "success": False,
            "status": "failed",
            "error_type": "InvalidExecutionDeadline",
            "error": str(error),
        }
    script = Path(script_path).resolve()
    gate = check_script_allowed(script)
    if gate["overridden"]:
        logger.warning(
            "GNN_ALLOW_UNSAFE_EXEC bypassed the Stan script gate for %s", script
        )
    if not gate["ok"]:
        return {
            "script": str(script),
            "framework": "stan",
            "success": False,
            "status": "failed",
            "return_code": -1,
            "error_type": gate.get("error_type", "SecurityGateBlocked"),
            "error": f"Pre-execution security gate blocked {script}: {gate['reason']}",
            "security_findings": gate["blocked"],
        }
    diagnosis = _check_stan_status(
        python_executable,
        min(resolved_timeout, DEFAULT_PROBE_TIMEOUT_SECONDS),
        deadline_monotonic=deadline,
    )
    if not diagnosis.available:
        return _unavailable_stan_result(script, diagnosis)
    precondition = execution_precondition(
        resolved_timeout, cancel_token, deadline_monotonic=deadline
    )
    if precondition is not None:
        return {"script": str(script), "framework": "stan", **precondition}
    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    execution_timeout: float = min(resolved_timeout, deadline - time.monotonic())
    if context is not None:
        execution_timeout = context.bounded_timeout(execution_timeout)
    if execution_timeout <= 0:
        return {
            "script": str(script),
            "framework": "stan",
            "success": False,
            "status": "timed_out",
            "return_code": -1,
            "error_type": "TimeoutExpired",
            "error": "Invocation deadline exhausted",
        }
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    envelope = run_subprocess_envelope(
        [python_executable or sys.executable, str(script)],
        timeout=execution_timeout,
        env={"STAN_OUTPUT_DIR": str(out_dir)},
        cwd=str(out_dir),
        cancel_token=cancel_token,
        deadline_monotonic=deadline,
    )
    result: Dict[str, Any] = {
        "script": str(script),
        "framework": "stan",
        "return_code": envelope["return_code"],
        "success": envelope["success"],
        "stdout": envelope["stdout"],
        "stderr": envelope["stderr"],
        "execution_time_seconds": round(envelope["duration_seconds"], 3),
        "results_file": str(out_dir / "simulation_results.json"),
    }
    for key in (
        "error_type",
        "error",
        "cancelled",
        "containment",
        "cleanup_verified",
        "streams_drained",
        "cleanup_timeout_seconds",
        "cleanup_error",
        "execution_error_type",
        "observed_descendant_count",
    ):
        if key in envelope and (
            key not in {"error", "error_type"} or envelope[key] is not None
        ):
            result[key] = envelope[key]
    if not envelope["success"]:
        if envelope.get("error_type") == "TimeoutExpired":
            logger.error(f"Stan driver timed out after {timeout}s: {script.name}")
        else:
            logger.error(
                f"Stan driver failed ({envelope['return_code']}): {script.name}"
            )
    return result


def run_stan_scripts(
    render_output_dir: Union[str, Path],
    output_dir: Union[str, Path],
    timeout: int = 1800,
    cancel_token: Optional[CancelToken] = None,
) -> List[Dict[str, Any]]:
    """Execute every rendered Stan driver; skip all with a reason if unavailable."""
    scripts = find_stan_scripts(render_output_dir)
    precondition = execution_precondition(timeout, cancel_token)
    if precondition is not None:
        return [
            {"script": str(script), "framework": "stan", **precondition}
            for script in scripts
        ]
    if not scripts:
        return []
    diagnosis = _check_stan_status(timeout=min(timeout, DEFAULT_PROBE_TIMEOUT_SECONDS))
    if not diagnosis.available:
        return [_unavailable_stan_result(script, diagnosis) for script in scripts]
    return [
        execute_stan_script(
            s,
            Path(output_dir) / s.parent.parent.name,
            timeout,
            cancel_token=cancel_token,
        )
        for s in scripts
    ]
