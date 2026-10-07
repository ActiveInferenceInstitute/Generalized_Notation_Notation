"""bnlearn Runner for executing rendered bnlearn scripts.

Step 11's generator-backed bnlearn renderer emits, per model, a Python
program under ``<model>/bnlearn/`` (``import bnlearn as bn`` +
``bn.make_DAG`` + ``bn.parameter_learning.fit``). Step 12 discovers and runs
it like any other Python framework script (framework directory ``bnlearn/``,
output env var ``BNLEARN_OUTPUT_DIR``).

This module provides the dependency probes and a language-aware direct
runner used by tests and callers outside the pipeline. The execution
language is derived from each emitted file's suffix — ``.py`` runs under a
Python interpreter with the ``bnlearn`` module, ``.R`` runs under Rscript
with the R ``bnlearn`` package — never assumed. Missing runtimes produce an
explicit ``skipped`` record (mirroring the Stan executor's
skip-on-missing-toolchain semantics via ``gnn.utils.runtime_safety.framework_availability``).
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from gnn.execute.preconditions import execution_precondition
from gnn.execute.security_gate import check_script_allowed
from gnn.execute.subprocess_envelope import CancelToken, run_subprocess_envelope
from gnn.utils.runtime_safety.framework_availability import (
    DEFAULT_PROBE_TIMEOUT_SECONDS,
    FrameworkStatus,
    check_framework,
)

logger = logging.getLogger(__name__)

FRAMEWORK: str = "bnlearn"
OUTPUT_ENV_VAR: str = "BNLEARN_OUTPUT_DIR"

_PYTHON_SUFFIXES = frozenset({".py"})
_R_SUFFIXES = frozenset({".r"})


def _check_bnlearn_status(
    python_executable: str | None = None, timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS
) -> FrameworkStatus:
    """Retain the selected interpreter's structured dependency diagnosis."""
    return check_framework(
        FRAMEWORK, executor=python_executable, logger=logger, timeout=timeout
    )


def is_bnlearn_available(python_executable: Optional[str] = None) -> bool:
    """True when a bounded child imports Python bnlearn successfully."""
    return _check_bnlearn_status(python_executable).available


def _check_r_bnlearn_status(
    rscript_executable: str = "Rscript", timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS
) -> FrameworkStatus:
    """Distinguish absent R/package from a failed or exhausted package probe."""
    rscript = shutil.which(rscript_executable)
    hint = "install.packages('bnlearn')"
    if rscript is None:
        return FrameworkStatus(
            FRAMEWORK,
            False,
            install_hint=hint,
            reason_code="executor_unavailable",
            reason="Rscript not found on PATH",
        )
    probe = run_subprocess_envelope(
        [
            rscript,
            "-e",
            "if (!requireNamespace('bnlearn', quietly=TRUE)) quit(status=42); suppressMessages(library(bnlearn))",
        ],
        timeout=timeout,
        sandbox=False,
    )
    if probe["success"]:
        return FrameworkStatus(FRAMEWORK, True)
    missing = (
        probe.get("return_code") == 42 and probe.get("cleanup_verified") is not False
    )
    timed_out = probe.get("error_type") == "TimeoutExpired"
    return FrameworkStatus(
        FRAMEWORK,
        False,
        missing_module="bnlearn" if missing else None,
        install_hint=hint,
        reason_code="missing_module"
        if missing
        else "probe_timeout"
        if timed_out
        else "probe_failed",
        reason="R bnlearn package not installed"
        if missing
        else "R bnlearn package probe failed: "
        + str(probe.get("error_type") or probe.get("return_code")),
        execution_error_type=probe.get("execution_error_type")
        or probe.get("error_type"),
        cleanup_verified=probe.get("cleanup_verified"),
        streams_drained=probe.get("streams_drained"),
    )


def is_r_bnlearn_available(rscript_executable: str = "Rscript") -> bool:
    """True when a bounded Rscript child loads the R bnlearn package."""
    return _check_r_bnlearn_status(rscript_executable).available


def _record_unavailable(
    record: Dict[str, Any], diagnosis: FrameworkStatus
) -> Dict[str, Any]:
    """Only positively identified absent prerequisites qualify as skips."""
    skipped = diagnosis.reason_code in {
        "missing_module",
        "missing_toolchain",
        "executor_unavailable",
        "unsupported_python",
        "unsupported_version",
    }
    cleanup_failed = (
        diagnosis.cleanup_verified is False or diagnosis.streams_drained is False
    )
    error_type = (
        "ProcessCleanupFailure"
        if cleanup_failed
        else diagnosis.execution_error_type
        or ("DependencyUnavailable" if skipped else "FrameworkProbeFailure")
    )
    record.update(
        skipped=skipped and not cleanup_failed,
        status="failed"
        if cleanup_failed
        else "skipped"
        if skipped
        else "timed_out"
        if diagnosis.reason_code == "probe_timeout"
        else "failed",
        reason=diagnosis.reason,
        reason_code=diagnosis.reason_code,
        install_hint=diagnosis.install_hint,
        error_type=error_type,
        execution_error_type=diagnosis.execution_error_type,
        cleanup_verified=diagnosis.cleanup_verified,
        streams_drained=diagnosis.streams_drained,
    )
    return record


def script_language(script_path: Union[str, Path]) -> str:
    """Return ``"python"``, ``"r"``, or ``"unknown"`` from the file suffix."""
    suffix = Path(script_path).suffix.lower()
    if suffix in _PYTHON_SUFFIXES:
        return "python"
    if suffix in _R_SUFFIXES:
        return "r"
    return "unknown"


def find_bnlearn_scripts(render_output_dir: Union[str, Path]) -> List[Path]:
    """Return every rendered bnlearn script (``.py``/``.R`` in ``bnlearn/`` dirs)."""
    root = Path(render_output_dir)
    found: set[Path] = set()
    for pattern in ("*.py", "*.R"):
        for path in root.rglob(pattern):
            if path.parent.name.lower() == "bnlearn":
                found.add(path)
    return sorted(found)


def execute_bnlearn_script(
    script_path: Union[str, Path],
    output_dir: Union[str, Path],
    timeout: int = 1800,
    python_executable: Optional[str] = None,
    rscript_executable: str = "Rscript",
    cancel_token: Optional[CancelToken] = None,
) -> Dict[str, Any]:
    """Run one rendered bnlearn script; return a structured result dict.

    The execution lane is derived from the script's suffix. Missing runtimes
    produce a ``skipped`` record without spawning a subprocess.
    """
    script = Path(script_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    language = script_language(script)
    record: Dict[str, Any] = {
        "script": str(script),
        "framework": FRAMEWORK,
        "language": language,
        "return_code": None,
        "success": False,
        "skipped": False,
        "status": "failed",
        "stdout": "",
        "stderr": "",
        "execution_time_seconds": 0.0,
        # Informational pointer to the conventional results path; rendered
        # bnlearn programs are not required to write it.
        "results_file": str(out_dir / "simulation_results.json"),
    }

    precondition = execution_precondition(timeout, cancel_token)
    if precondition is not None:
        record.update(precondition)
        return record

    # Shared pre-execution security gate (fail closed; GNN_ALLOW_UNSAFE_EXEC
    # is the only operator opt-out). Runs before any lane probe, command
    # construction, or subprocess spawn.
    gate_verdict = check_script_allowed(script)
    if gate_verdict["overridden"]:
        logger.warning(
            "GNN_ALLOW_UNSAFE_EXEC set: pre-execution security gate "
            "bypassed for %s (trusted-local use only)",
            script,
        )
    if not gate_verdict["ok"]:
        record["error_type"] = gate_verdict.get("error_type", "SecurityGateBlocked")
        record["security_findings"] = gate_verdict["blocked"]
        record["error"] = (
            f"Pre-execution security gate blocked {script_path}: "
            f"{gate_verdict['reason']}"
        )
        logger.error(record["error"])
        return record

    if language == "python":
        diagnosis = _check_bnlearn_status(
            python_executable, min(timeout, DEFAULT_PROBE_TIMEOUT_SECONDS)
        )
        if not diagnosis.available:
            return _record_unavailable(record, diagnosis)
        command: List[str] = [python_executable or sys.executable, str(script)]
    elif language == "r":
        diagnosis = _check_r_bnlearn_status(
            rscript_executable, min(timeout, DEFAULT_PROBE_TIMEOUT_SECONDS)
        )
        if not diagnosis.available:
            return _record_unavailable(record, diagnosis)
        command = [rscript_executable, str(script)]
    else:
        record["skipped"] = True
        record["status"] = "unsupported"
        record["reason"] = (
            f"Unsupported bnlearn script language: {script.suffix or '<none>'}"
        )
        logger.info("Skipping bnlearn script (unknown language): %s", script.name)
        return record

    precondition = execution_precondition(timeout, cancel_token)
    if precondition is not None:
        record.update(precondition)
        return record
    envelope = run_subprocess_envelope(
        command,
        timeout=timeout,
        env={OUTPUT_ENV_VAR: str(out_dir)},
        cwd=str(out_dir),
        cancel_token=cancel_token,
    )
    record.update(envelope)
    record["status"] = (
        "success"
        if envelope["success"]
        else "cancelled"
        if envelope.get("cancelled")
        else "timed_out"
        if envelope.get("error_type") == "TimeoutExpired"
        else "failed"
    )
    record["return_code"] = envelope["return_code"]
    record["success"] = envelope["success"]
    record["stdout"] = envelope["stdout"]
    record["stderr"] = envelope["stderr"]
    record["execution_time_seconds"] = round(envelope["duration_seconds"], 3)
    if not envelope["success"]:
        error_type = envelope.get("error_type")
        if error_type == "TimeoutExpired":
            record["error_type"] = "TimeoutExpired"
            record["error"] = (
                f"bnlearn script timed out after {timeout}s: {script.name}"
            )
        else:
            record["error_type"] = error_type or "RuntimeError"
            record["error"] = envelope.get("error") or (
                f"bnlearn script failed ({envelope['return_code']}): {script.name}"
            )
        logger.error(record["error"])
    return record


def run_bnlearn_scripts(
    render_output_dir: Union[str, Path],
    output_dir: Union[str, Path],
    timeout: int = 1800,
    python_executable: Optional[str] = None,
    rscript_executable: str = "Rscript",
) -> List[Dict[str, Any]]:
    """Execute every rendered bnlearn script; each records skip/fail/explicitly."""
    scripts = find_bnlearn_scripts(render_output_dir)
    return [
        execute_bnlearn_script(
            s,
            Path(output_dir) / s.parent.parent.name,
            timeout,
            python_executable=python_executable,
            rscript_executable=rscript_executable,
        )
        for s in scripts
    ]
