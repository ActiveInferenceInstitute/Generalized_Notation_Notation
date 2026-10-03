"""Supervised THRML execution with fresh artifact and sampling witnesses."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from gnn.execute.preconditions import (
    execution_precondition,
    script_deadline,
    unavailable_framework_result,
)
from gnn.execute.subprocess_envelope import CancelToken
from gnn.pipeline._io import atomic_write_text
from gnn.pipeline.output_lease import OutputLease, OutputLeaseError
from gnn.utils.runtime_safety.framework_availability import (
    DEFAULT_PROBE_TIMEOUT_SECONDS,
    check_framework,
)

logger = logging.getLogger(__name__)
_MAX_RESULT_BYTES = 16 * 1024 * 1024


def validate_native_result(
    payload: dict[str, Any], *, execution_id: str, script_sha256: str
) -> dict[str, Any]:
    """Share sampling validation and fresh runtime identity binding with Step 12."""
    from gnn.analysis.thrml.adapter import adapt_result

    summary = adapt_result(payload)
    runtime = payload["runtime_metadata"]
    if (
        runtime.get("execution_id") != execution_id
        or runtime.get("script_sha256") != script_sha256
    ):
        raise ValueError(
            "THRML result identity does not match this execution and script"
        )
    return summary


def _unfinished(result: dict[str, Any], blocked: dict[str, Any]) -> None:
    """Keep already-observed child output and cleanup truth after budget exhaustion."""
    result.update(
        {
            key: blocked[key]
            for key in ("success", "status", "error_type", "error", "cancelled")
            if key in blocked
        },
        execution_result_success=True,
    )


def is_thrml_available() -> bool:
    """Use the shared bounded current-interpreter readiness diagnosis."""
    return check_framework("thrml", executor=sys.executable).available


def find_thrml_scripts(base_dir: str | Path, recursive: bool = True) -> list[Path]:
    """Discover deterministic THRML script identities for standalone use."""
    root = Path(base_dir)
    if not root.is_dir():
        return []
    paths = root.rglob("*.py") if recursive else root.glob("*.py")
    return sorted(
        {
            path.resolve()
            for path in paths
            if path.name.lower().endswith("_thrml.py") or path.parent.name == "thrml"
        }
    )


def _persist(result: dict[str, Any], output: Path) -> None:
    """Preserve bounded process evidence without duplicating bulk scientific data."""
    for field in ("stdout", "stderr"):
        atomic_write_text(output / f"{field}.txt", str(result.get(field, "")))
    receipt = {
        key: value
        for key, value in result.items()
        if key not in {"stdout", "stderr", "simulation_data"}
    }
    atomic_write_text(output / "execution_log.json", json.dumps(receipt, indent=2))


def execute_thrml_script(
    script_path: str | Path,
    verbose: bool = False,
    output_dir: str | Path | None = None,
    timeout: float | None = 300,
    *,
    cancel_token: CancelToken | None = None,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    """Execute one script under one ceiling, accepting only its fresh native JSON.

    THRML and JAX load only in supervised children. An explicit output directory
    is leased until dispatch, result validation and receipt publication finish.
    """
    script = Path(script_path).resolve()
    started = time.monotonic()
    result: dict[str, Any] = {
        "framework": "thrml",
        "experimental": True,
        "script": str(script),
        "script_path": str(script),
        "success": False,
    }
    blocked = execution_precondition(
        timeout, cancel_token, deadline_monotonic=deadline_monotonic
    )
    if blocked is not None:
        return {**result, **blocked}
    if not script.is_file() or script.suffix.lower() != ".py":
        return {
            **result,
            "status": "failed",
            "error_type": "InvalidScript",
            "error": "THRML requires an existing Python script",
        }
    deadline = script_deadline(timeout, deadline_monotonic)
    from gnn.execute.security_gate import check_script_allowed

    verdict = check_script_allowed(script)
    if not verdict["ok"]:
        return {
            **result,
            "status": "failed",
            "error_type": verdict.get("error_type", "SecurityGateBlocked"),
            "error": verdict.get("reason", "Script blocked"),
            "security_findings": verdict["blocked"],
        }
    blocked = execution_precondition(timeout, cancel_token, deadline_monotonic=deadline)
    if blocked is not None:
        return {**result, **blocked}
    diagnosis = check_framework(
        "thrml",
        executor=sys.executable,
        timeout=DEFAULT_PROBE_TIMEOUT_SECONDS,
        deadline_monotonic=deadline,
        cancel_token=cancel_token,
    )
    if not diagnosis.available and (
        diagnosis.cleanup_verified is False or diagnosis.streams_drained is False
    ):
        return {**result, **unavailable_framework_result(diagnosis)}
    blocked = execution_precondition(timeout, cancel_token, deadline_monotonic=deadline)
    if blocked is not None:
        return {**result, **blocked, "readiness": diagnosis.reason_code}
    if not diagnosis.available:
        return {**result, **unavailable_framework_result(diagnosis)}
    execution_id = uuid4().hex
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else script.parent / f"{script.stem}_execution"
    )
    result.update(
        execution_id=execution_id,
        output_dir=str(output),
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
    )
    try:
        with OutputLease(output, execution_id):
            artifact = output / "simulation_data" / "simulation_results.json"
            if artifact.parent.is_symlink() or artifact.is_symlink():
                raise OutputLeaseError("THRML result destination cannot be a symlink")
            from gnn.execute.executor import execute_script_safely

            envelope = execute_script_safely(
                script,
                timeout=timeout,
                deadline_monotonic=deadline,
                cancel_token=cancel_token,
                cwd=output,
                env={
                    "THRML_OUTPUT_DIR": str(output),
                    "GNN_THRML_EXECUTION_ID": execution_id,
                },
            )
            result.update(envelope)
            if not result.get("success"):
                result["status"] = (
                    "failed"
                    if result.get("cleanup_verified") is False
                    or result.get("streams_drained") is False
                    else "cancelled"
                    if result.get("cancelled")
                    or result.get("error_type") == "Cancelled"
                    else "timed_out"
                    if result.get("error_type") == "TimeoutExpired"
                    else "failed"
                )
            if result.get("success"):
                try:
                    blocked = execution_precondition(
                        timeout, cancel_token, deadline_monotonic=deadline
                    )
                    if blocked is not None:
                        _unfinished(result, blocked)
                    else:
                        if (
                            artifact.parent.is_symlink()
                            or artifact.is_symlink()
                            or not artifact.is_file()
                            or artifact.stat().st_size > _MAX_RESULT_BYTES
                        ):
                            raise ValueError(
                                "THRML native result is unsafe, missing or exceeds 16 MiB"
                            )
                        payload = json.loads(artifact.read_text(encoding="utf-8"))
                        analysis = validate_native_result(
                            payload,
                            execution_id=execution_id,
                            script_sha256=result["script_sha256"],
                        )
                        blocked = execution_precondition(
                            timeout, cancel_token, deadline_monotonic=deadline
                        )
                        if blocked is not None:
                            _unfinished(result, blocked)
                        else:
                            result.update(
                                status="success",
                                simulation_data=payload,
                                scientific_analysis=analysis,
                                result_file=str(artifact),
                            )
                except (OSError, TypeError, ValueError, KeyError) as error:
                    result.update(
                        success=False,
                        status="failed",
                        error_type="InvalidThrmlResult",
                        error=str(error),
                        execution_result_success=True,
                    )
            if verbose:
                logger.info("THRML %s: %s", script.name, result.get("status", "failed"))
            result["runner_duration_seconds"] = time.monotonic() - started
            _persist(result, output)
            blocked = execution_precondition(
                timeout, cancel_token, deadline_monotonic=deadline
            )
            if result.get("success") and blocked is not None:
                _unfinished(result, blocked)
                _persist(result, output)
    except (OSError, OutputLeaseError) as error:
        result.update(
            success=False,
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
        )
    blocked = execution_precondition(timeout, cancel_token, deadline_monotonic=deadline)
    if result.get("success") and blocked is not None:
        _unfinished(result, blocked)
        # Lease release is required cleanup too. Reacquire non-blockingly before
        # updating a late verdict, so a new writer's evidence is never replaced.
        try:
            with OutputLease(output, execution_id):
                _persist(result, output)
        except (OSError, OutputLeaseError) as error:
            result["receipt_update_error"] = str(error)
    result["runner_duration_seconds"] = time.monotonic() - started
    return result


def run_thrml_records(
    rendered_simulators_dir: str | Path,
    execution_output_dir: str | Path | None = None,
    recursive_search: bool = True,
    verbose: bool = False,
    timeout: float | None = 300,
    *,
    selected_scripts: Sequence[str | Path] | None = None,
    cancel_token: CancelToken | None = None,
    deadline_monotonic: float | None = None,
) -> list[dict[str, Any]]:
    """Execute each selected identity once, preserving independent outcomes."""
    blocked = execution_precondition(
        timeout, cancel_token, deadline_monotonic=deadline_monotonic
    )
    if blocked is not None:
        return [{"framework": "thrml", **blocked}]
    deadline = script_deadline(timeout, deadline_monotonic)
    root = Path(rendered_simulators_dir).resolve()
    scripts = (
        find_thrml_scripts(root, recursive_search)
        if selected_scripts is None
        else sorted({Path(path).resolve() for path in selected_scripts})
    )
    results = []
    for script in scripts:
        try:
            relative = script.relative_to(root).as_posix()
        except ValueError:
            results.append(
                {
                    "framework": "thrml",
                    "script_path": str(script),
                    "success": False,
                    "status": "failed",
                    "error_type": "InvalidSelection",
                    "error": "Selected THRML script is outside the render root",
                }
            )
            continue
        identity = hashlib.sha256(relative.encode()).hexdigest()[:16]
        output = (
            Path(execution_output_dir) / f"{script.stem}_{identity}"
            if execution_output_dir is not None
            else None
        )
        results.append(
            execute_thrml_script(
                script,
                verbose,
                output,
                timeout,
                cancel_token=cancel_token,
                deadline_monotonic=deadline,
            )
        )
    return results


def run_thrml_scripts(
    rendered_simulators_dir: str | Path,
    execution_output_dir: str | Path | None = None,
    recursive_search: bool = True,
    verbose: bool = False,
    timeout: float | None = 300,
    *,
    selected_scripts: Sequence[str | Path] | None = None,
    cancel_token: CancelToken | None = None,
    deadline_monotonic: float | None = None,
) -> bool:
    """Success requires every selected record; an explicit empty list does no work."""
    records = run_thrml_records(
        rendered_simulators_dir,
        execution_output_dir,
        recursive_search,
        verbose,
        timeout,
        selected_scripts=selected_scripts,
        cancel_token=cancel_token,
        deadline_monotonic=deadline_monotonic,
    )
    explicit_empty = selected_scripts is not None and len(selected_scripts) == 0
    return (bool(records) or explicit_empty) and all(
        result.get("success") is True for result in records
    )
