#!/usr/bin/env python3
"""Per-script execution-result envelope factories for Step 12 execution."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from gnn.execute.metadata import _load_rxinfer_execution_metadata_from_script
from gnn.utils.runtime_safety.framework_availability import (
    FRAMEWORK_IMPORT_CHECK as _FRAMEWORK_IMPORT_CHECK,
)
from gnn.utils.runtime_safety.framework_availability import (
    last_unavailable_status as _last_unavailable_status,
)


def _is_python_framework_dependency_available(
    framework: str, executor: str, logger: Any
) -> bool:
    """Return True if the framework's required Python module is importable.

    Delegates to ``gnn.utils.runtime_safety.framework_availability.is_framework_available``, passing
    ``executor`` so the check targets the subprocess-invoked interpreter rather
    than the caller's. Preserves the pre-Phase-2.3 call-site signature.
    """
    from gnn.execute import processor as _processor_facade

    return _processor_facade._is_framework_available_by_name(
        framework, executor=executor, logger=logger
    )


def _base_execution_envelope(
    *,
    script_path: str,
    script_name: str,
    framework: str,
    model_name: str,
    executor: str,
    status: str,
    skipped: bool,
) -> Dict[str, Any]:
    """Shared 14-key prefix for every per-script execution result envelope.

    Failure and skip builders append their ``error``/``error_type`` (and any
    dispatch-specific extras) on top; the success path mutates the envelope in
    place. Keeping one construction site prevents key-set drift between the
    factories.
    """
    return {
        "script_path": script_path,
        "script_name": script_name,
        "framework": framework,
        "model_name": model_name,
        "executor": executor,
        "success": False,
        "skipped": skipped,
        "status": status,
        "attempts_started": 0,
        "return_code": None,
        "stdout": "",
        "stderr": "",
        "execution_time": 0,
        "timestamp": datetime.now().isoformat(),
    }


def _model_framework_from_path(script_info: Dict[str, Any]) -> Tuple[str, str]:
    """Derive ``(model_name, framework)`` fallbacks from a rendered script path.

    Rendered layouts are ``<...>/<model>/<framework>/<script>``, so the third
    and second path components from the end are authoritative; shorter paths
    fall back to the discovery metadata.
    """
    path_parts = Path(script_info["path"]).parts
    model_name = path_parts[-3] if len(path_parts) >= 3 else "unknown_model"
    framework = path_parts[-2] if len(path_parts) >= 3 else script_info["framework"]
    return model_name, framework


def _make_skipped_result(
    script_info: Dict[str, Any],
    framework: str,
    model_name: str,
    executor: str,
    logger: Any,
) -> Dict[str, Any]:
    """Build an execution result dict for a script skipped on a dependency pre-flight.

    The reason comes from the probe's last negative diagnosis for this
    ``(framework, executor)`` (module missing, interpreter too old, toolchain
    probe failed, or probe timed out); without one it falls back to the
    generic "Dependency not installed" wording.
    """
    module_name, install_hint = _FRAMEWORK_IMPORT_CHECK.get(framework, ("", ""))
    reason = (
        f"Dependency not installed: {module_name}"
        if module_name
        else "Dependency not installed"
    )
    skip_category = "missing_module"
    diagnosis = _last_unavailable_status(framework, executor)
    if diagnosis is not None and diagnosis.reason:
        reason = diagnosis.reason
        install_hint = diagnosis.install_hint or install_hint
        skip_category = diagnosis.skip_category or skip_category
    if install_hint and not logger.isEnabledFor(logging.DEBUG):
        logger.info(
            f"Skipping {script_info['name']} ({framework}): {reason}. Install with: {install_hint}"
        )
    envelope = _base_execution_envelope(
        script_path=str(script_info["path"]),
        script_name=script_info["name"],
        framework=framework,
        model_name=model_name,
        executor=executor,
        status="skipped",
        skipped=True,
    )
    envelope["error"] = reason
    envelope["error_type"] = "DependencyNotInstalled"
    envelope["skip_category"] = skip_category
    envelope["execution_metadata"] = (
        _load_rxinfer_execution_metadata_from_script(Path(script_info["path"]))
        if framework == "rxinfer"
        else {}
    )
    return envelope


def _make_local_worker_pool_failure_result(
    script_info: Dict[str, Any],
    exc: BaseException,
) -> Dict[str, Any]:
    """Return a per-script failure envelope when local process dispatch fails."""
    script_path = Path(script_info["path"])
    model_name, framework = _model_framework_from_path(script_info)
    envelope = _base_execution_envelope(
        script_path=str(script_path),
        script_name=script_info["name"],
        framework=framework,
        model_name=model_name,
        executor=script_info["executor"],
        status="failed",
        skipped=False,
    )
    envelope["error"] = f"Local worker pool failed before script completion: {exc}"
    envelope["error_type"] = "LocalWorkerPoolError"
    envelope["worker_pool_error_type"] = type(exc).__name__
    return envelope


def _make_distributed_dispatch_failure_result(
    script_info: Dict[str, Any],
    exc: BaseException,
    backend: str,
    max_retries: int,
) -> Dict[str, Any]:
    """Return one explicit failure when distributed dispatch cannot complete."""
    script_path = Path(script_info["path"])
    model_name, framework = _model_framework_from_path(script_info)
    envelope = _base_execution_envelope(
        script_path=str(script_path),
        script_name=script_info["name"],
        framework=framework,
        model_name=model_name,
        executor=script_info["executor"],
        status="failed",
        skipped=False,
    )
    envelope["error"] = (
        f"Distributed {backend} dispatch failed before completion: {exc}"
    )
    envelope["error_type"] = "DistributedDispatchError"
    envelope["dispatch_error_type"] = type(exc).__name__
    envelope["dispatch_max_retries"] = max_retries
    return envelope
