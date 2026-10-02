"""Tests for executor timeout propagation into framework runners.

``_execute_framework_spec`` and ``_execute_configured_frameworks`` must hand
the caller's ``timeout`` to the backend runner as a keyword; the default is
``None`` (each runner then applies its own historical ceiling), and
``run_lean_scripts`` accepts the same optional ``timeout``.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

import pytest

import gnn.execute.executor as executor_module
import gnn.execute.lean.lean_runner as lean_runner
from gnn.execute.executor import (
    ExecutorFrameworkSpec,
    _execute_configured_frameworks,
    _execute_framework_spec,
)


def _recording_spec(captured: list[dict[str, Any]]) -> ExecutorFrameworkSpec:
    """Build a pymdp-shaped spec whose runner records its keyword call."""

    def recording_runner(**kwargs: Any) -> bool:
        captured.append(kwargs)
        return True

    return ExecutorFrameworkSpec(
        framework_dir_key="pymdp",
        result_key="pymdp_executions",
        available=True,
        runner=recording_runner,
        operation_name="execute_pymdp_scripts",
        start_message="start",
        success_message="done",
        failure_message="failed",
        unavailable_log="unavailable",
        unavailable_message="unavailable",
        success_log="completed",
        warning_log_prefix="warning",
    )


def _execution_results() -> dict[str, Any]:
    return {"pymdp_executions": [], "total_successes": 0, "total_failures": 0}


def test_execute_framework_spec_propagates_timeout(tmp_path: Path) -> None:
    """An explicit timeout reaches the runner as a keyword and counts a success."""
    captured: list[dict[str, Any]] = []
    spec = _recording_spec(captured)
    framework_dirs = {spec.framework_dir_key: tmp_path / "pymdp"}
    results = _execution_results()

    _execute_framework_spec(
        spec,
        tmp_path,
        framework_dirs,
        results,
        logging.getLogger(__name__),
        recursive=False,
        verbose=False,
        timeout=42,
    )

    assert captured[0]["timeout"] == 42
    assert results["total_successes"] == 1
    assert results["pymdp_executions"][0]["status"] == "SUCCESS"


def test_execute_framework_spec_default_timeout_is_none(tmp_path: Path) -> None:
    """Without an explicit timeout the runner receives ``timeout=None``."""
    captured: list[dict[str, Any]] = []
    spec = _recording_spec(captured)
    framework_dirs = {spec.framework_dir_key: tmp_path / "pymdp"}

    _execute_framework_spec(
        spec,
        tmp_path,
        framework_dirs,
        _execution_results(),
        logging.getLogger(__name__),
        recursive=False,
        verbose=False,
    )

    assert captured[0]["timeout"] is None


def test_execute_configured_frameworks_threads_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batch funnel forwards its timeout to every configured spec runner."""
    captured: list[dict[str, Any]] = []
    spec = _recording_spec(captured)
    monkeypatch.setattr(executor_module, "_framework_specs", lambda **kwargs: (spec,))
    framework_dirs = {spec.framework_dir_key: tmp_path / "pymdp"}
    results = _execution_results()

    _execute_configured_frameworks(
        tmp_path,
        framework_dirs,
        results,
        logging.getLogger(__name__),
        recursive=False,
        verbose=False,
        timeout=42,
    )

    assert captured[0]["timeout"] == 42
    assert results["total_successes"] == 1


def test_run_lean_scripts_signature_has_timeout() -> None:
    """``run_lean_scripts`` accepts an optional ``timeout`` defaulting to None."""
    parameter = inspect.signature(lean_runner.run_lean_scripts).parameters["timeout"]
    assert parameter.default is None
