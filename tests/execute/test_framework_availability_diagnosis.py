#!/usr/bin/env python3
"""Step 12 skip reasons must name the real cause, not always "not installed".

A full pipeline run reported ``Dependency not installed: cmdstanpy`` while
cmdstanpy was installed (the CmdStan toolchain was missing) and
``Dependency not installed: bnlearn`` while bnlearn was installed (a 10 s probe
timed out under load). These tests run the real probe subprocess against the
project interpreter and pin each skip category end to end.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from gnn.execute.processor.envelope import _make_skipped_result
from gnn.utils.runtime_safety import framework_availability as fa


def _script_info(framework: str) -> dict:
    return {
        "path": Path("model_a") / framework / f"model_a_{framework}.py",
        "name": f"model_a_{framework}.py",
        "framework": framework,
        "executor": sys.executable,
    }


def test_available_framework_reports_no_reason() -> None:
    status = fa.diagnose_framework("jax", executor=sys.executable)
    assert status.available is True
    assert status.reason is None
    assert fa.last_unavailable_status("jax", sys.executable) is None


def test_python_version_gate_is_named_not_called_missing() -> None:
    status = fa.diagnose_framework("ngclearn", executor=sys.executable)
    if sys.version_info[:2] < fa.FRAMEWORK_MIN_PYTHON["ngclearn"]:
        assert status.available is False
        assert status.skip_category == fa.SKIP_PYTHON_TOO_OLD
        assert "requires Python >= 3.12" in (status.reason or "")
        assert f"{sys.version_info[0]}.{sys.version_info[1]}" in (status.reason or "")
    else:
        assert status.skip_category in (None, fa.SKIP_MISSING_MODULE)


def test_probe_timeout_is_its_own_category(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fa, "PROBE_TIMEOUT_SECONDS", 0.001)
    status = fa.diagnose_framework("jax", executor=sys.executable)
    assert status.available is False
    assert status.skip_category == fa.SKIP_PROBE_TIMEOUT
    assert "timed out" in (status.reason or "")
    assert "not installed" not in (status.reason or "")


def test_skipped_envelope_carries_the_diagnosed_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fa, "PROBE_TIMEOUT_SECONDS", 0.001)
    fa.diagnose_framework("jax", executor=sys.executable)
    result = _make_skipped_result(
        _script_info("jax"), "jax", "model_a", sys.executable, logging.getLogger("t")
    )
    assert result["error_type"] == "DependencyNotInstalled"
    assert result["skip_category"] == fa.SKIP_PROBE_TIMEOUT
    assert "timed out" in result["error"]


def test_missing_module_keeps_the_classic_wording() -> None:
    result = _make_skipped_result(
        _script_info("pymdp"),
        "pymdp",
        "model_a",
        "/nonexistent/never-probed-python",
        logging.getLogger("t"),
    )
    assert result["error"] == "Dependency not installed: pymdp"
    assert result["skip_category"] == fa.SKIP_MISSING_MODULE


@pytest.mark.needs_cmdstan
def test_missing_cmdstan_toolchain_is_a_probe_failure_with_toolchain_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CMDSTAN", str(tmp_path / "no-cmdstan-here"))
    status = fa.diagnose_framework("stan", executor=sys.executable)
    assert status.available is False
    assert status.skip_category == fa.SKIP_PROBE_FAILED
    assert "not installed" not in (status.reason or "")
    assert status.install_hint == fa.FRAMEWORK_TOOLCHAIN_HINT["stan"]
