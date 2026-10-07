#!/usr/bin/env python3
"""Headless GR default for every Julia subprocess GNN builds.

Pins the shared seam (``execute.julia_env.julia_subprocess_env``): every
Julia subprocess environment carries ``GKSwstype=100`` (headless GR — no
``gksqt`` Qt window on display-less hosts) unless the caller's environment
explicitly sets ``GKSwstype``. Covers the helper itself, the Step 12
rendered-script env builder, the standalone runner execution paths, and the
Julia package-load probe.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from gnn.execute import julia_env
from gnn.execute import processor as execute_processor
from gnn.execute.activeinference_jl import activeinference_runner
from gnn.execute.julia_env import julia_subprocess_env
from gnn.execute.rxinfer import rxinfer_runner
from gnn.execute.types import ScriptExecutionContext


class _EnvelopeSpy:
    """Capture ``run_subprocess_envelope`` calls; return canned envelopes."""

    def __init__(self, results: Optional[List[Dict[str, Any]]] = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.results = list(results or [])

    def __call__(self, command: List[str], **kwargs: Any) -> Dict[str, Any]:
        self.calls.append({"command": list(command), "kwargs": kwargs})
        queued = self.results.pop(0) if self.results else _canned()
        return dict(queued)


def _canned() -> Dict[str, Any]:
    """Full canned envelope with every key the persistence path reads."""
    return {
        "success": True,
        "return_code": 0,
        "stdout": "ok",
        "stderr": "",
        "duration_seconds": 0.01,
        "sandbox_mode": "off",
        "sandboxed": False,
        "error": None,
        "error_type": None,
    }


# ── Shared helper ──────────────────────────────────────────────────────────


def test_julia_subprocess_env_defaults_to_headless_gr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GKSwstype", raising=False)

    env = julia_subprocess_env()

    assert env["GKSwstype"] == "100"
    assert "PATH" in env  # parent environment preserved


def test_julia_subprocess_env_explicit_caller_override_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GKSwstype", "nehe")

    env = julia_subprocess_env()

    assert env["GKSwstype"] == "nehe"


def test_julia_subprocess_env_overrides_apply_after_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GKSwstype", raising=False)

    env = julia_subprocess_env({"JULIA_PROJECT": "/tmp/project"})

    assert env["JULIA_PROJECT"] == "/tmp/project"
    assert env["GKSwstype"] == "100"


# ── Step 12 rendered-script env builder ────────────────────────────────────


def _rxinfer_context(script: Path) -> ScriptExecutionContext:
    return ScriptExecutionContext(
        script_path=script,
        script_name=script.name,
        framework="rxinfer",
        model_name="model_a",
        executor="julia",
    )


def test_rendered_script_env_defaults_headless(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("GKSwstype", raising=False)
    script = tmp_path / "model_a_rxinfer.jl"

    env = execute_processor._build_execution_environment(
        _rxinfer_context(script), tmp_path / "12"
    )

    assert env["GKSwstype"] == "100"
    assert env["JULIA_PROJECT"].endswith("rxinfer")


def test_rendered_script_env_explicit_override_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GKSwstype", "nehe")
    script = tmp_path / "model_a_rxinfer.jl"

    env = execute_processor._build_execution_environment(
        _rxinfer_context(script), tmp_path / "12"
    )

    assert env["GKSwstype"] == "nehe"


# ── Standalone runner paths ────────────────────────────────────────────────


def test_rxinfer_execution_carries_headless_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = tmp_path / "model_a_rxinfer.jl"
    script.write_text('println("ok")\n')
    monkeypatch.delenv("GKSwstype", raising=False)
    spy = _EnvelopeSpy()
    monkeypatch.setattr(rxinfer_runner, "run_subprocess_envelope", spy)

    ok = rxinfer_runner.execute_rxinfer_script(script, output_dir=tmp_path / "logs")

    assert ok is True
    assert spy.calls[0]["kwargs"]["env"]["GKSwstype"] == "100"


def test_rxinfer_execution_explicit_override_survives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = tmp_path / "model_a_rxinfer.jl"
    script.write_text('println("ok")\n')
    monkeypatch.setenv("GKSwstype", "nehe")
    spy = _EnvelopeSpy()
    monkeypatch.setattr(rxinfer_runner, "run_subprocess_envelope", spy)

    ok = rxinfer_runner.execute_rxinfer_script(script, output_dir=tmp_path / "logs")

    assert ok is True
    assert spy.calls[0]["kwargs"]["env"]["GKSwstype"] == "nehe"


def test_activeinference_execution_headless_with_julia_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_dir = tmp_path / "activeinference_jl"
    project_dir.mkdir()
    script = project_dir / "model_a_activeinference.jl"
    script.write_text('println("ok")\n')
    monkeypatch.delenv("GKSwstype", raising=False)
    spy = _EnvelopeSpy()
    monkeypatch.setattr(activeinference_runner, "run_subprocess_envelope", spy)

    ok = activeinference_runner.execute_activeinference_script(
        script, setup_environment=False
    )

    assert ok is True
    env = spy.calls[0]["kwargs"]["env"]
    assert env["JULIA_PROJECT"] == str(project_dir)
    assert env["GKSwstype"] == "100"


# ── Julia package-load probe ───────────────────────────────────────────────


def test_julia_package_probe_runs_headless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []

    from gnn.execute import julia_setup
    from gnn.utils.runtime_safety import framework_availability as availability

    def fake_run(cmd: List[str], **kwargs: Any) -> Dict[str, Any]:
        calls.append({"cmd": cmd, "kwargs": kwargs})
        return {
            "success": True,
            "return_code": 0,
            "stdout": 'GNN_FRAMEWORK_STATUS:{"available":true}',
        }

    monkeypatch.setattr(availability, "_run_probe_envelope", fake_run)
    monkeypatch.setattr(julia_setup, "julia_executable", lambda: "/test/julia")

    assert julia_env.check_julia_dependencies(False) is True

    probe_calls = [c for c in calls if "-e" in c["cmd"]]
    assert probe_calls, "expected the package-load probe to run"
    assert probe_calls[0]["kwargs"]["env"]["GKSwstype"] == "100"
