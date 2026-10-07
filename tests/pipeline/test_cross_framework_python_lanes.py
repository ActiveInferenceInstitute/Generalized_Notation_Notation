#!/usr/bin/env python3
"""Unit tests for the Python cross-framework comparison lanes (JAX/PyTorch/NumPyro).

Covers the shared ``_execute_python_lane`` body through each thin lane
function: the missing-dependency probe and the renderer import failure both
produce ``unavailable`` skip receipts (never execution failures), the happy
path renders ``model_<framework>.py`` and executes it under ``sys.executable``
with the framework's output environment variable pointed at ``fw_dir``, and
the rendered comparison page carries one column per registered framework.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import gnn.analysis.rxinfer.cross_framework as cross_framework
from gnn.analysis.rxinfer.cross_framework import (
    FRAMEWORKS,
    PROJECT_ROOT,
    SRC_ROOT,
    FrameworkRun,
    _execute_jax,
    _execute_numpyro,
    _execute_pymdp,
    _execute_pytorch,
    render_comparison_html,
)

# framework -> (lane callable, output env var, renderer module path,
# renderer function name, detail emitted by the missing-dependency probe)
PYTHON_LANES: dict[str, tuple[Any, str, str, str, str]] = {
    "jax": (
        _execute_jax,
        "GNN_OUTPUT_DIR",
        "gnn.render.jax.jax_renderer",
        "render_gnn_to_jax",
        "jax not installed (uv sync)",
    ),
    "pytorch": (
        _execute_pytorch,
        "PYTORCH_OUTPUT_DIR",
        "gnn.render.pytorch.pytorch_renderer",
        "render_gnn_to_pytorch",
        "torch not installed (uv sync --extra torch)",
    ),
    "numpyro": (
        _execute_numpyro,
        "NUMPYRO_OUTPUT_DIR",
        "gnn.render.numpyro.numpyro_renderer",
        "render_gnn_to_numpyro",
        "numpyro not installed (uv sync)",
    ),
}


def _always_available(framework: str) -> bool:
    return True


def _never_available(framework: str) -> bool:
    return False


def _fake_render(
    spec: dict[str, Any], output_path: Path, options: Any = None
) -> tuple[bool, str, list[str]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("# generated for test\n", encoding="utf-8")
    return True, "ok", []


class TestUnavailableReceipts:
    """Missing dependencies and renderer import failures are skip receipts."""

    @pytest.mark.unit
    @pytest.mark.parametrize("framework", ["jax", "pytorch", "numpyro"])
    def test_missing_dependency_probe_yields_unavailable_before_render(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, framework: str
    ) -> None:
        execute, _env_var, _module, _fn, detail = PYTHON_LANES[framework]
        monkeypatch.setattr(cross_framework, "is_framework_available", _never_available)

        fw_dir = tmp_path / framework
        run = execute({}, fw_dir)

        assert run.framework == framework
        assert run.status == "unavailable"
        assert run.detail == detail
        assert run.results is None
        # The probe runs before rendering: no script is ever written.
        assert not fw_dir.exists() or not any(fw_dir.iterdir())

    @pytest.mark.unit
    @pytest.mark.parametrize("framework", ["jax", "pytorch", "numpyro"])
    def test_renderer_import_failure_yields_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, framework: str
    ) -> None:
        execute, _env_var, module, _fn, _detail = PYTHON_LANES[framework]
        monkeypatch.setattr(
            cross_framework, "is_framework_available", _always_available
        )
        # A None module entry makes the from-import raise ImportError.
        monkeypatch.setitem(sys.modules, module, None)

        run = execute({}, tmp_path / framework)

        assert run.status == "unavailable"
        assert "not importable" in run.detail


class TestHappyPath:
    """A present dependency renders the script and runs it under sys.executable."""

    @pytest.mark.unit
    @pytest.mark.parametrize("framework", ["jax", "pytorch", "numpyro"])
    def test_lane_renders_then_runs_subprocess_with_env_redirect(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, framework: str
    ) -> None:
        execute, env_var, module, fn_name, _detail = PYTHON_LANES[framework]
        monkeypatch.setattr(
            cross_framework, "is_framework_available", _always_available
        )
        renderer_module = importlib.import_module(module)
        monkeypatch.setattr(renderer_module, fn_name, _fake_render)

        captured: dict[str, Any] = {}

        def fake_run_subprocess(
            framework: str,
            command: list[str],
            cwd: Path,
            timeout: int,
            results_path: Path,
            script_path: Path,
            env: dict[str, str] | None = None,
            runtime: Any = None,
        ) -> FrameworkRun:
            captured.update(
                {
                    "framework": framework,
                    "command": command,
                    "cwd": cwd,
                    "timeout": timeout,
                    "results_path": results_path,
                    "script_path": script_path,
                    "env": env or {},
                }
            )
            return FrameworkRun(framework, "success", "ok")

        monkeypatch.setattr(cross_framework, "_run_subprocess", fake_run_subprocess)

        fw_dir = tmp_path / framework
        run = execute({}, fw_dir, timeout=99)

        assert run.status == "success"
        assert captured["framework"] == framework
        script = fw_dir / f"model_{framework}.py"
        assert captured["command"] == [sys.executable, str(script)]
        assert captured["cwd"] == fw_dir
        assert captured["timeout"] == 99
        assert captured["results_path"] == fw_dir / "simulation_results.json"
        assert captured["script_path"] == script
        assert captured["env"][env_var] == str(fw_dir)
        pythonpath = captured["env"]["PYTHONPATH"].split(os.pathsep)
        assert str(PROJECT_ROOT) in pythonpath
        assert str(SRC_ROOT) in pythonpath

    @pytest.mark.unit
    def test_pymdp_lane_keeps_its_env_contract(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The PyMDP lane still redirects results and pins the project root."""
        monkeypatch.setattr(cross_framework, "is_framework_available", _never_available)
        # The pymdp lane disables the probe; inject a fake renderer module so
        # the lane runs without the pymdp dependency installed.
        stub = types.ModuleType("gnn.render.pymdp.pymdp_renderer")
        stub.render_gnn_to_pymdp = _fake_render  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "gnn.render.pymdp.pymdp_renderer", stub)

        captured: dict[str, Any] = {}

        def fake_run_subprocess(
            framework: str,
            command: list[str],
            cwd: Path,
            timeout: int,
            results_path: Path,
            script_path: Path,
            env: dict[str, str] | None = None,
            runtime: Any = None,
        ) -> FrameworkRun:
            captured["env"] = env or {}
            return FrameworkRun(framework, "success", "ok")

        monkeypatch.setattr(cross_framework, "_run_subprocess", fake_run_subprocess)

        fw_dir = tmp_path / "pymdp"
        run = _execute_pymdp({}, fw_dir)

        assert run.status == "success"
        assert captured["env"]["PYMDP_OUTPUT_DIR"] == str(fw_dir)
        assert captured["env"]["GNN_PROJECT_ROOT"] == str(PROJECT_ROOT)


class TestComparisonPage:
    """The comparison page renders one column per registered framework."""

    @pytest.mark.unit
    def test_renders_all_six_framework_columns(self, tmp_path: Path) -> None:
        runs = [
            FrameworkRun(
                framework,
                "success",
                "ok",
                {
                    "framework": framework,
                    "beliefs": [[0.6, 0.4], [0.3, 0.7]],
                    "validation": {"all_valid": True},
                },
            )
            for framework in FRAMEWORKS
        ]
        out_path = tmp_path / "model_comparison.html"

        rendered = render_comparison_html("model", runs, out_path)

        assert rendered == str(out_path)
        text = out_path.read_text(encoding="utf-8")
        assert text.count("<th>") == len(FRAMEWORKS) + 1  # metric header column
        for framework in FRAMEWORKS:
            assert f"<th>{framework}</th>" in text
        # Every successful run contributes a colour-coded legend entry.
        for framework in FRAMEWORKS:
            assert f">{framework}</span>" in text
