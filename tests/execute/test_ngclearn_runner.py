#!/usr/bin/env python3
"""ngc-learn executor: discovery, envelope wiring, env routing, skip semantics.

The shared environment is py3.11, where the marker-gated ``ngclearn`` extra
resolves empty, so every runtime path must skip (never fail). Tests that need
the "installed" state simulate it with monkeypatching to stay deterministic on
both python splits.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from gnn.utils.runtime_safety.framework_availability import FrameworkStatus

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnn.execute import executor as executor_module  # noqa: E402
from gnn.execute import planning as execute_planning  # noqa: E402
from gnn.execute import processor as execute_processor  # noqa: E402
from gnn.execute.executor import execute_rendered_simulators  # noqa: E402
from gnn.execute.ngclearn import ngclearn_runner  # noqa: E402
from gnn.execute.ngclearn.ngclearn_runner import (  # noqa: E402
    execute_ngclearn_script,
    find_ngclearn_scripts,
    run_ngclearn_scripts,
)
from gnn.execute.types import ScriptExecutionContext  # noqa: E402
from gnn.utils.runtime_safety.framework_availability import (  # noqa: E402
    FRAMEWORK_IMPORT_CHECK,
)


def _write_render_script(
    root: Path, model: str = "m", name: str = "m_ngclearn.py"
) -> Path:
    """Create a rendered script under ``<root>/<model>/ngclearn/`` (generic
    Step-12 discovery layout)."""
    script = root / model / "ngclearn" / name
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("print('ok')\n")
    return script


def _write_runner_lane_script(
    root: Path, model: str = "model_a", name: str = "a_ngclearn.py"
) -> Path:
    """Create a rendered script under ``<root>/ngclearn/<model>/`` — the
    layout the registry-runner lane scans (``rendered_dir/ngclearn``)."""
    script = root / "ngclearn" / model / name
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("print('ok')\n")
    return script


def _envelope(**overrides: Any) -> Dict[str, Any]:
    envelope: Dict[str, Any] = {
        "success": True,
        "return_code": 0,
        "stdout": "ok",
        "stderr": "",
        "duration_seconds": 0.01,
    }
    envelope.update(overrides)
    return envelope


class _ExecuteSafelySpy:
    """Capture ``execute_script_safely`` calls; return the queued envelope."""

    def __init__(self, result: Optional[Dict[str, Any]] = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.result = result or _envelope()

    def __call__(self, script_path: Any, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append({"script_path": script_path, "kwargs": kwargs})
        return dict(self.result)


@pytest.fixture
def fake_ngclearn_import(monkeypatch: Any) -> None:
    """Control the bounded diagnosis, never import optional backends in parent."""
    monkeypatch.setattr(
        ngclearn_runner,
        "check_framework",
        lambda *args, **kwargs: FrameworkStatus("ngclearn", True),
    )


# ── Discovery and shared probe registration ────────────────────────────────


def test_find_ngclearn_scripts_matches_only_ngclearn_render_dirs(
    tmp_path: Path,
) -> None:
    script = _write_render_script(tmp_path, "model_a", "a_ngclearn.py")
    stray = tmp_path / "model_b" / "jax" / "b_jax.py"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_text("print('ok')\n")

    assert find_ngclearn_scripts(tmp_path) == [script]


def test_ngclearn_import_check_is_registered() -> None:
    """The shared probe maps ngclearn → the ngclearn module + install hint."""
    assert FRAMEWORK_IMPORT_CHECK["ngclearn"] == (
        "ngclearn",
        "uv sync --extra ngclearn",
    )


# ── Envelope wiring and env routing (direct runner) ────────────────────────


def test_run_delegates_to_execute_script_safely(
    tmp_path: Path, monkeypatch: Any, fake_ngclearn_import: None
) -> None:
    script = _write_runner_lane_script(tmp_path)
    monkeypatch.setattr(ngclearn_runner, "is_ngclearn_available", lambda: True)
    spy = _ExecuteSafelySpy()
    monkeypatch.setattr(executor_module, "execute_script_safely", spy)

    out_dir = tmp_path / "exec"
    assert run_ngclearn_scripts(tmp_path, out_dir) is True

    assert len(spy.calls) == 1
    assert spy.calls[0]["script_path"] == script.resolve()
    assert spy.calls[0]["kwargs"]["cwd"] == script.parent
    # The canonical safe executor owns timing + log persistence receipts.
    log_file = out_dir / "a_ngclearn" / "execution_log.json"
    assert json.loads(log_file.read_text())["script"] == str(script.resolve())


def test_execute_script_sets_ngclearn_output_dir_env(
    tmp_path: Path, monkeypatch: Any, fake_ngclearn_import: None
) -> None:
    script = _write_runner_lane_script(tmp_path)
    monkeypatch.setattr(ngclearn_runner, "is_ngclearn_available", lambda: True)
    spy = _ExecuteSafelySpy()
    monkeypatch.setattr(executor_module, "execute_script_safely", spy)

    out_dir = tmp_path / "exec" / "a_ngclearn"
    assert execute_ngclearn_script(script, output_dir=out_dir) is True

    env = spy.calls[0]["kwargs"]["env"]
    assert env["NGCLEARN_OUTPUT_DIR"] == str(out_dir)
    assert out_dir.is_dir()


def test_run_fails_closed_when_availability_probe_is_false(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _write_runner_lane_script(tmp_path)
    monkeypatch.setattr(ngclearn_runner, "is_ngclearn_available", lambda: False)
    spy = _ExecuteSafelySpy()
    monkeypatch.setattr(executor_module, "execute_script_safely", spy)

    assert run_ngclearn_scripts(tmp_path) is False
    assert spy.calls == []


# ── Step 12 wiring ─────────────────────────────────────────────────────────


def test_step12_execution_environment_sets_ngclearn_output_dir(
    tmp_path: Path,
) -> None:
    script = _write_render_script(tmp_path, "model_a", "a_ngclearn.py")
    context = ScriptExecutionContext(
        script_path=script,
        script_name=script.name,
        framework="ngclearn",
        model_name="model_a",
        executor=sys.executable,
    )

    env = execute_processor._build_execution_environment(context, tmp_path / "12")

    simulation_data = tmp_path / "12" / "model_a" / "ngclearn" / "simulation_data"
    assert env["NGCLEARN_OUTPUT_DIR"] == str(simulation_data)
    assert simulation_data.is_dir()


def test_step12_preflight_skips_ngclearn_scripts_without_module(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The documented skip story: ngclearn scripts skip at the Python
    pre-flight check when the module is absent."""
    script = _write_render_script(tmp_path, "model_a", "a_ngclearn.py")
    script_info = {
        "path": script,
        "name": script.name,
        "framework": "ngclearn",
        "executor": sys.executable,
        "relative_path": script,
        "size_bytes": script.stat().st_size,
    }
    monkeypatch.setattr(
        execute_processor,
        "_check_framework_by_name",
        lambda *a, **k: FrameworkStatus(
            "ngclearn",
            False,
            "ngclearn",
            reason_code="missing_module",
            reason="Dependency not installed: ngclearn",
        ),
    )

    result = execute_processor.execute_single_script(
        script_info, tmp_path / "12", False, execute_processor.logger, 60
    )

    assert result["skipped"] is True
    assert result["success"] is False
    assert result["status"] == "skipped"
    assert result["error_type"] == "DependencyNotInstalled"
    assert "ngclearn" in result["error"]


def test_executor_reports_ngclearn_skipped_when_unavailable(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A missing runtime retains its skip receipt and unfinished required work.

    A nonempty selected batch cannot report success without executing its
    required scripts. Readiness is controlled to make the proof deterministic.
    """
    _write_render_script(tmp_path, "model_a", "a_ngclearn.py")

    def _all_unavailable(framework_dir_key: str) -> executor_module._RunnerState:
        return executor_module._RunnerState(False, None)

    monkeypatch.setattr(executor_module, "_runner_state", _all_unavailable)
    output_dir = tmp_path / "out"
    logger = logging.getLogger("test_ngclearn_runner")

    result = execute_rendered_simulators(
        target_dir=tmp_path,
        output_dir=output_dir,
        logger=logger,
        recursive=False,
        verbose=False,
    )

    summary_file = (
        output_dir / "12_execute_output" / "summaries" / "execution_summary.json"
    )
    summary = json.loads(summary_file.read_text())
    ngclearn_records = summary["ngclearn_executions"]
    assert len(ngclearn_records) == 1
    assert ngclearn_records[0]["status"] == "SKIPPED"
    assert "uv sync --extra ngclearn" in ngclearn_records[0]["message"]
    assert summary["total_failures"] == 0
    assert summary["required_unfinished"] == 1
    assert ngclearn_records[0]["required_work"] is True
    assert result is False


def test_plan_disposition_marks_ngclearn_dependency_skip(monkeypatch: Any) -> None:
    script_info = {
        "framework": "ngclearn",
        "executor": sys.executable,
        "path": "/render/m/ngclearn/m_ngclearn.py",
        "name": "m_ngclearn.py",
    }
    monkeypatch.setattr(
        execute_planning, "is_framework_available", lambda *a, **k: False
    )
    assert execute_planning._disposition(script_info) == "skip_dependency"
    monkeypatch.setattr(
        execute_planning, "is_framework_available", lambda *a, **k: True
    )
    assert execute_planning._disposition(script_info) == "execute"


@pytest.mark.parametrize(
    "reason_code",
    ["unsupported_python", "missing_module", "probe_failed", "probe_timeout"],
)
def test_direct_runner_uses_structured_probe_without_parent_dependency_import(
    tmp_path: Path, monkeypatch: Any, reason_code: str
) -> None:
    script = _write_render_script(tmp_path)
    diagnosis = FrameworkStatus(
        "ngclearn", False, reason_code=reason_code, reason="specific readiness cause"
    )
    calls = []

    def probe(*args, **kwargs):
        calls.append((args, kwargs))
        return diagnosis

    monkeypatch.setattr(ngclearn_runner, "check_framework", probe)
    assert ngclearn_runner.is_ngclearn_available() is False
    assert execute_ngclearn_script(script, output_dir=tmp_path / "receipt") is False
    receipt = json.loads((tmp_path / "receipt/execution_log.json").read_text())
    assert receipt["status"] == "skipped" and receipt["reason_code"] == reason_code
    assert receipt["reason"] == diagnosis.reason
    assert calls[1][1]["executor"] == sys.executable
