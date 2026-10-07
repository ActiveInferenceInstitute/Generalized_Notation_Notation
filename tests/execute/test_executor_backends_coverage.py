"""Coverage pins for the stan and bnlearn backend wiring in the executor.

Covers the registry/spec alignment for the two added backends, the
skip-not-fail receipts they produce in ``execute_rendered_simulators``, the
new dispatch methods' envelope contracts, and the PyMDP runner's aligned
failure semantics (any script failure fails the run).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from gnn.execute import executor as executor_module
from gnn.execute.executor import GNNExecutor, execute_rendered_simulators
from gnn.execute.pymdp import pymdp_runner


def _fake_runner_state(
    available_by_key: Dict[str, bool],
) -> Callable[[str], Any]:
    """Stub ``_runner_state`` with controlled availability per backend key."""

    def state(framework_dir_key: str) -> Any:
        if available_by_key.get(framework_dir_key):
            return executor_module._RunnerState(True, lambda **kwargs: True)
        return executor_module._RunnerState(False, None)

    return state


def _canned_envelope() -> Dict[str, Any]:
    """Envelope shape matching a completed successful subprocess run."""
    return {
        "success": True,
        "return_code": 0,
        "stdout": "ok",
        "stderr": "",
        "duration_seconds": 0.01,
        "error": None,
        "error_type": None,
    }


class _EnvelopeSpy:
    """Capture ``run_subprocess_envelope`` calls; return queued envelopes."""

    def __init__(self, results: Optional[List[Dict[str, Any]]] = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.results = list(results or [])

    def __call__(self, command: List[str], **kwargs: Any) -> Dict[str, Any]:
        self.calls.append({"command": list(command), "kwargs": kwargs})
        queued = self.results.pop(0) if self.results else _canned_envelope()
        return dict(queued)


def _spy(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    results: Optional[List[Dict[str, Any]]] = None,
) -> _EnvelopeSpy:
    spy = _EnvelopeSpy(results)
    monkeypatch.setattr(target, spy)
    return spy


# ── registry/spec alignment ────────────────────────────────────────────────


def test_stan_and_bnlearn_specs_are_availability_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = [
        spec.framework_dir_key
        for spec in executor_module._framework_specs(resolve_availability=False)
    ]
    assert set(keys) == set(executor_module.FRAMEWORK_DIR_NAMES)
    assert keys[-2:] == ["stan", "bnlearn"]

    monkeypatch.setattr(executor_module, "_runner_state", _fake_runner_state({}))
    specs = {s.framework_dir_key: s for s in executor_module._framework_specs()}
    assert specs["stan"].available is False
    assert specs["bnlearn"].available is False
    assert callable(specs["bnlearn"].runner)
    assert "uv sync --extra bnlearn" in specs["bnlearn"].unavailable_message
    assert "BNLEARN_OUTPUT_DIR" in specs["bnlearn"].unavailable_message

    monkeypatch.setattr(
        executor_module,
        "_runner_state",
        _fake_runner_state({"stan": True}),
    )
    specs = {s.framework_dir_key: s for s in executor_module._framework_specs()}
    assert specs["stan"].available is True
    assert specs["stan"].runner is executor_module._run_stan_registry


# ── execute_rendered_simulators receipts ───────────────────────────────────


def _run_empty_render_tree(tmp_path: Path) -> Dict[str, Any]:
    render_dir = tmp_path / "11_render_output"
    render_dir.mkdir()
    output_dir = tmp_path / "out"

    assert execute_rendered_simulators(
        target_dir=render_dir,
        output_dir=output_dir,
        logger=logging.getLogger("test_executor_backends_coverage"),
        recursive=False,
        verbose=False,
    )
    summary_file = (
        output_dir / "12_execute_output" / "summaries" / "execution_summary.json"
    )
    return json.loads(summary_file.read_text())


def test_empty_render_tree_yields_skip_receipts_for_new_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor_module, "_runner_state", _fake_runner_state({}))

    summary = _run_empty_render_tree(tmp_path)

    stan_records = summary["stan_executions"]
    assert len(stan_records) == 1 and stan_records[0]["status"] == "SKIPPED"
    bnlearn_records = summary["bnlearn_executions"]
    assert len(bnlearn_records) == 1 and bnlearn_records[0]["status"] == "SKIPPED"
    assert summary["total_failures"] == 0


def test_stan_available_empty_tree_reports_skip_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        executor_module, "_runner_state", _fake_runner_state({"stan": True})
    )
    monkeypatch.setattr(
        "gnn.execute.stan.stan_runner.run_stan_scripts", lambda **kwargs: []
    )

    summary = _run_empty_render_tree(tmp_path)

    stan_records = summary["stan_executions"]
    assert len(stan_records) == 1 and stan_records[0]["status"] == "SKIPPED"
    bnlearn_records = summary["bnlearn_executions"]
    assert len(bnlearn_records) == 1 and bnlearn_records[0]["status"] == "SKIPPED"
    assert summary["total_failures"] == 0


# ── dispatch method contracts ──────────────────────────────────────────────


def test_activeinference_dispatch_command_shape_and_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_dir = tmp_path / "activeinference_jl"
    script_dir.mkdir()
    script = script_dir / "m_ai.jl"
    script.write_text("println(1)\n")
    canned = _canned_envelope()
    spy = _spy(monkeypatch, "gnn.execute.executor.run_subprocess_envelope", [canned])

    result = GNNExecutor()._execute_activeinference_script(str(script))

    project_dir = Path(executor_module.__file__).parent / "activeinference_jl"
    assert spy.calls[0]["command"] == [
        "julia",
        f"--project={project_dir}",
        str(script),
    ]
    assert spy.calls[0]["kwargs"]["timeout"] == 600
    assert spy.calls[0]["kwargs"]["cwd"] == str(script.parent)
    assert spy.calls[0]["kwargs"]["env"]["JULIA_PROJECT"] == str(project_dir)
    assert result == canned


@pytest.mark.parametrize(
    "dispatch_name",
    ["_execute_numpyro_script", "_execute_pytorch_script", "_execute_ngclearn_script"],
)
def test_python_framework_dispatches_run_sys_executable_timeout_300(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dispatch_name: str,
) -> None:
    script = tmp_path / "m.py"
    script.write_text("print('ok')\n")
    canned = _canned_envelope()
    spy = _spy(monkeypatch, "gnn.execute.executor.run_subprocess_envelope", [canned])

    result = getattr(GNNExecutor(), dispatch_name)(str(script))

    assert spy.calls[0]["command"] == [sys.executable, str(script)]
    assert spy.calls[0]["kwargs"]["timeout"] == 300
    assert result == canned


def test_stan_dispatch_skips_without_cmdstanpy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.utils.runtime_safety.framework_availability import FrameworkStatus

    monkeypatch.setattr(
        "gnn.execute.stan.stan_runner._check_stan_status",
        lambda *args, **kwargs: FrameworkStatus(
            "stan", False, reason_code="missing_module"
        ),
    )
    script = tmp_path / "m_stan.py"
    script.write_text("pass\n")

    result = GNNExecutor()._execute_stan_script(str(script))

    assert not result["success"] and result["skipped"] is True
    assert result["status"] == "skipped"
    assert result["reason_code"] == "missing_module"
    assert result["error_type"] == "DependencyUnavailable"


def test_stan_dispatch_available_routes_execute_stan_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "m_stan.py"
    script.write_text("pass\n")
    canned: Dict[str, Any] = {
        "script": str(script),
        "framework": "stan",
        "success": True,
    }
    captured: Dict[str, Any] = {}

    def fake_execute_stan_script(
        script_path: Any,
        output_dir: Any,
        timeout: int = 1800,
        python_executable: Optional[str] = None,
        cancel_token=None,
    ) -> Dict[str, Any]:
        captured["script_path"] = script_path
        captured["output_dir"] = output_dir
        captured["timeout"] = timeout
        return dict(canned)

    monkeypatch.setattr(
        executor_module, "_runner_state", _fake_runner_state({"stan": True})
    )
    monkeypatch.setattr(
        "gnn.execute.stan.stan_runner.execute_stan_script", fake_execute_stan_script
    )

    result = GNNExecutor()._execute_stan_script(str(script))

    assert result == canned
    assert Path(captured["output_dir"]) == script.parent
    assert captured["timeout"] == 1800


def test_bnlearn_dispatch_without_script_returns_skip_receipt() -> None:
    result = GNNExecutor()._execute_bnlearn_script()

    assert result["success"] is False
    assert result["skipped"] is True
    assert result["status"] == "skipped"
    assert "Supply a rendered bnlearn script" in result["error"]
    assert "BNLEARN_OUTPUT_DIR" in result["error"]


# ── pymdp runner semantics ─────────────────────────────────────────────────


def _run_pymdp(render_root: Path, monkeypatch: pytest.MonkeyPatch) -> bool:
    monkeypatch.setattr(
        pymdp_runner,
        "execute_pymdp_script_with_outputs",
        lambda *args, **kwargs: {"success": False},
    )
    return pymdp_runner.run_pymdp_scripts(
        rendered_simulators_dir=str(render_root),
        execution_output_dir=str(render_root / "out"),
        recursive_search=False,
        verbose=False,
        timeout=60,
    )


def test_pymdp_runner_fails_when_all_scripts_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pymdp_dir = tmp_path / "pymdp"
    pymdp_dir.mkdir()
    (pymdp_dir / "broken.py").write_text("raise RuntimeError('boom')\n")

    assert _run_pymdp(tmp_path, monkeypatch) is False


def test_pymdp_runner_fails_on_mixed_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One success plus one failure is a failed run (aligned with siblings)."""
    pymdp_dir = tmp_path / "pymdp"
    pymdp_dir.mkdir()
    (pymdp_dir / "ok.py").write_text("print('ok')\n")
    (pymdp_dir / "broken.py").write_text("raise RuntimeError('boom')\n")

    outcomes = iter([{"success": True}, {"success": False}])
    monkeypatch.setattr(
        pymdp_runner,
        "execute_pymdp_script_with_outputs",
        lambda *args, **kwargs: next(outcomes),
    )

    assert (
        pymdp_runner.run_pymdp_scripts(
            rendered_simulators_dir=str(tmp_path),
            execution_output_dir=str(tmp_path / "out"),
            recursive_search=False,
            verbose=False,
            timeout=60,
        )
        is False
    )


def test_pymdp_runner_succeeds_when_no_scripts_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run_pymdp(tmp_path, monkeypatch) is True


@pytest.mark.parametrize(
    ("reason_code", "execution_error", "cleanup", "status", "error_type"),
    [
        ("missing_module", None, None, "skipped", "DependencyUnavailable"),
        ("probe_timeout", "TimeoutExpired", True, "timed_out", "TimeoutExpired"),
        ("probe_failed", "TimeoutExpired", False, "failed", "ProcessCleanupFailure"),
        ("probe_failed", None, True, "failed", "FrameworkProbeFailure"),
    ],
)
def test_cpomdp_direct_diagnosis_preserves_failed_work(
    tmp_path, monkeypatch, reason_code, execution_error, cleanup, status, error_type
):
    from gnn.execute.executor import GNNExecutor
    from gnn.utils.runtime_safety import framework_availability

    diagnosis = framework_availability.FrameworkStatus(
        "cpomdp",
        False,
        reason_code=reason_code,
        execution_error_type=execution_error,
        cleanup_verified=cleanup,
    )
    monkeypatch.setattr(
        framework_availability, "check_framework", lambda *args, **kwargs: diagnosis
    )
    script = tmp_path / "cpomdp.py"
    script.write_text("raise AssertionError('unavailable script must not run')")
    result = GNNExecutor(output_dir=tmp_path / "out").execute_gnn_model(
        str(script), execution_type="cpomdp", timeout=3
    )
    assert not result["success"]
    assert result["status"] == status
    assert result["error_type"] == error_type
    assert result["reason_code"] == reason_code


def test_batch_readiness_census_uses_bounded_shared_diagnoses(monkeypatch):
    import logging

    from gnn.execute.executor_report import (
        _check_julia_availability,
        _check_python_dependencies,
    )
    from gnn.utils.runtime_safety import framework_availability

    seen = []

    def unavailable(name, **kwargs):
        seen.append(name)
        return framework_availability.FrameworkStatus(
            name, False, reason_code="probe_timeout", reason="probe deadline exhausted"
        )

    monkeypatch.setattr(framework_availability, "check_framework", unavailable)
    results = {"dependency_issues": []}
    _check_python_dependencies(results, logging.getLogger(__name__))
    _check_julia_availability(results, logging.getLogger(__name__))
    assert seen == ["jax", "pymdp", "rxinfer", "activeinference_jl"]
    assert all(
        row["reason_code"] == "probe_timeout" for row in results["readiness"].values()
    )
    assert len(results["dependency_issues"]) == 4
