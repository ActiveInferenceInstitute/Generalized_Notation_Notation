#!/usr/bin/env python3
"""bnlearn executor: language-aware discovery, probes, and skip semantics.

All tests are offline — subprocess envelopes and runtime probes are
monkeypatched, so neither the Python ``bnlearn`` module nor R/Rscript is
required. The Step 12 wiring points (execution environment, planner
disposition, pre-flight skip) are exercised directly.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from gnn.utils.runtime_safety.framework_availability import FrameworkStatus

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnn.execute import planning as execute_planning  # noqa: E402
from gnn.execute import processor as execute_processor  # noqa: E402
from gnn.execute.bnlearn import bnlearn_runner  # noqa: E402
from gnn.execute.bnlearn.bnlearn_runner import (  # noqa: E402
    OUTPUT_ENV_VAR,
    execute_bnlearn_script,
    find_bnlearn_scripts,
    is_bnlearn_available,
    is_r_bnlearn_available,
    run_bnlearn_scripts,
    script_language,
)
from gnn.execute.types import ScriptExecutionContext  # noqa: E402
from gnn.utils.runtime_safety.framework_availability import (  # noqa: E402
    FRAMEWORK_IMPORT_CHECK,
)


def _write_render_script(
    root: Path, model: str = "m", name: str = "m_bnlearn.py"
) -> Path:
    """Create a rendered script under ``<root>/<model>/bnlearn/``."""
    script = root / model / "bnlearn" / name
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
        "sandbox_mode": "off",
        "sandboxed": False,
    }
    envelope.update(overrides)
    return envelope


class _EnvelopeSpy:
    """Capture ``run_subprocess_envelope`` calls; return queued envelopes."""

    def __init__(self, results: Optional[List[Dict[str, Any]]] = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.results = list(results or [])

    def __call__(self, command: List[str], **kwargs: Any) -> Dict[str, Any]:
        self.calls.append({"command": list(command), "kwargs": kwargs})
        queued = self.results.pop(0) if self.results else _envelope()
        return dict(queued)


# ── Discovery and language detection ───────────────────────────────────────


def test_find_bnlearn_scripts_matches_only_bnlearn_render_dirs(
    tmp_path: Path,
) -> None:
    py_script = _write_render_script(tmp_path, "model_a", "a_bnlearn.py")
    r_script = _write_render_script(tmp_path, "model_a", "b.R")
    (tmp_path / "model_a" / "jax").mkdir()
    (tmp_path / "model_a" / "jax" / "other_stan.py").write_text("print()\n")
    (tmp_path / "model_a" / "jax" / "x.py").write_text("print()\n")
    (tmp_path / "model_a" / "pymdp").mkdir()
    (tmp_path / "model_a" / "pymdp" / "y.py").write_text("print()\n")

    assert find_bnlearn_scripts(tmp_path) == sorted([py_script, r_script])


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("m_bnlearn.py", "python"),
        ("m.R", "r"),
        ("m.r", "r"),
        ("m.jl", "unknown"),
        ("m", "unknown"),
    ],
)
def test_script_language_derived_from_suffix(name: str, expected: str) -> None:
    assert script_language(Path(name)) == expected


# ── Availability probes ────────────────────────────────────────────────────


def test_bnlearn_import_check_is_registered() -> None:
    """The shared probe maps bnlearn → the bnlearn module + install hint."""
    assert FRAMEWORK_IMPORT_CHECK["bnlearn"] == ("bnlearn", "uv sync --extra bnlearn")


def test_python_probe_delegates_to_shared_availability(monkeypatch: Any) -> None:
    seen: Dict[str, Any] = {}

    def _fake_probe(
        framework: str,
        executor: Optional[str] = None,
        logger: Any = None,
        timeout: float = 30,
    ) -> FrameworkStatus:
        seen["framework"] = framework
        seen["executor"] = executor
        return FrameworkStatus(framework, True)

    monkeypatch.setattr(bnlearn_runner, "check_framework", _fake_probe)
    assert is_bnlearn_available("python3") is True
    assert seen == {"framework": "bnlearn", "executor": "python3"}


def test_r_probe_requires_rscript_on_path(monkeypatch: Any) -> None:
    monkeypatch.setattr(bnlearn_runner.shutil, "which", lambda name: None)
    assert is_r_bnlearn_available() is False


def test_r_probe_probes_library_load(monkeypatch: Any) -> None:
    monkeypatch.setattr(bnlearn_runner.shutil, "which", lambda name: "/usr/bin/Rscript")
    spy = _EnvelopeSpy([_envelope()])
    monkeypatch.setattr(bnlearn_runner, "run_subprocess_envelope", spy)
    assert is_r_bnlearn_available() is True
    call = spy.calls[0]
    assert call["command"][0] == "/usr/bin/Rscript"
    assert "library(bnlearn)" in call["command"][2]
    assert call["kwargs"].get("sandbox", True) is False


# ── Direct runner records ──────────────────────────────────────────────────


def test_execute_python_script_success_sets_output_env(
    tmp_path: Path, monkeypatch: Any
) -> None:
    script = _write_render_script(tmp_path, "model_a", "a_bnlearn.py")
    out_dir = tmp_path / "exec" / "model_a"
    spy = _EnvelopeSpy()
    monkeypatch.setattr(bnlearn_runner, "run_subprocess_envelope", spy)
    monkeypatch.setattr(
        bnlearn_runner,
        "_check_bnlearn_status",
        lambda executor=None, timeout=30: FrameworkStatus("bnlearn", True),
    )

    record = execute_bnlearn_script(script, out_dir)

    assert record["success"] is True
    assert record["skipped"] is False
    assert record["framework"] == "bnlearn"
    assert record["language"] == "python"
    assert record["return_code"] == 0
    assert out_dir.is_dir()
    call = spy.calls[0]
    assert call["command"] == [sys.executable, str(script)]
    assert call["kwargs"]["env"] == {OUTPUT_ENV_VAR: str(out_dir)}
    assert call["kwargs"]["cwd"] == str(out_dir)
    # Sandbox semantics delegated to the shared envelope (default enabled).
    assert call["kwargs"].get("sandbox", True) is True


def test_execute_python_script_skips_when_module_missing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    script = _write_render_script(tmp_path, "model_a", "a_bnlearn.py")
    spy = _EnvelopeSpy()
    monkeypatch.setattr(bnlearn_runner, "run_subprocess_envelope", spy)
    monkeypatch.setattr(
        bnlearn_runner,
        "_check_bnlearn_status",
        lambda executor=None, timeout=30: FrameworkStatus(
            "bnlearn",
            False,
            install_hint="uv sync --extra bnlearn",
            reason="Missing bnlearn; uv sync --extra bnlearn",
            reason_code="missing_module",
        ),
    )

    record = execute_bnlearn_script(script, tmp_path / "exec")

    assert record["skipped"] is True
    assert record["success"] is False
    assert "uv sync --extra bnlearn" in record["reason"]
    assert record["return_code"] is None
    assert spy.calls == []


def test_execute_python_script_nonzero_exit_fails_explicitly(
    tmp_path: Path, monkeypatch: Any
) -> None:
    script = _write_render_script(tmp_path, "model_a", "a_bnlearn.py")
    spy = _EnvelopeSpy(
        [
            _envelope(
                success=False,
                return_code=2,
                stdout="",
                stderr="boom",
                duration_seconds=0.5,
            )
        ]
    )
    monkeypatch.setattr(bnlearn_runner, "run_subprocess_envelope", spy)
    monkeypatch.setattr(
        bnlearn_runner,
        "_check_bnlearn_status",
        lambda executor=None, timeout=30: FrameworkStatus("bnlearn", True),
    )

    record = execute_bnlearn_script(script, tmp_path / "exec")

    assert record["success"] is False
    assert record["skipped"] is False
    assert record["return_code"] == 2
    assert record["stderr"] == "boom"
    assert record["error_type"] == "RuntimeError"
    assert "failed (2)" in record["error"]


def test_execute_python_script_timeout_reports_error_type(
    tmp_path: Path, monkeypatch: Any
) -> None:
    script = _write_render_script(tmp_path, "model_a", "a_bnlearn.py")
    spy = _EnvelopeSpy(
        [
            _envelope(
                success=False,
                return_code=-1,
                stdout="partial",
                stderr="partial",
                duration_seconds=2.0,
                error_type="TimeoutExpired",
            )
        ]
    )
    monkeypatch.setattr(bnlearn_runner, "run_subprocess_envelope", spy)
    monkeypatch.setattr(
        bnlearn_runner,
        "_check_bnlearn_status",
        lambda executor=None, timeout=30: FrameworkStatus("bnlearn", True),
    )

    record = execute_bnlearn_script(script, tmp_path / "exec", timeout=5)

    assert record["success"] is False
    assert record["skipped"] is False
    assert record["error_type"] == "TimeoutExpired"
    assert "timed out after 5s" in record["error"]


def test_execute_r_script_runs_under_rscript(tmp_path: Path, monkeypatch: Any) -> None:
    script = _write_render_script(tmp_path, "model_a", "a.R")
    spy = _EnvelopeSpy()
    monkeypatch.setattr(bnlearn_runner, "run_subprocess_envelope", spy)
    monkeypatch.setattr(
        bnlearn_runner,
        "_check_r_bnlearn_status",
        lambda r="Rscript", timeout=30: FrameworkStatus("bnlearn", True),
    )
    # The pre-exec gate's file-type policy denies the R lane; the operator
    # override is the documented path for exercising the spawn itself.
    monkeypatch.setenv("GNN_ALLOW_UNSAFE_EXEC", "1")

    record = execute_bnlearn_script(script, tmp_path / "exec", rscript_executable="myR")

    assert record["success"] is True
    assert record["language"] == "r"
    assert spy.calls[0]["command"] == ["myR", str(script)]


def test_execute_unknown_language_skips_without_subprocess(tmp_path: Path) -> None:
    script = _write_render_script(tmp_path, "model_a", "a.jl")

    record = execute_bnlearn_script(script, tmp_path / "exec")

    assert record["skipped"] is True
    assert record["success"] is False
    assert "unknown" in record["language"] or record["language"] == "unknown"
    assert "Unsupported bnlearn script language" in record["reason"]


def test_run_bnlearn_scripts_mixed_lanes(tmp_path: Path, monkeypatch: Any) -> None:
    py_script = _write_render_script(tmp_path, "model_a", "a_bnlearn.py")
    r_script = _write_render_script(tmp_path, "model_b", "b.R")
    spy = _EnvelopeSpy()
    monkeypatch.setattr(bnlearn_runner, "run_subprocess_envelope", spy)
    monkeypatch.setattr(
        bnlearn_runner,
        "_check_bnlearn_status",
        lambda executor=None, timeout=30: FrameworkStatus(
            "bnlearn",
            False,
            install_hint="uv sync --extra bnlearn",
            reason="Missing bnlearn; uv sync --extra bnlearn",
            reason_code="missing_module",
        ),
    )
    monkeypatch.setattr(
        bnlearn_runner,
        "_check_r_bnlearn_status",
        lambda r="Rscript", timeout=30: FrameworkStatus("bnlearn", True),
    )
    monkeypatch.setenv("GNN_ALLOW_UNSAFE_EXEC", "1")

    records = run_bnlearn_scripts(tmp_path, tmp_path / "exec")

    assert records[0]["script"] == str(py_script)
    assert records[0]["skipped"] is True
    assert records[1]["script"] == str(r_script)
    assert records[1]["success"] is True
    assert spy.calls[0]["command"] == ["Rscript", str(r_script)]
    assert spy.calls[0]["kwargs"]["cwd"] == str(tmp_path / "exec" / "model_b")


# ── Step 12 wiring ─────────────────────────────────────────────────────────


def test_step12_execution_environment_sets_bnlearn_output_dir(tmp_path: Path) -> None:
    script = _write_render_script(tmp_path, "model_a", "a_bnlearn.py")
    context = ScriptExecutionContext(
        script_path=script,
        script_name=script.name,
        framework="bnlearn",
        model_name="model_a",
        executor=sys.executable,
    )

    env = execute_processor._build_execution_environment(context, tmp_path / "12")

    simulation_data = tmp_path / "12" / "model_a" / "bnlearn" / "simulation_data"
    assert env["BNLEARN_OUTPUT_DIR"] == str(simulation_data)
    assert simulation_data.is_dir()


def test_plan_disposition_marks_bnlearn_dependency_skip(monkeypatch: Any) -> None:
    script_info = {
        "framework": "bnlearn",
        "executor": sys.executable,
        "path": "/render/m/bnlearn/m_bnlearn.py",
        "name": "m_bnlearn.py",
    }
    monkeypatch.setattr(
        execute_planning, "is_framework_available", lambda *a, **k: False
    )
    assert execute_planning._disposition(script_info) == "skip_dependency"
    monkeypatch.setattr(
        execute_planning, "is_framework_available", lambda *a, **k: True
    )
    assert execute_planning._disposition(script_info) == "execute"


def test_step12_preflight_skips_bnlearn_scripts_without_module(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The documented skip story: bnlearn scripts skip at the Python
    pre-flight check when the module is absent."""
    script = _write_render_script(tmp_path, "model_a", "a_bnlearn.py")
    script_info = {
        "path": script,
        "name": script.name,
        "framework": "bnlearn",
        "executor": sys.executable,
        "relative_path": script,
        "size_bytes": script.stat().st_size,
    }
    monkeypatch.setattr(
        execute_processor,
        "_check_framework_by_name",
        lambda *a, **k: FrameworkStatus(
            "bnlearn",
            False,
            "bnlearn",
            reason_code="missing_module",
            reason="Dependency not installed: bnlearn",
        ),
    )

    result = execute_processor.execute_single_script(
        script_info, tmp_path / "12", False, execute_processor.logger, 60
    )

    assert result["skipped"] is True
    assert result["success"] is False
    assert result["status"] == "skipped"
    assert result["error_type"] == "DependencyNotInstalled"
    assert "bnlearn" in result["error"]


def test_direct_relative_script_survives_output_cwd(tmp_path, monkeypatch):
    import os

    from gnn.execute.bnlearn import bnlearn_runner as runner

    source = tmp_path / "input" / "relative.py"
    source.parent.mkdir()
    source.write_text(
        "import os; from pathlib import Path; Path(os.environ['BNLEARN_OUTPUT_DIR'], 'written.txt').write_text('actual child')\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GNN_ALLOW_UNSAFE_EXEC", "1")
    monkeypatch.setattr(
        runner, "_check_bnlearn_status", lambda *_: FrameworkStatus("bnlearn", True)
    )
    result = runner.execute_bnlearn_script("input/relative.py", "output", timeout=10)
    assert result["success"] and result["status"] == "success"
    assert result["cleanup_verified"] and result["streams_drained"]
    assert (tmp_path / "output/written.txt").read_text() == "actual child"


def test_direct_pre_cancel_precedes_missing_runtime_probe(tmp_path, monkeypatch):
    from gnn.execute.bnlearn import bnlearn_runner as runner
    from gnn.execute.subprocess_envelope import CancelToken

    source = tmp_path / "unused.py"
    source.write_text("raise RuntimeError('must not start')\n")
    token = CancelToken()
    token.cancel("stop")

    def forbidden(*_):
        raise AssertionError("cancelled work cannot probe a runtime")

    monkeypatch.setattr(runner, "_check_bnlearn_status", forbidden)
    result = runner.execute_bnlearn_script(source, tmp_path / "out", cancel_token=token)
    assert result["cancelled"] and result["status"] == "cancelled"
    assert result["containment"] == "not_started"


def test_direct_timeout_preserves_containment_receipt(tmp_path, monkeypatch):
    from gnn.execute.bnlearn import bnlearn_runner as runner

    source = tmp_path / "slow.py"
    source.write_text(
        "import time; print('partial evidence', flush=True); time.sleep(10)\n"
    )
    monkeypatch.setenv("GNN_ALLOW_UNSAFE_EXEC", "1")
    monkeypatch.setattr(
        runner, "_check_bnlearn_status", lambda *_: FrameworkStatus("bnlearn", True)
    )
    result = runner.execute_bnlearn_script(source, tmp_path / "out", timeout=0.2)
    assert not result["success"] and result["status"] == "timed_out"
    assert result["cleanup_verified"] and result["streams_drained"]
    assert "partial evidence" in result["stdout"]


@pytest.mark.parametrize(
    ("diagnosis", "expected_status", "expected_error"),
    [
        (
            FrameworkStatus(
                "bnlearn",
                False,
                reason_code="probe_timeout",
                execution_error_type="TimeoutExpired",
                cleanup_verified=True,
                streams_drained=True,
            ),
            "timed_out",
            "TimeoutExpired",
        ),
        (
            FrameworkStatus(
                "bnlearn",
                False,
                reason_code="probe_failed",
                execution_error_type="RuntimeError",
                cleanup_verified=True,
                streams_drained=True,
            ),
            "failed",
            "RuntimeError",
        ),
        (
            FrameworkStatus(
                "bnlearn",
                False,
                reason_code="probe_failed",
                execution_error_type="TimeoutExpired",
                cleanup_verified=False,
                streams_drained=False,
            ),
            "failed",
            "ProcessCleanupFailure",
        ),
    ],
)
def test_direct_bnlearn_retains_failed_probe_cause(
    tmp_path, monkeypatch, diagnosis, expected_status, expected_error
):
    script = _write_render_script(tmp_path)
    monkeypatch.setattr(bnlearn_runner, "_check_bnlearn_status", lambda *_: diagnosis)

    def forbid_dispatch(*args, **kwargs):
        raise AssertionError("failed readiness must prevent script dispatch")

    monkeypatch.setattr(bnlearn_runner, "run_subprocess_envelope", forbid_dispatch)
    record = execute_bnlearn_script(script, tmp_path / "out")
    assert record["status"] == expected_status
    assert record["error_type"] == expected_error
    assert record["skipped"] is False
    assert record["reason_code"] == diagnosis.reason_code


@pytest.mark.parametrize(
    ("envelope", "reason_code"),
    [
        (
            _envelope(
                success=False,
                return_code=42,
                cleanup_verified=True,
                streams_drained=True,
            ),
            "missing_module",
        ),
        (
            _envelope(
                success=False,
                return_code=1,
                error_type="RuntimeError",
                cleanup_verified=True,
                streams_drained=True,
            ),
            "probe_failed",
        ),
        (
            _envelope(
                success=False,
                return_code=-1,
                error_type="TimeoutExpired",
                cleanup_verified=True,
                streams_drained=True,
            ),
            "probe_timeout",
        ),
    ],
)
def test_r_readiness_distinguishes_absence_failure_and_timeout(
    monkeypatch, envelope, reason_code
):
    monkeypatch.setattr(bnlearn_runner.shutil, "which", lambda _: "/usr/bin/Rscript")
    monkeypatch.setattr(
        bnlearn_runner, "run_subprocess_envelope", _EnvelopeSpy([envelope])
    )
    diagnosis = bnlearn_runner._check_r_bnlearn_status()
    assert diagnosis.available is False
    assert diagnosis.reason_code == reason_code
