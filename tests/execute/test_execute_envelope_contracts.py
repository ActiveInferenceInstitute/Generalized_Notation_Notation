#!/usr/bin/env python3
"""Subprocess-envelope contract pins for execution surfaces without coverage.

Pins CURRENT behavior of the already-migrated shared-envelope callers that
no other test file exercises with a spy:

- ``GNNExecutor`` dispatch defaults (``executor.py``): per-type command
  vectors and timeout fallbacks, plus envelope passthrough.
- ``activeinference_runner``: script execution call shape
  (JULIA_PROJECT + GKSwstype=100 headless env, timeout 600, cwd) and result mapping, plus the
  package-validation probe (timeout 30).
- ``stan_runner.execute_stan_script``: command/env/cwd/timeout and the
  result-dict mapping.
- ``rxinfer_runner`` TOML branch: the ``rxinfer_runner.jl`` command shape.

No Julia, Stan toolchain, or real subprocess is needed — the envelope is
spied and returns canned results (same pattern as ``test_execute_bnlearn``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from gnn.execute import executor as executor_module
from gnn.execute.activeinference_jl import activeinference_runner
from gnn.execute.executor import (
    GNNExecutor,
    clear_execution_cache,
    execute_script_safely,
)
from gnn.execute.result_cache import ExecutionResultCache
from gnn.execute.rxinfer import rxinfer_runner
from gnn.execute.rxinfer.rxinfer_runner import execute_rxinfer_script
from gnn.execute.stan.stan_runner import execute_stan_script
from gnn.execute.subprocess_envelope import CancelToken
from gnn.utils.runtime_safety.framework_availability import FrameworkStatus


def _envelope(**overrides: Any) -> Dict[str, Any]:
    """Full canned envelope, including fields only failures populate."""
    envelope: Dict[str, Any] = {
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


def _spy(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    results: Optional[List[Dict[str, Any]]] = None,
) -> _EnvelopeSpy:
    spy = _EnvelopeSpy(results)
    monkeypatch.setattr(target, spy)
    return spy


# ── GNNExecutor dispatch defaults ──────────────────────────────────────────


def test_pymdp_dispatch_runs_python_with_default_timeout_600(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "m.py"
    script.write_text("print('ok')\n")
    canned = _envelope(stdout="sim done")
    spy = _spy(monkeypatch, "gnn.execute.executor.run_subprocess_envelope", [canned])

    result = GNNExecutor()._execute_pymdp_script(str(script))

    assert spy.calls[0]["command"] == [sys.executable, str(script)]
    assert spy.calls[0]["kwargs"]["timeout"] == 600
    # The dispatch methods return the envelope untouched — every field
    # (including sandbox bookkeeping and error slots) passes through.
    assert result == canned


def test_rxinfer_dispatch_routes_config_through_committed_runner_timeout_300(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TOML configs execute via execute_rxinfer_script, not a raw julia spawn."""
    config = tmp_path / "m_config.toml"
    config.write_text("[model]\n")
    canned = _envelope(stdout="sim done")
    spy = _spy(
        monkeypatch,
        "gnn.execute.rxinfer.rxinfer_runner.run_subprocess_envelope",
        [canned],
    )

    result = GNNExecutor()._execute_rxinfer_config(str(config))

    runner_project = Path(rxinfer_runner.__file__).parent.resolve()
    assert spy.calls[0]["command"] == [
        "julia",
        "--startup-file=no",
        f"--project={runner_project}",
        str(runner_project / "rxinfer_runner.jl"),
        str(config),
    ]
    assert spy.calls[0]["kwargs"]["timeout"] == 300
    # The dispatch synthesizes its envelope from the runner verdict plus the
    # evidence sidecars the runner persists beside the script.
    assert result == {
        "success": True,
        "return_code": 0,
        "stdout": "sim done",
        "stderr": "",
        "elapsed_seconds": 0.01,
    }
    assert (tmp_path / "m_config_stdout.txt").read_text() == "sim done"
    assert (tmp_path / "m_config_stderr.txt").read_text() == ""


@pytest.mark.parametrize("execution_type", ["discopy", "jax"])
def test_discopy_and_jax_dispatch_run_sys_executable_with_default_timeout_300(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execution_type: str
) -> None:
    target = tmp_path / f"m_{execution_type}.py"
    target.write_text("print('ok')\n")
    canned = _envelope()
    spy = _spy(monkeypatch, "gnn.execute.executor.run_subprocess_envelope", [canned])

    executor = GNNExecutor()
    if execution_type == "discopy":
        result = executor._execute_discopy_diagram(str(target))
    else:
        result = executor._execute_jax_script(str(target))

    assert spy.calls[0]["command"] == [sys.executable, str(target)]
    assert spy.calls[0]["kwargs"]["timeout"] == 300
    assert result == canned


# ── activeinference_runner ─────────────────────────────────────────────────


def _activeinference_script(tmp_path: Path) -> tuple[Path, Path]:
    """Script under an ``activeinference_jl`` dir; returns (script, project_dir)."""
    project_dir = tmp_path / "activeinference_jl"
    project_dir.mkdir()
    script = project_dir / "m_activeinference.jl"
    script.write_text('println("ok")\n')
    return script, project_dir


def test_activeinference_script_success_call_shape_and_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, project_dir = _activeinference_script(tmp_path)
    spy = _spy(
        monkeypatch,
        "gnn.execute.activeinference_jl.activeinference_runner.run_subprocess_envelope",
        [_envelope(stdout="ran")],
    )

    ok = activeinference_runner.execute_activeinference_script(
        script, setup_environment=False
    )

    assert ok is True
    assert spy.calls[0]["command"] == [
        "julia",
        f"--project={project_dir}",
        str(script.resolve()),
    ]
    assert spy.calls[0]["kwargs"]["timeout"] == 600
    env = spy.calls[0]["kwargs"]["env"]
    assert env["JULIA_PROJECT"] == str(project_dir)
    assert env["GKSwstype"] == "100"
    assert spy.calls[0]["kwargs"]["cwd"] == str(project_dir)


def test_activeinference_script_appends_output_dir_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, project_dir = _activeinference_script(tmp_path)
    out_dir = tmp_path / "results"
    spy = _spy(
        monkeypatch,
        "gnn.execute.activeinference_jl.activeinference_runner.run_subprocess_envelope",
    )

    ok = activeinference_runner.execute_activeinference_script(
        script, setup_environment=False, output_dir=out_dir
    )

    assert ok is True
    assert spy.calls[0]["command"][:3] == [
        "julia",
        f"--project={project_dir}",
        str(script.resolve()),
    ]
    assert spy.calls[0]["command"][-2:] == ["--output-dir", str(out_dir)]
    assert out_dir.is_dir()


@pytest.mark.parametrize(
    ("canned", "expected"),
    [
        (
            _envelope(success=False, return_code=1, stderr="ERROR: package missing"),
            False,
        ),
        (_envelope(success=False, return_code=-1, error="spawn failed"), False),
        (
            _envelope(
                success=False,
                return_code=-1,
                error="killed",
                error_type="TimeoutExpired",
            ),
            False,
        ),
    ],
)
def test_activeinference_script_failure_mappings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canned: Dict[str, Any],
    expected: bool,
) -> None:
    script, _ = _activeinference_script(tmp_path)
    spy = _spy(
        monkeypatch,
        "gnn.execute.activeinference_jl.activeinference_runner.run_subprocess_envelope",
        [canned],
    )

    ok = activeinference_runner.execute_activeinference_script(
        script, setup_environment=False
    )

    assert ok is expected
    assert len(spy.calls) == 1


def test_activeinference_package_probe_uses_timeout_30(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "activeinference_jl"
    project_dir.mkdir()
    spy = _spy(
        monkeypatch,
        "gnn.execute.activeinference_jl.activeinference_runner.run_subprocess_envelope",
    )

    ok = activeinference_runner._validate_package(project_dir, "ActiveInference")

    assert ok is True
    assert spy.calls[0]["command"] == [
        "julia",
        f"--project={project_dir}",
        "-e",
        'using ActiveInference; println("✅ ActiveInference loaded")',
    ]
    assert spy.calls[0]["kwargs"]["timeout"] == 30
    assert spy.calls[0]["kwargs"]["cwd"] == project_dir  # passed as Path
    assert spy.calls[0]["kwargs"]["env"]["GKSwstype"] == "100"


def test_activeinference_package_probe_failure_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "activeinference_jl"
    project_dir.mkdir()
    _spy(
        monkeypatch,
        "gnn.execute.activeinference_jl.activeinference_runner.run_subprocess_envelope",
        [_envelope(success=False, return_code=1)],
    )

    assert activeinference_runner._validate_package(project_dir, "Missing") is False


# ── stan_runner.execute_stan_script ────────────────────────────────────────


def test_stan_script_default_call_shape_and_result_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "gnn.execute.stan.stan_runner._check_stan_status",
        lambda *args, **kwargs: FrameworkStatus("stan", available=True),
    )
    script = tmp_path / "m_stan.py"
    script.write_text("print('ok')\n")
    out_dir = tmp_path / "stan_out"
    spy = _spy(
        monkeypatch,
        "gnn.execute.stan.stan_runner.run_subprocess_envelope",
        [_envelope(stdout="sampled")],
    )

    result = execute_stan_script(script, out_dir)

    assert out_dir.is_dir()  # created up front so the child can write into it
    assert spy.calls[0]["command"] == [sys.executable, str(script)]
    assert 0 < spy.calls[0]["kwargs"]["timeout"] <= 1800
    assert spy.calls[0]["kwargs"]["deadline_monotonic"] is not None
    assert spy.calls[0]["kwargs"]["env"] == {"STAN_OUTPUT_DIR": str(out_dir)}
    assert spy.calls[0]["kwargs"]["cwd"] == str(out_dir)
    assert result == {
        "script": str(script),
        "framework": "stan",
        "return_code": 0,
        "success": True,
        "stdout": "sampled",
        "stderr": "",
        "execution_time_seconds": round(0.01, 3),
        "results_file": str(out_dir / "simulation_results.json"),
    }


def test_stan_script_failure_maps_envelope_and_honors_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "gnn.execute.stan.stan_runner._check_stan_status",
        lambda *args, **kwargs: FrameworkStatus("stan", available=True),
    )
    script = tmp_path / "m_stan.py"
    script.write_text("print('ok')\n")
    out_dir = tmp_path / "stan_out"
    spy = _spy(
        monkeypatch,
        "gnn.execute.stan.stan_runner.run_subprocess_envelope",
        [_envelope(success=False, return_code=2, stderr="compile failed")],
    )

    result = execute_stan_script(
        script, out_dir, timeout=60, python_executable="custompython"
    )

    assert spy.calls[0]["command"] == ["custompython", str(script)]
    assert 0 < spy.calls[0]["kwargs"]["timeout"] <= 60
    assert spy.calls[0]["kwargs"]["deadline_monotonic"] is not None
    assert result["success"] is False
    assert result["return_code"] == 2
    assert result["stderr"] == "compile failed"


# ── rxinfer_runner TOML branch ─────────────────────────────────────────────


def test_rxinfer_toml_config_runs_committed_runner_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "model_a_config.toml"
    config.write_text("[model]\n")
    runner_script = Path(rxinfer_runner.__file__).parent / "rxinfer_runner.jl"
    spy = _spy(
        monkeypatch,
        "gnn.execute.rxinfer.rxinfer_runner.run_subprocess_envelope",
        [_envelope()],
    )

    ok = execute_rxinfer_script(config, output_dir=tmp_path / "logs")

    assert ok is True
    assert spy.calls[0]["command"] == [
        "julia",
        "--startup-file=no",
        f"--project={Path(rxinfer_runner.__file__).parent.resolve()}",
        str(runner_script),
        str(config),
    ]
    assert spy.calls[0]["kwargs"]["timeout"] == 300


def test_rxinfer_toml_config_missing_runner_script_refuses_without_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without rxinfer_runner.jl beside the module, TOML configs refuse cleanly."""
    config = tmp_path / "model_a_config.toml"
    config.write_text("[model]\n")
    spy = _spy(
        monkeypatch, "gnn.execute.rxinfer.rxinfer_runner.run_subprocess_envelope"
    )

    calls: Dict[str, Any] = {}
    original_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if self.name == "rxinfer_runner.jl":
            calls["queried"] = True
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)

    assert execute_rxinfer_script(config, output_dir=tmp_path / "logs") is False
    assert spy.calls == []
    assert calls.get("queried") is True


# ── execute_script_safely + execution result cache ─────────────────────────


def _isolate_shared_cache(
    monkeypatch: pytest.MonkeyPatch, cache_dir: Path, *, enabled: Optional[bool] = None
) -> None:
    """Point the executor's shared cache at a tmp_path dir (never the repo)."""
    monkeypatch.setattr(
        executor_module,
        "_EXECUTION_RESULT_CACHE",
        ExecutionResultCache(cache_dir=cache_dir, enabled=enabled),
    )


def test_execute_script_safely_forwards_args_and_cancel_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "m.py"
    script.write_text("print('ok')\n")
    _isolate_shared_cache(monkeypatch, tmp_path / "cache", enabled=False)
    spy = _spy(
        monkeypatch, "gnn.execute.executor.run_subprocess_envelope", [_envelope()]
    )
    token = CancelToken()

    result = execute_script_safely(script, args=["--flag"], cancel_token=token)

    assert spy.calls[0]["command"] == [sys.executable, str(script), "--flag"]
    assert spy.calls[0]["kwargs"]["cancel_token"] is token
    assert result["success"] is True
    assert result["script_path"] == str(script)


def test_execute_script_safely_cache_default_off_spawns_every_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GNN_EXEC_CACHE", raising=False)
    script = tmp_path / "m.py"
    script.write_text("print('ok')\n")
    cache_dir = tmp_path / "cache"
    _isolate_shared_cache(monkeypatch, cache_dir)
    spy = _spy(monkeypatch, "gnn.execute.executor.run_subprocess_envelope")

    first = execute_script_safely(script)
    second = execute_script_safely(script)

    assert len(spy.calls) == 2
    assert "cache_hit" not in first
    assert "cache_hit" not in second
    assert not cache_dir.exists()  # default-off never even creates the dir


def test_execute_script_safely_env_enabled_cache_short_circuits_second_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GNN_EXEC_CACHE", "1")
    script = tmp_path / "m.py"
    script.write_text("print('ok')\n")
    cache_dir = tmp_path / "cache"
    _isolate_shared_cache(monkeypatch, cache_dir)
    spy = _spy(
        monkeypatch,
        "gnn.execute.executor.run_subprocess_envelope",
        [_envelope(stdout="first-run")],
    )

    first = execute_script_safely(script)
    second = execute_script_safely(script)

    assert len(spy.calls) == 1  # second call short-circuits before the spawn
    assert "cache_hit" not in first
    assert first["stdout"] == "first-run"
    assert second["cache_hit"] is True
    assert second["stdout"] == first["stdout"]
    assert second["return_code"] == first["return_code"]
    assert second["script_path"] == str(script)


def test_execute_script_safely_cache_key_tracks_script_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GNN_EXEC_CACHE", "1")
    script = tmp_path / "m.py"
    script.write_text("print('v1')\n")
    _isolate_shared_cache(monkeypatch, tmp_path / "cache")
    spy = _spy(monkeypatch, "gnn.execute.executor.run_subprocess_envelope")

    execute_script_safely(script)
    script.write_text("print('v2')\n")
    second = execute_script_safely(script)

    assert len(spy.calls) == 2  # content edit → new key → spawns again
    assert "cache_hit" not in second


def test_execute_script_safely_failure_envelope_is_never_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GNN_EXEC_CACHE", "1")
    script = tmp_path / "m.py"
    script.write_text("print('ok')\n")
    cache_dir = tmp_path / "cache"
    _isolate_shared_cache(monkeypatch, cache_dir)
    canned_failure = _envelope(success=False, return_code=3, error="boom")
    spy = _spy(
        monkeypatch,
        "gnn.execute.executor.run_subprocess_envelope",
        [canned_failure, dict(canned_failure)],
    )

    first = execute_script_safely(script)
    second = execute_script_safely(script)

    assert len(spy.calls) == 2  # failure envelopes are never stored
    assert first["success"] is False
    assert second["success"] is False
    assert "cache_hit" not in second
    assert not cache_dir.exists()  # nothing stored → lazy mkdir never ran


def test_clear_execution_cache_invalidates_shared_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GNN_EXEC_CACHE", raising=False)
    script = tmp_path / "m.py"
    script.write_text("print('ok')\n")
    _isolate_shared_cache(monkeypatch, tmp_path / "cache", enabled=True)
    spy = _spy(monkeypatch, "gnn.execute.executor.run_subprocess_envelope")

    execute_script_safely(script)  # populates the shared cache (1 entry)
    removed = clear_execution_cache()
    third = execute_script_safely(script)

    assert removed == 1
    assert len(spy.calls) == 2  # spawn again after invalidation
    assert "cache_hit" not in third


@pytest.mark.parametrize(("config_timeout", "expected"), [(77, 77), (None, 600)])
def test_run_simulation_passes_config_timeout_to_pymdp_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_timeout: Optional[int],
    expected: int,
) -> None:
    script = tmp_path / "m.py"
    script.write_text("print('ok')\n")
    spy = _spy(monkeypatch, "gnn.execute.executor.run_subprocess_envelope")
    executor = GNNExecutor(
        cache=ExecutionResultCache(cache_dir=tmp_path / "cache", enabled=False)
    )
    config: Dict[str, Any] = {"model_path": str(script), "execution_type": "pymdp"}
    if config_timeout is not None:
        config["timeout"] = config_timeout

    result = executor.run_simulation(config)

    assert spy.calls[0]["kwargs"]["timeout"] == expected
    assert result["success"] is True


def test_pymdp_dispatch_cache_hit_avoids_second_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "m.py"
    script.write_text("print('ok')\n")
    executor = GNNExecutor(
        cache=ExecutionResultCache(cache_dir=tmp_path / "cache", enabled=True)
    )
    spy = _spy(monkeypatch, "gnn.execute.executor.run_subprocess_envelope")

    first = executor._execute_pymdp_script(str(script))
    second = executor._execute_pymdp_script(str(script))

    assert len(spy.calls) == 1
    assert first["success"] is True
    assert "cache_hit" not in first
    assert second["cache_hit"] is True
    assert second["stdout"] == first["stdout"]


def test_pymdp_non_py_source_model_path_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.md"
    model.write_text("# source model\n")
    executor = GNNExecutor(
        cache=ExecutionResultCache(cache_dir=tmp_path / "cache", enabled=True)
    )
    spy = _spy(monkeypatch, "gnn.execute.executor.run_subprocess_envelope")

    first = executor._execute_pymdp_script(str(model))
    second = executor._execute_pymdp_script(str(model))

    assert spy.calls == []  # canned path never spawns, never caches
    assert not first["success"] and first["skipped"]
    assert first["error_type"] == "RenderedScriptRequired"
    assert not second["success"] and second["skipped"]
    assert "cache_hit" not in first
    assert "cache_hit" not in second
    assert not (tmp_path / "cache").exists()
