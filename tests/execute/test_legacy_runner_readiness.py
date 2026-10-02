"""Legacy runner readiness stays supervised, cold and within one local budget."""

from __future__ import annotations

import importlib
import json
import os
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import pytest

from gnn.execute.executor import GNNExecutor
from gnn.utils.runtime_safety import framework_availability

_RUNNERS = (
    ("jax", "gnn.execute.jax.jax_runner", "execute_jax_script"),
    ("numpyro", "gnn.execute.numpyro.numpyro_runner", "execute_numpyro_script"),
    ("pytorch", "gnn.execute.pytorch.pytorch_runner", "execute_pytorch_script"),
    ("discopy", "gnn.execute.discopy.discopy_executor", "execute_discopy_script"),
)
_HEAVY = ("jax", "numpyro", "torch", "discopy", "matplotlib", "networkx")


def test_ready_registry_and_public_helpers_do_not_import_backends_in_parent(
    tmp_path: Path,
) -> None:
    """Actual ready child probes must not make the registry loader import packages."""
    shims = tmp_path / "shims"
    shims.mkdir()
    for name in ("jax", "numpyro", "torch", "discopy"):
        (shims / f"{name}.py").write_text("VERSION = 'test-only'\n")
    project = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(shims), str(project / "src")))
    code = f"""
import importlib, json, sys
from gnn.execute.executor import _runner_state
for framework, module, function in {_RUNNERS!r}:
    assert _runner_state(framework).available
    runner = importlib.import_module(module)
    if framework != 'discopy':
        assert getattr(runner, 'is_' + framework + '_available')()
print(json.dumps([name for name in {_HEAVY!r} if name in sys.modules]))
"""
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=45,
    )
    assert json.loads(result.stdout.splitlines()[-1]) == []


@pytest.mark.parametrize("framework,module_name,function_name", _RUNNERS)
def test_unavailable_readiness_never_loads_runner(
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
    module_name: str,
    function_name: str,
) -> None:
    """Readiness runs before any loader, even if loading would hang."""
    from gnn.execute import executor

    del module_name, function_name
    diagnosis = framework_availability.FrameworkStatus(
        framework,
        False,
        reason_code="probe_timeout",
        execution_error_type="TimeoutExpired",
    )
    monkeypatch.setattr(
        framework_availability, "check_framework", lambda *a, **kw: diagnosis
    )

    def forbidden() -> Any:
        raise AssertionError("unavailable optional runner must never load")

    monkeypatch.setitem(executor._RUNNER_LOADERS, framework, forbidden)
    state = executor._runner_state(framework)
    assert state.diagnosis is diagnosis and not state.available and state.runner is None


@pytest.mark.parametrize("framework,module_name,function_name", _RUNNERS)
@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan")])
def test_direct_runner_invalid_timeout_never_probes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
    module_name: str,
    function_name: str,
    timeout: Any,
) -> None:
    del framework
    runner = importlib.import_module(module_name)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("invalid timeout must not launch a readiness probe")

    monkeypatch.setattr(framework_availability, "check_framework", forbidden)
    script = tmp_path / "model.py"
    script.write_text("print('not executed')")
    assert not getattr(runner, function_name)(script, timeout=timeout)


@pytest.mark.needs_posix
@pytest.mark.parametrize("framework,module_name,function_name", _RUNNERS)
def test_actual_slow_import_is_reaped_within_direct_script_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
    module_name: str,
    function_name: str,
) -> None:
    runner = importlib.import_module(module_name)
    module = {"pytorch": "torch"}.get(framework, framework)
    pid_file = tmp_path / "probe_pid"
    (tmp_path / f"{module}.py").write_text(
        "import os, time\nfrom pathlib import Path\n"
        f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    script = tmp_path / "model.py"
    script.write_text("print('must not execute')")
    started = time.monotonic()
    assert not getattr(runner, function_name)(script, timeout=1.5)
    assert time.monotonic() - started < 2.5
    assert pid_file.exists(), "a genuine optional import child must have started"
    assert not psutil.pid_exists(int(pid_file.read_text()))


@pytest.mark.parametrize("framework,module_name,function_name", _RUNNERS)
def test_probe_elapsed_time_is_not_reset_before_script_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
    module_name: str,
    function_name: str,
) -> None:
    runner = importlib.import_module(module_name)
    script = tmp_path / "model.py"
    script.write_text("print('safe')")
    captured: dict[str, Any] = {}

    def ready(name: str, **kwargs: Any) -> framework_availability.FrameworkStatus:
        assert name == framework
        captured["probe_deadline"] = kwargs["deadline_monotonic"]
        time.sleep(0.03)
        return framework_availability.FrameworkStatus(name, True)

    def completed(path: Path, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "success": True,
            "return_code": 0,
            "stdout": "safe",
            "stderr": "",
            "duration_seconds": 0.01,
        }

    monkeypatch.setattr(framework_availability, "check_framework", ready)
    monkeypatch.setattr("gnn.execute.executor.execute_script_safely", completed)
    assert getattr(runner, function_name)(
        script, output_dir=tmp_path / "out", timeout=2
    )
    assert captured["deadline_monotonic"] == captured["probe_deadline"]
    assert 0 < captured["deadline_monotonic"] - time.monotonic() < 1.99


@pytest.mark.parametrize(
    "cause,status,error_type",
    [
        ("missing_module", "skipped", "DependencyUnavailable"),
        ("missing_toolchain", "skipped", "DependencyUnavailable"),
        ("probe_failed", "failed", "FrameworkProbeFailure"),
        ("probe_timeout", "timed_out", "TimeoutExpired"),
        ("cleanup_failed", "failed", "ProcessCleanupFailure"),
    ],
)
def test_direct_executor_stan_preserves_readiness_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cause: str,
    status: str,
    error_type: str,
) -> None:
    from gnn.execute.stan import stan_runner

    diagnosis = framework_availability.FrameworkStatus(
        "stan",
        False,
        reason_code="probe_timeout" if cause == "cleanup_failed" else cause,
        execution_error_type="TimeoutExpired"
        if cause in {"probe_timeout", "cleanup_failed"}
        else None,
        cleanup_verified=False if cause == "cleanup_failed" else True,
        streams_drained=False if cause == "cleanup_failed" else True,
    )
    monkeypatch.setattr(stan_runner, "_check_stan_status", lambda *a, **kw: diagnosis)
    script = tmp_path / "m_stan.py"
    script.write_text("print('not executed')")
    result = GNNExecutor()._execute_stan_script(str(script))
    assert not result["success"] and result["status"] == status
    assert result["error_type"] == error_type
    if cause == "cleanup_failed":
        assert result["execution_error_type"] == "TimeoutExpired"


@pytest.mark.needs_posix
def test_actual_stan_probe_cannot_spend_thirty_seconds_before_short_script_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_file = tmp_path / "probe_pid"
    (tmp_path / "cmdstanpy.py").write_text(
        "import os, time\nfrom pathlib import Path\n"
        f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    script = tmp_path / "m_stan.py"
    script.write_text("print('not executed')")
    started = time.monotonic()
    result = GNNExecutor()._execute_stan_script(str(script), timeout=1.5)
    assert time.monotonic() - started < 2.5
    assert not result["success"] and result["status"] == "timed_out"
    assert result["error_type"] == "TimeoutExpired"
    assert result["cleanup_verified"] is True and result["streams_drained"] is True
    assert pid_file.exists() and not psutil.pid_exists(int(pid_file.read_text()))


def test_expired_absolute_probe_deadline_precedes_cache_or_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "exhausted local deadline must not launch or use cached readiness"
        )

    monkeypatch.setattr(framework_availability, "_run_probe_envelope", forbidden)
    status = framework_availability.check_framework(
        "jax", deadline_monotonic=time.monotonic() - 1
    )
    assert not status.available and status.reason_code == "probe_timeout"


def test_safe_executor_expired_deadline_rejects_even_enabled_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.execute.executor import execute_script_safely
    from gnn.execute.result_cache import ExecutionResultCache

    script = tmp_path / "model.py"
    script.write_text("print('safe')")
    cache = ExecutionResultCache(enabled=True)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an exhausted local deadline must precede cache retrieval")

    monkeypatch.setattr(cache, "lookup", forbidden)
    result = execute_script_safely(
        script, cache=cache, deadline_monotonic=time.monotonic() - 1
    )
    assert not result["success"] and result["status"] == "timed_out"
    assert result["containment"] == "not_started"
    assert result["error_type"] == "TimeoutExpired"


def test_jax_batch_passes_requested_output_and_leaves_render_tree_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.execute.jax import jax_runner

    render = tmp_path / "render"
    (render / "jax").mkdir(parents=True)
    script = render / "jax" / "model_jax.py"
    script.write_text("print('safe')")
    selected = []

    def execute(
        path: Path, verbose: bool, device: Any, output_dir: Path, **kwargs: Any
    ) -> bool:
        selected.append((path, output_dir, kwargs["timeout"]))
        return True

    monkeypatch.setattr(jax_runner, "execute_jax_script", execute)
    out = tmp_path / "results"
    assert jax_runner.run_jax_scripts(render, out, timeout=7)
    assert selected == [(script, out / "model_jax", 7)]
    assert list((render / "jax").iterdir()) == [script]


def test_discopy_analysis_preserves_successes_without_accepting_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.execute.discopy import discopy_executor

    (tmp_path / "discopy").mkdir()
    monkeypatch.setattr(
        discopy_executor.DisCoPyExecutor,
        "execute_directory",
        lambda *args: {
            "analysis_summary": {"total_files_processed": 2},
            "successes": 1,
            "failures": 1,
        },
    )
    assert not discopy_executor.run_discopy_analysis(tmp_path, tmp_path / "out")


def test_stan_expired_explicit_deadline_precedes_gate_and_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.execute.stan import stan_runner

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("expired direct deadline must precede gate and probe")

    monkeypatch.setattr(stan_runner, "check_script_allowed", forbidden)
    monkeypatch.setattr(stan_runner, "_check_stan_status", forbidden)
    result = stan_runner.execute_stan_script(
        tmp_path / "missing.py",
        tmp_path / "out",
        deadline_monotonic=time.monotonic() - 1,
    )
    assert not result["success"] and result["status"] == "timed_out"
    assert result["containment"] == "not_started"
    assert not (tmp_path / "out").exists()


def test_stan_direct_none_timeout_uses_declared_default_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.execute.stan import stan_runner

    captured = []

    def diagnose(
        python_executable: Any, timeout: float, **kwargs: Any
    ) -> framework_availability.FrameworkStatus:
        captured.append((timeout, kwargs["deadline_monotonic"]))
        return framework_availability.FrameworkStatus(
            "stan", False, reason_code="missing_module"
        )

    monkeypatch.setenv("GNN_EXECUTE_DEFAULT_TIMEOUT", "7")
    monkeypatch.setattr(stan_runner, "_check_stan_status", diagnose)
    script = tmp_path / "model_stan.py"
    script.write_text("print('safe')")
    result = stan_runner.execute_stan_script(script, tmp_path / "out", timeout=None)
    assert result["skipped"] and not result["success"]
    assert captured[0][0] == 7
    assert 0 < captured[0][1] - time.monotonic() <= 7
