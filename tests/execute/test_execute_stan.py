"""Stan executor: discovery, dependency gating, and a real CmdStan run."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from gnn.execute.stan import execute_stan_script, find_stan_scripts
from gnn.execute.subprocess_envelope import CancelToken
from gnn.render.stan.stan_renderer import render_gnn_to_stan

DISCRETE_SPEC = {
    "name": "Tiny HMM",
    "model_name": "Tiny HMM",
    "initialparameterization": {
        "A": [[0.85, 0.15], [0.15, 0.85]],
        "B": [[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]],
        "C": [0.0, 1.0],
        "D": [0.5, 0.5],
    },
    "model_parameters": {
        "num_hidden_states": 2,
        "num_obs": 2,
        "num_actions": 2,
        "num_timesteps": 6,
        "random_seed": 3,
    },
}


def test_find_stan_scripts_only_matches_drivers_in_stan_dirs(tmp_path: Path) -> None:
    ok, _, arts = render_gnn_to_stan(
        DISCRETE_SPEC, tmp_path / "m" / "stan" / "m_stan.py"
    )
    assert ok
    (tmp_path / "m" / "jax").mkdir()
    (tmp_path / "m" / "jax" / "other_stan.py").write_text("print()")
    found = find_stan_scripts(tmp_path)
    assert found == [Path(arts[0])]


def test_render_summary_contract_treats_stan_program_as_companion(
    tmp_path: Path,
) -> None:
    """The .stan program beside the driver is not a 'missing executable script'."""
    import json
    import logging

    from gnn.execute.processor import _load_render_summary_contract

    render_dir = tmp_path / "11"
    ok, _, arts = render_gnn_to_stan(
        DISCRETE_SPEC, render_dir / "tiny" / "stan" / "tiny_stan.py"
    )
    assert ok
    summary = {
        "file_results": {
            "input/tiny.md": {
                "framework_results": {
                    "stan": {"success": True, "output_files": arts},
                    "pymdp": {
                        "success": False,
                        "unsupported": True,
                        "message": "continuous-state model",
                    },
                }
            }
        }
    }
    (render_dir / "render_processing_summary.json").write_text(json.dumps(summary))
    allowed, failures, _unsup = _load_render_summary_contract(
        render_dir, ["stan", "pymdp"], logging.getLogger("t"), target_dir=None
    )
    assert allowed == {Path(arts[0]).resolve()}
    assert failures == []


def test_discrete_program_declares_forward_algorithm(tmp_path: Path) -> None:
    ok, _, arts = render_gnn_to_stan(DISCRETE_SPEC, tmp_path / "m_stan.py")
    assert ok
    text = Path(arts[1]).read_text()
    assert "dirichlet(alpha_A[s])" in text
    assert "filtered_state" in text and "log_marginal" in text
    assert "B[u[t - 1]] * alpha" in text  # vectorised scaled forward recursion


@pytest.mark.parametrize("mode", ["pre_cancel", "cancel", "timeout"])
def test_direct_stan_driver_obeys_cancellation_and_timeout(
    tmp_path: Path,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner contract uses a real child without requiring CmdStan."""
    script = tmp_path / "m_stan.py"
    script.write_text("print('stan-partial',flush=True); import time; time.sleep(30)")
    from gnn.utils.runtime_safety.framework_availability import FrameworkStatus

    monkeypatch.setattr(
        "gnn.execute.stan.stan_runner._check_stan_status",
        lambda *args, **kwargs: FrameworkStatus("stan", available=True),
    )
    token = CancelToken()
    timer = None
    if mode == "pre_cancel":
        token.cancel("before Stan launch")
    elif mode == "cancel":
        timer = threading.Timer(0.3, token.cancel, args=("stop Stan",))
        timer.start()
    started = time.monotonic()
    try:
        result = execute_stan_script(
            script,
            tmp_path / "out",
            timeout=1 if mode == "timeout" else 30,
            cancel_token=token,
        )
    finally:
        if timer:
            timer.cancel()
            timer.join(timeout=1)
    assert time.monotonic() - started < 2.9, result
    assert not result["success"]
    assert result["error_type"] == (
        "TimeoutExpired" if mode == "timeout" else "Cancelled"
    )
    assert result["cancelled"] == (mode != "timeout")
    if mode == "pre_cancel":
        assert result["containment"] == "not_started"
        assert result["stdout"] == ""
    else:
        assert result["cleanup_verified"] is True and result["streams_drained"] is True
        assert "stan-partial" in result["stdout"]


@pytest.mark.parametrize("mode", ["pre_cancel", "invalid_timeout"])
def test_stan_batch_preconditions_do_not_probe_or_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    from gnn.execute.stan import stan_runner

    script = tmp_path / "m" / "stan" / "m_stan.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('must not run')")

    def fail(*args: object, **kwargs: object) -> bool:
        raise AssertionError("precondition must precede availability probe")

    monkeypatch.setattr(stan_runner, "_check_stan_status", fail)
    token = CancelToken()
    if mode == "pre_cancel":
        token.cancel("stop before probe")
    results = stan_runner.run_stan_scripts(
        tmp_path,
        tmp_path / "out",
        timeout=0 if mode == "invalid_timeout" else 30,
        cancel_token=token,
    )
    assert len(results) == 1 and not results[0]["success"]
    assert results[0]["error_type"] == (
        "Cancelled" if mode == "pre_cancel" else "InvalidExecutionTimeout"
    )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("entry", ["direct", "batch"])
@pytest.mark.parametrize(
    "cause,skipped",
    [
        ("missing_module", True),
        ("missing_toolchain", True),
        ("probe_failed", False),
        ("probe_timeout", False),
    ],
)
def test_stan_dependency_and_probe_failure_are_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
    cause: str,
    skipped: bool,
) -> None:
    from gnn.execute.stan import stan_runner
    from gnn.utils.runtime_safety.framework_availability import FrameworkStatus

    script = tmp_path / "m" / "stan" / "m_stan.py"
    script.parent.mkdir(parents=True)
    script.write_text("raise AssertionError('must not execute')")
    diagnosis = FrameworkStatus(
        "stan",
        available=False,
        reason_code=cause,
        reason="bounded probe receipt",
        install_hint="uv sync --extra stan",
        execution_error_type="TimeoutExpired" if cause == "probe_timeout" else None,
    )
    monkeypatch.setattr(
        stan_runner, "_check_stan_status", lambda *args, **kwargs: diagnosis
    )
    out = tmp_path / "out"
    result = (
        stan_runner.execute_stan_script(script, out)
        if entry == "direct"
        else stan_runner.run_stan_scripts(tmp_path, out)[0]
    )
    assert not result["success"] and result["skipped"] is skipped
    assert result["status"] == (
        "skipped" if skipped else "timed_out" if cause == "probe_timeout" else "failed"
    )
    assert result["reason_code"] == cause and result["reason"] == diagnosis.reason
    assert result["install_hint"] == diagnosis.install_hint
    if cause == "probe_timeout":
        assert result["error_type"] == "TimeoutExpired"
    assert not out.exists()


def test_stan_readiness_uses_selected_interpreter_and_bounded_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gnn.execute.stan import stan_runner
    from gnn.utils.runtime_safety.framework_availability import FrameworkStatus

    calls: list[dict[str, object]] = []

    def diagnose(framework: str, **kwargs: object) -> FrameworkStatus:
        calls.append({"framework": framework, **kwargs})
        return FrameworkStatus(
            framework,
            available=False,
            reason_code="missing_module",
            reason="selected interpreter lacks cmdstanpy",
        )

    monkeypatch.setattr(stan_runner, "check_framework", diagnose)
    script = tmp_path / "m_stan.py"
    script.write_text("raise AssertionError('wrong interpreter')")
    result = stan_runner.execute_stan_script(
        script, tmp_path / "out", timeout=2, python_executable="target-python"
    )
    assert result["skipped"] and not result["success"]
    assert calls[0]["framework"] == "stan"
    assert calls[0]["executor"] == "target-python"
    assert calls[0]["timeout"] == 2
    assert not (tmp_path / "out").exists()


def test_stan_blocks_actual_unsafe_script_before_probe_or_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gnn.execute.stan import stan_runner

    monkeypatch.delenv("GNN_ALLOW_UNSAFE_EXEC", raising=False)
    marker = tmp_path / "executed"
    script = tmp_path / "unsafe_stan.py"
    script.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\neval('2+2')\n"
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("blocked script must not reach the runtime probe")

    monkeypatch.setattr(stan_runner, "_check_stan_status", fail)
    result = stan_runner.execute_stan_script(script, tmp_path / "out")
    assert not result["success"] and result["error_type"] == "SecurityGateBlocked"
    assert result["security_findings"]
    assert not marker.exists() and not (tmp_path / "out").exists()


@pytest.mark.parametrize("entry", ["direct", "batch"])
def test_stan_probe_cleanup_failure_preserves_priority_and_original_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
) -> None:
    from gnn.execute.stan import stan_runner
    from gnn.utils.runtime_safety.framework_availability import FrameworkStatus

    script = tmp_path / "m" / "stan" / "m_stan.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('safe')")
    diagnosis = FrameworkStatus(
        "stan",
        available=False,
        reason_code="probe_timeout",
        reason="probe process cleanup could not be verified",
        execution_error_type="TimeoutExpired",
        cleanup_verified=False,
        streams_drained=False,
    )
    monkeypatch.setattr(
        stan_runner, "_check_stan_status", lambda *args, **kwargs: diagnosis
    )
    result = (
        stan_runner.execute_stan_script(script, tmp_path / "out")
        if entry == "direct"
        else stan_runner.run_stan_scripts(tmp_path, tmp_path / "out")[0]
    )
    assert not result["success"] and not result["skipped"]
    assert result["error_type"] == "ProcessCleanupFailure"
    assert result["execution_error_type"] == "TimeoutExpired"
    assert result["cleanup_verified"] is False and result["streams_drained"] is False


@pytest.mark.needs_cmdstan
def test_stan_execution_end_to_end(tmp_path: Path) -> None:
    ok, _, arts = render_gnn_to_stan(DISCRETE_SPEC, tmp_path / "m_stan.py")
    assert ok
    result = execute_stan_script(arts[0], tmp_path / "out", timeout=900)
    assert result["success"], result["stderr"][-2000:]
    res = json.loads(Path(result["results_file"]).read_text())
    assert res["framework"] == "stan" and res["model_kind"] == "discrete"
    assert len(res["beliefs"]) == 6 and len(res["beliefs"][0]) == 2
    assert len(res["A_posterior_mean"]) == 2 and len(res["A_posterior_mean"][0]) == 2
    assert res["validation"]["all_valid"] is True
