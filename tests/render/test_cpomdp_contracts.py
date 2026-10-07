"""Strict admission plus native released-wheel Kalman/EFE acceptance."""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gnn.analysis.cpomdp import analyze_payload
from gnn.execute.subprocess_envelope import run_subprocess_envelope
from gnn.render.cpomdp import render_gnn_to_cpomdp
from gnn.render.cpomdp.admission import admit_search
from gnn.render.cpomdp.script_template import resolve_render_options


def _spec(controlled: bool = True) -> dict[str, Any]:
    initial: dict[str, Any] = {
        "F": [[0.9, 0.1], [0.0, 0.8]],
        "H": [[1.0, 0.2]],
        "Q": [[0.01, 0.0], [0.0, 0.02]],
        "R": [[0.1]],
        "prior_mean": [-1.0, 0.5],
        "prior_cov": [[1.0, 0.0], [0.0, 0.5]],
    }
    if controlled:
        initial.update(goal_mean=[1.0, 0.0], control_gain=0.1)
    return {
        "model_kind": "continuous",
        "model_name": "asymmetric_cpomdp_acceptance",
        "initialparameterization": initial,
        "model_parameters": {"num_timesteps": 5, "random_seed": 7},
    }


@pytest.mark.parametrize("value", [True, 1.5, 0, -1, "2"])
def test_horizon_rejects_boolean_fractional_or_noninteger(value: Any) -> None:
    with pytest.raises(ValueError):
        resolve_render_options({"horizon": value})


@pytest.mark.parametrize("name", ["action_scale", "goal_precision"])
@pytest.mark.parametrize("value", [float("inf"), float("nan"), True, 0, -0.5])
def test_nonfinite_or_nonpositive_options_are_rejected(name: str, value: Any) -> None:
    with pytest.raises(ValueError):
        resolve_render_options({name: value})


def test_admission_matches_conservative_formula_and_rejects_before_write(
    tmp_path: Path,
) -> None:
    receipt = admit_search(2, 1, 2, 5, 1, True)
    assert receipt["policy_count"] == 9
    assert receipt["estimated_bytes"] == 29696
    assert admit_search(2, 1, 2, 5, 1000000, False)["policy_count"] == 0
    destination = tmp_path / "rejected.py"
    success, message, artifacts = render_gnn_to_cpomdp(
        _spec(), destination, {"horizon": 1000000}
    )
    assert not success and "max_policies" in message
    assert not artifacts and not destination.exists()
    with pytest.raises(ValueError, match="max_step_evaluations"):
        admit_search(2, 1, 2, 5, 1, True, max_step_evaluations=8)
    with pytest.raises(ValueError, match="estimated allocation"):
        admit_search(2, 1, 2, 1000000, 1, True)
    with pytest.raises(ValueError, match="ceiling"):
        admit_search(2, 1, 2, 5, 1, True, max_policies=4097)


def test_backend_requires_explicit_selection() -> None:
    from gnn.render.framework_registry import FRAMEWORK_REGISTRY, get_default_frameworks

    assert FRAMEWORK_REGISTRY["cpomdp"]["experimental"]
    assert "cpomdp" not in get_default_frameworks()


def _kalman_reference(
    initial: dict[str, Any], results: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    F, H, Q, R = (np.asarray(initial[key]) for key in ("F", "H", "Q", "R"))
    mean = np.asarray(initial["prior_mean"])
    cov = np.asarray(initial["prior_cov"])
    means, covariances = [], []
    for timestep, observation in enumerate(results["observations_continuous"]):
        if timestep:
            mean = F @ mean + np.asarray(results["controls"][timestep - 1])
            cov = F @ cov @ F.T + Q
        gain = np.linalg.solve(H @ cov @ H.T + R, H @ cov).T
        mean = mean + gain @ (np.asarray(observation) - H @ mean)
        residual = np.eye(len(mean)) - gain @ H
        cov = residual @ cov @ residual.T + gain @ R @ gain.T
        means.append(mean.copy())
        covariances.append(cov.copy())
    return np.asarray(means), np.asarray(covariances)


def _efe_reference(
    initial: dict[str, Any], mean: np.ndarray, cov: np.ndarray, policy: Any
) -> tuple[float, float, float]:
    F, H, Q, R = (np.asarray(initial[key]) for key in ("F", "H", "Q", "R"))
    goal = H @ np.asarray(initial["goal_mean"])
    pragmatic = epistemic = 0.0
    for action in policy:
        mean = F @ mean + action
        cov = F @ cov @ F.T + Q
        sensor_cov = H @ cov @ H.T + R
        delta = H @ mean - goal
        pragmatic += float(0.5 * (delta @ delta + np.trace(sensor_cov)))
        epistemic += float(
            0.5 * (np.linalg.slogdet(sensor_cov)[1] - np.linalg.slogdet(R)[1])
        )
        gain = np.linalg.solve(sensor_cov, H @ cov).T
        residual = np.eye(len(mean)) - gain @ H
        cov = residual @ cov @ residual.T + gain @ R @ gain.T
    return pragmatic - epistemic, pragmatic, epistemic


@pytest.mark.needs_cpomdp
@pytest.mark.parametrize("mode", ["passive", "parity", "efe"])
def test_released_wheel_native_filter_and_policy_scores(
    mode: str, tmp_path: Path
) -> None:
    from PIL import Image

    spec = _spec(controlled=mode != "passive")
    script = tmp_path / "model_cpomdp.py"
    options = {"control_mode": "parity" if mode == "parity" else "efe", "horizon": 2}
    success, message, _ = render_gnn_to_cpomdp(spec, script, options)
    assert success, message
    output = tmp_path / "results"
    env = dict(os.environ, CPOMDP_OUTPUT_DIR=str(output))
    envelope = run_subprocess_envelope(
        [sys.executable, str(script)], env=env, timeout=45
    )
    assert envelope["success"], envelope
    results = json.loads((output / "simulation_results.json").read_text())
    assert results["cpomdp_version"] == "0.4.4"
    assert results["validation"]["all_valid"]
    assert results["source_identity"]["model_contract_sha256"]
    assert results["measured_resources"]["process_peak_rss_bytes"] > 0
    means, covariances = _kalman_reference(spec["initialparameterization"], results)
    np.testing.assert_allclose(results["beliefs"], means, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        results["posterior_cov"], covariances, rtol=1e-9, atol=1e-10
    )
    if mode == "passive":
        assert results["n_policies"] == 0
        assert not results["efe_history"] and not results["selected_policy"]
        assert np.all(np.asarray(results["controls"]) == 0)
    else:
        actions = np.asarray(results["action_set"])
        assert len(actions) == 9
        policies = list(itertools.product(actions, repeat=2))
        assert results["n_policies"] == len(policies) == 81
        for timestep, (mean, cov) in enumerate(zip(means, covariances)):
            scored = np.asarray(
                [
                    _efe_reference(spec["initialparameterization"], mean, cov, policy)
                    for policy in policies
                ]
            )
            np.testing.assert_allclose(
                results["efe_history"][timestep], scored[:, 0], rtol=1e-9, atol=1e-10
            )
            selected = results["selected_policy_index"][timestep]
            assert selected == int(np.argmin(scored[:, 0]))
            np.testing.assert_allclose(
                results["pragmatic_term"][timestep], scored[selected, 1], rtol=1e-9
            )
            np.testing.assert_allclose(
                results["epistemic_term"][timestep], scored[selected, 2], rtol=1e-9
            )
    analysis = analyze_payload(results, tmp_path / "analysis")
    assert "categorical_entropy" in analysis["unavailable_metrics"]
    assert "entropy" not in analysis["metrics"]
    for path in analysis["artifacts"]:
        with Image.open(path) as image:
            image.verify()


@pytest.mark.needs_cpomdp
def test_all_invalid_scores_fail_before_policy_selection(tmp_path: Path) -> None:
    script = tmp_path / "all_invalid_cpomdp.py"
    assert render_gnn_to_cpomdp(_spec(), script)[0]
    source = script.read_text().replace(
        "g, parts = score(belief)",
        "g, parts = score(belief)\n            g = jnp.full_like(g, jnp.nan)",
    )
    script.write_text(source)
    envelope = run_subprocess_envelope(
        [sys.executable, str(script)],
        timeout=45,
        env={"CPOMDP_OUTPUT_DIR": str(tmp_path / "results")},
    )
    assert not envelope["success"]
    assert "All enumerated cpomdp policy scores are invalid" in envelope["stderr"]
    assert not (tmp_path / "results/simulation_results.json").exists()


def _ready_cpomdp(monkeypatch: pytest.MonkeyPatch) -> None:
    from gnn.utils.runtime_safety import framework_availability

    monkeypatch.setattr(
        framework_availability,
        "check_framework",
        lambda name, **kwargs: framework_availability.FrameworkStatus(name, True),
    )


def test_direct_cpomdp_probe_dispatch_and_hardware_share_the_same_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from gnn.execute import executor
    from gnn.execute.preconditions import current_script_deadline
    from gnn.utils.runtime_safety import framework_availability

    script = tmp_path / "model_cpomdp.py"
    script.write_text("print('safe')")
    seen: dict[str, Any] = {}

    def ready(name: str, **kwargs: Any) -> Any:
        assert name == "cpomdp"
        seen["probe_deadline"] = kwargs["deadline_monotonic"]
        time.sleep(0.03)
        return framework_availability.FrameworkStatus(name, True)

    def completed(command: Any, **kwargs: Any) -> dict[str, Any]:
        seen["dispatch_deadline"] = kwargs["deadline_monotonic"]
        seen["dispatch_remaining"] = kwargs["deadline_monotonic"] - time.monotonic()
        return {"success": True, "stdout": "science-ready", "cleanup_verified": True}

    def hardware() -> list[str]:
        seen["hardware_deadline"] = current_script_deadline()
        return ["cpu"]

    monkeypatch.setattr(framework_availability, "check_framework", ready)
    monkeypatch.setattr(executor, "run_subprocess_envelope", completed)
    monkeypatch.setattr(executor, "get_available_hardware", hardware)
    result = executor.GNNExecutor(str(tmp_path / "out")).execute_gnn_model(
        str(script), "cpomdp", timeout=2
    )
    assert result["success"]
    assert (
        seen["probe_deadline"] == seen["dispatch_deadline"] == seen["hardware_deadline"]
    )
    assert 0 < seen["dispatch_remaining"] < 1.99
    assert current_script_deadline() is None


@pytest.mark.needs_posix
def test_direct_cpomdp_actual_slow_dependency_import_is_budgeted_and_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    import psutil

    from gnn.execute.executor import GNNExecutor
    from gnn.execute.preconditions import current_script_deadline

    pid_file = tmp_path / "probe_pid"
    (tmp_path / "cpomdp.py").write_text(
        "import os, time\nfrom pathlib import Path\n"
        f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    script = tmp_path / "model_cpomdp.py"
    script.write_text("print('must not execute')")
    destination = tmp_path / "results"
    started = time.monotonic()
    result = GNNExecutor(str(tmp_path / "executor")).execute_gnn_model(
        str(script), "cpomdp", options={"output_dir": str(destination)}, timeout=1.5
    )
    assert time.monotonic() - started < 2.5
    assert not result["success"] and result["status"] == "timed_out"
    assert result["error_type"] == "TimeoutExpired"
    assert result["cleanup_verified"] is True and result["streams_drained"] is True
    assert pid_file.exists() and not psutil.pid_exists(int(pid_file.read_text()))
    assert not destination.exists()
    assert current_script_deadline() is None


@pytest.mark.needs_posix
@pytest.mark.parametrize("mode", ["timeout", "cancel"])
def test_direct_cpomdp_actual_script_exhaustion_preserves_partial_output_and_reaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    import threading
    import time

    from gnn.execute.executor import GNNExecutor
    from gnn.execute.preconditions import current_script_deadline
    from gnn.execute.subprocess_envelope import CancelToken

    _ready_cpomdp(monkeypatch)
    script = tmp_path / "model_cpomdp.py"
    script.write_text("print('cpomdp-partial',flush=True)\nimport time\ntime.sleep(30)")
    token = CancelToken()
    timer = (
        threading.Timer(0.3, token.cancel, args=("stop cpomdp",))
        if mode == "cancel"
        else None
    )
    if timer is not None:
        timer.start()
    started = time.monotonic()
    try:
        result = GNNExecutor(str(tmp_path / "out")).execute_gnn_model(
            str(script),
            "cpomdp",
            timeout=1.5 if mode == "timeout" else 10,
            cancel_token=token,
        )
    finally:
        if timer is not None:
            timer.cancel()
            timer.join(timeout=1)
    assert time.monotonic() - started < 2.5
    assert not result["success"]
    assert result["error_type"] == (
        "TimeoutExpired" if mode == "timeout" else "Cancelled"
    )
    assert result["cleanup_verified"] is True and result["streams_drained"] is True
    assert "cpomdp-partial" in result["stdout"]
    assert current_script_deadline() is None


@pytest.mark.parametrize("cancel_when", ["before", "during_probe"])
def test_direct_cpomdp_cancellation_never_dispatches_or_runs_hardware_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_when: str
) -> None:
    from gnn.execute import executor
    from gnn.execute.preconditions import current_script_deadline
    from gnn.execute.subprocess_envelope import CancelToken
    from gnn.utils.runtime_safety import framework_availability

    token = CancelToken()
    calls = []
    script = tmp_path / "model_cpomdp.py"
    script.write_text("print('not executed')")

    def ready(name: str, **kwargs: Any) -> Any:
        calls.append(name)
        token.cancel("cancelled during readiness")
        return framework_availability.FrameworkStatus(name, True)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("cancelled work must not dispatch or query hardware")

    if cancel_when == "before":
        token.cancel("pre-cancelled")
    monkeypatch.setattr(framework_availability, "check_framework", ready)
    monkeypatch.setattr(executor, "run_subprocess_envelope", forbidden)
    monkeypatch.setattr(executor, "get_available_hardware", forbidden)
    result = executor.GNNExecutor(str(tmp_path / "out")).execute_gnn_model(
        str(script), "cpomdp", cancel_token=token
    )
    assert not result["success"] and result["cancelled"]
    assert result["error_type"] == "Cancelled"
    assert calls == ([] if cancel_when == "before" else ["cpomdp"])
    assert current_script_deadline() is None


def test_invocation_deadline_does_not_leak_between_calls_or_after_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.execute import executor
    from gnn.execute.preconditions import current_script_deadline

    _ready_cpomdp(monkeypatch)
    script = tmp_path / "model_cpomdp.py"
    script.write_text("print('safe')")
    seen = []

    def raise_during_execution(command: Any, **kwargs: Any) -> Any:
        seen.append(current_script_deadline())
        raise ValueError("known child boundary failure")

    model_executor = executor.GNNExecutor(str(tmp_path / "out"))
    monkeypatch.setattr(executor, "run_subprocess_envelope", raise_during_execution)
    failed = model_executor.execute_gnn_model(str(script), "cpomdp", timeout=1)
    assert not failed["success"] and failed["error_type"] == "ValueError"
    assert current_script_deadline() is None

    def completed(command: Any, **kwargs: Any) -> dict[str, Any]:
        seen.append(current_script_deadline())
        return {"success": True, "stdout": "science-ready"}

    monkeypatch.setattr(executor, "run_subprocess_envelope", completed)
    monkeypatch.setattr(executor, "get_available_hardware", lambda: ["cpu"])
    accepted = model_executor.execute_gnn_model(str(script), "cpomdp", timeout=10)
    assert accepted["success"] and seen[1] > seen[0] + 8
    assert current_script_deadline() is None


@pytest.mark.needs_posix
def test_explicit_later_child_deadline_cannot_extend_the_scoped_ceiling() -> None:
    import time

    from gnn.execute.preconditions import (
        current_script_deadline,
        script_execution_scope,
    )

    started = time.monotonic()
    with script_execution_scope(1):
        result = run_subprocess_envelope(
            [
                sys.executable,
                "-c",
                "print('partial',flush=True);import time;time.sleep(30)",
            ],
            timeout=30,
            deadline_monotonic=time.monotonic() + 30,
        )
    assert time.monotonic() - started < 1.8
    assert not result["success"] and result["error_type"] == "TimeoutExpired"
    assert result["cleanup_verified"] is True and result["streams_drained"] is True
    assert "partial" in result["stdout"]
    assert current_script_deadline() is None


def test_expired_optional_hardware_diagnostic_is_omitted_without_changing_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.execute import executor

    _ready_cpomdp(monkeypatch)
    script = tmp_path / "model_cpomdp.py"
    script.write_text("print('safe')")
    original_precondition = executor._execution_precondition
    diagnostic_checks: list[int] = []

    def exhausted_diagnostic(timeout: Any, *args: Any, **kwargs: Any) -> Any:
        if timeout == 5:
            diagnostic_checks.append(timeout)
            return {
                "success": False,
                "status": "timed_out",
                "error_type": "TimeoutExpired",
                "error": "Scoped execution deadline exhausted",
            }
        return original_precondition(timeout, *args, **kwargs)

    def completed(command: Any, **kwargs: Any) -> dict[str, Any]:
        return {"success": True, "stdout": "science-ready", "cleanup_verified": True}

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "exhausted diagnostics cannot obtain a fresh five-second budget"
        )

    monkeypatch.setattr(executor, "run_subprocess_envelope", completed)
    monkeypatch.setattr(executor, "get_available_hardware", forbidden)
    monkeypatch.setattr(executor, "_execution_precondition", exhausted_diagnostic)
    result = executor.GNNExecutor(str(tmp_path / "out")).execute_gnn_model(
        str(script), "cpomdp", timeout=2
    )
    assert result["success"] and result["stdout"] == "science-ready"
    assert result["execution_device"] == "unknown"
    assert "omitted" in result["execution_device_note"]
    assert diagnostic_checks == [5]


def test_known_hardware_cleanup_failure_invalidates_containment_and_preserves_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.execute import executor
    from gnn.execute.preconditions import current_script_deadline

    _ready_cpomdp(monkeypatch)
    script = tmp_path / "model_cpomdp.py"
    script.write_text("print('safe')")

    def completed_or_failed_hardware(command: Any, **kwargs: Any) -> dict[str, Any]:
        if len(command) >= 3 and command[1] == "-c":
            return {
                "success": False,
                "error_type": "ProcessCleanupFailure",
                "execution_error_type": "TimeoutExpired",
                "cleanup_verified": False,
                "streams_drained": False,
                "containment": "observed_process_tree",
            }
        return {
            "success": True,
            "stdout": "science-ready",
            "simulation_data": {"validated": "preserved"},
            "cleanup_verified": True,
            "streams_drained": True,
        }

    monkeypatch.setattr(
        executor, "run_subprocess_envelope", completed_or_failed_hardware
    )
    result = executor.GNNExecutor(str(tmp_path / "out")).execute_gnn_model(
        str(script), "cpomdp"
    )
    assert not result["success"] and result["error_type"] == "ProcessCleanupFailure"
    assert result["execution_result_success"] is True
    assert result["stdout"] == "science-ready" and result["simulation_data"] == {
        "validated": "preserved"
    }
    assert result["cleanup_verified"] is False and result["streams_drained"] is False
    assert result["execution_cleanup_verified"] is True
    assert result["hardware_probe"]["execution_error_type"] == "TimeoutExpired"
    assert current_script_deadline() is None
