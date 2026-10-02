"""Candidate-scoped readiness and honest required-work aggregation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from gnn.execute import executor
from gnn.execute import executor_report as report
from gnn.utils.runtime_safety.framework_availability import FrameworkStatus


def _summary(output: Path) -> dict[str, Any]:
    path = output / "12_execute_output" / "summaries" / "execution_summary.json"
    return json.loads(path.read_text())


def _forbid_probe(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("metadata/zero work must not probe optional backends")


def test_metadata_initialization_count_and_empty_batch_never_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor, "_runner_state", _forbid_probe)
    source = tmp_path / "render"
    source.mkdir()
    metadata = report._initialize_execution_results(source, {})
    assert report._count_framework_execution_records(metadata) == 0
    assert executor.execute_rendered_simulators(
        source, tmp_path / "out", logging.getLogger(__name__)
    )
    summary = _summary(tmp_path / "out")
    assert summary["total_successes"] == summary["total_failures"] == 0
    assert all(
        summary[spec.result_key][0]["required_work"] is False
        for spec in executor._framework_specs(resolve_availability=False)
    )


@pytest.mark.parametrize(
    "reason_code,expected_status,error_type",
    [
        ("missing_module", "SKIPPED", "DependencyUnavailable"),
        ("probe_failed", "FAILED", "FrameworkProbeFailure"),
        ("probe_timeout", "FAILED", "TimeoutExpired"),
        ("cleanup_failed", "FAILED", "ProcessCleanupFailure"),
    ],
)
def test_candidate_readiness_preserves_failure_type_and_never_accepts_unfinished_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason_code: str,
    expected_status: str,
    error_type: str,
) -> None:
    source = tmp_path / "render"
    (source / "model" / "jax").mkdir(parents=True)
    script = source / "model" / "jax" / "model_jax.py"
    script.write_text("print('this must not execute')")
    calls = []

    def state(key: str) -> executor._RunnerState:
        calls.append(key)
        assert key == "jax", "unrelated frameworks must not be probed"
        diagnosis = FrameworkStatus(
            key,
            False,
            reason_code=reason_code,
            reason="known readiness evidence",
            execution_error_type="TimeoutExpired"
            if reason_code in {"probe_timeout", "cleanup_failed"}
            else None,
            cleanup_verified=False if reason_code == "cleanup_failed" else None,
        )
        return executor._RunnerState(False, _forbid_probe, diagnosis)

    monkeypatch.setattr(executor, "_runner_state", state)
    assert not executor.execute_rendered_simulators(
        source, tmp_path / "out", logging.getLogger(__name__)
    )
    result = _summary(tmp_path / "out")["jax_executions"][0]
    assert calls == ["jax"]
    assert result["status"] == expected_status and result["error_type"] == error_type
    assert result["required_work"] is True
    assert result["script_results"][0]["script"] == str(script)
    assert result["script_results"][0]["reason_code"] == reason_code
    if reason_code == "cleanup_failed":
        assert result["execution_error_type"] == "TimeoutExpired"
    assert not report._log_execution_outcome(
        _summary(tmp_path / "out"), logging.getLogger(__name__)
    )
    assert calls == ["jax"], "summary counting must not repeat readiness probes"


@pytest.mark.parametrize("empty_records", [False, True])
def test_available_runner_cannot_accept_nonempty_required_scripts_as_all_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_records: bool
) -> None:
    source = tmp_path / "render"
    (source / "model" / "stan").mkdir(parents=True)
    script = source / "model" / "stan" / "model_stan.py"
    script.write_text("print('driver')")
    monkeypatch.setattr(
        executor, "_runner_state", lambda key: executor._RunnerState(True, None)
    )
    monkeypatch.setattr(
        "gnn.execute.stan.stan_runner.run_stan_scripts",
        lambda **kwargs: (
            []
            if empty_records
            else [{"script": str(script), "success": False, "skipped": True}]
        ),
    )
    assert not executor.execute_rendered_simulators(
        source, tmp_path / "out", logging.getLogger(__name__)
    )
    summary = _summary(tmp_path / "out")
    assert summary["required_unfinished"] == 1
    assert summary["stan_executions"][0]["status"] == "SKIPPED"


def test_invalid_batch_timeout_never_probes_or_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "render"
    (source / "jax").mkdir(parents=True)
    (source / "jax" / "model_jax.py").write_text("print('driver')")
    monkeypatch.setattr(executor, "_runner_state", _forbid_probe)
    assert not executor.execute_rendered_simulators(
        source, tmp_path / "out", logging.getLogger(__name__), timeout=0
    )
    result = _summary(tmp_path / "out")["jax_executions"][0]
    assert result["error_type"] == "InvalidExecutionTimeout"


def test_explicit_list_readiness_retains_probe_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def state(key: str) -> executor._RunnerState:
        return executor._RunnerState(
            False,
            None,
            FrameworkStatus(
                key,
                False,
                reason_code="probe_timeout",
                execution_error_type="TimeoutExpired",
            ),
        )

    monkeypatch.setattr(executor, "_runner_state", state)
    assert all(
        record["readiness"]["reason_code"] == "probe_timeout"
        and record["readiness"]["execution_error_type"] == "TimeoutExpired"
        for record in executor.list_frameworks()
    )


def test_r_only_trusted_bnlearn_batch_does_not_require_python_bnlearn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.execute.bnlearn import bnlearn_runner

    source = tmp_path / "render"
    (source / "model" / "bnlearn").mkdir(parents=True)
    script = source / "model" / "bnlearn" / "model_bnlearn.R"
    script.write_text("print('R driver')")
    # R has no maintained static parser in the shared gate; its established
    # direct-runner contract requires explicit trusted-local opt-out.
    monkeypatch.setenv("GNN_ALLOW_UNSAFE_EXEC", "1")
    monkeypatch.setattr(executor, "_runner_state", _forbid_probe)
    monkeypatch.setattr(bnlearn_runner, "_check_bnlearn_status", _forbid_probe)
    monkeypatch.setattr(
        bnlearn_runner,
        "_check_r_bnlearn_status",
        lambda *args: FrameworkStatus("bnlearn", True),
    )
    commands = []

    def completed(command: list[str], **kwargs: Any) -> dict[str, Any]:
        commands.append(command)
        return {
            "success": True,
            "return_code": 0,
            "stdout": "R driver",
            "stderr": "",
            "duration_seconds": 0.1,
            "cleanup_verified": True,
            "streams_drained": True,
        }

    monkeypatch.setattr(bnlearn_runner, "run_subprocess_envelope", completed)
    assert executor.execute_rendered_simulators(
        source, tmp_path / "out", logging.getLogger(__name__)
    )
    assert commands == [["Rscript", str(script.resolve())]]
    record = _summary(tmp_path / "out")["bnlearn_executions"][0]
    assert record["status"] == "SUCCESS"
    assert record["script_results"][0]["language"] == "r"
