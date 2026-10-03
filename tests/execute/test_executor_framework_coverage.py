"""Smoke tests for ``execute.executor.execute_rendered_simulators`` framework coverage.

Verifies that the NumPyro and PyTorch branches are wired in alongside PyMDP,
RxInfer, DisCoPy, ActiveInference.jl, and JAX — and that all framework result
buckets are populated even when the render input is empty (everything should
be either SKIPPED or run-and-report-no-scripts, not missing).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.execute import executor as executor_module  # noqa: E402
from gnn.execute.executor import execute_rendered_simulators  # noqa: E402

EXPECTED_FRAMEWORK_KEYS: set[Any] = {
    "pymdp_executions",
    "rxinfer_executions",
    "discopy_executions",
    "activeinference_executions",
    "jax_executions",
    "numpyro_executions",
    "pytorch_executions",
    "ngclearn_executions",
    "thrml_executions",
    "lean_executions",
    "stan_executions",
    "bnlearn_executions",
}


def test_framework_spec_registry_matches_summary_contract() -> None:
    specs = executor_module._framework_specs(resolve_availability=False)

    assert [spec.framework_dir_key for spec in specs] == list(
        executor_module.FRAMEWORK_DIR_NAMES
    )
    assert {spec.result_key for spec in specs} == EXPECTED_FRAMEWORK_KEYS
    assert all(spec.runner is not None or not spec.available for spec in specs)
    assert all(spec.readiness_pending for spec in specs)


def test_executor_covers_all_frameworks(tmp_path: Path) -> None:
    import logging

    empty_render_dir = tmp_path / "11_render_output"
    empty_render_dir.mkdir()
    output_dir = tmp_path / "out"

    logger = logging.getLogger("test_executor_framework_coverage")

    result = execute_rendered_simulators(
        target_dir=empty_render_dir,
        output_dir=output_dir,
        logger=logger,
        recursive=False,
        verbose=False,
    )

    # With an empty render dir every framework should either skip or succeed
    # without exploding. The function returns True on all-success / all-skip.
    assert isinstance(result, bool)

    summary_file = (
        output_dir / "12_execute_output" / "summaries" / "execution_summary.json"
    )
    assert summary_file.exists(), (
        f"Expected execution summary at {summary_file}, got tree:"
        f" {[p.relative_to(tmp_path) for p in tmp_path.rglob('*') if p.is_file()]}"
    )

    summary = json.loads(summary_file.read_text())
    assert EXPECTED_FRAMEWORK_KEYS.issubset(summary.keys()), (
        f"Missing framework buckets: {EXPECTED_FRAMEWORK_KEYS - summary.keys()}"
    )

    framework_dirs = summary.get("framework_execution_dirs", {})
    assert {"numpyro", "pytorch"}.issubset(framework_dirs.keys()), (
        "numpyro/pytorch output directories not declared in framework_execution_dirs"
    )


def test_failed_framework_runner_logs_warning_not_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import logging

    log_events: list[tuple[str, str]] = []

    def sample_runner(**kwargs: Any) -> bool:
        return False

    def sample_log_step_success(logger: logging.Logger, message: str) -> None:
        log_events.append(("success", message))

    def sample_log_step_warning(logger: logging.Logger, message: str) -> None:
        log_events.append(("warning", message))

    monkeypatch.setattr(executor_module, "log_step_success", sample_log_step_success)
    monkeypatch.setattr(executor_module, "log_step_warning", sample_log_step_warning)

    spec = executor_module.ExecutorFrameworkSpec(
        framework_dir_key="sample",
        result_key="sample_executions",
        available=True,
        runner=sample_runner,
        operation_name="sample_operation",
        start_message="Running sample framework",
        success_message="Sample framework succeeded",
        failure_message="Sample framework failed",
        unavailable_log="Sample framework unavailable",
        unavailable_message="Sample framework unavailable",
        success_log="Sample framework completed",
        warning_log_prefix="Sample framework warning",
    )
    framework_dirs = {"sample": tmp_path / "sample"}
    framework_dirs["sample"].mkdir()
    execution_results: dict[str, Any] = {
        "total_successes": 0,
        "total_failures": 0,
        "sample_executions": [],
    }

    executor_module._execute_framework_spec(
        spec=spec,
        target_dir=tmp_path,
        framework_dirs=framework_dirs,
        execution_results=execution_results,
        logger=logging.getLogger("test_failed_framework_runner_logs_warning"),
        recursive=False,
        verbose=False,
    )

    assert execution_results["total_successes"] == 0
    assert execution_results["total_failures"] == 1
    assert execution_results["sample_executions"][0]["status"] == "FAILED"
    assert ("warning", "Sample framework failed") in log_events
    assert all(event_type != "success" for event_type, _ in log_events)
