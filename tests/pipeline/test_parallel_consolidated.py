"""Parallel-tier consolidation: --consolidated-steps under --parallel (V4-STAGE slice D).

Both execution tiers share one dispatcher (`_execute_selected_step`), so a
parallel run produces consolidated in-process receipts for whitelisted stems
and canonical subprocess receipts for everything else. These tests drive the
real orchestration with monkeypatched step functions — no subprocesses, no
network — and pin receipt ordering, mode routing, and aggregation parity
with the serial consolidated run.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

import gnn.main as orchestrator
from gnn.api.pipeline_runner import PIPELINE_SUMMARY
from gnn.utils.arguments.pipeline_arguments import PipelineArguments

# Two waves after 0_template: {3_gnn, 5_type_checker} share a tier (exercising
# the ThreadPoolExecutor submit path), 9_advanced_viz is not whitelisted and
# must keep the subprocess path in the same run.
STEPS: list[tuple[str, str]] = [
    ("0_template.py", "Template"),
    ("3_gnn.py", "Parse GNN files"),
    ("5_type_checker.py", "Type check"),
    ("9_advanced_viz.py", "Advanced visualization"),
]
DEPENDENCIES: dict[int, list[int]] = {0: [], 3: [0], 5: [0], 9: [3, 5]}

LOGGER = logging.getLogger("test_parallel_consolidated")


class RecordingVisualLogger:
    """Typed visual-logger stand-in recording the calls the pipeline makes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def set_correlation_id(self, correlation_id: str) -> None:
        self.calls.append(("set_correlation_id", (correlation_id,)))

    def print_progress(self, done: int, total: int, message: str) -> None:
        self.calls.append(("print_progress", (done, total, message)))

    def print_step_header(self, *args: object) -> None:
        self.calls.append(("print_step_header", args))


@pytest.fixture
def isolated_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PipelineArguments:
    target = tmp_path / "input"
    target.mkdir()
    (target / "model.md").write_text("# Model\n")
    args = PipelineArguments(target_dir=target, output_dir=tmp_path / "output")
    args.consolidated_steps = True
    args.parallel = True
    monkeypatch.setattr(orchestrator, "_start_pipeline_run", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_print_pipeline_completion", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_write_performance_dashboard", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_write_final_pipeline_report", lambda *a: None)
    monkeypatch.delenv("GNN_RUN_ID", raising=False)
    return args


def _inject_steps(monkeypatch: pytest.MonkeyPatch, args: PipelineArguments) -> None:
    """Force the pipeline onto the fixed four-step selection and dependency map."""

    def context(
        override_args: PipelineArguments | None,
        override_config: dict[str, Any] | None,
    ) -> tuple[Any, ...]:
        summary = orchestrator._initialize_pipeline_summary(args, STEPS, {})
        return (
            args,
            {},
            STEPS,
            summary,
            RecordingVisualLogger(),
            "correlation",
            LOGGER,
        )

    monkeypatch.setattr(orchestrator, "_prepare_pipeline_context", context)
    monkeypatch.setattr(
        "gnn.utils.pipeline_orchestration.pipeline_step_dependencies.PIPELINE_STEP_DEPENDENCIES",
        DEPENDENCIES,
    )


def _subprocess_fake(calls: list[Any], lock: threading.Lock) -> Any:
    """Canonical subprocess stand-in: records calls, returns a bare receipt.

    Deliberately omits ``execution_mode`` so the shared recording tail's
    setdefault behavior stays pinned at "subprocess".
    """

    def fake(
        script_name: str,
        args: Any,
        logger: Any,
        *,
        run_id: str | None = None,
        pipeline_config: Any = None,
    ) -> dict[str, Any]:
        with lock:
            calls.append(("subprocess", script_name))
        return {
            "status": "SUCCESS",
            "stdout": f"subprocess {script_name}\n",
            "stderr": "",
            "memory_usage_mb": 0.0,
            "peak_memory_mb": 0.0,
            "memory_delta_mb": 0.0,
            "exit_code": 0,
            "retry_count": 0,
            "prerequisite_check": True,
            "dependency_warnings": [],
        }

    return fake


def _consolidated_fake(calls: list[Any], lock: threading.Lock) -> Any:
    """Consolidated executor stand-in mirroring its receipt contract."""

    def fake(
        script_name: str,
        args: Any,
        logger: Any,
        *,
        run_id: str | None = None,
        pipeline_config: Any = None,
    ) -> dict[str, Any]:
        with lock:
            calls.append(("consolidated", script_name, run_id))
        return {
            "status": "SUCCESS",
            "stdout": f"consolidated {script_name}\n",
            "stderr": "",
            "memory_usage_mb": 0.0,
            "peak_memory_mb": 0.0,
            "memory_delta_mb": 0.0,
            "exit_code": 0,
            "retry_count": 0,
            "prerequisite_check": True,
            "dependency_warnings": [],
            "execution_mode": "consolidated",
        }

    return fake


def read_receipt(args: PipelineArguments) -> dict[str, Any]:
    receipt: dict[str, Any] = json.loads(
        (args.output_dir / PIPELINE_SUMMARY).read_text()
    )
    return receipt


def _aggregation(receipt: dict[str, Any]) -> list[tuple[str, str, str, int]]:
    """Pinned aggregation shape: mode, status, and exit code per recorded step."""
    return [
        (step["script_name"], step["status"], step["execution_mode"], step["exit_code"])
        for step in receipt["steps"]
    ]


def test_parallel_consolidated_preserves_hard_process_boundary(
    isolated_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch
) -> None:
    _inject_steps(monkeypatch, isolated_run)
    lock = threading.Lock()
    calls: list[Any] = []
    monkeypatch.setattr(
        orchestrator, "execute_pipeline_step", _subprocess_fake(calls, lock)
    )
    monkeypatch.setattr(
        "gnn.pipeline.step_executor.execute_step_in_process",
        _consolidated_fake(calls, lock),
    )
    serial_tier_calls: list[str] = []
    real_iteration = orchestrator._execute_pipeline_iteration

    def record_serial_iteration(*call_args: Any) -> Any:
        serial_tier_calls.append(call_args[1])
        return real_iteration(*call_args)

    monkeypatch.setattr(
        orchestrator,
        "_execute_pipeline_iteration",
        record_serial_iteration,
    )

    assert orchestrator.main(isolated_run) == 0
    receipt = read_receipt(isolated_run)

    modes = {step["script_name"]: step["execution_mode"] for step in receipt["steps"]}
    assert modes == {
        "0_template.py": "subprocess",
        "3_gnn.py": "subprocess",
        "5_type_checker.py": "subprocess",
        "9_advanced_viz.py": "subprocess",
    }
    # Dependency ordering preserved: wave 1, then the shared wave, then viz.
    assert [step["script_name"] for step in receipt["steps"]] == [
        step[0] for step in STEPS
    ]
    # The two same-tier steps went through the pool submit path, not the
    # single-step serial iteration helper.
    assert sorted(serial_tier_calls) == ["0_template.py", "9_advanced_viz.py"]
    # Consolidated workers ran under the run's GNN_RUN_ID scope.
    assert not any(call[0] == "consolidated" for call in calls)
    assert all(
        step["execution_fallback_reason"] == "hard_deadline_requires_process_boundary"
        for step in receipt["steps"]
    )
    assert "GNN_RUN_ID" not in os.environ


def test_serial_and_parallel_consolidated_aggregation_match(
    isolated_run: PipelineArguments, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _inject_steps(monkeypatch, isolated_run)
    lock = threading.Lock()
    calls: list[Any] = []
    subprocess_fake = _subprocess_fake(calls, lock)
    consolidated_fake = _consolidated_fake(calls, lock)
    monkeypatch.setattr(orchestrator, "execute_pipeline_step", subprocess_fake)
    monkeypatch.setattr(
        "gnn.pipeline.step_executor.execute_step_in_process", consolidated_fake
    )

    assert orchestrator.main(isolated_run) == 0
    parallel_receipt = read_receipt(isolated_run)

    serial_args = PipelineArguments(
        target_dir=isolated_run.target_dir,
        output_dir=tmp_path / "output_serial",
    )
    serial_args.consolidated_steps = True
    _inject_steps(monkeypatch, serial_args)
    assert orchestrator.main(serial_args) == 0
    serial_receipt = read_receipt(serial_args)

    assert _aggregation(serial_receipt) == _aggregation(parallel_receipt)
    assert sorted(serial_receipt["steps"][0]) == sorted(parallel_receipt["steps"][0])
    assert [step["script_name"] for step in serial_receipt["steps"]] == [
        step["script_name"] for step in parallel_receipt["steps"]
    ]


def test_parallel_consolidated_does_not_start_uncontained_threads(
    isolated_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch
) -> None:
    _inject_steps(monkeypatch, isolated_run)
    lock = threading.Lock()
    executed: list[str] = []

    def fake_resolver(script_name: str) -> Any:
        stem = script_name.replace(".py", "")

        def step_fn(**kwargs: Any) -> bool:
            output_dir = Path(kwargs["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "marker.txt").write_text(stem)
            with lock:
                executed.append(stem)
            return True

        return step_fn

    monkeypatch.setattr(
        "gnn.pipeline.step_executor.resolve_step_function", fake_resolver
    )
    monkeypatch.setattr(
        "gnn.pipeline.step_executor.validate_step_prerequisites",
        lambda *args, **kwargs: {"passed": True, "warnings": []},
    )
    monkeypatch.setattr(
        orchestrator, "execute_pipeline_step", _subprocess_fake([], lock)
    )

    assert orchestrator.main(isolated_run) == 0
    receipt = read_receipt(isolated_run)

    # The real executor ran the (fake) module functions in-process.
    assert executed == []
    modes = {step["script_name"]: step["execution_mode"] for step in receipt["steps"]}
    assert modes["3_gnn.py"] == "subprocess"
    assert modes["9_advanced_viz.py"] == "subprocess"
    assert all(step["status"] == "SUCCESS" for step in receipt["steps"])
    for stem in ("0_template", "3_gnn", "5_type_checker"):
        assert not (isolated_run.output_dir / f"{stem}_output" / "marker.txt").is_file()


def test_parallel_receipts_measure_workers_before_ordered_future_collection(
    isolated_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed sibling keeps its own runtime while an earlier task waits."""
    _inject_steps(monkeypatch, isolated_run)
    finished_fast = threading.Event()
    observed: dict[str, tuple[float, float]] = {}
    fake = _subprocess_fake([], threading.Lock())

    def worker(script_name, *args, **kwargs):
        started = time.monotonic()
        if script_name == "3_gnn.py":
            assert finished_fast.wait(timeout=5)
            time.sleep(0.15)
        elif script_name == "5_type_checker.py":
            time.sleep(0.03)
            finished_fast.set()
        result = fake(script_name, *args, **kwargs)
        observed[script_name] = (started, time.monotonic())
        return result

    monkeypatch.setattr(orchestrator, "execute_pipeline_step", worker)
    assert orchestrator.main(isolated_run) == 0
    records = {
        step["script_name"]: step for step in read_receipt(isolated_run)["steps"]
    }
    for name in ("3_gnn.py", "5_type_checker.py"):
        actual_start, actual_end = observed[name]
        assert records[name]["duration_seconds"] >= actual_end - actual_start > 0
        recorded_wall_duration = (
            datetime.fromisoformat(records[name]["end_time"])
            - datetime.fromisoformat(records[name]["start_time"])
        ).total_seconds()
        assert abs(recorded_wall_duration - records[name]["duration_seconds"]) < 0.05
    assert datetime.fromisoformat(
        records["5_type_checker.py"]["end_time"]
    ) < datetime.fromisoformat(records["3_gnn.py"]["end_time"])
