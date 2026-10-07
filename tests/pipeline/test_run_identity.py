"""Run identity through real orchestration and persistence, without pipeline recursion."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, NoReturn

import pytest

import gnn.main as orchestrator
from gnn.api.pipeline_runner import PIPELINE_SUMMARY, read_pipeline_summary
from gnn.utils.arguments.pipeline_arguments import PipelineArguments


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
    logger = logging.getLogger(__name__)

    def context(
        *_args: object,
    ) -> tuple[
        PipelineArguments,
        dict[Any, Any],
        list[Any],
        dict[str, Any],
        RecordingVisualLogger,
        str,
        logging.Logger,
    ]:
        steps: list[Any] = [("0_template.py", "Identity probe")]
        summary = orchestrator._initialize_pipeline_summary(args, steps, {})
        logger_double = RecordingVisualLogger()
        return args, {}, steps, summary, logger_double, "correlation", logger

    monkeypatch.setattr(orchestrator, "_prepare_pipeline_context", context)
    monkeypatch.setattr(orchestrator, "_start_pipeline_run", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_print_pipeline_completion", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_write_performance_dashboard", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_write_final_pipeline_report", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_read_input_config", lambda *a: {})
    monkeypatch.setattr(
        "gnn.utils.arguments.arg_parsing.build_step_command_args",
        lambda *a: [sys.executable, "-c", "import os; print(os.environ['GNN_RUN_ID'])"],
    )
    monkeypatch.delenv("GNN_RUN_ID", raising=False)
    return args


def read_receipt(args: PipelineArguments) -> dict[str, Any]:
    receipt: dict[str, Any] = json.loads(
        (args.output_dir / PIPELINE_SUMMARY).read_text()
    )
    return receipt


def test_new_runs_have_distinct_ids_stable_hash_and_child_identity(
    isolated_run: PipelineArguments,
) -> None:
    receipts = []
    for _ in range(2):
        assert orchestrator.main(isolated_run) == 0
        assert "GNN_RUN_ID" not in os.environ
        receipt = read_receipt(isolated_run)
        assert receipt["steps"][0]["stdout"].strip() == receipt["run_id"]
        assert (
            read_pipeline_summary(
                isolated_run.output_dir, expected_run_id=receipt["run_id"]
            )
            == receipt["steps"]
        )
        receipts.append(receipt)
    assert receipts[0]["run_id"] != receipts[1]["run_id"]
    assert receipts[0]["run_hash"] == receipts[1]["run_hash"]


def test_incoming_api_id_is_preserved(
    isolated_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GNN_RUN_ID", "api-request-123")
    assert orchestrator.main(isolated_run) == 0
    receipt = read_receipt(isolated_run)
    assert receipt["run_id"] == "api-request-123"
    assert receipt["steps"][0]["stdout"].strip() == "api-request-123"
    assert os.environ["GNN_RUN_ID"] == "api-request-123"
    assert (
        read_pipeline_summary(
            isolated_run.output_dir, expected_run_id="api-request-123"
        )
        == receipt["steps"]
    )


def test_explicit_child_identity_overrides_ambient(
    isolated_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GNN_RUN_ID", "unrelated")
    result = orchestrator.execute_pipeline_step(
        "0_template.py", isolated_run, logging.getLogger(__name__), run_id="selected"
    )
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "selected"
    assert os.environ["GNN_RUN_ID"] == "unrelated"


@pytest.mark.parametrize("minimal", [False, True])
def test_failure_summaries_preserve_identity(
    isolated_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch, minimal: bool
) -> None:
    monkeypatch.setenv("GNN_RUN_ID", "failed-api-request")

    def fail(*args: object) -> NoReturn:
        raise RuntimeError("intentional failure")

    monkeypatch.setattr(
        orchestrator,
        "validate_pipeline_summary" if minimal else "_start_pipeline_run",
        fail,
    )
    result = orchestrator.main(isolated_run)
    assert result == 1
    receipt = read_receipt(isolated_run)
    assert receipt["overall_status"] == "FAILED"
    assert receipt["run_id"] == "failed-api-request"
    assert receipt["run_hash"]
    assert (
        read_pipeline_summary(
            isolated_run.output_dir, expected_run_id="failed-api-request"
        )
        == receipt["steps"]
    )
    assert os.environ["GNN_RUN_ID"] == "failed-api-request"


def test_startup_failure_persists_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = PipelineArguments(target_dir=tmp_path, output_dir=tmp_path / "output")
    monkeypatch.setattr(orchestrator, "_build_main_args", lambda *a: (args, None))

    def fail(*args: object) -> NoReturn:
        raise RuntimeError("startup failure")

    monkeypatch.setattr(orchestrator, "_create_pipeline_visual_logger", fail)
    monkeypatch.delenv("GNN_RUN_ID", raising=False)
    assert orchestrator.main(args) == 1
    receipt = read_receipt(args)
    assert receipt["run_id"]
    assert receipt["overall_status"] == "FAILED"
    assert (
        read_pipeline_summary(args.output_dir, expected_run_id=receipt["run_id"]) == []
    )
    assert "GNN_RUN_ID" not in os.environ


def test_early_return_restores_environment(
    isolated_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated_run.autonomous = True
    monkeypatch.setattr(
        "gnn.pipeline.autonomous.run_autonomous_proposal_loop",
        lambda *a: {"candidate_count": 0},
    )
    assert orchestrator.main(isolated_run) == 0
    assert "GNN_RUN_ID" not in os.environ


def test_parallel_children_receive_same_run_identity(
    isolated_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated_run.parallel = True
    prepare = orchestrator._prepare_pipeline_context

    def context(
        override_args: PipelineArguments | None,
        override_config: dict[str, Any] | None,
    ) -> tuple[object, ...]:
        values = list(prepare(override_args, override_config))
        steps = values[2]
        assert isinstance(steps, list)
        steps.append(("1_setup.py", "Second identity probe"))
        return tuple(values)

    monkeypatch.setattr(orchestrator, "_prepare_pipeline_context", context)
    monkeypatch.setattr(
        "gnn.utils.pipeline_orchestration.pipeline_step_dependencies.PIPELINE_STEP_DEPENDENCIES",
        {0: [], 1: []},
    )
    assert orchestrator.main(isolated_run) == 0
    receipt = read_receipt(isolated_run)
    assert len(receipt["steps"]) == 2
    assert {step["stdout"].strip() for step in receipt["steps"]} == {receipt["run_id"]}
    assert "GNN_RUN_ID" not in os.environ


def test_interrupt_restores_environment(
    isolated_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(*args: object) -> NoReturn:
        assert os.environ["GNN_RUN_ID"]
        raise KeyboardInterrupt

    monkeypatch.setattr(orchestrator, "_start_pipeline_run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        orchestrator.main(isolated_run)
    assert "GNN_RUN_ID" not in os.environ


def test_overlapping_top_level_calls_serialize_environment(
    isolated_run: PipelineArguments,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    entered = Event()
    release = Event()
    second_started = Event()
    second_entered = Event()
    ids: list[str] = []
    monkeypatch.delenv("GNN_RUN_ID", raising=False)

    def run(*args: object) -> int:
        ids.append(os.environ["GNN_RUN_ID"])
        if len(ids) == 1:
            entered.set()
            assert release.wait(5)
        else:
            second_entered.set()
        return 0

    def second() -> int:
        second_started.set()
        return orchestrator.main(isolated_run)

    monkeypatch.setattr(orchestrator, "_run_pipeline", run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(orchestrator.main, isolated_run)
        try:
            assert entered.wait(5)
            other = pool.submit(second)
            assert second_started.wait(5)
            assert not second_entered.wait(0.1)
        finally:
            release.set()
        assert first.result() == other.result() == 0
    assert len(set(ids)) == 2
    assert "GNN_RUN_ID" not in os.environ
