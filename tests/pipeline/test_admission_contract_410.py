"""Real public execution admission and current-run identity acceptance."""

from __future__ import annotations

import json
import asyncio
import shutil
import subprocess
import sys
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gnn import run_pipeline
from gnn.api import path_utils, processor
from gnn.api.app import create_app as create_runs_app
from gnn.api.mcp import register_tools
from gnn.api.server import create_app
from gnn.mcp.mcp import MCP
from gnn.pipeline import execute_pipeline_step, execute_pipeline_steps
from gnn.utils.arguments.pipeline_arguments import PipelineArguments

pytestmark = pytest.mark.pipeline
ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "input/gnn_files/discrete/two_state_bistable.md"


@pytest.mark.parametrize("steps", [None, True, "3,5", 3])
def test_batch_facade_rejects_nonbatch_values_without_widening(
    tmp_path: Path, steps: object
) -> None:
    outcomes = execute_pipeline_steps(
        steps, {"target_dir": tmp_path, "output_dir": tmp_path / "output"}
    )
    assert outcomes and all(
        not outcome.success and outcome.error for outcome in outcomes
    )
    assert not (tmp_path / "output").exists()


def test_empty_convenience_batch_returns_no_work(tmp_path: Path) -> None:
    assert (
        execute_pipeline_steps(
            [], {"target_dir": tmp_path, "output_dir": tmp_path / "output"}
        )
        == []
    )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"only_steps": []},
        {"only_steps": "3,99"},
        {"only_steps": True},
        {"only_steps": 3.0},
        {"skip_steps": "3,3"},
        {"skip_steps": "99"},
        {"verbose": "false"},
        {"strict": 1},
        {"parallel": "false"},
        {"consolidated_steps": 1},
    ],
)
def test_direct_main_override_admits_control_before_output_ownership(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    options: dict,
) -> None:
    from gnn.main import main

    output = tmp_path / "output"
    args = PipelineArguments(target_dir=tmp_path, output_dir=output, **options)
    assert main(override_args=args, override_config={}) == 1
    assert "Pipeline admission failure" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("selection", ["", "3,99", "3,3"])
def test_direct_module_main_rejects_selection_before_output_creation(
    tmp_path: Path,
    selection: str,
) -> None:
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gnn.main",
            "--target-dir",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--only-steps",
            selection,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "Pipeline admission failure" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("step", ["3,99", "3,5", "3,3", "", True, 3.0])
def test_single_step_facade_never_reduces_invalid_or_multiple_requests(
    tmp_path: Path, step: object
) -> None:
    result = execute_pipeline_step(
        step, {}, {"target_dir": tmp_path, "output_dir": tmp_path / "output"}
    )
    assert result.success is False and result.error
    assert not (tmp_path / "output").exists()


def test_single_step_script_override_cannot_replace_the_requested_step(
    tmp_path: Path,
) -> None:
    result = execute_pipeline_step(
        "3",
        {"script_path": ROOT / "src/gnn/5_type_checker.py"},
        {"target_dir": tmp_path, "output_dir": tmp_path / "output"},
    )
    assert result.success is False and "match" in result.error
    assert not (tmp_path / "output").exists()


def test_default_api_command_consumes_workspace_configuration_in_a_checkout(
    tmp_path: Path,
) -> None:
    from gnn.api.pipeline_runner import build_pipeline_command

    target = _source(tmp_path)
    shadow = tmp_path / "gnn"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("")
    (shadow / "main.py").write_text(
        "from pathlib import Path; Path('workspace_shadow_executed').write_text('bad')"
    )
    (target / "config.yaml").write_text(
        "pipeline:\n  skip_steps: [5]\n", encoding="utf-8"
    )
    output = tmp_path / "output"
    completed = subprocess.run(
        build_pipeline_command(
            target, output, only_steps=[3, 5], consolidated_steps=True
        ),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert completed.returncode in (0, 2), completed.stdout + completed.stderr
    assert not (tmp_path / "workspace_shadow_executed").exists()
    summary = _summary(output)
    assert [step["script_name"] for step in summary["steps"]] == ["3_gnn.py"]
    context = json.loads((output / "00_pipeline_summary/run_context.json").read_text())
    assert json.loads(context["config_json"])["pipeline"]["skip_steps"] == [5]


def test_public_step_results_retain_enclosing_evidence_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.pipeline import finalization

    target = _source(tmp_path)

    def fail_durable_evidence(*args: object, **kwargs: object) -> None:
        raise OSError("Durable evidence could not be committed")

    monkeypatch.setattr(finalization, "finalize_run_evidence", fail_durable_evidence)
    output = tmp_path / "output"
    outcomes = execute_pipeline_steps(
        ["3"], {"target_dir": target, "output_dir": output, "consolidated_steps": True}
    )
    assert len(outcomes) == 1
    assert outcomes[0].status == "SUCCESS"
    assert outcomes[0].success is False
    assert "Durable evidence" in outcomes[0].error
    assert outcomes[0].run_id == _summary(output)["run_id"]


@pytest.mark.parametrize(
    "steps", [[], [True], [3.0], ["3", "unknown"], [3, 99], [3, 3]]
)
def test_python_execution_rejects_incomplete_or_ambiguous_selection(
    tmp_path: Path, steps: object
) -> None:
    output = tmp_path / "output"
    result = run_pipeline(target_dir=tmp_path, output_dir=output, steps=steps)
    assert result["success"] is False
    assert result["steps_executed"] == []
    assert result["errors"]
    assert not output.exists()


@pytest.mark.parametrize(
    "option", [{"strict": "false"}, {"parallel": 1}, {"unknown": True}]
)
def test_python_flags_and_unknown_options_are_not_silently_ignored(
    tmp_path: Path, option: dict
) -> None:
    result = run_pipeline(
        option, target_dir=tmp_path, output_dir=tmp_path / "output", steps=[3]
    )
    assert result["success"] is False
    assert result["errors"]
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("selection", [[], [True], [3.0], ["3"], [3, 99], [3, 3]])
def test_rest_and_mcp_reject_the_same_invalid_json_selection(selection: object) -> None:
    before = set(processor._JOBS)
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/process", json={"target_dir": "input/gnn_files", "steps": selection}
    )
    assert response.status_code == 422
    assert response.json()["status"] == "error"
    registry = MCP(
        enable_caching=False, enable_rate_limiting=False, strict_validation=False
    )
    register_tools(registry)
    result = registry.execute_tool(
        "gnn_submit_job", {"target_dir": "input/gnn_files", "steps": selection}
    )
    assert result["status"] == "error"
    assert set(processor._JOBS) == before


@pytest.mark.parametrize(
    "flags", [["--only-steps"], ["--only-steps", "99"], ["--only-steps", "3", "3"]]
)
def test_actual_cli_rejects_empty_unknown_and_duplicate_steps(
    tmp_path: Path, flags: list[str]
) -> None:
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gnn.cli",
            "run",
            "--target-dir",
            str(tmp_path),
            "--output-dir",
            str(output),
            *flags,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert not output.exists()


def _source(root: Path) -> Path:
    target = root / "input"
    target.mkdir()
    shutil.copy2(MODEL, target / "model.md")
    return target


def test_fresh_second_invocation_cannot_replace_first_facade_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gnn import main as main_module

    monkeypatch.delenv("GNN_RUN_ID", raising=False)
    target = _source(tmp_path)
    second = tmp_path / "second"
    second.mkdir()
    shutil.copy2(MODEL, second / "second.md")
    output = tmp_path / "output"
    native_main = main_module.main
    actual_ids = []

    def two_owned_invocations(**kwargs: object) -> int:
        first_id = os.environ["GNN_RUN_ID"]
        code = native_main(**kwargs)
        actual_ids.append(_summary(output)["run_id"])
        # Represent an independent caller rather than inheriting the first ID.
        os.environ.pop("GNN_RUN_ID")
        try:
            second_code = native_main(
                override_args=PipelineArguments(
                    target_dir=second,
                    output_dir=output,
                    only_steps="3",
                    consolidated_steps=True,
                ),
                override_config={"testing_matrix": {"enabled": False}},
            )
            assert second_code in (0, 2)
            actual_ids.append(_summary(output)["run_id"])
        finally:
            os.environ["GNN_RUN_ID"] = first_id
        return code

    monkeypatch.setattr(main_module, "main", two_owned_invocations)
    result = run_pipeline(
        target_dir=target, output_dir=output, steps=[3], consolidated_steps=True
    )
    assert len(set(actual_ids)) == 2
    assert result["success"] is False
    assert result["steps_executed"] == []
    assert "current invocation" in " ".join(result["errors"])
    assert "run_id" not in result and "model_selection" not in result
    assert "GNN_RUN_ID" not in os.environ


@pytest.mark.parametrize(
    "containment,verified",
    [("direct_worker_only", True), ("process_group_only", False)],
)
def test_http_status_preserves_supervisor_receipts_from_both_stores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    containment: str,
    verified: bool,
) -> None:
    target = _source(tmp_path)
    monkeypatch.setattr(path_utils, "get_repo_root", lambda: tmp_path)
    job_id = processor.create_job(str(target), "output", steps=[3])
    receipt = {
        "containment": containment,
        "cleanup_verified": verified,
        "observed_descendants": [123],
        "errors": [] if verified else ["Unverified descendant cleanup"],
    }
    processor._JOBS[job_id].update(
        status="completed" if verified else "failed", process_cleanup=receipt
    )
    try:
        response = TestClient(create_app()).get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["data"]["process_cleanup"] == receipt
        run_store = {
            "receipt-run": {
                "status": "completed" if verified else "failed",
                "process_cleanup": receipt,
            }
        }
        response = TestClient(create_runs_app(runs_store=run_store)).get(
            "/api/v1/runs/receipt-run"
        )
        assert response.status_code == 200
        assert response.json()["data"]["process_cleanup"] == receipt
    finally:
        processor._JOBS.pop(job_id, None)


def test_workspace_failure_at_job_launch_reaches_a_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gnn.api.path_utils import PathValidationError

    target = _source(tmp_path)
    monkeypatch.setattr(path_utils, "get_repo_root", lambda: tmp_path)
    job_id = processor.create_job(str(target), "output", steps=[3])

    def unavailable_workspace() -> Path:
        raise PathValidationError("Configured workspace is unavailable")

    monkeypatch.setattr(path_utils, "get_repo_root", unavailable_workspace)
    try:
        asyncio.run(processor.execute_job_async(job_id))
        job = processor.get_job(job_id)
        assert job["status"] == "failed"
        assert job["completed_at"] is not None
        assert "unavailable" in job["error_message"]
        assert processor._JOBS[job_id]["process"] is None
        assert not list((tmp_path / "output").rglob("*.py"))
    finally:
        processor._JOBS.pop(job_id, None)


def test_serial_parallel_and_public_multi_step_share_real_selection_and_outcomes(
    tmp_path: Path,
) -> None:
    target = _source(tmp_path)
    records = []
    for name, parallel in (("serial", False), ("parallel", True)):
        result = run_pipeline(
            {"parallel": not parallel, "verbose": True},
            target_dir=target,
            output_dir=tmp_path / name,
            steps=[3, 5],
            parallel=parallel,
            verbose=False,
            consolidated_steps=True,
        )
        assert result["success"], result["errors"]
        summary = json.loads(Path(result["summary_file"]).read_text())
        assert summary["arguments"]["parallel"] is parallel
        assert summary["arguments"]["verbose"] is False
        assert result["model_selection"] == summary["model_selection"]
        assert len(result["model_selection"]) == 1
        assert all(step["run_id"] == result["run_id"] for step in summary["steps"])
        assert {step["script_name"] for step in summary["steps"]} == {
            "3_gnn.py",
            "5_type_checker.py",
        }
        assert all(
            step["selected_model_ids"] == [result["model_selection"][0]["model_id"]]
            for step in summary["steps"]
        )
        records.append((result, summary))
    assert records[0][0]["run_id"] != records[1][0]["run_id"]
    assert records[0][0]["model_selection"] == records[1][0]["model_selection"]
    assert [
        (step["script_name"], step["status"]) for step in records[0][1]["steps"]
    ] == [(step["script_name"], step["status"]) for step in records[1][1]["steps"]]
    outcomes = execute_pipeline_steps(
        ["3", "5_type_checker"],
        {
            "target_dir": target,
            "output_dir": tmp_path / "multi",
            "consolidated_steps": True,
        },
    )
    assert len(outcomes) == 2
    assert all(outcome.success for outcome in outcomes)
    assert len({outcome.run_id for outcome in outcomes}) == 1
    assert outcomes[0].run_id is not None
    assert all(outcome.status.startswith("SUCCESS") for outcome in outcomes)
    assert all(outcome.artifacts for outcome in outcomes)


def test_python_failed_admission_never_ingests_an_inherited_summary(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    summary = output / "00_pipeline_summary/pipeline_execution_summary.json"
    summary.parent.mkdir(parents=True)
    original = (
        '{"run_id":"previous","steps":[{"script_name":"3_gnn.py","status":"SUCCESS"}]}'
    )
    summary.write_text(original)
    result = run_pipeline(target_dir=tmp_path, output_dir=output, steps=[3, "unknown"])
    assert result["success"] is False
    assert result["steps_executed"] == []
    assert "run_id" not in result
    assert summary.read_text() == original


def _summary(output: Path) -> dict:
    return json.loads(
        (output / "00_pipeline_summary/pipeline_execution_summary.json").read_text()
    )


def test_actual_rest_and_mcp_execution_use_one_owned_core_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _source(tmp_path)
    # Configure only the existing workspace boundary; real dispatch and workers run.
    monkeypatch.setattr(path_utils, "get_repo_root", lambda: tmp_path)
    reference = run_pipeline(
        target_dir=target,
        output_dir=tmp_path / "python",
        steps=[5],
        consolidated_steps=True,
    )
    assert reference["success"], reference
    summaries = [_summary(tmp_path / "python")]
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "gnn.cli",
            "run",
            "--target-dir",
            str(target),
            "--output-dir",
            str(tmp_path / "cli"),
            "--only-steps",
            "5",
            "--parallel",
            "--consolidated-steps",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert cli.returncode in (0, 2), cli.stderr
    summaries.append(_summary(tmp_path / "cli"))
    jobs_client = TestClient(create_app())
    posted = jobs_client.post(
        "/api/v1/process",
        json={
            "target_dir": "input",
            "output_dir": "jobs",
            "steps": [5],
            "skip_steps": [13, 24],
            "parallel": True,
            "consolidated_steps": True,
        },
    )
    assert posted.status_code == 200, posted.json()
    job_id = posted.json()["data"]["job_id"]
    assert (
        jobs_client.get(f"/api/v1/jobs/{job_id}").json()["data"]["status"]
        == "completed"
    )
    assert processor.get_job(job_id)["steps_completed"] == [3, 5]
    summaries.append(_summary(tmp_path / "jobs"))

    runs_client = TestClient(create_runs_app())
    posted = runs_client.post(
        "/api/v1/run",
        json={
            "target_dir": "input",
            "output_dir": "runs",
            "steps": [5],
            "skip_llm": True,
            "skip_steps": [24],
            "parallel": True,
            "consolidated_steps": True,
        },
    )
    assert posted.status_code == 200, posted.json()
    run_hash = posted.json()["data"]["run_hash"]
    status = runs_client.get(f"/api/v1/runs/{run_hash}").json()["data"]
    assert status["status"] == "completed", status
    assert status["total_steps"] == status["steps_completed"] == 2
    summaries.append(_summary(tmp_path / "runs"))

    registry = MCP(
        enable_caching=False, enable_rate_limiting=False, strict_validation=True
    )
    register_tools(registry)
    submitted = registry.execute_tool(
        "gnn_submit_job",
        {
            "target_dir": "input",
            "output_dir": "mcp",
            "steps": [5],
            "parallel": True,
            "consolidated_steps": True,
        },
    )
    assert submitted["status"] == "success", submitted
    assert processor.get_job(submitted["job_id"])["status"] == "pending"
    asyncio.run(processor.execute_job_async(submitted["job_id"]))
    assert (
        registry.execute_tool("gnn_get_job_status", {"job_id": submitted["job_id"]})[
            "job"
        ]["status"]
        == "completed"
    )
    summaries.append(_summary(tmp_path / "mcp"))
    expected = [(step["script_name"], step["status"]) for step in summaries[0]["steps"]]
    assert len({summary["run_id"] for summary in summaries}) == 5
    for summary in summaries:
        assert summary["model_selection"] == summaries[0]["model_selection"]
        assert [
            (step["script_name"], step["status"]) for step in summary["steps"]
        ] == expected
        assert all(
            step["run_id"] == summary["run_id"] and step["artifacts"]
            for step in summary["steps"]
        )


def test_matrix_executor_keeps_duplicate_sources_and_explicit_empty_models(
    tmp_path: Path,
) -> None:
    from gnn.main import main

    target = tmp_path / "input"
    for folder in ("first", "second"):
        (target / folder).mkdir(parents=True)
        shutil.copy2(MODEL, target / folder / "model.md")
    ordinary = run_pipeline(
        target_dir=target,
        output_dir=tmp_path / "ordinary",
        steps=[3, 5],
        consolidated_steps=True,
    )
    assert ordinary["success"], ordinary
    config = {
        "pipeline": {"skip_steps": []},
        "testing_matrix": {
            "enabled": True,
            "default_steps": [],
            "folders": {"first": [3, 5], "second": [3, 5]},
        },
    }
    matrix = tmp_path / "matrix"
    assert main(
        override_args=PipelineArguments(
            target_dir=target,
            output_dir=matrix,
            only_steps="3,5",
            consolidated_steps=True,
            parallel=True,
        ),
        override_config=config,
    ) in (0, 2)
    summary = _summary(matrix)
    assert summary["model_selection"] == ordinary["model_selection"]
    selected = [model["model_id"] for model in summary["model_selection"]]
    assert len(set(selected)) == 2
    assert len({model["artifact_stem"] for model in summary["model_selection"]}) == 2
    assert all(step["selected_model_ids"] == selected for step in summary["steps"])
    assert [(s["script_name"], s["status"]) for s in summary["steps"]] == [
        (s["step_name"], s["status"]) for s in ordinary["steps_executed"]
    ]
    assert all(len(step["artifacts"]) >= 2 for step in summary["steps"])

    # A present empty MODEL selection must not widen to the root directory.
    empty_output = tmp_path / "empty"
    config["testing_matrix"]["folders"] = {"first": [], "second": []}
    assert main(
        override_args=PipelineArguments(
            target_dir=target,
            output_dir=empty_output,
            only_steps="3,5",
            consolidated_steps=True,
        ),
        override_config=config,
    ) in (0, 2)
    empty = _summary(empty_output)
    assert empty["model_selection"] == []
    assert all(
        step["status"] == "SKIPPED"
        and step["selected_model_ids"] == []
        and step["artifacts"] == []
        for step in empty["steps"]
    )
    assert not (empty_output / "3_gnn_output").exists()


@pytest.mark.parametrize("output", ["input", "."])
def test_api_output_cannot_contain_and_clobber_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    target = _source(tmp_path)
    before = (target / "model.md").read_bytes()
    monkeypatch.setattr(path_utils, "get_repo_root", lambda: tmp_path)
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/process",
        json={"target_dir": "input", "output_dir": output, "steps": [3]},
    )
    assert response.status_code == 400
    assert "must not equal or contain" in str(response.json())
    assert (target / "model.md").read_bytes() == before
