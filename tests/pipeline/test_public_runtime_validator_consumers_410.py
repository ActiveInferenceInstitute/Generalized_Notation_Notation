"""The documented runtime validator consumes an actual refused CLI invocation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from gnn.pipeline import pipeline_runtime_validator as runtime_validator
from gnn.pipeline.pipeline_runtime_validator import PipelineValidator

pytestmark = pytest.mark.pipeline


def _package_run_receipts() -> dict[str, str | None]:
    """Observe authoritative checkout outputs, without creating package files."""
    package_parent = Path(runtime_validator.__file__).resolve().parents[3]
    paths = [
        package_parent / "output/.gnn_run.lock",
        package_parent / "output/00_pipeline_summary/pipeline_execution_summary.json",
        package_parent / "output/00_pipeline_summary/run_context.json",
    ]
    step_output = package_parent / "output/3_gnn_output"
    paths += sorted(path for path in step_output.rglob("*") if path.is_file())
    receipts = {}
    for path in paths:
        if path.is_file():
            with path.open("rb") as handle:
                receipts[str(path)] = hashlib.file_digest(handle, "sha256").hexdigest()
        else:
            receipts[str(path)] = None
    return receipts


def test_invalid_native_selection_cannot_validate_improvements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    validator = PipelineValidator(verbose=False)
    result = validator.test_pipeline_execution(["999"])
    assert result["execution_successful"] is False, result
    assert "Pipeline admission failure:" in result["stderr"], result
    assert result["step_results"] == {}
    assert not (tmp_path / "output").exists()
    validation = validator.validate_success_status(result)
    assert validation == {
        "success_rate_improved": False,
        "warnings_reduced": False,
        "no_critical_failures": False,
        "step_analysis": {},
    }
    report_path = validator.save_validation_report(
        {
            "improvements_validated": False,
            "pipeline_execution": result,
            "success_status_validation": validation,
        }
    )
    saved = json.loads(report_path.read_text())
    assert report_path == Path("output/pipeline_runtime_validator_report.json")
    assert saved["improvements_validated"] is False
    assert saved["pipeline_execution"]["stderr"] == result["stderr"]
    assert saved["success_status_validation"]["no_critical_failures"] is False


def test_validator_uses_one_caller_model_and_preserves_parent_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = Path(__file__).resolve().parents[2]
    model = project / "input/gnn_files/discrete/two_state_bistable.md"
    source = tmp_path / "input/gnn_files/caller_model.md"
    source.parent.mkdir(parents=True)
    content = model.read_bytes().replace(
        b"Two State Bistable POMDP", b"Caller Workspace POMDP"
    )
    source.write_bytes(content)
    # A workspace package must not shadow the trusted importable GNN package.
    shadow = tmp_path / "gnn"
    shadow.mkdir()
    (shadow / "__init__.py").write_text(
        "from pathlib import Path\n"
        "(Path(__file__).parent / 'executed.txt').write_text('shadowed')\n"
        "raise RuntimeError('workspace package executed')\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GNN_RUN_ID", "parent-owned-run")
    monkeypatch.setenv("GNN_RUN_CONTEXT_FILE", str(tmp_path / "parent-context.json"))
    before = _package_run_receipts()
    validator = PipelineValidator(verbose=False)
    result = validator.test_pipeline_execution(["3"])
    assert result["execution_successful"] is True, result
    assert set(result["step_results"]) == {"3_gnn.py"}, result
    assert result["step_results"]["3_gnn.py"]["status"] == "SUCCESS"
    summary_path = (
        tmp_path / "output/00_pipeline_summary/pipeline_execution_summary.json"
    )
    summary = json.loads(summary_path.read_text())
    context = json.loads((summary_path.parent / "run_context.json").read_text())
    assert summary["run_id"] == context["run_id"] != "parent-owned-run"
    assert summary["planned_steps"] == ["3_gnn.py"]
    (selected,) = summary["model_selection"]
    assert selected["source_path"] == str(source.resolve())
    assert selected["sha256"] == hashlib.sha256(content).hexdigest()
    assert context["output_root"] == str((tmp_path / "output").resolve())
    assert summary["steps"][0]["artifacts"]
    for artifact in summary["steps"][0]["artifacts"]:
        path = tmp_path / "output" / artifact["path"]
        assert path.is_file()
        with path.open("rb") as handle:
            assert (
                hashlib.file_digest(handle, "sha256").hexdigest() == artifact["sha256"]
            )
    assert not (shadow / "executed.txt").exists()
    assert _package_run_receipts() == before
    assert os.environ["GNN_RUN_ID"] == "parent-owned-run"
    assert os.environ["GNN_RUN_CONTEXT_FILE"] == str(tmp_path / "parent-context.json")
    saved = validator.save_validation_report({"pipeline_execution": result})
    assert (
        json.loads(saved.read_text())["pipeline_execution"]["step_results"]
        == result["step_results"]
    )


def test_empty_caller_workspace_cannot_consume_checkout_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    before = _package_run_receipts()
    result = PipelineValidator(verbose=False).test_pipeline_execution(["3"])
    assert result["execution_successful"] is False, result
    assert result["step_results"] == {}, result
    summary = json.loads(
        (
            tmp_path / "output/00_pipeline_summary/pipeline_execution_summary.json"
        ).read_text()
    )
    assert summary["overall_status"] == "FAILED"
    assert "Pipeline input must be a directory:" in summary["error"]
    assert str((tmp_path / "input/gnn_files").resolve()) in summary["error"]
    assert not (tmp_path / "output/3_gnn_output").exists()
    assert _package_run_receipts() == before


@pytest.mark.parametrize(
    "location",
    [
        "00_pipeline_summary/pipeline_execution_summary.json",
        "pipeline_execution_summary.json",
    ],
)
def test_prior_summary_is_not_admitted_after_real_selector_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, location: str
) -> None:
    monkeypatch.chdir(tmp_path)
    prior_summary = tmp_path / "output" / location
    prior_summary.parent.mkdir(parents=True)
    previous = {
        "run_id": "previous-validator-run",
        "steps": [{"script_name": "3_gnn.py", "status": "SUCCESS", "exit_code": 0}],
    }
    prior_summary.write_text(json.dumps(previous))
    original = prior_summary.read_bytes()
    # Even an inherited identity matching the prior file cannot admit it as new.
    monkeypatch.setenv("GNN_RUN_ID", previous["run_id"])
    before = _package_run_receipts()
    validator = PipelineValidator(verbose=False)
    result = validator.test_pipeline_execution(["999"])
    assert result["execution_successful"] is False, result
    assert "Pipeline admission failure:" in result["stderr"], result
    assert result["step_results"] == {}, result
    assert result["error"] in {
        "Pipeline summary does not belong to the current invocation",
        "No current-invocation pipeline summary is available",
    }, result
    assert prior_summary.read_bytes() == original
    assert _package_run_receipts() == before
    assert os.environ["GNN_RUN_ID"] == previous["run_id"]
    validation = validator.validate_success_status(result)
    assert validation["step_analysis"] == {}
    assert validation["no_critical_failures"] is False
