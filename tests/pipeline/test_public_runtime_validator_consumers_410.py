"""The documented runtime validator consumes an actual refused CLI invocation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnn.pipeline.pipeline_runtime_validator import PipelineValidator

pytestmark = pytest.mark.pipeline


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
