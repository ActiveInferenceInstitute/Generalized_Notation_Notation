"""Contract tests for the validation workflow and shared structure helpers.

Pins the step-6 orchestrator contract (template kwargs handling, stage
receipts, accumulation across passes, score averaging), the uniform
best-effort error contract, exact cycle detection, and the shared
structure utilities the validators compose.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import gnn.validation as validation
from gnn.validation import StageServices, process_validation, validate_directory
from gnn.validation.performance_profiler import (
    extract_content_from_dict as profiler_extract,
)
from gnn.validation.semantic_validator import (
    extract_content_from_dict as semantic_extract,
)
from gnn.validation.structure import (
    DirectedEdge,
    clamp01,
    cycle_nodes,
    extract_content_from_dict,
)


def test_strict_probability_receipt_uses_authored_values_and_preserves_default(
    tmp_path,
):
    from gnn.parsers.markdown_parser import MarkdownGNNParser
    from gnn.validation import validate_content
    from gnn.validation.mcp import validate_gnn_file_mcp

    root = Path(__file__).resolve().parents[2]
    source = root / "input" / "gnn_files" / "discrete" / "two_state_bistable.md"
    content = source.read_text(encoding="utf-8")
    assert (
        validate_content(content, strict_probabilities=True)["probability_validation"][
            "status"
        ]
        == "passed"
    )
    content = content.replace("(0.8, 0.2),", "(-0.1, 0.2),", 1).replace(
        "D={(0.5, 0.5)}",
        "D={(0.0, 0.0)}",
    )
    assert validate_content(content, "strict")["valid"]
    assert "probability_validation" not in validate_content(content, "strict")
    source_copy = tmp_path / "invalid.md"
    source_copy.write_text(content, encoding="utf-8")
    result = validate_gnn_file_mcp(str(source_copy), strict_probabilities=True)
    assert result["is_valid"] is False
    probability = result["semantic"]["probability_validation"]
    assert probability["status"] == "failed"
    assert len(probability["diagnostics"]) == 2
    assert all(row["code"] == "GNN-E007" for row in probability["diagnostics"])

    parsed = MarkdownGNNParser().parse_string(content)
    assert parsed.success
    parsed_path = tmp_path / "model.json"
    parsed_path.write_text(json.dumps(parsed.model.to_dict()), encoding="utf-8")
    run = tmp_path / "run"
    _write_manifest(run, parsed_path)
    out = run / "6_validation_output"
    assert process_validation(tmp_path, out) is True
    command = [
        sys.executable,
        str(root / "src" / "gnn" / "6_validation.py"),
        "--target-dir",
        str(tmp_path),
        "--output-dir",
        str(run),
        "--strict-probabilities",
    ]
    checked = subprocess.run(
        command, cwd=root, capture_output=True, text=True, timeout=30
    )
    assert checked.returncode == 1, checked.stdout + checked.stderr
    receipt = json.loads((out / "validation_results.json").read_text(encoding="utf-8"))
    assert receipt["context"]["strict_probabilities"] is True
    assert receipt["current_summary"]["failed_validations"] == 1
    parsed_probability = receipt["files_validated"][0]["validations"]["semantic"][
        "probability_validation"
    ]
    assert parsed_probability["line_origin"] == "reconstructed_model"
    assert probability["line_origin"] == "source_text"
    assert [
        (row["code"], row["message"]) for row in parsed_probability["diagnostics"]
    ] == [(row["code"], row["message"]) for row in probability["diagnostics"]]
    assert source_copy.read_text(encoding="utf-8") == content


def test_strict_probability_structured_receipt_without_raw_sections():
    model = {
        "variables": [{"name": "s", "dimensions": [2]}],
        "parameters": [{"name": "D", "value": [0.0, 0.0]}],
    }
    result = validation.process_semantic_validation(model, strict_probabilities=True)
    probability = result["probability_validation"]
    assert probability["status"] == "failed"
    assert "prior sums to 0" in probability["diagnostics"][0]["message"]


def _write_manifest(base_output: Path, parsed_file: Path) -> None:
    """Write a minimal step-3 manifest pointing at one parsed model."""
    gnn_output = base_output / "3_gnn_output"
    gnn_output.mkdir(parents=True, exist_ok=True)
    (gnn_output / "gnn_processing_results.json").write_text(
        json.dumps(
            {
                "processed_files": [
                    {
                        "file_name": parsed_file.name,
                        "file_path": str(parsed_file),
                        "parse_success": True,
                        "parsed_model_file": str(parsed_file),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_parsed_model(path: Path) -> None:
    """Write a parsed model dictionary with canonical raw sections."""
    path.write_text(
        json.dumps(
            {
                "file_path": str(path),
                "raw_sections": {
                    "ModelName": "ContractModel",
                    "StateSpaceBlock": "s [3]\no [3]",
                    "InitialParameterization": "A",
                    "Connections": "s > o",
                },
            }
        ),
        encoding="utf-8",
    )


def _capture_semantic_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """Record the kwargs process_validation forwards to the semantic stage."""
    real_semantic = validation.process_semantic_validation
    captured: list[dict[str, Any]] = []

    def recorder(
        model_data: str | Path | dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        captured.append(dict(kwargs))
        return real_semantic(model_data, **kwargs)

    monkeypatch.setattr(validation, "process_semantic_validation", recorder)
    return captured


class TestProcessValidationKwargs:
    """The orchestrator entry point must honor its documented kwargs."""

    def test_validation_level_kwarg_is_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _capture_semantic_kwargs(monkeypatch)
        parsed = tmp_path / "model.json"
        _write_parsed_model(parsed)
        _write_manifest(tmp_path / "run", parsed)
        output = tmp_path / "run" / "6_validation_output"

        success = process_validation(
            tmp_path / "models", output, validation_level="strict"
        )

        assert success is True
        assert captured == [{"validation_level": "strict"}]

    def test_strict_flag_maps_to_strict_level(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _capture_semantic_kwargs(monkeypatch)
        parsed = tmp_path / "model.json"
        _write_parsed_model(parsed)
        _write_manifest(tmp_path / "run", parsed)
        output = tmp_path / "run" / "6_validation_output"

        success = process_validation(tmp_path / "models", output, strict=True)

        assert success is True
        assert captured == [{"validation_level": "strict"}]

    def test_default_level_remains_standard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _capture_semantic_kwargs(monkeypatch)
        parsed = tmp_path / "model.json"
        _write_parsed_model(parsed)
        _write_manifest(tmp_path / "run", parsed)
        output = tmp_path / "run" / "6_validation_output"

        assert process_validation(tmp_path / "models", output) is True
        assert captured == [{"validation_level": "standard"}]

    def test_pipeline_template_kwargs_are_tolerated(self, tmp_path: Path) -> None:
        """logger/recursive/profile arrive from the pipeline template."""
        parsed = tmp_path / "model.json"
        _write_parsed_model(parsed)
        _write_manifest(tmp_path / "run", parsed)
        output = tmp_path / "run" / "6_validation_output"

        success = process_validation(
            tmp_path / "models",
            output,
            logger=logging.getLogger("step6-test"),
            recursive=False,
            profile=True,
        )

        assert success is True
        receipt = json.loads(
            (output / "validation_results.json").read_text(encoding="utf-8")
        )
        assert receipt["files_validated"][0]["success"] is True


class TestStageFailureReceipts:
    """Stage exceptions and recovery mode must persist uniform receipts."""

    def test_performance_stage_exception_persists_recovery_receipt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_performance(*_args: object, **_kwargs: object) -> Any:
            raise RuntimeError("performance probe failed")

        monkeypatch.setattr(validation, "profile_performance", fail_performance)
        parsed = tmp_path / "model.json"
        _write_parsed_model(parsed)
        _write_manifest(tmp_path / "run", parsed)
        output = tmp_path / "run" / "6_validation_output"

        success = process_validation(tmp_path / "models", output)

        assert success is False
        receipt = json.loads(
            (output / "validation_results.json").read_text(encoding="utf-8")
        )
        file_result = receipt["files_validated"][0]
        assert file_result["success"] is False
        assert "performance probe failed" in file_result["errors"]
        assert file_result["validations"]["performance"] == {
            "status": "error",
            "error": "performance probe failed",
            "recovery": True,
        }
        assert receipt["summary"]["failed_validations"] == 1

    def test_consistency_recovery_mode_failure_is_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            validation,
            "check_consistency",
            lambda *_args, **_kwargs: {
                "status": "error",
                "error": "consistency boom",
                "recovery": True,
            },
        )
        parsed = tmp_path / "model.json"
        _write_parsed_model(parsed)
        _write_manifest(tmp_path / "run", parsed)
        output = tmp_path / "run" / "6_validation_output"

        success = process_validation(tmp_path / "models", output)

        assert success is False
        receipt = json.loads(
            (output / "validation_results.json").read_text(encoding="utf-8")
        )
        file_result = receipt["files_validated"][0]
        assert file_result["errors"] == ["consistency boom"]
        assert receipt["summary"]["failed_validations"] == 1


class TestAccumulationAndScores:
    """Repeated passes accumulate files, sources, and average scores."""

    def test_results_accumulate_across_passes(self, tmp_path: Path) -> None:
        output = tmp_path / "run" / "6_validation_output"

        first_parsed = tmp_path / "first.json"
        _write_parsed_model(first_parsed)
        _write_manifest(tmp_path / "run", first_parsed)
        assert process_validation(tmp_path / "models_a", output) is True

        second_parsed = tmp_path / "second.json"
        _write_parsed_model(second_parsed)
        _write_manifest(tmp_path / "run", second_parsed)
        assert process_validation(tmp_path / "models_b", output) is True

        receipt = json.loads(
            (output / "validation_results.json").read_text(encoding="utf-8")
        )
        summary = receipt["summary"]
        assert summary["total_files"] == 2
        assert summary["successful_validations"] == 2
        assert len(receipt["files_validated"]) == 2
        assert receipt["source_directories"] == [
            str(tmp_path / "models_a"),
            str(tmp_path / "models_b"),
        ]
        for score_key in (
            "avg_semantic_score",
            "avg_performance_score",
            "avg_consistency_score",
        ):
            assert 0.0 <= summary["validation_scores"][score_key] <= 1.0
        assert (output / "validation_summary.json").exists()

    def test_missing_manifest_returns_false_without_receipt(
        self, tmp_path: Path
    ) -> None:
        output = tmp_path / "6_validation_output"

        assert process_validation(tmp_path / "models", output) is False
        assert not (output / "validation_results.json").exists()


class TestInjectedStageServices:
    """validate_directory composes arbitrary stage callables."""

    def test_custom_stage_functions_drive_the_receipt(self, tmp_path: Path) -> None:
        calls: list[str] = []

        def make_stage(name: str) -> Any:
            def stage(model_data: object, **_kwargs: object) -> dict[str, Any]:
                calls.append(name)
                return {f"{name}_score": 0.5, "recovery": False}

            return stage

        parsed = tmp_path / "model.json"
        _write_parsed_model(parsed)
        _write_manifest(tmp_path / "run", parsed)
        output = tmp_path / "run" / "6_validation_output"

        success = validate_directory(
            tmp_path / "models",
            output,
            services=StageServices(
                semantic=make_stage("semantic"),
                performance=make_stage("performance"),
                consistency=make_stage("consistency"),
            ),
        )

        assert success is True
        assert calls == ["semantic", "performance", "consistency"]
        receipt = json.loads(
            (output / "validation_results.json").read_text(encoding="utf-8")
        )
        assert receipt["summary"]["validation_scores"]["avg_semantic_score"] == 0.5


class TestProfilerErrorContract:
    """profile_performance must honor the module's documented best-effort contract."""

    def test_success_result_declares_recovery_false(self) -> None:
        result = validation.profile_performance(
            {
                "file_path": "model.gnn",
                "raw_sections": {
                    "ModelName": "M",
                    "StateSpaceBlock": "s { Name: s Dimensions: 2 }",
                },
            }
        )
        assert result["recovery"] is False
        assert 0.0 <= result["performance_score"] <= 1.0
        assert result["file_name"] == "model.gnn"

    def test_error_result_is_uniform(self, tmp_path: Path) -> None:
        result = validation.profile_performance(tmp_path / "missing.md")
        assert result["status"] == "error"
        assert result["recovery"] is True
        assert result["performance_score"] == 0.0
        assert result["file_name"] == "missing.md"
        assert result["metrics"] == {}
        assert result["warnings"]


class TestExactCycleDetection:
    """Semantic validator reports exact cycle members, not cycle-reachers."""

    def test_cycle_warning_lists_only_cycle_members(self) -> None:
        content = """ModelName: CycleModel
StateSpaceBlock {
    Name: a
    Dimensions: 2
}
StateSpaceBlock {
    Name: b
    Dimensions: 2
}
StateSpaceBlock {
    Name: c
    Dimensions: 2
}
StateSpaceBlock {
    Name: d
    Dimensions: 2
}
Connection{
    From: a
    To: b
}
Connection{
    From: b
    To: c
}
Connection{
    From: c
    To: b
}
Connection{
    From: d
    To: a
}
"""
        result = validation.SemanticValidator(validation_level="strict").validate(
            content
        )
        cycle_warnings = [
            w
            for w in result["warnings"]
            if "Potential circular dependencies detected" in w
        ]
        assert cycle_warnings == [
            "Potential circular dependencies detected in blocks: b, c"
        ]

    def test_acyclic_models_produce_no_cycle_warning(self) -> None:
        content = """ModelName: DagModel
StateSpaceBlock {
    Name: a
    Dimensions: 2
}
StateSpaceBlock {
    Name: b
    Dimensions: 2
}
Connection{
    From: a
    To: b
}
"""
        result = validation.SemanticValidator().validate(content)
        assert not any("circular dependencies" in w for w in result["warnings"])


class TestStructureHelpers:
    """Shared structure utilities used by every validator."""

    def test_clamp01_bounds(self) -> None:
        assert clamp01(-0.5) == 0.0
        assert clamp01(0.42) == 0.42
        assert clamp01(1.5) == 1.0

    def test_cycle_nodes_covers_self_loops_and_components_only(self) -> None:
        nodes = ["a", "b", "c", "d"]
        edges = [
            DirectedEdge(source="a", target="a"),
            DirectedEdge(source="b", target="c"),
            DirectedEdge(source="c", target="b"),
        ]
        assert cycle_nodes(nodes, edges) == ["a", "b", "c"]

    def test_display_file_name_preserves_unknown_sentinel(self) -> None:
        from gnn.validation.structure import display_file_name

        assert display_file_name("dir/model.gnn") == "model.gnn"
        assert display_file_name("unknown") == "unknown"

    def test_content_extraction_is_shared_between_validators(self) -> None:
        """The deduplicated extractor is one implementation, not two."""
        assert profiler_extract is semantic_extract is extract_content_from_dict

    def test_extract_content_from_dict_renders_fallback(self) -> None:
        content = extract_content_from_dict(
            {
                "variables": [
                    {"name": "s", "var_type": "categorical", "dimensions": [3]}
                ],
                "connections": [{"source_variables": ["s"], "target_variables": ["o"]}],
            }
        )
        assert content == "StateSpaceBlock:\ns[3] # categorical\n\nConnections:\ns > o"


class TestMcpSemanticResults:
    """validate_gnn_file_mcp carries the deep semantic result additively."""

    def test_success_result_includes_semantic_key(self, tmp_path: Path) -> None:
        from gnn.validation.mcp import validate_gnn_file_mcp

        model = tmp_path / "model.gnn"
        model.write_text(
            "## ModelName\nDemo\n\n## StateSpaceBlock\ns [3]\n\n## Connections\ns > s\n",
            encoding="utf-8",
        )
        result = validate_gnn_file_mcp(str(model), "standard")

        assert result["success"] is True
        assert result["is_valid"] is True
        semantic = result["semantic"]
        assert semantic["recovery"] is False
        assert semantic["valid"] is True
        assert 0.0 <= semantic["semantic_score"] <= 1.0

    def test_missing_file_still_fails_fast(self, tmp_path: Path) -> None:
        from gnn.validation.mcp import validate_gnn_file_mcp

        result = validate_gnn_file_mcp(str(tmp_path / "nope.gnn"))
        assert result == {
            "success": False,
            "error": f"File not found: {tmp_path / 'nope.gnn'}",
        }
