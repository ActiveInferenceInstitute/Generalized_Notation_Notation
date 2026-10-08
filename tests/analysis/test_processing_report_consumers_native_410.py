"""Native processing/report consumers: real files and externally supplied snapshots."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from gnn.processing import (
    GNNProcessor,
    ProcessingContext,
    ProcessingPhase,
    check_gnn_file_structure,
)
from gnn.report import ReportGenerator
from gnn.types import ValidationResult

REPO = Path(__file__).resolve().parents[2]


def _native_context(tmp_path: Path, **options) -> ProcessingContext:
    target = tmp_path / "models"
    target.mkdir()
    shutil.copyfile(
        REPO / "input/gnn_files/basics/static_perception.md", target / "perception.md"
    )
    return ProcessingContext(
        target_dir=target, output_dir=tmp_path / "reports", **options
    )


def _read_reports(report: dict) -> tuple[dict, str, str]:
    files = report["report_files"]
    assert set(files) == {"json", "markdown", "html"}
    assert all(
        Path(path).is_file() and Path(path).stat().st_size > 0
        for path in files.values()
    )
    return (
        json.loads(Path(files["json"]).read_text()),
        Path(files["markdown"]).read_text(),
        Path(files["html"]).read_text(),
    )


def test_directory_processor_writes_all_required_native_reports(tmp_path: Path) -> None:
    context = _native_context(tmp_path)
    assert GNNProcessor().process(context) is True
    assert "report" in context.processing_results
    data, markdown, html = _read_reports(context.processing_results["report"])
    summary = data["processing_summary"]
    assert summary["total_files_discovered"] == summary["valid_files_found"] == 1
    assert summary["validation_success_rate"] == 100.0
    assert data["validation_analysis"]["valid"] == 1
    assert data["validation_analysis"]["invalid"] == 0
    assert (
        data["file_analysis"]["sizes"]["total_size"]
        == (context.target_dir / "perception.md").stat().st_size
    )
    assert "**Files Discovered:** 1" in markdown and "**Valid Files:** 1" in markdown
    assert "100.0%" in html and '<span class="success">' in html
    assert data["metadata"]["output_directory"] == str(context.output_dir)
    assert ProcessingPhase.REPORTING in context.phase_logs


def test_processor_refuses_actual_blocked_report_destination(tmp_path: Path) -> None:
    context = _native_context(tmp_path)
    context.output_dir.write_bytes(b"caller-owned occupied destination")
    assert GNNProcessor().process(context) is False
    assert "report_error" in context.processing_results
    assert context.output_dir.read_bytes() == b"caller-owned occupied destination"
    assert not list(tmp_path.glob("gnn_processing_report*"))


def test_processor_keeps_partial_reports_and_refuses_native_html_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.report import processing_report

    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 2, 3, 4, 5, tzinfo=tz)

    # Freeze only filename time. The native HTML open fails on a real directory;
    # JSON/Markdown are genuine files written by the ordinary generator.
    monkeypatch.setattr(processing_report, "datetime", FixedClock)
    context = _native_context(tmp_path)
    context.output_dir.mkdir()
    blocked = context.output_dir / "gnn_processing_report_20260102_030405.html"
    blocked.mkdir()
    assert GNNProcessor().process(context) is False
    report = context.processing_results["report"]
    assert set(report["report_files"]) == {"json", "markdown", "error"}
    assert "report_error" in context.processing_results
    assert (
        json.loads(Path(report["report_files"]["json"]).read_text())[
            "validation_analysis"
        ]["valid"]
        == 1
    )
    assert Path(report["report_files"]["markdown"]).is_file()
    assert blocked.is_dir()


@pytest.mark.parametrize("failure", ["omitted-format", "removed-file", "empty-file"])
def test_processor_refuses_incomplete_required_reports_at_the_public_generator_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    context = _native_context(tmp_path)
    processor = GNNProcessor()
    generate = processor.report_generator.generate_processing_report

    def incomplete_report(**kwargs):
        # First produce all three native files. Inject only a returned-metadata
        # omission or filesystem damage at the public dependency boundary.
        report = generate(**kwargs)
        html = Path(report["report_files"]["html"])
        assert html.is_file() and html.stat().st_size > 0
        if failure == "omitted-format":
            del report["report_files"]["html"]
        elif failure == "removed-file":
            html.unlink()
        else:
            html.write_bytes(b"")
        return report

    monkeypatch.setattr(
        processor.report_generator, "generate_processing_report", incomplete_report
    )
    assert processor.process(context) is False
    assert "Required html report" in context.processing_results["report_error"]
    report = context.processing_results["report"]
    assert Path(report["report_files"]["json"]).is_file()
    assert Path(report["report_files"]["markdown"]).is_file()
    assert "failed" in context.phase_logs[ProcessingPhase.REPORTING]


def test_report_preserves_typed_snapshot_counts_and_optional_check_receipts(
    tmp_path: Path,
) -> None:
    context = _native_context(
        tmp_path, recursive=True, enable_round_trip=True, enable_cross_format=True
    )
    first = context.target_dir / "perception.md"
    second, invalid = (
        context.target_dir / "other.md",
        context.target_dir / "invalid.json",
    )
    second.write_bytes(first.read_bytes())
    invalid.write_text('{"missing":')
    context.discovered_files = [first, second, invalid]
    context.valid_files = [first, second]
    results = {
        str(first): ValidationResult(True, warnings=["Precision: caller warning"]),
        str(second): ValidationResult(True),
        str(invalid): ValidationResult(
            False, errors=["Schema: required field missing", "Malformed syntax"]
        ),
    }
    context.processing_results["validation_results"] = results
    # These are caller-recorded check snapshots, not inference/performance claims.
    context.processing_results["round_trip_results"] = {
        "success": True,
        "summary": {
            "total_files": 2,
            "successful_files": 1,
            "failed_files": 1,
            "average_success_rate": 50.0,
            "format_performance": {"json": 50.0},
            "common_errors": ["Schema: round-trip mismatch"],
        },
    }
    context.processing_results["cross_format_results"] = {
        "success": True,
        "summary": {
            "total_files": 2,
            "consistent_files": 1,
            "inconsistent_files": 1,
            "average_consistency_rate": 50.0,
            "common_inconsistencies": ["one missing variable"],
        },
    }
    context.log_phase(ProcessingPhase.VALIDATION, "Recorded typed validation snapshot")
    context.log_phase(ProcessingPhase.ROUND_TRIP, "Recorded round-trip snapshot")
    report = ReportGenerator().generate_processing_report(context)
    data, markdown, html = _read_reports(report)
    assert data["validation_analysis"] == {
        "total": 3,
        "valid": 2,
        "invalid": 1,
        "success_rate": pytest.approx(200 / 3),
        "error_patterns": {"Schema": 1, "General": 1},
        "warning_patterns": {"Precision": 1},
    }
    assert data["processing_summary"]["validation_success_rate"] == pytest.approx(
        200 / 3
    )
    assert data["file_analysis"]["formats"] == {".md": 2, ".json": 1}
    assert data["file_analysis"]["sizes"]["total_size"] == sum(
        p.stat().st_size for p in context.discovered_files
    )
    assert data["round_trip_analysis"]["successful_files"] == 1
    assert data["round_trip_analysis"]["failed_files"] == 1
    assert data["cross_format_analysis"]["inconsistent_files"] == 1
    assert (
        "Round-Trip Testing Results" in markdown
        and "Cross-Format Validation Results" in markdown
    )
    assert '<span class="warning">' in html and "66.7%" in html
    assert context.processing_results["validation_results"] is results
    assert results[str(invalid)].errors == [
        "Schema: required field missing",
        "Malformed syntax",
    ]
    assert (
        data["processing_summary"]["phase_logs"]["validation"]
        == "Recorded typed validation snapshot"
    )


def test_report_uses_actual_structural_errors_and_warnings_without_mutation(
    tmp_path: Path,
) -> None:
    context = _native_context(tmp_path)
    good = context.target_dir / "perception.md"
    short = context.target_dir / "short.md"
    empty = context.target_dir / "empty.md"
    short.write_text("GNN model_name")
    empty.write_text("")
    context.discovered_files = [good, short, empty]
    results = {
        str(path): check_gnn_file_structure(path) for path in context.discovered_files
    }
    assert results[str(empty)]["valid"] is False and results[str(empty)]["errors"]
    assert results[str(short)]["valid"] is True and results[str(short)]["warnings"]
    context.valid_files = [
        path for path in context.discovered_files if results[str(path)]["valid"]
    ]
    context.processing_results["validation_results"] = results
    before = json.dumps(results, sort_keys=True)
    data, markdown, _ = _read_reports(
        ReportGenerator().generate_processing_report(context)
    )
    analysis = data["validation_analysis"]
    assert (
        analysis["total"] == 3 and analysis["valid"] == 2 and analysis["invalid"] == 1
    )
    assert sum(analysis["error_patterns"].values()) == sum(
        len(result["errors"]) for result in results.values()
    )
    assert sum(analysis["warning_patterns"].values()) == sum(
        len(result["warnings"]) for result in results.values()
    )
    assert "Common Validation Issues" in markdown
    assert json.dumps(results, sort_keys=True) == before


@pytest.mark.parametrize(
    "entry, error",
    [
        ({"valid": "False", "errors": [], "warnings": []}, TypeError),
        ({"valid": True, "warnings": []}, KeyError),
    ],
)
def test_malformed_saved_validation_is_refused_instead_of_counted_as_valid(
    tmp_path: Path, entry: dict, error: type[Exception]
) -> None:
    context = _native_context(tmp_path)
    context.processing_results["validation_results"] = {"caller-model": entry}
    with pytest.raises(error):
        ReportGenerator().generate_processing_report(context)
    assert not list(context.output_dir.glob("gnn_processing_report*"))
    assert context.processing_results["validation_results"]["caller-model"] is entry


def test_empty_report_is_zero_attempts_without_inventing_checks(tmp_path: Path) -> None:
    context = ProcessingContext(target_dir=tmp_path, output_dir=tmp_path / "empty")
    data, markdown, html = _read_reports(
        ReportGenerator().generate_processing_report(context)
    )
    assert data["file_analysis"] == {"total": 0, "formats": {}, "sizes": {}}
    assert (
        data["validation_analysis"]["valid"]
        == data["validation_analysis"]["invalid"]
        == 0
    )
    assert data["performance_metrics"]["files_per_second"] == 0
    assert data["performance_metrics"]["average_file_processing_time"] == 0
    assert "round_trip_analysis" not in data and "cross_format_analysis" not in data
    assert any("No GNN files found" in text for text in data["recommendations"])
    assert "**Files Discovered:** 0" in markdown and "0.0%" in html


def test_failed_optional_check_snapshots_are_not_rendered_as_success(
    tmp_path: Path,
) -> None:
    context = ProcessingContext(
        target_dir=tmp_path,
        output_dir=tmp_path / "failed_checks",
        enable_round_trip=True,
        enable_cross_format=True,
    )
    context.processing_results = {
        "round_trip_results": {
            "success": False,
            "error": "actual caller-recorded round-trip failure",
        },
        "cross_format_results": {
            "success": False,
            "error": "actual caller-recorded cross-format failure",
        },
    }
    data, markdown, _ = _read_reports(
        ReportGenerator().generate_processing_report(context)
    )
    assert data["round_trip_analysis"] == {
        "enabled": False,
        "error": "actual caller-recorded round-trip failure",
    }
    assert data["cross_format_analysis"] == {
        "enabled": False,
        "error": "actual caller-recorded cross-format failure",
    }
    assert (
        "Round-Trip Testing Results" not in markdown
        and "Cross-Format Validation Results" not in markdown
    )


@pytest.mark.parametrize(
    "valid_count, css_class", [(0, "error"), (3, "warning"), (4, "success")]
)
def test_saved_html_success_rate_uses_actual_snapshot_population(
    tmp_path: Path, valid_count: int, css_class: str
) -> None:
    context = ProcessingContext(
        target_dir=tmp_path, output_dir=tmp_path / "rates", validation_level="basic"
    )
    files = []
    for index in range(5):
        path = tmp_path / f"source_{index}.md"
        path.write_text(f"## ModelName\nExternal snapshot {index}\n")
        files.append(path)
    context.discovered_files, context.valid_files = files, files[:valid_count]
    context.processing_results["validation_results"] = {
        str(path): ValidationResult(index < valid_count)
        for index, path in enumerate(files)
    }
    data, _, html = _read_reports(ReportGenerator().generate_processing_report(context))
    assert data["validation_analysis"]["valid"] == valid_count
    assert data["validation_analysis"]["invalid"] == 5 - valid_count
    assert f'<span class="{css_class}">' in html
    assert f"{20 * valid_count:.1f}%" in html
    assert any("standard" in item for item in data["recommendations"])
