"""Public reports over genuinely generated local pipeline evidence.

Only Steps 3, 8 and 23 execute. Saved-run replay is explicitly historical;
its timestamps and run identity must come from its original receipt.
No registry replacements, fabricated execution summaries or provider calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest
from PIL import Image

from gnn.pipeline import run_pipeline
from gnn.report import generate_comprehensive_report, process_report

REPO = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
LOGGER.propagate = False
pytestmark = pytest.mark.integration


def _hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


class _Document(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.tags: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        self.links.extend(
            value for key, value in attrs if key == "href" and value is not None
        )

    def handle_data(self, data: str) -> None:
        self.text.append(data)


@pytest.fixture(scope="module")
def native_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    workspace = tmp_path_factory.mktemp("native-report-run")
    source = workspace / "source"
    source.mkdir()
    for relative in (
        "basics/static_perception.md",
        "discrete/actinf_pomdp_agent.md",
    ):
        original = REPO / "input/gnn_files" / relative
        shutil.copyfile(original, source / original.name)
    before = _hashes(source)
    output = workspace / "output"
    outcome = run_pipeline(
        target_dir=source,
        output_dir=output,
        steps=[3, 8, 23],
        strict=False,
        consolidated_steps=True,
    )
    assert outcome["success"], outcome
    assert _hashes(source) == before
    final = json.loads(Path(outcome["summary_file"]).read_text())
    assert final["run_id"] == outcome["run_id"]
    assert {step["script_name"] for step in final["steps"]} == {
        "3_gnn.py",
        "8_visualization.py",
        "23_report.py",
    }
    return {"source": source, "output": output, "outcome": outcome, "final": final}


def _saved_copy(native_run: dict[str, Any], tmp_path: Path) -> Path:
    destination = tmp_path / "saved-pipeline"
    shutil.copytree(native_run["output"], destination)
    return destination


def test_native_step23_snapshot_and_gallery_have_actual_owned_evidence(
    native_run: dict[str, Any],
) -> None:
    output = native_run["output"]
    folder = output / "23_report_output"
    report = json.loads((folder / "report_summary.json").read_text())
    receipt = report["pipeline_summary"]
    assert receipt["run_id"] == native_run["outcome"]["run_id"]
    assert receipt["evidence_phase"] == "current_run_snapshot"
    assert receipt["preliminary"] is True
    assert receipt["model_selection"] == native_run["final"]["model_selection"]
    assert len(receipt["model_selection"]) == 2
    assert report["health_score_basis"] == "pipeline_execution_summary"
    assert report["summary"]["status_source"] == "pipeline_execution_summary"
    assert report["report_timestamp_source"] == "pipeline_execution_summary"
    assert report["report_generation_time"] == receipt["end_time"]

    statuses = [step["status"] for step in receipt["steps"]]
    assert statuses
    successes = sum(status.startswith("SUCCESS") for status in statuses)
    assert report["summary"]["success_rate"] == pytest.approx(
        100 * successes / len(statuses)
    )
    assert report["summary"]["successful_steps"] == successes
    assert report["summary"]["failed_steps"] == statuses.count("FAILED")

    owned = {
        record["path"]: record["sha256"]
        for step in receipt["steps"]
        for record in step.get("artifacts", [])
    }
    gallery = report["visualizations"]
    images = gallery["by_type"]["image"]
    assert len(images) > 5
    assert gallery["total_count"] == sum(
        len(items) for items in gallery["by_type"].values()
    )
    assert gallery["total_count"] == len(gallery["all_visualizations"])
    for image in gallery["all_visualizations"]:
        path = output / image["relative_path"]
        assert image["relative_path"] in owned
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == owned[image["relative_path"]]
        )
        assert image["size_bytes"] == path.stat().st_size
        if path.suffix == ".svg":
            assert ElementTree.parse(path).getroot().tag.endswith("svg")
        elif image["type"] == "image":
            with Image.open(path) as decoded:
                decoded.verify()
        else:
            document = _Document()
            document.feed(path.read_text())
            assert "html" in document.tags and "body" in document.tags

    html = _Document()
    html.feed((folder / "comprehensive_analysis_report.html").read_text())
    assert "html" in html.tags and "body" in html.tags
    gallery_links = [link for link in html.links if link.startswith("../8_")]
    assert gallery_links
    assert all((folder / link).is_file() for link in gallery_links)
    markdown = (folder / "comprehensive_analysis_report.md").read_text()
    assert f"{report['summary']['success_rate']:.1f}%" in markdown
    assert "Visualizations" in markdown
    assert "more" in "".join(html.text)
    processing = json.loads((folder / "report_processing_summary.json").read_text())
    assert processing["processing_status"] == "completed"
    assert processing["visualizations_discovered"] == gallery["total_count"]
    assert processing["html_validation_status"] == "valid"


def test_public_saved_run_replay_keeps_historical_identity_and_source_bytes(
    native_run: dict[str, Any],
    tmp_path: Path,
) -> None:
    saved = _saved_copy(native_run, tmp_path)
    source_directories = [saved / "3_gnn_output", saved / "8_visualization_output"]
    before = {str(path): _hashes(path) for path in source_directories}
    final_bytes = (
        saved / "00_pipeline_summary/pipeline_execution_summary.json"
    ).read_bytes()
    reports = saved / "23_report_output"
    assert process_report(native_run["source"], reports, logger=LOGGER) is True
    data = json.loads((reports / "report_summary.json").read_text())
    assert data["pipeline_summary"]["run_id"] == native_run["final"]["run_id"]
    assert data["report_generation_time"] == native_run["final"]["end_time"]
    assert data["pipeline_summary"]["steps"] == native_run["final"]["steps"]
    assert {str(path): _hashes(path) for path in source_directories} == before
    assert (
        saved / "00_pipeline_summary/pipeline_execution_summary.json"
    ).read_bytes() == final_bytes
    assert (saved / "PIPELINE_REPORT.md").is_file()


@pytest.mark.parametrize("formats", [["json"], ["markdown"], ["html"]])
def test_public_saved_report_writes_only_requested_formats(
    native_run: dict[str, Any],
    tmp_path: Path,
    formats: list[str],
) -> None:
    output = tmp_path / "selected-formats"
    saved = native_run["output"]
    before = _hashes(saved)
    assert generate_comprehensive_report(saved, output, LOGGER, report_formats=formats)
    filenames = {
        "json": "report_summary.json",
        "markdown": "comprehensive_analysis_report.md",
        "html": "comprehensive_analysis_report.html",
    }
    assert {path.name for path in output.iterdir()} == {
        filenames[formats[0]],
        "report_generation_summary.json",
    }
    summary = json.loads((output / "report_generation_summary.json").read_text())
    assert summary["report_generation_summary"]["generated_files"] == [
        filenames[formats[0]]
    ]
    assert _hashes(saved) == before


@pytest.mark.parametrize("failure", ["missing-input", "file-input", "blocked-output"])
def test_public_report_refuses_native_filesystem_failures(
    native_run: dict[str, Any],
    tmp_path: Path,
    failure: str,
) -> None:
    saved = native_run["output"]
    output = tmp_path / "report"
    before = _hashes(saved)
    if failure == "missing-input":
        saved = tmp_path / "absent-input"
    elif failure == "file-input":
        saved = tmp_path / "input-file"
        saved.write_text("This is a file, not a saved pipeline directory.\n")
    else:
        output.write_text("Retain the caller's blocked output.\n")
    assert generate_comprehensive_report(saved, output, LOGGER) is False
    if failure == "blocked-output":
        assert output.read_text() == "Retain the caller's blocked output.\n"
    else:
        assert not output.exists()
    assert _hashes(native_run["output"]) == before


def test_invalid_standalone_summary_is_reported_as_artifact_coverage(
    native_run: dict[str, Any],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    saved = _saved_copy(native_run, tmp_path)
    summary = saved / "00_pipeline_summary/pipeline_execution_summary.json"
    summary.write_text("{ invalid archived summary", encoding="utf-8")
    before = _hashes(saved)
    output = tmp_path / "degraded-report"
    # The caller owns this logger. Capture its genuine diagnostics directly
    # while keeping it isolated from the historical run's root file handlers.
    LOGGER.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=LOGGER.name):
            assert generate_comprehensive_report(saved, output, LOGGER) is True
    finally:
        LOGGER.removeHandler(caplog.handler)
    data = json.loads((output / "report_summary.json").read_text())
    assert "Failed to read pipeline summary" in caplog.text
    assert data["health_score_basis"] == "artifact_coverage_only"
    assert data["summary"]["status_source"] == "artifacts_only"
    assert data["report_timestamp_source"] == "unavailable"
    assert data["report_generation_time"] == "unavailable"
    assert "pipeline_summary" not in data
    assert "Artifact" in (output / "comprehensive_analysis_report.md").read_text()
    assert _hashes(saved) == before


@pytest.mark.parametrize("formats", [[], ["pdf"], ["pdf", "json"]])
def test_empty_and_unsupported_formats_keep_existing_best_effort_contract(
    native_run: dict[str, Any],
    tmp_path: Path,
    formats: list[str],
) -> None:
    request_before = formats.copy()
    source_before = _hashes(native_run["output"])
    output = tmp_path / "best-effort-formats"
    success = generate_comprehensive_report(
        native_run["output"],
        output,
        LOGGER,
        report_formats=formats,
    )
    supported = "json" in formats
    assert success is supported
    assert formats == request_before
    expected_files = {"report_generation_summary.json"}
    if supported:
        expected_files.add("report_summary.json")
        report = json.loads((output / "report_summary.json").read_text())
        assert report["report_metadata"]["formats_generated"] == request_before
        assert report["pipeline_summary"]["run_id"] == native_run["final"]["run_id"]
    assert {path.name for path in output.iterdir()} == expected_files
    summary = json.loads((output / "report_generation_summary.json").read_text())
    assert summary["report_generation_summary"]["generated_files"] == (
        ["report_summary.json"] if supported else []
    )
    assert _hashes(native_run["output"]) == source_before


def test_native_partial_writer_failure_keeps_completed_formats_and_blocked_path(
    native_run: dict[str, Any],
    tmp_path: Path,
) -> None:
    output = tmp_path / "partial-output"
    blocked = output / "report_summary.json"
    blocked.mkdir(parents=True)
    sentinel = blocked / "retain.txt"
    sentinel.write_text("Caller owns this blocking directory.\n")
    before = _hashes(native_run["output"])
    assert generate_comprehensive_report(native_run["output"], output, LOGGER) is True
    assert sentinel.read_text() == "Caller owns this blocking directory.\n"
    assert blocked.is_dir()
    assert (output / "comprehensive_analysis_report.html").is_file()
    assert (output / "comprehensive_analysis_report.md").is_file()
    receipt = json.loads((output / "report_generation_summary.json").read_text())
    assert receipt["report_generation_summary"]["generated_files"] == [
        "comprehensive_analysis_report.html",
        "comprehensive_analysis_report.md",
    ]
    assert _hashes(native_run["output"]) == before


def test_public_step23_wrapper_preserves_requested_subset_and_actual_saved_inventory(
    native_run: dict[str, Any],
    tmp_path: Path,
) -> None:
    saved = _saved_copy(native_run, tmp_path)
    source_before = _hashes(saved / "8_visualization_output")
    report_dir = saved / "requested-reports"
    formats = ["markdown"]
    assert process_report(
        native_run["source"], report_dir, logger=LOGGER, report_formats=formats
    )
    assert formats == ["markdown"]
    assert (report_dir / "comprehensive_analysis_report.md").is_file()
    assert not (report_dir / "comprehensive_analysis_report.html").exists()
    assert not (report_dir / "report_summary.json").exists()
    receipt = json.loads((report_dir / "report_processing_summary.json").read_text())
    assert receipt["formats_generated"] == ["markdown"]
    assert receipt["reports_generated"] == [
        "comprehensive_analysis_report.md",
        "PIPELINE_REPORT.md",
    ]
    assert receipt["html_reports_generated"] == 0
    assert receipt["html_validation_status"] == "no_html_reports"
    assert _hashes(saved / "8_visualization_output") == source_before
