"""Current-run report collection must not admit historical snapshot evidence."""

import hashlib
import json
import logging
from argparse import Namespace
from pathlib import Path

import pytest

from gnn.pipeline.run_context import build_run_context
from gnn.pipeline.run_manifest import emit_run_manifests
from gnn.report.analyzer import collect_pipeline_data
from gnn.report.pipeline_report import generate_pipeline_report

pytestmark = pytest.mark.pipeline


@pytest.mark.parametrize("current", [None, {"run_id": "old-run"}, []])
def test_current_run_report_rejects_missing_stale_and_malformed_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, current
) -> None:
    source = tmp_path / "input"
    source.mkdir()
    (source / "model.md").write_text("## ModelName\nModel\n")
    context = build_run_context(source, tmp_path / "output", "current-run", [23], {})
    context_file = tmp_path / "context.json"
    context.write(context_file)
    monkeypatch.setenv("GNN_RUN_CONTEXT_FILE", str(context_file))
    monkeypatch.setenv("GNN_RUN_ID", context.run_id)
    summaries = Path(context.output_root) / "00_pipeline_summary"
    summaries.mkdir(parents=True)
    (summaries / "pipeline_execution_summary.json").write_text(
        json.dumps({"run_id": "old-run", "overall_status": "SUCCESS"})
    )
    if current is not None:
        (summaries / "current_summary.json").write_text(json.dumps(current))
    with pytest.raises(ValueError, match="Current-run report snapshot"):
        collect_pipeline_data(Path(context.output_root), logging.getLogger(__name__))


def test_terminal_report_uses_verified_inventory(tmp_path: Path, capsys) -> None:
    from gnn.cli import _cmd_report

    summaries = tmp_path / "00_pipeline_summary"
    summaries.mkdir()
    session = summaries / "run_session.json"
    session.write_text(json.dumps({"session_id": "run-one", "final_status": "SUCCESS"}))
    parse_output = tmp_path / "3_gnn_output"
    parse_output.mkdir()
    (parse_output / "gnn_results.json").write_text('{"total_files": 991}')
    owned = parse_output / "current.json"
    owned.write_text("{}")
    other = tmp_path / "7_export_output"
    other.mkdir()
    (other / "old.json").write_text("{}")
    summary_path = summaries / "pipeline_execution_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_id": "run-one",
                "overall_status": "SUCCESS",
                "artifact_inventory_contract": "current-run-v1",
                "evidence_integrity": {"status": "verified"},
                "run_session_sha256": hashlib.sha256(session.read_bytes()).hexdigest(),
                "steps": [
                    {
                        "script_name": "3_gnn.py",
                        "status": "SUCCESS",
                        "artifacts": [
                            {
                                "path": "3_gnn_output/current.json",
                                "sha256": hashlib.sha256(
                                    owned.read_bytes()
                                ).hexdigest(),
                            }
                        ],
                    }
                ],
            }
        )
    )
    emit_run_manifests(tmp_path)

    report = generate_pipeline_report(tmp_path)
    assert "Verified terminal run inventory" in report
    assert "current.json" in report
    assert "gnn_results.json" not in report
    assert "Files parsed" not in report
    assert "old.json" not in report
    assert "7_export_output" not in report

    comprehensive = collect_pipeline_data(tmp_path, logging.getLogger(__name__))
    assert comprehensive["evidence_scope"] == "Verified terminal run inventory"
    assert comprehensive["steps"]["3_gnn_output"]["file_count"] == 1
    assert not comprehensive["steps"]["7_export_output"]["exists"]

    published = tmp_path / "PIPELINE_REPORT.md"
    published.write_text(report)
    owned.write_text('{"changed": true}')
    assert _cmd_report(Namespace(output_dir=str(tmp_path), json=True)) != 0
    assert json.loads(capsys.readouterr().out)["status"] == "error"
    assert published.read_text() == report
    with pytest.raises(ValueError, match="Report evidence verification failed"):
        collect_pipeline_data(tmp_path, logging.getLogger(__name__))
