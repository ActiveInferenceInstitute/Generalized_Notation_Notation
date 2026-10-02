"""Current-run report collection must not admit historical snapshot evidence."""

import json
import logging
from pathlib import Path

import pytest

from gnn.pipeline.run_context import build_run_context
from gnn.report.analyzer import collect_pipeline_data

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
