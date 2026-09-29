#!/usr/bin/env python3
"""The preliminary summary handed to steps 23/24 must carry the real status.

It used to hard-code ``overall_status = "SUCCESS"``, so step 24 analysed a run
whose Step 12 had failed as if everything had passed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from gnn.main import _write_preliminary_pipeline_summary


def _summary(failed: int, warnings: int) -> dict:
    return {
        "run_id": "r1",
        "start_time": datetime.now().isoformat(),
        "steps": [{"script_name": "12_execute.py", "status": "FAILED"}],
        "performance_summary": {
            "total_steps": 22,
            "failed_steps": failed,
            "critical_failures": 0,
            "warnings": warnings,
        },
    }


def _written(tmp_path: Path, summary: dict) -> dict:
    _write_preliminary_pipeline_summary(summary, tmp_path, logging.getLogger("t"))
    return json.loads(
        (tmp_path / "00_pipeline_summary" / "preliminary_summary.json").read_text()
    )


def test_failed_step_is_not_reported_as_success(tmp_path: Path) -> None:
    prelim = _written(tmp_path, _summary(failed=1, warnings=3))
    assert prelim["overall_status"] == "SUCCESS_WITH_WARNINGS"
    assert prelim["preliminary"] is True
    assert prelim["run_id"] == "r1"


def test_clean_run_is_success(tmp_path: Path) -> None:
    summary = _summary(failed=0, warnings=0)
    summary["steps"] = [{"script_name": "3_gnn.py", "status": "SUCCESS"}]
    assert _written(tmp_path, summary)["overall_status"] == "SUCCESS"


def test_live_summary_is_not_mutated(tmp_path: Path) -> None:
    summary = _summary(failed=1, warnings=0)
    summary["overall_status"] = "RUNNING"
    _written(tmp_path, summary)
    assert summary["overall_status"] == "RUNNING"
    assert "end_time" not in summary
