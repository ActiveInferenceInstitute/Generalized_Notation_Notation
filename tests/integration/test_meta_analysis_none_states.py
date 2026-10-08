"""Regression test for the meta-analysis resource-efficiency table.

LOC records come from the render summary, which does not always carry sweep
parameters, so ``num_states`` can be None. Sorting the raw values mixed None
with int and crashed the whole meta-analysis (reported as "Meta-analysis
failed (non-fatal)" in Step 17). The fix buckets None under 0; this test
locks that contract with real SweepRecord objects (no doubles).
"""

from __future__ import annotations

import logging
from pathlib import Path

from gnn.integration.meta_analysis.collector import SweepRecord
from gnn.integration.meta_analysis.reporter import SweepReporter


def _record(
    num_states: int | None, lines_of_code: int | None = 120, framework: str = "pymdp"
) -> SweepRecord:
    return SweepRecord(
        model_name="pymdp_scaling_N3_T100",
        framework=framework,
        num_states=num_states,
        num_timesteps=100,
        lines_of_code=lines_of_code,
        success=True,
        execution_time=1.5,
    )


def test_resource_efficiency_survives_none_num_states(tmp_path: Path) -> None:
    records = [
        _record(None),  # LOC record without sweep parameters — the crash case
        _record(3),
        _record(9),
    ]
    reporter = SweepReporter(records, [], tmp_path)
    # Must not raise TypeError: '<' not supported between NoneType and int
    section = reporter._resource_efficiency()
    assert "Recorded Source Sizes by State-Space Size" in section
    assert (
        "| N | pymdp LOC |\n"
        "|---|---|\n"
        "| 0 | 120 |\n"
        "| 3 | 120 |\n"
        "| 9 | 120 |"
    ) in section
    # The authored inputs declare 100 steps in 1.5 seconds: 66.67 steps/sec.
    assert (
        "| T=100 | pymdp | 66.67 |\n"
        "| N=3, T=100 | pymdp | 66.67 |\n"
        "| N=9, T=100 | pymdp | 66.67 |"
    ) in section


def test_resource_efficiency_empty_when_no_loc(tmp_path: Path) -> None:
    records = [_record(3, lines_of_code=None)]
    reporter = SweepReporter(records, [], tmp_path)
    assert reporter._resource_efficiency() == ""
