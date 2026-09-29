#!/usr/bin/env python3
"""
Unit tests for the intelligent_analysis composability surface added in the
2026-09-04 fleet3 pass: the deterministic analysis core
(``compute_full_analysis`` / ``FullAnalysisResult``), the run-history module
(``history.py``), and the shared helpers they expose.

All tests are pure (no network, no LLM, no pipeline execution).
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.intelligent_analysis import (  # noqa: E402
    FullAnalysisResult,
    StepAnalysis,
    compute_full_analysis,
)
from gnn.intelligent_analysis.history import (  # noqa: E402
    RunSnapshot,
    StepDelta,
    analyze_run_history,
    build_run_snapshot,
    classify_trend,
    compute_step_deltas,
)
from gnn.intelligent_analysis.processor import (  # noqa: E402
    resolve_analysis_output_dir,
    resolve_pipeline_summary_path,
)


def _successful_summary() -> dict[str, Any]:
    """Pipeline summary with a fast clean run."""
    return {
        "overall_status": "SUCCESS",
        "total_duration_seconds": 10.0,
        "end_time": "2026-09-04T12:00:00Z",
        "performance_summary": {"peak_memory_mb": 100.0},
        "steps": [
            {
                "step_number": 3,
                "script_name": "3_gnn.py",
                "description": "GNN parsing",
                "status": "SUCCESS",
                "duration_seconds": 2.0,
                "peak_memory_mb": 50.0,
                "exit_code": 0,
                "stdout": "Parsed 1 model",
                "stderr": "",
            },
            {
                "step_number": 12,
                "script_name": "12_execute.py",
                "description": "Execute simulations",
                "status": "SUCCESS",
                "duration_seconds": 4.0,
                "peak_memory_mb": 80.0,
                "exit_code": 0,
                "stdout": "Completed",
                "stderr": "",
            },
        ],
    }


def _failed_summary() -> dict[str, Any]:
    """Pipeline summary with a failure and slow/memory-heavy step."""
    return {
        "overall_status": "FAILED",
        "total_duration_seconds": 200.0,
        "end_time": "2026-09-04T13:00:00Z",
        "performance_summary": {"peak_memory_mb": 900.0},
        "steps": [
            {
                "step_number": 3,
                "script_name": "3_gnn.py",
                "description": "GNN parsing",
                "status": "SUCCESS",
                "duration_seconds": 2.0,
                "peak_memory_mb": 50.0,
                "exit_code": 0,
            },
            {
                "step_number": 12,
                "script_name": "12_execute.py",
                "description": "Execute simulations",
                "status": "FAILED",
                "duration_seconds": 90.0,
                "peak_memory_mb": 900.0,
                "exit_code": 1,
                "stderr": "ModuleNotFoundError: No module named 'pymdp'",
            },
        ],
    }


class TestComputeFullAnalysis:
    """The deterministic analysis core."""

    @pytest.mark.unit
    def test_clean_run_yields_no_red_flags(self) -> None:
        result = compute_full_analysis(_successful_summary())
        assert result.red_count == 0
        assert result.yellow_count == 0
        assert result.green_count == 2
        assert result.analysis["overall_status"] == "SUCCESS"
        assert result.recovery_plan == []

    @pytest.mark.unit
    def test_failed_run_yields_red_flag_and_recovery_plan(self) -> None:
        result = compute_full_analysis(_failed_summary())
        assert result.red_count == 1
        assert result.failures[0]["step_name"] == "12_execute.py"
        assert len(result.recovery_plan) >= 1
        # Failure severity recorded per failed step
        assert result.analysis["failures"][0]["step"] == "12_execute.py"

    @pytest.mark.unit
    def test_bottleneck_threshold_passthrough(self) -> None:
        # 90s step is under the default-vs-threshold boundary here; lower
        # the threshold far enough to guarantee detection.
        detected = compute_full_analysis(
            _failed_summary(), bottleneck_threshold=10.0
        ).bottlenecks
        assert any(b["step"] == "12_execute.py" for b in detected)

    @pytest.mark.unit
    def test_to_dict_matches_file_payload_shape(self) -> None:
        result = compute_full_analysis(_failed_summary())
        payload = result.to_dict()
        assert payload["flags_summary"] == {
            "red_count": result.red_count,
            "yellow_count": result.yellow_count,
            "green_count": result.green_count,
        }
        assert len(payload["step_analyses"]) == 2
        assert payload["step_analyses"][0]["script_name"] == "3_gnn.py"
        # Snippets are excluded from the default serialization payload.
        assert "stdout_snippet" not in payload["step_analyses"][0]
        assert set(payload) == {
            "analysis",
            "step_analyses",
            "flags_summary",
            "bottlenecks",
            "failures",
            "recommendations",
            "recovery_plan",
        }

    @pytest.mark.unit
    def test_is_pure_no_filesystem_or_logging(self, tmp_path: Path) -> None:
        """Repeated calls on the same input return equal, side-effect-free
        results (nothing written under tmp_path)."""
        a = compute_full_analysis(_failed_summary())
        b = compute_full_analysis(_failed_summary())
        assert a.to_dict() == b.to_dict()
        assert list(tmp_path.iterdir()) == []


class TestStepAnalysisToDict:
    """StepAnalysis.to_dict serialization contract."""

    @pytest.mark.unit
    def test_default_payload_omits_snippets(self) -> None:
        from gnn.intelligent_analysis import StepAnalysis

        sa = StepAnalysis(
            step_number=3,
            script_name="3_gnn.py",
            description="parse",
            status="SUCCESS",
            duration_seconds=1.5,
            memory_mb=42.0,
            exit_code=0,
            flags=["Slow: 70.0s (>60.0s threshold)"],
            flag_type="yellow",
            summary="Completed with 1 flag(s) in 1.50s",
            stdout_snippet="out",
            stderr_snippet="err",
        )
        data = sa.to_dict()
        assert data["flag_type"] == "yellow"
        assert "stdout_snippet" not in data
        assert "stderr_snippet" not in data
        assert data["flags"] == ["Slow: 70.0s (>60.0s threshold)"]

    @pytest.mark.unit
    def test_include_snippets_flag(self) -> None:
        from gnn.intelligent_analysis import StepAnalysis

        sa = StepAnalysis(
            step_number=1,
            script_name="1_setup.py",
            description="setup",
            status="FAILED",
            duration_seconds=2.0,
            memory_mb=10.0,
            exit_code=2,
            stdout_snippet="OUT",
            stderr_snippet="ERR",
        )
        data = sa.to_dict(include_snippets=True)
        assert data["stdout_snippet"] == "OUT"
        assert data["stderr_snippet"] == "ERR"


class TestResolveHelpers:
    """Path-resolution helpers extracted from the entry point."""

    @pytest.mark.unit
    def test_summary_path_prefers_output_dir(self, tmp_path: Path) -> None:
        summary_dir = tmp_path / "00_pipeline_summary"
        summary_dir.mkdir()
        expected = summary_dir / "pipeline_execution_summary.json"
        expected.write_text("{}")
        assert resolve_pipeline_summary_path(tmp_path) == expected

    @pytest.mark.unit
    def test_summary_path_falls_back_to_parent(self, tmp_path: Path) -> None:
        child = tmp_path / "24_intelligent_analysis_output"
        child.mkdir()
        parent_summary = (
            tmp_path / "00_pipeline_summary" / "pipeline_execution_summary.json"
        )
        parent_summary.parent.mkdir()
        parent_summary.write_text("{}")
        assert resolve_pipeline_summary_path(child) == parent_summary

    @pytest.mark.unit
    def test_current_run_preliminary_summary_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Step 24 runs before the final summary exists; a final summary on disk
        # is the previous run's, so the current run's preliminary must win.
        summary_dir = tmp_path / "00_pipeline_summary"
        summary_dir.mkdir()
        (summary_dir / "pipeline_execution_summary.json").write_text(
            json.dumps({"run_id": "previous"})
        )
        prelim = summary_dir / "preliminary_summary.json"
        prelim.write_text(json.dumps({"run_id": "current"}))
        monkeypatch.setenv("GNN_RUN_ID", "current")
        assert resolve_pipeline_summary_path(tmp_path) == prelim
        monkeypatch.setenv("GNN_RUN_ID", "another-run")
        assert resolve_pipeline_summary_path(tmp_path) == (
            summary_dir / "pipeline_execution_summary.json"
        )

    @pytest.mark.unit
    def test_output_dir_passthrough_when_named_for_step(self, tmp_path: Path) -> None:
        step_dir = tmp_path / "24_intelligent_analysis_output"
        step_dir.mkdir()
        assert resolve_analysis_output_dir(step_dir) == step_dir

    @pytest.mark.unit
    def test_output_dir_creates_subdirectory_path(self, tmp_path: Path) -> None:
        assert (
            resolve_analysis_output_dir(tmp_path)
            == tmp_path / "24_intelligent_analysis_output"
        )


class TestRunHistory:
    """history.py run-over-run analysis."""

    @pytest.mark.unit
    def test_snapshot_counts_statuses(self) -> None:
        snap = build_run_snapshot(_failed_summary())
        assert isinstance(snap, RunSnapshot)
        assert snap.failure_count == 1
        assert snap.successful_count == 1
        assert snap.step_count == 2
        assert snap.overall_status == "FAILED"

    @pytest.mark.unit
    def test_snapshot_prefers_end_time(self) -> None:
        snap = build_run_snapshot(_failed_summary())
        assert snap.timestamp == "2026-09-04T13:00:00Z"

    @pytest.mark.unit
    def test_snapshot_timestamp_unavailable(self) -> None:
        summary = _successful_summary()
        del summary["end_time"]
        assert build_run_snapshot(summary).timestamp == "unavailable"

    @pytest.mark.unit
    def test_step_deltas_match_by_script_name(self) -> None:
        deltas = compute_step_deltas(_failed_summary(), _successful_summary())
        by_name = {d.script_name: d for d in deltas}
        assert set(by_name) == {"3_gnn.py", "12_execute.py"}
        assert by_name["3_gnn.py"].duration_delta_seconds == pytest.approx(0.0)
        # Present in current but absent from previous -> omitted.
        only_current = {"steps": [{"script_name": "9_new.py", "status": "SUCCESS"}]}
        assert compute_step_deltas(only_current, _successful_summary()) == []

    @pytest.mark.unit
    def test_deltas_sorted_by_absolute_duration(self) -> None:
        deltas = compute_step_deltas(_failed_summary(), _successful_summary())
        magnitudes = [abs(d.duration_delta_seconds) for d in deltas]
        assert magnitudes == sorted(magnitudes, reverse=True)

    @pytest.mark.unit
    def test_classify_trend_directions(self) -> None:
        assert classify_trend(90.0, [70.0])["direction"] == "improving"
        assert classify_trend(50.0, [80.0])["direction"] == "degrading"
        assert classify_trend(72.0, [70.0])["direction"] == "stable"
        assert classify_trend(70.0, [])["direction"] == "insufficient_history"
        no_history = classify_trend(70.0, [])
        assert no_history["mean_previous"] is None
        assert no_history["delta"] is None

    @pytest.mark.unit
    def test_analyze_run_history_shape(self) -> None:
        history = analyze_run_history(
            _failed_summary(), [_successful_summary()], current_label="run-9"
        )
        assert history["current"]["script_name"] == "run-9"
        assert len(history["previous"]) == 1
        assert history["trend"]["direction"] in {
            "improving",
            "degrading",
            "stable",
        }
        assert isinstance(history["step_deltas"], list)

    @pytest.mark.unit
    def test_max_deltas_limits_output(self) -> None:
        current = _failed_summary()
        current["steps"] = [
            {
                "script_name": f"s{i}.py",
                "status": "SUCCESS",
                "duration_seconds": float(i),
                "peak_memory_mb": 10.0,
            }
            for i in range(10)
        ]
        previous = {
            "steps": [
                {
                    "script_name": f"s{i}.py",
                    "status": "SUCCESS",
                    "duration_seconds": float(i) * 2,
                    "peak_memory_mb": 20.0,
                }
                for i in range(10)
            ]
        }
        history = analyze_run_history(current, [previous], max_deltas=3)
        assert len(history["step_deltas"]) == 3


class TestPackageExports:
    """New API is reachable from the package root."""

    @pytest.mark.unit
    def test_new_names_in_all(self) -> None:
        import gnn.intelligent_analysis as ia

        for name in (
            "FullAnalysisResult",
            "compute_full_analysis",
            "RunSnapshot",
            "StepDelta",
            "build_run_snapshot",
            "compute_step_deltas",
            "classify_trend",
            "analyze_run_history",
        ):
            assert name in ia.__all__, name
            assert getattr(ia, name) is not None

    @pytest.mark.unit
    def test_full_analysis_result_is_dataclass(self) -> None:
        import dataclasses

        assert dataclasses.is_dataclass(FullAnalysisResult)
        assert dataclasses.is_dataclass(StepAnalysis)


class TestEntryPointDelegation:
    """process_intelligent_analysis delegates to the deterministic core."""

    @pytest.mark.unit
    def test_process_uses_core_and_writes_outputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json
        import logging

        summary_dir = tmp_path / "00_pipeline_summary"
        summary_dir.mkdir()
        (summary_dir / "pipeline_execution_summary.json").write_text(
            json.dumps(_failed_summary())
        )
        logger = logging.getLogger("test-entrypoint")
        logger.setLevel(logging.DEBUG)

        from gnn.intelligent_analysis.processor import process_intelligent_analysis

        ok = process_intelligent_analysis(
            target_dir=tmp_path,
            output_dir=tmp_path,
            logger=logger,
            skip_llm=True,
        )
        assert ok is True
        out_dir = tmp_path / "24_intelligent_analysis_output"
        assert (out_dir / "intelligent_analysis_report.md").exists()
        data = json.loads((out_dir / "analysis_data.json").read_text())
        assert data["flags_summary"]["red_count"] == 1
        assert data["analysis_source"] == "rule_based"
        assert data["timestamp"] == "2026-09-04T13:00:00Z"
        assert data["timestamp_source"] == "pipeline_execution_summary"

    @pytest.mark.unit
    def test_missing_summary_writes_partial_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No summary anywhere: partial report, success exit, no crash."""
        monkeypatch.setattr(
            "gnn.intelligent_analysis.processor.time.sleep", lambda s: None
        )
        logger = logging.getLogger("test-partial")

        from gnn.intelligent_analysis.processor import process_intelligent_analysis

        ok = process_intelligent_analysis(
            target_dir=tmp_path, output_dir=tmp_path, logger=logger
        )
        assert ok is True
        report = (
            tmp_path
            / "24_intelligent_analysis_output"
            / "intelligent_analysis_report.md"
        ).read_text()
        assert "not available" in report
