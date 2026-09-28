"""
Test Intelligent Analysis Overall Tests

This file contains comprehensive tests for the intelligent_analysis module functionality.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
SAMPLE_PIPELINE_SUMMARY: dict[str, Any] = {
    "start_time": "2024-01-23T10:00:00",
    "end_time": "2024-01-23T10:05:30",
    "total_duration_seconds": 330.5,
    "overall_status": "SUCCESS_WITH_WARNINGS",
    "steps": [
        {
            "step_number": 1,
            "script_name": "0_template.py",
            "description": "Template initialization",
            "status": "SUCCESS",
            "duration_seconds": 2.5,
            "exit_code": 0,
            "peak_memory_mb": 150.0,
        },
        {
            "step_number": 2,
            "script_name": "1_setup.py",
            "description": "Environment setup",
            "status": "SUCCESS",
            "duration_seconds": 15.3,
            "exit_code": 0,
            "peak_memory_mb": 200.0,
        },
        {
            "step_number": 3,
            "script_name": "2_tests.py",
            "description": "Test execution",
            "status": "SUCCESS_WITH_WARNINGS",
            "duration_seconds": 120.0,
            "exit_code": 0,
            "peak_memory_mb": 512.0,
            "stdout": "WARNING: Some tests skipped",
        },
        {
            "step_number": 4,
            "script_name": "3_gnn.py",
            "description": "GNN processing",
            "status": "SUCCESS",
            "duration_seconds": 45.2,
            "exit_code": 0,
            "peak_memory_mb": 350.0,
        },
    ],
    "performance_summary": {
        "peak_memory_mb": 512.0,
        "successful_steps": 4,
        "failed_steps": 0,
        "warnings": 1,
        "total_steps": 4,
    },
    "environment_info": {"python_version": "3.12.0", "platform": "darwin"},
}
SAMPLE_FAILED_SUMMARY: dict[str, Any] = {
    "start_time": "2024-01-23T10:00:00",
    "end_time": "2024-01-23T10:02:30",
    "total_duration_seconds": 150.0,
    "overall_status": "FAILED",
    "steps": [
        {
            "step_number": 1,
            "script_name": "0_template.py",
            "status": "SUCCESS",
            "duration_seconds": 2.0,
            "exit_code": 0,
            "peak_memory_mb": 100.0,
        },
        {
            "step_number": 2,
            "script_name": "1_setup.py",
            "status": "FAILED",
            "duration_seconds": 30.0,
            "exit_code": 1,
            "peak_memory_mb": 200.0,
            "stderr": "ModuleNotFoundError: No module named 'missing_dependency'",
        },
        {
            "step_number": 3,
            "script_name": "2_tests.py",
            "status": "FAILED",
            "duration_seconds": 5.0,
            "exit_code": 1,
            "peak_memory_mb": 150.0,
            "stderr": "Error: Prerequisite step failed",
        },
    ],
    "performance_summary": {
        "peak_memory_mb": 200.0,
        "successful_steps": 1,
        "failed_steps": 2,
        "warnings": 0,
        "total_steps": 3,
    },
}


class TestIntelligentAnalysisModuleComprehensive:
    """Comprehensive tests for the intelligent_analysis module."""

    @pytest.mark.unit
    def test_module_version(self) -> Any:
        """Test module version tracks the package single source."""
        import tomllib
        from pathlib import Path

        import gnn.intelligent_analysis as intelligent_analysis

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with pyproject.open("rb") as stream:
            expected = tomllib.load(stream)["project"]["version"]
        assert intelligent_analysis.__version__ == expected

    @pytest.mark.unit
    def test_features_dict(self) -> Any:
        """Test FEATURES dictionary contains expected capabilities."""
        import gnn.intelligent_analysis as intelligent_analysis

        expected_features: list[Any] = [
            "pipeline_analysis",
            "failure_root_cause",
            "performance_optimization",
            "llm_powered_insights",
            "executive_reports",
        ]
        for feature in expected_features:
            assert feature in intelligent_analysis.FEATURES

    @pytest.mark.unit
    def test_get_supported_analysis_types(self) -> Any:
        """Test analysis types retrieval."""
        from gnn.intelligent_analysis import get_supported_analysis_types

        types = get_supported_analysis_types()
        assert isinstance(types, list)
        assert len(types) > 0
        expected_types: list[Any] = [
            "failure_analysis",
            "performance_analysis",
            "executive_summary",
        ]
        for t in expected_types:
            assert t in types

    @pytest.mark.unit
    def test_validate_pipeline_summary(self) -> Any:
        """Test pipeline summary validation."""
        from gnn.intelligent_analysis import validate_pipeline_summary

        assert validate_pipeline_summary(SAMPLE_PIPELINE_SUMMARY) is True
        invalid_summary: dict[str, Any] = {"foo": "bar"}
        assert validate_pipeline_summary(invalid_summary) is False


class TestProcessorFunctions:
    """Tests for processor functions."""

    @pytest.mark.unit
    def test_analyze_pipeline_summary(self) -> Any:
        """Test pipeline summary analysis."""
        from gnn.intelligent_analysis.processor import analyze_pipeline_summary

        analysis = analyze_pipeline_summary(SAMPLE_PIPELINE_SUMMARY)
        assert isinstance(analysis, dict)
        assert "overall_status" in analysis
        assert "health_score" in analysis
        assert "failures" in analysis
        assert "warnings" in analysis
        assert analysis["overall_status"] == "SUCCESS_WITH_WARNINGS"
        assert 0 <= analysis["health_score"] <= 100

    @pytest.mark.unit
    def test_analyze_failed_summary(self) -> Any:
        """Test analysis of failed pipeline summary."""
        from gnn.intelligent_analysis.processor import analyze_pipeline_summary

        analysis = analyze_pipeline_summary(SAMPLE_FAILED_SUMMARY)
        assert analysis["overall_status"] == "FAILED"
        assert len(analysis["failures"]) == 2
        assert analysis["health_score"] < 50

    @pytest.mark.unit
    def test_unknown_step_status_is_not_counted_as_success(self) -> None:
        from gnn.intelligent_analysis.processor import analyze_pipeline_summary

        analysis = analyze_pipeline_summary(
            {"overall_status": "UNKNOWN", "steps": [{"status": "UNKNOWN"}]}
        )

        assert analysis["health_score"] == 0.0

    @pytest.mark.unit
    def test_llm_report_requires_all_contract_headings(self) -> None:
        from gnn.intelligent_analysis.processor import _is_complete_llm_report

        assert _is_complete_llm_report("### Executive Summary\nPartial") is False

    @pytest.mark.unit
    def test_identify_bottlenecks(self) -> Any:
        """Test bottleneck identification."""
        from gnn.intelligent_analysis.processor import identify_bottlenecks

        bottlenecks = identify_bottlenecks(SAMPLE_PIPELINE_SUMMARY)
        assert isinstance(bottlenecks, list)
        if bottlenecks:
            assert any((b["step"] == "2_tests.py" for b in bottlenecks))

    @pytest.mark.unit
    def test_identify_bottlenecks_custom_threshold(self) -> Any:
        """Test bottleneck identification with custom threshold."""
        from gnn.intelligent_analysis.processor import identify_bottlenecks

        bottlenecks = identify_bottlenecks(
            SAMPLE_PIPELINE_SUMMARY, threshold_seconds=10.0
        )
        assert isinstance(bottlenecks, list)
        assert len(bottlenecks) >= 1

    @pytest.mark.unit
    def test_extract_failure_context(self) -> Any:
        """Test failure context extraction."""
        from gnn.intelligent_analysis.processor import extract_failure_context

        failures = extract_failure_context(SAMPLE_FAILED_SUMMARY)
        assert isinstance(failures, list)
        assert len(failures) == 2
        first_failure = failures[0]
        assert "step_name" in first_failure
        assert "exit_code" in first_failure
        assert "error_output" in first_failure

    @pytest.mark.unit
    def test_generate_recommendations(self) -> Any:
        """Test recommendation generation."""
        from gnn.intelligent_analysis.processor import (
            analyze_individual_steps,
            analyze_pipeline_summary,
            generate_recommendations,
            identify_bottlenecks,
        )

        analysis = analyze_pipeline_summary(SAMPLE_FAILED_SUMMARY)
        bottlenecks = identify_bottlenecks(SAMPLE_FAILED_SUMMARY)
        step_analyses, flags_by_type = analyze_individual_steps(SAMPLE_FAILED_SUMMARY)
        recommendations = generate_recommendations(analysis, bottlenecks, flags_by_type)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any(
            (
                "critical" in r.lower() or "red" in r.lower() or "failed" in r.lower()
                for r in recommendations
            )
        )

    @pytest.mark.unit
    def test_generate_executive_report(self) -> Any:
        """Test executive report generation."""
        from gnn.intelligent_analysis.processor import (
            analyze_individual_steps,
            analyze_pipeline_summary,
            extract_failure_context,
            generate_executive_report,
            generate_recommendations,
            identify_bottlenecks,
        )

        analysis = analyze_pipeline_summary(SAMPLE_PIPELINE_SUMMARY)
        bottlenecks = identify_bottlenecks(SAMPLE_PIPELINE_SUMMARY)
        failures = extract_failure_context(SAMPLE_PIPELINE_SUMMARY)
        step_analyses, flags_by_type = analyze_individual_steps(SAMPLE_PIPELINE_SUMMARY)
        recommendations = generate_recommendations(analysis, bottlenecks, flags_by_type)
        report = generate_executive_report(
            analysis,
            bottlenecks,
            failures,
            recommendations,
            step_analyses,
            flags_by_type,
        )
        assert isinstance(report, str)
        assert "# Pipeline Intelligent Analysis Report" in report
        assert "Quick Overview" in report or "Executive Summary" in report
        assert "Recommendations" in report
        assert "Per-Step" in report


class TestAnalyzerClass:
    """Tests for the IntelligentAnalyzer class."""

    @pytest.mark.unit
    def test_analyzer_instantiation(self) -> Any:
        """Test IntelligentAnalyzer class instantiation."""
        from gnn.intelligent_analysis import AnalysisContext, IntelligentAnalyzer

        analyzer = IntelligentAnalyzer()
        assert isinstance(analyzer, IntelligentAnalyzer)
        assert analyzer.context is None

    @pytest.mark.unit
    def test_analyzer_with_context(self) -> Any:
        """Test analyzer with context."""
        from gnn.intelligent_analysis import AnalysisContext, IntelligentAnalyzer

        context = AnalysisContext(summary_data=SAMPLE_PIPELINE_SUMMARY)
        analyzer = IntelligentAnalyzer(context=context)
        assert analyzer.context is not None
        assert analyzer.context.overall_status == "SUCCESS_WITH_WARNINGS"

    @pytest.mark.unit
    def test_analyzer_set_context(self) -> Any:
        """Test setting context on analyzer."""
        from gnn.intelligent_analysis import AnalysisContext, IntelligentAnalyzer

        analyzer = IntelligentAnalyzer()
        context = AnalysisContext(summary_data=SAMPLE_PIPELINE_SUMMARY)
        analyzer.set_context(context)
        assert analyzer.context is not None

    @pytest.mark.unit
    def test_analyzer_analyze(self) -> Any:
        """Test full analysis."""
        from gnn.intelligent_analysis import AnalysisContext, IntelligentAnalyzer

        context = AnalysisContext(summary_data=SAMPLE_PIPELINE_SUMMARY)
        analyzer = IntelligentAnalyzer(context=context)
        results = analyzer.analyze()
        assert isinstance(results, dict)
        assert "health_score" in results
        assert "failure_analysis" in results
        assert "performance_analysis" in results
        assert "patterns" in results
        assert "optimizations" in results

    @pytest.mark.unit
    def test_analyzer_no_context_error(self) -> Any:
        """Test that analyze raises error without context."""
        from gnn.intelligent_analysis import IntelligentAnalyzer

        analyzer = IntelligentAnalyzer()
        with pytest.raises(ValueError):
            analyzer.analyze()


class TestAnalyzerFunctions:
    """Tests for analyzer module functions."""

    @pytest.mark.unit
    def test_calculate_health_score(self) -> Any:
        """Test health score calculation."""
        from gnn.intelligent_analysis import calculate_pipeline_health_score

        healthy_score = calculate_pipeline_health_score(SAMPLE_PIPELINE_SUMMARY)
        assert 50 <= healthy_score <= 100
        failed_score = calculate_pipeline_health_score(SAMPLE_FAILED_SUMMARY)
        assert 0 <= failed_score < healthy_score

    @pytest.mark.unit
    def test_classify_failure_severity(self) -> Any:
        """Test failure severity classification."""
        from gnn.intelligent_analysis import classify_failure_severity

        critical_step: dict[str, Any] = {
            "stderr": "MemoryError: Out of memory",
            "exit_code": 137,
        }
        assert classify_failure_severity(critical_step) == "critical"
        major_step: dict[str, Any] = {
            "stderr": "FileNotFoundError: No such file",
            "exit_code": 1,
        }
        assert classify_failure_severity(major_step) == "major"
        minor_step: dict[str, Any] = {"stderr": "Some warning message", "exit_code": 1}
        assert classify_failure_severity(minor_step) == "minor"

    @pytest.mark.unit
    def test_detect_performance_patterns(self) -> Any:
        """Test performance pattern detection."""
        from gnn.intelligent_analysis import detect_performance_patterns

        patterns = detect_performance_patterns(SAMPLE_PIPELINE_SUMMARY)
        assert isinstance(patterns, list)
        for pattern in patterns:
            assert "type" in pattern
            assert "description" in pattern
            assert "severity" in pattern

    @pytest.mark.unit
    def test_generate_optimization_suggestions(self) -> Any:
        """Test optimization suggestion generation."""
        from gnn.intelligent_analysis import generate_optimization_suggestions

        suggestions = generate_optimization_suggestions(SAMPLE_PIPELINE_SUMMARY)
        assert isinstance(suggestions, list)
        for suggestion in suggestions:
            assert "type" in suggestion
            assert "impact" in suggestion
            assert "description" in suggestion


class TestAnalysisContext:
    """Tests for AnalysisContext dataclass."""

    @pytest.mark.unit
    def test_context_creation(self) -> Any:
        """Test AnalysisContext creation."""
        from gnn.intelligent_analysis import AnalysisContext

        context = AnalysisContext(summary_data=SAMPLE_PIPELINE_SUMMARY)
        assert context.summary_data == SAMPLE_PIPELINE_SUMMARY
        assert context.overall_status == "SUCCESS_WITH_WARNINGS"

    @pytest.mark.unit
    def test_context_properties(self) -> Any:
        """Test AnalysisContext properties."""
        from gnn.intelligent_analysis import AnalysisContext

        context = AnalysisContext(summary_data=SAMPLE_PIPELINE_SUMMARY)
        assert context.total_duration == 330.5
        assert len(context.steps) == 4
        assert isinstance(context.performance_summary, dict)

    @pytest.mark.unit
    def test_context_get_steps_methods(self) -> Any:
        """Test step retrieval methods."""
        from gnn.intelligent_analysis import AnalysisContext

        context = AnalysisContext(summary_data=SAMPLE_FAILED_SUMMARY)
        failed_steps = context.get_failed_steps()
        assert len(failed_steps) == 2
        successful_steps = context.get_successful_steps()
        assert len(successful_steps) == 1
        warning_steps = context.get_warning_steps()
        assert len(warning_steps) == 0


class TestIntegration:
    """Integration tests for intelligent_analysis module."""

    @pytest.mark.integration
    def test_full_analysis_workflow(self, isolated_temp_dir: Any) -> Any:
        """Test complete analysis workflow."""
        from gnn.intelligent_analysis.processor import (
            analyze_individual_steps,
            analyze_pipeline_summary,
            extract_failure_context,
            generate_executive_report,
            generate_recommendations,
            identify_bottlenecks,
        )

        analysis = analyze_pipeline_summary(SAMPLE_FAILED_SUMMARY)
        step_analyses, flags_by_type = analyze_individual_steps(SAMPLE_FAILED_SUMMARY)
        bottlenecks = identify_bottlenecks(SAMPLE_FAILED_SUMMARY)
        failures = extract_failure_context(SAMPLE_FAILED_SUMMARY)
        recommendations = generate_recommendations(analysis, bottlenecks, flags_by_type)
        report = generate_executive_report(
            analysis,
            bottlenecks,
            failures,
            recommendations,
            step_analyses,
            flags_by_type,
        )
        report_path = isolated_temp_dir / "analysis_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        assert report_path.exists()
        assert report_path.stat().st_size > 0
        assert "Red Flags" in report or "Yellow Flags" in report or "Per-Step" in report

    @pytest.mark.integration
    def test_process_skip_llm_uses_rule_based_summary(
        self, isolated_temp_dir: Any, monkeypatch: Any
    ) -> Any:
        """Skip-LLM mode must avoid external provider calls."""
        import json
        import logging

        from gnn.intelligent_analysis import processor

        output_dir = isolated_temp_dir / "output"
        summary_dir = output_dir / "00_pipeline_summary"
        summary_dir.mkdir(parents=True)
        with open(summary_dir / "pipeline_execution_summary.json", "w") as f:
            json.dump(SAMPLE_PIPELINE_SUMMARY, f)

        async def fail_on_llm_call(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("LLM analysis should not run when skip_llm is true")

        monkeypatch.setattr(processor, "_run_llm_analysis", fail_on_llm_call)

        assert processor.process_intelligent_analysis(
            isolated_temp_dir / "input",
            output_dir,
            logging.getLogger("test_intelligent_analysis_skip_llm"),
            skip_llm=True,
        )
        assert (
            output_dir
            / "24_intelligent_analysis_output"
            / "intelligent_analysis_report.md"
        ).exists()
        analysis_dir = output_dir / "24_intelligent_analysis_output"
        report = (analysis_dir / "intelligent_analysis_report.md").read_text()
        data = json.loads((analysis_dir / "analysis_data.json").read_text())
        assert "## AI-Powered Analysis" not in report
        assert "## Analysis Summary" in report
        assert "### Root Cause Analysis" in report
        assert "### Optimization Opportunities" in report
        assert data["analysis_source"] == "rule_based"
        assert data["llm_analysis_available"] is False
        assert data["timestamp"] == SAMPLE_PIPELINE_SUMMARY["end_time"]

    @pytest.mark.integration
    def test_check_analysis_tools(self) -> Any:
        """Test analysis tools check."""
        from gnn.intelligent_analysis import check_analysis_tools

        tools = check_analysis_tools()
        assert isinstance(tools, dict)
        if "numpy" in tools:
            assert "available" in tools["numpy"]


def test_module_completeness() -> Any:
    """Test that intelligent_analysis module has all required components."""
    required_components: list[Any] = [
        "__version__",
        "FEATURES",
        "get_module_info",
        "get_supported_analysis_types",
        "validate_pipeline_summary",
        "process_intelligent_analysis",
        "analyze_pipeline_summary",
        "generate_executive_report",
        "IntelligentAnalyzer",
        "AnalysisContext",
    ]
    try:
        import gnn.intelligent_analysis as intelligent_analysis

        for component in required_components:
            assert hasattr(intelligent_analysis, component), (
                f"Missing component: {component}"
            )
    except ImportError:
        raise AssertionError("intelligent_analysis module not available")


@pytest.mark.slow
def test_module_performance() -> Any:
    """Test intelligent_analysis module performance characteristics."""
    import time

    from gnn.intelligent_analysis.processor import analyze_pipeline_summary

    start_time = time.time()
    for _ in range(10):
        analyze_pipeline_summary(SAMPLE_PIPELINE_SUMMARY)
    processing_time = time.time() - start_time
    assert processing_time < 1.0


class TestIntelligentAnalysisMCP:
    def _import_mcp(self) -> Any:
        try:
            from gnn.intelligent_analysis import mcp

            return mcp
        except Exception:
            raise AssertionError("gnn.intelligent_analysis.mcp not importable")

    def test_process_intelligent_analysis_mcp_nonexistent(self, tmp_path: Any) -> Any:
        mcp = self._import_mcp()
        result = mcp.process_intelligent_analysis_mcp(
            str(tmp_path / "nonexistent"), str(tmp_path / "out")
        )
        assert isinstance(result, dict)
        assert "success" in result or "error" in result

    def test_get_module_info_mcp(self) -> Any:
        """Real function is get_intelligent_analysis_module_info_mcp."""
        from gnn.intelligent_analysis.mcp import (
            get_intelligent_analysis_module_info_mcp,
        )

        result = get_intelligent_analysis_module_info_mcp()
        assert isinstance(result, dict)

    def test_get_analysis_capabilities_mcp(self) -> Any:
        """Real API: get_analysis_capabilities_mcp (not get_supported_analysis_types_mcp)."""
        from gnn.intelligent_analysis.mcp import get_analysis_capabilities_mcp

        result = get_analysis_capabilities_mcp()
        assert isinstance(result, dict)


class TestIntelligentAnalysisAnalyzer:
    def test_analysis_context_instantiable(self) -> Any:
        from gnn.intelligent_analysis.analyzer import AnalysisContext

        ctx = AnalysisContext(summary_data={})
        assert isinstance(ctx, AnalysisContext)
        assert ctx.overall_status == "UNKNOWN"

    def test_analysis_context_with_data(self) -> Any:
        from gnn.intelligent_analysis.analyzer import AnalysisContext

        data: dict[str, Any] = {
            "overall_status": "SUCCESS",
            "total_duration_seconds": 1.5,
            "steps": [],
        }
        ctx = AnalysisContext(summary_data=data)
        assert ctx.overall_status == "SUCCESS"
        assert ctx.get_failed_steps() == []
        assert ctx.get_successful_steps() == []

    def test_intelligent_analyzer_instantiable(self) -> Any:
        from gnn.intelligent_analysis.analyzer import IntelligentAnalyzer

        ia = IntelligentAnalyzer()
        assert isinstance(ia, IntelligentAnalyzer)
