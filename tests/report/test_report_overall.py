"""
Test Report Overall Tests

This file contains comprehensive tests for the report module functionality.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestReportModuleComprehensive:
    """Comprehensive tests for the report module."""

    @pytest.mark.unit
    def test_report_module_imports(self) -> None:
        """Test that report module can be imported."""
        import gnn.report as report

        assert hasattr(report, "__version__")
        assert hasattr(report, "ReportGenerator")
        assert hasattr(report, "ReportFormatter")
        assert hasattr(report, "get_module_info")

    @pytest.mark.unit
    def test_report_generator_instantiation(self) -> None:
        """Test ReportGenerator class instantiation."""
        from gnn.report import ReportGenerator

        generator = ReportGenerator()
        assert generator is not None
        assert hasattr(generator, "generate_report")
        assert hasattr(generator, "format_report")

    @pytest.mark.unit
    def test_report_formatter_instantiation(self) -> None:
        """Test ReportFormatter class instantiation."""
        from gnn.report import ReportFormatter

        formatter = ReportFormatter()
        assert formatter is not None
        assert hasattr(formatter, "format_markdown")
        assert hasattr(formatter, "format_html")

    @pytest.mark.unit
    def test_report_module_info(self) -> None:
        """Test report module information retrieval."""
        from gnn.report import get_module_info

        info = get_module_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "description" in info
        assert "report_formats" in info

    @pytest.mark.unit
    def test_report_formats(self) -> None:
        """Test report formats retrieval."""
        from gnn.report import get_supported_formats

        formats = get_supported_formats()
        assert isinstance(formats, list)
        assert len(formats) > 0
        expected_formats: list[Any] = ["markdown", "html", "json", "pdf"]
        for fmt in expected_formats:
            assert fmt in formats


class TestReportFunctionality:
    """Tests for report functionality."""

    @pytest.mark.unit
    def test_report_generation(self, comprehensive_test_data: Any) -> None:
        """Test report generation functionality."""
        from gnn.report import ReportGenerator

        generator = ReportGenerator()
        report_data = comprehensive_test_data.get("report_data", {})
        result = generator.generate_report(report_data)
        assert result is not None

    @pytest.mark.unit
    def test_report_formatting(self) -> None:
        """Test report formatting functionality."""
        from gnn.report import ReportFormatter

        formatter = ReportFormatter()
        content = "Test report content"
        result = formatter.format_markdown(content)
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_report_validation(self) -> None:
        """Test report validation functionality."""
        from gnn.report import validate_report

        result = validate_report({"content": "test report content"})
        assert isinstance(result, bool)


class TestReportIntegration:
    """Integration tests for report module."""

    @pytest.mark.integration
    def test_report_pipeline_integration(
        self, sample_gnn_files: Any, isolated_temp_dir: Any
    ) -> None:
        """Test report module integration with pipeline."""
        from gnn.report import ReportGenerator

        generator = ReportGenerator()
        gnn_file = list(sample_gnn_files.values())[0]
        with open(gnn_file) as f:
            gnn_content = f.read()
        result = generator.generate_report({"gnn_content": gnn_content})
        assert result is not None

    @pytest.mark.integration
    def test_report_mcp_integration(self) -> None:
        """Test report MCP integration."""
        from gnn.report.mcp import register_tools

        assert callable(register_tools)


def test_report_module_completeness() -> None:
    """Test that report module has all required components."""
    required_components: list[Any] = [
        "ReportGenerator",
        "ReportFormatter",
        "get_module_info",
        "get_supported_formats",
        "validate_report",
    ]
    try:
        import gnn.report as report

        for component in required_components:
            assert hasattr(report, component), f"Missing component: {component}"
    except ImportError:
        raise AssertionError("Report module not available")


@pytest.mark.slow
def test_report_module_performance() -> None:
    """Test report module performance characteristics."""
    import time

    from gnn.report import ReportGenerator

    generator = ReportGenerator()
    start_time = time.time()
    generator.generate_report({"test": "data"})
    processing_time = time.time() - start_time
    assert processing_time < 5.0
