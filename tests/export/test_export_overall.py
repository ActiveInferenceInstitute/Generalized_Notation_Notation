"""
Test Export Overall Tests

This file contains comprehensive tests for the export module functionality.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestExportModuleComprehensive:
    """Comprehensive tests for the export module."""

    @pytest.mark.unit
    def test_export_module_imports(self) -> None:
        """Test that export module can be imported."""
        import gnn.export as export

        assert hasattr(export, "__version__")
        assert hasattr(export, "Exporter")
        assert hasattr(export, "MultiFormatExporter")
        assert hasattr(export, "get_supported_formats")

    @pytest.mark.unit
    def test_exporter_instantiation(self) -> None:
        """Test Exporter class instantiation."""
        from gnn.export import Exporter

        exporter = Exporter()
        assert exporter is not None
        assert hasattr(exporter, "export_gnn_model")
        assert hasattr(exporter, "validate_format")

    @pytest.mark.unit
    def test_multi_format_exporter_instantiation(self) -> None:
        """Test MultiFormatExporter class instantiation."""
        from gnn.export import MultiFormatExporter

        exporter = MultiFormatExporter()
        assert exporter is not None
        assert hasattr(exporter, "export_to_multiple_formats")
        assert hasattr(exporter, "get_supported_formats")

    @pytest.mark.unit
    def test_export_module_info(self) -> None:
        """Test export module information retrieval."""
        from gnn.export import get_module_info

        info = get_module_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "description" in info
        assert "supported_formats" in info

    @pytest.mark.unit
    def test_supported_formats(self) -> None:
        """Test supported formats retrieval."""
        from gnn.export import get_supported_formats

        formats = get_supported_formats()
        assert isinstance(formats, list)
        assert len(formats) > 0
        expected_formats: list[Any] = ["json", "xml", "graphml", "gexf", "pickle"]
        for fmt in expected_formats:
            assert fmt in formats


class TestExportFunctionality:
    """Tests for export functionality."""

    @pytest.mark.unit
    def test_export_gnn_model(self, comprehensive_test_data: Any) -> None:
        """Test GNN model export functionality."""
        from gnn.export import Exporter

        exporter = Exporter()
        gnn_data = comprehensive_test_data.get("gnn_content", "test content")
        result = exporter.export_gnn_model(gnn_data, "json")
        assert result is not None

    @pytest.mark.unit
    def test_export_format_validation(self) -> None:
        """Test export format validation."""
        from gnn.export import validate_export_format

        result = validate_export_format("json")
        assert isinstance(result, bool)
        assert result is True
        result = validate_export_format("invalid_format")
        assert isinstance(result, bool)
        assert result is False

    @pytest.mark.unit
    def test_multi_format_export(self, comprehensive_test_data: Any) -> None:
        """Test multi-format export functionality."""
        from gnn.export import MultiFormatExporter

        exporter = MultiFormatExporter()
        gnn_data = comprehensive_test_data.get("gnn_content", "test content")
        formats: list[Any] = ["json", "xml"]
        result = exporter.export_to_multiple_formats(gnn_data, formats)
        assert result is not None
        assert isinstance(result, dict)


class TestExportIntegration:
    """Integration tests for export module."""

    @pytest.mark.integration
    def test_export_pipeline_integration(
        self, sample_gnn_files: Any, isolated_temp_dir: Any
    ) -> None:
        """Test export module integration with pipeline."""
        from gnn.export import Exporter

        exporter = Exporter()
        gnn_file = list(sample_gnn_files.values())[0]
        with open(gnn_file) as f:
            gnn_content = f.read()
        result = exporter.export_gnn_model(gnn_content, "json")
        assert result is not None

    @pytest.mark.integration
    def test_export_mcp_integration(self) -> None:
        """Test export MCP integration."""
        from gnn.export.mcp import register_tools

        assert callable(register_tools)


def test_export_module_completeness() -> None:
    """Test that export module has all required components."""
    required_components: list[Any] = [
        "Exporter",
        "MultiFormatExporter",
        "get_module_info",
        "get_supported_formats",
        "validate_export_format",
    ]
    try:
        import gnn.export as export

        for component in required_components:
            assert hasattr(export, component), f"Missing component: {component}"
    except ImportError:
        raise AssertionError("Export module not available")


@pytest.mark.slow
def test_export_module_performance() -> None:
    """Test export module performance characteristics."""
    import time

    from gnn.export import Exporter

    exporter = Exporter()
    start_time = time.time()
    exporter.export_gnn_model("test content", "json")
    processing_time = time.time() - start_time
    assert processing_time < 5.0


class TestExportUtils:
    """Smoke tests for export.utils sub-module."""

    def test_module_importable(self) -> Any:
        from gnn.export import utils

    def test_get_module_info_returns_dict(self) -> Any:
        from gnn.export.utils import get_module_info

        result = get_module_info()
        assert isinstance(result, dict)

    def test_get_supported_formats_returns_dict(self) -> Any:
        from gnn.export.utils import get_supported_formats

        result = get_supported_formats()
        assert isinstance(result, dict)
        assert len(result) > 0
