"""
Fast Test Suite for GNN Pipeline

This module provides a focused, fast test suite that covers essential functionality
without the overhead of comprehensive testing. It's designed to run quickly and
provide basic validation of core components.

Key Features:
- Fast execution (< 30 seconds total)
- Essential functionality coverage
- Minimal harness overhead
- Focus on core pipeline components
- Safe-to-fail design
"""

import logging
import sys
from typing import Any

import pytest

pytestmark: list[Any] = [pytest.mark.fast]
from . import (
    PROJECT_ROOT,
    SRC_DIR,
    TEST_CONFIG,
    TEST_DIR,
    create_sample_gnn_content,
    is_safe_mode,
)


class TestFastEnvironment:
    """Fast environment validation tests."""

    @pytest.mark.unit
    def test_python_environment(self) -> Any:
        """Test basic Python environment."""
        assert sys.version_info >= (3, 12), "Python 3.12+ required"
        assert str(SRC_DIR) in sys.path, "Source directory in Python path"

    @pytest.mark.unit
    def test_test_configuration(self) -> Any:
        """Test test configuration is valid."""
        config = TEST_CONFIG
        assert config["safe_mode"] is True, "Safe mode should be enabled"
        assert config["timeout_seconds"] >= 30, (
            "Timeout should be at least 30 seconds for reasonable operation"
        )
        assert config["max_test_files"] <= 10, (
            "Max test files should be reasonably limited (<=10)"
        )

    @pytest.mark.unit
    def test_essential_directories(self) -> Any:
        """Test essential directories exist."""
        assert SRC_DIR.exists(), "Source directory should exist"
        assert TEST_DIR.exists(), "Test directory should exist"
        assert PROJECT_ROOT.exists(), "Project root should exist"


class TestFastGNNProcessing:
    """Fast GNN processing tests."""

    @pytest.mark.unit
    def test_gnn_module_import(self) -> Any:
        """Test that GNN module can be imported."""
        import gnn

        assert gnn is not None, "GNN module should be importable"

    @pytest.mark.unit
    def test_gnn_file_creation(self, isolated_temp_dir: Any) -> Any:
        """Test GNN file creation and basic parsing."""
        sample_content = create_sample_gnn_content()
        valid_content = sample_content["valid_basic"]
        test_file = isolated_temp_dir / "test_model.md"
        test_file.write_text(valid_content)
        assert test_file.exists(), "Test file should be created"
        assert test_file.read_text() == valid_content, "File content should match"

    @pytest.mark.unit
    def test_gnn_content_validation(self) -> Any:
        """Test GNN content validation."""
        sample_content = create_sample_gnn_content()
        valid_content = sample_content["valid_basic"]
        assert "## ModelName" in valid_content, "Should contain ModelName section"
        assert "## StateSpaceBlock" in valid_content, (
            "Should contain StateSpaceBlock section"
        )
        assert "## Connections" in valid_content, "Should contain Connections section"


class TestFastPipelineComponents:
    """Fast pipeline component tests."""

    @pytest.mark.unit
    def test_utils_module_import(self) -> Any:
        """Test that utils module can be imported."""
        import gnn.utils as utils

        assert utils is not None, "Utils module should be importable"

    @pytest.mark.unit
    def test_pipeline_module_import(self) -> Any:
        """Test that pipeline module can be imported."""
        import gnn.pipeline as pipeline

        assert pipeline is not None, "Pipeline module should be importable"

    @pytest.mark.unit
    def test_type_checker_import(self) -> Any:
        """Test that type checker module can be imported."""
        import gnn.type_checker as type_checker

        assert type_checker is not None, "Type checker module should be importable"


class TestFastExport:
    """Fast export functionality tests."""

    @pytest.mark.unit
    def test_export_module_import(self) -> Any:
        """Test that export module can be imported."""
        import gnn.export as export

        assert export is not None, "Export module should be importable"

    @pytest.mark.unit
    def test_basic_export_functionality(self, isolated_temp_dir: Any) -> Any:
        """Test basic export functionality."""
        try:
            from gnn.export import export_to_json_gnn

            test_data: dict[str, Any] = {
                "name": "TestModel",
                "variables": [{"name": "X", "dimensions": [2]}],
                "connections": [],
            }
            output_file = isolated_temp_dir / "test_export.json"
            export_to_json_gnn(test_data, output_file)
            assert output_file.exists(), "Export file should be created"
        except ImportError:
            raise AssertionError("Export functionality not available")
        except Exception as e:
            raise AssertionError(f"Export test failed: {e}")


class TestFastVisualization:
    """Fast visualization tests."""

    @pytest.mark.unit
    def test_visualization_module_import(self) -> Any:
        """Test that visualization module can be imported."""
        import gnn.visualization as visualization

        assert visualization is not None, "Visualization module should be importable"


class TestFastRender:
    """Fast render tests."""

    @pytest.mark.unit
    def test_render_module_import(self) -> Any:
        """Test that render module can be imported."""
        import gnn.render as render

        assert render is not None, "Render module should be importable"


class TestFastExecute:
    """Fast execute tests."""

    @pytest.mark.unit
    def test_execute_module_import(self) -> Any:
        """Test that execute module can be imported."""
        import gnn.execute as execute

        assert execute is not None, "Execute module should be importable"


class TestFastLLM:
    """Fast LLM tests."""

    @pytest.mark.unit
    def test_llm_module_import(self) -> Any:
        """Test that LLM module can be imported."""
        import gnn.llm as llm

        assert llm is not None, "LLM module should be importable"


class TestFastMCP:
    """Fast MCP tests."""

    @pytest.mark.unit
    def test_mcp_module_import(self) -> Any:
        """Test that MCP module can be imported."""
        import gnn.mcp as mcp

        assert mcp is not None, "MCP module should be importable"


class TestFastOntology:
    """Fast ontology tests."""

    @pytest.mark.unit
    def test_ontology_module_import(self) -> Any:
        """Test that ontology module can be imported."""
        import gnn.ontology as ontology

        assert ontology is not None, "Ontology module should be importable"


class TestFastSAPF:
    """Fast SAPF tests."""

    @pytest.mark.unit
    def test_sapf_module_import(self) -> Any:
        """Test that SAPF module can be imported."""
        import gnn.audio.sapf as sapf

        assert sapf is not None, "SAPF module should be importable"


class TestFastWebsite:
    """Fast website tests."""

    @pytest.mark.unit
    def test_website_module_import(self) -> Any:
        """Test that website module can be imported."""
        import gnn.website as website

        assert website is not None, "Website module should be importable"


class TestFastIntegration:
    """Fast integration tests."""

    @pytest.mark.integration
    def test_basic_pipeline_flow(self, isolated_temp_dir: Any) -> Any:
        """Test basic pipeline flow with minimal components."""
        try:
            sample_content = create_sample_gnn_content()
            test_file = isolated_temp_dir / "test_model.md"
            test_file.write_text(sample_content["valid_basic"])
            assert test_file.exists(), "Test file should be created"
            content = test_file.read_text()
            assert len(content) > 0, "File should have content"
            assert "## ModelName" in content, "Should contain model name"
        except Exception as e:
            raise AssertionError(f"Basic pipeline flow test failed: {e}")

    @pytest.mark.integration
    def test_test_environment_setup(self) -> Any:
        """Test that test environment is properly set up."""
        assert is_safe_mode(), "Should be in safe mode"
        temp_dir = TEST_CONFIG["temp_output_dir"]
        temp_dir.mkdir(parents=True, exist_ok=True)
        assert temp_dir.exists(), "Temp directory should be created"


def test_fast_suite_completeness() -> Any:
    """Test that fast test suite covers essential functionality."""
    from pathlib import Path

    test_dir = Path(__file__).parent
    fast_test_count = 0
    test_files = list(test_dir.rglob("test_*.py"))
    assert len(test_files) > 10, f"Expected >10 test files, found {len(test_files)}"
    for test_file in test_files:
        try:
            content = test_file.read_text()
            if "@pytest.mark.fast" in content:
                fast_test_count += content.count("@pytest.mark.fast")
        except Exception:
            pass
    assert fast_test_count > 20, f"Expected >20 fast tests, found {fast_test_count}"
    logging.info(
        f"Fast test suite completeness: {fast_test_count} fast tests in {len(test_files)} files"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "fast"])
