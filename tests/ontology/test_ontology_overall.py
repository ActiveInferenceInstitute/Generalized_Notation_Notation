"""
Test Ontology Overall Tests

This file contains comprehensive tests for the ontology module functionality.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestOntologyModuleComprehensive:
    """Comprehensive tests for the ontology module."""

    @pytest.mark.unit
    def test_ontology_module_imports(self) -> Any:
        """Test that ontology module can be imported."""
        import gnn.ontology as ontology

        assert hasattr(ontology, "__version__")
        assert hasattr(ontology, "OntologyProcessor")
        assert hasattr(ontology, "OntologyValidator")
        assert hasattr(ontology, "get_module_info")

    @pytest.mark.unit
    def test_ontology_processor_instantiation(self) -> Any:
        """Test OntologyProcessor class instantiation."""
        from gnn.ontology import OntologyProcessor

        processor = OntologyProcessor()
        assert processor is not None
        assert hasattr(processor, "process_ontology")
        assert hasattr(processor, "validate_terms")

    @pytest.mark.unit
    def test_ontology_validator_instantiation(self) -> Any:
        """Test OntologyValidator class instantiation."""
        from gnn.ontology import OntologyValidator

        validator = OntologyValidator()
        assert validator is not None
        assert hasattr(validator, "validate_ontology")
        assert hasattr(validator, "check_consistency")

    @pytest.mark.unit
    def test_ontology_module_info(self) -> Any:
        """Test ontology module information retrieval."""
        from gnn.ontology import get_module_info

        info = get_module_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "description" in info
        assert "ontology_types" in info

    @pytest.mark.unit
    def test_ontology_processing_options(self) -> Any:
        """Test ontology processing options retrieval."""
        from gnn.ontology import get_ontology_processing_options

        options = get_ontology_processing_options()
        assert isinstance(options, dict)
        assert "validation_levels" in options
        assert "output_formats" in options


class TestOntologyFunctionality:
    """Tests for ontology functionality."""

    @pytest.mark.unit
    def test_ontology_processing(self, comprehensive_test_data: Any) -> Any:
        """Test ontology processing functionality."""
        from gnn.ontology import OntologyProcessor

        processor = OntologyProcessor()
        ontology_data = comprehensive_test_data.get("ontology_data", {})
        result = processor.process_ontology(ontology_data)
        assert result is not None

    @pytest.mark.unit
    def test_ontology_validation(self) -> Any:
        """Test ontology validation functionality."""
        from gnn.ontology import OntologyValidator

        validator = OntologyValidator()
        ontology_content = "test ontology content"
        result = validator.validate_ontology(ontology_content)
        assert isinstance(result, bool)

    @pytest.mark.unit
    def test_term_validation(self) -> Any:
        """Test term validation functionality."""
        from gnn.ontology import validate_ontology_terms

        result = validate_ontology_terms("test term")
        assert isinstance(result, bool)


class TestOntologyIntegration:
    """Integration tests for ontology module."""

    @pytest.mark.integration
    def test_ontology_pipeline_integration(
        self, sample_gnn_files: Any, isolated_temp_dir: Any
    ) -> Any:
        """Test ontology module integration with pipeline."""
        from gnn.ontology import OntologyProcessor

        processor = OntologyProcessor()
        gnn_file = list(sample_gnn_files.values())[0]
        with open(gnn_file) as f:
            gnn_content = f.read()
        result = processor.process_ontology({"content": gnn_content})
        assert result is not None

    @pytest.mark.integration
    def test_ontology_mcp_integration(self) -> Any:
        """Test ontology MCP integration."""
        from gnn.ontology.mcp import register_tools

        assert callable(register_tools)


def test_ontology_module_completeness() -> Any:
    """Test that ontology module has all required components."""
    required_components: list[Any] = [
        "OntologyProcessor",
        "OntologyValidator",
        "get_module_info",
        "get_ontology_processing_options",
        "validate_ontology_terms",
    ]
    try:
        import gnn.ontology as ontology

        for component in required_components:
            assert hasattr(ontology, component), f"Missing component: {component}"
    except ImportError:
        raise AssertionError("Ontology module not available")


@pytest.mark.slow
def test_ontology_module_performance() -> Any:
    """Test ontology module performance characteristics."""
    import time

    from gnn.ontology import OntologyProcessor

    processor = OntologyProcessor()
    start_time = time.time()
    processor.process_ontology({"test": "data"})
    processing_time = time.time() - start_time
    assert processing_time < 10.0


class TestOntologyMCP:
    def test_module_importable(self) -> Any:
        from gnn.ontology import mcp

    def test_process_ontology_mcp_nonexistent(self, tmp_path: Any) -> Any:
        from gnn.ontology.mcp import process_ontology_mcp

        result = process_ontology_mcp(
            str(tmp_path / "nonexistent"), str(tmp_path / "out")
        )
        assert isinstance(result, dict)
        assert "success" in result or "error" in result

    def test_validate_ontology_terms_mcp_empty(self) -> Any:
        from gnn.ontology.mcp import validate_ontology_terms_mcp

        result = validate_ontology_terms_mcp([])
        assert isinstance(result, dict)

    def test_validate_ontology_terms_mcp_string(self) -> Any:
        from gnn.ontology.mcp import validate_ontology_terms_mcp

        result = validate_ontology_terms_mcp("LikelihoodMatrix")
        assert isinstance(result, dict)


class TestOntologyUtils:
    def test_module_importable(self) -> Any:
        from gnn.ontology import utils

    def test_get_module_info(self) -> Any:
        from gnn.ontology.utils import get_module_info

        result = get_module_info()
        assert isinstance(result, dict)
        assert "module_name" in result

    def test_get_ontology_processing_options(self) -> Any:
        from gnn.ontology.utils import get_ontology_processing_options

        result = get_ontology_processing_options()
        assert isinstance(result, dict)
