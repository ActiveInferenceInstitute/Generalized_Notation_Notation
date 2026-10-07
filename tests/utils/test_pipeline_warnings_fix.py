#!/usr/bin/env python3
"""
Pipeline Validation and Prerequisites Tests

This module validates the pipeline's prerequisite checking and output directory management.
Tests verify that:

1. Output directories are correctly named for each pipeline step
2. Prerequisite validation detects required dependencies
3. Nested directory structures (previous patterns) are detected and warned about
4. Missing prerequisites generate appropriate warnings without failing execution
5. The get_output_dir_for_script function prevents unintended directory nesting

All tests use real implementations of validate_step_prerequisites() and
get_output_dir_for_script() using real functions and filesystem operations.
Tests use temporary directories for isolation and automatic cleanup.
"""

from typing import Any

import pytest

pytestmark = pytest.mark.pipeline
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.pipeline.config import get_output_dir_for_script
from gnn.utils.pipeline_orchestration.pipeline_validator import (
    check_pipeline_readiness,
    validate_step_prerequisites,
)


@dataclass
class _PipelineArgs:
    """Simulated-free replacement for pipeline arguments."""

    output_dir: Path
    target_dir: Path | None = None
    verbose: bool = False


class TestPipelineWarningsFix:
    """Test suite for pipeline warning fixes."""

    def test_prerequisite_validation_correct_directories(self, tmp_path: Any) -> None:
        """Test that prerequisite validation uses correct output directory names."""
        # Setup
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create the correct prerequisite directories
        gnn_output_dir = output_dir / "3_gnn_output"
        gnn_output_dir.mkdir()

        # Create a parsed GNN file (validator checks for *_parsed.json files)
        # Note: Parsed files in model-specific subdirectories is the correct/intended structure
        model_dir = gnn_output_dir / "test_model"
        model_dir.mkdir()
        parsed_file = model_dir / "test_model_parsed.json"
        parsed_file.write_text('{"ModelName": "test"}')

        type_checker_output_dir = output_dir / "5_type_checker_output"
        type_checker_output_dir.mkdir()

        # Create real args object
        args = _PipelineArgs(output_dir=output_dir)

        logger = logging.getLogger("test")

        # Test validation for step that depends on 3_gnn.py
        result = validate_step_prerequisites("5_type_checker.py", args, logger)

        # Should pass without warnings since 3_gnn_output exists with parsed files
        assert result["passed"], "Prerequisite validation should pass"
        assert len(result["warnings"]) == 0, (
            f"Should have no warnings, got: {result['warnings']}"
        )

    def test_prerequisite_validation_with_nested_directories(
        self, tmp_path: Any
    ) -> None:
        """Test that prerequisite validation detects and warns about nested directories."""
        # Setup
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create the expected output directory (which will be empty/just exist)
        expected_gnn_dir = output_dir / "3_gnn_output"
        expected_gnn_dir.mkdir()

        # Create nested directory structure (previous pattern)
        nested_gnn_dir = expected_gnn_dir / "3_gnn_output"
        nested_gnn_dir.mkdir()

        # Create a parsed GNN file in the nested location
        parsed_file = nested_gnn_dir / "test_model_parsed.json"
        parsed_file.write_text('{"ModelName": "test"}')

        # Create real args object
        args = _PipelineArgs(output_dir=output_dir)

        logger = logging.getLogger("test")

        # Test validation for step that depends on 3_gnn.py
        result = validate_step_prerequisites("5_type_checker.py", args, logger)

        # Should pass with a warning about nested structure
        assert result["passed"], "Prerequisite validation should pass"
        assert len(result["warnings"]) > 0, "Should have warning about nested structure"
        assert any("nested" in w.lower() for w in result["warnings"]), (
            f"Should warn about nested directory, got warnings: {result['warnings']}"
        )

    def test_prerequisite_validation_missing_directory(self, tmp_path: Any) -> None:
        """Test that prerequisite validation warns about missing directories."""
        # Setup
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Don't create the prerequisite directory

        # Create real args object
        args = _PipelineArgs(output_dir=output_dir)

        logger = logging.getLogger("test")

        # Test validation for step that depends on 3_gnn.py
        result = validate_step_prerequisites("5_type_checker.py", args, logger)

        # Missing prerequisite outputs should fail the prerequisite check.
        assert result["passed"] is False, "Prerequisite validation should fail"
        assert len(result["warnings"]) > 0, (
            "Should have warning about missing directory"
        )
        assert len(result["errors"]) > 0, "Should record missing output as an error"
        assert any("missing" in w.lower() for w in result["warnings"]), (
            "Should warn about missing prerequisite directory"
        )

    def test_get_output_dir_prevents_nesting(self, tmp_path: Any) -> None:
        """Test that get_output_dir_for_script prevents nested directories."""
        base_output_dir = tmp_path / "output"
        base_output_dir.mkdir()

        # First call should create the correct directory
        result1 = get_output_dir_for_script("10_ontology.py", base_output_dir)
        expected = base_output_dir / "10_ontology_output"
        assert result1 == expected, f"Expected {expected}, got {result1}"

        # Second call with already-nested directory should NOT create double nesting
        result2 = get_output_dir_for_script("10_ontology.py", result1)
        # Should return the same directory, not result1 / "10_ontology_output"
        assert result2 == expected, (
            f"Should prevent nesting, expected {expected}, got {result2}"
        )

    def test_get_output_dir_all_steps(self, tmp_path: Any) -> None:
        """Test that all pipeline steps get correct output directories."""
        base_output_dir = tmp_path / "output"
        base_output_dir.mkdir()

        # Test all 24 pipeline steps
        steps: list[Any] = [
            ("0_template.py", "0_template_output"),
            ("1_setup.py", "1_setup_output"),
            ("2_tests.py", "2_tests_output"),
            ("3_gnn.py", "3_gnn_output"),
            ("4_model_registry.py", "4_model_registry_output"),
            ("5_type_checker.py", "5_type_checker_output"),
            ("6_validation.py", "6_validation_output"),
            ("7_export.py", "7_export_output"),
            ("8_visualization.py", "8_visualization_output"),
            ("9_advanced_viz.py", "9_advanced_viz_output"),
            ("10_ontology.py", "10_ontology_output"),
            ("11_render.py", "11_render_output"),
            ("12_execute.py", "12_execute_output"),
            ("13_llm.py", "13_llm_output"),
            ("14_ml_integration.py", "14_ml_integration_output"),
            ("15_audio.py", "15_audio_output"),
            ("16_analysis.py", "16_analysis_output"),
            ("17_integration.py", "17_integration_output"),
            ("18_security.py", "18_security_output"),
            ("19_research.py", "19_research_output"),
            ("20_website.py", "20_website_output"),
            ("21_mcp.py", "21_mcp_output"),
            ("22_gui.py", "22_gui_output"),
            ("23_report.py", "23_report_output"),
        ]

        for script_name, expected_dir_name in steps:
            result = get_output_dir_for_script(script_name, base_output_dir)
            expected = base_output_dir / expected_dir_name
            assert result == expected, (
                f"Script {script_name} should map to {expected_dir_name}, got {result.name}"
            )

    def test_prerequisite_chain_validation(self, tmp_path: Any) -> None:
        """Test validation of prerequisite chains."""
        # Setup
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create prerequisite directories
        gnn_output = output_dir / "3_gnn_output"
        gnn_output.mkdir()
        model_output = gnn_output / "test_model"
        model_output.mkdir()
        (model_output / "test_model_parsed.json").write_text('{"ModelName": "test"}')
        (output_dir / "5_type_checker_output").mkdir()
        render_output = output_dir / "11_render_output"
        render_output.mkdir()
        (render_output / "test_render.py").write_text("print('rendered')\n")

        # Create real args object
        args = _PipelineArgs(output_dir=output_dir)

        logger = logging.getLogger("test")

        # Test validation for step that depends on 11_render.py (which depends on 3_gnn.py)
        result = validate_step_prerequisites("12_execute.py", args, logger)

        # Should pass without warnings since all prerequisites exist
        assert result["passed"], "Prerequisite validation should pass"

    def test_prerequisite_missing_execute_output_fails(self, tmp_path: Any) -> None:
        """Test validation fails when 12_execute prerequisite is missing."""
        output_dir = tmp_path / "output_missing_exec"
        output_dir.mkdir()
        gnn_output = output_dir / "3_gnn_output" / "test_model"
        gnn_output.mkdir(parents=True)
        (gnn_output / "test_model_parsed.json").write_text('{"ModelName": "test"}')
        (output_dir / "7_export_output").mkdir()

        args = _PipelineArgs(output_dir=output_dir)
        logger = logging.getLogger("test")
        result = validate_step_prerequisites("16_analysis.py", args, logger)
        # Empirical analysis also requires current execution results. Existing
        # parse/export directories cannot stand in for the missing producer.
        assert result["passed"] is False
        assert any("12_execute_output" in error for error in result["errors"])

    def test_readiness_uses_registered_gnn_extensions(self, tmp_path: Any) -> None:
        """Readiness should discover all parser-registered input extensions."""
        target_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        target_dir.mkdir()
        (target_dir / "model.json").write_text('{"model_name": "JsonModel"}')
        args = _PipelineArgs(output_dir=output_dir, target_dir=target_dir)

        result = check_pipeline_readiness([("5_type_checker.py", "Type check")], args)

        assert result["ready"] is True
        assert not result["blocking_issues"]

    # Note: If checking for transitive dependencies is outside scope,
    # there may still be a warning about 3_gnn.py not being a direct prerequisite

    def test_get_output_dir_with_path_object(self, tmp_path: Any) -> None:
        """Test get_output_dir_for_script with Path object."""
        base_output_dir = Path(tmp_path) / "output"
        base_output_dir.mkdir(parents=True)

        result = get_output_dir_for_script("3_gnn.py", base_output_dir)
        expected = base_output_dir / "3_gnn_output"

        assert result == expected
        assert isinstance(result, Path)

    def test_get_output_dir_without_py_extension(self, tmp_path: Any) -> None:
        """Test get_output_dir_for_script without .py extension."""
        base_output_dir = tmp_path / "output"
        base_output_dir.mkdir()

        result = get_output_dir_for_script("3_gnn", base_output_dir)
        expected = base_output_dir / "3_gnn_output"

        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
