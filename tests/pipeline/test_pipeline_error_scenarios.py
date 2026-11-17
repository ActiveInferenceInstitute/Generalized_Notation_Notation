"""
Pipeline Error Scenario Tests

This module tests the pipeline's behavior under various error conditions,
dependency failures, and edge cases to ensure robust operation.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.pipeline, pytest.mark.integration]
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class TestDependencyErrorScenarios:
    """Test pipeline behavior with missing dependencies."""

    @pytest.mark.unit
    def test_pymdp_graceful_degradation_with_real_error(self) -> Any:
        """PyMDP simulator handles malformed input with a real, structured error.

        Requires pymdp 1.0.0+ (JAX-backed). Skip when unavailable — the
        graceful degradation of the import itself is exercised in
        execute.pymdp.package_detector, not here.
        """
        import importlib.util

        if (
            importlib.util.find_spec("pymdp") is None
            or importlib.util.find_spec("jax") is None
        ):
            raise AssertionError(
                "pymdp + jax required for PyMDPSimulation instantiation"
            )
        from gnn.execute.pymdp.pymdp_simulation import PyMDPSimulation

        with pytest.raises(
            ValueError, match="requires explicit initialparameterization"
        ):
            PyMDPSimulation()

    @pytest.mark.unit
    def test_missing_discopy_graceful_degradation(self) -> Any:
        """Test that missing DisCoPy is handled gracefully."""
        from gnn.execute.discopy_translator_module.translator import (
            DISCOPY_AVAILABLE,
            JAX_FULLY_OPERATIONAL,
            gnn_file_to_discopy_diagram,
        )

        test_gnn_data: dict[str, Any] = {
            "Variables": {"test_var": {"type": "state", "dimensions": [3]}}
        }
        success, message, diagram = gnn_file_to_discopy_diagram(test_gnn_data)
        if not DISCOPY_AVAILABLE:
            assert not success
            assert "DisCoPy library not installed" in message
            assert diagram is None
        else:
            assert isinstance(success, bool)
            assert isinstance(message, str)

    @pytest.mark.unit
    def test_matplotlib_rendering_fallbacks(self) -> Any:
        """Test visualization fallbacks when matplotlib has issues."""
        import matplotlib.pyplot as plt

        from gnn.visualization.processor import save_plot_safely

        with tempfile.TemporaryDirectory() as temp_dir:
            test_plot_path = Path(temp_dir) / "test_plot.png"
            plt.figure()
            plt.plot([1, 2, 3], [1, 4, 9])
            assert save_plot_safely(test_plot_path, dpi=999999)
            assert test_plot_path.exists()
            plt.close()


class TestFileOperationErrorScenarios:
    """Test pipeline behavior with file operation errors."""

    @pytest.mark.unit
    def test_missing_input_directory(self, temp_directories: Any) -> Any:
        """Missing visualization input is a warning, not a hard crash."""
        non_existent_dir = temp_directories["temp_dir"] / "non_existent"
        from gnn.visualization import process_visualization

        result = process_visualization(
            target_dir=non_existent_dir,
            output_dir=temp_directories["output_dir"],
            verbose=True,
        )

        assert result == 2
        summary_path = temp_directories["output_dir"] / "visualization_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["success"] is False
        assert summary["processed_files"] == 0
        assert summary["total_visualizations"] == 0
        assert summary["warnings"]

    @pytest.mark.unit
    def test_readonly_output_directory(self, temp_directories: Any) -> Any:
        """A read-only output directory never receives partial artifacts.

        Processing must either raise ``PermissionError`` (graceful refusal) or
        complete without writing into the read-only directory.
        """
        readonly_dir = temp_directories["temp_dir"] / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)
        try:
            from gnn import process_gnn_directory

            try:
                result = process_gnn_directory(
                    temp_directories["input_dir"],
                    output_dir=readonly_dir,
                )
            except PermissionError:
                return  # graceful refusal is an acceptable outcome
            assert isinstance(result, dict)
            assert "status" in result
            assert list(readonly_dir.iterdir()) == [], (
                "read-only output directory received writes"
            )
        finally:
            try:
                readonly_dir.chmod(0o755)
            except Exception:
                pass

    @pytest.mark.unit
    def test_corrupted_gnn_file_handling(self, temp_directories: Any) -> Any:
        """Test handling of corrupted or invalid GNN files."""
        corrupted_file = temp_directories["input_dir"] / "corrupted.md"
        corrupted_file.write_text("This is not a valid GNN file\n\x00\x01\x02")
        try:
            from gnn.schema_validator import GNNParser

            parser = GNNParser()
            result = parser.parse_file(str(corrupted_file))
            assert isinstance(result, dict)
        except Exception as e:
            assert "parse" in str(e).lower() or "invalid" in str(e).lower()


class TestResourceConstraintScenarios:
    """Test pipeline behavior under resource constraints."""

    @pytest.mark.unit
    def test_large_gnn_file_handling(self, temp_directories: Any) -> Any:
        """Test handling of unusually large GNN files."""
        large_file = temp_directories["input_dir"] / "large_model.md"
        large_content = "# Large GNN Model\n\n## StateSpaceBlock\n"
        for i in range(1000):
            large_content += f"var_{i} [10,10] # Variable {i}\n"
        large_content += "\n## Connections\n"
        for i in range(500):
            large_content += f"var_{i} > var_{i + 1}\n"
        large_file.write_text(large_content)
        try:
            from gnn.parsers.unified_parser import UnifiedGNNParser

            parser = UnifiedGNNParser()
            result = parser.parse_file(str(large_file))
            if result is not None:
                assert isinstance(result, dict)
        except ImportError:
            raise AssertionError("Parser interface not available")
        except Exception as e:
            assert any(
                word in str(e).lower()
                for word in [
                    "memory",
                    "timeout",
                    "resource",
                    "size",
                    "parse",
                    "invalid",
                ]
            )

    @pytest.mark.unit
    def test_concurrent_pipeline_execution(self, temp_directories: Any) -> Any:
        """Test behavior when multiple pipeline steps run concurrently."""
        from concurrent.futures import ThreadPoolExecutor

        def run_visualization() -> Any:
            try:
                from gnn.visualization import process_visualization

                return process_visualization(
                    target_dir=temp_directories["input_dir"],
                    output_dir=temp_directories["output_dir"] / "viz_thread",
                    verbose=False,
                )
            except Exception as e:
                return str(e)

        def run_gnn_processing() -> Any:
            try:
                from gnn import process_gnn_directory

                return process_gnn_directory(
                    temp_directories["input_dir"],
                    output_dir=temp_directories["output_dir"] / "gnn_thread",
                )
            except Exception as e:
                return str(e)

        test_file = temp_directories["input_dir"] / "concurrent_test.md"
        test_file.write_text(
            "\n# Concurrent Test Model\n\n## StateSpaceBlock\nstate [3,3]\naction [2]\n\n## Connections  \nstate > action\n        "
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(run_visualization)
            future2 = executor.submit(run_gnn_processing)
            result1 = future1.result(timeout=30)
            result2 = future2.result(timeout=30)
            assert isinstance(result1, bool) or isinstance(result1, str)
            assert isinstance(result2, (dict, str))
            if isinstance(result2, dict):
                assert result2["status"] == "SUCCESS"
                assert result2["processed_files"]


class TestPipelineIntegrationScenarios:
    """Test full pipeline integration under various scenarios."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_pipeline_with_missing_dependencies(self, temp_directories: Any) -> Any:
        """Test full pipeline execution with some missing dependencies."""
        test_file = temp_directories["input_dir"] / "integration_test.md"
        test_file.write_text(
            "\n# Integration Test Model\n\n## StateSpaceBlock\nstate [2,2]\nobservation [2]\n\n## Connections\nstate > observation\n        "
        )
        try:
            cmd: list[Any] = [
                sys.executable,
                str(SRC_DIR / "gnn" / "main.py"),
                "--target-dir",
                str(temp_directories["input_dir"]),
                "--output-dir",
                str(temp_directories["output_dir"]),
                "--only-steps",
                "3,5,7,8",
                "--verbose",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, cwd=PROJECT_ROOT
            )
            assert result.returncode in [0, 2], (
                f"Pipeline failed with exit code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
            output_files = list(temp_directories["output_dir"].rglob("*"))
            assert len(output_files) > 0, "No output files produced"
        except subprocess.TimeoutExpired:
            pytest.fail("Pipeline execution timed out")
        except Exception as e:
            pytest.fail(f"Pipeline execution failed: {e}")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_error_recovery_and_continuation(self, temp_directories: Any) -> Any:
        """Test pipeline's ability to recover from errors and continue."""
        problematic_file = temp_directories["input_dir"] / "problematic.md"
        problematic_file.write_text(
            "\n# Problematic GNN Model\n\n## StateSpaceBlock\n# Invalid variable definition that may cause parsing issues\ninvalid_var [invalid_dimension, another_invalid]\nvalid_var [3,3]\n\n## Connections\n# Invalid connection\ninvalid_var > non_existent_var\nvalid_var > valid_var\n        "
        )
        valid_file = temp_directories["input_dir"] / "valid.md"
        valid_file.write_text(
            "\n# Valid GNN Model\n\n## StateSpaceBlock  \nstate [2,2]\naction [3]\n\n## Connections\nstate > action\n        "
        )
        try:
            cmd: list[Any] = [
                sys.executable,
                str(SRC_DIR / "gnn" / "main.py"),
                "--target-dir",
                str(temp_directories["input_dir"]),
                "--output-dir",
                str(temp_directories["output_dir"]),
                "--only-steps",
                "3,5",
                "--verbose",
                "--recursive",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, cwd=PROJECT_ROOT
            )
            assert result.returncode in [0, 2], (
                f"Expected success or warnings, got {result.returncode}"
            )
            output_files = list(temp_directories["output_dir"].rglob("*"))
            assert len(output_files) > 0, "No output produced despite valid input file"
        except subprocess.TimeoutExpired:
            pytest.fail("Pipeline execution timed out")


class TestErrorReportingAndDiagnostics:
    """Error reporting and diagnostic coverage.

    Phase 7.2: tests referencing utils.standardized_error_handling were
    removed — that module was aspirational and never implemented.
    utils.diagnostic_logging existed but had no production importer and was
    deleted in the wave-2 dead-surface cleanup (W2-02). The real facilities
    live in utils.error_handling and utils.error_recovery
    (see test_error_recovery_framework.py).
    """

    @pytest.mark.unit
    def test_dependency_validation_reporting(self) -> Any:
        """Test dependency validation reporting."""
        from gnn.utils.runtime_safety.dependency_validator import DependencyValidator

        validator = DependencyValidator()
        core_valid = validator.validate_dependency_group("core")
        assert isinstance(core_valid, bool)
        pymdp_valid = validator.validate_dependency_group("pymdp")
        assert isinstance(pymdp_valid, bool)
        instructions = validator.get_installation_instructions()
        assert isinstance(instructions, list)
