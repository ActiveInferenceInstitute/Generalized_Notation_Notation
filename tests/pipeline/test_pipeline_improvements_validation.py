"""
Pipeline Improvements Validation Tests

This module validates that the specific improvements made to the pipeline
(DisCoPy module creation, visualization fixes, error handling, etc.) work correctly.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.pipeline, pytest.mark.integration]
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class TestDiScoPyModuleCreation:
    """Test that the created DisCoPy translator module works correctly."""

    @pytest.mark.unit
    def test_discopy_translator_module_import(self) -> None:
        """Test that the DisCoPy translator module can be imported."""
        from gnn.execute.discopy_translator_module.translator import (
            JAX_FULLY_OPERATIONAL,
            MATPLOTLIB_AVAILABLE,
            gnn_file_to_discopy_diagram,
            gnn_file_to_discopy_matrix_diagram,
        )

        assert callable(gnn_file_to_discopy_diagram)
        assert callable(gnn_file_to_discopy_matrix_diagram)
        assert isinstance(JAX_FULLY_OPERATIONAL, bool)
        assert isinstance(MATPLOTLIB_AVAILABLE, bool)

    @pytest.mark.unit
    def test_discopy_diagram_creation_graceful_degradation(self) -> None:
        """Test DisCoPy diagram creation with graceful degradation."""
        try:
            from gnn.execute.discopy_translator_module.translator import (
                gnn_file_to_discopy_diagram,
            )

            test_gnn_data: dict[str, Any] = {
                "Variables": {
                    "state": {
                        "type": "state",
                        "dimensions": [3, 3],
                        "comment": "State space",
                    },
                    "action": {
                        "type": "action",
                        "dimensions": [2],
                        "comment": "Action space",
                    },
                },
                "Edges": [{"source": "state", "target": "action", "directed": True}],
            }
            success, message, diagram = gnn_file_to_discopy_diagram(test_gnn_data)
            assert isinstance(success, bool)
            assert isinstance(message, str)
            if success:
                assert diagram is not None
            else:
                assert "DisCoPy" in message or "not installed" in message
        except Exception as e:
            pytest.fail(f"DisCoPy diagram creation failed unexpectedly: {e}")

    @pytest.mark.unit
    def test_visualize_jax_output_module_import(self) -> None:
        """Test that the JAX visualization module can be imported."""
        from gnn.execute.discopy_translator_module.visualize_jax_output import (
            create_summary_visualization,
            plot_multiple_tensor_outputs,
            plot_tensor_output,
        )

        assert callable(plot_tensor_output)
        assert callable(plot_multiple_tensor_outputs)
        assert callable(create_summary_visualization)


class TestVisualizationBugFixes:
    """Test that visualization bug fixes work correctly."""

    @pytest.mark.unit
    def test_visualization_parser_type_safety(self) -> None:
        """Test that visualization parser handles type fields safely."""
        try:
            from gnn.visualization.parser import GNNParser

            parser = GNNParser()
            test_content = "\n# Test Model\n\n## StateSpaceBlock\nvar1 [3,3] # Test variable\nvar2 [2] # Another variable  \nvar3 [5,5,2] # 3D variable\n\n## Connections\nvar1 > var2\nvar2 > var3\n            "
            parsed_data = parser._parse_markdown_format(test_content)
            assert isinstance(parsed_data, dict)
            assert "Variables" in parsed_data
            variables = parsed_data["Variables"]
            for _var_name, var_info in variables.items():
                assert "type" in var_info
                assert isinstance(var_info["type"], str)
                assert var_info["type"] != ""
        except Exception as e:
            pytest.fail(f"Visualization parser failed: {e}")

    @pytest.mark.unit
    def test_matplotlib_dpi_corruption_fix(self) -> None:
        """Test that matplotlib DPI corruption is handled safely."""
        import matplotlib.pyplot as plt

        from gnn.visualization.processor import save_plot_safely

        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "test_plot.png"
            plt.figure()
            plt.plot([1, 2, 3], [1, 4, 9])
            corrupted_dpi_values: list[Any] = [
                28421050826,
                -1,
                0,
                float("inf"),
                "invalid",
            ]
            for bad_dpi in corrupted_dpi_values:
                result = save_plot_safely(test_path, dpi=bad_dpi)
                assert isinstance(result, bool)
                if result:
                    assert test_path.exists()
                    test_path.unlink()
            plt.close()

    @pytest.mark.unit
    def test_visualization_data_structure_compatibility(self) -> None:
        """Real API: visualization.processor.generate_matrix_visualizations."""
        from gnn.visualization.processor import generate_matrix_visualizations

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            parsed_data: dict[str, Any] = {
                "matrices": [
                    {"name": "A", "values": [[0.9, 0.1], [0.1, 0.9]]},
                ],
            }
            artifacts = generate_matrix_visualizations(
                parsed_data, output_dir, "test_model"
            )
            assert isinstance(artifacts, list)


class TestErrorHandlingImprovements:
    """Error handling coverage.

    Phase 7.2: tests referencing utils.standardized_error_handling were
    removed — the module was aspirational and never implemented. Real error
    handling lives in utils.error_handling.PipelineErrorHandler and is
    covered by test_error_recovery_framework.py.
    """


class TestDependencyValidationImprovements:
    """Test that dependency validation improvements work correctly."""

    @pytest.mark.unit
    def test_discopy_dependency_group(self) -> None:
        """Test that DisCoPy dependency group is properly defined."""
        from gnn.utils.runtime_safety.dependency_validator import DependencyValidator

        validator = DependencyValidator()
        assert "discopy" in validator.dependencies
        discopy_deps = validator.dependencies["discopy"]
        dep_names = [dep.name for dep in discopy_deps]
        assert "discopy" in dep_names
        assert "jax" in dep_names or "jaxlib" in dep_names
        for dep in discopy_deps:
            assert dep.is_optional, f"DisCoPy dependency {dep.name} should be optional"

    @pytest.mark.unit
    def test_pymdp_dependency_group(self) -> None:
        """Test that PyMDP dependency group is properly defined."""
        from gnn.utils.runtime_safety.dependency_validator import DependencyValidator

        validator = DependencyValidator()
        assert "pymdp" in validator.dependencies
        pymdp_deps = validator.dependencies["pymdp"]
        dep_names = [dep.name for dep in pymdp_deps]
        has_pymdp = any("pymdp" in name.lower() for name in dep_names)
        assert has_pymdp, f"Expected pymdp dependency, got: {dep_names}"
        pymdp_dep = next(dep for dep in pymdp_deps if "pymdp" in dep.name.lower())
        assert pymdp_dep.is_optional, "pymdp dependency should be marked optional"

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_improved_installation_instructions(self) -> None:
        """Test that installation instructions are provided for missing dependencies."""
        from gnn.utils.runtime_safety.dependency_validator import DependencyValidator

        validator = DependencyValidator()
        optional_groups: list[Any] = ["discopy", "pymdp", "rxinfer"]
        for group in optional_groups:
            if group in validator.dependencies:
                validator.validate_dependency_group(group)
        instructions = validator.get_installation_instructions()
        assert isinstance(instructions, list)
        for instruction in instructions:
            if "pip install" in instruction:
                assert "uv pip install" in instruction or "pip install" in instruction


class TestLoggingImprovements:
    """Real logging infrastructure coverage (utils.logging.logging_utils).

    Phase 7.2: replaced aspirational utils.diagnostic_logging tests with
    coverage of the actually-implemented correlation-aware logger helpers.
    """

    @pytest.mark.unit
    def test_structured_logging_exposes_correlation_helpers(self) -> None:
        from gnn.utils.logging.logging_utils import (
            log_step_error,
            log_step_start,
            log_step_success,
            log_step_warning,
            setup_step_logging,
        )

        for fn in (
            log_step_start,
            log_step_success,
            log_step_warning,
            log_step_error,
            setup_step_logging,
        ):
            assert callable(fn)

    @pytest.mark.unit
    def test_setup_step_logging_returns_logger(self) -> None:
        import logging as _logging

        from gnn.utils.logging_utils import setup_step_logging

        logger = setup_step_logging("test_step", verbose=False)
        assert isinstance(logger, _logging.Logger)


class TestIntegrationValidation:
    """Test that all improvements work together in integration."""

    @pytest.mark.integration
    def test_end_to_end_pipeline_with_improvements(self, temp_directories: Any) -> None:
        """Test that the pipeline runs with all improvements integrated."""
        test_file = temp_directories["input_dir"] / "improvements_test.md"
        test_file.write_text(
            "\n# Improvements Test Model\n\n## StateSpaceBlock\nstate [3,3,2] # Complex tensor\naction [2] # Simple vector  \nobservation [3] # Observation space\n\n## Connections\nstate > action\naction > observation\nobservation > state\n        "
        )
        try:
            from gnn.execute.discopy.discopy_executor import DisCoPyExecutor

            executor = DisCoPyExecutor(verbose=True)
            assert executor is not None
            from gnn.visualization.parser import GNNParser

            parser = GNNParser()
            parsed_data = parser.parse_file(str(test_file))
            assert isinstance(parsed_data, dict)
            assert "Variables" in parsed_data
            for _var_name, var_info in parsed_data["Variables"].items():
                assert "type" in var_info
                assert var_info["type"] is not None
                assert var_info["type"] != ""
        except Exception as e:
            pytest.fail(f"End-to-end integration test failed: {e}")


@pytest.mark.parametrize("step_numbers", [[3, 5, 7, 8], [3, 8, 12, 15]])
@pytest.mark.slow
@pytest.mark.integration
def test_specific_pipeline_steps_improvements(
    step_numbers: Any, temp_directories: Any
) -> None:
    """Test that specific pipeline steps work with improvements."""
    import subprocess

    test_file = temp_directories["input_dir"] / "step_test.md"
    test_file.write_text(
        """# Step Test Model

## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
Step Test Model

## StateSpaceBlock
A[3,3,type=float]
B[3,3,3,type=float]
C[3,type=float]
D[3,type=float]
E[3,type=float]
s[3,1,type=float]
o[3,1,type=int]
u[1,type=int]

## Connections
D>s
s-A
A-o
s-B
B>u
C>u
E>u

## InitialParameterization
A={
  (0.9, 0.05, 0.05),
  (0.05, 0.9, 0.05),
  (0.05, 0.05, 0.9)
}
B={
  ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
  ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
  ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
}
C={(0.1, 0.1, 1.0)}
D={(0.33333, 0.33333, 0.33333)}
E={(0.33333, 0.33333, 0.33333)}

## Time
Dynamic
Discrete
ModelTimeHorizon=5
"""
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
            ",".join(map(str, step_numbers)),
            "--verbose",
        ]
        if 12 in step_numbers:
            cmd.extend(["--frameworks", "pymdp"])
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, cwd=PROJECT_ROOT
        )
        assert result.returncode in [0, 2], (
            f"Steps {step_numbers} failed: {result.stderr}"
        )
        stderr_lower = result.stderr.lower()
        assert "discopy translator module not available" not in stderr_lower
        assert "incompatible constructor arguments" not in stderr_lower
        assert "nameerror" not in stderr_lower or "'type'" not in stderr_lower
    except subprocess.TimeoutExpired:
        pytest.fail(f"Pipeline steps {step_numbers} timed out")
    except Exception as e:
        pytest.fail(f"Pipeline steps {step_numbers} failed unexpectedly: {e}")
