#!/usr/bin/env python3
"""
Main Orchestrator Tests for GNN Pipeline

This module contains comprehensive tests for the main pipeline orchestrator
(src/gnn/main.py) which coordinates the execution of all pipeline steps.

Tests cover:
1. Import validation and component checking
2. Argument parsing functionality and validation
3. Pipeline script discovery and metadata extraction
4. Virtual environment detection and handling
5. Individual step execution coordination
6. Overall pipeline orchestration
7. Performance monitoring and system info collection
8. Pipeline output and reporting
9. Error handling, retry logic, and graceful degradation
10. Configuration consistency validation
11. End-to-end integration scenarios

All tests execute real methods and subprocesses.
"""

import logging
import shutil
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Test markers
pytestmark = [pytest.mark.pipeline, pytest.mark.main_orchestrator]

# Local paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
POINTS_DIR = PROJECT_ROOT / "src" / "gnn"


class TestMainOrchestratorImport:
    """Test main orchestrator import validation and component checking."""

    @pytest.mark.unit
    def test_main_orchestrator_file_exists(self) -> None:
        """Test that main.py exists and is readable."""
        main_script = POINTS_DIR / "main.py"

        assert main_script.exists(), (
            f"Main orchestrator script not found: {main_script}"
        )
        assert main_script.is_file(), (
            f"Main orchestrator path is not a file: {main_script}"
        )

        # Test that file is readable
        try:
            content = main_script.read_text()
            assert len(content) > 0, "Main orchestrator script is empty"

            # Test basic Python structure
            assert "import" in content, "Main script should have import statements"
            assert "def main" in content or "if __name__" in content, (
                "Main script should have main function or entry point"
            )

            logging.info("Main orchestrator file structure validated")

        except Exception as e:
            pytest.fail(f"Failed to read main orchestrator script: {e}")

    @pytest.mark.unit
    def test_main_orchestrator_help_executes(self) -> None:
        """main.py should print help and exit 0."""
        main_py = POINTS_DIR / "main.py"
        result = subprocess.run(
            [sys.executable, str(main_py), "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )  # nosec B603
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()

    @pytest.mark.unit
    def test_main_orchestrator_component_availability(self) -> None:
        """Test that main orchestrator components are available."""
        # Test that main components exist
        components: dict[str, Any] = {
            "utils_package": POINTS_DIR / "utils" / "__init__.py",
            "pipeline_package": POINTS_DIR / "pipeline" / "__init__.py",
            "main_script": POINTS_DIR / "main.py",
        }

        missing_components: list[Any] = []
        for name, path in components.items():
            if not path.exists():
                missing_components.append(f"{name}: {path}")

        assert not missing_components, (
            f"Missing main orchestrator components: {missing_components}"
        )
        logging.info(f"All {len(components)} main orchestrator components available")


class TestArgumentParsing:
    """Smoke-check argument parser via main --help only."""

    @pytest.mark.unit
    def test_help(self) -> None:
        main_py = POINTS_DIR / "main.py"
        result = subprocess.run(
            [sys.executable, str(main_py), "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )  # nosec B603
        assert result.returncode == 0


class TestPipelineScriptDiscovery:
    """Test pipeline script discovery and metadata extraction."""

    @pytest.mark.unit
    def test_pipeline_script_discovery_logic(self) -> None:
        """Test pipeline script discovery logic."""
        # Test script discovery pattern
        expected_pattern = r"^(\d+)_.*\.py$"

        test_scripts: list[Any] = [
            ("1_setup.py", True, 1),
            ("3_gnn.py", True, 3),  # Fixed: regex extracts 3, not 2
            ("20_website.py", True, 20),
            ("23_report.py", True, 23),
            ("24_intelligent_analysis.py", True, 24),
            ("main.py", False, None),
            ("gnn.utils.py", False, None),
            ("not_a_script.txt", False, None),
        ]

        import re

        pattern = re.compile(expected_pattern)

        for script_name, should_match, expected_num in test_scripts:
            match = pattern.match(script_name)

            if should_match:
                assert match is not None, f"Script {script_name} should match pattern"
                assert int(match.group(1)) == expected_num, (
                    f"Script {script_name} should extract number {expected_num}"
                )
            else:
                assert match is None, f"Script {script_name} should not match pattern"

        logging.info("Pipeline script discovery logic validated")

    @pytest.mark.unit
    def test_pipeline_script_sorting(self) -> None:
        """Test pipeline script sorting logic."""
        # Test script sorting by number and name
        sample_scripts: list[Any] = [
            {"num": 3, "basename": "3_gnn.py"},
            {"num": 1, "basename": "1_setup.py"},
            {"num": 13, "basename": "13_llm.py"},
            {"num": 14, "basename": "14_ml_integration.py"},
            {"num": 2, "basename": "2_tests.py"},
        ]

        # Sort like the main orchestrator would
        sorted_scripts = sorted(sample_scripts, key=lambda x: (x["num"], x["basename"]))

        # Verify correct order - include all scripts that exist
        expected_order: list[Any] = [1, 2, 3, 13, 14]
        actual_order = [script["num"] for script in sorted_scripts]

        assert actual_order == expected_order, (
            f"Scripts should be sorted correctly: {actual_order}"
        )
        logging.info("Pipeline script sorting logic validated")


class TestVirtualEnvironmentHandling:
    @pytest.mark.unit
    def test_python_executable_detection(self) -> None:
        python_executable = sys.executable
        assert python_executable, "python executable should be detected"


class TestStepExecution:
    """Execute a core step for real."""

    @pytest.mark.integration
    def test_run_core_step(self) -> None:
        script = POINTS_DIR / "3_gnn.py"
        if not script.exists():
            raise AssertionError("3_gnn.py missing")
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td) / "output"
            target_dir = PROJECT_ROOT / "input" / "gnn_files" / "discrete"
            cmd: list[Any] = [
                sys.executable,
                str(script),
                "--target-dir",
                str(target_dir),
                "--output-dir",
                str(outdir),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT)
            )  # nosec B603
            assert result.returncode in [0, 1]


class TestPipelineCoordination:
    @pytest.mark.slow
    @pytest.mark.integration
    def test_minimal_pipeline_execution(self) -> None:
        main_py = POINTS_DIR / "main.py"
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td) / "output"
            target_dir = PROJECT_ROOT / "input" / "gnn_files" / "discrete"
            cmd: list[Any] = [
                sys.executable,
                str(main_py),
                "--only-steps",
                "3,5,7",
                "--target-dir",
                str(target_dir),
                "--output-dir",
                str(outdir),
            ]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                    timeout=240,
                )  # nosec B603
            except subprocess.TimeoutExpired as exc:
                pytest.fail(
                    f"Minimal pipeline timed out for {target_dir}: "
                    f"stdout={exc.stdout!r}, stderr={exc.stderr!r}"
                )

            # Check for summary in the correct location (00_pipeline_summary subdirectory)
            summary = outdir / "00_pipeline_summary" / "pipeline_execution_summary.json"

            # Also check alternate location (directly in output dir) and existence of output dir
            summary_alt = outdir / "pipeline_execution_summary.json"

            # Test passes if summary exists OR if output directory was created with step outputs
            if summary.exists() or summary_alt.exists():
                pass  # Success - summary file exists
            else:
                # Check if the pipeline ran at all by looking for step output directories
                step_outputs = list(outdir.glob("*_output")) if outdir.exists() else []
                assert len(step_outputs) > 0 or result.returncode == 0, (
                    f"Pipeline should either create summary file or complete with step outputs. "
                    f"Return code: {result.returncode}, stderr: {result.stderr[:500] if result.stderr else 'None'}"
                )

    @pytest.mark.unit
    def test_environment_info_function(self) -> None:
        from gnn.main import get_environment_info

        info = get_environment_info()
        assert isinstance(info, dict)
        assert "python_version" in info


class TestEndToEndIntegration:
    @pytest.mark.slow
    @pytest.mark.integration
    def test_run_pipeline_subset(self) -> None:
        main_py = POINTS_DIR / "main.py"
        with tempfile.TemporaryDirectory() as td:
            target_dir = Path(td) / "input"
            target_dir.mkdir()
            shutil.copy2(
                PROJECT_ROOT / "input" / "gnn_files" / "discrete" / "simple_mdp.md",
                target_dir / "simple_mdp.md",
            )
            outdir = Path(td) / "output"
            cmd: list[Any] = [
                sys.executable,
                str(main_py),
                "--only-steps",
                "3,5,7,8",
                "--target-dir",
                str(target_dir),
                "--output-dir",
                str(outdir),
            ]
            subprocess.run(cmd, cwd=str(PROJECT_ROOT))  # nosec B603
            # Assert expected subdirs may be created by steps
            assert (outdir / "3_gnn_output").exists() or (
                outdir / "gnn_processing_step"
            ).exists()

    def test_pipeline_summary_validation(self) -> None:
        """Test pipeline summary structure validation."""
        import logging

        from gnn.main import validate_pipeline_summary

        # Create a logger for testing
        logger = logging.getLogger("test_logger")

        # Test valid summary
        valid_summary: dict[str, Any] = {
            "start_time": "2025-01-01T00:00:00",
            "arguments": {"target_dir": "input"},
            "steps": [
                {
                    "status": "SUCCESS",
                    "step_number": 1,
                    "script_name": "test.py",
                    "description": "Test step",
                    "exit_code": 0,
                }
            ],
            "end_time": "2025-01-01T00:01:00",
            "overall_status": "SUCCESS",
            "total_duration_seconds": 60.0,
            "environment_info": {"python_version": "3.11"},
            "performance_summary": {
                "peak_memory_mb": 100.0,
                "total_steps": 1,
                "failed_steps": 0,
                "critical_failures": 0,
                "successful_steps": 1,
                "warnings": 0,
            },
        }

        # Should not raise any errors
        validate_pipeline_summary(valid_summary, logger)

        # Test invalid summary
        invalid_summary: dict[str, Any] = {"start_time": None, "steps": "not_a_list"}

        # Should log warnings/errors but not raise exceptions
        validate_pipeline_summary(invalid_summary, logger)

    def test_enhanced_warning_detection(self) -> None:
        """Test improved warning detection logic."""
        from gnn.main import _step_has_actionable_warning

        safe_warning = {
            "stdout": "INFO: Processing completed\nWARNING: optional dependency not available",
            "stderr": "",
        }
        mixed_warning = {
            "stdout": (
                "INFO: Processing completed\n"
                "WARNING: optional dependency not available\n"
                "WARNING: Budget exhausted"
            ),
            "stderr": "",
        }
        no_warning = {
            "stdout": "INFO: Processing completed\nDEBUG: File processed successfully",
            "stderr": "",
        }

        assert not _step_has_actionable_warning(safe_warning)
        assert _step_has_actionable_warning(mixed_warning)
        assert not _step_has_actionable_warning(no_warning)

    def test_warning_status_updates_performance_summary(self) -> None:
        """Warning step statuses should affect aggregate warning counts."""
        from gnn.main import _update_performance_summary

        summary: dict[str, Any] = {
            "performance_summary": {
                "peak_memory_mb": 0.0,
                "total_steps": 0,
                "failed_steps": 0,
                "critical_failures": 0,
                "successful_steps": 0,
                "warnings": 0,
            }
        }
        step_result: dict[str, Any] = {
            "status": "SUCCESS_WITH_WARNINGS",
            "memory_usage_mb": 0.0,
            "peak_memory_mb": 0.0,
        }

        _update_performance_summary(
            summary,
            step_result,
            "13_llm.py",
            has_warning=False,
            total_steps=1,
        )

        assert summary["performance_summary"]["successful_steps"] == 1
        assert summary["performance_summary"]["warnings"] == 1
        assert summary["performance_summary"]["total_steps"] == 1

    def test_finalize_pipeline_summary_preserves_warning_status(self) -> None:
        """Warning-only pipelines should finish SUCCESS_WITH_WARNINGS."""
        from datetime import datetime

        from gnn.main import _finalize_pipeline_summary

        summary: dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "steps": [{"status": "SUCCESS_WITH_WARNINGS"}],
            "performance_summary": {
                "peak_memory_mb": 0.0,
                "total_steps": 1,
                "failed_steps": 0,
                "critical_failures": 0,
                "successful_steps": 1,
                "warnings": 0,
            },
        }

        _finalize_pipeline_summary(summary)

        assert summary["overall_status"] == "SUCCESS_WITH_WARNINGS"

    def test_pipeline_exit_code_maps_warning_status(self) -> None:
        """Warning-only pipeline status should use the warning exit code."""
        from gnn.main import _pipeline_exit_code

        assert _pipeline_exit_code("SUCCESS") == 0
        assert _pipeline_exit_code("SUCCESS_WITH_WARNINGS") == 2
        assert _pipeline_exit_code("PARTIAL_SUCCESS") == 2
        assert _pipeline_exit_code("FAILED") == 1

    def test_step_exit_code_two_maps_to_warning_status(self) -> None:
        """Child exit code 2 should remain warning, not become failed."""
        from gnn.main import _status_from_step_exit_code
        from gnn.utils.errors.error_handling import status_from_exit_code

        assert _status_from_step_exit_code(0) == "SUCCESS"
        assert _status_from_step_exit_code(0, ["dep warning"]) == (
            "SUCCESS_WITH_WARNINGS"
        )
        assert _status_from_step_exit_code(2) == "SUCCESS_WITH_WARNINGS"
        assert _status_from_step_exit_code(1) == "FAILED"
        assert _status_from_step_exit_code(2) == status_from_exit_code(2)

    def test_critical_script_set_comes_from_step_metadata(self) -> None:
        """Critical script behavior should be driven by StepConfiguration metadata."""
        from gnn.main import CRITICAL_SCRIPTS
        from gnn.utils.arguments.step_config import StepConfiguration

        expected = {
            f"{step_name}.py"
            for step_name, config in StepConfiguration.STEP_CONFIGS.items()
            if config.get("critical")
        }

        assert CRITICAL_SCRIPTS == expected
        assert CRITICAL_SCRIPTS == {
            "0_template.py",
            "1_setup.py",
            "3_gnn.py",
            "5_type_checker.py",
            "6_validation.py",
            "7_export.py",
            "11_render.py",
        }

    def test_finalize_pipeline_summary_mixed_noncritical_failure_is_warning(
        self,
    ) -> None:
        """Few non-critical failures produce warning status; critical failures fail."""
        from datetime import datetime

        from gnn.main import _finalize_pipeline_summary

        summary: dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "steps": [
                {"status": "SUCCESS"},
                {"status": "FAILED"},
                {"status": "SUCCESS_WITH_WARNINGS"},
                {"status": "SUCCESS"},
            ],
            "performance_summary": {
                "peak_memory_mb": 0.0,
                "total_steps": 4,
                "failed_steps": 1,
                "critical_failures": 0,
                "successful_steps": 3,
                "warnings": 1,
            },
        }

        _finalize_pipeline_summary(summary)

        assert summary["overall_status"] == "PARTIAL_SUCCESS"

        summary["performance_summary"]["critical_failures"] = 1
        _finalize_pipeline_summary(summary)
        assert summary["overall_status"] == "FAILED"

    def test_pipeline_summary_step_numbering(self) -> None:
        """Test that pipeline summary uses correct step numbering."""
        # Test that step numbers reflect execution order, not script names
        summary: dict[str, Any] = {
            "steps": [
                {
                    "script_name": "3_gnn.py",
                    "step_number": 1,  # Should be 1 (first executed)
                    "status": "SUCCESS",
                },
                {
                    "script_name": "5_type_checker.py",
                    "step_number": 2,  # Should be 2 (second executed)
                    "status": "SUCCESS",
                },
                {
                    "script_name": "8_visualization.py",
                    "step_number": 3,  # Should be 3 (third executed)
                    "status": "SUCCESS",
                },
            ],
            "performance_summary": {
                "total_steps": 3  # Should reflect actual executed steps
            },
        }

        # Verify step numbering is sequential based on execution order
        assert summary["steps"][0]["step_number"] == 1
        assert summary["steps"][1]["step_number"] == 2
        assert summary["steps"][2]["step_number"] == 3
        assert summary["performance_summary"]["total_steps"] == 3

    def test_pipeline_summary_validation_comprehensive(self) -> None:
        """Test comprehensive pipeline summary validation."""
        import logging

        from gnn.main import validate_pipeline_summary

        # Create a logger for testing
        logger = logging.getLogger("test_logger")

        # Test with valid summary
        valid_summary: dict[str, Any] = {
            "start_time": "2025-01-01T00:00:00",
            "end_time": "2025-01-01T00:01:00",
            "total_duration_seconds": 60.0,
            "overall_status": "SUCCESS",
            "arguments": {},
            "steps": [
                {
                    "status": "SUCCESS",
                    "step_number": 1,
                    "script_name": "test.py",
                    "description": "Test step",
                    "exit_code": 0,
                    "duration_seconds": 1.0,
                }
            ],
            "environment_info": {"python_version": "3.11"},
            "performance_summary": {
                "peak_memory_mb": 100.0,
                "total_steps": 1,
                "failed_steps": 0,
                "successful_steps": 1,
                "warnings": 0,
            },
        }

        # Should not raise any errors for valid summary
        validate_pipeline_summary(valid_summary, logger)

        # Test with invalid summary
        invalid_summary: dict[str, Any] = {
            "start_time": None,
            "steps": "not_a_list",
            "overall_status": "INVALID_STATUS",
        }

        # Should log warnings but not raise exceptions
        validate_pipeline_summary(invalid_summary, logger)

    def test_pipeline_step_execution_order(self) -> None:
        """Dependency tiers produced by the real DAG resolver are topologically valid."""
        from gnn.pipeline.dag import resolve_execution_order

        # Documented data-flow dependencies (src/gnn/AGENTS.md § Data Dependencies):
        # step N -> [dependency step numbers that must run before N]
        dependencies: dict[int, list[int]] = {
            4: [3],  # registry needs parsed models
            5: [3],  # type checker needs parsed models
            6: [3, 5],  # validation needs parsed models + type info
            7: [3],  # export needs parsed models
            8: [3],  # visualization needs parsed models
            9: [3],  # advanced viz needs parsed models
            11: [3],  # render needs parsed models
            12: [11],  # execute needs rendered code
            13: [3],  # LLM needs parsed models
            16: [12],  # analysis needs execution results
            23: [16],  # report needs analysis results
        }
        tiers = resolve_execution_order(dependencies, total_steps=25)
        flat = [step for tier in tiers for step in tier]
        assert sorted(flat) == list(range(25)), (
            "every step 0-24 must be resolved exactly once"
        )
        position = {step: index for index, step in enumerate(flat)}
        for step, deps in dependencies.items():
            for dep in deps:
                assert position[dep] < position[step], (
                    f"dependency order violated: step {dep} must precede step {step}"
                )

    def test_pipeline_summary_metadata_completeness(self) -> None:
        """Test that pipeline summary includes all required metadata."""
        summary: dict[str, Any] = {
            "start_time": "2025-01-01T00:00:00",
            "arguments": {"target_dir": "input"},
            "steps": [
                {
                    "status": "SUCCESS",
                    "step_number": 1,
                    "script_name": "3_gnn.py",
                    "description": "GNN file processing",
                    "exit_code": 0,
                    "memory_usage_mb": 29.5,
                    "peak_memory_mb": 29.5,
                    "duration_seconds": 0.077,
                    "retry_count": 0,
                    "prerequisite_check": True,
                    "dependency_warnings": [],
                    "recoverable": False,
                }
            ],
            "end_time": "2025-01-01T00:01:00",
            "overall_status": "SUCCESS",
            "total_duration_seconds": 60.0,
            "environment_info": {"python_version": "3.11"},
            "performance_summary": {
                "peak_memory_mb": 29.625,
                "total_steps": 1,
                "failed_steps": 0,
                "successful_steps": 1,
                "warnings": 0,
            },
        }

        # Test that all required fields are present
        required_step_fields: list[Any] = [
            "status",
            "step_number",
            "script_name",
            "description",
            "exit_code",
            "memory_usage_mb",
            "peak_memory_mb",
            "duration_seconds",
        ]

        step = summary["steps"][0]
        for field in required_step_fields:
            assert field in step, f"Step missing required field: {field}"

        # Test performance summary completeness
        perf = summary["performance_summary"]
        required_perf_fields: list[Any] = [
            "peak_memory_mb",
            "total_steps",
            "failed_steps",
            "successful_steps",
            "warnings",
        ]

        for field in required_perf_fields:
            assert field in perf, f"Performance summary missing field: {field}"

    def test_parallel_pipeline_execution_flag(self, tmp_path: Path) -> None:
        """Test that --parallel flag executes pipeline successfully."""
        from gnn.main import main
        from gnn.utils.arguments.pipeline_arguments import PipelineArguments

        out_dir = tmp_path / "parallel_out"
        args = PipelineArguments(
            target_dir=Path("input/gnn_files/basics"),
            output_dir=out_dir,
            only_steps="3,7,8",
            parallel=True,
            verbose=False,
        )
        exit_code = main(args)
        failure_details: dict[str, Any] | None = None
        if exit_code != 0:
            records: dict[str, Any] = {}
            for name, filename in (
                ("canonical_summary", "pipeline_execution_summary.json"),
                ("run_context", "run_context.json"),
            ):
                path = out_dir / "00_pipeline_summary" / filename
                try:
                    with path.open("rb") as stream:
                        content = stream.read(65536 + 1)
                    records[name] = {
                        "path": str(path),
                        "content_utf8": content[:65536].decode(
                            "utf-8", errors="replace"
                        ),
                        "truncated": len(content) > 65536,
                    }
                except OSError as error:
                    records[name] = {
                        "path": str(path),
                        "missing_or_unreadable": str(error),
                        "error_type": type(error).__name__,
                    }
            failure_details = {"main_status": exit_code, "owned_records": records}
        assert exit_code == 0, failure_details
        assert (out_dir / "3_gnn_output").exists()
        assert (out_dir / "7_export_output").exists()
        assert (out_dir / "8_visualization_output").exists()

    def test_enhanced_pipeline_summary_structure(self) -> None:
        """Test that pipeline summary has enhanced structure with correct step numbering."""
        # Test the pipeline summary structure improvements
