#!/usr/bin/env python3
"""
Pipeline Runtime Integration Tester

Role: Runs the GNN pipeline via subprocess and validates that outputs are produced
correctly. Tests end-to-end pipeline behavior. Does NOT analyze source code.

See also:
- utils/pipeline_orchestration/pipeline_validator.py: Pre-execution prerequisite checker (checks step outputs exist)
"""

import json
import logging
import os
import subprocess  # nosec B404
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# gnn.utils.* are same-package core imports; no path bootstrap or
# degraded-mode scaffolding is needed for them.
from gnn.utils.pipeline_orchestration.pipeline_dependencies import (
    get_pipeline_dependency_manager,
)

# Probe targets resolve against the installed ``gnn`` package root so the
# checks are independent of the working directory and survive repo-layout
# moves (post-3.3.0 the renderer tree lives at ``src/gnn/render/``).
_PROBE_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class PipelineValidator:
    """Comprehensive pipeline validator and improvement tester."""

    def __init__(self, verbose: bool = True) -> None:
        """Initialize the instance."""
        self.verbose = verbose
        self.logger = self._setup_logging()
        self.dependency_manager = get_pipeline_dependency_manager()
        self.validation_results: dict[str, Any] = {}

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the validator."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger

    def validate_code_generation_fixes(self) -> Dict[str, Any]:
        """Validate that code generation issues have been fixed."""
        self.logger.info("🔧 Validating code generation fixes...")

        fixes_validation: dict[str, Any] = {
            "pymdp_import_fix": False,
            "jax_flax_fix": False,
            "julia_matrix_fix": False,
            "dependency_handling": False,
        }

        try:
            # Probe targets live inside the gnn package; a missing target is
            # an error (the check cannot run), not a silent pass.
            package_root = _PROBE_PACKAGE_ROOT

            # Test PyMDP import fix: the renderer must not import the retired
            # configure_from_gnn_spec helper.
            pymdp_renderer_path = package_root / "render/pymdp/pymdp_renderer.py"
            if pymdp_renderer_path.exists():
                content = pymdp_renderer_path.read_text()
                # Should NOT contain configure_from_gnn_spec import
                if "configure_from_gnn_spec" not in content:
                    fixes_validation["pymdp_import_fix"] = True
                    self.logger.info("✅ PyMDP import fix validated")
                else:
                    self.logger.warning("⚠️ PyMDP import fix not applied")
            else:
                self.logger.error(f"❌ Probe target missing: {pymdp_renderer_path}")

            # Test JAX Flax fix: generated model constants must be derived
            # from matrix shapes (NUM_STATES = {num_states}), never attribute
            # access (.num_states) on a model object.
            jax_generator_path = package_root / "render/jax/jax_model_generator.py"
            if jax_generator_path.exists():
                content = jax_generator_path.read_text()
                if (
                    "NUM_STATES = {num_states}" in content
                    and ".num_states" not in content
                ):
                    fixes_validation["jax_flax_fix"] = True
                    self.logger.info("✅ JAX Flax attribute access fix validated")
                else:
                    self.logger.warning("⚠️ JAX Flax fix not applied")
            else:
                self.logger.error(f"❌ Probe target missing: {jax_generator_path}")

            # Test Julia matrix formatting fix: element conversion must handle
            # both scalar values and row containers.
            julia_renderer_path = (
                package_root / "render/activeinference_jl/activeinference_renderer.py"
            )
            if julia_renderer_path.exists():
                content = julia_renderer_path.read_text()
                if "isinstance(row, (tuple, list))" in content:
                    fixes_validation["julia_matrix_fix"] = True
                    self.logger.info("✅ Julia matrix formatting fix validated")
                else:
                    self.logger.warning("⚠️ Julia matrix fix not applied")
            else:
                self.logger.error(f"❌ Probe target missing: {julia_renderer_path}")

            # Test pipeline dependency manager
            dep_manager_path = (
                package_root / "utils/pipeline_orchestration/pipeline_dependencies.py"
            )
            if dep_manager_path.exists():
                fixes_validation["dependency_handling"] = True
                self.logger.info("✅ Pipeline dependency manager created")
            else:
                self.logger.error(
                    f"❌ Pipeline dependency manager missing: {dep_manager_path}"
                )

        except Exception as e:
            self.logger.error(f"❌ Error validating code generation fixes: {e}")

        return fixes_validation

    def test_dependency_validation(self) -> Dict[str, Any]:
        """Test the enhanced dependency validation system."""
        self.logger.info("📋 Testing enhanced dependency validation...")

        try:
            # Test key pipeline steps
            test_steps: list[Any] = [
                "8_visualization",
                "11_render",
                "12_execute",
                "15_audio",
            ]
            step_results: dict[Any, Any] = {}

            for step in test_steps:
                result = self.dependency_manager.check_step_dependencies(step)
                step_results[step] = {
                    "status": result["status"],
                    "required_satisfied": all(
                        dep.available for dep in result["required"].values()
                    ),
                    "optional_count": len(result["optional"]),
                    "fallbacks_available": len(result["fallbacks_available"]),
                    "warnings": len(result["warnings"]),
                    "errors": len(result["errors"]),
                }

                status_emoji = {
                    "healthy": "✅",
                    "degraded": "⚠️",
                    "failed": "❌",
                    "unknown": "❓",
                }.get(result["status"], "❓")

                self.logger.info(
                    f"{status_emoji} {step}: {result['status'].upper()} "
                    f"({len(result['errors'])} errors, {len(result['warnings'])} warnings)"
                )

            return {
                "validation_successful": True,
                "step_results": step_results,
                "overall_health": self._calculate_overall_health(step_results),
            }

        except Exception as e:
            self.logger.error(f"❌ Dependency validation test failed: {e}")
            return {"validation_successful": False, "error": str(e)}

    def _calculate_overall_health(self, step_results: Dict[str, Any]) -> str:
        """Calculate overall pipeline health from step results."""
        if not step_results:
            return "unknown"

        statuses = [result["status"] for result in step_results.values()]

        if all(status == "healthy" for status in statuses):
            return "healthy"
        elif any(status == "failed" for status in statuses):
            return "failed"
        else:
            return "degraded"

    def test_pipeline_execution(
        self, steps: (List[str]) | None = None
    ) -> Dict[str, Any]:
        """Test actual pipeline execution with improvements."""
        self.logger.info("🚀 Testing pipeline execution...")

        if steps is None:
            steps = ["3", "5", "7", "8", "11", "15"]  # Key steps for testing

        test_results: dict[str, Any] = {
            "execution_successful": False,
            "step_results": {},
            "execution_time": 0,
            "memory_usage": 0,
            "warnings_count": 0,
            "errors_count": 0,
        }

        try:
            start_time = datetime.now()

            # Module execution retains the caller's data/configuration cwd.
            # Safe-path mode also prevents a workspace gnn/ from shadowing
            # the installed/importable pipeline package (API launch contract).
            workspace = Path.cwd()
            expected_run_id = uuid.uuid4().hex
            child_environment = os.environ.copy()
            child_environment["GNN_RUN_ID"] = expected_run_id
            child_environment.pop("GNN_RUN_CONTEXT_FILE", None)
            cmd: list[Any] = [
                sys.executable,
                "-P",
                "-m",
                "gnn.main",
                "--target-dir",
                str(workspace / "input/gnn_files"),
                "--output-dir",
                str(workspace / "output"),
                "--only-steps",
                ",".join(steps),
                "--verbose",
            ]

            self.logger.info(f"Executing: {' '.join(cmd)}")

            result = subprocess.run(  # nosec B603
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=workspace,
                env=child_environment,
            )

            end_time = datetime.now()
            test_results["execution_time"] = (end_time - start_time).total_seconds()

            # Parse execution results
            if result.returncode == 0:
                self.logger.info(
                    "Pipeline subprocess exited with code 0; checking receipt"
                )
            else:
                self.logger.error(
                    f"❌ Pipeline execution failed with code {result.returncode}"
                )

            # Count warnings and errors in output
            combined_output = result.stdout + result.stderr
            test_results["warnings_count"] = combined_output.lower().count("warning")
            test_results["errors_count"] = combined_output.lower().count("error")
            test_results["stdout"] = result.stdout
            test_results["stderr"] = result.stderr

            # Canonical current-run identity admits evidence; directory
            # presence or timestamps cannot make another invocation current.
            summary_file = (
                workspace / "output/00_pipeline_summary/pipeline_execution_summary.json"
            )
            if not summary_file.is_file():
                raise ValueError("No current-invocation pipeline summary is available")
            with summary_file.open(encoding="utf-8") as handle:
                summary_data = json.load(handle)
            if (
                not isinstance(summary_data, dict)
                or summary_data.get("run_id") != expected_run_id
            ):
                raise ValueError(
                    "Pipeline summary does not belong to the current invocation"
                )
            summary_steps = summary_data.get("steps")
            if not isinstance(summary_steps, list) or not all(
                isinstance(step, dict) for step in summary_steps
            ):
                raise ValueError("Current pipeline summary has invalid step records")
            current_steps = {}
            for step_data in summary_steps:
                step_name = step_data.get("script_name", "unknown")
                current_steps[step_name] = {
                    "status": step_data.get("status", "unknown"),
                    "duration": step_data.get("duration_seconds", 0),
                    "exit_code": step_data.get("exit_code", -1),
                    "memory_usage": step_data.get("memory_usage_mb", 0),
                }
            test_results["step_results"] = current_steps
            test_results["execution_successful"] = result.returncode == 0 and bool(
                current_steps
            )

        except subprocess.TimeoutExpired:
            self.logger.error("❌ Pipeline execution timed out")
            test_results["error"] = "Execution timed out"
        except Exception as e:
            self.logger.error(f"❌ Pipeline execution error: {e}")
            test_results["error"] = str(e)

        return test_results

    def validate_success_status(
        self, execution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate that pipeline steps achieve SUCCESS status instead of SUCCESS_WITH_WARNINGS."""
        self.logger.info("📊 Validating SUCCESS status achievement...")

        validation: dict[str, Any] = {
            "success_rate_improved": False,
            "warnings_reduced": False,
            "no_critical_failures": False,
            "step_analysis": {},
        }

        try:
            if execution_results.get("execution_successful") and execution_results.get(
                "step_results"
            ):
                step_results = execution_results["step_results"]

                # Analyze step statuses
                success_count = 0
                success_with_warnings_count = 0
                failed_count = 0

                for step_name, result in step_results.items():
                    status = result.get("status", "unknown")
                    validation["step_analysis"][step_name] = status

                    if status == "SUCCESS":
                        success_count += 1
                    elif status == "SUCCESS_WITH_WARNINGS":
                        success_with_warnings_count += 1
                    elif status == "FAILED":
                        failed_count += 1

                total_steps = len(step_results)
                success_rate = success_count / total_steps if total_steps > 0 else 0

                # Consider improvement if success rate is high
                validation["success_rate_improved"] = success_rate >= 0.7
                validation["warnings_reduced"] = (
                    success_with_warnings_count <= total_steps * 0.3
                )
                validation["no_critical_failures"] = failed_count == 0

                self.logger.info(
                    f"📈 Success rate: {success_rate:.1%} ({success_count}/{total_steps})"
                )
                self.logger.info(f"⚠️ Warnings: {success_with_warnings_count}")
                self.logger.info(f"❌ Failures: {failed_count}")

            else:
                self.logger.warning("⚠️ No execution results to validate")

        except Exception as e:
            self.logger.error(f"❌ Error validating success status: {e}")

        return validation

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation and improvement report."""
        self.logger.info("📋 Generating comprehensive validation report...")

        report: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "validator_version": "1.0.0",
            "improvements_validated": False,
            "pipeline_health": "unknown",
            "recommendations": [],
        }

        try:
            # 1. Validate code generation fixes
            code_fixes = self.validate_code_generation_fixes()
            report["code_generation_fixes"] = code_fixes

            # 2. Test dependency validation
            dependency_test = self.test_dependency_validation()
            report["dependency_validation"] = dependency_test

            # 3. Test pipeline execution
            execution_test = self.test_pipeline_execution()
            report["pipeline_execution"] = execution_test

            # 4. Validate success status
            success_validation = self.validate_success_status(execution_test)
            report["success_status_validation"] = success_validation

            # 5. Determine overall improvement status
            code_fixes_success = all(code_fixes.values())
            dependency_success = dependency_test.get("validation_successful", False)
            execution_success = execution_test.get("execution_successful", False)
            success_status_improved = any(success_validation.values())

            report["improvements_validated"] = (
                code_fixes_success
                and dependency_success
                and execution_success
                and success_status_improved
            )

            # 6. Calculate pipeline health
            if execution_test.get("step_results"):
                health_scores: list[Any] = []
                for result in execution_test["step_results"].values():
                    if result["status"] == "SUCCESS":
                        health_scores.append(1.0)
                    elif result["status"] == "SUCCESS_WITH_WARNINGS":
                        health_scores.append(0.7)
                    else:
                        health_scores.append(0.0)

                avg_health = (
                    sum(health_scores) / len(health_scores) if health_scores else 0
                )
                if avg_health >= 0.8:
                    report["pipeline_health"] = "excellent"
                elif avg_health >= 0.6:
                    report["pipeline_health"] = "good"
                elif avg_health >= 0.4:
                    report["pipeline_health"] = "fair"
                else:
                    report["pipeline_health"] = "poor"

            # 7. Generate recommendations
            recommendations: list[Any] = []

            if not code_fixes_success:
                recommendations.append("Complete remaining code generation fixes")

            if execution_test.get("warnings_count", 0) > 0:
                recommendations.append(
                    f"Address {execution_test['warnings_count']} remaining warnings"
                )

            if execution_test.get("errors_count", 0) > 0:
                recommendations.append(
                    f"Fix {execution_test['errors_count']} remaining errors"
                )

            if not success_status_improved:
                recommendations.append("Continue improving SUCCESS status achievement")

            if not recommendations:
                recommendations.append("Pipeline improvements successfully validated!")

            report["recommendations"] = recommendations

            # Log final results
            status_emoji = {
                "excellent": "🌟",
                "good": "✅",
                "fair": "⚠️",
                "poor": "❌",
                "unknown": "❓",
            }.get(report["pipeline_health"], "❓")

            self.logger.info(
                f"{status_emoji} Pipeline Health: {report['pipeline_health'].upper()}"
            )
            self.logger.info(
                f"🔧 Improvements Validated: {'YES' if report['improvements_validated'] else 'NO'}"
            )

        except Exception as e:
            self.logger.error(f"❌ Error generating report: {e}")
            report["error"] = str(e)

        return report

    def save_validation_report(
        self, report: Dict[str, Any], output_path: (Path) | None = None
    ) -> Path:
        """Save validation report to file."""
        if output_path is None:
            output_path = Path("output/pipeline_runtime_validator_report.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"📄 Validation report saved to: {output_path}")
        return output_path


def main() -> int:
    """Main function to run comprehensive pipeline validation."""
    validator = PipelineValidator(verbose=True)

    main_logger = logging.getLogger(__name__)
    main_logger.info("🔍 Starting comprehensive pipeline validation...")
    main_logger.info("=" * 60)

    # Generate comprehensive report
    report = validator.generate_comprehensive_report()

    # Save report
    report_path = validator.save_validation_report(report)

    # Print summary
    main_logger.info("\n" + "=" * 60)
    main_logger.info("📊 VALIDATION SUMMARY")
    main_logger.info("=" * 60)
    main_logger.info(f"Pipeline Health: {report['pipeline_health'].upper()}")
    main_logger.info(
        f"Improvements Validated: {'✅ YES' if report['improvements_validated'] else '❌ NO'}"
    )
    main_logger.info(f"Report saved to: {report_path}")

    if report.get("recommendations"):
        main_logger.info("\n🎯 RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            main_logger.info(f"  {i}. {rec}")

    return 0 if report["improvements_validated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
