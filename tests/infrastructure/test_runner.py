"""
Test Runner for GNN Processing Pipeline.

Canonical single-source ``TestRunner``: monitors resources, builds and runs
pytest subprocesses, parses output, and writes execution reports.

``tests.runner`` re-exports this class so existing import paths keep working
(``from tests.runner import TestRunner``) keep resolving to the same
implementation. Do not add a second copy of the class elsewhere.
"""

import json
import logging
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .resource_monitor import ResourceMonitor
from .test_config import TestExecutionConfig, TestExecutionResult

# Calculate project root
project_root = Path(__file__).parent.parent.parent


class TestRunner:
    """Test runner with comprehensive monitoring and reporting."""

    def __init__(self, config: TestExecutionConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("test_runner")
        self.resource_monitor = ResourceMonitor(
            memory_limit_mb=config.memory_limit_mb,
            cpu_limit_percent=config.cpu_limit_percent,
        )
        self.execution_history: List[TestExecutionResult] = []
        self._history_lock = threading.Lock()

    def run_tests(
        self, test_paths: List[Path], output_dir: Path
    ) -> TestExecutionResult:
        """Execute tests with comprehensive monitoring."""
        start_time = time.time()

        try:
            # Start resource monitoring
            self.resource_monitor.start_monitoring()

            # Build pytest command
            cmd = self._build_pytest_command(test_paths, output_dir)

            # Execute tests
            result = self._execute_pytest(cmd, output_dir)

            # Stop monitoring and get stats
            self.resource_monitor.stop_monitoring()
            resource_stats = self.resource_monitor.get_stats()

            # Create execution result
            execution_result = TestExecutionResult(
                success=result["success"],
                tests_run=result["tests_run"],
                tests_passed=result["tests_passed"],
                tests_failed=result["tests_failed"],
                tests_skipped=result["tests_skipped"],
                execution_time=time.time() - start_time,
                memory_peak_mb=resource_stats["peak_memory_mb"],
                coverage_percentage=result.get("coverage_percentage"),
                error_message=result.get("error_message"),
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", ""),
            )

            # Store in history (thread-safe; run_tests may be called from
            # concurrent contexts).
            with self._history_lock:
                self.execution_history.append(execution_result)

            return execution_result

        except Exception as e:
            self.resource_monitor.stop_monitoring()
            return TestExecutionResult(
                success=False,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                execution_time=time.time() - start_time,
                memory_peak_mb=0.0,
                error_message=str(e),
            )
        finally:
            self.resource_monitor.stop_monitoring()

    def _build_pytest_command(
        self, test_paths: List[Path], output_dir: Path
    ) -> List[str]:
        """Build pytest command with appropriate options."""
        cmd: list[Any] = [
            sys.executable,
            "-m",
            "pytest",
            "--verbose",
            "--tb=short",
            "--log-cli-level=WARNING",
            f"--maxfail={self.config.max_failures}",
            "--durations=10",
            "--disable-warnings",
        ]

        # Add markers
        if self.config.markers:
            for marker in self.config.markers:
                cmd.extend(["-m", marker])

        # Add coverage if enabled
        if self.config.coverage:
            cov_json = output_dir / "coverage.json"
            cov_html = output_dir / "htmlcov"
            cmd.extend(
                [
                    "--cov=gnn",
                    f"--cov-report=json:{cov_json}",
                    f"--cov-report=html:{cov_html}",
                    "--cov-report=term-missing",
                ]
            )

        # Add test paths
        cmd.extend([str(path) for path in test_paths])

        return cmd

    def _execute_pytest(self, cmd: List[str], output_dir: Path) -> Dict[str, Any]:
        """Execute pytest command and capture results."""
        try:
            # Debug: Log what we're trying to do
            self.logger.debug(f"Creating output directory: {output_dir}")
            self.logger.debug(f"Output dir type: {type(output_dir)}")

            # Create output files
            stdout_file = output_dir / "pytest_stdout.txt"
            stderr_file = output_dir / "pytest_stderr.txt"

            # Execute command with streaming
            from gnn.utils.pipeline_orchestration.execution_utils import (
                execute_command_streaming,
            )

            result = execute_command_streaming(
                cmd,
                cwd=project_root,
                timeout=self.config.timeout_seconds,
                print_stdout=True,
                print_stderr=True,
                capture_output=True,
            )

            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")

            # Save output to files
            with open(stdout_file, "w") as f:
                f.write(stdout)
            with open(stderr_file, "w") as f:
                f.write(stderr)

            if result["status"] == "TIMEOUT":
                return {
                    "success": False,
                    "tests_run": 0,
                    "tests_passed": 0,
                    "tests_failed": 0,
                    "tests_skipped": 0,
                    "error_message": f"Test execution timed out after {self.config.timeout_seconds} seconds",
                    "stdout": stdout,
                    "stderr": stderr,
                }

            # Parse results
            results = self._parse_pytest_output(stdout, stderr)
            process_success = (
                result.get("status") == "SUCCESS" and result.get("exit_code") == 0
            )
            results["success"] = bool(results["success"] and process_success)
            if not process_success:
                results["error_message"] = result.get("cleanup_error") or (
                    f"pytest exited with status {result.get('status')} and code {result.get('exit_code')}"
                )
            results["stdout"] = stdout
            results["stderr"] = stderr

            return results

        except Exception as e:
            import traceback

            self.logger.error(f"Exception in _execute_pytest: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return {
                "success": False,
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "tests_skipped": 0,
                "error_message": str(e),
                "stdout": "",
                "stderr": "",
            }

    def _parse_pytest_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse pytest output to extract test statistics."""
        try:
            # Extract test counts from output
            lines = stdout.split("\n")
            tests_run = 0
            tests_passed = 0
            tests_failed = 0
            tests_skipped = 0

            # Look for the summary line at the end (e.g., "60 passed, 20 skipped in 1.85s")
            for line in reversed(lines):
                if "passed" in line or "failed" in line or "skipped" in line:
                    # Try to extract numbers from the summary line
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if i > 0:  # Check previous part is a number
                            try:
                                num = int(parts[i - 1])
                                if "passed" in part:
                                    tests_passed = num
                                elif "failed" in part:
                                    tests_failed = num
                                elif "skipped" in part:
                                    tests_skipped = num
                            except (ValueError, IndexError):
                                # Not a number, skip
                                continue
                    # Calculate total from summary
                    if tests_passed > 0 or tests_failed > 0 or tests_skipped > 0:
                        tests_run = tests_passed + tests_failed + tests_skipped
                        break

            # Recovery: look for collected items line
            if tests_run == 0:
                for line in lines:
                    if "collected" in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "collected" and i > 0:
                                try:
                                    tests_run = int(parts[i - 1])
                                except ValueError:
                                    pass
                                break

            # Check for collection errors
            collection_errors: list[Any] = []
            for line in lines:
                if "ERROR collecting" in line or "ERROR: No tests collected" in line:
                    collection_errors.append(line)

            # Determine success - fail if no tests collected or collection errors
            success = tests_failed == 0 and tests_passed > 0 and not collection_errors

            # Extract coverage if present
            coverage_percentage = None
            for line in lines:
                if "TOTAL" in line and "%" in line:
                    try:
                        coverage_percentage = float(line.split()[-1].replace("%", ""))
                    except (ValueError, IndexError):
                        pass

            return {
                "success": success,
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "tests_skipped": tests_skipped,
                "coverage_percentage": coverage_percentage,
                "collection_errors": collection_errors,
            }

        except Exception as e:
            import traceback

            self.logger.error(f"Failed to parse pytest output: {e}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            return {
                "success": False,
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "tests_skipped": 0,
                "error_message": f"Failed to parse pytest output: {e}",
            }

    def generate_report(self, output_dir: Path) -> Dict[str, Any]:
        """Generate comprehensive test execution report."""
        with self._history_lock:
            if not self.execution_history:
                return {"error": "No test execution history available"}
            latest_result = self.execution_history[-1]

        report: dict[str, Any] = {
            "execution_summary": asdict(latest_result),
            "resource_usage": self.resource_monitor.get_stats(),
            "execution_history": [asdict(result) for result in self.execution_history],
            "performance_metrics": {
                "average_execution_time": sum(
                    r.execution_time for r in self.execution_history
                )
                / len(self.execution_history),
                "peak_memory_usage": max(
                    r.memory_peak_mb for r in self.execution_history
                ),
                "success_rate": sum(1 for r in self.execution_history if r.success)
                / len(self.execution_history)
                * 100,
            },
        }

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save report
        report_file = output_dir / "test_execution_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        return report
