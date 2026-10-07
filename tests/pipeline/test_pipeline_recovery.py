#!/usr/bin/env python3
"""
Pipeline Recovery Tests

This module provides comprehensive testing for pipeline error recovery scenarios,
focusing on known failure modes and their recovery mechanisms.

Key test areas:
1. NumPy recursion error recovery
2. Async/await error handling
3. Lightweight processing recovery
4. Hardware initialization issues
5. Resource management
6. Error reporting
"""

import asyncio
import inspect
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Uses real implementations per testing policy.

# Import test utilities

# Test markers
pytestmark = [pytest.mark.pipeline, pytest.mark.recovery]


@pytest.fixture
def test_environment() -> Any:
    """Create an isolated environment for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Create necessary subdirectories
        (temp_path / "input").mkdir()
        (temp_path / "output").mkdir()
        yield temp_path


@pytest.fixture
def sample_gnn_file(test_environment: Any) -> Any:
    """Create a sample GNN file for testing."""
    file_path = test_environment / "input" / "test_model.md"
    file_path.write_text("""
    # ModelName
    Test Model

    # StateSpaceBlock
    s[3,1]

    # Connections
    s -> s
    """)
    return file_path


class TestRecursionErrorRecovery:
    """Test suite for NumPy recursion error recovery."""

    def test_numpy_import_recovery(self, test_environment: Any) -> Any:
        """Test recovery from NumPy import recursion error.

        Patch numpy.typing module import itself to raise RecursionError first,
        then increase recursion limit and import again to simulate recovery.
        """
        # Real import path: attempt a controlled recursion by tweaking recursionlimit first
        original_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(1000)
            with pytest.raises(RecursionError):
                # Force a recursion via a small function to simulate failure scenario
                def _recur(n: Any) -> Any:
                    return 1 if n == 0 else _recur(n - 1) + _recur(n - 1)

                _recur(2000)
        finally:
            sys.setrecursionlimit(original_limit)

        # Should recover by increasing recursion limit
        sys.setrecursionlimit(3000)
        # Remove any cached failed import so retry occurs
        sys.modules.pop("numpy.typing", None)
        import numpy as np  # noqa: F401

        assert "numpy" in sys.modules

    def test_render_step_recovery(
        self, test_environment: Any, sample_gnn_file: Any
    ) -> Any:
        """Test render step recovery from recursion errors."""
        from tests.helpers.render_recovery import render_gnn_files

        # Execute real render; recovery should be handled internally if needed
        result = render_gnn_files(
            target_dir=test_environment / "input",
            output_dir=test_environment / "output",
        )

        assert result["status"] == "SUCCESS"
        assert isinstance(result["recovery_actions"], list)


class TestAsyncAwaitRecovery:
    """Test suite for async/await error recovery."""

    @pytest.mark.slow
    def test_llm_analysis_recovery(
        self, test_environment: Any, sample_gnn_file: Any
    ) -> Any:
        """Test LLM analysis with proper async/await handling."""
        from gnn.llm.analyzer import analyze_gnn_file_with_llm

        # Real call: skip if no local providers and no keys available
        if not (
            os.getenv("OPENAI_API_KEY") or os.getenv("OLLAMA_DISABLED", "0") == "0"
        ):
            raise AssertionError(
                "No LLM providers available (no API keys and Ollama disabled)"
            )
        maybe_result = analyze_gnn_file_with_llm(sample_gnn_file)
        result = (
            asyncio.run(maybe_result)
            if inspect.isawaitable(maybe_result)
            else maybe_result
        )

        assert result["status"] == "SUCCESS"
        assert "analysis" in result

    @pytest.mark.slow
    def test_sync_wrapper_recovery(
        self, test_environment: Any, sample_gnn_file: Any
    ) -> Any:
        """Test synchronous wrapper for async LLM analysis."""
        from gnn.llm.analyzer import analyze_gnn_file_with_llm

        # Real call: skip if no local providers and no keys available
        if not (
            os.getenv("OPENAI_API_KEY") or os.getenv("OLLAMA_DISABLED", "0") == "0"
        ):
            raise AssertionError(
                "No LLM providers available (no API keys and Ollama disabled)"
            )
        result = analyze_gnn_file_with_llm(sample_gnn_file)

        assert isinstance(result, dict)
        assert result["status"] == "SUCCESS"
        assert "analysis" in result


class TestLightweightProcessingRecovery:
    """Test suite for lightweight processing recovery."""

    def test_gnn_lightweight_fallback(
        self, test_environment: Any, sample_gnn_file: Any
    ) -> Any:
        """Test recovery to lightweight GNN processing."""
        from gnn.processing.core_processor import process_gnn_directory

        # Call real method; it should choose a mode based on availability
        result = process_gnn_directory(
            test_environment / "input", test_environment / "output"
        )

        assert result["status"] == "SUCCESS"
        assert result["processing_mode"] == "lightweight"
        assert str(sample_gnn_file) in result["processed_files"]

    def test_lightweight_processing_output(
        self, test_environment: Any, sample_gnn_file: Any
    ) -> Any:
        """Test output quality of lightweight processing."""
        from gnn.processing.core_processor import process_gnn_directory_lightweight

        result = process_gnn_directory_lightweight(test_environment / "input")

        assert str(sample_gnn_file) in result
        assert result[str(sample_gnn_file)]["status"] == "processed"
        assert result[str(sample_gnn_file)]["format"] == "markdown"
        assert isinstance(result[str(sample_gnn_file)]["size"], int)


class TestHardwareInitialization:
    """Test suite for hardware initialization."""

    def test_jax_devices(self, test_environment: Any) -> Any:
        """Test real JAX device discovery."""
        from gnn.execute.jax.jax_runner import initialize_jax_devices

        devices = initialize_jax_devices()

        if not devices:
            raise AssertionError("JAX devices unavailable in this environment")
        assert str(devices[0])

    def test_execution_hardware_recovery(
        self, test_environment: Any, monkeypatch: pytest.MonkeyPatch
    ) -> Any:
        """A rendered Python workload executes on an actual CPU device."""
        from gnn.execute.executor import execute_gnn_model

        script = test_environment / "input" / "hardware_jax.py"
        script.write_text("import jax\nprint(jax.devices('cpu')[0].platform)\n")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")
        result = execute_gnn_model(
            str(script), "jax", {"output_dir": str(test_environment / "output")}
        )

        assert result["status"] == "SUCCESS"
        assert result["success"] is True
        assert result["return_code"] == 0
        assert result["stdout"].strip() == "cpu"
        assert result["execution_device"] == "cpu"


class TestResourceManagementRecovery:
    """Test suite for resource management recovery."""

    def test_memory_limit_recovery(self, test_environment: Any) -> Any:
        """Test recovery from memory limit issues."""
        from gnn.utils.runtime_safety.resource_manager import with_resource_limits

        @with_resource_limits(max_memory_mb=100)
        def memory_intensive_operation() -> Any:
            # Simulate memory-intensive operation
            [0] * (1024 * 1024)  # 1MB
            return "Success"

        result = memory_intensive_operation()
        assert result == "Success"

    def test_disk_space_recovery(self, test_environment: Any) -> Any:
        """Test recovery from disk space issues."""
        from gnn.utils.runtime_safety.resource_manager import check_disk_space

        # Real check: should succeed given temp dir has space; if not, skip gracefully
        try:
            assert check_disk_space(test_environment, required_mb=1)
        except RuntimeError:
            raise AssertionError("Insufficient disk space on test system")


class TestErrorReportingRecovery:
    """Test suite for error reporting mechanisms."""

    def test_error_collection(self, test_environment: Any) -> Any:
        """Test error collection and reporting."""
        from gnn.utils.errors.error_recovery import ErrorReporter

        reporter = ErrorReporter()

        # Simulate error collection
        reporter.collect_error("test_error", "Test error message")

        errors = reporter.get_errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "test_error"
        assert errors[0]["message"] == "Test error message"

    def test_error_recovery_logging(self, test_environment: Any) -> Any:
        """Test error recovery logging functionality."""
        import logging

        from gnn.utils.logging.logging_utils import log_step_error

        # Create a test logger
        logger = logging.getLogger("test_error_recovery")

        # Test error logging - log_step_error takes (logger, message) and returns None
        # It logs the error and doesn't raise an exception
        result = log_step_error(logger, "Test error occurred")

        # The structured logging version returns a status dict
        assert isinstance(result, dict)
        assert result["status"] == "ERROR"
        assert result["message"] == "Test error occurred"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
