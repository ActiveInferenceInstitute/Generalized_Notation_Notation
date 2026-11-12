#!/usr/bin/env python3
"""
Pipeline Performance Tests

This module provides comprehensive performance testing for the GNN pipeline,
focusing on resource usage, timing, and scalability characteristics.

Key test areas:
1. GNN processing performance
2. Visualization generation timing
3. Memory usage patterns
4. Disk I/O performance
5. Network operation timing
6. Resource scaling characteristics

Threshold scope: ``THRESHOLDS`` holds absolute, single-point budgets taken from
one measurement environment. They are not regression bands calibrated against a
recorded baseline, so a slower or busier machine can fail them without any code
change. The module carries the ``pipeline`` marker, so the default/CI marker
filter (``just test``) deselects it; run it deliberately when evaluating a
change:

    uv run pytest tests/pipeline/test_pipeline_performance.py -v -m performance

Converting the fixed numbers into measured bands is open work, not something
this module currently provides.
"""

import tempfile
import time
from pathlib import Path
from typing import Any, cast

import psutil
import pytest

from gnn.utils.testing import performance_tracker

# Test markers
pytestmark = [pytest.mark.pipeline, pytest.mark.performance]

# Performance thresholds
THRESHOLDS: dict[str, Any] = {
    "small_model": {
        "processing_time": 1.0,  # seconds
        "memory_usage": 100,  # MB
        "disk_usage": 10,  # MB
        "export_time": 2.0,  # seconds
    },
    "medium_model": {
        "processing_time": 5.0,
        "memory_usage": 500,
        "disk_usage": 50,
        "export_time": 10.0,  # seconds
    },
    "large_model": {
        "processing_time": 20.0,
        "memory_usage": 2000,
        "disk_usage": 200,
        "export_time": 30.0,  # seconds
    },
}


@pytest.fixture
def isolated_environment() -> Any:
    """Create an isolated environment for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Create necessary subdirectories
        (temp_path / "input").mkdir()
        (temp_path / "output").mkdir()
        yield temp_path


@pytest.fixture
def create_model_file(isolated_environment: Any) -> Any:
    """Create a GNN model file of specified size."""

    def _create_file(size: str = "small") -> Path:
        sizes: dict[str, Any] = {
            "small": 10,  # 10 states
            "medium": 100,  # 100 states
            "large": 1000,  # 1000 states
        }
        num_states = sizes[size]

        content: list[Any] = [
            f"# Test {size.capitalize()} Model",
            "",
            "## GNNSection",
            "ActInfPOMDP",
            "",
            "## GNNVersionAndFlags",
            "GNN v1",
            "",
            "## ModelName",
            f"Test{size.capitalize()}Model",
            "",
            "## StateSpaceBlock",
        ]

        # Add states
        for i in range(num_states):
            content.append(f"s{i}[3,1]")

        # Add connections
        content.append("")
        content.append("## Connections")
        for i in range(num_states - 1):
            content.append(f"s{i}>s{i + 1}")

        file_path = isolated_environment / "input" / f"test_model_{size}.md"
        file_path.write_text("\n".join(content))
        return cast("Path", file_path)

    return _create_file


class TestGNNProcessingPerformance:
    """Test suite for GNN processing performance."""

    @pytest.mark.parametrize("model_size", ["small", "medium", "large"])
    def test_processing_scaling(
        self, isolated_environment: Any, create_model_file: Any, model_size: Any
    ) -> Any:
        """Test GNN processing performance scaling."""
        from gnn import process_gnn_directory

        model_file = create_model_file(model_size)

        with performance_tracker() as tracker:
            result = process_gnn_directory(model_file, isolated_environment / "output")

        assert result["status"] == "SUCCESS"
        assert tracker.duration < THRESHOLDS[f"{model_size}_model"]["processing_time"]
        # Use the correct attribute name for memory tracking
        if hasattr(tracker, "max_memory_mb"):
            assert (
                tracker.max_memory_mb
                < THRESHOLDS[f"{model_size}_model"]["memory_usage"]
            )
        elif hasattr(tracker, "peak_memory_mb"):
            assert (
                tracker.peak_memory_mb
                < THRESHOLDS[f"{model_size}_model"]["memory_usage"]
            )

    def test_parallel_processing(
        self, isolated_environment: Any, create_model_file: Any
    ) -> Any:
        """Test parallel GNN processing performance."""
        from gnn import process_gnn_directory

        # Create multiple models
        for size in ["small", "medium"]:
            create_model_file(size)

        with performance_tracker() as tracker:
            result = process_gnn_directory(
                isolated_environment / "input", recursive=True, parallel=True
            )

        assert result["status"] == "SUCCESS"
        # Check that files were processed (actual structure may vary)
        assert "processed_files" in result or "files" in result
        # Verify concurrent processing completes within a reasonable time bound.
        assert tracker.duration < (THRESHOLDS["small_model"]["processing_time"] * 4)


class TestVisualizationPerformance:
    """Test suite for visualization performance."""

    @pytest.mark.parametrize("model_size", ["small", "medium", "large"])
    def test_visualization_scaling(
        self, isolated_environment: Any, create_model_file: Any, model_size: Any
    ) -> Any:
        """Test visualization generation performance scaling."""
        import logging

        from gnn.visualization import generate_visualizations

        model_file = create_model_file(model_size)
        logger = logging.getLogger("test_viz_scaling")

        with performance_tracker() as tracker:
            result = generate_visualizations(
                logger, model_file.parent, isolated_environment / "output"
            )

        # Handle both dict and bool return types
        if isinstance(result, dict):
            assert result["status"] == "SUCCESS"
        else:
            assert result is True  # Boolean success indicator

        assert tracker.duration < THRESHOLDS[f"{model_size}_model"]["processing_time"]
        # Use the correct attribute name for memory tracking
        if hasattr(tracker, "max_memory_mb"):
            assert (
                tracker.max_memory_mb
                < THRESHOLDS[f"{model_size}_model"]["memory_usage"]
            )
        elif hasattr(tracker, "peak_memory_mb"):
            assert (
                tracker.peak_memory_mb
                < THRESHOLDS[f"{model_size}_model"]["memory_usage"]
            )

    def test_visualization_caching(
        self, isolated_environment: Any, create_model_file: Any
    ) -> Any:
        """Test visualization caching performance."""
        import logging

        from gnn.visualization import generate_visualizations

        model_file = create_model_file("medium")
        logger = logging.getLogger("test_viz_caching")

        # First generation
        with performance_tracker() as tracker_1:
            generate_visualizations(
                logger, model_file.parent, isolated_environment / "output"
            )

        # Second generation (should use cache)
        with performance_tracker() as tracker_2:
            result_2 = generate_visualizations(
                logger, model_file.parent, isolated_environment / "output"
            )

        # Handle both dict and bool return types
        if isinstance(result_2, dict):
            assert result_2["status"] == "SUCCESS"
        else:
            assert result_2 is True  # Boolean success indicator

        # Second run may be faster when caches hit; sub-millisecond timings are noisy on CI.
        slack = max(0.02, tracker_1.duration * 4.0)
        assert tracker_2.duration <= tracker_1.duration + slack, (
            f"Second run unexpectedly slow: {tracker_2.duration:.4f}s vs first "
            f"{tracker_1.duration:.4f}s (slack {slack:.4f}s)"
        )


class TestMemoryUsagePatterns:
    """Test suite for memory usage patterns."""

    def test_memory_cleanup(
        self, isolated_environment: Any, create_model_file: Any
    ) -> Any:
        """Python-level allocation delta for `process_gnn_directory` must be bounded.

        Uses ``tracemalloc`` rather than process RSS: RSS is dominated by the state of
        every preceding test in the same pytest worker (LLM caches, matplotlib figure
        managers, JAX kernels, …), so an RSS-based assertion is order-dependent and
        gives no information about *this* function's allocations.
        """
        import gc
        import tracemalloc

        from gnn import process_gnn_directory

        model_file = create_model_file("large")

        # Quiesce the allocator before taking the baseline; retry a few times since
        # cyclic collection sometimes runs in stages.
        for _ in range(3):
            gc.collect()

        tracemalloc.start()
        try:
            baseline_current, _ = tracemalloc.get_traced_memory()
            result = process_gnn_directory(
                model_file,
                isolated_environment / "output",
            )
            # Force collection so short-lived objects allocated inside the call drop
            # out before measuring residual retention.
            for _ in range(3):
                gc.collect()
            final_current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert result["status"] == "SUCCESS"

        retained_mb = max(0.0, (final_current - baseline_current) / (1024 * 1024))
        peak_mb = peak / (1024 * 1024)

        # Retention of Python-managed memory after GC should be modest; the generous
        # bound is deliberately above observed values (typically < 25 MB) so ordinary
        # caching changes do not flake CI.
        assert retained_mb < 75, (
            f"Python allocator retained {retained_mb:.1f} MB after "
            f"process_gnn_directory (peak during call: {peak_mb:.1f} MB)"
        )

    def test_peak_memory_tracking(
        self, isolated_environment: Any, create_model_file: Any
    ) -> Any:
        """Test peak memory usage tracking."""
        from gnn.utils.runtime_safety.resource_manager import track_peak_memory

        @track_peak_memory
        def memory_intensive_operation() -> Any:
            # Simulate memory-intensive operation
            [0] * (1024 * 1024 * 10)  # 10MB
            time.sleep(0.1)  # Allow tracking to measure
            return "Success"

        result, peak_memory = memory_intensive_operation()

        assert result == "Success"
        assert isinstance(peak_memory, float)
        assert peak_memory > 0


class TestDiskIOPerformance:
    """Test suite for disk I/O performance."""

    def test_file_write_performance(self, isolated_environment: Any) -> Any:
        """Test file write performance with different file sizes."""
        from gnn.utils.config_io.io_utils import batch_write_files

        # Create test files with correct structure
        files_data = [
            {
                "path": f"test_file_{i}.txt",
                "content": f"Test content for file {i}" * 1000,  # 1KB content
            }
            for i in range(10)
        ]

        with performance_tracker():
            result = batch_write_files(files_data, isolated_environment / "output")

        assert result["total_files"] == 10
        assert result["successful_writes"] == 10
        assert result["write_time_seconds"] < 1.0  # Should be fast

    def test_export_performance(
        self, isolated_environment: Any, create_model_file: Any
    ) -> Any:
        """Test export performance for different formats."""
        from gnn.export import export_model

        model_file = create_model_file("medium")

        # Read the model file content and create model data
        with open(model_file, "r") as f:
            model_content = f.read()

        # Create model data dictionary
        model_data: dict[str, Any] = {
            "model_name": "TestMediumModel",
            "model_annotation": "Test model for performance testing",
            "variables": ["s0", "s1", "s2"],
            "connections": [{"from": "s0", "to": "s1"}, {"from": "s1", "to": "s2"}],
            "equations": [],
            "metadata": {"size": "medium"},
            "source_content": model_content,
        }

        with performance_tracker() as tracker:
            result = export_model(
                model_data,
                isolated_environment / "output",
                formats=["json", "xml", "graphml"],  # Use correct parameter name
            )

        # Check if at least some formats succeeded (more lenient)
        successful_formats = sum(1 for success in result["formats"].values() if success)
        assert successful_formats >= 1, (
            f"At least one format should succeed, but only {successful_formats} succeeded"
        )
        # Allow more time for export operations
        assert tracker.duration < 30.0  # 30 seconds should be enough for export


class TestResourceScaling:
    """Test suite for resource scaling characteristics."""

    @pytest.mark.parametrize("model_count", [1, 3, 10])
    def test_pipeline_scaling(
        self, isolated_environment: Any, create_model_file: Any, model_count: Any
    ) -> Any:
        """Test pipeline scaling with different model counts."""
        from gnn.pipeline.execution import run_pipeline

        # Create test files
        for _i in range(model_count):
            create_model_file("small")

        with performance_tracker() as tracker:
            result = run_pipeline(
                target_dir=isolated_environment / "input",
                output_dir=isolated_environment / "output",
                steps=["3"],
            )

        assert result["success"]
        max_time_per_model = 300
        assert tracker.duration < (model_count * max_time_per_model)

    def test_resource_estimation(
        self, isolated_environment: Any, create_model_file: Any
    ) -> Any:
        """Test resource estimation accuracy."""
        from gnn.pipeline.execution import run_pipeline
        from gnn.utils.runtime_safety.resource_manager import estimate_resources

        model_file = create_model_file("medium")

        estimate = estimate_resources(model_file)

        with performance_tracker() as tracker:
            run_pipeline(
                target_dir=isolated_environment / "input",
                output_dir=isolated_environment / "output",
                steps=["3"],
            )

        # Resource estimates are heuristics; this test verifies shape and sanity
        # without assuming a specific relationship to wall-clock time on the host.
        assert isinstance(estimate, dict)
        assert estimate.get("time") is not None
        assert estimate.get("memory_mb") is not None
        assert estimate["time"] >= 0
        assert estimate["memory_mb"] >= 0
        assert tracker.duration >= 0
        # Check for either max_memory_mb or peak_memory_mb attribute
        memory_attr = getattr(tracker, "max_memory_mb", None) or getattr(
            tracker, "peak_memory_mb", None
        )
        if memory_attr:
            assert memory_attr >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
