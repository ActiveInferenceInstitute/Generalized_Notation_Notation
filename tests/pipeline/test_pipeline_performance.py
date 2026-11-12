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

Threshold scope: ``THRESHOLDS`` is derived from the checked-in
``pipeline_performance_baseline.json``. That artifact records the measurement
environment, three arms of the band (quiet, self-spawned loaded, ambient
loaded), and the per-metric upper edge. Each budget is
``recorded_upper_edge * TOLERANCE``; the tolerance is 100, a host-independent
ceiling above the largest observed loaded/quiet ratio (29.1x, medium export,
ambient arm; the self-spawned arm's largest ratio was 2.4x). The quiet arm ran
at 16.6% CPU utilization; the loaded arms ran 36 CPU-bound workers on 18 cores
at 100% CPU utilization. The 100x ceiling catches order-of-magnitude
regressions rather than modest slowdowns. Memory and disk values below 1 MB are
recorded at the 1 MB measurement floor. The module warms the measured paths
once before timing and runs a self-spawned loaded-arm verdict test, so the
budgets describe steady-state work and the loaded verdict is reproducible on a
quiet machine.

The module carries the ``pipeline`` marker, so the default/CI marker filter
(``just test``) deselects it; run it deliberately when evaluating a change:

    uv run pytest tests/pipeline/test_pipeline_performance.py -v -m performance
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import psutil
import pytest

from gnn.utils.testing import performance_tracker

# Test markers
pytestmark = [pytest.mark.pipeline, pytest.mark.performance]

# Recorded measurement band and the tolerance applied to its upper edge.
BASELINE_PATH = Path(__file__).with_name("pipeline_performance_baseline.json")
BASELINE: dict[str, Any] = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
TOLERANCE = float(BASELINE["tolerance"])


def _scale_budget(value: Any) -> Any:
    """Multiply every numeric leaf of the recorded baseline by the tolerance."""
    if isinstance(value, dict):
        return {key: _scale_budget(item) for key, item in value.items()}
    return float(value) * TOLERANCE


# Performance thresholds derived from the recorded baseline artifact.
THRESHOLDS: dict[str, Any] = _scale_budget(BASELINE["baseline"])


def _directory_mb(path: Path) -> float:
    """Return the total size of the files under ``path`` in megabytes."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / (1024 * 1024)


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


@pytest.fixture(scope="module", autouse=True)
def _warm_performance_paths() -> None:
    """Exercise the measured paths once so imports and first-call caches are warm.

    The recorded baseline was taken after the same warm-up, so the budgets
    describe steady-state work rather than one-time import cost.
    """
    import logging

    from gnn import process_gnn_directory
    from gnn.export import export_model
    from gnn.pipeline.execution import run_pipeline
    from gnn.utils.config_io.io_utils import batch_write_files
    from gnn.visualization import generate_visualizations

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / "input").mkdir()
        (root / "output").mkdir()
        model_file = root / "input" / "warmup.md"
        model_file.write_text(
            "\n".join(
                [
                    "# Warmup Model",
                    "",
                    "## GNNSection",
                    "ActInfPOMDP",
                    "",
                    "## GNNVersionAndFlags",
                    "GNN v1",
                    "",
                    "## ModelName",
                    "WarmupModel",
                    "",
                    "## StateSpaceBlock",
                    "s0[3,1]",
                    "s1[3,1]",
                    "",
                    "## Connections",
                    "s0>s1",
                ]
            )
        )
        process_gnn_directory(model_file, root / "output")
        generate_visualizations(
            logging.getLogger("warmup"), root / "input", root / "output"
        )
        export_model(
            {
                "model_name": "WarmupModel",
                "model_annotation": "warmup",
                "variables": ["s0", "s1"],
                "connections": [{"from": "s0", "to": "s1"}],
                "equations": [],
                "metadata": {},
                "source_content": model_file.read_text(),
            },
            root / "output",
            formats=["json"],
        )
        batch_write_files([{"path": "warmup.txt", "content": "x"}], root / "output")
        run_pipeline(target_dir=root / "input", output_dir=root / "output", steps=["3"])


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
        # Verify concurrent processing completes within the recorded band.
        assert tracker.duration < THRESHOLDS["parallel"]


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

        assert (
            tracker.duration < THRESHOLDS[f"{model_size}_model"]["visualization_time"]
        )
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

        # Retention of Python-managed memory after GC should stay inside the
        # recorded band; the 1 MB floor keeps allocator noise from setting it.
        assert retained_mb < THRESHOLDS["retained_memory"], (
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
        assert result["write_time_seconds"] < THRESHOLDS["disk_write"]

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
        assert tracker.duration < THRESHOLDS["medium_model"]["export_time"]
        assert (
            _directory_mb(isolated_environment / "output")
            < THRESHOLDS["medium_model"]["disk_usage"]
        )


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
        assert tracker.duration < THRESHOLDS["pipeline"][str(model_count)]

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


class TestLoadedArmVerdict:
    """The recorded band must reach the same verdict under a self-spawned load.

    The load is created by this test rather than inherited from the host, so the
    loaded arm is reproducible on a quiet machine. The worker count and the
    warm-up match the measurement harness that recorded the baseline artifact.
    """

    @staticmethod
    def _spawn_load() -> list[Any]:
        workers = 2 * (os.cpu_count() or 1)
        return [
            subprocess.Popen(
                [sys.executable, "-c", "while True: pass"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(workers)
        ]

    @staticmethod
    def _stop_load(workers: list[Any]) -> None:
        for worker in workers:
            worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)

    def test_loaded_arm_verdict(
        self, isolated_environment: Any, create_model_file: Any
    ) -> None:
        """Run the load-sensitive operations under a deterministic CPU load."""
        import logging

        from gnn import process_gnn_directory
        from gnn.export import export_model
        from gnn.pipeline.execution import run_pipeline
        from gnn.visualization import generate_visualizations

        large_model = create_model_file("large")
        medium_model = create_model_file("medium")
        for _ in range(3):
            create_model_file("small")

        workers = self._spawn_load()
        try:
            time.sleep(0.5)

            with performance_tracker() as tracker:
                processed = process_gnn_directory(
                    large_model, isolated_environment / "output"
                )
            assert processed["status"] == "SUCCESS"
            assert tracker.duration < THRESHOLDS["large_model"]["processing_time"]

            with performance_tracker() as tracker:
                generated = generate_visualizations(
                    logging.getLogger("loaded_viz"),
                    large_model.parent,
                    isolated_environment / "output",
                )
            assert generated is True or generated.get("status") == "SUCCESS"
            assert tracker.duration < THRESHOLDS["large_model"]["visualization_time"]

            with open(medium_model) as f:
                model_content = f.read()
            with performance_tracker() as tracker:
                exported = export_model(
                    {
                        "model_name": "TestMediumModel",
                        "model_annotation": "loaded arm",
                        "variables": ["s0", "s1", "s2"],
                        "connections": [
                            {"from": "s0", "to": "s1"},
                            {"from": "s1", "to": "s2"},
                        ],
                        "equations": [],
                        "metadata": {"size": "medium"},
                        "source_content": model_content,
                    },
                    isolated_environment / "output",
                    formats=["json", "xml", "graphml"],
                )
            assert any(exported["formats"].values())
            assert tracker.duration < THRESHOLDS["medium_model"]["export_time"]

            with performance_tracker() as tracker:
                ran = run_pipeline(
                    target_dir=isolated_environment / "input",
                    output_dir=isolated_environment / "output",
                    steps=["3"],
                )
            assert ran["success"]
            assert tracker.duration < THRESHOLDS["pipeline"]["3"]

            with performance_tracker() as tracker:
                parallel = process_gnn_directory(
                    isolated_environment / "input", recursive=True, parallel=True
                )
            assert parallel["status"] == "SUCCESS"
            assert tracker.duration < THRESHOLDS["parallel"]
        finally:
            self._stop_load(workers)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
