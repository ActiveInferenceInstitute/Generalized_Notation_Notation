"""
Performance Utilities for AXIOM Implementation
=============================================

Implements performance tracking, timing, memory monitoring, and
efficiency metrics for AXIOM agent analysis and optimization.

Authors: AXIOM Research Team
Institution: VERSES AI / Active Inference Institute
"""

import json
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import psutil


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    # Timing metrics
    total_time: float = 0.0
    operation_times: Dict[str, float] = field(default_factory=dict)
    operation_counts: Dict[str, int] = field(default_factory=dict)

    # Memory metrics
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    memory_history: List[float] = field(default_factory=list)

    # Computational metrics
    total_operations: int = 0
    operations_per_second: float = 0.0

    # Model metrics
    parameter_count: int = 0
    model_complexity: Dict[str, int] = field(default_factory=dict)

    # Learning metrics
    learning_rate: float = 0.0
    convergence_metrics: Dict[str, float] = field(default_factory=dict)


class PerformanceTracker:
    """
    Comprehensive performance tracking for AXIOM agent.
    Monitors timing, memory usage, and computational efficiency.
    """

    def __init__(
        self,
        max_history_size: int = 10000,
        memory_sampling_interval: float = 0.1,
        enable_detailed_tracking: bool = True,
    ):
        """
        Initialize performance tracker.

        Args:
            max_history_size: Maximum number of measurements to keep
            memory_sampling_interval: Interval for memory sampling (seconds)
            enable_detailed_tracking: Enable detailed per-operation tracking
        """

        self.max_history_size = max_history_size
        self.memory_sampling_interval = memory_sampling_interval
        self.enable_detailed_tracking = enable_detailed_tracking

        # Timing data
        self.operation_times = defaultdict(list)
        self.operation_counts = defaultdict(int)
        self.start_times = {}

        # Memory tracking
        self.memory_history = deque(maxlen=max_history_size)
        self.peak_memory = 0.0
        self.memory_monitor_thread = None
        self.monitoring = False

        # Performance metrics
        self.metrics_history = deque(maxlen=max_history_size)
        self.global_start_time = time.time()

        # Process handle for memory monitoring
        self.process = psutil.Process()

        # Start memory monitoring
        if enable_detailed_tracking:
            self.start_memory_monitoring()

    def start_memory_monitoring(self):
        """Start background memory monitoring thread."""

        if self.memory_monitor_thread is not None:
            return

        self.monitoring = True
        self.memory_monitor_thread = threading.Thread(
            target=self._memory_monitor_loop, daemon=True
        )
        self.memory_monitor_thread.start()

    def stop_memory_monitoring(self):
        """Stop background memory monitoring."""

        self.monitoring = False
        if self.memory_monitor_thread is not None:
            self.memory_monitor_thread.join(timeout=1.0)
            self.memory_monitor_thread = None

    def _memory_monitor_loop(self):
        """Background loop for memory monitoring."""

        while self.monitoring:
            try:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                self.memory_history.append(memory_mb)
                self.peak_memory = max(self.peak_memory, memory_mb)
                time.sleep(self.memory_sampling_interval)
            except Exception:
                break

    @contextmanager
    def track_operation(self, operation_name: str):
        """
        Context manager for tracking operation timing.

        Args:
            operation_name: Name of the operation to track

        Usage:
            with tracker.track_operation("my_operation"):
                # Do work
                pass
        """

        start_time = time.time()
        self.start_times[operation_name] = start_time

        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time

            if self.enable_detailed_tracking:
                self.operation_times[operation_name].append(duration)
                if len(self.operation_times[operation_name]) > self.max_history_size:
                    self.operation_times[operation_name].pop(0)

            self.operation_counts[operation_name] += 1

    def record_metric(self, metric_name: str, value: Union[float, int]):
        """Record a custom metric value."""

        timestamp = time.time() - self.global_start_time

        if not hasattr(self, "custom_metrics"):
            self.custom_metrics = defaultdict(list)

        self.custom_metrics[metric_name].append((timestamp, value))

        # Limit history size
        if len(self.custom_metrics[metric_name]) > self.max_history_size:
            self.custom_metrics[metric_name].pop(0)

    def get_operation_stats(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for a specific operation."""

        if operation_name not in self.operation_times:
            return {
                "count": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "min_time": 0.0,
                "max_time": 0.0,
                "std_time": 0.0,
            }

        times = self.operation_times[operation_name]

        return {
            "count": len(times),
            "total_time": sum(times),
            "avg_time": np.mean(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "std_time": np.std(times),
        }

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""

        if not self.memory_history:
            current_memory = self.process.memory_info().rss / 1024 / 1024
            return {
                "current_mb": current_memory,
                "peak_mb": max(self.peak_memory, current_memory),
                "avg_mb": current_memory,
                "min_mb": current_memory,
                "max_mb": current_memory,
            }

        memory_array = np.array(list(self.memory_history))

        return {
            "current_mb": memory_array[-1] if len(memory_array) > 0 else 0.0,
            "peak_mb": self.peak_memory,
            "avg_mb": np.mean(memory_array),
            "min_mb": np.min(memory_array),
            "max_mb": np.max(memory_array),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""

        total_elapsed = time.time() - self.global_start_time
        total_operations = sum(self.operation_counts.values())

        summary = {
            "total_elapsed_time": total_elapsed,
            "total_operations": total_operations,
            "operations_per_second": total_operations / total_elapsed
            if total_elapsed > 0
            else 0.0,
            "memory_stats": self.get_memory_stats(),
            "operation_stats": {},
        }

        # Add per-operation statistics
        for op_name in self.operation_times.keys():
            summary["operation_stats"][op_name] = self.get_operation_stats(op_name)

        # Add custom metrics
        if hasattr(self, "custom_metrics"):
            summary["custom_metrics"] = {}
            for metric_name, values in self.custom_metrics.items():
                if values:
                    recent_values = [v[1] for v in values[-100:]]  # Last 100 values
                    summary["custom_metrics"][metric_name] = {
                        "current": recent_values[-1] if recent_values else 0.0,
                        "avg": np.mean(recent_values),
                        "min": np.min(recent_values),
                        "max": np.max(recent_values),
                    }

        return summary

    def reset(self):
        """Reset all performance tracking data."""

        self.operation_times.clear()
        self.operation_counts.clear()
        self.start_times.clear()
        self.memory_history.clear()
        self.metrics_history.clear()
        self.peak_memory = 0.0
        self.global_start_time = time.time()

        if hasattr(self, "custom_metrics"):
            self.custom_metrics.clear()

    def save_report(self, filepath: Path):
        """Save performance report to file."""

        report = {
            "timestamp": time.time(),
            "summary": self.get_summary(),
            "detailed_timings": dict(self.operation_times),
            "operation_counts": dict(self.operation_counts),
            "memory_history": list(self.memory_history),
        }

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj

        def clean_for_json(data):
            if isinstance(data, dict):
                return {k: clean_for_json(v) for k, v in data.items()}
            elif isinstance(data, (list, tuple)):
                return [clean_for_json(item) for item in data]
            else:
                return convert_numpy(data)

        clean_report = clean_for_json(report)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(clean_report, f, indent=2)

    def load_report(self, filepath: Path):
        """Load performance report from file."""

        with open(filepath, "r") as f:
            report = json.load(f)

        # Restore data structures
        if "detailed_timings" in report:
            self.operation_times = defaultdict(list, report["detailed_timings"])

        if "operation_counts" in report:
            self.operation_counts = defaultdict(int, report["operation_counts"])

        if "memory_history" in report:
            self.memory_history = deque(
                report["memory_history"], maxlen=self.max_history_size
            )
            if self.memory_history:
                self.peak_memory = max(self.memory_history)

    def __del__(self):
        """Cleanup when tracker is destroyed."""
        self.stop_memory_monitoring()


class EfficiencyAnalyzer:
    """Analyze efficiency and identify performance bottlenecks."""

    def __init__(self, tracker: PerformanceTracker):
        self.tracker = tracker

    def analyze_bottlenecks(self, top_n: int = 5) -> Dict[str, Any]:
        """Identify top performance bottlenecks."""

        operation_stats = {}
        total_time = 0.0

        for op_name in self.tracker.operation_times.keys():
            stats = self.tracker.get_operation_stats(op_name)
            operation_stats[op_name] = stats
            total_time += stats["total_time"]

        # Sort by total time
        sorted_ops = sorted(
            operation_stats.items(), key=lambda x: x[1]["total_time"], reverse=True
        )

        bottlenecks = []
        for op_name, stats in sorted_ops[:top_n]:
            percentage = (
                (stats["total_time"] / total_time * 100) if total_time > 0 else 0
            )
            bottlenecks.append(
                {
                    "operation": op_name,
                    "total_time": stats["total_time"],
                    "percentage": percentage,
                    "avg_time": stats["avg_time"],
                    "count": stats["count"],
                }
            )

        return {
            "bottlenecks": bottlenecks,
            "total_tracked_time": total_time,
            "analysis_timestamp": time.time(),
        }

    def compute_efficiency_metrics(self) -> Dict[str, float]:
        """Compute various efficiency metrics."""

        summary = self.tracker.get_summary()
        memory_stats = summary["memory_stats"]

        metrics = {
            "ops_per_second": summary["operations_per_second"],
            "memory_efficiency": 1.0 / memory_stats["peak_mb"]
            if memory_stats["peak_mb"] > 0
            else 0.0,
            "time_efficiency": summary["total_operations"]
            / summary["total_elapsed_time"]
            if summary["total_elapsed_time"] > 0
            else 0.0,
        }

        # Compute operation-specific efficiency
        for op_name, stats in summary["operation_stats"].items():
            if stats["count"] > 0:
                metrics[f"{op_name}_efficiency"] = 1.0 / stats["avg_time"]

        return metrics

    def suggest_optimizations(self) -> List[str]:
        """Suggest potential optimizations based on performance data."""

        suggestions = []

        # Analyze bottlenecks
        bottlenecks = self.analyze_bottlenecks()

        for bottleneck in bottlenecks["bottlenecks"][:3]:  # Top 3
            if bottleneck["percentage"] > 20:
                suggestions.append(
                    f"Optimize '{bottleneck['operation']}' operation - "
                    f"consumes {bottleneck['percentage']:.1f}% of total time"
                )

        # Memory analysis
        memory_stats = self.tracker.get_memory_stats()
        if memory_stats["peak_mb"] > 1000:  # > 1GB
            suggestions.append(
                f"High memory usage detected: {memory_stats['peak_mb']:.1f} MB peak. "
                f"Consider memory optimization."
            )

        # Operation frequency analysis
        summary = self.tracker.get_summary()
        for op_name, stats in summary["operation_stats"].items():
            if stats["avg_time"] > 0.1:  # > 100ms average
                suggestions.append(
                    f"Operation '{op_name}' has high average time ({stats['avg_time']:.3f}s). "
                    f"Consider optimization or caching."
                )

        return suggestions


class BenchmarkSuite:
    """Benchmark suite for AXIOM components."""

    def __init__(self):
        self.benchmarks = {}
        self.results = {}

    def register_benchmark(self, name: str, benchmark_func, description: str = ""):
        """Register a benchmark function."""

        self.benchmarks[name] = {"func": benchmark_func, "description": description}

    def run_benchmark(self, name: str, *args, **kwargs) -> Dict[str, Any]:
        """Run a specific benchmark."""

        if name not in self.benchmarks:
            raise ValueError(f"Benchmark '{name}' not found")

        tracker = PerformanceTracker(enable_detailed_tracking=True)

        start_time = time.time()

        with tracker.track_operation(name):
            result = self.benchmarks[name]["func"](*args, **kwargs)

        end_time = time.time()

        benchmark_result = {
            "name": name,
            "description": self.benchmarks[name]["description"],
            "execution_time": end_time - start_time,
            "result": result,
            "memory_stats": tracker.get_memory_stats(),
            "timestamp": time.time(),
        }

        self.results[name] = benchmark_result
        return benchmark_result

    def run_all_benchmarks(self, *args, **kwargs) -> Dict[str, Any]:
        """Run all registered benchmarks."""

        results = {}

        for name in self.benchmarks.keys():
            try:
                results[name] = self.run_benchmark(name, *args, **kwargs)
            except Exception as e:
                results[name] = {
                    "name": name,
                    "error": str(e),
                    "timestamp": time.time(),
                }

        return results

    def compare_results(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results with baseline."""

        comparison = {}

        for name, current in self.results.items():
            if name in baseline_results:
                baseline = baseline_results[name]

                if "execution_time" in current and "execution_time" in baseline:
                    speedup = baseline["execution_time"] / current["execution_time"]
                    comparison[name] = {
                        "speedup": speedup,
                        "current_time": current["execution_time"],
                        "baseline_time": baseline["execution_time"],
                        "improvement": (speedup - 1.0) * 100,  # Percentage improvement
                    }

        return comparison


# Pre-defined benchmarks for AXIOM components
def benchmark_matrix_operations(size: int = 1000, iterations: int = 10):
    """Benchmark basic matrix operations."""

    results = {}

    # Matrix multiplication
    A = np.random.randn(size, size)
    B = np.random.randn(size, size)

    start_time = time.time()
    for _ in range(iterations):
        A @ B
    results["matrix_multiply_time"] = (time.time() - start_time) / iterations

    # Eigenvalue decomposition
    start_time = time.time()
    for _ in range(min(iterations, 5)):  # Fewer iterations for expensive operations
        eigenvals, eigenvecs = np.linalg.eigh(A)
    results["eigendecomp_time"] = (time.time() - start_time) / min(iterations, 5)

    # Matrix inversion
    start_time = time.time()
    for _ in range(iterations):
        np.linalg.inv(A + 1e-6 * np.eye(size))
    results["matrix_inverse_time"] = (time.time() - start_time) / iterations

    return results


def benchmark_bayesian_inference(
    n_data: int = 1000, n_components: int = 10, iterations: int = 5
):
    """Benchmark Bayesian inference operations."""

    from .math_utils import VariationalInference

    results = {}

    # Generate test data
    data = np.random.randn(n_data, 5)

    # Benchmark variational inference E-step
    log_likelihoods = np.random.randn(n_data, n_components)
    mixing_weights = np.ones(n_components) / n_components

    start_time = time.time()
    for _ in range(iterations):
        VariationalInference.update_assignment_probabilities(
            log_likelihoods, mixing_weights
        )
    results["vi_estep_time"] = (time.time() - start_time) / iterations

    # Benchmark NIW parameter updates
    responsibilities_single = np.random.rand(n_data)
    responsibilities_single /= np.sum(responsibilities_single)

    prior_params = {"m": np.zeros(5), "kappa": 1.0, "nu": 7.0, "psi": np.eye(5)}

    start_time = time.time()
    for _ in range(iterations):
        m_new, kappa_new, nu_new, psi_new = VariationalInference.update_niw_parameters(
            data, responsibilities_single, **prior_params
        )
    results["niw_update_time"] = (time.time() - start_time) / iterations

    return results


# Example usage and testing
def test_performance_utilities():
    """Test performance utilities."""

    print("Testing AXIOM Performance Utilities...")

    # Test basic performance tracking
    tracker = PerformanceTracker()

    # Simulate some operations
    with tracker.track_operation("test_operation_1"):
        time.sleep(0.01)
        np.random.randn(1000, 1000) @ np.random.randn(1000, 1000)

    with tracker.track_operation("test_operation_2"):
        time.sleep(0.005)
        np.linalg.eigh(np.random.randn(500, 500) + np.eye(500))

    # Repeat operations
    for _ in range(5):
        with tracker.track_operation("test_operation_1"):
            time.sleep(0.002)

    # Get summary
    summary = tracker.get_summary()
    print(f"Total operations: {summary['total_operations']}")
    print(f"Operations per second: {summary['operations_per_second']:.2f}")
    print(f"Peak memory: {summary['memory_stats']['peak_mb']:.2f} MB")

    # Test efficiency analyzer
    analyzer = EfficiencyAnalyzer(tracker)
    bottlenecks = analyzer.analyze_bottlenecks()
    print(f"Top bottleneck: {bottlenecks['bottlenecks'][0]['operation']}")

    suggestions = analyzer.suggest_optimizations()
    if suggestions:
        print(f"Optimization suggestion: {suggestions[0]}")

    # Test benchmark suite
    suite = BenchmarkSuite()
    suite.register_benchmark(
        "matrix_ops", benchmark_matrix_operations, "Basic matrix operations benchmark"
    )

    result = suite.run_benchmark("matrix_ops", size=100, iterations=3)
    print(f"Matrix multiply time: {result['result']['matrix_multiply_time']:.4f}s")

    tracker.stop_memory_monitoring()
    print("All performance tests completed successfully!")


if __name__ == "__main__":
    test_performance_utilities()

# Module-level exports
get_memory_stats = PerformanceTracker.get_memory_stats
track_operation = PerformanceTracker.track_operation
record_metric = PerformanceTracker.record_metric
