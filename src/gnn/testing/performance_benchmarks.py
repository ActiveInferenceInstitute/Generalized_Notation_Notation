"""
GNN Performance Benchmarking System

This module provides comprehensive performance benchmarking and complexity analysis
for GNN models and Active Inference computations.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for GNN operations."""

    operation_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_ops_per_sec: float
    complexity_score: float
    accuracy_score: float
    error_count: int = 0


class GNNPerformanceBenchmark:
    """Performance benchmarking for GNN operations."""

    def __init__(self, test_iterations: int = 100) -> None:
        """Initialize the instance."""
        self.test_iterations = test_iterations
        self.results: List[PerformanceMetrics] = []

    def benchmark_parsing(self, file_path: Path) -> PerformanceMetrics:
        """Benchmark GNN file parsing performance."""
        from ..schema_validator import GNNParser

        parser = GNNParser()

        with open(file_path) as f:
            content = f.read()

        # Measure performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        errors = 0
        for _ in range(self.test_iterations):
            try:
                parser.parse_content(content)
            except Exception:
                errors += 1

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        execution_time = (end_time - start_time) / self.test_iterations
        memory_usage = end_memory - start_memory
        throughput = 1.0 / execution_time if execution_time > 0 else 0

        metrics = PerformanceMetrics(
            operation_name=f"parse_{file_path.stem}",
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=0,  # Simplified
            throughput_ops_per_sec=throughput,
            complexity_score=len(content) * 0.001,  # Simple complexity estimate
            accuracy_score=1.0 - (errors / self.test_iterations),
            error_count=errors,
        )

        self.results.append(metrics)
        return metrics

    def benchmark_validation(self, file_path: Path) -> PerformanceMetrics:
        """Benchmark GNN file validation performance."""
        from ..schema_validator import GNNValidator

        validator = GNNValidator()

        # Measure performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        errors = 0
        for _ in range(self.test_iterations):
            try:
                result = validator.validate_file(file_path)
                if not result.is_valid:
                    errors += len(result.errors)
            except Exception:
                errors += 1

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        execution_time = (end_time - start_time) / self.test_iterations
        memory_usage = end_memory - start_memory
        throughput = 1.0 / execution_time if execution_time > 0 else 0

        metrics = PerformanceMetrics(
            operation_name=f"validate_{file_path.stem}",
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=0,  # Simplified
            throughput_ops_per_sec=throughput,
            complexity_score=file_path.stat().st_size * 0.0001,
            accuracy_score=1.0 - (errors / (self.test_iterations * 10)),  # Normalized
            error_count=errors,
        )

        self.results.append(metrics)
        return metrics

    def generate_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        if not self.results:
            return {"error": "No benchmark results available"}

        total_time = sum(m.execution_time for m in self.results)
        avg_memory = sum(m.memory_usage_mb for m in self.results) / len(self.results)
        avg_throughput = sum(m.throughput_ops_per_sec for m in self.results) / len(
            self.results
        )
        total_errors = sum(m.error_count for m in self.results)

        return {
            "summary": {
                "total_operations": len(self.results),
                "total_execution_time": total_time,
                "average_memory_usage_mb": avg_memory,
                "average_throughput": avg_throughput,
                "total_errors": total_errors,
            },
            "operations": [
                {
                    "name": m.operation_name,
                    "execution_time": m.execution_time,
                    "memory_mb": m.memory_usage_mb,
                    "throughput": m.throughput_ops_per_sec,
                    "accuracy": m.accuracy_score,
                    "errors": m.error_count,
                }
                for m in self.results
            ],
        }


def benchmark_gnn_files(file_paths: List[Path]) -> Dict[str, Any]:
    """Benchmark multiple GNN files."""
    benchmark = GNNPerformanceBenchmark()

    for file_path in file_paths:
        benchmark.benchmark_parsing(file_path)
        benchmark.benchmark_validation(file_path)

    return benchmark.generate_report()
