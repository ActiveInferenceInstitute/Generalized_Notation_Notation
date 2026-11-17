"""
Performance Profiler

This module provides performance profiling for GNN models, including
computational complexity estimation, memory usage prediction, and
identification of potential performance bottlenecks.
"""

import logging
import re
from pathlib import Path
from typing import Any

from .structure import (
    clamp01,
    display_file_name,
    extract_content_from_dict,
)

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Profiler for performance aspects of GNN models."""

    def profile(self, content: str) -> dict[str, Any]:
        """
        Profile the performance characteristics of a GNN model.

        Args:
            content: GNN model content

        Returns:
            Performance profile with metrics and warnings
        """
        # Extract model structure
        state_blocks = re.findall(r"StateSpaceBlock\s*\{([^}]*)\}", content)
        connections = re.findall(r"Connection\s*\{([^}]*)\}", content)

        # Extract dimensions and other properties
        block_dims = self._extract_block_dimensions(state_blocks)
        block_types = self._extract_block_types(state_blocks)
        connection_types = self._extract_connection_types(connections)

        # Calculate metrics
        metrics = self._calculate_metrics(block_dims, block_types, connection_types)

        # Generate warnings
        warnings = self._generate_warnings(metrics)

        return {"metrics": metrics, "warnings": warnings}

    def _extract_block_dimensions(
        self, state_blocks: list[str]
    ) -> dict[str, list[int]]:
        """Extract dimensions from state blocks."""
        block_dims: dict[Any, Any] = {}

        for block in state_blocks:
            name_match = re.search(r"Name:\s*([^\n]+)", block)
            dim_match = re.search(r"Dimensions:\s*([^\n]+)", block)

            if name_match and dim_match:
                name = name_match.group(1).strip()
                try:
                    dims = [
                        int(d.strip()) for d in dim_match.group(1).strip().split(",")
                    ]
                    block_dims[name] = dims
                except ValueError:
                    logger.debug("Skipping invalid dimensions for '%s'", name)

        return block_dims

    def _extract_block_types(self, state_blocks: list[str]) -> dict[str, str]:
        """Extract block types from state blocks."""
        block_types: dict[Any, Any] = {}

        for block in state_blocks:
            name_match = re.search(r"Name:\s*([^\n]+)", block)
            type_match = re.search(r"Type:\s*([^\n]+)", block)

            if name_match:
                name = name_match.group(1).strip()
                block_type = type_match.group(1).strip() if type_match else "Generic"
                block_types[name] = block_type

        return block_types

    def _extract_connection_types(self, connections: list[str]) -> list[str]:
        """Extract connection types from connections."""
        connection_types: list[Any] = []

        for conn in connections:
            type_match = re.search(r"Type:\s*([^\n]+)", conn)
            if type_match:
                connection_types.append(type_match.group(1).strip())
            else:
                connection_types.append("Generic")

        return connection_types

    def _calculate_metrics(
        self,
        block_dims: dict[str, list[int]],
        block_types: dict[str, str],
        connection_types: list[str],
    ) -> dict[str, Any]:
        """Calculate performance metrics."""
        metrics: dict[Any, Any] = {}

        # Basic counts
        metrics["state_block_count"] = len(block_dims)
        metrics["connection_count"] = len(connection_types)

        # Dimension statistics
        all_dims: list[Any] = []
        for dims in block_dims.values():
            all_dims.extend(dims)

        metrics["total_dimensions"] = sum(all_dims) if all_dims else 0
        metrics["max_dimension"] = max(all_dims) if all_dims else 0
        metrics["avg_dimension"] = sum(all_dims) / len(all_dims) if all_dims else 0

        # Memory usage estimation
        # Assume 8 bytes per dimension for double precision
        metrics["estimated_memory_bytes"] = metrics["total_dimensions"] * 8
        metrics["estimated_memory_mb"] = metrics["estimated_memory_bytes"] / (
            1024 * 1024
        )

        # Computational complexity estimation
        metrics["computational_complexity"] = self._estimate_computational_complexity(
            block_dims, block_types, connection_types
        )

        # Parallelization potential
        metrics["parallelization_potential"] = self._estimate_parallelization_potential(
            block_dims, block_types, connection_types
        )

        return metrics

    def _estimate_computational_complexity(
        self,
        block_dims: dict[str, list[int]],
        block_types: dict[str, str],
        connection_types: list[str],
    ) -> dict[str, Any]:
        """Estimate computational complexity of the model."""
        # Initialize complexity metrics
        complexity: dict[str, Any] = {
            "inference_operations": 0,
            "learning_operations": 0,
            "complexity_class": "Unknown",
            "bottleneck_block": None,
            "bottleneck_operations": 0,
        }

        # Calculate operations for each block
        block_operations: dict[Any, Any] = {}
        for name, dims in block_dims.items():
            block_type = block_types.get(name, "Generic")

            # Calculate operations based on block type and dimensions
            if block_type.lower() in ["categorical", "discrete"]:
                # Categorical distributions: operations proportional to number of categories
                ops = sum(dims)
            elif block_type.lower() in ["gaussian", "normal", "continuous"]:
                # Gaussian distributions: operations proportional to dimensions squared
                ops = sum(d * d for d in dims)
            elif block_type.lower() in ["dirichlet", "beta"]:
                # Dirichlet/Beta: operations proportional to dimensions
                ops = sum(dims) * 2
            else:
                # Generic: assume quadratic complexity
                ops = sum(d * d for d in dims)

            block_operations[name] = ops

        # Find bottleneck block
        if block_operations:
            bottleneck_block = max(block_operations.items(), key=lambda x: x[1])
            complexity["bottleneck_block"] = bottleneck_block[0]
            complexity["bottleneck_operations"] = bottleneck_block[1]

        # Calculate total operations
        inference_ops = sum(block_operations.values())
        complexity["inference_operations"] = inference_ops

        # Learning operations (typically more expensive than inference)
        complexity["learning_operations"] = inference_ops * 2

        # Determine complexity class
        total_dims = sum(sum(dims) for dims in block_dims.values())
        if total_dims == 0:
            complexity["complexity_class"] = "O(1)"
        elif inference_ops < total_dims * 2:
            complexity["complexity_class"] = "O(n)"
        elif inference_ops < total_dims * total_dims * 1.5:
            complexity["complexity_class"] = "O(n²)"
        else:
            complexity["complexity_class"] = "O(n³) or higher"

        return complexity

    def _estimate_parallelization_potential(
        self,
        block_dims: dict[str, list[int]],
        block_types: dict[str, str],
        connection_types: list[str],
    ) -> dict[str, Any]:
        """Estimate parallelization potential of the model."""
        # Initialize parallelization metrics
        parallelization: dict[str, Any] = {
            "parallelizable_blocks": 0,
            "sequential_dependencies": 0,
            "parallel_efficiency": 0.0,
            "gpu_amenable": False,
            "distributed_amenable": False,
        }

        # Dependency graph parsing is intentionally coarse here; connection count feeds the
        # sequential-dependency heuristic below.
        sequential_connection_count = len(connection_types)

        # Estimate parallel efficiency
        total_blocks = len(block_dims)
        if total_blocks > 0:
            # Simple heuristic: assume 70% of blocks can be parallelized
            parallelizable_blocks = int(total_blocks * 0.7)
            parallelization["parallelizable_blocks"] = parallelizable_blocks
            sequential_dependencies = (
                total_blocks
                - parallelizable_blocks
                + int(sequential_connection_count > 0)
            )
            parallelization["sequential_dependencies"] = min(
                total_blocks, sequential_dependencies
            )
            parallelization["parallel_efficiency"] = (
                parallelizable_blocks / total_blocks
            )

        # Determine GPU amenability
        # Simple heuristic: if there are large dimensions, it's GPU amenable
        large_dims = any(max(dims) > 100 for dims in block_dims.values() if dims)
        many_blocks = len(block_dims) > 10
        parallelization["gpu_amenable"] = large_dims or many_blocks

        # Determine distributed amenability
        # Simple heuristic: if there are many blocks with moderate coupling, it's distributed amenable
        many_connections = len(connection_types) > 20
        parallelization["distributed_amenable"] = many_blocks and not many_connections

        return parallelization

    def _generate_warnings(self, metrics: dict[str, Any]) -> list[str]:
        """Generate warnings based on performance metrics."""
        warnings: list[Any] = []

        # Memory usage warnings
        memory_mb = metrics.get("estimated_memory_mb", 0)
        if memory_mb > 1000:
            warnings.append(f"Very high memory usage: {memory_mb:.2f} MB")
        elif memory_mb > 100:
            warnings.append(f"High memory usage: {memory_mb:.2f} MB")

        # Computational complexity warnings
        complexity = metrics.get("computational_complexity", {})
        complexity_class = complexity.get("complexity_class", "Unknown")
        if complexity_class in ["O(n³) or higher"]:
            warnings.append(f"High computational complexity: {complexity_class}")

        bottleneck_block = complexity.get("bottleneck_block")
        bottleneck_ops = complexity.get("bottleneck_operations", 0)
        if bottleneck_block and bottleneck_ops > 10000:
            warnings.append(
                f"Performance bottleneck in block '{bottleneck_block}': {bottleneck_ops} operations"
            )

        # Parallelization warnings
        parallelization = metrics.get("parallelization_potential", {})
        parallel_efficiency = parallelization.get("parallel_efficiency", 0)
        if parallel_efficiency < 0.5:
            warnings.append(f"Low parallelization potential: {parallel_efficiency:.2f}")

        # Structure warnings
        state_block_count = metrics.get("state_block_count", 0)
        connection_count = metrics.get("connection_count", 0)

        if state_block_count > 20:
            warnings.append(f"Large number of state blocks ({state_block_count})")

        if connection_count > 50:
            warnings.append(f"Large number of connections ({connection_count})")

        if connection_count > state_block_count * 3:
            warnings.append(
                f"High connection density: {connection_count} connections for {state_block_count} blocks"
            )

        return warnings


def profile_performance(
    model_path: str | Path | dict[str, Any],
) -> dict[str, Any]:
    """
    Profile the performance characteristics of a GNN model.

    Args:
        model_path: Path to the GNN model file, Path object, or model data dictionary

    Returns:
        Performance profile with metrics and warnings
    """
    try:
        # Handle different input types
        if isinstance(model_path, dict):
            # If it's already a dictionary, extract content from it
            content = extract_content_from_dict(model_path)
            model_path_str = model_path.get("file_path", "unknown")
        else:
            # Convert string path to Path object
            model_path = Path(model_path)
            model_path_str = str(model_path)

            # Read model content
            with open(model_path, encoding="utf-8") as f:
                content = f.read()

        # Performance profiling
        performance_profiler = PerformanceProfiler()
        profile_result = performance_profiler.profile(content)

        return {
            "file_path": model_path_str,
            "file_name": display_file_name(str(model_path_str)),
            "metrics": profile_result.get("metrics", {}),
            "warnings": profile_result.get("warnings", []),
            "performance_score": _calculate_performance_score(profile_result),
            "recovery": False,
        }

    except Exception as e:
        return {
            "status": "error",
            "file_path": str(model_path)
            if not isinstance(model_path, dict)
            else "unknown",
            "file_name": "unknown"
            if isinstance(model_path, dict)
            else Path(str(model_path)).name,
            "error": str(e),
            "metrics": {},
            "warnings": [str(e)],
            "performance_score": 0.0,
            "recovery": True,
        }


def _calculate_performance_score(profile_result: dict[str, Any]) -> float:
    """Calculate a performance score from profile results."""
    metrics = profile_result.get("metrics", {})
    warnings = profile_result.get("warnings", [])

    # Base score starts at 1.0
    score = 1.0

    # Deduct points for warnings
    score -= len(warnings) * 0.1

    # Deduct points for high memory usage
    memory_mb = metrics.get("estimated_memory_mb", 0)
    if memory_mb > 1000:
        score -= 0.3
    elif memory_mb > 100:
        score -= 0.1

    # Deduct points for high computational complexity
    complexity = metrics.get("computational_complexity", {})
    complexity_class = complexity.get("complexity_class", "Unknown")
    if complexity_class in ["O(n³) or higher"]:
        score -= 0.3
    elif complexity_class in ["O(n²)"]:
        score -= 0.1

    # Deduct points for low parallelization potential
    parallelization = metrics.get("parallelization_potential", {})
    parallel_efficiency = parallelization.get("parallel_efficiency", 1.0)
    if parallel_efficiency < 0.5:
        score -= 0.2

    # Ensure score is between 0 and 1
    return clamp01(score)
