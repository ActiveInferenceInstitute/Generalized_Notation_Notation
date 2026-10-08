#!/usr/bin/env python3
"""
Analysis analyzer module for GNN statistical analysis.

Mechanical split facade: implementations live in ``analysis_*``,
``simulation_visualizations``, and ``framework_comparison`` sibling
modules; every previously public and private name is re-exported here
so consumer import paths are unchanged.
"""

import hashlib
import io
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
)

import numpy as np

from .analysis_complexity import (
    calculate_cognitive_complexity,
    calculate_complexity_metrics,
    calculate_cyclomatic_complexity,
    calculate_maintainability_index,
    calculate_structural_complexity,
    calculate_technical_debt,
)
from .analysis_extraction import (
    extract_connections,
    extract_sections,
    extract_variables,
)
from .analysis_statistics import (
    SCIPY_AVAILABLE,
    analyze_distributions,
    calculate_connection_statistics,
    calculate_correlations,
    calculate_section_statistics,
    calculate_variable_statistics,
)
from .framework_comparison import (
    _calculate_aggregate_metrics,
    _compare_framework_results,
    _discopy_inline_circuit_from_structured,
    _extract_simulation_metrics,
    analyze_framework_outputs,
    generate_framework_comparison_report,
)
from .simulation_visualizations import (
    SEABORN_AVAILABLE,
    generate_matrix_visualizations,
    parse_matrix_data,
    visualize_cross_framework_metrics,
    visualize_simulation_results,
)

logger = logging.getLogger(__name__)

# Explicit re-export surface: every name consumers may import through this
# facade (analysis/__init__.py, processor.py). Third-party aliases (``stats``,
# ``sns``) and internal helpers are intentionally excluded.
__all__ = [
    "analyze_distributions",
    "analyze_framework_outputs",
    "calculate_cognitive_complexity",
    "calculate_complexity_metrics",
    "calculate_connection_statistics",
    "calculate_correlations",
    "calculate_cyclomatic_complexity",
    "calculate_maintainability_index",
    "calculate_section_statistics",
    "calculate_structural_complexity",
    "calculate_technical_debt",
    "calculate_variable_statistics",
    "extract_connections",
    "extract_sections",
    "extract_variables",
    "generate_analysis_summary",
    "generate_framework_comparison_report",
    "generate_matrix_visualizations",
    "perform_model_comparisons",
    "perform_statistical_analysis",
    "run_performance_benchmarks",
    "visualize_cross_framework_metrics",
    "visualize_simulation_results",
]


def perform_statistical_analysis(
    file_path: Path, verbose: bool = False
) -> Dict[str, Any]:
    """Perform comprehensive statistical analysis on a GNN file."""
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Extract structural elements
        variables = extract_variables(content)
        connections = extract_connections(content)
        sections = extract_sections(content)

        # Calculate statistics
        var_stats = calculate_variable_statistics(variables)
        conn_stats = calculate_connection_statistics(connections)
        section_stats = calculate_section_statistics(sections)

        # Analyze distributions
        distributions = analyze_distributions(variables, connections)

        # Calculate correlations
        correlations = calculate_correlations(variables, connections)

        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "line_count": len(content.splitlines()),
            "variables": variables,
            "connections": connections,
            "sections": sections,
            "variable_statistics": var_stats,
            "connection_statistics": conn_stats,
            "section_statistics": section_stats,
            "distributions": distributions,
            "correlations": correlations,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise RuntimeError(f"Failed to analyze {file_path}: {e}") from e


def run_performance_benchmarks(
    file_path: Path, verbose: bool = False
) -> Dict[str, Any]:
    """Measure native parser wall/CPU time and qualify retained size/runtime estimates."""
    try:
        source_bytes = file_path.read_bytes()
        # Match open(..., "r") locale encoding and universal-newline behavior,
        # while hashing exactly the bytes consumed by this measurement.
        with io.TextIOWrapper(io.BytesIO(source_bytes)) as stream:
            content = stream.read()

        start_time = time.perf_counter()
        start_cpu = time.process_time()
        variables = extract_variables(content)
        connections = extract_connections(content)
        end_time = time.perf_counter()

        real_parse_time = end_time - start_time
        parse_cpu_time = time.process_time() - start_cpu

        # Retain the existing shallow Python object-size estimate; this is not RSS.
        memory_usage = (
            sys.getsizeof(content)
            + sys.getsizeof(variables)
            + sys.getsizeof(connections)
        )
        for var in variables:
            memory_usage += sys.getsizeof(var)
        for conn in connections:
            memory_usage += sys.getsizeof(conn)

        complexity = len(variables) + len(connections)

        # Measured parse time and explicitly qualified compatibility estimates
        benchmarks: dict[str, Any] = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "parse_time": real_parse_time,
            "parse_cpu_time": parse_cpu_time,
            "memory_usage": memory_usage,
            "memory_measurement": "shallow Python object-size estimate; not measured RSS or JIT allocation",
            "complexity_score": complexity,
            "estimated_runtime": complexity * 0.01,
            "runtime_estimate_method": "existing heuristic: complexity_score * 0.01 seconds; not a native execution measurement",
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "measured_phase": "extract_variables and extract_connections after source read",
            "benchmark_timestamp": datetime.now().isoformat(),
        }

        return benchmarks

    except Exception as e:
        raise RuntimeError(f"Failed to run benchmarks for {file_path}: {e}") from e


def perform_model_comparisons(
    statistical_analyses: List[Dict[str, Any]], verbose: bool = False
) -> Dict[str, Any]:
    """Perform comparisons between multiple models."""
    if len(statistical_analyses) < 2:
        return {"error": "Need at least 2 models for comparison"}

    comparisons: dict[str, Any] = {
        "model_count": len(statistical_analyses),
        "complexity_comparison": {},
        "size_comparison": {},
        "structure_comparison": {},
        "comparison_timestamp": datetime.now().isoformat(),
    }

    # Compare complexity metrics
    complexity_scores: list[Any] = []
    file_sizes: list[Any] = []
    variable_counts: list[Any] = []
    connection_counts: list[Any] = []

    for analysis in statistical_analyses:
        complexity_metrics = analysis.get("distributions", {}).get(
            "complexity_metrics", {}
        )
        complexity_scores.append(complexity_metrics.get("total_elements", 0))
        file_sizes.append(analysis.get("file_size", 0))
        variable_counts.append(len(analysis.get("variables", [])))
        connection_counts.append(len(analysis.get("connections", [])))

    comparisons["complexity_comparison"] = {
        "mean": np.mean(complexity_scores),
        "std": np.std(complexity_scores),
        "min": np.min(complexity_scores),
        "max": np.max(complexity_scores),
    }

    comparisons["size_comparison"] = {
        "mean": np.mean(file_sizes),
        "std": np.std(file_sizes),
        "min": np.min(file_sizes),
        "max": np.max(file_sizes),
    }

    comparisons["structure_comparison"] = {
        "variable_counts": {
            "mean": np.mean(variable_counts),
            "std": np.std(variable_counts),
        },
        "connection_counts": {
            "mean": np.mean(connection_counts),
            "std": np.std(connection_counts),
        },
    }

    return comparisons


def generate_analysis_summary(results: Dict[str, Any]) -> str:
    """Generate a summary report of analysis results."""
    summary = f"""
# Analysis Summary

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Processing Results
- **Files Processed**: {results.get("processed_files", 0)}
- **Success**: {results.get("success", False)}
- **Errors**: {len(results.get("errors", []))}

## Analysis Results
- **Statistical Analyses**: {len(results.get("statistical_analysis", []))}
- **Complexity Metrics**: {len(results.get("complexity_metrics", []))}
- **Performance Benchmarks**: {len(results.get("performance_benchmarks", []))}
- **Model Comparisons**: {len(results.get("model_comparisons", []))}

## Error Summary
"""

    errors = results.get("errors", [])
    if errors:
        for error in errors:
            if isinstance(error, dict):
                summary += f"- **{error.get('file', 'Unknown')}**: {error.get('error', 'Unknown error')}\n"
            else:
                summary += f"- {error}\n"
    else:
        summary += "- No errors encountered\n"

    summary += "\n## Model Statistics\n"

    # Add statistics from analyses
    analyses = results.get("statistical_analysis", [])
    if analyses:
        total_variables = sum(
            len(analysis.get("variables", [])) for analysis in analyses
        )
        total_connections = sum(
            len(analysis.get("connections", [])) for analysis in analyses
        )

        summary += f"- Total variables across all models: {total_variables}\n"
        summary += f"- Total connections across all models: {total_connections}\n"
        summary += (
            f"- Average variables per model: {total_variables / len(analyses):.1f}\n"
        )
        summary += f"- Average connections per model: {total_connections / len(analyses):.1f}\n"

    return summary
