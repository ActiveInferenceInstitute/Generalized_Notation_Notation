#!/usr/bin/env python3
"""
Complexity, maintainability, and technical-debt metrics for GNN Step 16 analysis.

Extracted from ``analysis.analyzer``.
"""

from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    cast,
)

import numpy as np

from .analysis_extraction import (
    extract_connections,
    extract_variables,
)


def calculate_cyclomatic_complexity(
    variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]
) -> float:
    """Calculate cyclomatic complexity of the model."""
    return len(connections) - len(variables) + 2


def calculate_cognitive_complexity(
    variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]
) -> float:
    """Calculate cognitive complexity of the model."""
    # Cognitive complexity considers nesting, branching, and logical operators
    complexity = 0.0

    # Base complexity from number of elements
    complexity += len(variables) * 0.5
    complexity += len(connections) * 1.0

    # Additional complexity for high connectivity
    if variables and connections:
        density = len(connections) / len(variables)
        if density > 2.0:
            complexity += density * 0.5

    return complexity


def calculate_structural_complexity(
    variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]
) -> float:
    """Calculate structural complexity of the model."""
    # Structural complexity considers the graph structure
    complexity = 0.0

    # Base complexity
    complexity += len(variables) * 0.3
    complexity += len(connections) * 0.7

    # Graph density penalty
    if variables and connections:
        density = len(connections) / len(variables)
        if density > 1.5:
            complexity += density * 0.2

    return complexity


def calculate_complexity_metrics(
    file_path: Path, verbose: bool = False
) -> Dict[str, Any]:
    """Calculate comprehensive complexity metrics for a GNN file."""
    try:
        with open(file_path) as f:
            content = f.read()

        variables = extract_variables(content)
        connections = extract_connections(content)

        metrics: dict[str, Any] = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "cyclomatic_complexity": calculate_cyclomatic_complexity(
                variables, connections
            ),
            "cognitive_complexity": calculate_cognitive_complexity(
                variables, connections
            ),
            "structural_complexity": calculate_structural_complexity(
                variables, connections
            ),
            "maintainability_index": calculate_maintainability_index(
                content, variables, connections
            ),
            "technical_debt": calculate_technical_debt(content, variables, connections),
            "analysis_timestamp": datetime.now().isoformat(),
        }

        return metrics

    except Exception as e:
        raise RuntimeError(
            f"Failed to calculate complexity metrics for {file_path}: {e}"
        ) from e


def calculate_maintainability_index(
    content: str, variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]
) -> float:
    """Calculate maintainability index."""
    # Simplified maintainability index calculation
    lines = len(content.splitlines())
    complexity = len(variables) + len(connections)

    if lines == 0:
        return 100.0

    # Higher index = more maintainable
    maintainability = 171 - 5.2 * np.log(complexity) - 0.23 * np.log(lines)
    return cast("float", max(0.0, min(100.0, maintainability)))


def calculate_technical_debt(
    content: str, variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]
) -> float:
    """Calculate technical debt score."""
    debt = 0.0

    # Complexity debt
    if len(variables) > 20:
        debt += (len(variables) - 20) * 0.1

    if len(connections) > 50:
        debt += (len(connections) - 50) * 0.05

    # Documentation debt (simplified)
    if len(content) < 1000:  # Assuming short content means poor documentation
        debt += 0.5

    return debt
