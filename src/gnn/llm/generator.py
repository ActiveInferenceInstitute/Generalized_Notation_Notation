#!/usr/bin/env python3
"""
LLM generator module for insights, suggestions, and documentation.
"""

from datetime import datetime
from typing import Any, Dict

from .analyzer import variable_type_counts


def generate_model_insights(file_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate insights about the GNN model."""
    insights: dict[str, Any] = {
        "model_complexity": "medium",
        "recommendations": [],
        "strengths": [],
        "weaknesses": [],
        "generation_timestamp": datetime.now().isoformat(),
    }

    # Analyze complexity
    complexity_metrics = file_analysis.get("complexity_metrics", {})
    total_elements = complexity_metrics.get("total_elements", 0)

    if total_elements < 10:
        insights["model_complexity"] = "low"
        insights["strengths"].append("Simple and maintainable model")
    elif total_elements > 50:
        insights["model_complexity"] = "high"
        insights["weaknesses"].append("Complex model may be difficult to understand")
        insights["recommendations"].append(
            "Consider breaking down into smaller modules"
        )

    # Analyze variables
    variables = file_analysis.get("variables", [])
    if len(variables) > 20:
        insights["recommendations"].append(
            "High variable count - consider grouping related variables"
        )

    # Analyze connections
    connections = file_analysis.get("connections", [])
    if len(connections) > len(variables) * 3:
        insights["recommendations"].append(
            "High connectivity - consider simplifying the graph structure"
        )

    # Check for patterns
    patterns = file_analysis.get("patterns", {})
    if patterns.get("anti_patterns"):
        insights["weaknesses"].extend(patterns["anti_patterns"])

    if patterns.get("suggestions"):
        insights["recommendations"].extend(patterns["suggestions"])

    return insights


def generate_code_suggestions(file_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate code suggestions for the GNN model."""
    suggestions: dict[str, Any] = {
        "optimizations": [],
        "improvements": [],
        "best_practices": [],
        "generation_timestamp": datetime.now().isoformat(),
    }

    # Analyze for optimization opportunities
    complexity_metrics = file_analysis.get("complexity_metrics", {})
    density = complexity_metrics.get("density", 0)

    if density > 2.0:
        suggestions["optimizations"].append(
            "High graph density - consider sparse representations"
        )

    # Check variable patterns
    variables = file_analysis.get("variables", [])
    connections = file_analysis.get("connections", [])
    var_types = variable_type_counts(variables)

    # Suggest type improvements
    if "unknown" in var_types and var_types["unknown"] > len(variables) * 0.5:
        suggestions["improvements"].append(
            "Many variables lack type annotations - consider adding explicit types"
        )

    # Suggest best practices
    if len(variables) > 0:
        suggestions["best_practices"].append("Use descriptive variable names")
        suggestions["best_practices"].append("Group related variables together")

    if len(connections) > 0:
        suggestions["best_practices"].append("Document connection semantics")
        suggestions["best_practices"].append(
            "Consider connection weights or probabilities"
        )

    return suggestions


def generate_documentation(file_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate documentation for the GNN model."""
    documentation: dict[str, Any] = {
        "file_path": file_analysis.get(
            "file_path", file_analysis.get("file_name", "Unknown")
        ),
        "model_overview": "",
        "variable_documentation": [],
        "connection_documentation": [],
        "usage_examples": [],
        "generation_timestamp": datetime.now().isoformat(),
    }

    # Generate model overview
    file_name = file_analysis.get("file_name", "Unknown")
    variables = file_analysis.get("variables", [])
    connections = file_analysis.get("connections", [])

    documentation["model_overview"] = f"""
# {file_name} Model Documentation

This GNN model contains {len(variables)} variables and {len(connections)} connections.

## Model Structure
- **Variables**: {len(variables)} defined
- **Connections**: {len(connections)} defined
- **Complexity**: {file_analysis.get("complexity_metrics", {}).get("total_elements", 0)} total elements

## Key Components
"""

    # Document variables
    for var in variables:
        doc: dict[str, Any] = {
            "name": var.get("name", "Unknown"),
            "definition": var.get("definition", ""),
            "line": var.get("line", 0),
            "description": f"Variable defined at line {var.get('line', 0)}",
        }
        documentation["variable_documentation"].append(doc)

    # Document connections
    for conn in connections:
        doc = {
            "source": conn.get("source", "Unknown"),
            "target": conn.get("target", "Unknown"),
            "connection": conn.get("connection", ""),
            "line": conn.get("line", 0),
            "description": f"Connection from {conn.get('source', 'Unknown')} to {conn.get('target', 'Unknown')}",
        }
        documentation["connection_documentation"].append(doc)

    # Generate usage examples
    if variables:
        example_vars = [var.get("name", "var") for var in variables[:3]]
        documentation["usage_examples"].append(
            {
                "title": "Variable Usage",
                "description": f"Example variables: {', '.join(example_vars)}",
                "code": f"# Example variable usage\n{chr(10).join([f'{var} = ...' for var in example_vars])}",
            }
        )

    if connections:
        example_conns = connections[:2]
        conn_examples: list[Any] = []
        for conn in example_conns:
            conn_examples.append(
                f"{conn.get('source', 'source')} -> {conn.get('target', 'target')}"
            )

        documentation["usage_examples"].append(
            {
                "title": "Connection Usage",
                "description": f"Example connections: {', '.join(conn_examples)}",
                "code": f"# Example connections\n{chr(10).join(conn_examples)}",
            }
        )

    return documentation


def generate_llm_summary(results: Dict[str, Any]) -> str:
    """Generate a summary report of LLM processing results."""
    summary = f"""
# LLM Processing Summary

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Processing Results
- **Files Processed**: {results.get("processed_files", 0)}
- **Success**: {results.get("success", False)}
- **Errors**: {len(results.get("errors", []))}
"""

    for label, key in (
        ("Status", "status"),
        ("Selected Files", "selected_files"),
        ("Discovered Files", "total_files_discovered"),
        ("Provider", "provider"),
        ("Model", "selected_model"),
        ("Budget (seconds)", "budget_seconds"),
    ):
        if key in results:
            summary += f"- **{label}**: {results[key]}\n"

    coverage = results.get("coverage", {})
    if coverage:
        summary += "\n## Selected-source Coverage\n"
        for label, kind in (
            ("Structural Analyses", "structural"),
            ("LLM Summaries", "summaries"),
            ("LLM Prompts", "prompts"),
        ):
            completed, required = f"{kind}_completed", f"{kind}_required"
            if completed in coverage and required in coverage:
                summary += (
                    f"- **{label}**: {coverage[completed]}/{coverage[required]}\n"
                )
        if "unfinished_work" in results:
            summary += f"- **Unfinished Prompts**: {len(results['unfinished_work'])}\n"
        summary += (
            "\nStructural analysis does not count as an LLM response. "
            "Required unfinished prompts prevent complete-source success.\n"
        )

    summary += f"""
## Analysis Results
- **Files Analyzed**: {len(results.get("analysis_results", []))}
- **Insights Generated**: {len(results.get("model_insights", []))}
- **Suggestions Generated**: {len(results.get("code_suggestions", []))}
- **Documentation Generated**: {len(results.get("documentation_generated", []))}

## Error Summary
"""

    errors = results.get("errors", [])
    if errors:
        for error in errors:
            if isinstance(error, dict):
                location = error.get("file") or error.get("stage") or "Unknown"
                summary += f"- **{location}**: {error.get('error', 'Unknown error')}\n"
            else:
                summary += f"- {error}\n"
    else:
        summary += "- No errors encountered\n"

    summary += "\n## Recommendations\n"

    # Add recommendations based on analysis
    analysis_results = results.get("analysis_results", [])
    if analysis_results:
        total_variables = sum(
            len(analysis.get("variables", [])) for analysis in analysis_results
        )
        total_connections = sum(
            len(analysis.get("connections", [])) for analysis in analysis_results
        )

        summary += f"- Total variables across all models: {total_variables}\n"
        summary += f"- Total connections across all models: {total_connections}\n"

        if total_variables > 50:
            summary += "- Consider modularizing large models\n"

        if total_connections > 100:
            summary += "- Consider simplifying complex graph structures\n"

    return summary
