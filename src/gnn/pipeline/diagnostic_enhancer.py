#!/usr/bin/env python3
"""
Pipeline Diagnostic Enhancer

This module provides enhanced diagnostic capabilities for pipeline execution summaries.
It analyzes pipeline results and provides actionable insights and recommendations.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, cast

logger = logging.getLogger(__name__)


class PipelineDiagnosticEnhancer:
    """
    Enhanced diagnostic analyzer for pipeline execution summaries.

    Provides detailed analysis of pipeline performance, failures, warnings,
    and actionable recommendations for improvements.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.known_issues = {
            "gradio.*has no attribute.*Blocks": {
                "category": "dependency_version",
                "solution": "Upgrade Gradio to version 4.0+ using: uv add gradio",
                "priority": "critical",
            },
            "POMDP processing modules not available.*attempted relative import": {
                "category": "import_error",
                "solution": "Fix relative imports in render/processor/ submodules",
                "priority": "high",
            },
            "matplotlib.*incompatible constructor arguments": {
                "category": "matplotlib_dpi",
                "solution": "Fix matplotlib DPI calculation in visualization modules",
                "priority": "medium",
            },
            "PyMDP not available": {
                "category": "optional_dependency",
                "solution": "Install PyMDP for full simulation capabilities: uv add pymdp",
                "priority": "low",
            },
        }

    def enhance_summary(self, summary_path: Path) -> Dict[str, Any]:
        """
        Enhance an existing pipeline summary with diagnostic information.

        Args:
            summary_path: Path to the pipeline execution summary JSON file

        Returns:
            Enhanced summary dictionary with diagnostic insights
        """
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)

            # Add diagnostic analysis
            diagnostics = self._analyze_pipeline_execution(summary)
            summary["diagnostics"] = diagnostics

            # Add recommendations
            recommendations = self._generate_recommendations(summary)
            summary["recommendations"] = recommendations

            # Add health score
            health_score = self._calculate_health_score(summary)
            summary["health_score"] = health_score

            # Save enhanced summary
            enhanced_path = summary_path.parent / f"enhanced_{summary_path.name}"
            with open(enhanced_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            logger.info(f"Enhanced pipeline summary saved to: {enhanced_path}")
            return cast("dict[str, Any]", summary)

        except Exception as e:
            logger.error(f"Failed to enhance pipeline summary: {e}")
            return {}

    def _analyze_pipeline_execution(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pipeline execution for issues and patterns."""
        diagnostics: dict[str, Any] = {
            "execution_analysis": {},
            "performance_analysis": {},
            "error_analysis": {},
            "warning_analysis": {},
            "dependency_analysis": {},
        }

        steps = summary.get("steps", [])
        if not steps:
            return diagnostics

        # Execution analysis
        total_steps = len(steps)
        successful_steps = sum(1 for step in steps if step.get("status") == "SUCCESS")
        warning_steps = sum(
            1 for step in steps if step.get("status") == "SUCCESS_WITH_WARNINGS"
        )
        failed_steps = sum(
            1 for step in steps if step.get("status", "").startswith("FAILED")
        )

        diagnostics["execution_analysis"] = {
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "warning_steps": warning_steps,
            "failed_steps": failed_steps,
            "success_rate": (successful_steps + warning_steps) / total_steps * 100
            if total_steps > 0
            else 0,
            "critical_failure_rate": failed_steps / total_steps * 100
            if total_steps > 0
            else 0,
        }

        # Performance analysis
        durations = [step.get("duration_seconds", 0) for step in steps]
        if durations:
            diagnostics["performance_analysis"] = {
                "total_duration": sum(durations),
                "average_duration": sum(durations) / len(durations),
                "median_duration": sorted(durations)[len(durations) // 2],
                "slowest_step": max(steps, key=lambda s: s.get("duration_seconds", 0)),
                "fastest_step": min(steps, key=lambda s: s.get("duration_seconds", 0)),
            }

        # Error and warning analysis
        all_errors: list[Any] = []
        all_warnings: list[Any] = []

        for step in steps:
            stderr = step.get("stderr", "")
            stdout = step.get("stdout", "")

            # Extract error patterns
            for pattern, info in self.known_issues.items():
                if re.search(pattern, stderr + stdout, re.IGNORECASE):
                    issue_info: dict[Any, Any] = {
                        "step": step.get("script_name", "unknown"),
                        "pattern": pattern,
                        **info,
                    }
                    if info["priority"] in ["critical", "high"]:
                        all_errors.append(issue_info)
                    else:
                        all_warnings.append(issue_info)

        diagnostics["error_analysis"] = all_errors
        diagnostics["warning_analysis"] = all_warnings

        # Dependency analysis
        diagnostics["dependency_analysis"] = self._analyze_dependencies(steps)

        return diagnostics

    def _analyze_dependencies(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze dependency-related issues."""
        missing_deps: set[Any] = set()
        optional_deps: set[Any] = set()
        version_issues: set[Any] = set()

        for step in steps:
            stderr = step.get("stderr", "")

            # Common dependency patterns
            if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
                # Extract module name if possible
                module_match = re.search(r"No module named ['\"]([^'\"]+)['\"]", stderr)
                if module_match:
                    missing_deps.add(module_match.group(1))

            if "not available" in stderr.lower():
                # Extract optional dependency names
                optional_match = re.search(r"(\w+) not available", stderr)
                if optional_match:
                    optional_deps.add(optional_match.group(1))

        return {
            "missing_dependencies": list(missing_deps),
            "optional_dependencies": list(optional_deps),
            "version_issues": list(version_issues),
        }

    def _generate_recommendations(
        self, summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis."""
        recommendations: list[Any] = []
        diagnostics = summary.get("diagnostics", {})

        # Check for critical failures
        execution = diagnostics.get("execution_analysis", {})
        if execution.get("failed_steps", 0) > 0:
            recommendations.append(
                {
                    "type": "critical",
                    "category": "execution",
                    "title": "Critical Step Failures Detected",
                    "description": f"Pipeline has {execution['failed_steps']} failed steps that need immediate attention",
                    "action": "Review failed step logs and resolve blocking issues before proceeding",
                }
            )

        # Performance recommendations
        performance = diagnostics.get("performance_analysis", {})
        if performance.get("average_duration", 0) > 5.0:
            slowest = performance.get("slowest_step", {})
            recommendations.append(
                {
                    "type": "optimization",
                    "category": "performance",
                    "title": "Performance Optimization Opportunity",
                    "description": f"Step {slowest.get('script_name', 'unknown')} takes {slowest.get('duration_seconds', 0):.2f}s",
                    "action": "Consider optimizing visualization or processing algorithms",
                }
            )

        # Dependency recommendations
        deps = diagnostics.get("dependency_analysis", {})
        if deps.get("missing_dependencies"):
            recommendations.append(
                {
                    "type": "dependency",
                    "category": "setup",
                    "title": "Missing Dependencies",
                    "description": f"Missing: {', '.join(deps['missing_dependencies'])}",
                    "action": f"Install missing dependencies: uv add {' '.join(deps['missing_dependencies'])}",
                }
            )

        # Error-specific recommendations
        for error in diagnostics.get("error_analysis", []):
            recommendations.append(
                {
                    "type": "error_fix",
                    "category": error["category"],
                    "title": f"Fix {error['step']} Issue",
                    "description": f"{error['category'].title()} issue in {error['step']}",
                    "action": error["solution"],
                }
            )

        return recommendations

    def _calculate_health_score(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall pipeline health score."""
        diagnostics = summary.get("diagnostics", {})
        execution = diagnostics.get("execution_analysis", {})

        base_score = 100
        deductions = 0

        # Deduct for failures
        failed_steps = execution.get("failed_steps", 0)
        deductions += failed_steps * 25  # 25 points per failed step

        # Deduct for warnings
        warning_steps = execution.get("warning_steps", 0)
        deductions += warning_steps * 5  # 5 points per warning

        # Deduct for performance issues
        performance = diagnostics.get("performance_analysis", {})
        if performance.get("average_duration", 0) > 10.0:
            deductions += 10  # Performance penalty

        final_score = max(0, base_score - deductions)

        # Health rating
        if final_score >= 90:
            rating = "excellent"
        elif final_score >= 75:
            rating = "good"
        elif final_score >= 50:
            rating = "fair"
        else:
            rating = "poor"

        return {
            "score": final_score,
            "rating": rating,
            "deductions": deductions,
            "breakdown": {
                "base_score": base_score,
                "failure_penalty": failed_steps * 25,
                "warning_penalty": warning_steps * 5,
                "performance_penalty": min(
                    10, max(0, (performance.get("average_duration", 0) - 10) * 2)
                ),
            },
        }


def enhance_pipeline_summary(summary_path: Path) -> Dict[str, Any]:
    """
    Convenience function to enhance a pipeline summary.

    Args:
        summary_path: Path to pipeline execution summary JSON

    Returns:
        Enhanced summary with diagnostic information
    """
    enhancer = PipelineDiagnosticEnhancer()
    return enhancer.enhance_summary(summary_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        summary_path = Path(sys.argv[1])
        if summary_path.exists():
            enhanced = enhance_pipeline_summary(summary_path)
            print(
                f"Enhanced summary with health score: {enhanced.get('health_score', {}).get('score', 0)}/100"
            )
        else:
            print(f"Summary file not found: {summary_path}")
    else:
        print("Usage: python diagnostic_enhancer.py <pipeline_summary.json>")
