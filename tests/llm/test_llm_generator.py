"""Deterministic tests for llm.generator heuristic outputs."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.llm.generator import (
    generate_code_suggestions,
    generate_documentation,
    generate_llm_summary,
    generate_model_insights,
)

pytestmark = pytest.mark.unit


def _analysis(total_elements: int, variables: int, connections: int) -> dict:
    return {
        "file_name": "agent.md",
        "file_path": "/tmp/agent.md",
        "variables": [
            {"name": f"s{i}", "definition": f"s{i}: boolean"} for i in range(variables)
        ],
        "connections": [
            {"source": "a", "target": "b", "connection": "a > b"}
            for _ in range(connections)
        ],
        "complexity_metrics": {"total_elements": total_elements, "density": 0.5},
        "patterns": {
            "anti_patterns": ["No connections defined"],
            "suggestions": ["Consider splitting"],
        },
    }


class TestGenerateModelInsights:
    def test_low_complexity(self) -> None:
        insights = generate_model_insights(_analysis(5, 3, 2))
        assert insights["model_complexity"] == "low"
        assert "Simple and maintainable model" in insights["strengths"]

    def test_high_complexity_and_anti_patterns_flow_through(self) -> None:
        insights = generate_model_insights(_analysis(80, 25, 55))
        assert insights["model_complexity"] == "high"
        assert "No connections defined" in insights["weaknesses"]
        assert "Consider splitting" in insights["recommendations"]


class TestGenerateCodeSuggestions:
    def test_unknown_type_ratio_triggers_improvement(self) -> None:
        analysis = {
            "variables": [{"definition": "a = 1"}, {"definition": "b = 2"}],
            "connections": [],
            "complexity_metrics": {"density": 0.1},
        }
        suggestions = generate_code_suggestions(analysis)
        assert any("type annotations" in s for s in suggestions["improvements"])

    def test_high_density_triggers_optimization(self) -> None:
        analysis = {
            "variables": [{"definition": "a: boolean"}],
            "connections": [],
            "complexity_metrics": {"density": 3.0},
        }
        suggestions = generate_code_suggestions(analysis)
        assert any("sparse" in s for s in suggestions["optimizations"])


class TestGenerateDocumentation:
    def test_overview_counts_and_sections(self) -> None:
        docs = generate_documentation(_analysis(9, 3, 2))
        assert "3 variables and 2 connections" in docs["model_overview"]
        assert len(docs["variable_documentation"]) == 3
        assert len(docs["connection_documentation"]) == 2
        assert docs["usage_examples"]

    def test_empty_model(self) -> None:
        docs = generate_documentation(
            {"file_name": "empty.md", "variables": [], "connections": []}
        )
        assert "0 variables and 0 connections" in docs["model_overview"]


class TestGenerateLLMSummary:
    def test_partial_coverage_preserves_llm_and_structural_distinction(self) -> None:
        summary = generate_llm_summary(
            {
                "processed_files": 38,
                "selected_files": 38,
                "total_files_discovered": 38,
                "success": False,
                "status": "timed_out",
                "provider": "ollama",
                "selected_model": "configured:exact",
                "budget_seconds": 900,
                "coverage": {
                    "structural_completed": 38,
                    "structural_required": 38,
                    "summaries_completed": 33,
                    "summaries_required": 38,
                    "prompts_completed": 82,
                    "prompts_required": 342,
                },
                "unfinished_work": [f"unfinished:{i}" for i in range(260)],
            }
        )
        for fact in (
            "**Status**: timed_out",
            "**Structural Analyses**: 38/38",
            "**LLM Summaries**: 33/38",
            "**LLM Prompts**: 82/342",
            "**Unfinished Prompts**: 260",
            "**Provider**: ollama",
            "**Model**: configured:exact",
            "**Budget (seconds)**: 900",
            "Structural analysis does not count as an LLM response.",
        ):
            assert fact in summary

    def test_missing_model_is_attributed_to_preflight(self) -> None:
        summary = generate_llm_summary(
            {
                "success": False,
                "errors": [
                    {
                        "stage": "preflight",
                        "error": "Configured Ollama model is not installed",
                    }
                ],
                "coverage": {
                    "structural_completed": 1,
                    "structural_required": 1,
                    "summaries_completed": 0,
                    "summaries_required": 1,
                    "prompts_completed": 0,
                    "prompts_required": 9,
                },
                "unfinished_work": [f"case:{i}" for i in range(9)],
            }
        )
        assert "**preflight**: Configured Ollama model is not installed" in summary
        assert "**LLM Summaries**: 0/1" in summary
        assert "**LLM Prompts**: 0/9" in summary
        assert "**Unfinished Prompts**: 9" in summary

    def test_legacy_results_do_not_invent_coverage_or_budget(self) -> None:
        summary = generate_llm_summary({"processed_files": 1, "success": True})
        assert "**Files Processed**: 1" in summary
        assert "Selected-source Coverage" not in summary
        assert "Budget (seconds)" not in summary

    def test_error_lines_rendered(self) -> None:
        summary = generate_llm_summary(
            {
                "processed_files": 1,
                "success": False,
                "errors": [{"file": "a.md", "error": "boom"}],
                "analysis_results": [],
            }
        )
        assert "**a.md**: boom" in summary

    def test_recommendations_for_large_corpus(self) -> None:
        big = [{"variables": [{"n": i} for i in range(60)], "connections": []}]
        summary = generate_llm_summary(
            {
                "processed_files": 1,
                "success": True,
                "errors": [],
                "analysis_results": big,
            }
        )
        assert "modularizing large models" in summary
