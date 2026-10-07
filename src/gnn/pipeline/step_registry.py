#!/usr/bin/env python3
"""Canonical pipeline step registry — single source of truth for all 25 steps.

Every step list in the codebase should derive from ``STEPS`` below.
Add ONE entry here when adding a new pipeline step; all downstream lists
(main.py, config.py, pipeline_orchestration/pipeline_template.py, justfile) update automatically.

Usage:

    from gnn.pipeline.step_registry import STEPS, step_for_name, step_output_dir

    for step in STEPS:
        print(step.script_name, step.description)

    step = step_for_name("11_render")
    print(step.module_function)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class StepInfo:
    """Immutable descriptor for one pipeline step."""

    script_stem: str  # e.g. "11_render"
    description: str  # e.g. "Code rendering"
    module_function: str  # e.g. "process_render"
    tags: frozenset[str] = frozenset(("core",))
    stage: str = (
        "core"  # logical stage key: discovery_schema, export_static_viz,
        # simulation_execution, intelligence_analysis, presentation_reporting
    )

    execution_scope: str = "source"
    prerequisites: tuple[int, ...] = ()
    optional_producers: tuple[int, ...] = ()

    @property
    def script_name(self) -> str:
        """Full script filename, e.g. ``11_render.py``."""
        return f"{self.script_stem}.py"

    @property
    def output_dir_name(self) -> str:
        """Standard output subdirectory, e.g. ``11_render_output``."""
        return f"{self.script_stem}_output"


# ---------------------------------------------------------------------------
# Canonical step list — ordered, authoritative
# ---------------------------------------------------------------------------
STEPS: List[StepInfo] = [
    StepInfo(
        "0_template",
        "Template initialization",
        "process_template_standardized",
        frozenset({"core"}),
        stage="discovery_schema",
        execution_scope="corpus",
    ),
    StepInfo(
        "1_setup",
        "Environment setup",
        "setup_orchestrator",
        frozenset({"core"}),
        stage="discovery_schema",
        execution_scope="run",
    ),
    StepInfo(
        "2_tests",
        "Test suite execution",
        "_test_runner_wrapper",
        frozenset({"core", "tests"}),
        stage="discovery_schema",
        execution_scope="run",
        prerequisites=(1,),
    ),
    StepInfo(
        "3_gnn",
        "GNN file processing",
        "process_gnn_multi_format",
        frozenset({"core"}),
        stage="discovery_schema",
    ),
    StepInfo(
        "4_model_registry",
        "Model registry",
        "process_model_registry",
        frozenset({"core"}),
        stage="discovery_schema",
        prerequisites=(3,),
    ),
    StepInfo(
        "5_type_checker",
        "Type checking",
        "_type_check_dispatch",
        frozenset({"core"}),
        stage="discovery_schema",
        prerequisites=(3,),
    ),
    StepInfo(
        "6_validation",
        "Validation",
        "process_validation",
        frozenset({"core"}),
        stage="discovery_schema",
        execution_scope="artifacts",
        prerequisites=(3, 5),
    ),
    StepInfo(
        "7_export",
        "Multi-format export",
        "_export_with_geo",
        frozenset({"core"}),
        stage="export_static_viz",
        execution_scope="artifacts",
        prerequisites=(3,),
    ),
    StepInfo(
        "8_visualization",
        "Visualization",
        "process_visualization",
        frozenset({"core"}),
        stage="export_static_viz",
        prerequisites=(3,),
    ),
    StepInfo(
        "9_advanced_viz",
        "Advanced visualization",
        "process_advanced_viz",
        frozenset({"core"}),
        stage="export_static_viz",
        execution_scope="artifacts",
        prerequisites=(3, 8),
    ),
    StepInfo(
        "10_ontology",
        "Ontology processing",
        "process_ontology",
        frozenset({"core"}),
        stage="export_static_viz",
        prerequisites=(3,),
    ),
    StepInfo(
        "11_render",
        "Code rendering",
        "process_render",
        frozenset({"core"}),
        stage="simulation_execution",
        prerequisites=(3,),
    ),
    StepInfo(
        "12_execute",
        "Execution",
        "process_execute",
        frozenset({"core"}),
        stage="simulation_execution",
        execution_scope="artifacts",
        prerequisites=(3, 11),
    ),
    StepInfo(
        "13_llm",
        "LLM processing",
        "process_llm",
        frozenset({"llm"}),
        stage="intelligence_analysis",
        execution_scope="corpus",
        prerequisites=(3,),
        optional_producers=(10, 12),
    ),
    StepInfo(
        "14_ml_integration",
        "ML integration",
        "process_ml_integration",
        frozenset({"core"}),
        stage="intelligence_analysis",
        prerequisites=(3,),
    ),
    StepInfo(
        "15_audio",
        "Audio processing",
        "process_audio",
        frozenset({"core"}),
        stage="simulation_execution",
        prerequisites=(3,),
        optional_producers=(12,),
    ),
    StepInfo(
        "16_analysis",
        "Analysis",
        "process_analysis",
        frozenset({"core"}),
        stage="simulation_execution",
        execution_scope="corpus",
        prerequisites=(3, 7, 12),
    ),
    StepInfo(
        "17_integration",
        "Integration",
        "process_integration",
        frozenset({"core"}),
        stage="intelligence_analysis",
        execution_scope="corpus",
        prerequisites=(3,),
        optional_producers=(11, 12),
    ),
    StepInfo(
        "18_security",
        "Security",
        "process_security",
        frozenset({"core"}),
        stage="intelligence_analysis",
        prerequisites=(11,),
    ),
    StepInfo(
        "19_research",
        "Research",
        "process_research",
        frozenset({"core"}),
        stage="intelligence_analysis",
        prerequisites=(3,),
    ),
    StepInfo(
        "20_website",
        "Website generation",
        "process_website",
        frozenset({"core"}),
        stage="presentation_reporting",
        execution_scope="corpus",
        prerequisites=(8,),
        optional_producers=(9, 12, 16, 17, 18, 19),
    ),
    StepInfo(
        "21_mcp",
        "Model Context Protocol processing",
        "process_mcp",
        frozenset({"core"}),
        stage="presentation_reporting",
        execution_scope="run",
    ),
    StepInfo(
        "22_gui",
        "GUI (Interactive GNN Constructor)",
        "process_gui",
        frozenset({"core"}),
        stage="presentation_reporting",
        execution_scope="corpus",
        prerequisites=(3,),
    ),
    StepInfo(
        "23_report",
        "Report generation",
        "process_report",
        frozenset({"core"}),
        stage="presentation_reporting",
        execution_scope="run",
        prerequisites=(3,),
        optional_producers=(
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
        ),
    ),
    StepInfo(
        "24_intelligent_analysis",
        "Intelligent pipeline analysis",
        "process_intelligent_analysis",
        frozenset({"core"}),
        stage="presentation_reporting",
        execution_scope="run",
        optional_producers=(
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Derived lookup maps
# ---------------------------------------------------------------------------
_STEPS_BY_STEM: Dict[str, StepInfo] = {s.script_stem: s for s in STEPS}
_STEPS_BY_SCRIPT: Dict[str, StepInfo] = {s.script_name: s for s in STEPS}


def step_for_stem(stem: str) -> Optional[StepInfo]:
    """Look up a step by its script stem (e.g. ``\"11_render\"``), supporting consolidated aliases."""
    resolved = canonical_step_stem(stem)
    return _STEPS_BY_STEM.get(resolved)


def step_for_name(name: str) -> Optional[StepInfo]:
    """Look up a step by script name (e.g. ``\"11_render.py\"``) or stem, supporting consolidated aliases."""
    if name.endswith(".py"):
        stem = name[:-3]
        resolved_stem = canonical_step_stem(stem)
        resolved_name = f"{resolved_stem}.py"
        return _STEPS_BY_SCRIPT.get(resolved_name, _STEPS_BY_SCRIPT.get(name))
    return step_for_stem(name)


def output_dir_for_stem(stem: str) -> Optional[str]:
    """Return the standard output dir name for a step stem, or None."""
    step = step_for_stem(stem)
    return step.output_dir_name if step else None


# ---------------------------------------------------------------------------
# Re-export aliases for existing code and tests
# ---------------------------------------------------------------------------
PIPELINE_STEPS_TUPLE = tuple((step.script_name, step.description) for step in STEPS)

STEP_METADATA_DICT: Dict[str, Dict[str, str]] = {
    step.script_stem: {"name": step.description, "description": step.description}
    for step in STEPS
}

STEP_OUTPUT_DIR_MAP: Dict[str, str] = {
    step.script_stem: step.output_dir_name for step in STEPS
}

STANDARD_MODULE_FUNCTION_NAMES: Dict[str, str] = {
    step.script_stem: step.module_function for step in STEPS
}


def get_step_tags(stem: str) -> frozenset[str]:
    """Return the tags for a given step stem (for pipeline filtering)."""
    step = step_for_stem(stem)
    return step.tags if step else frozenset()


def get_core_steps() -> List[StepInfo]:
    """Return only core-tagged steps (excludes LLM-only steps)."""
    return [s for s in STEPS if "core" in s.tags]


def get_llm_steps() -> List[StepInfo]:
    """Return only LLM-tagged steps."""
    return [s for s in STEPS if "llm" in s.tags]


# ---------------------------------------------------------------------------
# Logical pipeline stages
# ---------------------------------------------------------------------------
STAGE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "discovery_schema": {
        "name": "Discovery & Schema",
        "description": "Initialization, environment, tests, discovery, typing, and validation",
        "steps": [0, 1, 2, 3, 4, 5, 6],
    },
    "export_static_viz": {
        "name": "Export & Static Visualization",
        "description": "Multi-format export, structural visualization, dashboards, and ontology",
        "steps": [7, 8, 9, 10],
    },
    "simulation_execution": {
        "name": "Simulation & Execution",
        "description": "Code rendering, execution rollouts, audio sonification, and post-simulation analysis",
        "steps": [11, 12, 15, 16],
    },
    "intelligence_analysis": {
        "name": "Intelligence & Analysis",
        "description": "LLM cognitive analysis, ML integration, system integration, security, and hypotheses",
        "steps": [13, 14, 17, 18, 19],
    },
    "presentation_reporting": {
        "name": "Presentation & Reporting",
        "description": "Static website generation, MCP tool server, GUI constructors, and executive summaries",
        "steps": [20, 21, 22, 23, 24],
    },
}


def get_stage_steps(stage: str) -> List[StepInfo]:
    """Return steps belonging to a given logical stage."""
    return [s for s in STEPS if s.stage == stage]


def get_pipeline_stages() -> Dict[str, Dict[str, Any]]:
    """Return the canonical pipeline stage definitions."""
    return dict(STAGE_DEFINITIONS)


# ---------------------------------------------------------------------------
# Aliases for consolidated step referencing & renumbering transition
# ---------------------------------------------------------------------------
CONSOLIDATED_STEP_ALIASES: Dict[str, str] = {
    "13_audio": "15_audio",
    "14_analysis": "16_analysis",
    "15_llm": "13_llm",
    "16_ml_integration": "14_ml_integration",
}


def canonical_step_stem(step_alias: str) -> str:
    """Resolve a possibly renumbered/aliased step stem to its canonical script stem."""
    return CONSOLIDATED_STEP_ALIASES.get(step_alias, step_alias)


# ---------------------------------------------------------------------------
# Consolidated in-process execution (S2-11 / V4-STAGE)
# ---------------------------------------------------------------------------
CONSOLIDATED_IN_PROCESS_STEMS: frozenset[str] = frozenset(
    {
        "0_template",
        "3_gnn",
        "5_type_checker",
        "7_export",
        "8_visualization",
        "11_render",
    }
)
"""Canonical stems the opt-in consolidated executor runs in-process.

The numbered scripts stay the CLI surface; these steps additionally run
in-process under ``--consolidated-steps`` via ``gnn.pipeline.step_executor``.
Slice 1 (ADR 0001, ``docs/decisions/0001-consolidated-pipeline-execution.md``)
proved the equivalence contract on the discovery/schema steps; slice 2 adds
the export/viz/render steps after an in-process-safety audit (no subprocess
spawns, no cwd-relative writes in the step path). Extend this set as further
steps are proven equivalent — resolution always flows through
``StepInfo.module_function``, never around it.
"""


def discover_steps() -> Dict[int, StepInfo]:
    """Return a dict mapping step number (from script_stem) to StepInfo."""
    return {int(s.script_stem.split("_")[0]): s for s in STEPS}
