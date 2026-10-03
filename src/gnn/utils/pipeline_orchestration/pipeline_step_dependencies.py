"""Shared pipeline step dependency definitions.

This module captures execution-order dependencies between numbered pipeline
steps. It intentionally models step-output prerequisites, not import-time source
dependencies such as Step 21 discovering module ``mcp.py`` files.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Iterable

from gnn.pipeline.step_registry import STEPS

PIPELINE_STEP_SCRIPTS = MappingProxyType(
    {int(step.script_stem.split("_")[0]): step.script_name for step in STEPS}
)
PIPELINE_SCRIPT_STEPS = MappingProxyType(
    {script: step for step, script in PIPELINE_STEP_SCRIPTS.items()}
)
PIPELINE_STEP_DEPENDENCIES = MappingProxyType(
    {int(step.script_stem.split("_")[0]): step.prerequisites for step in STEPS}
)
PIPELINE_OPTIONAL_PRODUCERS = MappingProxyType(
    {int(step.script_stem.split("_")[0]): step.optional_producers for step in STEPS}
)


def normalize_script_name(script_name: str | Path) -> str:
    """Return a canonical script filename for a step identifier."""
    return Path(str(script_name)).name


def step_number_for_script(script_name: str | Path) -> int | None:
    """Return the numeric step for a script filename, if known."""
    return PIPELINE_SCRIPT_STEPS.get(normalize_script_name(script_name))


def dependency_steps_for_step(step_number: int) -> tuple[int, ...]:
    """Return direct dependency step numbers for ``step_number``."""
    return tuple(PIPELINE_STEP_DEPENDENCIES.get(step_number, ()))


def dependency_scripts_for_script(script_name: str | Path) -> list[str]:
    """Return direct dependency script filenames for ``script_name``."""
    step_number = step_number_for_script(script_name)
    if step_number is None:
        return []
    return [
        PIPELINE_STEP_SCRIPTS[dep]
        for dep in dependency_steps_for_step(step_number)
        if dep in PIPELINE_STEP_SCRIPTS
    ]


def resolve_step_dependencies(requested_steps: Iterable[int]) -> list[int]:
    """Return requested steps plus recursive dependencies in pipeline order."""
    resolved: set[int] = set()
    visiting: set[int] = set()

    def visit(step_number: int) -> None:
        """Provide visit behavior."""
        if step_number in resolved:
            return
        if step_number in visiting:
            raise ValueError(
                f"Cycle detected in pipeline dependencies at step {step_number}"
            )
        visiting.add(step_number)
        for dependency in dependency_steps_for_step(step_number):
            visit(dependency)
        visiting.remove(step_number)
        resolved.add(step_number)

    for step_number in requested_steps:
        visit(step_number)

    return sorted(resolved)
