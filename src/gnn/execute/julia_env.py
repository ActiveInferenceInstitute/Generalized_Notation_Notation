#!/usr/bin/env python3
"""
Julia environment resolution for GNN Step 12.

Owns the committed Julia project environments for Julia-backed frameworks,
the rendered-script command builder, and the Julia package availability
check. Extracted from ``execute.processor``.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from .types import ScriptExecutionContext

logger = logging.getLogger(__name__)

GKSWSTYPE_VAR = "GKSwstype"

#: GR workspace type that renders headlessly (no gksqt Qt window): rendered
#: scripts may plot via Plots.jl/GR, and without it GR spawns the Qt window
#: process, which hangs indefinitely on display-less hosts (CI, agents).
GKSWSTYPE_HEADLESS = "100"


def julia_subprocess_env(
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build the environment for one Julia subprocess.

    Parent environment plus the headless GR default ``GKSwstype=100``:
    forces Plots.jl/GR to render without the ``gksqt`` Qt window, which
    hangs indefinitely on display-less hosts. A ``GKSwstype`` already
    present in the caller's environment wins (explicit override).
    ``overrides`` (e.g. ``{"JULIA_PROJECT": ...}``) apply last and win
    over both.
    """
    env = dict(os.environ)
    env.setdefault(GKSWSTYPE_VAR, GKSWSTYPE_HEADLESS)
    if overrides:
        env.update(overrides)
    return env


def _julia_project_for_framework(framework: str) -> Optional[Path]:
    """Return the committed Julia environment for a supported framework."""
    execute_dir = Path(__file__).resolve().parent
    projects = {
        "rxinfer": execute_dir / "rxinfer",
        "activeinference_jl": execute_dir / "activeinference_jl",
    }
    return projects.get(framework)


def _build_script_execution_command(
    context: ScriptExecutionContext, sandbox_prefix: List[str]
) -> List[str]:
    """Build an explicit, reproducible command for a rendered script."""
    command = list(sandbox_prefix)
    if context.executor == "julia":
        project_dir = _julia_project_for_framework(context.framework)
        if project_dir is None:
            raise ValueError(
                f"No committed Julia project is registered for {context.framework}"
            )
        command.extend(
            [
                context.executor,
                "--startup-file=no",
                f"--project={project_dir}",
                context.script_path.name,
            ]
        )
        return command
    command.extend([context.executor, context.script_path.name])
    return command


def check_julia_dependencies(
    verbose: bool,
    log: Optional[logging.Logger] = None,
    frameworks: Optional[List[str]] = None,
) -> bool:
    """Check if required Julia packages are available.

    Args:
        verbose: Enable verbose logging.
        log: Optional logger instance; defaults to module logger if not provided.

    Returns:
        True if dependencies ok, False otherwise.
    """
    from gnn.utils.runtime_safety.framework_availability import (
        FRAMEWORK_JULIA_PACKAGES,
        check_framework,
    )

    log = log or logger
    for framework in sorted(set(frameworks or FRAMEWORK_JULIA_PACKAGES)):
        if framework not in FRAMEWORK_JULIA_PACKAGES:
            log.warning("Unknown Julia framework '%s'; skipping check", framework)
            continue
        status = check_framework(framework, logger=log)
        if not status.available:
            if verbose:
                log.warning(
                    "Julia readiness failed for %s: %s (%s)",
                    framework,
                    status.reason,
                    status.reason_code,
                )
            return False
    return True
