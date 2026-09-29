#!/usr/bin/env python3
"""
Dry-run execution planning for GNN Step 12.

``plan_execute`` composes the same discovery / render-contract / dependency
primitives that :func:`execute.processor.process_execute` uses at run time,
but performs **no script execution and no Julia package probing**. It answers
"what would Step 12 do?" for preflight checks, CI gates, and interactive
debugging: which rendered scripts would run, which would be skipped because
their backend dependency is absent, and which the render-summary contract
references but cannot discover on disk.

The planner is deliberately cheap and deterministic: it only does
filesystem reads, a JSON contract load, and Python-side importability probes
(``gnn.utils.runtime_safety.framework_availability.is_framework_available``) plus a PATH-only
Julia lookup (``execute.julia_setup.check_julia_availability``). It never
shells out to ``julia --project=... -e 'using ...'`` (that is the expensive
per-package probe used at run time by
:func:`execute.julia_env.check_julia_dependencies`).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from gnn.utils.runtime_safety.framework_availability import is_framework_available

from .detection import (
    _resolve_render_output_dir,
    find_executable_scripts,
    parse_frameworks_parameter,
)
from .julia_setup import check_julia_availability
from .metadata import _load_render_summary_contract
from .types import ExecutionPlan

logger = logging.getLogger(__name__)

# Frameworks that run under a Julia interpreter. Their run-time availability
# is gated on a Julia toolchain; for planning we only check that ``julia`` is
# on PATH (not the heavier per-package probe).
_JULIA_FRAMEWORKS = frozenset({"rxinfer", "activeinference_jl"})

# Frameworks whose Python dependency importability is probed via the shared
# ``gnn.utils.runtime_safety.framework_availability`` helper (mirrors
# ``execute.processor._is_python_framework_dependency_available``).
_PYTHON_FRAMEWORKS = frozenset(
    {
        "pymdp",
        "jax",
        "discopy",
        "pytorch",
        "numpyro",
        "stan",
        "bnlearn",
        "ngclearn",
        "cpomdp",
    }
)


def _python_dependency_available(framework: str, executor: str) -> bool:
    """Cheap Python-side importability probe for a rendered-script backend.

    Mirrors the pre-flight check in
    :func:`execute.processor.execute_single_script`: a script whose executor is
    the current Python interpreter is only run when its framework module
    imports cleanly.
    """
    if executor != sys.executable:
        return True
    if framework not in _PYTHON_FRAMEWORKS:
        return True
    return is_framework_available(framework, executor=executor, logger=logger)


def _script_entry(script: Dict[str, Any]) -> Dict[str, str]:
    """Project one discovered script into the compact plan entry shape."""
    script_path = Path(str(script["path"]))
    parts = script_path.parts
    # find_executable_scripts does not emit a model_name field; derive it from
    # the rendered layout ``<...>/<model>/<framework>/<script>`` (third path
    # component from the end), matching _build_script_execution_context.
    model_name = parts[-3] if len(parts) >= 3 else "unknown_model"
    return {
        "script_name": str(script["name"]),
        "framework": str(script["framework"]),
        "model_name": model_name,
        "script_path": str(script_path),
    }


def _julia_available() -> bool:
    """PATH-only Julia availability (no version/package probe) for planning."""
    available, _path = check_julia_availability()
    return bool(available)


def _disposition(
    script: Dict[str, Any],
) -> str:
    """Classify one discovered script as 'execute' | 'skip_dependency' | 'unknown'.

    The classification is the same predicate ``execute_single_script`` applies
    before dispatching, minus the security gate (which is a run-time-only
    concern): Python frameworks skip when their module is not importable, and
    Julia frameworks skip when no ``julia`` binary is on PATH.
    """
    framework = str(script["framework"])
    executor = str(script["executor"])
    if framework == "unknown":
        return "unknown"
    if framework in _JULIA_FRAMEWORKS:
        return "execute" if _julia_available() else "skip_dependency"
    if framework in _PYTHON_FRAMEWORKS:
        return (
            "execute"
            if _python_dependency_available(framework, executor)
            else "skip_dependency"
        )
    return "execute"


def plan_execute(
    target_dir: Path,
    output_dir: Path,
    frameworks: str = "all",
    **config: Any,
) -> ExecutionPlan:
    """Plan a Step 12 run without executing any rendered scripts.

    Composes :func:`execute.detection.parse_frameworks_parameter`,
    :func:`execute.detection._resolve_render_output_dir`,
    :func:`execute.metadata._load_render_summary_contract`, and
    :func:`execute.detection.find_executable_scripts` — the same primitives
    :func:`execute.processor.process_execute` uses — then classifies each
    discovered script by the same skip-vs-execute predicate
    :func:`execute.processor.execute_single_script` applies at run time.

    Args:
        target_dir: Directory containing (or sibling to) the Step 11 render
            output, exactly as passed to ``process_execute``.
        output_dir: Step 12 output directory; only used to resolve a sibling
            render-output directory (no files are written).
        frameworks: ``"all"``, ``"lite"``, or a comma-separated subset — same
            semantics as ``process_execute``'s ``frameworks`` argument.
        **config: Forwarded to ``_resolve_render_output_dir``; the only
            recognised key is ``render_output_dir`` (an explicit Step 11
            output path), matching ``process_execute``'s ``kwargs``.

    Returns:
        An :class:`~execute.types.ExecutionPlan` describing the run that
        ``process_execute`` would perform. ``status`` is one of:

        * ``"invalid_frameworks"`` — the frameworks argument was rejected by
          ``validate_frameworks_arg`` (mirrors ``process_execute``'s early
          ``return False``).
        * ``"no_render_output"`` — no render output directory could be
          resolved.
        * ``"no_executable_scripts"`` — the render output exists but no
          executable scripts were discovered.
        * ``"ready"`` — at least one script would be executed.

    Raises:
        ValueError: if ``frameworks`` is rejected by
          ``gnn.utils.runtime_safety.validation_schemas.validate_frameworks_arg`` (the same
          exception ``process_execute`` catches and converts to ``return
          False``; planners wanting the typed plan instead should catch it).
    """
    from gnn.utils.runtime_safety.validation_schemas import validate_frameworks_arg

    frameworks = validate_frameworks_arg(frameworks, context="plan_execute")
    requested_frameworks = parse_frameworks_parameter(frameworks, logger)

    plan: ExecutionPlan = {
        "requested_frameworks": list(requested_frameworks),
        "target_directory": str(target_dir),
        "output_directory": str(output_dir),
        "render_output_dir": None,
        "render_contract_found": False,
        "status": "ready",
        "total_scripts": 0,
        "would_execute": [],
        "would_skip_dependency": [],
        "unknown_framework_scripts": [],
        "missing_render_scripts": [],
        "render_failures": [],
        "unsupported_render_receipts": [],
    }

    render_output_dir = _resolve_render_output_dir(
        target_dir, config, output_dir=output_dir
    )
    plan["render_output_dir"] = (
        str(render_output_dir) if render_output_dir is not None else None
    )

    if render_output_dir is None or not render_output_dir.exists():
        plan["status"] = "no_render_output"
        return plan

    # Scope the render contract to the current invocation, mirroring
    # process_execute: a folder-scoped target_dir must not pull in every other
    # folder's scripts. When target_dir is itself the render-output dir
    # (direct invocation), don't scope.
    scope_target: Optional[Path] = (
        target_dir if render_output_dir != target_dir else None
    )
    (
        allowed_render_scripts,
        render_failures,
        unsupported_receipts,
    ) = _load_render_summary_contract(
        render_output_dir,
        requested_frameworks,
        logger,
        target_dir=scope_target,
    )
    plan["render_contract_found"] = allowed_render_scripts is not None
    plan["render_failures"] = list(render_failures)
    plan["unsupported_render_receipts"] = list(unsupported_receipts)

    if allowed_render_scripts is None and not render_output_dir.exists():
        # Defensive: contract missing and directory gone between checks.
        plan["status"] = "no_render_output"
        return plan

    executable_scripts = find_executable_scripts(
        render_output_dir,
        False,
        logger,
        requested_frameworks,
        allowed_scripts=allowed_render_scripts,
    )

    if allowed_render_scripts is not None:
        executable_scripts = [
            script
            for script in executable_scripts
            if script["path"].resolve() in allowed_render_scripts
        ]
        found_allowed_scripts = {
            script["path"].resolve() for script in executable_scripts
        }
        plan["missing_render_scripts"] = sorted(
            str(path) for path in allowed_render_scripts - found_allowed_scripts
        )

    plan["total_scripts"] = len(executable_scripts)

    for script in executable_scripts:
        entry = _script_entry(script)
        disposition = _disposition(script)
        if disposition == "execute":
            plan["would_execute"].append(entry)
        elif disposition == "skip_dependency":
            plan["would_skip_dependency"].append(entry)
        else:
            plan["unknown_framework_scripts"].append(entry)

    if not executable_scripts:
        plan["status"] = "no_executable_scripts"
    else:
        plan["status"] = "ready"

    return plan
