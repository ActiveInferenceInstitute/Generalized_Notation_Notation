#!/usr/bin/env python3
"""GNN capability doctor: one structured environment + readiness probe.

Composes two existing surfaces into a single JSON-serializable report so
MCP clients, CI preflights, and interactive debugging need one call
instead of re-assembling the answer from scattered probes:

1. **Per-framework availability** — the canonical
   ``FRAMEWORK_IMPORT_CHECK``/``check_framework`` primitives from
   ``gnn.utils.runtime_safety.framework_availability`` (the same source
   Step 11/12 use to decide run vs. skip), including the committed-project package gate for the Julia
   frameworks (``rxinfer``, ``activeinference_jl``).
2. **Step 12 execution readiness** — ``gnn.execute.planning.plan_execute``
   dry-run over a target/output directory pair, classified by the same
   four planner statuses (``ready`` / ``no_render_output`` /
   ``no_executable_scripts`` / ``invalid_frameworks``).

The probe is offline: supervised import/package loading and directory reads.
It executes no rendered models and performs no installs or Pkg operations.

The MCP exposure is the thin ``get_doctor_report_mcp`` wrapper in this
package's ``mcp.py`` behind the canonical ``run_tool_envelope``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from gnn.utils.runtime_safety.framework_availability import (
    FRAMEWORK_IMPORT_CHECK,
    FRAMEWORK_JULIA_PACKAGES,
    FRAMEWORK_PROBE_STATEMENT,
    FrameworkStatus,
    check_framework,
)

from .julia_setup import check_julia_availability
from .planning import _JULIA_FRAMEWORKS, plan_execute

# planning's framework classification is reused verbatim (not duplicated):
# the frozensets there are the planner's truth, and duplicating them here
# would let the doctor drift from Step 12's actual skip/execute semantics.

logger = logging.getLogger(__name__)

__all__ = ["collect_doctor_report"]

PathLike = Union[str, Path]


def _python_framework_entry(name: str) -> Dict[str, Any]:
    """Build the report entry for one ``FRAMEWORK_IMPORT_CHECK`` framework."""
    status = check_framework(name, logger=logger)
    probe_module = FRAMEWORK_IMPORT_CHECK[name][0]
    entry: Dict[str, Any] = {
        "kind": "python_import",
        "probe_module": probe_module,
        "available": status.available,
        "requires_toolchain_probe": name in FRAMEWORK_PROBE_STATEMENT,
    }
    if name in FRAMEWORK_PROBE_STATEMENT:
        entry["toolchain_probe"] = FRAMEWORK_PROBE_STATEMENT[name]
    entry.update(_readiness_fields(status))
    return entry


def _readiness_fields(status: FrameworkStatus) -> Dict[str, Any]:
    """The shared structured diagnosis, including supervision evidence."""
    entry: Dict[str, Any] = {"available": status.available}
    if not status.available:
        for key in ("missing_module", "install_hint", "reason_code", "reason"):
            entry[key] = getattr(status, key)
    for key in (
        "execution_error_type",
        "cleanup_verified",
        "streams_drained",
        "runtime_version",
        "backend_version",
    ):
        value = getattr(status, key)
        if value is not None:
            entry[key] = value
    return entry


def _julia_framework_entry(name: str) -> Dict[str, Any]:
    """Load the registered packages in the committed project, without installs."""
    return {
        "kind": "julia_project",
        "probe_packages": list(FRAMEWORK_JULIA_PACKAGES[name]),
        **_readiness_fields(check_framework(name, logger=logger)),
    }


def collect_doctor_report(
    target_dir: Optional[PathLike] = None,
    output_dir: Optional[PathLike] = None,
    frameworks: str = "all",
) -> Dict[str, Any]:
    """Return one structured capability report for the GNN pipeline.

    Args:
        target_dir: Directory containing (or sibling to) the Step 11 render
            output, exactly as passed to :func:`execute.planning.plan_execute`.
            Supply together with ``output_dir`` or not at all.
        output_dir: Step 12 output directory used only to resolve the sibling
            render-output directory; no files are written.
        frameworks: ``"all"``, ``"lite"``, or a comma-separated subset —
            forwarded to ``plan_execute`` unchanged. Invalid values raise
            ``ValueError`` (the planner's contract).

    Returns:
        A JSON-serializable dict with:

        - ``frameworks``: per-framework records keyed by name. Python-probe
          frameworks (``FRAMEWORK_IMPORT_CHECK``) carry ``kind`` =
          ``"python_import"`` with ``probe_module``, importability, and the
          ``install_hint``/``missing_module`` fields when unavailable; Julia
          frameworks carry ``kind`` = ``"julia_project"`` and the committed-project
          package diagnosis; the ``julia`` section separately reports PATH evidence.
        - ``julia``: the shared ``check_julia_availability`` PATH probe.
        - ``frameworks_available`` / ``frameworks_missing``: name lists.
        - ``execution``: the full ``plan_execute`` dry-run plan, or
          ``{"status": "not_probed", "reason": ...}`` when no directory pair
          was supplied.
        - ``execution_ready``: at least one entry in ``would_execute``; ``None`` when
          the execution section was not probed.

    Raises:
        ValueError: if exactly one of ``target_dir``/``output_dir`` is
            supplied, or when ``plan_execute`` rejects ``frameworks``.
    """
    if (target_dir is None) != (output_dir is None):
        raise ValueError(
            "target_dir and output_dir must be supplied together to probe "
            "execution readiness (or both omitted to probe availability only)"
        )

    julia_available, julia_path = check_julia_availability()

    frameworks_section: Dict[str, Any] = {}
    for name in sorted(FRAMEWORK_IMPORT_CHECK):
        frameworks_section[name] = _python_framework_entry(name)
    for name in sorted(_JULIA_FRAMEWORKS):
        frameworks_section[name] = _julia_framework_entry(name)

    available_names = sorted(
        name for name, entry in frameworks_section.items() if entry["available"]
    )
    missing_names = sorted(set(frameworks_section) - set(available_names))

    report: Dict[str, Any] = {
        "success": True,
        "frameworks": frameworks_section,
        "julia": {
            "available": bool(julia_available),
            "path": str(julia_path) if julia_path else None,
        },
        "frameworks_available": available_names,
        "frameworks_missing": missing_names,
    }

    if target_dir is None or output_dir is None:
        report["execution"] = {
            "status": "not_probed",
            "reason": "no target/output directories supplied",
        }
        report["execution_ready"] = None
    else:
        plan = plan_execute(
            Path(target_dir),
            Path(output_dir),
            frameworks=frameworks,
        )
        report["execution"] = plan
        report["execution_ready"] = bool(plan["would_execute"])

    return report
