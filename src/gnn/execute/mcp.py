"""
MCP integration for the execute module.

Exposes GNN execution tools: pipeline execution driver, single-model
GNN execution, PyMDP simulation runner, cross-framework comparison,
dependency checker, and module introspection through MCP.
"""

import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from gnn.utils.mcp.dispatch import run_pipeline_step_mcp, run_tool_envelope
from gnn.utils.runtime_safety.workspace_paths import (
    PathValidationError,
    resolve_repo_path,
)

from . import (
    check_dependencies,
    execute_simulation_from_gnn,
    process_execute,
)
from .doctor import collect_doctor_report
from .subprocess_envelope import run_subprocess_envelope
from .validator import ValidationResult

_GNN_MODEL_SUFFIXES = {".md", ".json", ".yaml", ".yml"}


def _resolve_gnn_model_path(path_value: str, *, purpose: str) -> Path:
    path = resolve_repo_path(
        path_value,
        purpose=purpose,
        must_exist=True,
        must_be_dir=False,
    )
    if not path.is_file():
        raise PathValidationError(f"{purpose} must be a file: {path_value}")
    if path.suffix.lower() not in _GNN_MODEL_SUFFIXES:
        allowed = ", ".join(sorted(_GNN_MODEL_SUFFIXES))
        raise PathValidationError(
            f"{purpose} must be a GNN source file ({allowed}): {path_value}"
        )
    return path


def _resolve_output_directory(path_value: str, *, purpose: str) -> Path:
    return resolve_repo_path(
        path_value,
        purpose=purpose,
        must_be_dir=True,
        create=True,
    )


def _resolve_render_output_directory(path_value: str) -> Path:
    path = resolve_repo_path(
        path_value,
        purpose="Render output directory",
        must_exist=True,
        must_be_dir=True,
    )
    summary_file = path / "render_processing_summary.json"
    if not summary_file.is_file():
        raise PathValidationError(
            "Render output directory must contain render_processing_summary.json"
        )
    return path


# ── Domain tools ─────────────────────────────────────────────────────────────


def process_execute_mcp(
    target_directory: str,
    output_directory: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run execution for trusted Step 11 rendered implementations.

    Executes only scripts referenced by the Step 11
    `render_processing_summary.json` in `target_directory`.

    Args:
        target_directory: Repository-local Step 11 render output directory.
        output_directory: Directory to save execution results.
        verbose: Enable verbose logging.

    Returns:
        Dictionary with success flag and processing summary.
    """

    def _resolve(target: str, output: str) -> tuple[Path, Path]:
        return (
            _resolve_render_output_directory(target),
            _resolve_output_directory(output, purpose="Execution output directory"),
        )

    def _interpret(raw: Any) -> tuple[bool, Dict[str, Any], str | None]:
        # Phase 1.1 contract: process_execute may return bool OR int (0/1/2).
        # Coerce to MCP bool envelope, surfacing the "skipped" case separately.
        if isinstance(raw, bool):
            success, skipped = raw, False
        else:  # int
            success, skipped = raw in (0, 2), raw == 2  # 2 = skipped/warnings
        message = (
            "Execute processing skipped (no work found)"
            if skipped
            else "Execute processing completed"
            if success
            else "Execute processing failed"
        )
        return success, {"skipped": skipped}, message

    return run_pipeline_step_mcp(
        process_execute,
        wrapper_name="process_execute_mcp",
        logger=logger,
        target_directory=target_directory,
        output_directory=output_directory,
        verbose=verbose,
        resolve_paths=_resolve,
        extra_step_kwargs=lambda target_path, _output_path: {
            "render_output_dir": target_path,
            "require_render_summary": True,
        },
        interpret_result=_interpret,
        echo_resolved=True,
    )


def execute_gnn_model_mcp(
    gnn_file_path: str,
    output_directory: str,
) -> Dict[str, Any]:
    """
    Execute a single GNN model file via the pipeline executor.

    Delegates to ``execute.execute_simulation_from_gnn`` which dispatches to
    ``GNNExecutor`` (PyMDP by default). The timestep count is read from the
    GNN spec's ``Time`` section by the underlying simulator; callers that need
    to override it should edit the model file.

    Args:
        gnn_file_path: Path to the ``.md`` GNN model file.
        output_directory: Directory to save execution artifacts.

    Returns:
        Dictionary with success flag and execution metadata. The underlying
        return value is merged into the top-level dict.
    """
    try:
        gnn_path = _resolve_gnn_model_path(
            gnn_file_path,
            purpose="GNN model file",
        )
        output_path = _resolve_output_directory(
            output_directory,
            purpose="Execution output directory",
        )
        result: object = execute_simulation_from_gnn(
            gnn_path,
            output_path,
        )
        if isinstance(result, dict):
            merged: dict[str, Any] = {"success": bool(result.get("success", True))}
            merged.update(result)
            return merged
        return {"success": bool(result), "result": result}
    except Exception as e:
        logger.error(f"execute_gnn_model_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def _pymdp_child_environment() -> Dict[str, str]:
    """Environment overlay for the gated PyMDP child process.

    Mirrors the step-12 PyMDP script environment: the project root is
    prepended to ``PYTHONPATH`` so ``-m gnn.execute.pymdp.mcp_child``
    resolves regardless of how the parent imported ``gnn``, the project
    root is exported, and JAX/TF log noise is muted unless the operator
    already set values.
    """
    import gnn

    project_root = Path(gnn.__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["GNN_PROJECT_ROOT"] = str(project_root)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    jax_platform = os.environ.get("GNN_JAX_PLATFORM")
    if jax_platform and str(jax_platform).strip():
        env["JAX_PLATFORM_NAME"] = str(jax_platform).strip()
    return env


def _pymdp_child_payload(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a child-run envelope into the tool's result payload.

    Successful runs merge the simulator's results dict at the top level,
    matching the tool's historical contract; every failure carries the
    envelope's structured receipt (``error_type``, ``return_code``,
    captured streams) so MCP callers can classify the outcome.
    """
    if envelope.get("success"):
        for line in reversed(envelope.get("stdout", "").splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "success" in data:
                payload: Dict[str, Any] = {"success": bool(data["success"])}
                results = data.get("results")
                if isinstance(results, dict):
                    payload.update(results)
                else:
                    payload["result"] = results
                return payload
        return {
            "success": False,
            "error": (
                "PyMDP child reported success but produced an unreadable result line"
            ),
            "stdout": envelope.get("stdout", ""),
            "stderr": envelope.get("stderr", ""),
        }
    payload = {
        "success": False,
        "error": envelope.get("error")
        or f"PyMDP child exited with code {envelope.get('return_code')}",
        "return_code": envelope.get("return_code"),
        "stdout": envelope.get("stdout", ""),
        "stderr": envelope.get("stderr", ""),
    }
    if envelope.get("error_type"):
        payload["error_type"] = envelope["error_type"]
    if envelope.get("cancelled"):
        payload["cancelled"] = True
    return payload


def execute_pymdp_simulation_mcp(
    gnn_file_path: str,
    output_directory: str,
) -> Dict[str, Any]:
    """
    Run a PyMDP Active Inference simulation from a GNN model file.

    Routes the simulation through the shared gated subprocess envelope
    (``run_subprocess_envelope``) — the same gate every other execute
    backend honors — by running ``gnn.execute.pymdp.mcp_child`` in a
    fresh process: the ``GNN_SANDBOX`` prefix semantics apply, the run
    is bounded by the envelope's wall-clock default, and failures come
    back as structured receipts (``error_type``, ``return_code``,
    captured streams) instead of an unbounded in-process run. The child
    calls ``execute.pymdp.execute_pymdp.execute_from_gnn_file``, which
    converts the spec to pymdp matrices (A, B, C, D), instantiates a
    pymdp ``Agent``, and runs the perception-action loop for the
    timesteps declared in the model.

    Args:
        gnn_file_path: Path to the GNN model file.
        output_directory: Directory to write the simulation log and plots.

    Returns:
        Dictionary with success flag and the simulator's results dict merged
        at the top level (keys like ``timesteps_run``, ``output_files``, etc.).
        Gate refusals (for example ``GNN_SANDBOX=require`` with no sandbox
        backend installed) return ``success: False`` with
        ``error_type: "SandboxUnavailable"`` and ``return_code: -1`` — the
        model is never executed.
    """
    try:
        gnn_path = _resolve_gnn_model_path(
            gnn_file_path,
            purpose="GNN model file",
        )
        output_path = _resolve_output_directory(
            output_directory,
            purpose="PyMDP output directory",
        )
        envelope = run_subprocess_envelope(
            [
                sys.executable,
                "-m",
                "gnn.execute.pymdp.mcp_child",
                str(gnn_path),
                str(output_path),
            ],
            env=_pymdp_child_environment(),
            timeout=None,
        )
        return _pymdp_child_payload(envelope)
    except Exception as e:
        logger.error(f"execute_pymdp_simulation_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def run_cross_framework_comparison_mcp(
    gnn_file_path: str,
    output_directory: str,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Render and compare one GNN model across every registered backend.

    Delegates to ``analysis.rxinfer.cross_framework.compare_with_status``,
    which renders the model to RxInfer.jl, PyMDP, ActiveInference.jl, JAX,
    PyTorch, and NumPyro from one parsed spec, executes each backend, and
    writes a self-contained HTML comparison page. Backends whose
    dependencies are missing yield ``unavailable`` skip receipts — never
    execution failures.

    Args:
        gnn_file_path: Path to the ``.md`` GNN model file.
        output_directory: Directory receiving per-framework artifacts and
            the comparison HTML.
        timeout: Optional per-backend execution timeout in seconds.

    Returns:
        Dictionary with ``success``, the ``comparison_html`` path, and a
        per-framework status breakdown (``frameworks`` list with
        ``framework``, ``status``, ``detail`` records).
    """
    try:
        gnn_path = _resolve_gnn_model_path(
            gnn_file_path,
            purpose="GNN model file",
        )
        output_path = _resolve_output_directory(
            output_directory,
            purpose="Cross-framework comparison output directory",
        )

        def _build() -> Dict[str, Any]:
            from gnn.analysis.rxinfer.cross_framework import compare_with_status

            html_path, runs = compare_with_status(gnn_path, output_path, timeout)
            succeeded = sum(1 for run in runs if run.status == "success")
            return {
                "success": True,
                "comparison_html": html_path,
                "frameworks": [dataclasses.asdict(run) for run in runs],
                "frameworks_succeeded": succeeded,
                "frameworks_total": len(runs),
                "message": (
                    f"Cross-framework comparison: {succeeded}/{len(runs)} "
                    "frameworks succeeded"
                ),
            }

        return run_tool_envelope(
            _build,
            wrapper_name="run_cross_framework_comparison_mcp",
            logger=logger,
        )
    except Exception as e:
        logger.error(f"run_cross_framework_comparison_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def check_execute_dependencies_mcp() -> Dict[str, Any]:
    """Check which execution backend dependencies are installed.

    Probes importability of the required Python packages (numpy, matplotlib,
    networkx, pandas, pyyaml, scipy, scikit-learn) and the optional pymdp
    backend, returning one plain-dict record per package so the MCP payload
    is JSON-serializable.

    Returns:
        Dictionary with ``success`` and a ``dependencies`` list of plain dicts
        (``component``, ``status``, ``message``, ``details``, ``suggestion``).
    """

    def _build() -> Dict[str, Any]:
        result: list[ValidationResult] | Dict[str, Any] = check_dependencies()
        if isinstance(result, dict):
            return {"success": True, **result}
        # ``check_dependencies`` returns ``List[ValidationResult]`` (dataclasses);
        # serialize each to a plain dict so the MCP response is JSON-serializable.
        dependencies = [
            dataclasses.asdict(item) if dataclasses.is_dataclass(item) else item
            for item in result
        ]
        return {"success": True, "dependencies": dependencies}

    return run_tool_envelope(
        _build,
        wrapper_name="check_execute_dependencies_mcp",
        logger=logger,
    )


def get_execute_module_info_mcp() -> Dict[str, Any]:
    """
    Return version, feature flags, and API surface of the execute module.

    Includes: module version, execution backends available, PyMDP status,
    error-recovery availability, and supported GNN model types.

    Returns:
        Dictionary with module metadata and feature inventory.
    """

    def _build() -> Dict[str, Any]:
        import importlib

        mod = importlib.import_module(__package__)
        return {
            "success": True,
            "module": __package__,
            "version": getattr(mod, "__version__", "unknown"),
            "features": getattr(mod, "FEATURES", {}),
            "tools": [
                "process_execute",
                "execute_gnn_model",
                "execute_pymdp_simulation",
                "check_execute_dependencies",
                "get_execute_module_info",
                "get_doctor_report",
                "run_cross_framework_comparison",
            ],
        }

    return run_tool_envelope(
        _build,
        wrapper_name="get_execute_module_info_mcp",
        logger=logger,
    )


def get_doctor_report_mcp(
    target_directory: str | None = None,
    output_directory: str | None = None,
    frameworks: str = "all",
) -> Dict[str, Any]:
    """Return one structured capability report: frameworks + Step 12 readiness.

    Composes per-framework availability (``FRAMEWORK_IMPORT_CHECK`` /
    ``check_framework`` plus the Julia PATH gate) with the
    ``plan_execute`` dry-run over an optional target/output directory
    pair. Both directories must be supplied together to probe execution
    readiness; omitting them reports framework availability only.

    Returns:
        Dictionary with ``success``, per-framework ``frameworks`` records,
        ``julia`` availability, availability name lists, the ``execution``
        plan section, and ``execution_ready``.
    """

    def _build() -> Dict[str, Any]:
        return collect_doctor_report(
            target_dir=target_directory,
            output_dir=output_directory,
            frameworks=frameworks,
        )

    return run_tool_envelope(
        _build,
        wrapper_name="get_doctor_report_mcp",
        logger=logger,
    )


# ── MCP Registration ──────────────────────────────────────────────────────────


def register_tools(mcp_instance: Any) -> None:
    """Register execute domain tools with the MCP server."""

    mcp_instance.register_tool(
        "process_execute",
        process_execute_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {
                    "type": "string",
                    "description": "Repository-local Step 11 render output directory",
                },
                "output_directory": {
                    "type": "string",
                    "description": "Execution output directory",
                },
                "verbose": {"type": "boolean", "default": False},
            },
            "required": ["target_directory", "output_directory"],
        },
        "Run trusted Step 11 rendered scripts listed in render_processing_summary.json.",
        module=__package__,
        category="execute",
    )

    mcp_instance.register_tool(
        "execute_gnn_model",
        execute_gnn_model_mcp,
        {
            "type": "object",
            "properties": {
                "gnn_file_path": {
                    "type": "string",
                    "description": "Path to the GNN model file (.md)",
                },
                "output_directory": {
                    "type": "string",
                    "description": "Directory to save execution results",
                },
            },
            "required": ["gnn_file_path", "output_directory"],
        },
        "Execute a single GNN model file via GNNExecutor (PyMDP default); timesteps come from the model's Time section.",
        module=__package__,
        category="execute",
    )

    mcp_instance.register_tool(
        "execute_pymdp_simulation",
        execute_pymdp_simulation_mcp,
        {
            "type": "object",
            "properties": {
                "gnn_file_path": {
                    "type": "string",
                    "description": "Path to the GNN model file",
                },
                "output_directory": {
                    "type": "string",
                    "description": "Directory for simulation outputs",
                },
            },
            "required": ["gnn_file_path", "output_directory"],
        },
        "Run a PyMDP Active Inference simulation from a GNN model (A/B/C/D matrices -> Agent -> perception-action loop).",
        module=__package__,
        category="execute",
    )

    mcp_instance.register_tool(
        "run_cross_framework_comparison",
        run_cross_framework_comparison_mcp,
        {
            "type": "object",
            "properties": {
                "gnn_file_path": {
                    "type": "string",
                    "description": "Path to the GNN model file",
                },
                "output_directory": {
                    "type": "string",
                    "description": "Directory for per-framework artifacts and the comparison HTML",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional per-backend execution timeout in seconds",
                },
            },
            "required": ["gnn_file_path", "output_directory"],
        },
        "Render one GNN model to every registered backend, execute each, and write a cross-framework comparison HTML page with per-framework status receipts.",
        module=__package__,
        category="execute",
    )

    mcp_instance.register_tool(
        "check_execute_dependencies",
        check_execute_dependencies_mcp,
        {},
        "Check which execution backend dependencies (pymdp, numpy, scipy, jax) are installed.",
        module=__package__,
        category="execute",
    )

    mcp_instance.register_tool(
        "get_execute_module_info",
        get_execute_module_info_mcp,
        {},
        "Return version, feature flags, and API surface of the GNN execute module.",
        module=__package__,
        category="execute",
    )

    mcp_instance.register_tool(
        "get_doctor_report",
        get_doctor_report_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {
                    "type": "string",
                    "description": "Directory containing (or sibling to) the Step 11 render output",
                },
                "output_directory": {
                    "type": "string",
                    "description": "Execution output directory used to resolve the sibling render output",
                },
                "frameworks": {
                    "type": "string",
                    "description": '"all", "lite", or a comma-separated framework subset',
                    "default": "all",
                },
            },
            "required": [],
        },
        "Return one structured capability report: per-framework availability plus a Step 12 execution-readiness dry run (no scripts run).",
        module=__package__,
        category="execute",
    )

    logger.info("execute module MCP tools registered (7 real domain tools).")
