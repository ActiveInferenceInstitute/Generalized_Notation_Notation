"""
Batch execution and report helpers for the GNN executor: framework
directory/result bookkeeping, dependency checks, per-framework dispatch,
markdown/JSON summaries, and the rendered-simulators entry point.

Split from ``gnn.execute.executor`` (W7-A band split); the moved ranges are
byte-verbatim. ``_framework_specs`` and the ``log_step_*`` helpers resolve
through ``gnn.execute.executor`` at call time so executor-namespace
monkeypatches stay observable.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from gnn.execute.executor_specs import FRAMEWORK_DIR_NAMES, ExecutorFrameworkSpec


def _framework_specs(*args: Any, **kwargs: Any) -> Any:
    """Resolve through ``gnn.execute.executor`` (call-time indirection)."""
    from gnn.execute import executor

    return executor._framework_specs(*args, **kwargs)


def _log_step(name: str, *args: Any, **kwargs: Any) -> Any:
    """Resolve a ``log_step_*`` helper through ``gnn.execute.executor``."""
    from gnn.execute import executor

    return getattr(executor, name)(*args, **kwargs)


def log_step_error(*args: Any, **kwargs: Any) -> Any:
    """See ``gnn.execute.executor.log_step_error`` (call-time indirection)."""
    return _log_step("log_step_error", *args, **kwargs)


def log_step_start(*args: Any, **kwargs: Any) -> Any:
    """See ``gnn.execute.executor.log_step_start`` (call-time indirection)."""
    return _log_step("log_step_start", *args, **kwargs)


def log_step_success(*args: Any, **kwargs: Any) -> Any:
    """See ``gnn.execute.executor.log_step_success`` (call-time indirection)."""
    return _log_step("log_step_success", *args, **kwargs)


def log_step_warning(*args: Any, **kwargs: Any) -> Any:
    """See ``gnn.execute.executor.log_step_warning`` (call-time indirection)."""
    return _log_step("log_step_warning", *args, **kwargs)


from gnn.pipeline.config import get_output_dir_for_script
from gnn.utils import performance_tracker


def _create_framework_dirs(
    execution_output_dir: Path, logger: logging.Logger
) -> dict[str, Path]:
    """Create and return framework-specific execution directories."""
    framework_dirs = {name: execution_output_dir / name for name in FRAMEWORK_DIR_NAMES}
    for framework_dir in framework_dirs.values():
        framework_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created framework directory: {framework_dir}")
    return framework_dirs


def _initialize_execution_results(
    target_dir: Path, framework_dirs: dict[str, Path]
) -> dict[str, Any]:
    """Build the common execution summary envelope.

    The per-framework ``*_executions`` lists are derived from the
    :func:`_framework_specs` registry so adding a framework is a one-line
    change (the registry is the single source of truth for both the result
    keys and the dispatch wiring).
    """
    result: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_directory": str(target_dir),
        "framework_execution_dirs": {k: str(v) for k, v in framework_dirs.items()},
        "total_successes": 0,
        "total_failures": 0,
        "required_unfinished": 0,
        "dependency_issues": [],
        "syntax_errors": [],
        "execution_details": {},
    }
    for spec in _framework_specs(resolve_availability=False):
        result[spec.result_key] = []
    return result


def _check_python_dependencies(
    execution_results: dict[str, Any], logger: logging.Logger
) -> None:
    """Record bounded core-framework diagnoses for explicit census callers."""
    _record_readiness_census(execution_results, logger, ("jax", "pymdp"))


def _check_julia_availability(
    execution_results: dict[str, Any], logger: logging.Logger
) -> None:
    """Record bounded committed-project diagnoses, without package installs."""
    _record_readiness_census(
        execution_results, logger, ("rxinfer", "activeinference_jl")
    )


def _record_readiness_census(
    execution_results: dict[str, Any],
    logger: logging.Logger,
    frameworks: tuple[str, ...],
) -> None:
    from dataclasses import asdict

    from gnn.utils.runtime_safety.framework_availability import check_framework

    diagnoses = execution_results.setdefault("readiness", {})
    for framework in frameworks:
        diagnosis = check_framework(framework, logger=logger)
        diagnoses[framework] = asdict(diagnosis)
        if not diagnosis.available:
            execution_results["dependency_issues"].append(
                f"{framework}: {diagnosis.reason_code}: {diagnosis.reason}"
            )


def _validate_pymdp_script_syntax(
    target_dir: Path, execution_results: dict[str, Any], logger: logging.Logger
) -> None:
    """Compile rendered PyMDP scripts so syntax errors appear in the summary."""
    pymdp_dir = target_dir / "pymdp"
    if not pymdp_dir.exists():
        return

    for script in pymdp_dir.glob("*.py"):
        try:
            with open(script, "r") as f:
                compile(f.read(), script.name, "exec")
            logger.debug(f"✅ PyMDP script syntax valid: {script.name}")
        except SyntaxError as e:
            logger.warning(f"⚠️ PyMDP script syntax error in {script.name}: {e}")
            execution_results["syntax_errors"].append(f"PyMDP: {script.name} - {e}")


def _append_framework_result(
    execution_results: dict[str, Any],
    spec: ExecutorFrameworkSpec,
    status: str,
    message: str,
    output_dir: Path,
    records: list[dict[str, Any]] | None = None,
    evidence: dict[str, Any] | None = None,
) -> None:
    """Append a normalized framework record with its original script receipts."""
    execution_results[spec.result_key].append(
        {
            "status": status,
            "message": message,
            "output_dir": str(output_dir),
            **({"script_results": records} if records is not None else {}),
            **(evidence or {}),
        }
    )


def _execute_framework_spec(
    spec: ExecutorFrameworkSpec,
    target_dir: Path,
    framework_dirs: dict[str, Path],
    execution_results: dict[str, Any],
    logger: logging.Logger,
    recursive: bool,
    verbose: bool,
    timeout: Optional[int] = None,
    required_scripts: tuple[Path, ...] | None = None,
) -> None:
    """Execute one framework runner and record its status."""
    from gnn.execute.preconditions import execution_precondition

    output_dir = framework_dirs[spec.framework_dir_key]
    precondition = execution_precondition(timeout, None)
    if precondition is not None:
        _record_precondition_failure(
            spec, output_dir, execution_results, precondition, required_scripts
        )
        return
    if not spec.available:
        from gnn.execute.preconditions import unavailable_framework_result

        evidence = (
            unavailable_framework_result(spec.diagnosis)
            if spec.diagnosis is not None
            else {"success": False, "skipped": True, "reason": spec.unavailable_message}
        )
        required = bool(required_scripts)
        evidence["required_work"] = required
        if required:
            execution_results["required_unfinished"] = (
                execution_results.get("required_unfinished", 0) + 1
            )
        if not evidence["skipped"]:
            execution_results["total_failures"] += 1
        status = "SKIPPED" if evidence["skipped"] else "FAILED"
        logger.warning("%s: %s", spec.framework_dir_key, evidence.get("reason"))
        _append_framework_result(
            execution_results,
            spec,
            status,
            str(evidence.get("reason") or spec.unavailable_message),
            output_dir,
            records=[{"script": str(script), **evidence} for script in required_scripts]
            if required_scripts
            else None,
            evidence={key: value for key, value in evidence.items() if key != "status"},
        )
        return

    try:
        with performance_tracker.track_operation(spec.operation_name):
            logger.info(spec.start_message)
            if spec.framework_dir_key == "pymdp":
                _validate_pymdp_script_syntax(target_dir, execution_results, logger)

            success = spec.runner(
                rendered_simulators_dir=target_dir,
                execution_output_dir=output_dir,
                recursive_search=recursive,
                verbose=verbose,
                timeout=timeout,
            )

            records = None
            if isinstance(success, list):
                records = success
                if not records or all(record.get("skipped") for record in records):
                    required = bool(records) or bool(required_scripts)
                    if required:
                        execution_results["required_unfinished"] = (
                            execution_results.get("required_unfinished", 0) + 1
                        )
                    _append_framework_result(
                        execution_results,
                        spec,
                        "SKIPPED",
                        "No runnable scripts; inspect per-script skip receipts",
                        output_dir,
                        records,
                        evidence={"required_work": required},
                    )
                    return
                if any(record.get("skipped") for record in records) and all(
                    record.get("success") or record.get("skipped") for record in records
                ):
                    execution_results["total_failures"] += 1
                    _append_framework_result(
                        execution_results,
                        spec,
                        "PARTIAL",
                        "Successful scripts and unfinished skipped work",
                        output_dir,
                        records,
                    )
                    return
                success = all(record.get("success") for record in records)

            if success:
                execution_results["total_successes"] += 1
                _append_framework_result(
                    execution_results,
                    spec,
                    "SUCCESS",
                    spec.success_message,
                    output_dir,
                    records,
                )
                log_step_success(logger, spec.success_log)
            else:
                execution_results["total_failures"] += 1
                _append_framework_result(
                    execution_results,
                    spec,
                    "FAILED",
                    spec.failure_message,
                    output_dir,
                    records,
                )
                log_step_warning(logger, spec.failure_message)
    except Exception as e:
        execution_results["total_failures"] += 1
        _append_framework_result(execution_results, spec, "ERROR", str(e), output_dir)
        log_step_warning(logger, f"{spec.warning_log_prefix}: {e}")


def _execute_configured_frameworks(
    target_dir: Path,
    framework_dirs: dict[str, Path],
    execution_results: dict[str, Any],
    logger: logging.Logger,
    recursive: bool,
    verbose: bool,
    timeout: Optional[int] = None,
) -> None:
    """Probe only frameworks with scripts; preserve controlled spec seams."""
    from gnn.execute.executor_specs import _runner_state

    candidates = _framework_candidates(target_dir)
    for spec in _framework_specs(resolve_availability=False):
        required_scripts = candidates.get(spec.framework_dir_key, ())
        if spec.readiness_pending:
            if not required_scripts:
                _append_framework_result(
                    execution_results,
                    spec,
                    "SKIPPED",
                    "No rendered scripts for this framework",
                    framework_dirs[spec.framework_dir_key],
                    records=[],
                    evidence={"required_work": False},
                )
                continue
            from gnn.execute.preconditions import execution_precondition

            precondition = execution_precondition(timeout, None)
            if precondition is not None:
                _record_precondition_failure(
                    spec,
                    framework_dirs[spec.framework_dir_key],
                    execution_results,
                    precondition,
                    required_scripts,
                )
                continue
            if spec.framework_dir_key in {"stan", "bnlearn"}:
                # These maintained list runners own per-script readiness.
                # In particular R-only bnlearn must not require Python bnlearn.
                spec = replace(spec, available=True, readiness_pending=False)
            else:
                state = _runner_state(spec.framework_dir_key)
                spec = replace(
                    spec,
                    available=state.available,
                    runner=state.runner,
                    diagnosis=state.diagnosis,
                    readiness_pending=False,
                )
        _execute_framework_spec(
            spec,
            target_dir,
            framework_dirs,
            execution_results,
            logger,
            recursive,
            verbose,
            timeout,
            required_scripts=required_scripts,
        )


def _framework_candidates(target_dir: Path) -> dict[str, tuple[Path, ...]]:
    """Discover executable framework candidates once, without readiness probes."""
    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    candidates: dict[str, list[Path]] = {}
    names = set(FRAMEWORK_DIR_NAMES)
    for path in target_dir.rglob("*"):
        if context is not None:
            context.raise_if_expired()
        if not path.is_file():
            continue
        parents = (target_dir.name, *path.relative_to(target_dir).parts[:-1])
        key = next((part for part in reversed(parents) if part in names), None)
        if key is None:
            continue
        suffix = path.suffix.lower()
        if key == "stan":
            executable = path.name.endswith("_stan.py")
        elif key in {"rxinfer", "activeinference_jl"}:
            executable = suffix == ".jl"
        elif key == "lean":
            executable = suffix in {".lean", ".md"}
        elif key == "bnlearn":
            executable = suffix in {".py", ".r"}
        else:
            executable = suffix == ".py"
        if executable:
            candidates.setdefault(key, []).append(path)
    return {key: tuple(sorted(paths)) for key, paths in candidates.items()}


def _record_precondition_failure(
    spec: ExecutorFrameworkSpec,
    output_dir: Path,
    execution_results: dict[str, Any],
    precondition: dict[str, Any],
    scripts: tuple[Path, ...] | None,
) -> None:
    """Keep invalid or exhausted required work visible before readiness probes."""
    execution_results["total_failures"] += 1
    _append_framework_result(
        execution_results,
        spec,
        "FAILED",
        precondition["error"],
        output_dir,
        records=[{"script": str(script), **precondition} for script in scripts]
        if scripts
        else None,
        evidence={
            **{key: value for key, value in precondition.items() if key != "status"},
            "required_work": True,
        },
    )


def _write_framework_report_section(
    file_obj: Any,
    title: str,
    executions: list[dict[str, Any]],
    default_script: str,
    include_type: bool = False,
) -> None:
    """Write one framework subsection to the markdown execution report."""
    if not executions:
        return

    file_obj.write(f"## {title}\n\n")
    for exec_info in executions:
        status_icon = "✅" if exec_info.get("status") == "SUCCESS" else "❌"
        type_text = f" ({exec_info.get('type', 'analysis')})" if include_type else ""
        file_obj.write(
            f"- {status_icon} **{exec_info.get('script', default_script)}**{type_text}: {exec_info.get('status', 'Unknown')}\n"
        )
        file_obj.write(f"  - {exec_info.get('message', 'No message')}\n")
        file_obj.write(f"  - Output Directory: {exec_info.get('output_dir', 'N/A')}\n")
        if "scripts_processed" in exec_info:
            file_obj.write(f"  - Scripts processed: {exec_info['scripts_processed']}\n")
    file_obj.write("\n")


def _write_execution_report(
    report_file: Path, execution_results: dict[str, Any]
) -> None:
    """Write the enhanced markdown execution report."""
    with open(report_file, "w") as f:
        f.write("# Enhanced Execution Results Report\n\n")
        f.write(f"**Generated:** {execution_results['timestamp']}\n")
        f.write(f"**Target Directory:** {execution_results['target_directory']}\n")
        f.write(f"**Total Successes:** {execution_results['total_successes']}\n")
        f.write(f"**Total Failures:** {execution_results['total_failures']}\n\n")

        f.write("## Framework-Specific Output Directories\n\n")
        for framework, framework_dir in execution_results[
            "framework_execution_dirs"
        ].items():
            f.write(f"- **{framework.upper()}**: {framework_dir}\n")
        f.write("\n")

        if execution_results["dependency_issues"]:
            f.write("## Dependency Issues\n\n")
            for issue in execution_results["dependency_issues"]:
                f.write(f"- ⚠️ {issue}\n")
            f.write("\n")

        if execution_results["syntax_errors"]:
            f.write("## Syntax Errors\n\n")
            for error in execution_results["syntax_errors"]:
                f.write(f"- ❌ {error}\n")
            f.write("\n")

        _write_framework_report_section(
            f,
            "PyMDP Executions",
            execution_results["pymdp_executions"],
            "PyMDP Scripts",
        )
        _write_framework_report_section(
            f,
            "RxInfer Executions",
            execution_results["rxinfer_executions"],
            "RxInfer Scripts",
        )
        _write_framework_report_section(
            f,
            "DisCoPy Analyses",
            execution_results["discopy_executions"],
            "DisCoPy Analysis",
            include_type=True,
        )
        _write_framework_report_section(
            f,
            "ActiveInference.jl Analyses",
            execution_results["activeinference_executions"],
            "ActiveInference.jl Scripts",
        )
        _write_framework_report_section(
            f, "JAX Executions", execution_results["jax_executions"], "JAX Scripts"
        )
        _write_framework_report_section(
            f,
            "NumPyro Executions",
            execution_results["numpyro_executions"],
            "NumPyro Scripts",
        )
        _write_framework_report_section(
            f,
            "PyTorch Executions",
            execution_results["pytorch_executions"],
            "PyTorch Scripts",
        )
        _write_framework_report_section(
            f,
            "ngc-learn Executions",
            execution_results["ngclearn_executions"],
            "ngc-learn Scripts",
        )
        _write_framework_report_section(
            f,
            "Lean verification",
            execution_results["lean_executions"],
            "Lean Documents",
        )
        _write_framework_report_section(
            f,
            "Stan Executions",
            execution_results["stan_executions"],
            "Stan Drivers",
        )
        _write_framework_report_section(
            f,
            "bnlearn Executions",
            execution_results["bnlearn_executions"],
            "bnlearn Scripts",
        )

        f.write("## Recommendations\n\n")
        if execution_results["dependency_issues"]:
            f.write("### Install Missing Dependencies\n\n")
            for issue in execution_results["dependency_issues"]:
                if "Python dependencies" in issue:
                    f.write(
                        "- Install missing Python packages: `uv pip install <package_name>` or add to pyproject and run `uv sync`\n"
                    )
                elif "Julia" in issue:
                    f.write("- Install Julia from https://julialang.org/downloads/\n")
            f.write("\n")

        if execution_results["syntax_errors"]:
            f.write("### Fix Syntax Errors\n\n")
            f.write("- Review and fix syntax errors in rendered scripts\n")
            f.write("- Check for stray characters or malformed code\n")
            f.write(
                "- Re-run the rendering step (src/gnn/11_render.py) to regenerate scripts\n\n"
            )


def _write_execution_artifacts(
    execution_output_dir: Path, execution_results: dict[str, Any]
) -> None:
    """Write JSON and markdown execution summaries."""
    summaries_dir = execution_output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    with open(summaries_dir / "execution_summary.json", "w") as f:
        json.dump(execution_results, f, indent=2)
    _write_execution_report(summaries_dir / "execution_report.md", execution_results)


def _count_framework_execution_records(execution_results: dict[str, Any]) -> int:
    """Count framework result records across all supported backends."""
    return sum(
        sum(
            record.get("required_work") is not False
            for record in execution_results[spec.result_key]
        )
        for spec in _framework_specs(resolve_availability=False)
    )


def _log_execution_outcome(
    execution_results: dict[str, Any], logger: logging.Logger
) -> bool:
    """Log aggregate execution outcome and return success status."""
    successful = bool(
        execution_results["total_failures"] == 0
        and execution_results.get("required_unfinished", 0) == 0
    )
    total_executions = _count_framework_execution_records(execution_results)
    if total_executions == 0:
        log_step_warning(
            logger, "No simulator scripts or outputs found to execute/analyze"
        )
        return successful

    success_rate = execution_results["total_successes"] / total_executions * 100
    (log_step_success if successful else log_step_warning)(
        logger,
        f"Execution completed with framework-specific organization. Success rate: {success_rate:.1f}% ({execution_results['total_successes']}/{total_executions})",
    )

    if execution_results["dependency_issues"]:
        logger.warning(
            f"⚠️ Dependency issues found: {len(execution_results['dependency_issues'])}"
        )
    if execution_results["syntax_errors"]:
        logger.warning(
            f"⚠️ Syntax errors found: {len(execution_results['syntax_errors'])}"
        )

    return successful


def execute_rendered_simulators(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Execute rendered simulator scripts with enhanced error handling and dependency checking.
    Framework outputs are organized in separate subdirectories.

    Args:
        target_dir: Directory containing rendered simulator scripts
        output_dir: Output directory for results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional execution options

    Returns:
        True if execution succeeded, False otherwise
    """
    timeout: Optional[int] = kwargs.pop("timeout", None)
    log_step_start(
        logger,
        "Executing rendered simulator scripts with framework-specific organization",
    )

    execution_output_dir = get_output_dir_for_script("12_execute.py", output_dir)
    execution_output_dir.mkdir(parents=True, exist_ok=True)
    framework_dirs = _create_framework_dirs(execution_output_dir, logger)

    try:
        execution_results = _initialize_execution_results(target_dir, framework_dirs)
        logger.info("🔍 Pre-execution validation and dependency checking...")
        # Each selected runner owns a bounded diagnosis; avoid unrelated eager imports.

        _execute_configured_frameworks(
            target_dir,
            framework_dirs,
            execution_results,
            logger,
            recursive,
            verbose,
            timeout,
        )
        _write_execution_artifacts(execution_output_dir, execution_results)
        return _log_execution_outcome(execution_results, logger)

    except Exception as e:
        log_step_error(logger, f"Execution failed: {e}")
        return False
