#!/usr/bin/env python3
"""Programmatic pipeline execution adapters.

The main orchestration engine lives in :mod:`main`. This module provides a
small importable surface that delegates actual work to the numbered step
scripts through ``main.py``.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, List, Optional

from ._version import __version__ as _PACKAGE_VERSION
from .admission import validate_boolean, validate_steps
from .config import DEFAULT_OUTPUT_DIR, DEFAULT_TARGET_DIR, STEP_METADATA

_STEP_SCRIPT_BY_NUM = {int(key.split("_", 1)[0]): f"{key}.py" for key in STEP_METADATA}
_STEP_NUM_BY_ALIAS: dict[str, int] = {}
for _num, _script in _STEP_SCRIPT_BY_NUM.items():
    _stem = Path(_script).stem
    _suffix = _stem.split("_", 1)[1] if "_" in _stem else _stem
    _STEP_NUM_BY_ALIAS[str(_num)] = _num
    _STEP_NUM_BY_ALIAS[_stem] = _num
    _STEP_NUM_BY_ALIAS[_script] = _num
    _STEP_NUM_BY_ALIAS[_suffix] = _num
_STEP_NUM_BY_ALIAS["advanced_visualization"] = 9

# Step outcomes that count as success across the pipeline summary contract.
_SUCCESS_STATUSES = frozenset({"SUCCESS", "SUCCESS_WITH_WARNINGS", "SKIPPED"})


def _ensure_src_on_path() -> None:
    """Handle ensure src on path for internal callers."""
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _main_module() -> Any:
    """Handle main module for internal callers."""
    _ensure_src_on_path()
    from gnn import main as main_module

    return main_module


@contextmanager
def _owned_invocation_identity(main: Any) -> Iterator[str]:
    """Bind a known UUID through main using its existing environment owner."""
    with main._run_environment_lock:
        inherited = os.environ.get("GNN_RUN_ID")
        expected = inherited or uuid.uuid4().hex
        os.environ["GNN_RUN_ID"] = expected
        try:
            yield expected
        finally:
            if inherited is None:
                os.environ.pop("GNN_RUN_ID", None)
            else:
                os.environ["GNN_RUN_ID"] = inherited


def resolve_step_numbers(
    steps: Any,
    pipeline_data: dict | None = None,
) -> list[int]:
    """Normalize user-provided step identifiers to registered step numbers.

    Accepts: ``None``/""/"all"/"pipeline" (→ every registered step), a single
    step number or name ("11", "11_render", "11_render.py"), comma-separated
    lists ("3,5" — mirrors the ``--only-steps`` CLI form), or any iterable
    mixing those forms. ``pipeline_data`` may carry a fallback ``steps`` /
    ``only_steps`` entry when ``steps`` is ``None``. Unknown or empty tokens
    are dropped; duplicates collapse; output is sorted and de-duplicated.

    Raises no exceptions for unrecognized names — callers decide whether an
    empty result is an error (see :func:`run_pipeline`).
    """
    return _coerce_steps(steps, pipeline_data)


def _coerce_steps(steps: Any, pipeline_data: dict | None = None) -> list[int]:
    """Internal wrapper for :func:`resolve_step_numbers`."""
    if steps is None and pipeline_data:
        steps = pipeline_data.get("steps") or pipeline_data.get("only_steps")

    if steps in (None, "", "pipeline", "all"):
        return list(_STEP_SCRIPT_BY_NUM)
    if isinstance(steps, (str, int)):
        steps = [steps]

    out: list[int] = []
    for step in steps:
        tokens = str(step).split(",") if isinstance(step, str) else [str(step)]
        for token in tokens:
            key = token.strip()
            if not key:
                continue
            key = key[:-3] if key.endswith(".py") else key
            num = _STEP_NUM_BY_ALIAS.get(key)
            if num is None and key.isdigit():
                num = int(key)
            if num is not None and num in _STEP_SCRIPT_BY_NUM:
                out.append(num)
    return sorted(dict.fromkeys(out))


def _script_for_step(step_name: str) -> str | None:
    """Handle script for step for internal callers."""
    steps = validate_steps(
        step_name, field_name="step_name", aliases=True, allow_empty=False
    )
    if steps is None or len(steps) != 1:
        raise ValueError("step_name must identify exactly one registered step")
    return _STEP_SCRIPT_BY_NUM.get(steps[0])


def _path_from_sources(
    key: str,
    *,
    pipeline_data: dict | None,
    step_config: dict | None = None,
    fallback: str,
) -> Path:
    """Handle path from sources for internal callers."""
    for source in (step_config or {}, pipeline_data or {}):
        value = source.get(key)
        if value is not None:
            return Path(value)
    if key == "target_dir" and pipeline_data:
        for alias in ("input_dir", "temp_dir"):
            value = pipeline_data.get(alias)
            if value is not None:
                return Path(value)
    return Path(fallback)


@dataclass
class StepExecutionResult:
    """Result of a pipeline step execution."""

    step_name: str
    success: bool
    duration: float
    output: Optional[str] = None
    error: Optional[str] = None
    warnings: Optional[List[str]] = None
    remediation: Optional[str] = None
    status: Optional[str] = None
    run_id: Optional[str] = None
    artifacts: Optional[List[dict[str, Any]]] = None

    def __post_init__(self) -> Any:
        """Normalize fields after dataclass initialization."""
        if self.warnings is None:
            self.warnings = []


def run_pipeline(
    pipeline_data: dict | None = None,
    *,
    target_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    steps: List[str] | str | None = None,
    verbose: bool | None = None,
    strict: bool | None = None,
    parallel: bool | None = None,
    consolidated_steps: bool | None = None,
) -> dict:
    """Execute pipeline steps through ``main.py`` and return a compact summary."""
    start = datetime.now()
    if pipeline_data is None:
        pipeline_data = {}
    if not isinstance(pipeline_data, dict):
        raise ValueError("pipeline_data must be a mapping")
    resolved_target = (
        Path(target_dir)
        if target_dir is not None
        else _path_from_sources(
            "target_dir", pipeline_data=pipeline_data, fallback=DEFAULT_TARGET_DIR
        )
    )
    resolved_output = (
        Path(output_dir)
        if output_dir is not None
        else _path_from_sources(
            "output_dir", pipeline_data=pipeline_data, fallback=DEFAULT_OUTPUT_DIR
        )
    )
    results: dict[str, Any] = {
        "success": False,
        "steps_executed": [],
        "errors": [],
        "warnings": [],
        "target_dir": str(resolved_target),
        "output_dir": str(resolved_output),
        "exit_code": None,
    }

    try:
        allowed = {
            "target_dir",
            "input_dir",
            "temp_dir",
            "output_dir",
            "steps",
            "only_steps",
            "skip_steps",
            "verbose",
            "strict",
            "parallel",
            "consolidated_steps",
        }
        unknown = set(pipeline_data) - allowed
        if unknown:
            raise ValueError(f"Unknown pipeline options: {sorted(unknown)}")
        requested = steps
        if requested is None:
            requested = pipeline_data.get("steps", pipeline_data.get("only_steps"))
        step_numbers = (
            list(_STEP_SCRIPT_BY_NUM)
            if requested is None or requested in ("all", "pipeline")
            else validate_steps(
                requested, field_name="steps", aliases=True, allow_empty=False
            )
        )
        skipped = validate_steps(
            pipeline_data.get("skip_steps"), field_name="skip_steps", aliases=True
        )
        if (
            requested is not None
            and requested not in ("all", "pipeline")
            and set(step_numbers or ()) & set(skipped or ())
        ):
            raise ValueError("Pipeline steps cannot be both requested and skipped")
        flags = {
            "verbose": verbose,
            "strict": strict,
            "parallel": parallel,
            "consolidated_steps": consolidated_steps,
        }
        for name, explicit in flags.items():
            flags[name] = validate_boolean(
                pipeline_data.get(name, False) if explicit is None else explicit,
                field_name=name,
            )

        main = _main_module()
        from gnn.utils.arguments.pipeline_arguments import PipelineArguments

        args = PipelineArguments(
            target_dir=resolved_target,
            output_dir=resolved_output,
            only_steps=",".join(str(step) for step in step_numbers or ()),
            verbose=bool(flags["verbose"]),
            strict=bool(flags["strict"]),
            parallel=bool(flags["parallel"]),
            consolidated_steps=bool(flags["consolidated_steps"]),
            skip_steps=",".join(str(step) for step in skipped) if skipped else None,
        )
        config_override: dict[str, Any] = {
            "pipeline": {"only_steps": args.only_steps, "skip_steps": skipped or []},
            "testing_matrix": {"enabled": False},
        }
        with _owned_invocation_identity(main) as expected_run_id:
            invocation_start_ns = time.time_ns()
            exit_code = int(
                main.main(override_args=args, override_config=config_override)
            )
        results["exit_code"] = exit_code
        results["success"] = exit_code == 0 or (exit_code == 2 and not flags["strict"])
        summary_file = (
            resolved_output / "00_pipeline_summary" / "pipeline_execution_summary.json"
        )
        if (
            summary_file.is_file()
            and summary_file.stat().st_mtime_ns >= invocation_start_ns
        ):
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            if (
                not isinstance(summary, dict)
                or summary.get("run_id") != expected_run_id
            ):
                raise ValueError(
                    "Pipeline summary does not belong to the current invocation"
                )
            results["summary_file"] = str(summary_file)
            results["overall_status"] = summary.get("overall_status")
            if summary.get("error"):
                results["errors"].append(str(summary["error"]))
            results["run_id"] = summary.get("run_id")
            results["run_hash"] = summary.get("run_hash")
            results["model_selection"] = summary.get("model_selection", [])
            results["artifact_inventory_contract"] = summary.get(
                "artifact_inventory_contract"
            )
            for step in summary.get("steps", []):
                results["steps_executed"].append(
                    {
                        "step_name": step.get("script_name"),
                        "success": step.get("status") in _SUCCESS_STATUSES,
                        "duration": step.get("duration_seconds", 0.0),
                        "output": step.get("stdout", ""),
                        "status": step.get("status"),
                        "error": step.get("error") or step.get("stderr", ""),
                        "warnings": step.get("dependency_warnings", []),
                        "artifacts": step.get("artifacts", []),
                    }
                )
        else:
            results["success"] = False
            results["errors"].append(
                "No current-invocation pipeline summary is available"
            )

    except Exception as e:
        results["success"] = False
        results["errors"].append(f"Pipeline execution failed: {e}")

    results["duration"] = (datetime.now() - start).total_seconds()
    return results


def get_pipeline_status() -> dict:
    """Get the current pipeline status."""
    return {
        "status": "ready",
        "timestamp": datetime.now().isoformat(),
        "steps_available": len(_STEP_SCRIPT_BY_NUM),
        "steps_completed": 0,
    }


def validate_pipeline_config(config: dict) -> bool:
    """Validate pipeline configuration."""
    try:
        required_keys: list[str] = ["steps", "output_dir"]
        return all(key in config for key in required_keys)
    except (TypeError, KeyError):
        return False


def get_pipeline_info() -> dict:
    """Get pipeline information."""
    return {
        "name": "GNN Pipeline",
        "version": _PACKAGE_VERSION,
        "description": "GeneralizedNotationNotation processing pipeline",
        "steps": sorted(_STEP_SCRIPT_BY_NUM),
    }


def create_pipeline_config() -> dict:
    """Create a default pipeline configuration."""
    return {
        "project_name": "GeneralizedNotationNotation",
        "version": _PACKAGE_VERSION,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "steps": {},
    }


def execute_pipeline_step(
    step_name: str, step_config: dict, pipeline_data: dict
) -> StepExecutionResult:
    """Execute a registered step and its prerequisites in one owned invocation."""
    try:
        script_name = _script_for_step(step_name)
    except ValueError as error:
        return StepExecutionResult(step_name, False, 0.0, error=str(error))
    if step_config.get("script_path"):
        candidate = Path(step_config["script_path"])
        if not candidate.exists():
            return StepExecutionResult(
                step_name=step_name,
                success=False,
                duration=0.0,
                error=f"Step script not found: {candidate}",
            )
        if (
            candidate.resolve()
            != (Path(__file__).resolve().parents[1] / candidate.name).resolve()
        ):
            return StepExecutionResult(
                step_name=step_name,
                success=False,
                duration=0.0,
                error="script_path must identify a packaged registered step",
            )
        if candidate.name != script_name:
            return StepExecutionResult(
                step_name=step_name,
                success=False,
                duration=0.0,
                error="script_path must match the requested registered step",
            )
    if not script_name:
        return StepExecutionResult(
            step_name=step_name,
            success=False,
            duration=0.0,
            error=f"Unknown pipeline step: {step_name}",
        )

    unknown = set(step_config) - {
        "target_dir",
        "output_dir",
        "verbose",
        "strict",
        "parallel",
        "consolidated_steps",
        "script_path",
    }
    if unknown:
        return StepExecutionResult(
            step_name=script_name,
            success=False,
            duration=0.0,
            error=f"Unknown step options: {sorted(unknown)}",
        )
    data = {
        **pipeline_data,
        **{key: value for key, value in step_config.items() if key != "script_path"},
    }
    return execute_pipeline_steps([script_name], data)[0]


def execute_pipeline_steps(
    steps: List[str], pipeline_data: dict
) -> List[StepExecutionResult]:
    """Execute the complete requested plan once under one frozen run context."""
    # Public annotations do not validate values supplied by dynamic callers.
    supplied_steps: Any = steps
    supplied_data: Any = pipeline_data
    if not isinstance(supplied_steps, (list, tuple)):
        return [
            StepExecutionResult(
                "steps", False, 0.0, error="steps must be a list of registered steps"
            )
        ]
    if not steps:
        return []
    if not isinstance(supplied_data, dict):
        return [
            StepExecutionResult(
                str(step), False, 0.0, error="pipeline_data must be a mapping"
            )
            for step in steps
        ]
    try:
        numbers = validate_steps(
            steps, field_name="steps", aliases=True, allow_empty=False
        )
    except ValueError as error:
        return [
            StepExecutionResult(str(step), False, 0.0, error=str(error))
            for step in steps
        ]
    result = run_pipeline(pipeline_data, steps=steps)
    by_name = {record["step_name"]: record for record in result["steps_executed"]}
    outcomes = []
    for number in numbers or ():
        name = _STEP_SCRIPT_BY_NUM[number]
        record = by_name.get(name)
        if record is None:
            outcomes.append(
                StepExecutionResult(
                    name,
                    False,
                    0.0,
                    error="; ".join(result["errors"]) or "No current step receipt",
                    run_id=result.get("run_id"),
                )
            )
            continue
        outcomes.append(
            StepExecutionResult(
                name,
                bool(result["success"] and record["success"]),
                record["duration"],
                output=record["output"],
                error=(
                    record["error"]
                    or "; ".join(result["errors"])
                    or f"Owned pipeline ended {result.get('overall_status')} (exit {result['exit_code']})"
                )
                if not result["success"] or not record["success"]
                else None,
                warnings=record["warnings"],
                status=record["status"],
                run_id=result.get("run_id"),
                artifacts=record["artifacts"],
            )
        )
    return outcomes
