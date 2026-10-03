#!/usr/bin/env python3
"""
GNN Processing Pipeline

This script orchestrates the 25-step GNN processing pipeline (steps 0-24).
The pipeline transforms GNN specifications into executable simulations, visualizations,
and advanced representations including audio sonification.

Pipeline Steps:
0. Template initialization (0_template.py)
1. Environment setup (1_setup.py)
2. Test suite execution (2_tests.py)
3. GNN file processing (3_gnn.py)
4. Model registry (4_model_registry.py)
5. Type checking (5_type_checker.py)
6. Validation (6_validation.py)
7. Multi-format export (7_export.py)
8. Visualization (8_visualization.py)
9. Advanced visualization (9_advanced_viz.py)
10. Ontology processing (10_ontology.py)
11. Code rendering (11_render.py)
12. Execution (12_execute.py)
13. LLM processing (13_llm.py)
14. ML integration (14_ml_integration.py)
15. Audio processing (15_audio.py)
16. Analysis (16_analysis.py)
17. Integration (17_integration.py)
18. Security (18_security.py)
19. Research (19_research.py)
20. Website generation (20_website.py)
21. Model Context Protocol processing (21_mcp.py)
22. GUI (Interactive GNN Constructor) (22_gui.py)
23. Report generation (23_report.py)
24. Intelligent analysis (24_intelligent_analysis.py)

Usage:
    uv run --extra dev python src/gnn/main.py [options]

Examples:
    # Run full pipeline
    uv run --extra dev python src/gnn/main.py --target-dir input/gnn_files --verbose

    # Run specific steps only
    uv run --extra dev python src/gnn/main.py --only-steps "0,1,2,3" --verbose

    # Skip certain steps
    uv run --extra dev python src/gnn/main.py --skip-steps "15,16" --verbose

For complete usage information, see:
- AGENTS.md: Repository conventions and the command of record
- README.md: Project overview and quick start
- docs/pipeline/README.md: Detailed pipeline documentation
- src/gnn/README.md: Module and pipeline safety documentation
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional, Sequence

_module_logger = logging.getLogger(__name__)
# Environment scope is process-global; serialize programmatic top-level runs.
_run_environment_lock = RLock()

# Detect project root and ensure we're working from there
SCRIPT_DIR = Path(__file__).parent  # src/
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # project root (two levels up from src/gnn/)

# Script-entry side effects live inside the ``__main__`` guard below so that
# ``import gnn.main`` stays side-effect free. A direct repository script gets
# src/ on sys.path and anchors relative defaults to its checkout. Package
# module execution and installed scripts retain the caller's working directory
# so input/config.yaml and relative input/output paths belong to that caller.
if __name__ == "__main__":
    _src_dir = str(SCRIPT_DIR.parent)
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    if (
        not __package__
        and (PROJECT_ROOT / "pyproject.toml").is_file()
        and Path.cwd() != PROJECT_ROOT
    ):
        os.chdir(PROJECT_ROOT)
        logging.getLogger(__name__).info(
            f"Changed working directory to project root: {PROJECT_ROOT}"
        )

from dataclasses import dataclass, fields

from gnn.utils.arguments.arg_parsing import ArgumentParser
from gnn.utils.arguments.pipeline_arguments import PipelineArguments
from gnn.utils.arguments.pipeline_config_merge import apply_input_config_defaults

# Structured logging and visual progress tracking are maintained pipeline
# surfaces; import errors should fail loudly during startup.
from gnn.utils.logging_utils import (
    PipelineLogger,
    PipelineProgressTracker,
    log_step_start,
    reset_progress_tracker,
    rotate_logs,
    setup_step_logging,
)
from gnn.utils.observability.visual_logging import (
    VisualConfig,
    VisualLogger,
    create_visual_logger,
    print_pipeline_banner,
)
from gnn.utils.pipeline_orchestration.pipeline_step_dependencies import (
    resolve_step_dependencies,
)
from gnn.utils.pipeline_orchestration.pipeline_validator import (
    validate_pipeline_step_sequence,
)

STRUCTURED_LOGGING_AVAILABLE = True
PipelineStep = tuple[str, str]

# Derive PIPELINE_STEPS from the canonical step registry
from gnn.pipeline.run_session import RunSession  # noqa: E402
from gnn.pipeline.run_session_wiring import (  # noqa: E402
    close_run_session_guarded,  # noqa: F401 — preserved existing export
    mark_units_running_guarded,
    open_run_session_guarded,
    record_step_result_guarded,
)
from gnn.pipeline.step_registry import (
    PIPELINE_STEPS_TUPLE as PIPELINE_STEPS,  # noqa: E402
)
from gnn.pipeline.summary_wiring import (  # noqa: E402
    CRITICAL_SCRIPTS,  # noqa: F401
    _consolidated_step_selected,
    _fail_pipeline_startup,
    _finalize_pipeline_summary,
    _handle_pipeline_failure,
    _log_pipeline_step_start,
    _pipeline_exit_code,
    _pipeline_summary_path,
    _print_pipeline_completion,
    _read_input_config,
    _record_step_result,
    _save_minimal_pipeline_summary,
    _status_from_step_exit_code,  # noqa: F401
    _step_has_actionable_warning,  # noqa: F401
    _update_performance_summary,  # noqa: F401
    _write_final_pipeline_report,  # noqa: F401
    _write_performance_dashboard,  # noqa: F401
    _write_pipeline_summary_outputs,
    execute_pipeline_step,
    get_environment_info,
    parse_step_list,  # noqa: F401
    step_number_from_script_name,  # noqa: F401
    validate_pipeline_summary,  # noqa: F401
)


@dataclass(frozen=True)
class StepSelection:
    """Pure result of applying only/skip filters to the canonical step list.

    Attributes:
        selected: Ordered ``(script_name, description)`` pairs to execute.
        skipped: Step numbers excluded by the CLI/config skip lists.
        added_dependencies: Step numbers pulled in by dependency resolution
            that were not explicitly requested.
        requested_only: Raw requested step numbers from ``only_steps``.
        unknown_requested: Requested step numbers outside the canonical
            range (reported by callers, never executed).
    """

    selected: tuple[PipelineStep, ...]
    skipped: tuple[int, ...]
    added_dependencies: tuple[int, ...]
    requested_only: tuple[int, ...]
    unknown_requested: tuple[int, ...]


def parse_step_list_strict(step_input: Any) -> list[int]:
    """Parse step input like :func:`parse_step_list`, but reject invalid tokens.

    Args:
        step_input: Comma-separated string, list of step numbers, or None.

    Returns:
        Parsed step numbers.

    Raises:
        ValueError: If ``step_input`` contains tokens that are not step
            numbers (empty tokens are ignored, as in :func:`parse_step_list`).
    """
    if step_input is None:
        return []
    if isinstance(step_input, str):
        tokens = [token.strip() for token in step_input.split(",") if token.strip()]
        invalid = [token for token in tokens if not token.isdigit()]
        if invalid:
            raise ValueError(
                f"Invalid step number(s) {invalid!r} in step selection "
                f"{step_input!r}; expected comma-separated integers"
            )
        return [int(token) for token in tokens]
    if isinstance(step_input, (list, tuple)):
        numbers: list[int] = []
        invalid_items: list[Any] = []
        for item in step_input:
            if isinstance(item, bool):
                invalid_items.append(item)
            elif isinstance(item, int):
                numbers.append(item)
            elif isinstance(item, str) and item.strip().isdigit():
                numbers.append(int(item.strip()))
            else:
                invalid_items.append(item)
        if invalid_items:
            raise ValueError(
                f"Invalid step number(s) {invalid_items!r} in step selection"
            )
        return numbers
    raise ValueError(f"Unsupported step selection input: {step_input!r}")


def select_pipeline_steps(
    pipeline_steps: Sequence[PipelineStep],
    only_steps: Any = None,
    cli_skip_steps: Any = None,
    config_skip_steps: Any = None,
) -> StepSelection:
    """Pure only/skip step selection over an explicit step list.

    Dependency resolution pulls in required prerequisite steps for requested
    numbers. Out-of-range requested numbers are reported as
    ``unknown_requested`` and never executed; invalid tokens raise
    ``ValueError`` via :func:`parse_step_list_strict`.

    Args:
        pipeline_steps: Ordered ``(script_name, description)`` pairs; index
            equals the canonical step number.
        only_steps: CLI or config ``only_steps`` selection, or None.
        cli_skip_steps: CLI ``--skip-steps`` value, or None.
        config_skip_steps: Config ``pipeline.skip_steps`` value, or None.

    Returns:
        A frozen :class:`StepSelection`.
    """
    steps = list(pipeline_steps)
    selected = steps
    requested: list[int] = []
    added: list[int] = []
    unknown: list[int] = []

    if only_steps:
        requested = parse_step_list_strict(only_steps)
        valid = [n for n in requested if 0 <= n < len(steps)]
        unknown = [n for n in requested if n not in set(valid)]
        resolved = resolve_step_dependencies(valid)
        added = sorted(set(resolved) - set(valid))
        selected = [steps[i] for i in resolved if 0 <= i < len(steps)]

    skip_numbers = sorted(
        set(parse_step_list_strict(cli_skip_steps or []))
        | set(parse_step_list_strict(config_skip_steps or []))
    )
    if skip_numbers:
        original_indices = {script: i for i, (script, _) in enumerate(steps)}
        selected = [
            step
            for step in selected
            if original_indices.get(step[0], -1) not in skip_numbers
        ]

    return StepSelection(
        selected=tuple(selected),
        skipped=tuple(skip_numbers),
        added_dependencies=tuple(added),
        requested_only=tuple(requested),
        unknown_requested=tuple(unknown),
    )


def _build_main_args(
    override_args: Optional[PipelineArguments],
) -> tuple[PipelineArguments, Optional[argparse.Namespace]]:
    """Build pipeline arguments from an override object or CLI parsing."""
    if override_args is not None:
        return override_args, None

    parser = ArgumentParser.create_main_parser()
    parsed = parser.parse_args()
    field_names = {f.name for f in fields(PipelineArguments)}
    kwargs = {k: getattr(parsed, k) for k in field_names if hasattr(parsed, k)}
    args = PipelineArguments(**kwargs)

    if getattr(args, "skip_llm", False):
        existing = args.skip_steps or ""
        existing_nums = (
            [s.strip() for s in str(existing).split(",") if s.strip()]
            if existing
            else []
        )
        if "13" not in existing_nums:
            existing_nums.append("13")
        args.skip_steps = ",".join(existing_nums)

    return args, parsed


def _create_pipeline_visual_logger(
    args: PipelineArguments,
) -> tuple[VisualLogger, str]:
    """Create the visual logger and attach a short correlation ID."""
    visual_config = VisualConfig(
        enable_colors=True,
        enable_progress_bars=True,
        enable_emoji=True,
        enable_animation=True,
        show_timestamps=args.verbose,
        show_correlation_ids=True,
        compact_mode=False,
    )
    visual_logger = create_visual_logger("pipeline", visual_config)
    correlation_id = str(uuid.uuid4())[:8]
    visual_logger.set_correlation_id(correlation_id)
    return visual_logger, correlation_id


def _setup_pipeline_logger(args: PipelineArguments) -> logging.Logger:
    """Initialize structured pipeline logging for a fresh run."""
    if STRUCTURED_LOGGING_AVAILABLE:
        log_dir = args.output_dir / "00_pipeline_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        rotate_logs(log_dir)

        PipelineLogger._initialized = False
        if PipelineLogger._log_file_handler:
            logging.getLogger().removeHandler(PipelineLogger._log_file_handler)
            PipelineLogger._log_file_handler.close()
            PipelineLogger._log_file_handler = None

        PipelineLogger.initialize(
            log_dir=log_dir, enable_structured=True, log_format=args.log_format
        )
        PipelineLogger.enable_json_logging(log_dir)
        logger = setup_step_logging(
            "pipeline", args.verbose, enable_structured=True, log_format=args.log_format
        )
        reset_progress_tracker()
        return logger

    return setup_step_logging(
        "pipeline",
        args.verbose,
        log_format=args.log_format if hasattr(args, "log_format") else "human",
    )


def _log_backend_versions(logger: logging.Logger) -> None:
    """Log optional execution backend versions during verbose runs."""
    parts: list[str] = []
    for mod, label in (
        ("jax", "jax"),
        ("jaxlib", "jaxlib"),
        ("torch", "torch"),
        ("numpyro", "numpyro"),
        ("discopy", "discopy"),
    ):
        try:
            module = __import__(mod)
            parts.append(f"{label}={getattr(module, '__version__', '?')}")
        except ImportError:
            parts.append(f"{label}=missing")
    logger.info("Step 12 backends: %s", "; ".join(parts))


def _load_pipeline_config(
    override_config: Optional[Dict[str, Any]],
    logger: logging.Logger,
) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """Load `input/config.yaml` unless a test or caller supplies config directly."""
    if override_config is not None:
        return override_config, override_config.get("pipeline", {})

    full_config: dict[Any, Any] = {}
    config_pipeline_settings: dict[Any, Any] = {}
    try:
        full_config = _read_input_config(Path("input/config.yaml"))
        config_pipeline_settings = full_config.get("pipeline", {})
    except Exception as e:
        logger.warning(f"Could not load pipeline settings from input/config.yaml: {e}")

    return full_config, config_pipeline_settings


def _preflight_config_gate(
    override_config: Optional[Dict[str, Any]], logger: logging.Logger
) -> None:
    """Fail startup on config errors before any pipeline step executes.

    Runs the config-only preflight (``gnn.pipeline.preflight.validate_config``)
    against the same on-disk ``input/config.yaml`` that ``_load_pipeline_config``
    reads. Caller-supplied ``override_config`` bypasses the file contract and is
    not re-validated here. A missing config file is reported by preflight as a
    warning and stays tolerated.
    """
    if override_config is not None:
        return
    from gnn.pipeline.preflight import validate_config

    report = validate_config(Path("input/config.yaml"))
    # PreflightReport.checks_failed counts only severity=="error" issues
    # (warnings — e.g. a missing config file — never fail the gate).
    if report.checks_failed:
        errors = [issue for issue in report.issues if issue.severity == "error"]
        details = "; ".join(f"[{i.category}] {i.message}" for i in errors)
        raise ValueError(
            f"Preflight config validation failed with {report.checks_failed} "
            f"error(s): {details}"
        )
    for issue in report.issues:
        logger.warning(f"Preflight config: [{issue.category}] {issue.message}")


def resolve_steps_to_execute(
    args: PipelineArguments,
    config_pipeline_settings: dict[Any, Any],
    logger: logging.Logger,
) -> list[PipelineStep]:
    """Apply only/skip step filters and automatic dependency resolution.

    Invalid step tokens raise ``ValueError`` (fail fast instead of silently
    running nothing); unknown out-of-range step numbers are logged and
    dropped; a request that resolves to zero executable steps raises
    ``ValueError`` so startup fails loudly.
    """
    only_steps_val = args.only_steps or config_pipeline_settings.get("only_steps")
    selection = select_pipeline_steps(
        list(PIPELINE_STEPS),
        only_steps=only_steps_val,
        cli_skip_steps=args.skip_steps,
        config_skip_steps=config_pipeline_settings.get("skip_steps"),
    )

    if selection.added_dependencies:
        logger.info(
            f"Auto-including dependency steps: {list(selection.added_dependencies)}"
        )
    if selection.requested_only:
        logger.info(f"Executing steps: {[step[0] for step in selection.selected]}")
    if selection.unknown_requested:
        logger.warning(
            f"Ignoring unknown step number(s) in only_steps selection: "
            f"{list(selection.unknown_requested)}"
        )
    if selection.skipped:
        logger.info(
            f"Skipping steps: {[PIPELINE_STEPS[i][0] for i in selection.skipped if 0 <= i < len(PIPELINE_STEPS)]}"
        )

    if selection.requested_only and not selection.selected:
        raise ValueError(
            "Step selection resolved to no executable steps "
            f"(only_steps={only_steps_val!r}, "
            f"unknown={list(selection.unknown_requested)}, "
            f"skipped={list(selection.skipped)})"
        )

    return list(selection.selected)


def _initialize_pipeline_summary(
    args: PipelineArguments,
    steps_to_execute: list[PipelineStep],
    config_pipeline_settings: dict[Any, Any],
    *,
    input_config: Optional[Dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create the initial summary payload before step execution starts."""
    from gnn.pipeline.hasher import (
        RUN_HASH_SCHEMA,
        compute_run_hash_with_files,
        effective_run_config,
    )

    identity_config = effective_run_config(
        args.to_dict(),
        config_pipeline_settings,
        [step[0] for step in steps_to_execute],
        input_config,
    )
    run_hash, file_hashes = compute_run_hash_with_files(
        args.target_dir,
        config=identity_config,
    )

    return {
        "run_id": os.environ.get("GNN_RUN_ID"),
        "run_hash": run_hash,
        "run_hash_schema": RUN_HASH_SCHEMA,
        "identity_config": identity_config,
        "file_hashes": file_hashes,
        "start_time": datetime.now().isoformat(),
        "start_monotonic": time.monotonic(),
        "arguments": args.to_dict(),
        "steps": [],
        "end_time": None,
        "overall_status": "RUNNING",
        "total_duration_seconds": None,
        "environment_info": get_environment_info(),
        "performance_summary": {
            "peak_memory_mb": 0.0,
            "total_steps": len(steps_to_execute),
            "failed_steps": 0,
            "critical_failures": 0,
            "successful_steps": 0,
            "warnings": 0,
        },
    }


def _prepare_pipeline_context(
    override_args: Optional[PipelineArguments],
    override_config: Optional[Dict[str, Any]],
) -> tuple[
    PipelineArguments,
    dict[Any, Any],
    list[PipelineStep],
    dict[str, Any],
    VisualLogger,
    str,
    logging.Logger,
]:
    """Prepare args, config, logging, step list, and initial summary."""
    args, parsed = _build_main_args(override_args)
    try:
        visual_logger, correlation_id = _create_pipeline_visual_logger(args)
        logger = _setup_pipeline_logger(args)

        if args.verbose:
            _log_backend_versions(logger)

        full_config, config_pipeline_settings = _load_pipeline_config(
            override_config, logger
        )
        _preflight_config_gate(override_config, logger)
        apply_input_config_defaults(args, full_config, parsed)

        steps_to_execute = resolve_steps_to_execute(
            args, config_pipeline_settings, logger
        )
        pipeline_summary = _initialize_pipeline_summary(
            args, steps_to_execute, config_pipeline_settings, input_config=full_config
        )

        return (
            args,
            config_pipeline_settings,
            steps_to_execute,
            pipeline_summary,
            visual_logger,
            correlation_id,
            logger,
        )
    except Exception as error:
        # Args are resolved, so even a config/logging/selection failure has a receipt.
        failed_summary: dict[str, Any] = {
            "run_id": os.environ.get("GNN_RUN_ID"),
            "start_time": datetime.now().isoformat(),
            "start_monotonic": time.monotonic(),
            "arguments": args.to_dict(),
            "steps": [],
        }
        _save_minimal_pipeline_summary(
            _pipeline_summary_path(args.output_dir),
            failed_summary,
            error,
            _module_logger,
        )
        _handle_pipeline_failure(error, args, failed_summary, _module_logger)
        raise


def _start_pipeline_run(
    args: PipelineArguments,
    steps_to_execute: list[PipelineStep],
    pipeline_summary: dict[str, Any],
    visual_logger: VisualLogger,
    correlation_id: str,
    logger: logging.Logger,
) -> Optional[PipelineProgressTracker]:
    """Print start banners, wire progress tracking, and validate the step sequence."""
    print_pipeline_banner(
        "Generalized Notation Notation (GNN)",
        "https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation | Active Inference Institute",
    )
    print_pipeline_banner(
        "🚀 GNN Processing Pipeline",
        f"Starting execution with {len(steps_to_execute)} steps | Correlation ID: {correlation_id}",
    )
    visual_logger.print_progress(0, len(steps_to_execute), "Pipeline initialization")

    if STRUCTURED_LOGGING_AVAILABLE:
        PipelineLogger.log_structured(
            logger,
            logging.INFO,
            "🚀 Starting GNN Processing Pipeline",
            total_steps=len(steps_to_execute),
            target_dir=str(args.target_dir),
            output_dir=str(args.output_dir),
            event_type="pipeline_start",
        )
    else:
        log_step_start(logger, "Starting GNN Processing Pipeline")

    pipeline_summary["performance_summary"]["total_steps"] = len(steps_to_execute)

    progress_tracker = None
    if STRUCTURED_LOGGING_AVAILABLE:
        progress_tracker = PipelineProgressTracker(len(steps_to_execute))
        from gnn.utils.logging_utils import set_global_progress_tracker

        set_global_progress_tracker(progress_tracker)

    sequence_validation = validate_pipeline_step_sequence(steps_to_execute, logger)
    for warning in sequence_validation["warnings"]:
        logger.warning(f"Pipeline sequence: {warning}")
    for rec in sequence_validation["recommendations"]:
        logger.info(f"Recommendation: {rec}")

    return progress_tracker


def _write_preliminary_pipeline_summary(
    pipeline_summary: dict[str, Any],
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Write a current-run summary for report and intelligent-analysis steps."""
    try:
        prelim_summary = dict(pipeline_summary)
        prelim_end = datetime.now()
        prelim_summary["end_time"] = prelim_end.isoformat()
        if "start_monotonic" in prelim_summary:
            prelim_summary["total_duration_seconds"] = (
                time.monotonic() - prelim_summary["start_monotonic"]
            )
        else:
            prelim_start = datetime.fromisoformat(prelim_summary["start_time"])
            prelim_summary["total_duration_seconds"] = (
                prelim_end - prelim_start
            ).total_seconds()
        prelim_summary["preliminary"] = True
        prelim_summary["evidence_phase"] = "current_run_snapshot"

        prelim_path = output_dir / "00_pipeline_summary" / "current_summary.json"
        prelim_path.parent.mkdir(parents=True, exist_ok=True)
        from gnn.pipeline._io import atomic_write_text

        snapshot_path = (
            prelim_path.parent
            / "snapshots"
            / str(prelim_summary.get("run_id", "standalone"))
            / f"{len(prelim_summary.get('steps', [])):03d}-{uuid.uuid4().hex}.json"
        )
        prelim_summary["snapshot_path"] = str(snapshot_path.resolve())
        atomic_write_text(
            snapshot_path, json.dumps(prelim_summary, indent=4, default=str)
        )
        atomic_write_text(
            prelim_path, json.dumps(prelim_summary, indent=4, default=str)
        )
        atomic_write_text(
            prelim_path.with_name("preliminary_summary.json"),
            json.dumps(prelim_summary, indent=4, default=str),
        )
        logger.info(
            f"📝 Preliminary pipeline summary written ({len(prelim_summary.get('steps', []))} steps — "
            f"final summary path is 00_pipeline_summary/pipeline_execution_summary.json)"
        )
    except Exception as prelim_err:
        if os.environ.get("GNN_RUN_CONTEXT_FILE"):
            raise
        logger.warning(f"Could not write preliminary summary: {prelim_err}")


def _execute_selected_step(
    script_name: str,
    args: PipelineArguments,
    pipeline_summary: dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Dispatch one selected step to the consolidated executor or subprocess.

    Shared by the serial loop and the parallel tier's worker submission so
    ``--consolidated-steps`` applies to both execution tiers: whitelisted
    steps run in-process, everything else keeps the canonical subprocess
    path (mixed modes within one run are legal).
    """
    from copy import copy

    from gnn.pipeline.artifact_ownership import changed_artifacts, snapshot_files
    from gnn.pipeline.config import get_output_dir_for_script
    from gnn.pipeline.run_context import current_run_context
    from gnn.pipeline.step_registry import step_for_name

    context = current_run_context()
    step = step_for_name(script_name)
    if context is not None:
        remaining = context.remaining_seconds()
        if remaining is not None and remaining <= 0:
            return {
                "status": "FAILED",
                "exit_code": -1,
                "stop_reason": "total_timeout",
                "stdout": "",
                "stderr": "Pipeline total deadline expired",
                "artifacts": [],
            }
        context.verify_sources()
        if step is not None and step.execution_scope != "run":
            number = int(step.script_stem.split("_")[0])
            if not context.selected_models(number):
                return {
                    "status": "SKIPPED",
                    "exit_code": 0,
                    "skip_reason": "no_selected_models",
                    "stdout": "No selected model sources; no work dispatched",
                    "stderr": "",
                    "artifacts": [],
                }
            args = copy(args)
            args.target_dir = context.input_view(number)
        # The frozen view already contains the exact matrix selection. Dispatch once.
        pipeline_config = context.input_config
        pipeline_config["testing_matrix"] = {"enabled": False}
    else:
        pipeline_config = pipeline_summary.get("identity_config", {}).get(
            "input_config"
        )
    output_root = Path(args.output_dir)
    step_output = get_output_dir_for_script(script_name, output_root)
    before = snapshot_files(step_output)
    run_id = pipeline_summary.get("run_id")
    if _consolidated_step_selected(script_name, args, pipeline_summary):
        from gnn.pipeline.step_executor import execute_step_in_process

        result = execute_step_in_process(
            script_name,
            args,
            logger,
            run_id=run_id,
            pipeline_config=pipeline_config,
        )
    else:
        result = execute_pipeline_step(
            script_name, args, logger, run_id=run_id, pipeline_config=pipeline_config
        )
        if context is not None and getattr(args, "consolidated_steps", False):
            result["execution_fallback_reason"] = (
                "hard_deadline_requires_process_boundary"
            )
    try:
        result["artifacts"] = changed_artifacts(step_output, output_root, before)
    except TimeoutError:
        result.update(
            status="FAILED",
            exit_code=-1,
            stop_reason="total_timeout",
            artifacts=[],
            partial_output_dir=str(step_output),
        )
    if context is not None:
        result["run_id"] = context.run_id
        result["selected_model_ids"] = [
            model.model_id
            for model in context.selected_models(int(script_name.split("_")[0]))
        ]
    return result


def _execute_timed_selected_step(
    script_name: str,
    args: PipelineArguments,
    pipeline_summary: dict[str, Any],
    logger: logging.Logger,
) -> tuple[Dict[str, Any], datetime, datetime, float]:
    """Measure a parallel worker's execution before its future is collected."""
    started = time.monotonic()
    start_datetime = datetime.now()
    result = _execute_selected_step(script_name, args, pipeline_summary, logger)
    duration = time.monotonic() - started
    return result, start_datetime, datetime.now(), duration


def _execute_pipeline_iteration(
    step_index: int,
    script_name: str,
    description: str,
    steps_to_execute: list[PipelineStep],
    args: PipelineArguments,
    pipeline_summary: dict[str, Any],
    visual_logger: VisualLogger,
    progress_tracker: Optional[PipelineProgressTracker],
    logger: logging.Logger,
    session: Optional[RunSession] = None,
) -> Optional[RunSession]:
    """Execute and record one selected pipeline step.

    When a run session is supplied, the unit is marked RUNNING before the
    step executes and folded to its terminal status after the summary
    record is written (so the warning-upgrade is already applied). Returns
    the updated session, or None when no session was supplied.

    Note: ``session`` is a trailing positional-or-keyword param (deliberately
    NOT keyword-only) so variadic positional-forwarding wrappers around this
    function stay compatible when callers append the session positionally.
    """
    actual_step_number = step_index + 1
    step_start_time = time.monotonic()
    step_start_datetime = datetime.now()

    visual_logger.print_step_header(
        actual_step_number, description, len(steps_to_execute)
    )
    _log_pipeline_step_start(
        actual_step_number,
        script_name,
        description,
        len(steps_to_execute),
        progress_tracker,
        logger,
    )

    if script_name in ("20_website.py", "23_report.py", "24_intelligent_analysis.py"):
        _write_preliminary_pipeline_summary(pipeline_summary, args.output_dir, logger)

    session = mark_units_running_guarded(session, [script_name], args, logger)

    step_result: dict[str, Any] = _execute_selected_step(
        script_name, args, pipeline_summary, logger
    )
    step_duration = time.monotonic() - step_start_time
    step_end_datetime = datetime.now()

    _record_step_result(
        step_result,
        actual_step_number,
        script_name,
        description,
        step_start_datetime,
        step_end_datetime,
        step_duration,
        pipeline_summary,
        len(steps_to_execute),
        progress_tracker,
        logger,
    )

    session = record_step_result_guarded(
        session, script_name, step_result, args, logger
    )
    return session


def main(
    override_args: Optional[PipelineArguments] = None,
    override_config: Optional[Dict[str, Any]] = None,
) -> int:
    """Run once with an invocation ID, restoring the caller's environment on exit.

    An incoming API ID is authoritative. Fresh calls receive distinct UUIDs;
    the separately computed run_hash continues to identify stable input/config.
    """
    with _run_environment_lock:
        incoming_run_id = os.environ.get("GNN_RUN_ID")
        incoming_context = os.environ.get("GNN_RUN_CONTEXT_FILE")
        os.environ["GNN_RUN_ID"] = incoming_run_id or uuid.uuid4().hex
        try:
            from gnn.pipeline.output_lease import OutputLease, OutputLeaseError

            probe_args, _ = _build_main_args(override_args)
            try:
                with OutputLease(Path(probe_args.output_dir), os.environ["GNN_RUN_ID"]):
                    return _run_pipeline(override_args, override_config)
            except OutputLeaseError as error:
                # A previous invocation may still own an open file handler to a
                # now-rejected hardlinked log. Refusal must not write through it.
                print(f"Pipeline ownership failure: {error}", file=sys.stderr)
                return 1
        finally:
            if incoming_run_id is None:
                os.environ.pop("GNN_RUN_ID", None)
            else:
                os.environ["GNN_RUN_ID"] = incoming_run_id
            if incoming_context is None:
                os.environ.pop("GNN_RUN_CONTEXT_FILE", None)
            else:
                os.environ["GNN_RUN_CONTEXT_FILE"] = incoming_context


def _run_pipeline(
    override_args: Optional[PipelineArguments],
    override_config: Optional[Dict[str, Any]],
) -> int:
    """Execute one invocation inside main's environment scope."""
    try:
        (
            args,
            config_pipeline_settings,
            steps_to_execute,
            pipeline_summary,
            visual_logger,
            correlation_id,
            logger,
        ) = _prepare_pipeline_context(override_args, override_config)
    except Exception as e:
        return _fail_pipeline_startup(e)

    from gnn.pipeline.run_context import CONTEXT_ENV, build_run_context

    try:
        context = build_run_context(
            args.target_dir,
            args.output_dir,
            str(pipeline_summary["run_id"]),
            [int(step[0].split("_")[0]) for step in steps_to_execute],
            pipeline_summary["identity_config"]["input_config"],
            frameworks=str(getattr(args, "frameworks", "all")).split(","),
            recursive=bool(getattr(args, "recursive", True)),
        )
        context_path = args.output_dir / "00_pipeline_summary" / "run_context.json"
        context.write(context_path)
        os.environ[CONTEXT_ENV] = str(context_path.resolve())
        pipeline_summary["model_selection"] = [
            model.__dict__ for model in context.models if model.steps
        ]
        pipeline_summary["source_inventory"] = [
            model.__dict__ for model in context.models
        ]
        pipeline_summary["selection_exclusions"] = list(context.exclusions)
        pipeline_summary["deadline_monotonic"] = context.deadline_monotonic
        pipeline_summary["artifact_inventory_contract"] = "current-run-v1"
        pipeline_summary["planned_steps"] = [step[0] for step in steps_to_execute]
        _write_preliminary_pipeline_summary(pipeline_summary, args.output_dir, logger)
    except Exception as error:
        return _handle_pipeline_failure(error, args, pipeline_summary, logger)

    if getattr(args, "autonomous", False):
        from gnn.pipeline.autonomous import run_autonomous_proposal_loop

        report = run_autonomous_proposal_loop(args.target_dir, args.output_dir)
        logger.info(
            "Autonomous proposal loop wrote %d candidate(s) under %s/autonomous",
            report.get("candidate_count", 0),
            args.output_dir,
        )
        return 0

    run_session: Optional[RunSession] = None
    progress_tracker: Optional[PipelineProgressTracker] = None
    try:
        run_session = open_run_session_guarded(
            args, steps_to_execute, pipeline_summary, logger
        )
        progress_tracker = _start_pipeline_run(
            args,
            steps_to_execute,
            pipeline_summary,
            visual_logger,
            correlation_id,
            logger,
        )

        if getattr(args, "parallel", False):
            from concurrent.futures import ThreadPoolExecutor

            from gnn.pipeline.dag import resolve_execution_order
            from gnn.utils.pipeline_orchestration.pipeline_step_dependencies import (
                PIPELINE_OPTIONAL_PRODUCERS,
                PIPELINE_STEP_DEPENDENCIES,
            )

            deps_dict = {
                step_num: sorted(
                    set(deps) | set(PIPELINE_OPTIONAL_PRODUCERS.get(step_num, ()))
                )
                for step_num, deps in PIPELINE_STEP_DEPENDENCIES.items()
            }
            exec_indices = {
                int(s[0].split("_")[0]): s for s in steps_to_execute if "_" in s[0]
            }
            tiers = resolve_execution_order(
                deps_dict,
                total_steps=25,
                skip_steps=set(range(25)) - set(exec_indices.keys()),
            )

            current_step_counter = 0
            for tier_idx, tier in enumerate(tiers):
                tier_steps = [exec_indices[n] for n in tier if n in exec_indices]
                if not tier_steps:
                    continue
                if len(tier_steps) == 1:
                    script_name, description = tier_steps[0]
                    updated_session = _execute_pipeline_iteration(
                        current_step_counter,
                        script_name,
                        description,
                        steps_to_execute,
                        args,
                        pipeline_summary,
                        visual_logger,
                        progress_tracker,
                        logger,
                        run_session,
                    )
                    if updated_session is not None:
                        run_session = updated_session
                    current_step_counter += 1
                else:
                    max_cpu = os.cpu_count() or 4
                    # Calculate estimated tier memory requirement to prevent overcommit
                    tier_memory_est = sum(
                        (
                            500
                            if "14_ml_integration" in s[0]
                            else 300
                            if "9_advanced_viz" in s[0]
                            else 200
                            if "8_visualization" in s[0]
                            else 150
                            if "2_tests" in s[0]
                            else 100
                            if "13_llm" in s[0]
                            else 50
                        )
                        for s in tier_steps
                    )
                    # Dynamically throttle worker concurrency if aggregate memory footprint is high
                    worker_cap = max_cpu
                    if tier_memory_est > 800:
                        worker_cap = min(worker_cap, 4)
                    dynamic_workers = max(1, min(len(tier_steps), worker_cap, 8))
                    logger.info(
                        "⚡ Running Tier %d in parallel (%d steps, %d workers, ~%dMB est memory): %s",
                        tier_idx,
                        len(tier_steps),
                        dynamic_workers,
                        tier_memory_est,
                        [s[0] for s in tier_steps],
                    )
                    with ThreadPoolExecutor(max_workers=dynamic_workers) as pool:
                        _write_preliminary_pipeline_summary(
                            pipeline_summary, args.output_dir, logger
                        )
                        futures = []
                        for s_idx_offset, (script_name, description) in enumerate(
                            tier_steps
                        ):
                            f = pool.submit(
                                _execute_timed_selected_step,
                                script_name,
                                args,
                                pipeline_summary,
                                logger,
                            )
                            futures.append(
                                (
                                    current_step_counter + s_idx_offset,
                                    script_name,
                                    description,
                                    f,
                                )
                            )
                        current_step_counter += len(tier_steps)

                        run_session = mark_units_running_guarded(
                            run_session, [s[0] for s in tier_steps], args, logger
                        )

                        for step_num, script_name, description, future in futures:
                            (
                                step_result,
                                step_start_datetime,
                                step_end_datetime,
                                step_duration,
                            ) = future.result()

                            _record_step_result(
                                step_result,
                                step_num + 1,
                                script_name,
                                description,
                                step_start_datetime,
                                step_end_datetime,
                                step_duration,
                                pipeline_summary,
                                len(steps_to_execute),
                                progress_tracker,
                                logger,
                            )

                            run_session = record_step_result_guarded(
                                run_session, script_name, step_result, args, logger
                            )
        else:
            for step_index, (script_name, description) in enumerate(steps_to_execute):
                updated_session = _execute_pipeline_iteration(
                    step_index,
                    script_name,
                    description,
                    steps_to_execute,
                    args,
                    pipeline_summary,
                    visual_logger,
                    progress_tracker,
                    logger,
                    run_session,
                )
                if updated_session is not None:
                    run_session = updated_session

        context.verify_sources()
        from gnn.pipeline.artifact_ownership import verify_owned_artifacts

        verify_owned_artifacts(args.output_dir, pipeline_summary)
        pipeline_summary["evidence_integrity"] = {
            "status": "verified",
            "source_count": len(context.models),
        }
        _finalize_pipeline_summary(pipeline_summary)
        from gnn.pipeline.finalization import finalize_run_evidence

        finalize_run_evidence(args.output_dir, pipeline_summary, run_session)
        _write_pipeline_summary_outputs(
            args, config_pipeline_settings, pipeline_summary, logger
        )
        context.raise_if_expired()
        _print_pipeline_completion(pipeline_summary, progress_tracker, logger)
        return _pipeline_exit_code(pipeline_summary["overall_status"])

    except Exception as e:
        return _handle_pipeline_failure(e, args, pipeline_summary, logger)


if __name__ == "__main__":
    raise SystemExit(main())
