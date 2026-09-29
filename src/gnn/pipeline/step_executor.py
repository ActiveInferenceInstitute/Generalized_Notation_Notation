#!/usr/bin/env python3
"""Consolidated in-process step execution (S2-11 / V4-STAGE, first slice).

The 25-step numbered-script contract stays the pipeline's CLI surface:
``main.py`` still launches one subprocess per numbered script by default.
This module adds the opt-in counterpart selected by ``--consolidated-steps``:
steps whitelisted in ``step_registry.CONSOLIDATED_IN_PROCESS_STEMS`` run
inside the main process. Resolution stays registry-driven — the executor
imports ``gnn.<stem>`` and calls ``StepInfo.module_function``, exactly the
surface asserted by ``tests/pipeline/test_step_registry.py`` — so the
numbered scripts remain the single source of the step contract.

Each call mirrors the argument contract of
``gnn.utils.pipeline_orchestration.pipeline_template.create_standardized_pipeline_script``: the
per-step CLI surface from ``StepConfiguration`` is forwarded as keyword
values, ``target_dir``/``output_dir`` resolve to the standard numbered
``<stem>_output`` directory, and the return value coerces through the shared
``coerce_step_exit_code`` contract. The returned receipt dict matches
``execute_pipeline_step``'s ``step_result`` shape plus an ``execution_mode``
field, so ``main._record_step_result`` treats both modes identically.

Limits (docs/decisions/0001-consolidated-pipeline-execution.md): testing-matrix
folder dispatch stays on the subprocess path. Timeout containment is
cooperative (BC-13): the worker thread is a daemon and observes a
:class:`~gnn.execute.subprocess_envelope.CancelToken` at its safe points —
the capture tees raise at the step's next console write once the wall-clock
budget fires, and a thread that reaches no safe point is abandoned after a
bounded grace window without blocking interpreter exit. The parallel tier
runs consolidated steps on threads that share one process: steps 7/8 render
with matplotlib (Agg) — thread-safe in-process, but concurrent consolidated
runs of step 8 should expect shared matplotlib state (standing limit; no
behavioral change attempted for matplotlib).
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, TextIO, Tuple, cast

from gnn.pipeline.config import get_output_dir_for_script
from gnn.pipeline.step_registry import (
    CONSOLIDATED_IN_PROCESS_STEMS,
    canonical_step_stem,
    step_for_stem,
)
from gnn.pipeline.step_timeouts import get_step_timeout
from gnn.utils.arguments.step_config import StepConfiguration
from gnn.utils.errors.error_handling import coerce_step_exit_code, status_from_exit_code
from gnn.utils.observability.structured_logging import log_step_warning
from gnn.utils.pipeline_orchestration.pipeline_validator import (
    validate_step_prerequisites,
)
from gnn.utils.runtime_safety.resource_manager import get_current_memory_usage

__all__ = [
    "UnsupportedStepError",
    "can_execute_in_process",
    "clear_parsed_model_carrier",
    "execute_step_in_process",
    "resolve_step_function",
]

# Handled positionally exactly like the numbered-script wrapper handles them;
# never forwarded a second time through **kwargs.
_ARGS_EXCLUDED_FROM_KWARGS = frozenset(
    {"target_dir", "output_dir", "recursive", "verbose"}
)


class UnsupportedStepError(ValueError):
    """Raised when a step is refused by the consolidated executor."""


# ---------------------------------------------------------------------------
# Console capture: tee stdout/stderr into the receipt (console still streams)
# ---------------------------------------------------------------------------
class _StepCancelled(BaseException):
    """Unwinds a timed-out step at its next console write (BC-13).

    Deliberately a :class:`BaseException` subclass: the timeout receipt owns
    the failure shape, and the interrupt must not be swallowed by a step's
    broad ``except Exception`` handlers — it propagates like
    ``KeyboardInterrupt``. The executor discards it; the thread exiting is
    the point.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _ConsoleStreamTee:
    """Thread-scoped console tee used while a consolidated step runs.

    Writes from the step's worker thread are tee'd: forwarded to the real
    console stream and captured into that thread's sink for the receipt.
    Writes from every other thread pass through untouched, so pipeline
    logging is never mis-attributed to a step. Install/uninstall are
    reference-counted under a lock so concurrent consolidated steps (e.g.
    the parallel tier) never leave the process streams swapped.
    """

    def __init__(self, stream_attr: str) -> None:
        self._attr = stream_attr
        self._lock = threading.Lock()
        self._depth = 0
        self._saved: Any = None
        self._local = threading.local()

    def _live_target(self) -> TextIO:
        saved = self._saved
        target = saved if saved is not None else getattr(sys, self._attr)
        return cast("TextIO", target)

    def open_for_thread(
        self, sink: Any, cancel_gate: Optional[Callable[[], None]] = None
    ) -> None:
        """Route the calling thread's writes into *sink* (plus console).

        *cancel_gate* (BC-13) is consulted before every write from this
        thread and raises :class:`_StepCancelled` once the step's wall-clock
        budget has fired — the step terminates at its next print/log
        boundary with no module change.
        """
        self._local.sink = sink
        self._local.cancel_gate = cancel_gate

    def close_for_thread(self) -> None:
        """Stop capturing the calling thread's writes."""
        self._local.sink = None
        self._local.cancel_gate = None

    def install(self) -> None:
        with self._lock:
            if self._depth == 0:
                self._saved = getattr(sys, self._attr)
                setattr(sys, self._attr, self)
            self._depth += 1

    def uninstall(self) -> None:
        with self._lock:
            if self._depth > 0:
                self._depth -= 1
            if self._depth == 0 and self._saved is not None:
                setattr(sys, self._attr, self._saved)
                self._saved = None

    def write(self, text: str) -> int:
        gate = getattr(self._local, "cancel_gate", None)
        if gate is not None:
            gate()
        sink = getattr(self._local, "sink", None)
        if sink is not None:
            try:
                sink.write(text)
            except Exception:  # capture must never break the step's output
                pass
        return self._live_target().write(text)

    def flush(self) -> None:
        self._live_target().flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._live_target(), name)


_STDOUT_CAPTURE = _ConsoleStreamTee("stdout")
_STDERR_CAPTURE = _ConsoleStreamTee("stderr")


@contextlib.contextmanager
def _console_capture_window() -> Generator[None, None, None]:
    """Swap in the capture tees for the calling thread's step run."""
    _STDOUT_CAPTURE.install()
    _STDERR_CAPTURE.install()
    try:
        yield
    finally:
        _STDERR_CAPTURE.uninstall()
        _STDOUT_CAPTURE.uninstall()


# ---------------------------------------------------------------------------
# Cooperative force-kill (BC-13): token seam + console-write gate + daemon
# ---------------------------------------------------------------------------
#: Step module functions receive a ``cancel_token`` kwarg only when their
#: signature declares one (the same opt-in shape as the parsed-model carrier).
_CANCEL_TOKEN_PARAM = "cancel_token"

#: Seconds the executor waits after cancelling a step's token for the worker
#: thread to reach a safe point before it abandons it (daemon; cannot block
#: interpreter exit).
_CANCEL_GRACE_SECONDS = 2.0

#: Sentinel written into a timed-out step's output dir (artifacts are marked,
#: never deleted); a later successful run of the same step removes it.
_TIMEOUT_MARKER_NAME = ".gnn_step_timed_out"

#: Cap on the per-file candidate list recorded in the timeout marker.
_TIMEOUT_MARKER_CANDIDATE_CAP = 200

#: Slack subtracted from the step's start time before comparing file mtimes.
#: Filesystem timestamps come from a coarser clock than ``time.time()`` (Linux
#: stamps mtimes from the kernel's coarse clock, a few ms behind; HFS+/FAT
#: round to 1-2 s), so a file written just after the step started can carry
#: an mtime *earlier* than the recorded start. Over-reporting a stale file as
#: possibly partial is harmless; missing a truncated one is not.
_TIMEOUT_MARKER_MTIME_SLACK_SECONDS = 2.0


def _new_cancel_token() -> Any:
    """Create the step's cooperative cancel token (BC-13).

    The type is the subprocess envelope's :class:`CancelToken` — one
    cancellation vocabulary across the repo, mirroring
    ``run_subprocess_envelope``'s token. Imported lazily: a module-level
    import would couple the pipeline import chain to the ``gnn.execute``
    package init (the same coupling ADR 0001's D2 slice avoids for
    ``advanced_visualization``).
    """
    from gnn.execute.subprocess_envelope import CancelToken  # lazy: heavy init

    return CancelToken()


def _accepts_cancel_token(function: Callable[..., Any]) -> bool:
    """True when the resolved step function declares a ``cancel_token`` param."""
    try:
        return _CANCEL_TOKEN_PARAM in inspect.signature(function).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False


def _mark_timed_out_artifacts(
    step_output_dir: Path,
    stem: str,
    *,
    timeout_seconds: int,
    started: float,
    logger: logging.Logger,
) -> None:
    """Mark (never delete) artifacts written while a timed-out step ran.

    The sentinel names the step, the deadline, and every file under the
    step's output dir whose mtime postdates the step's start (less a
    timestamp-granularity slack) — candidates
    for mid-write truncation by the cancelled/abandoned thread. Downstream
    consumers treat a marked directory as suspect; a later successful run of
    the same step re-authors the artifacts and removes the sentinel.
    """
    if not step_output_dir.is_dir():
        # The step never created its output dir: nothing was written, so
        # there is nothing to mark (and no directory to hold a sentinel).
        return
    candidates: list[str] = []
    truncated = False
    mtime_floor = started - _TIMEOUT_MARKER_MTIME_SLACK_SECONDS
    try:
        for path in sorted(step_output_dir.rglob("*")):
            try:
                if not path.is_file() or path.name == _TIMEOUT_MARKER_NAME:
                    continue
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= mtime_floor:
                if len(candidates) >= _TIMEOUT_MARKER_CANDIDATE_CAP:
                    truncated = True
                    break
                candidates.append(path.relative_to(step_output_dir).as_posix())
    except OSError as error:
        logger.warning("Timeout marker: could not scan %s: %s", step_output_dir, error)
    payload = {
        "step": stem,
        "force_killed": True,
        "timeout_seconds": timeout_seconds,
        "timed_out_at": datetime.now(timezone.utc).isoformat(),
        "possibly_partial": candidates,
        "possibly_partial_truncated": truncated,
        "note": (
            "Artifacts in this directory may be partial: the step exceeded "
            "its wall-clock timeout and its worker was cancelled/abandoned. "
            "Files listed under possibly_partial were written during the "
            "timed-out run. Re-run the step to re-author them."
        ),
    }
    try:
        (step_output_dir / _TIMEOUT_MARKER_NAME).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as error:
        logger.warning(
            "Timeout marker: could not write %s in %s: %s",
            _TIMEOUT_MARKER_NAME,
            step_output_dir,
            error,
        )


def _clear_stale_timeout_marker(step_output_dir: Path, logger: logging.Logger) -> None:
    """Remove a stale timeout sentinel after this step re-authored its dir."""
    marker = step_output_dir / _TIMEOUT_MARKER_NAME
    try:
        if marker.is_file():
            marker.unlink()
    except OSError as error:
        logger.warning("Could not remove stale timeout marker %s: %s", marker, error)


# ---------------------------------------------------------------------------
# Parsed-model carrier: in-memory step 3 -> steps 7/8 handoff
# ---------------------------------------------------------------------------
#: Steps whose module functions consume a ``parsed_model`` carrier kwarg.
_PARSED_MODEL_CONSUMER_STEMS = frozenset({"7_export", "8_visualization"})

#: Step-3 parse artifacts collected once per run, keyed by the resolved
#: output dir. Serial within a run; lock-guarded for concurrent tiers.
_PARSED_MODEL_CARRIERS: Dict[str, Dict[str, Any]] = {}
_CARRIER_LOCK = threading.Lock()


def _carrier_key(output_dir: Any) -> str:
    """Carrier cache key: the run's resolved output directory."""
    return os.fspath(Path(output_dir).resolve())


def _parsed_model_carrier_for(args: Any) -> Optional[Dict[str, Any]]:
    """Return the collected step-3 carrier for *args*' run, if any."""
    with _CARRIER_LOCK:
        return _PARSED_MODEL_CARRIERS.get(_carrier_key(args.output_dir))


def _collect_parsed_model_carrier(
    output_dir: Path, logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """Read step 3's on-disk parse artifacts exactly once.

    The carrier bundles ``gnn_processing_results.json`` plus every
    successful ``{model}_parsed.json`` it references, keyed by source-file
    stem — the same bytes steps 7/8 would otherwise re-read from disk.
    """
    step3_dir = get_output_dir_for_script("3_gnn", Path(output_dir))
    results_file = step3_dir / "gnn_processing_results.json"
    if not results_file.is_file():
        return None
    try:
        with open(results_file, encoding="utf-8") as handle:
            results = json.load(handle)
    except (OSError, ValueError) as error:
        logger.warning(
            "Parsed-model carrier: could not read %s: %s", results_file, error
        )
        return None
    if not isinstance(results, dict) or not isinstance(
        results.get("processed_files"), list
    ):
        return None
    models: Dict[str, Any] = {}
    for entry in results["processed_files"]:
        if not isinstance(entry, dict) or not entry.get("parse_success"):
            continue
        parsed_path_text = entry.get("parsed_model_file")
        if not parsed_path_text:
            continue
        parsed_path = Path(parsed_path_text)
        try:
            with open(parsed_path, encoding="utf-8") as handle:
                model = json.load(handle)
        except (OSError, ValueError) as error:
            logger.warning(
                "Parsed-model carrier: could not read %s: %s", parsed_path, error
            )
            continue
        source_path = entry.get("file_path")
        key = Path(source_path).stem if source_path else parsed_path.parent.name
        models[key] = model
    return {"results": results, "models": models}


def _refresh_parsed_model_carrier(
    args: Any,
    logger: logging.Logger,
    *,
    collect: bool,
    succeeded: bool,
) -> None:
    """Refresh or drop the run's carrier after an in-process step 3.

    Collected only when the caller opted in and step 3 succeeded; a
    non-collecting (or failed) step 3 rewrites the artifacts the carrier
    describes, so any cached carrier for the run is dropped rather than
    left stale.
    """
    key = _carrier_key(args.output_dir)
    if not collect or not succeeded:
        with _CARRIER_LOCK:
            _PARSED_MODEL_CARRIERS.pop(key, None)
        return
    carrier = _collect_parsed_model_carrier(Path(args.output_dir), logger)
    with _CARRIER_LOCK:
        if carrier is None:
            _PARSED_MODEL_CARRIERS.pop(key, None)
        else:
            _PARSED_MODEL_CARRIERS[key] = carrier


def clear_parsed_model_carrier(output_dir: Optional[Path] = None) -> None:
    """Drop cached parsed-model carriers (one run's, or all when omitted).

    Exposed for callers that re-run steps outside the executor (tests,
    tooling): the cache must never outlive the artifacts it mirrors.
    """
    with _CARRIER_LOCK:
        if output_dir is None:
            _PARSED_MODEL_CARRIERS.clear()
        else:
            _PARSED_MODEL_CARRIERS.pop(_carrier_key(output_dir), None)


def canonical_stem_for_script(script_name: str) -> str:
    """Normalize ``11_render.py`` / ``15_llm`` style names to a canonical stem."""
    stem = script_name[:-3] if script_name.endswith(".py") else script_name
    return canonical_step_stem(stem)


def _global_steps_skip_note(
    stem: str, pipeline_config: Optional[Dict[str, Any]]
) -> Optional[str]:
    """Mirror execute_pipeline_step's global_steps skip for a whitelisted step."""
    if not pipeline_config:
        return None
    matrix = pipeline_config.get("testing_matrix", {})
    if not isinstance(matrix, dict):
        return None
    global_steps = matrix.get("global_steps", {})
    if (
        isinstance(global_steps, dict)
        and stem in global_steps
        and not global_steps[stem]
    ):
        return f"Skipped by global_steps config (testing_matrix.global_steps.{stem}: false)"
    return None


def _matrix_dispatch_active(
    stem: str, args: Any, pipeline_config: Optional[Dict[str, Any]]
) -> bool:
    """Mirror execute_pipeline_step's per-folder dispatch condition.

    The subprocess path forks one run per target subfolder only when the
    testing matrix is enabled and at least one subfolder allows this step;
    flat target dirs run standard mode and stay consolidated-eligible.
    """
    if not pipeline_config:
        return False
    matrix = pipeline_config.get("testing_matrix", {})
    if not isinstance(matrix, dict) or not matrix.get("enabled", False):
        return False
    try:
        step_num = int(stem.split("_")[0])
    except ValueError:
        return False
    if step_num < 3:  # execute_pipeline_step computes folders only for >= 3
        return False
    target_dir = Path(args.target_dir)
    if not (target_dir.exists() and target_dir.is_dir()):
        return False
    folders_config = matrix.get("folders", {})
    default_steps = matrix.get("default_steps", [])
    for item in target_dir.iterdir():
        if item.is_dir() and item.name != "archived_gnn_files":
            allowed_steps = folders_config.get(item.name, default_steps)
            if step_num in allowed_steps:
                return True
    return False


def can_execute_in_process(
    script_name: str,
    args: Any,
    *,
    pipeline_config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return True when *script_name* is eligible for consolidated execution.

    Eligible means: whitelisted in ``CONSOLIDATED_IN_PROCESS_STEMS``, not
    folder-dispatched by the testing matrix, and not skipped by
    ``testing_matrix.global_steps``.
    """
    stem = canonical_stem_for_script(script_name)
    if stem not in CONSOLIDATED_IN_PROCESS_STEMS:
        return False
    if _matrix_dispatch_active(stem, args, pipeline_config):
        return False
    return _global_steps_skip_note(stem, pipeline_config) is None


def resolve_step_function(script_name: str) -> Callable[..., Any]:
    """Resolve the registry's ``module_function`` for *script_name*.

    The numbered script module (``gnn.<stem>``) remains the resolution
    surface — registry stem to script module to function name — so the
    executor adds no second step mapping of its own.
    """
    step = step_for_stem(canonical_stem_for_script(script_name))
    if step is None:
        raise UnsupportedStepError(f"Unknown pipeline step: {script_name!r}")
    module = importlib.import_module(f"gnn.{step.script_stem}")
    function = getattr(module, step.module_function, None)
    if not callable(function):
        raise UnsupportedStepError(
            f"{step.script_stem}: registry function "
            f"{step.module_function!r} does not resolve to a callable"
        )
    return cast("Callable[..., Any]", function)


def _in_process_step_kwargs(stem: str, args: Any) -> Dict[str, Any]:
    """Forward exactly the step-configured CLI arguments as keyword values.

    Mirrors ``build_step_command_args``: the per-step argument surface comes
    from ``StepConfiguration`` (required_args + optional_args), values from
    ``PipelineArguments``, and optional ``None`` values are omitted exactly
    as the subprocess command omits them.
    """
    config = StepConfiguration.get_step_config(stem)
    required = frozenset(config.get("required_args", []))
    kwargs: Dict[str, Any] = {}
    for name in list(config.get("required_args", [])) + list(
        config.get("optional_args", [])
    ):
        if name in _ARGS_EXCLUDED_FROM_KWARGS or not hasattr(args, name):
            continue
        value = getattr(args, name)
        if value is None and name not in required:
            continue
        kwargs[name] = value
    return kwargs


def execute_step_in_process(
    script_name: str,
    args: Any,
    logger: logging.Logger,
    *,
    run_id: Optional[str] = None,
    pipeline_config: Optional[Dict[str, Any]] = None,
    timeout_seconds: Optional[int] = None,
    collect_parsed_model: bool = False,
) -> Dict[str, Any]:
    """Run one whitelisted step in-process and return its execution receipt.

    The receipt matches ``execute_pipeline_step``'s ``step_result`` contract
    (same keys, same status mapping) so the shared recording tail in ``main``
    treats both modes identically; ``execution_mode`` records which mode ran.
    Raises :class:`UnsupportedStepError` for steps outside the consolidated
    whitelist or under testing-matrix folder dispatch.

    The step runs on a daemon worker thread with a wall-clock timeout sourced
    from ``gnn.pipeline.step_timeouts`` — the same knob as the subprocess
    tier (``GNN_STEP_TIMEOUT_{N}`` / ``GNN_STEP_TIMEOUT_SCALE`` env overrides
    included) — and its ``print()``/``sys.stdout``/``sys.stderr`` output is
    tee'd to the console and captured into the receipt's ``stdout``/``stderr``
    fields. A timeout records the subprocess tier's timeout receipt semantics
    (exit code -1, FAILED status, partial captured streams) plus the additive
    ``force_killed`` field (``False`` on clean completion, ``True`` when the
    deadline fired). On timeout the executor cancels the step's cooperative
    :class:`~gnn.execute.subprocess_envelope.CancelToken` — injected as a
    ``cancel_token`` kwarg when the step function declares one, and enforced
    at the step's console-write safe points by the capture tee — then, after
    a bounded grace window, abandons any thread that reached no safe point
    (the daemon worker cannot block interpreter exit). Timed-out artifacts
    are marked, never deleted: a ``.gnn_step_timed_out`` sentinel lands in
    the step's output directory, removed by a later successful run.

    With ``collect_parsed_model=True`` the executor participates in the
    parsed-model carrier: a successful in-process step 3 is read from disk
    exactly once and cached for the run, and in-process steps 7/8 receive
    it as a ``parsed_model`` kwarg (absent flag: steps behave exactly as
    before).
    """
    stem = canonical_stem_for_script(script_name)
    if stem not in CONSOLIDATED_IN_PROCESS_STEMS:
        raise UnsupportedStepError(
            f"Step {script_name!r} is not enabled for consolidated in-process execution"
        )
    if _matrix_dispatch_active(stem, args, pipeline_config):
        raise UnsupportedStepError(
            f"Step {script_name!r} dispatches testing-matrix folders; "
            "the consolidated executor refuses it"
        )

    step = step_for_stem(stem)
    if step is None:  # pragma: no cover - whitelist check above guarantees this
        raise UnsupportedStepError(f"Unknown pipeline step: {script_name!r}")

    skip_note = _global_steps_skip_note(stem, pipeline_config)
    if skip_note is not None:
        return {
            "status": "SKIPPED",
            "stdout": f"{skip_note}\n",
            "stderr": "",
            "memory_usage_mb": 0.0,
            "peak_memory_mb": 0.0,
            "memory_delta_mb": 0.0,
            "exit_code": 0,
            "retry_count": 0,
            "prerequisite_check": True,
            "dependency_warnings": [],
            "execution_mode": "consolidated",
            "force_killed": False,
        }

    function = resolve_step_function(script_name)
    step_output_dir = get_output_dir_for_script(stem, Path(args.output_dir))

    start_memory = get_current_memory_usage()
    step_result: Dict[str, Any] = {
        "status": "UNKNOWN",
        "stdout": "",
        "stderr": "",
        "memory_usage_mb": 0.0,
        "peak_memory_mb": 0.0,
        "memory_delta_mb": 0.0,
        "exit_code": -1,
        "retry_count": 0,
        "prerequisite_check": True,
        "dependency_warnings": [],
        "execution_mode": "consolidated",
        "force_killed": False,
    }

    # Prerequisite validation mirrors execute_pipeline_step: record the
    # outcome on the receipt, never gate the step on it.
    config_skip_steps: list[Any] = []
    pipeline_section = (pipeline_config or {}).get("pipeline", {})
    if isinstance(pipeline_section, dict):
        config_skip_steps = list(pipeline_section.get("skip_steps", []) or [])
    prereq_result = validate_step_prerequisites(
        script_name, args, logger, skip_steps=config_skip_steps
    )
    step_result["prerequisite_check"] = prereq_result["passed"]
    step_result["dependency_warnings"] = prereq_result.get("warnings", [])
    for prereq_warning in prereq_result.get("warnings", []):
        logger.warning(f"Prerequisite notice for {script_name}: {prereq_warning}")

    if run_id is not None:
        os.environ["GNN_RUN_ID"] = run_id

    call_kwargs: Dict[str, Any] = {
        "target_dir": Path(args.target_dir),
        "output_dir": step_output_dir,
        "logger": logger,
        "recursive": args.recursive,
        "verbose": args.verbose,
    }
    call_kwargs.update(_in_process_step_kwargs(stem, args))
    if collect_parsed_model and stem in _PARSED_MODEL_CONSUMER_STEMS:
        carrier = _parsed_model_carrier_for(args)
        if carrier is not None:
            call_kwargs["parsed_model"] = carrier

    cancel_token = _new_cancel_token()
    if _accepts_cancel_token(function):
        # BC-13: token-aware steps receive the token and terminate promptly
        # at their safe points once it fires. Steps that do not declare the
        # parameter keep today's behavior (the tee's console-write gate and
        # the daemon worker still contain the timeout).
        call_kwargs[_CANCEL_TOKEN_PARAM] = cancel_token

    if timeout_seconds is None:
        # Same budget knob as the subprocess tier (main.py): the per-step
        # STEP_TIMEOUTS value with the GNN_STEP_TIMEOUT_{N} /
        # GNN_STEP_TIMEOUT_SCALE env overrides.
        comprehensive_requested = any("--comprehensive" in str(arg) for arg in sys.argv)
        timeout_seconds = get_step_timeout(
            script_name, comprehensive=comprehensive_requested
        )

    call_started = time.time()
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    holder: Dict[str, Any] = {}

    def _step_cancel_gate() -> None:
        """Kill surface: raise at the step's next console write once cancelled."""
        if cancel_token.cancelled:
            raise _StepCancelled(cancel_token.reason or "step timed out")

    def _invoke() -> Tuple[Any, Optional[BaseException]]:
        _STDOUT_CAPTURE.open_for_thread(captured_stdout, cancel_gate=_step_cancel_gate)
        _STDERR_CAPTURE.open_for_thread(captured_stderr, cancel_gate=_step_cancel_gate)
        try:
            with _console_capture_window():
                return function(**call_kwargs), None
        except BaseException as error:  # receipt contract owns the failure shape
            return None, error
        finally:
            _STDERR_CAPTURE.close_for_thread()
            _STDOUT_CAPTURE.close_for_thread()

    def _run_step() -> None:
        try:
            holder["payload"] = _invoke()
        except BaseException as error:  # pragma: no cover - _invoke is total
            holder["payload"] = (None, error)

    # Daemon worker (BC-13): an abandoned step thread can no longer block
    # interpreter exit — the replaced ThreadPoolExecutor threads were
    # non-daemon, so a hung step delayed process exit indefinitely.
    worker = threading.Thread(
        target=_run_step, name=f"gnn-consolidated-{stem}", daemon=True
    )
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        # The wall-clock budget fired. Mirror the subprocess tier's timeout
        # receipt (execute_command_streaming -> execute_pipeline_step): exit
        # code -1, FAILED status, partial captured streams, schema plus the
        # additive force_killed field. Cancel the step's token first: a
        # token-aware step unwinds at its safe points, and the tee gate
        # raises at the step's next console write; a thread that reaches no
        # safe point within the grace window is abandoned (daemon).
        cancel_token.cancel(
            reason=(
                f"{step.script_name} exceeded its {timeout_seconds}s wall-clock timeout"
            )
        )
        worker.join(_CANCEL_GRACE_SECONDS)
        thread_stopped = not worker.is_alive()
        end_memory = get_current_memory_usage()
        step_result["exit_code"] = -1
        step_result["status"] = status_from_exit_code(
            step_result["exit_code"], step_result["dependency_warnings"]
        )
        step_result["stdout"] = captured_stdout.getvalue()
        timeout_notice = (
            f"Consolidated in-process execution of {step.script_name} "
            f"exceeded its {timeout_seconds}s wall-clock timeout "
            "(TIMEOUT; recorded as FAILED with exit code -1). "
        )
        if thread_stopped:
            timeout_notice += (
                "The step thread stopped via the cooperative cancel token "
                "(consolidated steps observe it at their safe points; the "
                "console-write path enforces it) and was joined. Its "
                "post-deadline output is discarded from this receipt."
            )
        else:
            timeout_notice += (
                "The step thread did not reach a cancellation safe point "
                f"within the {_CANCEL_GRACE_SECONDS:g}s grace window and was "
                "abandoned: the worker runs as a daemon thread, so it cannot "
                "block interpreter exit, but its in-flight work was not "
                "interrupted mid-flight and any further output is discarded "
                "from this receipt."
            )
        partial_stderr = captured_stderr.getvalue()
        step_result["stderr"] = (
            partial_stderr + ("\n" if partial_stderr else "") + timeout_notice + "\n"
        )
        step_result["memory_usage_mb"] = end_memory
        step_result["peak_memory_mb"] = max(start_memory, end_memory)
        step_result["memory_delta_mb"] = end_memory - start_memory
        step_result["force_killed"] = True
        _mark_timed_out_artifacts(
            step_output_dir,
            stem,
            timeout_seconds=timeout_seconds,
            started=call_started,
            logger=logger,
        )
        logger.error(
            "Consolidated in-process execution of %s timed out after %ss "
            "(force_killed=true): %s.",
            step.script_name,
            timeout_seconds,
            (
                "the step thread stopped via the cancel token and was joined"
                if thread_stopped
                else "the step thread was abandoned as a daemon (no "
                "cancellation safe point reached within the grace window)"
            ),
        )
        return step_result

    payload = holder.get("payload")
    if payload is None:  # pragma: no cover - _invoke is total
        payload = (
            None,
            RuntimeError("consolidated step worker died without a result"),
        )
    result, error = cast("Tuple[Any, Optional[BaseException]]", payload)

    if error is not None:
        logger.error(
            "Consolidated in-process execution of %s failed: %s",
            step.script_name,
            error,
        )
        step_result["stderr"] = (
            captured_stderr.getvalue() + f"{type(error).__name__}: {error}\n"
        )
        result = False
    else:
        step_result["stderr"] = captured_stderr.getvalue()

    exit_code = coerce_step_exit_code(
        result,
        step_name=stem,
        logger=logger,
        warning_callback=lambda message: log_step_warning(logger, message),
    )
    end_memory = get_current_memory_usage()

    step_result["exit_code"] = exit_code
    step_result["status"] = status_from_exit_code(
        exit_code, step_result["dependency_warnings"]
    )
    step_result["stdout"] = captured_stdout.getvalue() + (
        f"{step.script_name}: consolidated in-process execution completed "
        f"in {time.time() - call_started:.2f}s\n"
    )
    step_result["memory_usage_mb"] = end_memory
    step_result["peak_memory_mb"] = max(start_memory, end_memory)
    step_result["memory_delta_mb"] = end_memory - start_memory
    if step_result["status"] in ("SUCCESS", "SUCCESS_WITH_WARNINGS"):
        # This run re-authored the step's artifacts: a timeout sentinel from
        # an earlier run of the same step into the same directory is stale.
        _clear_stale_timeout_marker(step_output_dir, logger)

    if stem == "3_gnn":
        _refresh_parsed_model_carrier(
            args,
            logger,
            collect=collect_parsed_model,
            succeeded=step_result["status"] in ("SUCCESS", "SUCCESS_WITH_WARNINGS"),
        )
    return step_result
