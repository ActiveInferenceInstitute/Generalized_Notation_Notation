#!/usr/bin/env python3
"""
GNN API Job Manager — in-memory job tracking and async pipeline execution.

Manages job lifecycle: create → execute → poll → result.
Uses asyncio for non-blocking pipeline execution.
No database dependency — jobs are stored in memory (lost on restart).
"""

import asyncio
import logging
import os
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from gnn.api.models import validate_step_numbers as _shared_validate_step_numbers
from gnn.api.path_utils import get_repo_root, resolve_request_paths
from gnn.api.pipeline_runner import (
    build_pipeline_command,
    normalize_summary_steps,
    pipeline_exit_succeeded,
    read_pipeline_summary,
    summarize_step_progress,
)
from gnn.api.process_supervision import (
    ProcessCleanupError,
    request_process_stop,
    supervise_api_process,
)
from gnn.pipeline.admission import validate_boolean
from gnn.pipeline.step_registry import STEPS
from gnn.utils.runtime_safety.filesystem import directory_handle, is_redirect

# In-memory job store (cleared on restart — research tool, not production service)
_JOBS: Dict[str, dict[str, Any]] = {}


RUNS_STORE: Dict[str, dict[str, Any]] = {}

# States a run record reaches and never leaves.
_TERMINAL_RUN_STATES = frozenset({"completed", "failed", "cancelled"})


class RunNotFoundError(Exception):
    """No run record matches the requested run hash or prefix."""


class RunAmbiguousError(Exception):
    """The requested run-hash prefix matches more than one run record."""


def _validate_step_numbers(
    values: Optional[List[int]], *, field_name: str
) -> Optional[List[int]]:
    """Validate an optional list of unique pipeline steps.

    Delegates to the shared contract validator in ``api.models`` so the job
    manager and the request models can never disagree on step semantics.
    """
    return _shared_validate_step_numbers(values, field_name=field_name)


def create_job(
    target_dir: str,
    output_dir: Optional[str] = None,
    steps: Optional[List[int]] = None,
    skip_steps: Optional[List[int]] = None,
    verbose: bool = False,
    strict: bool = False,
    parallel: bool = False,
    consolidated_steps: bool = False,
) -> str:
    """
    Create a new pipeline job and return its ID.

    Args:
        target_dir: Directory containing GNN files
        output_dir: Directory where pipeline outputs should be written
        steps: Specific steps to run (None = all)
        skip_steps: Steps to skip
        verbose: Enable verbose output
        strict: Treat warnings as errors

    Returns:
        Unique job ID string
    """
    steps = _validate_step_numbers(steps, field_name="steps")
    skip_steps = _validate_step_numbers(skip_steps, field_name="skip_steps")
    validate_boolean(verbose, field_name="verbose")
    validate_boolean(strict, field_name="strict")
    validate_boolean(parallel, field_name="parallel")
    validate_boolean(consolidated_steps, field_name="consolidated_steps")
    overlap = set(steps or ()) & set(skip_steps or ())
    if overlap:
        raise ValueError(f"steps and skip_steps must not overlap: {sorted(overlap)}")

    target_path, output_path = resolve_request_paths(target_dir, output_dir or "output")

    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "target_dir": str(target_path),
        "steps": steps,
        "skip_steps": skip_steps,
        "verbose": verbose,
        "strict": strict,
        "parallel": parallel,
        "consolidated_steps": consolidated_steps,
        "progress_step": None,
        "steps_completed": [],
        "steps_failed": [],
        "exit_code": None,
        "error_message": None,
        "output_dir": str(output_path),
        "process": None,  # subprocess handle (not serializable, stripped in get_job)
    }
    logger.info(
        f"Created job {job_id} for target={target_path}, output={output_path}, steps={steps}"
    )
    return job_id


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    """
    Retrieve job status by ID.

    Returns a serializable dict (subprocess handle is stripped).
    """
    job = _JOBS.get(job_id)
    if job is None:
        return None

    # Return copy without non-serializable fields
    serializable = {k: v for k, v in job.items() if k != "process"}
    return serializable


def cancel_job(job_id: str) -> bool:
    """
    Cancel a running or pending job.

    Returns True if cancelled, False if job not found or already terminal.
    """
    job = _JOBS.get(job_id)
    if job is None:
        return False

    if job["status"] in ("completed", "failed", "cancelled"):
        return False

    # Terminate the subprocess tree if running
    proc = job.get("process")
    if proc is not None:
        try:
            request_process_stop(proc)
        except (OSError, AttributeError) as exc:
            logger.warning(f"Could not request process stop for job {job_id}: {exc}")

    job["status"] = "cancelled"
    job["completed_at"] = datetime.now().isoformat()
    return True


def _remove_run_artifacts(entry: dict[str, Any]) -> tuple[Optional[bool], str]:
    """Remove a run's artifact directory, never raising.

    A run that targeted the repository's own ``output`` tree keeps its
    artifacts: that tree is shared working state, not disposable output.

    Returns ``(artifacts_removed, artifacts_note)`` where the boolean is
    ``True`` after a successful removal, ``False`` when nothing needed
    removing (with the reason in the note), and ``None`` when artifact
    removal was not requested.
    """
    request = entry.get("request") or {}
    raw_output_dir = request.get("output_dir")
    if not raw_output_dir:
        return False, "no artifacts on disk"
    output_dir = Path(str(raw_output_dir)).absolute()
    try:
        resolved = output_dir.resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        return False, f"could not resolve artifact directory: {exc}"
    try:
        workspace = get_repo_root().resolve()
        repo_output = (workspace / "output").resolve()
    except (OSError, ValueError) as exc:
        return False, f"could not resolve API workspace: {exc}"
    if resolved == repo_output:
        return False, "repository output tree retained"
    if resolved == workspace or resolved in workspace.parents:
        return False, "API workspace or ancestor retained"
    try:
        # Never resolve a requested artifact link to its target before removal.
        # POSIX rmtree is fd-based, including its recursive symlink race checks.
        # The opened parent also cannot be redirected by a replaced ancestor.
        if os.name == "posix":
            if not shutil.rmtree.avoids_symlink_attacks:
                return False, "safe artifact removal is unavailable on this platform"
            with directory_handle(output_dir.parent) as parent:
                directory_entry = os.stat(output_dir.name, dir_fd=parent, follow_symlinks=False)
                if is_redirect(directory_entry):
                    return False, "artifact directory symlink or reparse point retained"
                shutil.rmtree(output_dir.name, dir_fd=parent)
        else:
            # Windows rmtree does not descend junction targets; admission still
            # requires directory entries trusted against concurrent replacement.
            node = Path(output_dir.anchor)
            for part in output_dir.parts[1:]:
                node /= part
                if is_redirect(node.lstat()):
                    return False, "artifact directory symlink or reparse point retained"
            shutil.rmtree(output_dir)
    except FileNotFoundError:
        return False, "no artifacts on disk"
    except (OSError, ValueError) as exc:
        return False, f"artifact removal failed: {exc}"
    return True, ""


def delete_run(
    run_hash: str,
    *,
    runs_store: Optional[Dict[str, dict[str, Any]]] = None,
    remove_artifacts: bool = True,
    wait_timeout: float = 10.0,
    poll_interval: float = 0.05,
) -> dict[str, Any]:
    """Delete a run record, cancelling it first when it is still active.

    Resolves an exact run hash or a unique prefix, then — for queued or
    running runs — cancels through the record's CancelToken and waits for
    the executor to reach a terminal state (a queued run skips the wait:
    its background task may not have started, and the executor tolerates a
    vanished record). Artifacts are removed afterwards, the repository
    output tree excepted, and the record is popped from the store.

    Raises:
        RunNotFoundError: No record matches the hash or prefix.
        RunAmbiguousError: The prefix matches several records.
        RuntimeError: A cancelled run did not reach a terminal state within
            ``wait_timeout``; the record is left in place with its current
            status.
    """
    store = RUNS_STORE if runs_store is None else runs_store
    if run_hash in store:
        key = run_hash
    else:
        matches = [known for known in store if known.startswith(run_hash)]
        if len(matches) > 1:
            raise RunAmbiguousError(f"Run hash prefix is ambiguous: {run_hash}")
        if not matches:
            raise RunNotFoundError(f"Run not found: {run_hash}")
        key = matches[0]

    entry = store[key]
    cancelled = False
    if entry["status"] in ("queued", "running"):
        token = entry.get("cancel_token")
        if token is not None:
            token.cancel(reason="run deleted")
            cancelled = True
        if entry["status"] == "running":
            deadline = time.monotonic() + wait_timeout
            while entry["status"] not in _TERMINAL_RUN_STATES:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"Run {key} did not reach a terminal state after "
                        f"cancel; status: {entry['status']}"
                    )
                time.sleep(min(poll_interval, remaining))

    artifacts_removed: Optional[bool] = None
    artifacts_note = ""
    cleanup = entry.get("process_cleanup") or {}
    if cleanup.get("cleanup_verified") is False:
        raise RuntimeError(
            f"Run {key} cleanup could not be verified; record and artifacts retained"
        )
    if remove_artifacts:
        artifacts_removed, artifacts_note = _remove_run_artifacts(entry)

    store.pop(key, None)
    return {
        "deleted": key,
        "existed": True,
        "cancelled": cancelled,
        "artifacts_removed": artifacts_removed,
        "artifacts_note": artifacts_note,
    }


def list_jobs(limit: int = 50) -> List[dict[str, Any]]:
    """List recent jobs (most recent first)."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100")
    jobs = [get_job(jid) for jid in list(_JOBS.keys())[-limit:]]
    return [j for j in jobs if j is not None]


async def execute_job_async(job_id: str) -> None:
    """
    Execute a pipeline job asynchronously.

    Runs the packaged `gnn.main` module with admitted arguments in a subprocess.
    Updates job status as execution progresses.

    This coroutine is meant to be launched with asyncio.create_task().
    """
    job = _JOBS.get(job_id)
    if job is None:
        logger.error(f"Cannot execute unknown job: {job_id}")
        return

    if job["status"] in ("completed", "failed", "cancelled"):
        # A cancel raced us before execution started; a cancelled job must
        # never launch its pipeline.
        logger.info(f"Skipping execution of job {job_id}: already {job['status']}")
        return

    job["status"] = "running"
    job["started_at"] = datetime.now().isoformat()
    logger.info(f"Starting job {job_id}")

    # Build the real orchestrator command via the shared pure builder so the
    # job surface and the run surface can never drift on argv shape.
    from gnn.api.path_utils import get_repo_root

    repo_root: Optional[Path] = None
    try:
        repo_root = get_repo_root()
        output_dir = Path(job.get("output_dir") or (repo_root / "output"))
        job["output_dir"] = str(output_dir)
        cmd = build_pipeline_command(
            str(job["target_dir"]),
            str(output_dir),
            only_steps=job.get("steps"),
            skip_steps=job.get("skip_steps"),
            verbose=job.get("verbose", False),
            strict=job.get("strict", False),
            parallel=job.get("parallel", False),
            consolidated_steps=job.get("consolidated_steps", False),
        )
        invocation_start_ns = time.time_ns()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(repo_root),
            env={**os.environ, "GNN_RUN_ID": job_id},
            # Own session/group so a cancel can signal the whole process
            # tree (mirrors utils.pipeline_orchestration.execution_utils).
            start_new_session=os.name == "posix",
        )
        job["process"] = proc

        stdout, stderr, cleanup = await supervise_api_process(
            proc, cancelled=lambda: job["status"] == "cancelled",
        )
        job["process_cleanup"] = cleanup

        # A cancel can race this coroutine: cancel_job terminates the
        # subprocess and writes the terminal 'cancelled' state while
        # communicate() is still waiting, and a terminated process exits
        # nonzero — which would otherwise re-report the job as 'failed'
        # with a stderr error message. Snapshot the flag before any state
        # write; there are no awaits below, so the check and the guarded
        # writes are atomic on the single loop thread that mutates jobs.
        was_cancelled = job["status"] == "cancelled"

        job["exit_code"] = proc.returncode
        if not was_cancelled:
            job["completed_at"] = datetime.now().isoformat()

        # Populate per-step progress from the canonical summary the pipeline
        # writes; the single subprocess gives us no live per-step view, so
        # progress is observable only once the run has finished.
        progress = summarize_step_progress(
            normalize_summary_steps(
                read_pipeline_summary(
                    output_dir,
                    not_before_ns=invocation_start_ns,
                    expected_run_id=job_id,
                )
                or []
            )
        )
        job["steps_completed"] = progress["steps_completed"]
        job["steps_failed"] = progress["steps_failed"]

        if was_cancelled:
            # Keep cancel_job's terminal write: 'cancelled' with no
            # fabricated error message, whatever partial output exists.
            logger.info(f"Job {job_id} was cancelled; keeping cancelled state")
        elif pipeline_exit_succeeded(proc.returncode, strict=bool(job.get("strict"))):
            job["status"] = "completed"
            logger.info(f"Job {job_id} completed successfully")
        else:
            job["status"] = "failed"
            # Capture a sanitized tail of stderr for the error message. Raw
            # stderr leaks internal paths, library versions, and stack traces;
            # redact the repository root and other absolute paths first.
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
            job["error_message"] = _sanitize_stderr(stderr_text, repo_root)
            logger.error(f"Job {job_id} failed with exit code {proc.returncode}")

    except ProcessCleanupError as e:
        job["process_cleanup"] = e.receipt
        job["status"] = "failed"
        job["error_message"] = f"Process cleanup could not be verified: {e}"
        job["completed_at"] = datetime.now().isoformat()
        logger.error("Job %s cleanup failed: %s", job_id, e)
    except Exception as e:
        if job["status"] == "cancelled":
            # cancel_job won the race during the exception; keep its write.
            logger.info(f"Job {job_id} was cancelled; ignoring exception: {e}")
            return
        job["status"] = "failed"
        job["error_message"] = _sanitize_stderr(str(e), repo_root or Path.home())
        job["completed_at"] = datetime.now().isoformat()
        logger.error(f"Job {job_id} raised exception: {e}")
    finally:
        job["process"] = None


def _sanitize_stderr(stderr_text: str, repo_root: Path) -> str:
    """Redact internal paths from a stderr tail before exposing it to clients.

    Keeps the diagnostic value of the tail (the last 500 chars) while removing
    the repository root and other absolute filesystem paths that would disclose
    host layout to an API caller.
    """
    tail = stderr_text[-500:] if len(stderr_text) > 500 else stderr_text
    tail = tail.replace(str(repo_root), "<repo>")
    # Common absolute path prefixes (home/usr/tmp/var/etc.) become <path>.
    tail = re.sub(r"(?:/[A-Za-z0-9_.-]+){2,}(?:/[^\s\"']*)?", "<path>", tail)
    return tail


# Pipeline step registry for the /tools endpoint — derived once from the
# canonical ``pipeline.step_registry.STEPS`` so this module is never the
# authority on which steps exist (single source of truth).
def _derive_pipeline_steps() -> Dict[int, tuple[str, str]]:
    """Map step number → (name, description) from the canonical registry."""
    registry: Dict[int, tuple[str, str]] = {}
    for step in STEPS:
        num_text, _, name = step.script_stem.partition("_")
        registry[int(num_text)] = (name, step.description)
    return registry


PIPELINE_STEPS: Dict[int, tuple[str, str]] = _derive_pipeline_steps()


def get_pipeline_tools() -> List[dict[str, Any]]:
    """Return list of available pipeline tools."""
    return [
        {
            "step_number": step,
            "name": name,
            "description": desc,
            "script": f"src/gnn/{step}_{name}.py",
        }
        for step, (name, desc) in PIPELINE_STEPS.items()
    ]
