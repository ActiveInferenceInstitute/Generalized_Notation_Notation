"""Actual spawned file work, bounded submission and cancellation contracts."""

import json
import logging
import multiprocessing
import os
import re
import signal
import threading
import time
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from pickle import PicklingError
from types import SimpleNamespace
from typing import Any

import pytest

from gnn.pipeline.run_context import build_run_context
from gnn.processing import process_gnn_directory_lightweight
from gnn.processing.multi_format_processor import process_gnn_multi_format
from gnn.utils.file_workers import ordered_file_map, resolve_file_workers


def _delayed_identity(job: tuple[int, float]) -> int:
    value, delay = job
    time.sleep(delay)
    return value


def _mark_worker_ready(directory: str) -> None:
    """Announce completion of this spawned interpreter's cold imports."""
    (Path(directory) / str(os.getpid())).touch()


def _hard_exit(_job: int) -> int:
    os._exit(7)


def _job_timeout(_job: int) -> int:
    raise TimeoutError("job timeout")


def _ignore_termination(marker: str) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(marker).touch()
    time.sleep(10)


def _crash_with_ignoring_peer(job: tuple[bool, str]) -> None:
    crash, marker = job
    if crash:
        deadline = time.monotonic() + 3
        while not Path(marker).exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        os._exit(7)
    _ignore_termination(marker)


def _without_timestamps(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "<generated node ID>"
                if key == "id" and "node_type" in value
                else _without_timestamps(item)
            )
            for key, item in value.items()
            if key
            not in {
                "timestamp",
                "parse_timestamp",
                "validation_timestamp",
                "created_at",
                "modified_at",
            }
        }
    if isinstance(value, list):
        return [_without_timestamps(item) for item in value]
    return value


class _CancelWhenReady:
    reason = "worker ready cancellation"

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    @property
    def cancelled(self) -> bool:
        return self.marker.exists()


@pytest.mark.parametrize(
    "workers,executor", [(True, None), (0, None), (257, None), (2, "unknown"), (2, {})]
)
def test_file_worker_admission(workers: Any, executor: Any) -> None:
    with pytest.raises(ValueError):
        resolve_file_workers(workers, executor)


def test_file_worker_defaults_and_bounded_ordered_submission() -> None:
    assert resolve_file_workers(None, None) == (1, "thread")
    assert resolve_file_workers(2, None) == (2, "process")
    assert resolve_file_workers(None, None, parallel=True)[1] == "thread"
    queued: list[int] = []

    def jobs() -> Any:
        for value in range(5):
            queued.append(value)
            yield value, 0.03 if value == 0 else 0.0

    results = ordered_file_map(_delayed_identity, jobs(), workers=2, executor="thread")
    assert next(results) == 0
    assert queued == [0, 1]
    assert list(results) == [1, 2, 3, 4]


def test_spawned_file_workers_preserve_results_and_cancel_without_children(
    tmp_path: Path,
) -> None:
    for index in range(4):
        (tmp_path / f"model{index}.md").write_text(
            "## ModelName\nDemo\n## StateSpaceBlock\nx: float\nx = 1\n"
        )
    serial = process_gnn_directory_lightweight(tmp_path)
    parallel = process_gnn_directory_lightweight(tmp_path, workers=2)
    assert parallel["success"]
    assert [p["file_path"] for p in parallel["parsed_files"]] == [
        p["file_path"] for p in serial["parsed_files"]
    ]
    assert [p["variables"] for p in parallel["parsed_files"]] == [
        p["variables"] for p in serial["parsed_files"]
    ]
    assert _without_timestamps(parallel["parsed_files"]) == _without_timestamps(
        serial["parsed_files"]
    )
    assert _without_timestamps(parallel["validation_results"]) == _without_timestamps(
        serial["validation_results"]
    )
    execution = parallel["worker_execution"]
    assert execution["effective_workers"] == 2 and execution["executor"] == "process"
    assert execution["worker_pids"] and os.getpid() not in execution["worker_pids"]
    before = {p.pid for p in multiprocessing.active_children()}
    with pytest.raises(TimeoutError, match="deadline"):
        list(
            ordered_file_map(
                _delayed_identity,
                [(0, 5.0), (1, 5.0)],
                workers=2,
                deadline=time.monotonic() + 0.3,
            )
        )
    assert {p.pid for p in multiprocessing.active_children()} <= before
    results = ordered_file_map(_delayed_identity, [(0, 0.0), (1, 5.0)], workers=2)
    assert next(results) == 0
    results.close()
    assert {p.pid for p in multiprocessing.active_children()} <= before


@pytest.mark.parametrize("public_kill", [True, False])
def test_hard_worker_exit_and_job_timeout_fail_without_waiting_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, public_kill: bool
) -> None:
    # Four cold spawn pools can exceed five seconds under xdist/coverage. Each
    # actual fault still has its own five-second completion/cleanup deadline;
    # cold interpreter readiness has a separate bounded startup allowance.
    import gnn.utils.file_workers as workers_module

    pool_type = workers_module.ProcessPoolExecutor
    if not public_kill:
        monkeypatch.setattr(pool_type, "kill_workers", None, raising=False)
    before = {p.pid for p in multiprocessing.active_children()}
    phases: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []

    def is_alive(process: Any) -> bool:
        try:
            return process.is_alive()
        except ValueError:
            return False  # The executor already reaped and closed its Process.

    def kill_owned(processes: tuple[Any, ...]) -> None:
        for process in processes:
            try:
                process.kill()
            except (OSError, ValueError):
                pass

    def ready_pool(**kwargs: Any) -> Any:
        directory = tmp_path / f"pool-{len(phases)}-ready"
        directory.mkdir()
        started = time.monotonic()
        pool = pool_type(
            **kwargs, initializer=_mark_worker_ready, initargs=(str(directory),)
        )
        try:
            warm = [
                pool.submit(_delayed_identity, (index, 0.0))
                for index in range(kwargs["max_workers"])
            ]
            owned = tuple(pool._processes.values())
            startup_deadline = started + 30
            while not all((directory / str(p.pid)).is_file() for p in owned):
                assert time.monotonic() < startup_deadline, "spawn startup exceeded 30s"
                for future in warm:
                    if future.done():
                        future.result()
                time.sleep(0.01)
            for future in warm:
                future.result(timeout=max(0.001, startup_deadline - time.monotonic()))
            assert time.monotonic() < startup_deadline, "spawn startup exceeded 30s"
        except BaseException:
            owned = tuple((getattr(pool, "_processes", None) or {}).values())
            kill_owned(owned)
            for process in owned:
                try:
                    process.join(timeout=1)
                except ValueError:
                    pass
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        expired = threading.Event()

        def overrun() -> None:
            expired.set()
            # Test emergency cleanup only: retain the failure, and signal only
            # captured Process objects belonging to this one actual pool.
            kill_owned(owned)

        timer = threading.Timer(5, overrun)
        timer.daemon = True
        phase = {
            "owned": owned,
            "pids": tuple(p.pid for p in owned),
            "expired": expired,
            "timer": timer,
            "ready_at": time.monotonic(),
            "startup_seconds": time.monotonic() - started,
        }
        phases.append(phase)
        timer.start()
        return pool

    monkeypatch.setattr(workers_module, "ProcessPoolExecutor", ready_pool)

    def check_fault(
        function: Any, jobs: Any, expected: Any, match: str | None = None
    ) -> None:
        previous = len(phases)
        try:
            with pytest.raises(expected, match=match):
                list(ordered_file_map(function, jobs, workers=2))
        finally:
            if len(phases) > previous:
                phase = phases[-1]
                phase["timer"].cancel()
                phase["timer"].join(timeout=1)
                finished = time.monotonic()
                leaked = [
                    pid
                    for p, pid in zip(phase["owned"], phase["pids"], strict=True)
                    if is_alive(p)
                ]
                # Ensure an assertion failure does not leave test-owned children.
                if leaked:
                    kill_owned(phase["owned"])
                    for process in phase["owned"]:
                        try:
                            process.join(timeout=1)
                        except ValueError:
                            pass
                elapsed = finished - phase["ready_at"]
                timings.append(
                    {
                        "fault": function.__name__,
                        "startup_seconds": phase["startup_seconds"],
                        "lifecycle_seconds": elapsed,
                        "worker_pids": list(phase["pids"]),
                        "watchdog_fired": phase["expired"].is_set(),
                        "leaked_on_return": leaked,
                    }
                )
                assert not phase["timer"].is_alive(), "lifecycle watchdog did not stop"
                assert not phase["expired"].is_set(), (
                    "file-worker lifecycle exceeded 5s"
                )
                assert elapsed < 5, "ready-to-exception/cleanup exceeded 5s"
                assert not leaked, "file workers survived the expected failure"
                assert {p.pid for p in multiprocessing.active_children()} <= before
        assert len(phases) == previous + 1, "fault occurred before worker readiness"

    check_fault(_hard_exit, [0, 1], BrokenProcessPool)
    check_fault(_job_timeout, [0, 1], TimeoutError, "job timeout")
    check_fault(
        lambda _job: 1,
        [0, 1],
        (AttributeError, TypeError, PicklingError),
        "pickle|local",
    )
    if os.name != "nt":
        marker = str(tmp_path / "ignoring_peer_ready")
        check_fault(
            _crash_with_ignoring_peer,
            [(True, marker), (False, marker)],
            BrokenProcessPool,
        )
        assert Path(marker).is_file(), "SIGTERM-ignoring peer never became ready"
    (tmp_path / "lifecycle-timings.json").write_text(
        json.dumps(timings, indent=2) + "\n"
    )
    print("file_worker_lifecycle " + json.dumps(timings))


@pytest.mark.timeout(5)
@pytest.mark.pipeline
@pytest.mark.needs_posix
def test_cancellation_kills_worker_even_when_sigterm_is_ignored(tmp_path: Path) -> None:
    marker = tmp_path / "ready"
    before = {p.pid for p in multiprocessing.active_children()}
    with pytest.raises(TimeoutError, match="worker ready cancellation"):
        list(
            ordered_file_map(
                _ignore_termination,
                [str(marker), str(marker)],
                workers=2,
                cancel_token=_CancelWhenReady(marker),
                deadline=time.monotonic() + 4,
            )
        )
    assert marker.exists()
    assert {p.pid for p in multiprocessing.active_children()} <= before


def test_multiformat_process_workers_keep_all_parent_owned_outputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "models"
    source.mkdir()
    exemplar = Path(__file__).parents[2] / "input/gnn_files/discrete/simple_mdp.md"
    for name in ("first", "second"):
        (source / f"{name}.md").write_bytes(exemplar.read_bytes())
    output = tmp_path / "output"
    assert process_gnn_multi_format(
        source,
        output,
        logging.getLogger(__name__),
        serialize_preset="minimal",
        file_workers=2,
    )
    receipt = json.loads(
        (output / "3_gnn_output/gnn_processing_results.json").read_text()
    )
    assert receipt["summary"]["successful_parses"] == 2
    assert receipt["summary"]["total_formats_generated"] == 6
    assert os.getpid() not in receipt["worker_execution"]["worker_pids"]
    assert [p["file_name"] for p in receipt["processed_files"]] == [
        "first.md",
        "second.md",
    ]
    for result in receipt["processed_files"]:
        assert Path(result["parsed_model_file"]).is_file()
        assert all(
            Path(item["output_file"]).is_file()
            for item in result["generated_formats"].values()
        )
    serial_output = tmp_path / "serial"
    assert process_gnn_multi_format(
        source, serial_output, logging.getLogger(__name__), serialize_preset="minimal"
    )
    for result in receipt["processed_files"]:
        worker_model = json.loads(Path(result["parsed_model_file"]).read_text())
        serial_dir = serial_output / "3_gnn_output" / Path(result["file_path"]).stem
        serial_model = json.loads(
            (serial_dir / Path(result["parsed_model_file"]).name).read_text()
        )
        assert _without_timestamps(worker_model) == _without_timestamps(serial_model)
        for info in result["generated_formats"].values():
            parallel_path = Path(info["output_file"])
            serial_path = serial_dir / parallel_path.name
            if parallel_path.suffix == ".json":
                assert _without_timestamps(
                    json.loads(parallel_path.read_text())
                ) == _without_timestamps(json.loads(serial_path.read_text()))
            else:
                assert re.sub(
                    r"(?m)^Generated: [^\n]+",
                    "Generated: <timestamp>",
                    parallel_path.read_text(),
                ) == re.sub(
                    r"(?m)^Generated: [^\n]+",
                    "Generated: <timestamp>",
                    serial_path.read_text(),
                )


def test_parent_write_failure_keeps_healthy_selected_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "models"
    source.mkdir()
    exemplar = Path(__file__).parents[2] / "input/gnn_files/discrete/simple_mdp.md"
    for name in ("first", "second"):
        (source / f"{name}.md").write_bytes(exemplar.read_bytes())
    output = tmp_path / "output"
    context = build_run_context(source, output, "parent-write", [3], {})
    context_file = tmp_path / "context.json"
    context.write(context_file)
    monkeypatch.setenv("GNN_RUN_CONTEXT_FILE", str(context_file))
    monkeypatch.setenv("GNN_RUN_ID", context.run_id)
    write = Path.write_bytes

    def fail_first(path: Path, *args: Any, **kwargs: Any) -> int:
        if path.name == "first_parsed.json":
            raise OSError("injected parent write failure")
        return write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_bytes", fail_first)
    assert not process_gnn_multi_format(
        source,
        output,
        logging.getLogger(__name__),
        serialize_preset="minimal",
        file_workers=2,
    )
    receipt = json.loads(
        (output / "3_gnn_output/gnn_processing_results.json").read_text()
    )
    assert (
        receipt["summary"]["successful_parses"]
        == receipt["summary"]["failed_parses"]
        == 1
    )
    assert len(receipt["processed_files"]) == 2
    failed, healthy = receipt["processed_files"]
    assert (
        not failed["parse_success"]
        and "injected parent write failure" in failed["error"]
    )
    assert healthy["parse_success"] and Path(healthy["parsed_model_file"]).is_file()
    assert {row["model_id"] for row in receipt["processed_files"]} == {
        model.model_id for model in context.models
    }


@pytest.mark.parametrize("requested,process_capacity", [(61, 61), (62, 61), (256, 61)])
def test_windows_process_capacity_preserves_requests_and_thread_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: int,
    process_capacity: int,
) -> None:
    import gnn.utils.file_workers as workers

    capacities: list[int] = []

    class InlineExecutor:
        """Exercise admission without spawning a platform-sized native pool."""

        def __init__(self, max_workers: int, **_kwargs: Any) -> None:
            capacities.append(max_workers)

        def submit(self, function: Any, *args: Any) -> Future:
            future: Future = Future()
            try:
                future.set_result(function(*args))
            except Exception as exc:
                future.set_exception(exc)
            return future

        def shutdown(self, **_kwargs: Any) -> None:
            pass

    monkeypatch.setattr(workers, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(workers, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(workers, "ThreadPoolExecutor", InlineExecutor)
    assert workers.resolve_file_workers(requested, "process") == (requested, "process")
    for mode, capacity in (("process", process_capacity), ("thread", requested)):
        metadata = {"requested_workers": requested, "effective_workers": requested}
        assert list(
            workers.ordered_file_map(
                abs, range(300), workers=requested, executor=mode, metadata=metadata
            )
        ) == list(range(300))
        assert capacities[-1] == capacity
        assert metadata["requested_workers"] == requested
        assert metadata["effective_workers"] == capacity
        assert metadata["executor"] == mode
        failed_metadata = {
            "requested_workers": requested,
            "effective_workers": requested,
        }
        with pytest.raises(TimeoutError, match="job timeout"):
            list(
                workers.ordered_file_map(
                    _job_timeout,
                    [0],
                    workers=requested,
                    executor=mode,
                    metadata=failed_metadata,
                )
            )
        assert failed_metadata["requested_workers"] == requested
        assert failed_metadata["effective_workers"] == capacity
        assert failed_metadata["executor"] == mode
    empty = process_gnn_directory_lightweight(tmp_path, workers=requested)
    assert empty["success"]
    assert empty["worker_execution"] == {
        "requested_workers": requested,
        "effective_workers": 0,
        "executor": "serial",
        "worker_pids": [],
    }
    (tmp_path / "one.md").write_text(
        "## ModelName\nOne\n## StateSpaceBlock\nx: float\nx = 1\n"
    )
    one = process_gnn_directory_lightweight(tmp_path, workers=requested)
    assert one["success"]
    assert one["worker_execution"]["requested_workers"] == requested
    assert one["worker_execution"]["effective_workers"] == 1
    assert one["worker_execution"]["executor"] == "serial"
    assert one["worker_execution"]["worker_pids"] == [os.getpid()]

    if requested == 62:
        from gnn.processing.profiling import profile_directory

        selected = tmp_path / "many_models"
        selected.mkdir()
        for index in range(62):
            (selected / f"model_{index}.md").write_text(
                "## ModelName\nMany\n## StateSpaceBlock\nx: float\nx = 1\n"
            )
        report = profile_directory(
            selected, tmp_path / "profile", repeats=1, workers=256
        )
        assert report["controls"]["requested_workers"] == 256
        assert report["controls"]["effective_workers"] == 61
        assert report["worker_execution"]["requested_workers"] == 256
        assert report["worker_execution"]["effective_workers"] == 61
        assert (
            report["unprofiled_samples"][0]["worker_execution"]["effective_workers"]
            == 61
        )
        (selected / "bad.md").write_bytes(b"\xff")
        with pytest.raises(ValueError, match="did not complete"):
            profile_directory(
                selected, tmp_path / "failed_profile", repeats=1, workers=256
            )
        failed = json.loads(
            (tmp_path / "failed_profile/profile_report.json").read_text()
        )
        assert failed["status"] == "failed"
        assert failed["controls"]["requested_workers"] == 256
        assert failed["controls"]["effective_workers"] == 61
