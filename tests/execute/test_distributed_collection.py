"""Per-task receipts and bounded transfers on real optional cluster backends."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from gnn.execute.distributed import Dispatcher
from gnn.execute.processor.envelope import _bind_dispatcher_failure


def test_remote_failure_keeps_script_identity() -> None:
    info = {
        "path": "/render/model/jax/model_jax.py",
        "name": "model_jax.py",
        "framework": "jax",
        "executor": "python",
    }
    result = _bind_dispatcher_failure(
        info,
        {
            "success": False,
            "error": "boom",
            "error_type": "DistributedTaskError",
            "exception_type": "ValueError",
        },
        "dask",
        3,
    )
    assert result["model_name"] == "model" and result["framework"] == "jax"
    assert result["script_path"] == info["path"]
    assert result["error_type"] == "DistributedTaskError"
    assert result["dispatch_error_type"] == "ValueError"


def test_dispatcher_intersects_explicit_and_invocation_deadlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gnn.pipeline import run_context

    now = time.monotonic()
    monkeypatch.setattr(
        run_context,
        "current_run_context",
        lambda: SimpleNamespace(deadline_monotonic=now + 0.1),
    )
    deadline, budget = Dispatcher(
        "dask", deadline_monotonic=now + 1
    )._collection_deadline()
    assert deadline == now + 0.1
    assert 0 < budget <= 0.1


def test_boolean_deadline_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        Dispatcher("dask", deadline_monotonic=True)


def test_initialization_failure_does_not_execute_unbounded_sequential_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = Dispatcher("dask")
    monkeypatch.setattr(dispatcher, "connect_to_cluster", lambda: False)
    calls = []
    results = dispatcher.run_scripts_parallel(
        [{"id": 1}, {"id": 2}], lambda *args: calls.append(args)
    )
    sweep = dispatcher.parameter_sweep(
        lambda **kwargs: calls.append(kwargs), [{"x": 1}]
    )
    assert not calls
    assert len(results) == 2 and len(sweep) == 1
    assert all(
        result["error_type"] == "DistributedInitializationError"
        for result in results + sweep
    )


@pytest.mark.needs_distributed
def test_direct_dispatcher_requires_ready_caller_owned_client(
    cluster: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.pipeline import run_context

    monkeypatch.setattr(run_context, "current_run_context", lambda: None)
    assert Dispatcher("dask").connect_to_cluster() is False
    client, local = cluster
    dispatcher = Dispatcher("dask", client=client)
    assert dispatcher.connect_to_cluster()
    assert dispatcher.parameter_sweep(lambda value: value * 2, [{"value": 3}]) == [6]
    dispatcher.shutdown()
    from distributed.core import Status

    assert local.status == Status.running and client.status == "running"


@pytest.mark.needs_distributed
def test_owned_dask_cluster_closes_without_extending_expired_deadline(
    cluster: Any,
) -> None:
    from distributed.core import Status

    client, local = cluster
    dispatcher = Dispatcher("dask", deadline_monotonic=time.monotonic() - 1)
    dispatcher.client = client
    dispatcher._owned_cluster = local
    dispatcher._owns_client = True
    dispatcher._initialized = True
    started = time.monotonic()
    dispatcher.shutdown()
    assert time.monotonic() - started < 0.2
    deadline = time.monotonic() + 3
    while local.status != Status.closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert local.status == Status.closed
    assert client.status == "closed"


def test_preexisting_ray_runtime_is_not_owned(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    runtime = types.ModuleType("ray")
    closed = []
    runtime.shutdown = lambda: closed.append(True)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", runtime)
    dispatcher = Dispatcher("ray")
    dispatcher._initialized = True
    dispatcher.shutdown()
    assert not closed and not dispatcher._initialized


@pytest.fixture
def cluster() -> Any:
    from distributed import Client, LocalCluster

    local = LocalCluster(
        n_workers=2, threads_per_worker=2, processes=False, dashboard_address=None
    )
    client = Client(local, set_as_default=False)
    try:
        yield client, local
    finally:
        client.close(timeout=5)
        local.close(timeout=5)


@pytest.mark.needs_distributed
def test_real_dask_mixed_results_preserve_siblings_and_order(
    cluster: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dask.distributed
    from distributed.client import FuturesCancelledError

    cancelled_waits = []
    real_wait = dask.distributed.wait

    def record_wait(*args: Any, **kwargs: Any) -> Any:
        try:
            return real_wait(*args, **kwargs)
        except FuturesCancelledError:
            cancelled_waits.append(True)
            raise

    monkeypatch.setattr(dask.distributed, "wait", record_wait)
    client, _ = cluster
    dispatcher = Dispatcher("dask", deadline_monotonic=time.monotonic() + 0.6)
    dispatcher.client = client

    def work(name: str) -> dict[str, Any]:
        if name == "bad":
            raise ValueError("bad-model")
        if name in {"slow", "cancel"}:
            time.sleep(2)
        return {"success": True, "name": name}

    futures = [
        client.submit(work, name, pure=False)
        for name in ("fast", "bad", "cancel", "slow")
    ]
    client.loop.call_later(0.05, client.cancel, [futures[2]], asynchronous=True)
    start = time.monotonic()
    result = dispatcher._dask_gather_bounded(futures)
    assert time.monotonic() - start < 1.2
    assert result[0] == {"success": True, "name": "fast"}
    assert (
        result[1]["error_type"] == "DistributedTaskError"
        and result[1]["exception_type"] == "ValueError"
    )
    assert result[2]["error_type"] == "DistributedTaskCancelled"
    assert cancelled_waits, (
        "real wait must exercise Dask's typed cancellation exception"
    )
    assert result[3]["error_type"] == "DistributedWaitTimeout"
    deadline = time.monotonic() + 1
    while futures[3].status != "cancelled" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert futures[3].status == "cancelled"


@pytest.mark.needs_distributed
def test_remote_timeout_exception_is_not_collection_timeout(cluster: Any) -> None:
    client, _ = cluster
    dispatcher = Dispatcher("dask", deadline_monotonic=time.monotonic() + 1)
    dispatcher.client = client

    def fail() -> None:
        raise TimeoutError("remote model failed")

    result = dispatcher._dask_gather_bounded([client.submit(fail)])
    assert result[0]["error_type"] == "DistributedTaskError"
    assert result[0]["exception_type"] == "TimeoutError"


@pytest.mark.needs_distributed
def test_real_dask_retries_exhaustion_and_parameter_sweep(
    cluster: Any, tmp_path: Path
) -> None:
    client, _ = cluster
    dispatcher = Dispatcher("dask", max_retries=1)
    dispatcher.client, dispatcher._initialized = client, True

    def model(value: int) -> int:
        if value < 0:
            raise RuntimeError("exhausted")
        path = tmp_path / str(value)
        if not path.exists():
            path.write_text("retry")
            raise ValueError("retry once")
        return value * 2

    result = dispatcher.parameter_sweep(
        model, [{"value": 2}, {"value": -1}, {"value": 3}]
    )
    assert result[0] == 4 and result[2] == 6
    assert result[1]["error_type"] == "DistributedTaskError"


@pytest.mark.needs_distributed
def test_real_dask_recovers_after_worker_data_loss(cluster: Any) -> None:
    from distributed import wait

    client, local = cluster
    worker = next(iter(local.workers.values()))
    future = client.submit(
        lambda: {"success": True, "value": 42},
        workers=[worker.address],
        allow_other_workers=True,
        pure=False,
    )
    with client.as_current():
        wait([future], timeout=5)
    local.sync(worker.close)
    dispatcher = Dispatcher("dask", deadline_monotonic=time.monotonic() + 3)
    dispatcher.client = client
    assert dispatcher._dask_gather_bounded([future]) == [{"success": True, "value": 42}]


@pytest.mark.needs_distributed
def test_dask_transport_failure_preserves_finished_sibling(
    cluster: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dask.distributed

    client, _ = cluster
    fast = client.submit(lambda: 42, pure=False)
    assert fast.result(timeout=3) == 42
    slow = client.submit(time.sleep, 2, pure=False)

    def broken_wait(*args: Any, **kwargs: Any) -> None:
        raise OSError("scheduler transport unavailable")

    monkeypatch.setattr(dask.distributed, "wait", broken_wait)
    dispatcher = Dispatcher("dask", client=client)
    result = dispatcher._dask_gather_bounded([fast, slow])
    assert result[0] == 42
    assert result[1]["error_type"] == "DistributedTransportError"


@pytest.mark.needs_distributed
def test_dask_submit_channel_failure_preserves_prior_submissions(cluster: Any) -> None:
    client, _ = cluster

    class ClientProxy:
        submissions = 0

        def __getattr__(self, name: str) -> Any:
            return getattr(client, name)

        def submit(self, *args: Any, **kwargs: Any) -> Any:
            self.submissions += 1
            if self.submissions == 2:
                raise OSError("channel closed during submit")
            return client.submit(*args, **kwargs)

    dispatcher = Dispatcher("dask", client=ClientProxy())
    result = dispatcher.parameter_sweep(
        lambda value: value * 2, [{"value": 2}, {"value": 3}, {"value": 4}]
    )
    assert result[0] == 4 and result[2] == 8
    assert result[1]["error_type"] == "DistributedTransportError"


def test_ray_wait_channel_failure_preserves_retrieved_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    class RayError(Exception):
        pass

    class GetTimeoutError(RayError):
        pass

    class TaskCancelledError(RayError):
        pass

    calls = []
    cancelled = []

    def wait(futures: Any, **kwargs: Any) -> Any:
        calls.append(True)
        if len(calls) == 1:
            return ["fast"], ["pending"]
        raise RayError("typed channel loss")

    runtime = types.ModuleType("ray")
    runtime.wait = wait  # type: ignore[attr-defined]
    runtime.get = lambda ref, **kwargs: 42  # type: ignore[attr-defined]
    runtime.cancel = lambda ref, **kwargs: cancelled.append(ref)  # type: ignore[attr-defined]
    runtime.exceptions = SimpleNamespace(
        RayError=RayError,
        GetTimeoutError=GetTimeoutError,
        TaskCancelledError=TaskCancelledError,
    )  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", runtime)
    result = Dispatcher("ray")._ray_get_bounded(["fast", "pending"])
    assert result[0] == 42
    assert result[1]["error_type"] == "DistributedTransportError"
    assert cancelled == ["pending"]


@pytest.mark.needs_distributed
def test_result_transfer_is_bounded_and_does_not_starve_sibling(
    cluster: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from distributed import Future, wait

    client, _ = cluster
    stalled = client.submit(lambda: 1, pure=False)
    fast = client.submit(lambda: 2, pure=False)
    with client.as_current():
        wait([stalled, fast], timeout=5)
    real_result = Future.result
    received_timeouts = []

    def result(future: Any, timeout: float | None = None) -> Any:
        if future is stalled:
            assert timeout is not None and timeout > 0
            received_timeouts.append(timeout)
            time.sleep(timeout)
            raise TimeoutError("injected transfer stall")
        return real_result(future, timeout=timeout)

    monkeypatch.setattr(Future, "result", result)
    dispatcher = Dispatcher("dask", deadline_monotonic=time.monotonic() + 0.3)
    dispatcher.client = client
    started = time.monotonic()
    results = dispatcher._dask_gather_bounded([stalled, fast])
    assert time.monotonic() - started < 0.7
    assert received_timeouts and max(received_timeouts) <= 0.1
    assert results[0]["error_type"] == "DistributedWaitTimeout"
    assert results[1] == 2


@pytest.mark.needs_distributed
def test_worker_loss_without_recovery_expires_instead_of_being_terminal(
    cluster: Any,
) -> None:
    from distributed import wait

    client, local = cluster
    future = client.submit(lambda: 42, pure=False)
    with client.as_current():
        wait([future], timeout=5)
    for worker in tuple(local.workers.values()):
        local.sync(worker.close)
    dispatcher = Dispatcher("dask", deadline_monotonic=time.monotonic() + 0.2)
    dispatcher.client = client
    results = dispatcher._dask_gather_bounded([future])
    assert results[0]["error_type"] == "DistributedWaitTimeout"


def _isolated_cluster(
    script: str, tmp_path: Path, timeout: float = 60
) -> dict[str, Any]:
    """Bound startup and task execution, cleaning only this subprocess's fleet."""
    short_directory = tempfile.TemporaryDirectory(prefix="gnnr-", dir="/tmp")
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
    }
    process = subprocess.Popen(
        [sys.executable, "-c", script, short_directory.name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )
    descendants: set[psutil.Process] = set()
    deadline = time.monotonic() + timeout
    try:
        while True:
            with contextlib.suppress(psutil.Error):
                descendants.update(psutil.Process(process.pid).children(recursive=True))
            try:
                stdout, stderr = process.communicate(
                    timeout=min(0.5, max(0.01, deadline - time.monotonic()))
                )
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    pytest.fail("isolated cluster exceeded startup/execution deadline")
        assert process.returncode == 0, stderr[-4000:]
        receipt = [
            line for line in stdout.splitlines() if line.startswith("GNN_RECEIPT=")
        ]
        assert receipt, (stdout[-2000:], stderr[-2000:])
        return json.loads(receipt[-1].split("=", 1)[1])
    finally:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        for child in descendants:
            with contextlib.suppress(psutil.Error):
                child.kill()
        _, alive = psutil.wait_procs(tuple(descendants), timeout=2)
        process.communicate(timeout=5)
        short_directory.cleanup()
        survivors = []
        for child in alive:
            with contextlib.suppress(psutil.Error):
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    survivors.append(child.pid)
        assert not survivors, f"Owned cluster descendants survived cleanup: {survivors}"


@pytest.mark.needs_ray
@pytest.mark.needs_posix
def test_real_ray_mixed_results_and_bounded_cancellation(tmp_path: Path) -> None:
    receipt = _isolated_cluster(
        """
import json, sys, time, ray
from gnn.execute.distributed import Dispatcher
ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False, _temp_dir=sys.argv[1])
try:
    @ray.remote(max_retries=0)
    def task(name):
        if name == "bad": raise ValueError("bad-model")
        if name in ("slow", "cancel"): time.sleep(5)
        return {"success": True, "name": name}
    fast = task.remote("fast")
    ray.get(fast, timeout=15)
    bad, cancelled, slow = (task.remote(name) for name in ("bad", "cancel", "slow"))
    ray.cancel(cancelled)
    dispatcher = Dispatcher("ray", deadline_monotonic=time.monotonic() + 1)
    start = time.monotonic()
    result = dispatcher._ray_get_bounded([fast, bad, cancelled, slow])
    print("GNN_RECEIPT=" + json.dumps({"results": result, "elapsed": time.monotonic()-start}))
finally:
    ray.shutdown()
""",
        tmp_path,
    )
    result = receipt["results"]
    assert receipt["elapsed"] < 2
    assert result[0] == {"success": True, "name": "fast"}
    assert result[1]["error_type"] == "DistributedTaskError"
    assert result[2]["error_type"] == "DistributedTaskCancelled"
    assert result[3]["error_type"] == "DistributedWaitTimeout"
