"""Tests for ``execute.executor.execute_script_safely``.

Exercises the envelope contract: the function must always return a dict with
``success`` and never raise, regardless of the failure mode.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.execute import execute_script_safely  # noqa: E402

REQUIRED_KEYS: set[Any] = {
    "success",
    "script_path",
    "return_code",
    "stdout",
    "stderr",
    "duration_seconds",
}


def _assert_envelope(result: dict) -> None:
    assert isinstance(result, dict)
    assert REQUIRED_KEYS.issubset(result.keys()), (
        f"missing keys: {REQUIRED_KEYS - result.keys()}"
    )
    assert isinstance(result["success"], bool)
    assert isinstance(result["return_code"], int)
    assert isinstance(result["stdout"], str)
    assert isinstance(result["stderr"], str)
    assert isinstance(result["duration_seconds"], float)


def test_missing_file_returns_failure_envelope(tmp_path: Path) -> None:
    result = execute_script_safely(tmp_path / "does_not_exist.py")
    _assert_envelope(result)
    assert result["success"] is False
    assert result["error_type"] == "FileNotFoundError"


def test_non_python_suffix_rejected(tmp_path: Path) -> None:
    shell_script = tmp_path / "hello.sh"
    shell_script.write_text("echo hello\n")
    result = execute_script_safely(shell_script)
    _assert_envelope(result)
    assert result["success"] is False
    assert result["error_type"] == "ValueError"


def test_successful_script(tmp_path: Path) -> None:
    script = tmp_path / "ok.py"
    script.write_text("print('hello-from-script')\n")
    result = execute_script_safely(script, timeout=30)
    _assert_envelope(result)
    assert result["success"] is True
    assert result["return_code"] == 0
    assert "hello-from-script" in result["stdout"]


def test_failing_script(tmp_path: Path) -> None:
    script = tmp_path / "fail.py"
    script.write_text("import sys; sys.exit(3)\n")
    result = execute_script_safely(script, timeout=30)
    _assert_envelope(result)
    assert result["success"] is False
    assert result["return_code"] == 3


def test_timeout(tmp_path: Path) -> None:
    script = tmp_path / "hang.py"
    script.write_text("import time; time.sleep(5)\n")
    result = execute_script_safely(script, timeout=1)
    _assert_envelope(result)
    assert result["success"] is False
    assert result["error_type"] == "TimeoutExpired"


def test_env_override_merges(tmp_path: Path) -> None:
    script = tmp_path / "env.py"
    script.write_text("import os; print(os.environ.get('GNN_TEST_VAR', 'MISSING'))\n")
    result = execute_script_safely(script, timeout=30, env={"GNN_TEST_VAR": "present"})
    _assert_envelope(result)
    assert result["success"] is True
    assert "present" in result["stdout"]


# --- Phase 1.1 regression: silent-success in process_execute ---------------


def test_process_execute_returns_2_when_no_render_output(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Before Phase 1.1, process_execute returned True when the render output
    directory was missing — the pipeline reported step 12 as successful while
    silently skipping all work. After the fix it returns 2 ("skipped/warnings")
    so the widened pipeline_template contract can surface this as a warning.

    Isolation note: ``_resolve_render_output_dir`` searches CWD-relative
    ``Path("output")`` as a fallback, so we chdir into an isolated tmp dir
    before invoking the processor; otherwise the real project's render output
    would be discovered and executed.
    """
    from gnn.execute.processor import process_execute

    monkeypatch.chdir(tmp_path)  # isolate from real project output/
    empty_target = tmp_path / "no_render_output_here"
    empty_target.mkdir()
    output_dir = tmp_path / "execute_output"
    result = process_execute(
        target_dir=empty_target,
        output_dir=output_dir,
        verbose=False,
        frameworks="all",
        # Default flipped to True; this test pins the exit-2 contract for
        # callers that opt out of the render-summary requirement.
        require_render_summary=False,
    )
    # Per the new contract, "nothing to do" must be exit-code 2, not True.
    assert result == 2, f"Expected exit-code 2 for empty render output; got {result!r}"


def test_process_execute_records_local_worker_configuration(tmp_path: Path) -> None:
    """Step 12 should expose bounded local script-level concurrency."""
    from gnn.execute.processor import process_execute

    render_out = tmp_path / "render" / "11_render_output"
    for model_name in ("model_a", "model_b"):
        script_dir = render_out / model_name / "pymdp"
        script_dir.mkdir(parents=True)
        (script_dir / f"{model_name}_pymdp.py").write_text(
            "print('worker-ok')\n",
            encoding="utf-8",
        )

    output_dir = tmp_path / "execute_output"
    result = process_execute(
        target_dir=tmp_path / "input",
        output_dir=output_dir,
        verbose=False,
        frameworks="pymdp",
        timeout=10,
        render_output_dir=render_out,
        execution_workers=2,
        require_render_summary=False,
    )

    assert result is True
    summary = json.loads(
        (output_dir / "summaries" / "execution_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["execution_mode"] == "local"
    assert summary["execution_workers"] == 2
    assert summary["total_scripts_found"] == 2


def test_process_execute_records_local_worker_pool_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken local process pool should become per-script failures."""
    from gnn.execute import processor

    render_out = tmp_path / "render" / "11_render_output"
    for model_name in ("model_a", "model_b"):
        script_dir = render_out / model_name / "pymdp"
        script_dir.mkdir(parents=True)
        (script_dir / f"{model_name}_pymdp.py").write_text(
            "print('worker-ok')\n",
            encoding="utf-8",
        )

    class BrokenPool:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> "BrokenPool":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def map(self, *args: object, **kwargs: object) -> list[dict[str, object]]:
            raise RuntimeError("pool exploded")

    monkeypatch.setattr(processor, "ProcessPoolExecutor", BrokenPool)

    output_dir = tmp_path / "execute_output"
    result = processor.process_execute(
        target_dir=tmp_path / "input",
        output_dir=output_dir,
        verbose=False,
        frameworks="pymdp",
        timeout=10,
        render_output_dir=render_out,
        execution_workers=2,
        require_render_summary=False,
    )

    assert result is False
    summary = json.loads(
        (output_dir / "summaries" / "execution_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["success"] is False
    assert summary["status"] == "failed"
    assert summary["exit_code"] == 1
    assert summary["outcome_reason"] == "requested_framework_execution_incomplete"
    assert summary["failed_executions"] == 2
    assert {detail["error_type"] for detail in summary["execution_details"]} == {
        "LocalWorkerPoolError"
    }


def test_required_render_summary_fails_closed_and_is_recorded(tmp_path: Path) -> None:
    from gnn.execute.processor import process_execute

    render_out = tmp_path / "render" / "11_render_output"
    script_dir = render_out / "model" / "pymdp"
    script_dir.mkdir(parents=True)
    (script_dir / "model_pymdp.py").write_text("print('ok')\n", encoding="utf-8")
    output_dir = tmp_path / "execute_output"

    result = process_execute(
        target_dir=tmp_path / "input",
        output_dir=output_dir,
        frameworks="pymdp",
        render_output_dir=render_out,
        require_render_summary=True,
    )

    assert result is False
    summary = json.loads(
        (output_dir / "summaries" / "execution_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["success"] is False
    assert summary["status"] == "failed"
    assert summary["outcome_reason"] == "required_render_summary_missing"
    assert summary["total_scripts_found"] == 0


def test_distributed_dispatch_failure_becomes_per_script_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.execute import distributed, processor

    render_out = tmp_path / "render" / "11_render_output"
    script_dir = render_out / "model" / "pymdp"
    script_dir.mkdir(parents=True)
    (script_dir / "model_pymdp.py").write_text("print('ok')\n", encoding="utf-8")

    class BrokenDispatcher:
        shutdown_called = False

        def __init__(self, **kwargs: object) -> None:
            assert kwargs["max_retries"] == 4

        def run_scripts_parallel(self, *args: object, **kwargs: object) -> list[dict]:
            raise RuntimeError("cluster vanished")

        def shutdown(self) -> None:
            BrokenDispatcher.shutdown_called = True

    monkeypatch.setattr(distributed, "Dispatcher", BrokenDispatcher)
    output_dir = tmp_path / "execute_output"
    result = processor.process_execute(
        target_dir=tmp_path / "input",
        output_dir=output_dir,
        frameworks="pymdp",
        render_output_dir=render_out,
        distributed=True,
        backend="ray",
        distributed_max_retries=4,
        require_render_summary=False,
    )

    assert result is False
    assert BrokenDispatcher.shutdown_called is True
    summary = json.loads(
        (output_dir / "summaries" / "execution_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["dispatch_max_retries"] == 4
    assert summary["failed_executions"] == 1
    assert summary["execution_details"][0]["error_type"] == "DistributedDispatchError"


@pytest.mark.parametrize("framework", ["rxinfer", "activeinference_jl"])
def test_julia_execution_command_pins_committed_project(framework: str) -> None:
    from gnn.execute.processor import (
        ScriptExecutionContext,
        _build_script_execution_command,
    )

    context = ScriptExecutionContext(
        script_path=Path("/tmp/model") / framework / "model.jl",
        script_name="model.jl",
        framework=framework,
        model_name="model",
        executor="julia",
    )
    command = _build_script_execution_command(context, ["sandbox"])

    expected_project = (
        Path(__file__).resolve().parents[2] / "src" / "gnn" / "execute" / framework
    )
    assert command == [
        "sandbox",
        "julia",
        "--startup-file=no",
        f"--project={expected_project}",
        "model.jl",
    ]


def test_rxinfer_manifest_pins_5_5_0() -> None:
    manifest = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "gnn"
        / "execute"
        / "rxinfer"
        / "Manifest.toml"
    ).read_text(encoding="utf-8")
    assert "[[deps.RxInfer]]" in manifest
    rxinfer_section = manifest.split("[[deps.RxInfer]]", maxsplit=1)[1].split(
        "[[deps.", maxsplit=1
    )[0]
    assert 'version = "5.5.0"' in rxinfer_section


def test_dask_dispatch_without_distributed_applies_configured_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without dask.distributed importable the dispatcher gathers unbounded."""
    from gnn.execute.distributed import Dispatcher

    monkeypatch.setitem(sys.modules, "dask.distributed", None)
    submitted: list[dict[str, object]] = []

    class FakeClient:
        def submit(self, fn: object, info: object, **kwargs: object) -> object:
            submitted.append(kwargs)
            return {"info": info}

        def gather(self, futures: list[object]) -> list[dict[str, object]]:
            return [{"success": True} for _ in futures]

    dispatcher = Dispatcher(backend="dask", max_retries=5)
    dispatcher._initialized = True
    dispatcher.client = FakeClient()
    results = dispatcher.run_scripts_parallel(
        [{"name": "one"}, {"name": "two"}],
        lambda info, **kwargs: {"success": True},
        timeout=1,
    )

    assert len(results) == 2
    assert [item["retries"] for item in submitted] == [5, 5]
    assert [item["timeout"] for item in submitted] == [1, 1]


@pytest.fixture
def dask_client() -> Any:
    """In-process Dask client that is deliberately NOT the global default."""
    import distributed

    cluster = distributed.LocalCluster(
        n_workers=1,
        threads_per_worker=2,
        processes=False,
        protocol="inproc",
        dashboard_address=None,
    )
    client = distributed.Client(cluster, set_as_default=False)
    try:
        yield client
    finally:
        client.close()
        cluster.close()


@pytest.mark.needs_distributed
def test_dask_dispatch_applies_configured_retries(dask_client: Any) -> None:
    from gnn.execute.distributed import Dispatcher

    submitted: list[dict[str, object]] = []
    real_submit = dask_client.submit

    def recording_submit(fn: Any, *args: Any, **kwargs: Any) -> Any:
        submitted.append(kwargs)
        return real_submit(fn, *args, **kwargs)

    dask_client.submit = recording_submit
    dispatcher = Dispatcher(backend="dask", max_retries=5)
    dispatcher._initialized = True
    dispatcher.client = dask_client

    def execute(info: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return {"success": True, "name": info["name"], "timeout": kwargs["timeout"]}

    results = dispatcher.run_scripts_parallel(
        [{"name": "one"}, {"name": "two"}], execute, timeout=1
    )

    assert results == [
        {"success": True, "name": "one", "timeout": 1},
        {"success": True, "name": "two", "timeout": 1},
    ]
    assert [item["retries"] for item in submitted] == [5, 5]
    assert [item["timeout"] for item in submitted] == [1, 1]


@pytest.mark.needs_distributed
def test_dask_dispatch_reports_wait_timeout_as_failure(
    dask_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from gnn.execute.distributed import WAIT_TIMEOUT_ENV, Dispatcher

    monkeypatch.setenv(WAIT_TIMEOUT_ENV, "1")
    dispatcher = Dispatcher(backend="dask", max_retries=0)
    dispatcher._initialized = True
    dispatcher.client = dask_client

    def execute(info: dict[str, Any]) -> dict[str, Any]:
        if info["name"] == "slow":
            time.sleep(4)
        return {"success": True, "name": info["name"]}

    results = dispatcher.run_scripts_parallel(
        [{"name": "fast"}, {"name": "slow"}], execute
    )

    assert results[0] == {"success": True, "name": "fast"}
    assert results[1]["success"] is False
    assert results[1]["error_type"] == "DistributedWaitTimeout"


class _FlippingFuture:
    """Future whose status reads ``pending`` once, then ``finished``.

    Models a future completing between two status reads after the wait expired.
    """

    def __init__(self, name: str, statuses: list[str]) -> None:
        self.name = name
        self._statuses = statuses
        self.cancelled = False

    @property
    def status(self) -> str:
        return self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]

    def cancel(self) -> None:
        self.cancelled = True


class _FakeDaskClient:
    def as_current(self) -> Any:
        import contextlib

        return contextlib.nullcontext()

    def gather(self, futures: list[Any]) -> list[dict[str, Any]]:
        return [{"success": True, "name": fut.name} for fut in futures]


def test_dask_gather_classifies_each_future_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A future finishing mid-classification is gathered or cancelled, never dropped.

    With two separate status scans a future read ``pending`` by the first and
    ``finished`` by the second lands in neither bucket: it is reported as a
    ``DistributedWaitTimeout`` although it was never cancelled.
    """
    import types

    from gnn.execute.distributed import WAIT_TIMEOUT_ENV, Dispatcher

    def expired_wait(futures: Any, timeout: Any = None) -> None:
        raise TimeoutError

    fake_distributed = types.ModuleType("dask.distributed")
    fake_distributed.wait = expired_wait  # type: ignore[attr-defined]
    fake_dask = types.ModuleType("dask")
    fake_dask.distributed = fake_distributed  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dask", fake_dask)
    monkeypatch.setitem(sys.modules, "dask.distributed", fake_distributed)
    monkeypatch.setenv(WAIT_TIMEOUT_ENV, "1")

    dispatcher = Dispatcher(backend="dask", max_retries=0)
    dispatcher.client = _FakeDaskClient()
    fast = _FlippingFuture("fast", ["finished"])
    flipping = _FlippingFuture("flipping", ["pending", "finished"])
    stuck = _FlippingFuture("stuck", ["pending"])

    results = dispatcher._dask_gather_bounded([fast, flipping, stuck])

    assert results[0] == {"success": True, "name": "fast"}
    assert results[2]["error_type"] == "DistributedWaitTimeout"
    assert stuck.cancelled
    for fut, result in zip((fast, flipping, stuck), results):
        timed_out = result.get("error_type") == "DistributedWaitTimeout"
        assert timed_out == fut.cancelled, (
            f"{fut.name}: reported timed out={timed_out} but cancelled={fut.cancelled}"
        )


class _ErroredFuture(_FlippingFuture):
    """Future that finished with a remote exception (dask status ``error``)."""

    def __init__(self, name: str, exc: BaseException) -> None:
        super().__init__(name, ["error"])
        self._exc = exc

    def exception(self) -> BaseException:
        return self._exc


class _FinishedOnlyGatherClient(_FakeDaskClient):
    """Gather that raises like dask when handed a non-finished future."""

    def gather(self, futures: list[Any]) -> list[dict[str, Any]]:
        for fut in futures:
            if fut._statuses[0] != "finished":
                raise RuntimeError(f"gather raised for {fut.name}")
        return super().gather(futures)


@pytest.mark.parametrize("with_pending", [False, True])
def test_dask_gather_isolates_errored_futures(
    monkeypatch: pytest.MonkeyPatch, with_pending: bool
) -> None:
    """One remotely failed future becomes one failure record, not a batch raise."""
    import types

    from gnn.execute.distributed import WAIT_TIMEOUT_ENV, Dispatcher

    def wait(futures: Any, timeout: Any = None) -> None:
        if with_pending:
            raise TimeoutError

    fake_distributed = types.ModuleType("dask.distributed")
    fake_distributed.wait = wait  # type: ignore[attr-defined]
    fake_dask = types.ModuleType("dask")
    fake_dask.distributed = fake_distributed  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dask", fake_dask)
    monkeypatch.setitem(sys.modules, "dask.distributed", fake_distributed)
    monkeypatch.setenv(WAIT_TIMEOUT_ENV, "1")

    dispatcher = Dispatcher(backend="dask", max_retries=0)
    dispatcher.client = _FinishedOnlyGatherClient()
    futures: list[Any] = [
        _FlippingFuture("ok", ["finished"]),
        _ErroredFuture("bad", ValueError("model diverged")),
    ]
    if with_pending:
        futures.append(_FlippingFuture("stuck", ["pending"]))

    results = dispatcher._dask_gather_bounded(futures)

    assert results[0] == {"success": True, "name": "ok"}
    assert results[1]["success"] is False
    assert results[1]["error_type"] == "DistributedTaskError"
    assert results[1]["exception_type"] == "ValueError"
    assert "model diverged" in results[1]["error"]
    assert not futures[1].cancelled
    if with_pending:
        assert results[2]["error_type"] == "DistributedWaitTimeout"
        assert futures[2].cancelled


@pytest.mark.needs_distributed
def test_dask_dispatch_reports_remote_exception_per_script(dask_client: Any) -> None:
    from gnn.execute.distributed import Dispatcher

    dispatcher = Dispatcher(backend="dask", max_retries=0)
    dispatcher._initialized = True
    dispatcher.client = dask_client

    def execute(info: dict[str, Any]) -> dict[str, Any]:
        if info["name"] == "bad":
            raise ValueError("model diverged")
        return {"success": True, "name": info["name"]}

    results = dispatcher.run_scripts_parallel(
        [{"name": "ok"}, {"name": "bad"}], execute
    )

    assert results[0] == {"success": True, "name": "ok"}
    assert results[1]["success"] is False
    assert results[1]["error_type"] == "DistributedTaskError"
    assert results[1]["exception_type"] == "ValueError"


def test_bare_dispatcher_failure_is_bound_to_its_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-task dispatcher failures keep the script's identity and framework."""
    from gnn.execute import distributed, processor

    render_out = tmp_path / "render" / "11_render_output"
    script_dir = render_out / "model" / "pymdp"
    script_dir.mkdir(parents=True)
    script = script_dir / "model_pymdp.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    class TimingOutDispatcher:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run_scripts_parallel(
            self, infos: list[dict], *args: object, **kwargs: object
        ) -> list[dict]:
            return [
                {
                    "success": False,
                    "error": "remote boom",
                    "error_type": "DistributedTaskError",
                    "exception_type": "ValueError",
                }
                for _ in infos
            ]

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(distributed, "Dispatcher", TimingOutDispatcher)
    output_dir = tmp_path / "execute_output"
    result = processor.process_execute(
        target_dir=tmp_path / "input",
        output_dir=output_dir,
        frameworks="pymdp",
        render_output_dir=render_out,
        distributed=True,
        backend="dask",
        distributed_max_retries=2,
        require_render_summary=False,
    )

    assert result is False
    summary = json.loads(
        (output_dir / "summaries" / "execution_summary.json").read_text(
            encoding="utf-8"
        )
    )
    detail = summary["execution_details"][0]
    assert detail["error_type"] == "DistributedTaskError"
    assert detail["framework"] == "pymdp"
    assert Path(detail["script_path"]).resolve() == script.resolve()
    assert "remote boom" in detail["error"]
    assert "unknown" not in summary["framework_status"]
