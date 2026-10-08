"""Behavior tests for the run delete surface: cancellation and artifacts.

DELETE /api/v1/runs/{run_hash} upgrades the old housekeeping-only contract:
active (queued/running) runs are cancelled through the run's CancelToken
before the record is removed, artifacts are deleted unless the run targeted
the repository output tree, and unknown or ambiguous hashes fail explicitly.
"""

import asyncio
import importlib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from gnn.api import mcp as api_mcp
from gnn.api import processor as runs_processor
from gnn.execute.subprocess_envelope import CancelToken

#: A pid that cannot exist: os.killpg raises ProcessLookupError, exercising
#: the direct-child fallback in request_process_stop instead of ever
#: signalling a real process group.
_NO_SUCH_PID = 999_999_999


def _completed_entry(output_dir: Path) -> dict[str, Any]:
    """Build a completed run record pointing at an artifact directory."""
    return {
        "status": "completed",
        "started_at": "2026-01-01T00:00:00",
        "completed_at": "2026-01-01T00:05:00",
        "request": {"output_dir": str(output_dir)},
        "steps_completed": 3,
        "total_steps": 3,
        "errors": [],
        "events": [],
    }


def _store_client(store: dict[str, dict[str, Any]]) -> TestClient:
    """Create a test client over an app bound to the given store."""
    api_app = importlib.import_module("gnn.api.app")
    return TestClient(api_app.create_app(runs_store=store))


async def _wait_for_status(entry: dict[str, Any], status: str) -> None:
    """Await until the entry reports the requested status."""
    while entry["status"] != status:
        await asyncio.sleep(0.01)


def test_delete_completed_run_removes_artifacts_then_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A completed run's artifacts go away with the record; repeat is 404."""
    monkeypatch.setenv("GNN_RATE_LIMIT", "0")
    monkeypatch.delenv("GNN_API_KEY", raising=False)
    output_dir = tmp_path / "finished_out"
    (output_dir / "steps").mkdir(parents=True)
    (output_dir / "PIPELINE_REPORT.md").write_text("report", encoding="utf-8")
    (output_dir / "steps" / "done.txt").write_text("done", encoding="utf-8")

    store = {"deadbeef01": _completed_entry(output_dir)}
    client = _store_client(store)

    response = client.delete("/api/v1/runs/deadbeef01")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    data = body["data"]
    assert data["deleted"] == "deadbeef01"
    assert data["existed"] is True
    assert data["cancelled"] is False
    assert data["artifacts_removed"] is True
    assert not output_dir.exists()
    assert "deadbeef01" not in store

    repeat = client.delete("/api/v1/runs/deadbeef01")
    assert repeat.status_code == 404
    assert repeat.json()["status"] == "error"


@pytest.mark.asyncio
async def test_delete_running_run_cancels_process_and_removes_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Deleting a running run cancels its token, kills the process, and pops."""
    from gnn.api import app as api_app

    monkeypatch.setenv("GNN_RATE_LIMIT", "0")
    monkeypatch.delenv("GNN_API_KEY", raising=False)
    output_dir = tmp_path / "live_out"
    output_dir.mkdir()
    (output_dir / "artifact.bin").write_text("data", encoding="utf-8")

    run_hash = "liverun0001"
    entry: dict[str, Any] = {
        "status": "queued",
        "started_at": "2026-01-01T00:00:00",
        "request": {"output_dir": str(output_dir)},
        "steps_completed": 0,
        "total_steps": 3,
        "errors": [],
        "events": [],
        "cancel_token": CancelToken(),
    }
    store: dict[str, dict[str, Any]] = {run_hash: entry}
    app = api_app.create_app(runs_store=store)

    class _BlockingProcess:
        """Subprocess stand-in that blocks in communicate until terminated."""

        pid = _NO_SUCH_PID

        def __init__(self) -> None:
            self.terminate_calls = 0
            self.returncode: int | None = None
            self._released = asyncio.Event()

        async def communicate(self) -> tuple[bytes, bytes]:
            await self._released.wait()
            self.returncode = -15
            return b"partial stdout", b"terminated by signal"

        def terminate(self) -> None:
            self.terminate_calls += 1
            self._released.set()

    proc = _BlockingProcess()
    spawn_calls: list[list[str]] = []

    async def spawn_recorder(*command: str, **kwargs: Any) -> _BlockingProcess:
        spawn_calls.append(list(command))
        return proc

    monkeypatch.setattr(api_app.asyncio, "create_subprocess_exec", spawn_recorder)

    request = api_app.RunRequest(target_dir=str(tmp_path), output_dir=str(output_dir))
    task = asyncio.create_task(api_app._execute_pipeline(run_hash, request, store))
    await asyncio.wait_for(_wait_for_status(entry, "running"), timeout=10.0)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete(f"/api/v1/runs/{run_hash}")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["cancelled"] is True
    assert data["deleted"] == run_hash
    assert data["artifacts_removed"] is True
    assert proc.terminate_calls >= 1
    assert len(spawn_calls) == 1  # the run really was mid-flight when deleted
    assert not output_dir.exists()
    assert run_hash not in store

    await asyncio.wait_for(task, timeout=10.0)
    # The executor finished writing terminal state on the record it still
    # references, even though the store no longer holds it.
    assert entry["status"] == "cancelled"
    assert entry["completed_at"] is not None
    assert entry["exit_code"] == -15


@pytest.mark.asyncio
async def test_execute_pipeline_prespawn_cancel_never_spawns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pre-cancelled run record is marked cancelled with zero spawns."""
    from gnn.api import app as api_app

    token = CancelToken()
    token.cancel(reason="run deleted")
    run_hash = "prespawn11"
    entry: dict[str, Any] = {
        "status": "queued",
        "started_at": "2026-01-01T00:00:00",
        "request": {"output_dir": str(tmp_path / "never_out")},
        "steps_completed": 0,
        "total_steps": 3,
        "errors": [],
        "events": [],
        "cancel_token": token,
    }
    store: dict[str, dict[str, Any]] = {run_hash: entry}

    spawn_calls: list[list[str]] = []

    async def spawn_recorder(*command: str, **kwargs: Any) -> Any:
        spawn_calls.append(list(command))
        raise AssertionError("pipeline spawned for a pre-cancelled run")

    monkeypatch.setattr(api_app.asyncio, "create_subprocess_exec", spawn_recorder)

    request = api_app.RunRequest(
        target_dir=str(tmp_path), output_dir=str(tmp_path / "never_out")
    )
    await asyncio.wait_for(
        api_app._execute_pipeline(run_hash, request, store), timeout=10.0
    )

    assert spawn_calls == []
    assert entry["status"] == "cancelled"
    assert entry["completed_at"] is not None
    assert entry["duration_seconds"] == 0.0
    assert any(event["type"] == "run_cancelled" for event in entry["events"])
    assert "exit_code" not in entry


def test_delete_retains_repository_output_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run targeting the repository output tree keeps its artifacts."""
    monkeypatch.setenv("GNN_RATE_LIMIT", "0")
    monkeypatch.delenv("GNN_API_KEY", raising=False)
    processor_repo_output = (
        Path(runs_processor.__file__).resolve().parents[3] / "output"
    ).resolve()
    # Guard the test's own premise: the path under test must genuinely be the
    # repository output tree the guard compares against.
    assert processor_repo_output.name == "output"

    store = {"guardhash": _completed_entry(processor_repo_output)}
    client = _store_client(store)

    response = client.delete("/api/v1/runs/guardhash")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["artifacts_removed"] is False
    assert "repository output tree" in data["artifacts_note"]
    assert processor_repo_output.exists()
    assert "guardhash" not in store


def test_delete_run_wait_timeout_keeps_record_and_reports_status(
    tmp_path: Path,
) -> None:
    """A cancel that never lands leaves the record and reports its status."""
    token = CancelToken()
    run_hash = "slowrun001"
    store: dict[str, dict[str, Any]] = {
        run_hash: {
            "status": "running",
            "request": {"output_dir": str(tmp_path)},
            "cancel_token": token,
        }
    }
    with pytest.raises(RuntimeError) as excinfo:
        runs_processor.delete_run(
            run_hash, runs_store=store, wait_timeout=0.15, poll_interval=0.01
        )
    assert "running" in str(excinfo.value)
    assert run_hash in store


def test_delete_endpoint_maps_errors_to_http_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unknown hashes 404, ambiguous prefixes 409, wait timeouts 409."""
    monkeypatch.setenv("GNN_RATE_LIMIT", "0")
    monkeypatch.delenv("GNN_API_KEY", raising=False)
    output_dir = tmp_path / "either_out"
    output_dir.mkdir()
    store = {
        "aaaa1111": _completed_entry(output_dir),
        "baba1111": _completed_entry(output_dir),
        "baba2222": _completed_entry(output_dir),
    }
    client = _store_client(store)

    unknown = client.delete("/api/v1/runs/zzzz9999")
    assert unknown.status_code == 404
    assert unknown.json()["status"] == "error"

    ambiguous = client.delete("/api/v1/runs/baba")
    assert ambiguous.status_code == 409
    assert ambiguous.json()["status"] == "error"
    assert "aaaa1111" in store

    def raise_wait_timeout(run_hash: str, **kwargs: Any) -> dict[str, Any]:
        """Simulate a delete whose cancel request never lands."""
        raise RuntimeError(
            f"Run {run_hash} did not reach a terminal state after "
            "cancel; status: running"
        )

    monkeypatch.setattr(runs_processor, "delete_run", raise_wait_timeout)
    timed_out = client.delete("/api/v1/runs/aaaa1111")
    assert timed_out.status_code == 409
    assert "running" in timed_out.text
    assert "aaaa1111" in store


def test_mcp_delete_run_unknown_hash_reports_not_found() -> None:
    """The MCP tool surfaces unknown hashes as a not-found error."""
    result = api_mcp.gnn_delete_run_mcp("nosuchhash99")
    assert result["status"] == "error"
    assert "not found" in result["message"].lower()


def test_mcp_delete_run_success_returns_result_fields(tmp_path: Path) -> None:
    """The MCP tool reports deletion details and removes artifacts."""
    output_dir = tmp_path / "mcp_out"
    output_dir.mkdir()
    (output_dir / "artifact.txt").write_text("data", encoding="utf-8")
    run_hash = "mcpdelete1"
    runs_processor.RUNS_STORE[run_hash] = _completed_entry(output_dir)
    try:
        result = api_mcp.gnn_delete_run_mcp(run_hash)
        assert result["status"] == "success"
        assert result["deleted"] == run_hash
        assert result["existed"] is True
        assert result["cancelled"] is False
        assert result["artifacts_removed"] is True
        assert run_hash not in runs_processor.RUNS_STORE
        assert not output_dir.exists()
    finally:
        runs_processor.RUNS_STORE.pop(run_hash, None)


def test_mcp_manifest_contains_delete_run() -> None:
    """The serialized manifest advertises the delete tool with its schema."""
    manifest = api_mcp.register_mcp_tools()
    names = {tool["name"] for tool in manifest["tools"]}
    assert "gnn_delete_run" in names
    delete_tool = next(t for t in manifest["tools"] if t["name"] == "gnn_delete_run")
    assert delete_tool["inputSchema"].get("required") == ["run_hash"]
    assert "run_hash" in delete_tool["inputSchema"]["properties"]
