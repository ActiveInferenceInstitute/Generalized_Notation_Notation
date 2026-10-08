"""Tests for the GNN API endpoints."""

from pathlib import Path
from typing import Any

import pytest

from gnn.api.app import FASTAPI_AVAILABLE, create_app


def _assert_envelope(payload: Any, status: str = "success") -> dict[str, Any]:
    """Assert and return the canonical API response envelope."""
    assert set(payload) == {"status", "data", "error", "meta"}
    assert isinstance(payload, dict)
    assert payload["status"] == status
    assert isinstance(payload["data"], dict)
    assert isinstance(payload["meta"], dict)
    assert "timestamp" in payload["meta"]
    if status == "success":
        assert payload["error"] is None
    else:
        assert isinstance(payload["error"], dict)
        assert payload["error"]["code"]
        assert payload["error"]["message"]
    return payload


def test_api_health_endpoint() -> Any:
    """Test the health check endpoint."""
    from fastapi.testclient import TestClient

    assert FASTAPI_AVAILABLE is True
    app = create_app()
    client = TestClient(app)

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    assert envelope["data"]["status"] == "healthy"
    assert "renderers" in envelope["data"]
    assert "version" in envelope["data"]


def test_api_list_runs_empty() -> Any:
    """Test listing runs when none have been submitted."""
    from fastapi.testclient import TestClient

    # We need to clear the global _runs dict if we want a clean state,
    # but since it's module-level in app.py, isolations is tricky without reloading.
    # For baseline, we just check it returns a dict.
    app = create_app()
    client = TestClient(app)

    response = client.get("/api/v1/runs")
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    assert isinstance(envelope["data"]["runs"], dict)
    assert envelope["data"]["total"] >= 0


def test_api_submit_run_invalid_payload() -> Any:
    """Test submitting a run with invalid JSON."""
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    response = client.post("/api/v1/run", json={"skip_steps": ["not_an_int"]})
    # FastAPI returns 422 Unprocessable Entity for schema validation errors
    assert response.status_code == 422
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "validation_error"
    assert envelope["error"]["details"]


def test_api_submit_run_success() -> Any:
    """Test successful run submission."""
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    import gnn.api.app as api_app
    import gnn.pipeline as pipeline
    import gnn.pipeline.hasher

    orig_hasher = getattr(pipeline.hasher, "compute_run_hash", None)
    orig_execute = getattr(api_app, "_execute_pipeline", None)

    async def _complete_pipeline_immediately(*args: Any, **kwargs: Any) -> None:
        return None

    # Replace the reference directly for this endpoint wiring test.
    pipeline.hasher.compute_run_hash = lambda *args, **kwargs: "test_hash_123"
    api_app._execute_pipeline = _complete_pipeline_immediately

    try:
        response = client.post(
            "/api/v1/run",
            json={
                "target_dir": "input/gnn_files",
                "output_dir": "output",
                "skip_steps": [13],
            },
        )

        assert response.status_code == 200
        envelope = _assert_envelope(response.json())
        assert envelope["data"]["run_hash"] == "test_hash_123"
        assert envelope["data"]["status"] == "queued"
    finally:
        if orig_hasher:
            pipeline.hasher.compute_run_hash = orig_hasher
        if orig_execute:
            api_app._execute_pipeline = orig_execute


def test_api_submit_run_rejects_output_outside_repo() -> None:
    """The run API should validate output_dir as well as target_dir."""
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/api/v1/run",
        json={"target_dir": ".", "output_dir": "../outside-repo-output"},
    )

    assert response.status_code == 400
    envelope = _assert_envelope(response.json(), status="error")
    assert "Output directory" in envelope["error"]["message"]


@pytest.mark.parametrize(
    "payload",
    [
        {"skip_steps": [-1]},
        {"skip_steps": [25]},
        {"skip_steps": [3, 3]},
        {"config": {"ignored": True}},
        {"unknown_field": "ignored"},
    ],
)
def test_api_submit_run_rejects_unsupported_or_invalid_fields(
    payload: dict[str, Any],
) -> None:
    """Invalid step selections and silently ignored fields fail explicitly."""
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    response = client.post("/api/v1/run", json=payload)
    assert response.status_code == 422
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "validation_error"


def test_api_unknown_route_uses_error_envelope() -> None:
    """Framework-generated 404 responses preserve the API contract."""
    from fastapi.testclient import TestClient

    response = TestClient(create_app()).get("/api/v1/does-not-exist")
    assert response.status_code == 404
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "not_found"


def test_api_submit_run_stores_normalized_output_dir(monkeypatch: Any) -> None:
    """Queued run metadata should preserve the caller-selected output dir."""
    from fastapi.testclient import TestClient

    import gnn.api.app as api_app
    import gnn.pipeline as pipeline
    import gnn.pipeline.hasher

    app = create_app()
    client = TestClient(app)
    run_hash = "output_dir_contract_hash"
    output_dir = "output/api_contract_test"

    orig_hasher = getattr(pipeline.hasher, "compute_run_hash", None)
    orig_execute = getattr(api_app, "_execute_pipeline", None)
    api_app._runs.pop(run_hash, None)

    async def _complete_pipeline_immediately(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(pipeline.hasher, "compute_run_hash", lambda *a, **k: run_hash)
    monkeypatch.setattr(api_app, "_execute_pipeline", _complete_pipeline_immediately)

    try:
        response = client.post(
            "/api/v1/run",
            json={"target_dir": ".", "output_dir": output_dir},
        )

        assert response.status_code == 200
        stored_output = api_app._runs[run_hash]["request"]["output_dir"]
        assert (
            Path(stored_output).resolve()
            == (Path(__file__).resolve().parents[2] / output_dir).resolve()
        )
    finally:
        api_app._runs.pop(run_hash, None)
        if orig_hasher:
            pipeline.hasher.compute_run_hash = orig_hasher
        if orig_execute:
            api_app._execute_pipeline = orig_execute


@pytest.mark.asyncio
async def test_job_processor_uses_requested_output_dir(monkeypatch: Any) -> None:
    """Async job execution should pass the job's output_dir to src/gnn/main.py."""
    from gnn.api import processor as job_mgr

    captured_cmd: list[str] = []

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def fake_create_subprocess_exec(*cmd: str, **kwargs: Any) -> FakeProcess:
        captured_cmd.extend(cmd)
        return FakeProcess()

    monkeypatch.setattr(
        job_mgr.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    requested_output = "output/job_contract_test"
    job_id = job_mgr.create_job(target_dir=".", output_dir=requested_output, steps=[3])
    try:
        await job_mgr.execute_job_async(job_id)
        output_arg = captured_cmd[captured_cmd.index("--output-dir") + 1]
        assert (
            Path(output_arg).resolve()
            == (Path(__file__).resolve().parents[2] / requested_output).resolve()
        )
    finally:
        job_mgr._JOBS.pop(job_id, None)


@pytest.mark.asyncio
async def test_run_api_executes_real_main_subprocess_and_preserves_warning_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The run API must invoke src/gnn/main.py instead of simulating successful steps."""
    import json

    import gnn.api.app as api_app

    output_dir = tmp_path / "output"
    summary_dir = output_dir / "00_pipeline_summary"
    summary_dir.mkdir(parents=True)
    (summary_dir / "pipeline_execution_summary.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "script_name": "3_gnn.py",
                        "status": "SUCCESS_WITH_WARNINGS",
                        "duration_seconds": 0.25,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    captured_command: list[str] = []
    invocation_id = ""

    class FakeProcess:
        pid = 1234
        returncode = 2

        async def communicate(self) -> tuple[bytes, bytes]:
            # Model publication by this subprocess, after the invocation starts.
            summary = summary_dir / "pipeline_execution_summary.json"
            payload = json.loads(summary.read_text())
            payload["run_id"] = invocation_id
            summary.write_text(json.dumps(payload))
            return b"", b""

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        nonlocal invocation_id
        invocation_id = kwargs["env"]["GNN_RUN_ID"]
        captured_command.extend(command)
        return FakeProcess()

    monkeypatch.setattr(
        api_app.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    run_hash = "real_subprocess_contract"
    api_app._runs[run_hash] = {
        "status": "queued",
        "started_at": None,
        "steps_completed": 0,
        "total_steps": 23,
        "errors": [],
        "events": [],
    }
    request = api_app.RunRequest(
        target_dir=str(tmp_path),
        output_dir=str(output_dir),
        skip_steps=[4],
        skip_llm=True,
    )
    try:
        await api_app._execute_pipeline(run_hash, request)
        entry = api_app._runs[run_hash]
        assert captured_command[1:4] == ["-P", "-m", "gnn.main"]
        assert captured_command[captured_command.index("--skip-steps") + 1] == "4,13"
        assert entry["status"] == "completed"
        assert entry["exit_code"] == 2
        assert entry["process_id"] == 1234
        assert entry["steps_completed"] == 1
        assert entry["total_steps"] == 1
        assert any(event["type"] == "pipeline_warning" for event in entry["events"])
    finally:
        api_app._runs.pop(run_hash, None)


def test_api_availability_flag() -> Any:
    """Verify that the availability flag matches reality."""
    try:
        import fastapi

        expected = True
    except ImportError:
        expected = False

    assert FASTAPI_AVAILABLE == expected
