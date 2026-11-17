#!/usr/bin/env python3
"""Regression coverage for the canonical FastAPI response contract."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from gnn.api import processor as job_mgr
from gnn.api.app import create_app as create_run_app
from gnn.api.responses import install_exception_handlers
from gnn.api.server import app as job_app
from gnn.api.server import create_app as create_job_app


def _assert_envelope(payload: Any, *, status: str) -> dict[str, Any]:
    """Assert the exact top-level API contract and return the payload."""
    assert set(payload) == {"status", "data", "error", "meta"}
    assert isinstance(payload, dict)
    assert payload["status"] == status
    assert isinstance(payload["data"], dict)
    assert isinstance(payload["meta"], dict)
    assert "timestamp" in payload["meta"]
    if status == "success":
        assert payload["error"] is None
    else:
        assert set(payload["error"]) == {"code", "message", "details"}
    return payload


@pytest.fixture(autouse=True)
def disable_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep response-contract checks independent from global limiter history."""
    monkeypatch.setenv("GNN_RATE_LIMIT", "0")


@pytest.mark.parametrize("path", ["/api/v1/health", "/api/v1/jobs", "/api/v1/tools"])
def test_job_api_success_endpoints_use_envelope(path: str) -> None:
    response = TestClient(job_app).get(path)
    assert response.status_code == 200
    _assert_envelope(response.json(), status="success")


def test_job_api_framework_errors_use_envelope() -> None:
    client = TestClient(job_app)

    missing = client.get("/api/v1/jobs/not-a-job")
    assert missing.status_code == 404
    payload = _assert_envelope(missing.json(), status="error")
    assert payload["error"]["code"] == "not_found"

    invalid_limit = client.get("/api/v1/jobs?limit=0")
    assert invalid_limit.status_code == 422
    payload = _assert_envelope(invalid_limit.json(), status="error")
    assert payload["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    "payload",
    [
        {"steps": [-1]},
        {"steps": [25]},
        {"steps": [3, 3]},
        {"steps": [3], "skip_steps": [3]},
        {"unexpected": True},
    ],
)
def test_job_api_rejects_invalid_process_requests(payload: dict[str, Any]) -> None:
    response = TestClient(job_app).post("/api/v1/process", json=payload)
    assert response.status_code == 422
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "validation_error"


def test_tool_endpoint_rejects_silently_ignored_kwargs() -> None:
    response = TestClient(job_app).post(
        "/api/v1/tools/3",
        json={"target_dir": ".", "kwargs": {"unsupported": True}},
    )
    assert response.status_code == 422
    envelope = _assert_envelope(response.json(), status="error")
    assert "not supported" in str(envelope["error"]["details"])


@pytest.mark.parametrize("route", ["/api/v1/process", "/api/v1/tools/3"])
def test_job_routes_retain_worker_limits_and_reject_invalid_values(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def queued_without_execution(job_id: str) -> None:
        return None

    monkeypatch.setattr(job_mgr, "execute_job_async", queued_without_execution)
    client = TestClient(job_app)
    before = set(job_mgr._JOBS)
    invalid = client.post(route, json={"target_dir": ".", "file_workers": True})
    assert invalid.status_code == 422
    assert set(job_mgr._JOBS) == before
    response = client.post(
        route,
        json={"target_dir": ".", "file_workers": 4, "step_workers": 2},
    )
    assert response.status_code == 200
    data = _assert_envelope(response.json(), status="success")["data"]
    job_id = data["job_id"]
    try:
        assert data["file_workers"] == 4 and data["step_workers"] == 2
        job = job_mgr.get_job(job_id)
        assert job is not None
        assert job["file_workers"] == 4 and job["step_workers"] == 2
        polled = client.get(f"/api/v1/jobs/{job_id}").json()["data"]
        assert polled["file_workers"] == 4 and polled["step_workers"] == 2
    finally:
        job_mgr._JOBS.pop(job_id, None)


def test_processor_rejects_invalid_steps_before_registering_job() -> None:
    before = set(job_mgr._JOBS)
    with pytest.raises(ValueError, match="between 0 and 24"):
        job_mgr.create_job(target_dir=".", steps=[99])
    assert set(job_mgr._JOBS) == before


def test_processor_rejects_non_integer_steps_explicitly() -> None:
    with pytest.raises(ValueError, match="integers between 0 and 24"):
        job_mgr.create_job(target_dir=".", steps=[[3]])  # type: ignore[list-item]


def test_mcp_path_validation_rejects_blank_target() -> None:
    from gnn.api.mcp import gnn_submit_job_mcp

    result = gnn_submit_job_mcp("   ")
    assert result["status"] == "error"
    assert "must not be empty" in result["message"]


def test_nonstandard_http_error_code_still_uses_envelope() -> None:
    test_app = FastAPI()
    install_exception_handlers(test_app)

    @test_app.get("/custom")
    async def custom_error() -> None:
        raise HTTPException(status_code=499, detail={"reason": "custom"})

    response = TestClient(test_app).get("/custom")
    assert response.status_code == 499
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "http_error"
    assert envelope["error"]["message"] == "HTTP error"


def test_unexpected_exception_is_sanitized_and_enveloped() -> None:
    test_app = FastAPI()
    install_exception_handlers(test_app)

    @test_app.get("/explode")
    async def explode() -> None:
        raise RuntimeError("sensitive implementation detail")

    response = TestClient(test_app, raise_server_exceptions=False).get("/explode")
    assert response.status_code == 500
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "internal_error"
    assert "sensitive" not in envelope["error"]["message"]


def test_run_api_missing_report_uses_envelope(tmp_path: Any) -> None:
    """The run surface's mixed report contract: envelope 404 (no report yet)
    vs the documented plain-text 200 download once PIPELINE_REPORT.md exists."""
    from gnn.api.app import create_app as create_run_app

    runs_store: dict[str, dict[str, Any]] = {
        "w2testhash": {
            "status": "completed",
            "request": {"output_dir": str(tmp_path)},
        }
    }
    response = TestClient(create_run_app(runs_store=runs_store)).get(
        "/api/v1/runs/w2testhash/report"
    )
    assert response.status_code == 404
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "not_found"

    (tmp_path / "PIPELINE_REPORT.md").write_text("# report\n", encoding="utf-8")
    ok = TestClient(create_run_app(runs_store=runs_store)).get(
        "/api/v1/runs/w2testhash/report"
    )
    assert ok.status_code == 200
    assert ok.headers["content-type"].startswith("text/markdown")


def test_api_step_surface_is_registry_derived() -> None:
    """API step counts and validation bounds derive from the canonical
    step registry — hardcoding 25/{13} in an API surface must be impossible
    to reintroduce silently (W2-06)."""
    from gnn.api.models import RunHealthResponse, RunStatus
    from gnn.api.pipeline_runner import (
        LLM_STEP_NUMBERS,
        MAX_PIPELINE_STEP,
        PIPELINE_STEP_COUNT,
        VALID_STEP_NUMBERS,
    )
    from gnn.pipeline.step_registry import STEPS, get_llm_steps

    assert PIPELINE_STEP_COUNT == len(STEPS)
    assert VALID_STEP_NUMBERS == {int(s.script_stem.partition("_")[0]) for s in STEPS}
    assert MAX_PIPELINE_STEP == max(VALID_STEP_NUMBERS)
    assert LLM_STEP_NUMBERS == {
        int(s.script_stem.partition("_")[0]) for s in get_llm_steps()
    }
    assert RunStatus.model_fields["total_steps"].default == PIPELINE_STEP_COUNT
    health = RunHealthResponse()
    assert health.pipeline_steps == PIPELINE_STEP_COUNT
    assert health.version != "2.0.0"


@pytest.mark.parametrize("factory", [create_run_app, create_job_app])
def test_invalid_utf8_request_body_retains_json_safe_validation_details(
    factory: Any,
) -> None:
    response = TestClient(factory()).post(
        "/api/v1/render",
        content=b"\xff",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 422
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "validation_error"
    details = envelope["error"]["details"]
    assert isinstance(details, list) and details
    assert any(
        entry.get("loc") == ["body"] and entry.get("input") == "\\xff"
        for entry in details
    )
