#!/usr/bin/env python3
"""Behavior tests for the CLI-parity API endpoints (``src/gnn/api/parity.py``).

Every behavioral test runs against BOTH FastAPI surfaces via the parametrized
``client`` fixture — ``gnn.api.app.create_app`` (the ``gnn serve`` surface)
and ``gnn.api.server.create_app`` (the ``python -m gnn.api.server`` surface) —
and the route-table tests pin each surface's FULL route table: the shared
parity routes (``PARITY_ROUTES``), each factory's exclusive routes
(``RUN_FACTORY_ROUTES`` / ``JOB_FACTORY_ROUTES``), and the framework meta
routes (``META_ROUTES``). The parity set is additionally re-derived as the
actual cross-factory intersection, so no route can drift in or out of either
factory without failing a pinned assertion.

Pinned exit-code → HTTP mapping (see the module docstring of
``src/gnn/api/parity.py``; mirrors the strict ``pipeline_exit_succeeded``
policy in ``gnn.api.pipeline_runner``):

- CLI exit 0                        → 200 success envelope, data.exit_code == 0
- CLI exit 2 (warnings)             → 200 success envelope, data.exit_code == 2,
                                      warnings/errors listed in data
- would-be exit 2 under strict=True → 400 bad_request, errors in error.details
- missing file/directory            → 404 not_found
- path-boundary violation           → 400 bad_request
- backend/import/operation failure  → 400 bad_request with sanitized detail
"""

from __future__ import annotations

import importlib.util
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

import pytest
from fastapi.testclient import TestClient

import gnn.main as orchestrator
from gnn.api.app import create_app as create_run_app
from gnn.api.server import app as server_module_app
from gnn.api.server import create_app as create_job_app
from gnn.pipeline.hasher import index_run
from gnn.utils.arguments.pipeline_arguments import PipelineArguments

#: Committed exemplar with a full valid GNN model (11 variables, 10
#: connections) — positive-case input for validate/parse/extract/render/graph.
VALID_EXEMPLAR = "input/gnn_files/discrete/simple_mdp.md"

#: Committed fixture missing required sections → validate exit 2 / strict 400.
INVALID_EXEMPLAR = "tests/api/fixtures/invalid_gnn_missing_sections.md"

#: Committed fixture with all sections but an undeclared connection target
#: (`A>Z`) → parse exit 2 and validate exit 2.
PARSE_WARNING_EXEMPLAR = "tests/api/fixtures/parse_warning_unknown_variable.md"

#: The exact parity route surface that must exist on BOTH FastAPI factories.
PARITY_ROUTES: Dict[str, Set[str]] = {
    "/api/v1/validate": {"POST"},
    "/api/v1/parse": {"POST"},
    "/api/v1/extract": {"POST"},
    "/api/v1/render": {"POST"},
    "/api/v1/graph": {"POST"},
    "/api/v1/templates": {"GET"},
    "/api/v1/templates/{name}": {"GET"},
    "/api/v1/pull": {"POST"},
    "/api/v1/models": {"GET"},
    "/api/v1/preflight": {"POST"},
    "/api/v1/report": {"POST"},
    "/api/v1/reproduce": {"POST"},
}


#: Routes exclusive to the ``gnn.api.app.create_app`` run surface. Every other
#: route on that factory must be in ``PARITY_ROUTES`` or ``META_ROUTES``.
RUN_FACTORY_ROUTES: Dict[str, Set[str]] = {
    "/api/v1/run": {"POST"},
    "/api/v1/runs": {"GET"},
    "/api/v1/runs/{run_hash}": {"GET", "DELETE"},
    "/api/v1/runs/{run_hash}/report": {"GET"},
    "/api/v1/runs/{run_hash}/stream": {"GET"},
}

#: Routes exclusive to the ``gnn.api.server.create_app`` job surface. Every
#: other route on that factory must be in ``PARITY_ROUTES`` or ``META_ROUTES``.
JOB_FACTORY_ROUTES: Dict[str, Set[str]] = {
    "/api/v1/process": {"POST"},
    "/api/v1/jobs": {"GET"},
    "/api/v1/jobs/{job_id}": {"GET", "DELETE"},
    "/api/v1/tools": {"GET"},
    "/api/v1/tools/{step}": {"POST"},
}

#: Framework routes both factories mount (FastAPI/Starlette defaults). The
#: docs/openapi routes expose GET+HEAD because Starlette adds HEAD to its
#: built-ins; hand-registered APIRoutes expose only their declared methods.
META_ROUTES: Dict[str, Set[str]] = {
    "/api/v1/health": {"GET"},
    "/openapi.json": {"GET", "HEAD"},
    "/docs": {"GET", "HEAD"},
    "/docs/oauth2-redirect": {"GET", "HEAD"},
    "/redoc": {"GET", "HEAD"},
}


def _assert_envelope(payload: Any, status: str = "success") -> Dict[str, Any]:
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


@pytest.fixture(autouse=True)
def disable_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep parity checks independent from global limiter history."""
    monkeypatch.setenv("GNN_RATE_LIMIT", "0")


@pytest.fixture(params=["app_factory", "server_factory"])
def client(request: pytest.FixtureRequest) -> Iterator[TestClient]:
    """Serve every behavior test against both FastAPI surfaces."""
    factory: Callable[[], Any] = (
        create_run_app if request.param == "app_factory" else create_job_app
    )
    yield TestClient(factory())


# ── Route parity across surfaces ─────────────────────────────────────────────


def _route_table(app: Any) -> Dict[str, Set[str]]:
    """Collect path → method-set for every HTTP route on ``app``."""
    table: Dict[str, Set[str]] = {}
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if path and methods:
            table.setdefault(path, set()).update(methods)
    return table


@pytest.mark.unit
def test_parity_routes_present_on_both_surfaces() -> None:
    """Every parity route must exist with identical methods on both surfaces."""
    run_app_table = _route_table(create_run_app())
    job_app_table = _route_table(create_job_app())
    module_app_table = _route_table(server_module_app)
    for path, methods in PARITY_ROUTES.items():
        for surface, table in (
            ("gnn.api.app", run_app_table),
            ("gnn.api.server factory", job_app_table),
            ("gnn.api.server module app", module_app_table),
        ):
            assert table.get(path) == methods, f"{path} missing or wrong on {surface}"


@pytest.mark.unit
def test_factory_route_tables_match_pins() -> None:
    """Each factory's FULL route table must equal its pinned composition.

    Derived from the live ``_route_table`` of ``gnn.api.app.create_app`` and
    ``gnn.api.server.create_app`` (plus the ``gnn.api.server`` module app):
    run table == ``PARITY_ROUTES`` | ``RUN_FACTORY_ROUTES`` | ``META_ROUTES``,
    job-factory table == ``PARITY_ROUTES`` | ``JOB_FACTORY_ROUTES`` |
    ``META_ROUTES``, and the module app must match its own factory exactly.
    Adding or removing ANY route on either factory without updating the
    corresponding pin fails this test loudly — no hardcoded union list can
    mask drift.
    """
    run_app_table = _route_table(create_run_app())
    job_app_table = _route_table(create_job_app())
    module_app_table = _route_table(server_module_app)
    expected_run_table = {**PARITY_ROUTES, **RUN_FACTORY_ROUTES, **META_ROUTES}
    expected_job_table = {**PARITY_ROUTES, **JOB_FACTORY_ROUTES, **META_ROUTES}
    assert run_app_table == expected_run_table, (
        f"run-factory route table drifted: "
        f"unexpected={set(run_app_table) - set(expected_run_table)} "
        f"missing={set(expected_run_table) - set(run_app_table)}"
    )
    assert job_app_table == expected_job_table, (
        f"job-factory route table drifted: "
        f"unexpected={set(job_app_table) - set(expected_job_table)} "
        f"missing={set(expected_job_table) - set(job_app_table)}"
    )
    assert module_app_table == expected_job_table, (
        f"module app drifted from its factory: "
        f"unexpected={set(module_app_table) - set(expected_job_table)} "
        f"missing={set(expected_job_table) - set(module_app_table)}"
    )


@pytest.mark.unit
def test_parity_surface_is_the_real_cross_factory_intersection() -> None:
    """PARITY_ROUTES must equal the actual intersection of both factories.

    Computes ``shared`` from the REAL route tables of ``gnn.api.app`` and
    ``gnn.api.server`` (path present on both surfaces) and asserts it equals
    ``PARITY_ROUTES`` merged with ``META_ROUTES``. A route sneaking into both
    factories (breaking surface exclusivity) or drifting out of one is caught
    here even if the per-factory pins above were also updated.
    """
    run_app_table = _route_table(create_run_app())
    job_app_table = _route_table(create_job_app())
    shared = {
        path: methods
        for path, methods in run_app_table.items()
        if path in job_app_table
    }
    assert shared == {**PARITY_ROUTES, **META_ROUTES}


# ── POST /api/v1/validate ────────────────────────────────────────────────────


@pytest.mark.unit
def test_validate_happy_path_returns_exit_code_0(client: TestClient) -> None:
    """A fully valid exemplar validates cleanly: 200, exit_code 0, counts."""
    response = client.post("/api/v1/validate", json={"file_path": VALID_EXEMPLAR})
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    assert envelope["data"]["exit_code"] == 0
    assert envelope["data"]["valid"] is True
    assert envelope["data"]["errors"] == []
    assert envelope["data"]["variables_count"] == 11
    assert envelope["data"]["connections_count"] == 10


@pytest.mark.unit
def test_validate_non_strict_warnings_map_to_exit_code_2(
    client: TestClient,
) -> None:
    """Missing required sections: 200 success envelope carrying exit_code 2."""
    response = client.post("/api/v1/validate", json={"file_path": INVALID_EXEMPLAR})
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    assert envelope["data"]["exit_code"] == 2
    assert envelope["data"]["valid"] is False
    errors = envelope["data"]["errors"]
    assert errors
    assert all("code" in error and "message" in error for error in errors)
    assert any("Missing required section" in error["message"] for error in errors)


@pytest.mark.unit
def test_validate_strict_escalates_warnings_to_400(client: TestClient) -> None:
    """strict=True escalates a would-be exit 2 to a 400 bad_request envelope."""
    response = client.post(
        "/api/v1/validate",
        json={"file_path": INVALID_EXEMPLAR, "strict": True},
    )
    assert response.status_code == 400
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "bad_request"
    details = envelope["error"]["details"]
    assert isinstance(details, dict)
    assert details["errors"]
    assert all("message" in error for error in details["errors"])


@pytest.mark.unit
def test_validate_missing_file_maps_to_404(client: TestClient) -> None:
    """A nonexistent repository-local file mirrors _guard_input_file as 404."""
    response = client.post(
        "/api/v1/validate",
        json={"file_path": "input/gnn_files/no_such_model.md"},
    )
    assert response.status_code == 404
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "not_found"
    assert "not found" in envelope["error"]["message"]


@pytest.mark.unit
def test_validate_out_of_repo_path_maps_to_400(client: TestClient) -> None:
    """Client paths outside the repository boundary are rejected as 400."""
    response = client.post("/api/v1/validate", json={"file_path": "/etc/passwd"})
    assert response.status_code == 400
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "bad_request"
    assert "repository root" in envelope["error"]["message"]


@pytest.mark.unit
def test_validate_rejects_unknown_request_fields(client: TestClient) -> None:
    """Request models forbid extra fields explicitly (no silent ignoring)."""
    response = client.post(
        "/api/v1/validate",
        json={"file_path": VALID_EXEMPLAR, "unexpected": True},
    )
    assert response.status_code == 422
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "validation_error"


# ── POST /api/v1/parse ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_json_happy_path_returns_structure(client: TestClient) -> None:
    """Parsing a valid exemplar returns variables, connections, exit_code 0."""
    response = client.post(
        "/api/v1/parse", json={"file_path": VALID_EXEMPLAR, "format": "json"}
    )
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    data = envelope["data"]
    assert data["exit_code"] == 0
    assert data["format"] == "json"
    assert len(data["variables"]) == 11
    assert len(data["connections"]) == 10
    assert data["variables"][0]["name"]
    assert data["connections"][0]["source"]


@pytest.mark.unit
def test_parse_summary_format_returns_counts(client: TestClient) -> None:
    """Summary format mirrors the CLI's File/Variables/Connections report."""
    response = client.post(
        "/api/v1/parse", json={"file_path": VALID_EXEMPLAR, "format": "summary"}
    )
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    data = envelope["data"]
    assert data["exit_code"] == 0
    assert data["variables_count"] == 11
    assert data["connections_count"] == 10
    assert isinstance(data["metadata_keys"], list)


@pytest.mark.unit
def test_parse_yaml_format_embeds_yaml_text(client: TestClient) -> None:
    """YAML format carries the structured payload plus rendered YAML text."""
    response = client.post(
        "/api/v1/parse", json={"file_path": VALID_EXEMPLAR, "format": "yaml"}
    )
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    data = envelope["data"]
    assert data["format"] == "yaml"
    if importlib.util.find_spec("yaml") is not None:
        assert data["yaml"] is not None
        assert "variables:" in data["yaml"]
    else:
        # PyYAML absent: the endpoint degrades to the structured payload
        # exactly like the CLI; yaml_text stays None (the envelope stays
        # structured either way — see parity.py).
        assert data["yaml"] is None


@pytest.mark.unit
def test_parse_unknown_connection_maps_to_exit_code_2(client: TestClient) -> None:
    """An undeclared connection target is a warning outcome: 200, exit 2."""
    response = client.post("/api/v1/parse", json={"file_path": PARSE_WARNING_EXEMPLAR})
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    data = envelope["data"]
    assert data["exit_code"] == 2
    assert data["warnings"]
    assert any("Z" in warning for warning in data["warnings"])
    assert data["errors"]


@pytest.mark.unit
def test_parse_missing_file_maps_to_404(client: TestClient) -> None:
    """A nonexistent file maps to 404 before any parsing happens."""
    response = client.post(
        "/api/v1/parse", json={"file_path": "input/gnn_files/no_such_model.md"}
    )
    assert response.status_code == 404
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "not_found"


# ── POST /api/v1/extract ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_extract_happy_path_returns_pomdp_payload(client: TestClient) -> None:
    """Extracting the exemplar returns the POMDP payload with exit_code 0."""
    response = client.post("/api/v1/extract", json={"file_path": VALID_EXEMPLAR})
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    data = envelope["data"]
    assert data["exit_code"] == 0
    assert data["strict"] is True
    assert data["pomdp"]["model_name"]


@pytest.mark.unit
def test_extract_contentless_file_maps_to_400(client: TestClient) -> None:
    """A file with no POMDP content fails explicitly (CLI exit 1 → 400)."""
    response = client.post(
        "/api/v1/extract", json={"file_path": "input/gnn_files/INDEX.md"}
    )
    assert response.status_code == 400
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "bad_request"
    assert "no POMDP state-space content" in envelope["error"]["message"]


# ── POST /api/v1/render ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_render_happy_path_produces_artifact(client: TestClient) -> None:
    """Rendering the exemplar with pymdp yields 200 and a real artifact."""
    output_dir = "output/api_parity_render"
    shutil.rmtree(Path(output_dir), ignore_errors=True)
    try:
        response = client.post(
            "/api/v1/render",
            json={
                "file_path": VALID_EXEMPLAR,
                "framework": "pymdp",
                "output_dir": output_dir,
            },
        )
        assert response.status_code == 200
        envelope = _assert_envelope(response.json())
        data = envelope["data"]
        assert data["exit_code"] == 0
        assert data["framework"] == "pymdp"
        assert data["artifact"] is not None
        assert data["artifact"].endswith(".py")
        assert Path(data["artifact"]).is_file()
    finally:
        shutil.rmtree(Path(output_dir), ignore_errors=True)


@pytest.mark.unit
def test_render_rejects_unknown_framework_with_422(client: TestClient) -> None:
    """Frameworks outside the CLI choice set fail request validation."""
    response = client.post(
        "/api/v1/render",
        json={"file_path": VALID_EXEMPLAR, "framework": "matplotlib"},
    )
    assert response.status_code == 422
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "validation_error"


# ── POST /api/v1/graph ───────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("output_format", ["mermaid", "text"])
def test_graph_happy_path_returns_graph(client: TestClient, output_format: str) -> None:
    """Both graph formats return the rendered graph with exit_code 0."""
    response = client.post(
        "/api/v1/graph",
        json={"file_path": VALID_EXEMPLAR, "format": output_format},
    )
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    data = envelope["data"]
    assert data["exit_code"] == 0
    assert data["format"] == output_format
    assert data["graph"]
    if output_format == "mermaid":
        assert data["graph"].startswith("graph TD")


# ── GET /api/v1/templates ────────────────────────────────────────────────────


@pytest.mark.unit
def test_templates_list_returns_records(client: TestClient) -> None:
    """Listing templates returns checksummed records."""
    response = client.get("/api/v1/templates")
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    data = envelope["data"]
    assert data["total"] == len(data["templates"])
    assert data["total"] >= 3
    first = data["templates"][0]
    assert first["name"] and len(first["sha256"]) == 64


@pytest.mark.unit
def test_templates_show_known_name(client: TestClient) -> None:
    """Showing a maintained template returns its full record."""
    response = client.get("/api/v1/templates/actinf-pomdp-2state")
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    template = envelope["data"]["template"]
    assert template["name"] == "actinf-pomdp-2state"
    assert len(template["sha256"]) == 64


@pytest.mark.unit
def test_templates_show_unknown_name_maps_to_404(client: TestClient) -> None:
    """An unknown template name mirrors the CLI KeyError as 404."""
    response = client.get("/api/v1/templates/no-such-template")
    assert response.status_code == 404
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "not_found"
    assert "Unknown template" in envelope["error"]["message"]


# ── POST /api/v1/pull ────────────────────────────────────────────────────────


def _first_template_name(client: TestClient) -> str:
    """Fetch one real maintained template name from the list endpoint."""
    response = client.get("/api/v1/templates")
    envelope = _assert_envelope(response.json())
    return envelope["data"]["templates"][0]["name"]


@pytest.mark.unit
def test_pull_dry_run_returns_plan_without_writing(client: TestClient) -> None:
    """A dry-run pull reports the plan with a checksum and writes nothing."""
    name = _first_template_name(client)
    output_dir = "output/api_pull_test"
    shutil.rmtree(Path(output_dir), ignore_errors=True)
    try:
        response = client.post(
            "/api/v1/pull",
            json={"name": name, "output_dir": output_dir, "dry_run": True},
        )
        assert response.status_code == 200
        envelope = _assert_envelope(response.json())
        data = envelope["data"]
        assert data["dry_run"] is True
        assert data["copied"] is False
        assert len(data["sha256"]) == 64
        assert not Path(data["destination"]).exists()
    finally:
        shutil.rmtree(Path(output_dir), ignore_errors=True)


@pytest.mark.unit
def test_pull_copies_template_file(client: TestClient) -> None:
    """A real pull copies the template file into the output directory."""
    name = _first_template_name(client)
    output_dir = "output/api_pull_test"
    shutil.rmtree(Path(output_dir), ignore_errors=True)
    try:
        response = client.post(
            "/api/v1/pull",
            json={"name": name, "output_dir": output_dir},
        )
        assert response.status_code == 200
        envelope = _assert_envelope(response.json())
        data = envelope["data"]
        assert data["copied"] is True
        assert Path(data["destination"]).is_file()
    finally:
        shutil.rmtree(Path(output_dir), ignore_errors=True)


@pytest.mark.unit
def test_pull_unknown_name_maps_to_404(client: TestClient) -> None:
    """An unknown template name mirrors the CLI KeyError as 404."""
    response = client.post(
        "/api/v1/pull",
        json={"name": "no-such-template-xyz", "output_dir": "output/api_pull_test"},
    )
    assert response.status_code == 404
    envelope = _assert_envelope(response.json(), status="error")
    assert "Unknown template" in envelope["error"]["message"]


@pytest.mark.unit
def test_pull_conflict_maps_to_409_then_overwrite_succeeds(
    client: TestClient,
) -> None:
    """A differing destination maps to 409 until overwrite is requested."""
    name = _first_template_name(client)
    output_dir = "output/api_pull_test"
    shutil.rmtree(Path(output_dir), ignore_errors=True)
    try:
        first = client.post(
            "/api/v1/pull",
            json={"name": name, "output_dir": output_dir},
        )
        assert first.status_code == 200
        envelope = _assert_envelope(first.json())
        destination = Path(envelope["data"]["destination"])
        destination.write_bytes(b"different bytes")
        conflict = client.post(
            "/api/v1/pull",
            json={"name": name, "output_dir": output_dir},
        )
        assert conflict.status_code == 409
        conflict_envelope = _assert_envelope(conflict.json(), status="error")
        assert "overwrite" in conflict_envelope["error"]["message"]
        overwrite_response = client.post(
            "/api/v1/pull",
            json={"name": name, "output_dir": output_dir, "overwrite": True},
        )
        assert overwrite_response.status_code == 200
        overwrite_envelope = _assert_envelope(overwrite_response.json())
        assert overwrite_envelope["data"]["overwritten"] is True
        assert overwrite_envelope["data"]["copied"] is True
    finally:
        shutil.rmtree(Path(output_dir), ignore_errors=True)


# ── GET /api/v1/models ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_models_happy_path_returns_registry_counts(client: TestClient) -> None:
    """Registry processing over a committed directory returns model counts."""
    response = client.get(
        "/api/v1/models", params={"target_dir": "input/gnn_files/discrete"}
    )
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    data = envelope["data"]
    assert data["exit_code"] == 0
    assert data["total_models"] >= 1
    assert isinstance(data["matching_models"], list)


@pytest.mark.unit
def test_models_ontology_query_filters_matches(client: TestClient) -> None:
    """An ontology query with no hits returns an empty matching list."""
    response = client.get(
        "/api/v1/models",
        params={
            "target_dir": "input/gnn_files/discrete",
            "query_ontology": "no-such-ontology-term-xyz",
        },
    )
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    assert envelope["data"]["matching_models"] == []
    assert envelope["data"]["query_ontology"] == "no-such-ontology-term-xyz"


@pytest.mark.unit
def test_registry_requests_are_isolated_and_leave_tracked_output_unchanged(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real concurrent registry snapshots cannot share or retain scratch files."""
    import gnn.model_registry

    tracked = Path("output/model_registry_api/model_registry.json")
    before = tracked.read_bytes()
    original = gnn.model_registry.process_model_registry
    scratch_paths: List[Path] = []

    def capture_scratch(**kwargs: Any) -> Any:
        scratch_paths.append(kwargs["output_dir"])
        return original(**kwargs)

    monkeypatch.setattr(gnn.model_registry, "process_model_registry", capture_scratch)

    def query(term: str) -> Any:
        return client.get(
            "/api/v1/models",
            params={"target_dir": "input/gnn_files/discrete", "query_ontology": term},
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(query, ["", "absent-a", "", "absent-b"]))
    assert all(response.status_code == 200 for response in responses)
    assert responses[0].json()["data"]["total_models"] > 0
    assert responses[1].json()["data"]["matching_models"] == []
    assert responses[3].json()["data"]["matching_models"] == []
    assert len(set(scratch_paths)) == 4
    assert all(not path.exists() for path in scratch_paths)
    assert tracked.read_bytes() == before


@pytest.mark.unit
def test_models_missing_target_dir_maps_to_404(client: TestClient) -> None:
    """A nonexistent target directory maps to 404 before processing."""
    response = client.get(
        "/api/v1/models", params={"target_dir": "output/no_such_target_dir"}
    )
    assert response.status_code == 404
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "not_found"


# ── POST /api/v1/preflight ───────────────────────────────────────────────────


def _fake_preflight_report(severities: List[str]) -> Callable[[Optional[Path]], Any]:
    """Build a run_preflight stub returning issues with the given severities."""
    from gnn.pipeline.preflight import PreflightIssue, PreflightReport

    def fake_run_preflight(config_path: Optional[Path] = None) -> Any:
        return PreflightReport(
            issues=[
                PreflightIssue(
                    category="dependency",
                    severity=severity,
                    message=f"synthetic {severity} issue",
                )
                for severity in severities
            ],
            checks_passed=5,
            checks_failed=sum(severity == "error" for severity in severities),
        )

    return fake_run_preflight


@pytest.mark.unit
def test_preflight_warnings_map_to_exit_code_2(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warning-severity preflight issues: 200 success envelope, exit_code 2."""
    monkeypatch.setattr(
        "gnn.pipeline.preflight.run_preflight",
        _fake_preflight_report(["warning", "info"]),
    )
    response = client.post("/api/v1/preflight", json={})
    assert response.status_code == 200
    envelope = _assert_envelope(response.json())
    data = envelope["data"]
    assert data["exit_code"] == 2
    assert data["is_ok"] is True
    assert any(issue["severity"] == "warning" for issue in data["issues"])


@pytest.mark.unit
def test_preflight_errors_map_to_400_with_issue_details(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Error-severity preflight issues (CLI exit 1) become a 400 envelope."""
    monkeypatch.setattr(
        "gnn.pipeline.preflight.run_preflight",
        _fake_preflight_report(["error", "warning"]),
    )
    response = client.post("/api/v1/preflight", json={})
    assert response.status_code == 400
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "bad_request"
    details = envelope["error"]["details"]
    assert isinstance(details, dict)
    assert any(issue["severity"] == "error" for issue in details["issues"])


@pytest.mark.unit
def test_preflight_real_call_returns_canonical_envelope(
    client: TestClient,
) -> None:
    """Wiring smoke against the real backend.

    The real environment's dependency state decides the branch (missing
    optional deps here produce error issues → 400; a fully provisioned env
    returns 200 with exit_code 0 or 2) — either way the canonical envelope
    contract must hold.
    """
    response = client.post("/api/v1/preflight", json={})
    envelope = _assert_envelope(
        response.json(), status="success" if response.status_code == 200 else "error"
    )
    if response.status_code == 200:
        assert envelope["data"]["exit_code"] in (0, 2)
    else:
        assert response.status_code == 400
        assert envelope["error"]["code"] == "bad_request"


# ── POST /api/v1/report ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_report_happy_path_writes_pipeline_report(client: TestClient) -> None:
    """Reporting over an existing directory writes PIPELINE_REPORT.md."""
    output_dir = Path("output/api_parity_report")
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    try:
        response = client.post("/api/v1/report", json={"output_dir": str(output_dir)})
        assert response.status_code == 200
        envelope = _assert_envelope(response.json())
        data = envelope["data"]
        assert data["exit_code"] == 0
        assert data["report_path"].endswith("PIPELINE_REPORT.md")
        assert data["report_chars"] > 0
        assert Path(data["report_path"]).is_file()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


@pytest.mark.unit
def test_report_missing_dir_maps_to_404(client: TestClient) -> None:
    """A nonexistent output directory maps to 404 (CLI exit 1 → not_found)."""
    response = client.post(
        "/api/v1/report",
        json={"output_dir": "output/no_such_report_dir"},
    )
    assert response.status_code == 404
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "not_found"


# ── POST /api/v1/reproduce ───────────────────────────────────────────────────


REPRODUCE_TEST_ROOT = Path("output/api_parity_reproduce_test")


def _index_reproduce_fixture() -> Dict[str, Any]:
    """Index one valid run under a repo-local scratch root (wave2 pattern)."""
    shutil.rmtree(REPRODUCE_TEST_ROOT, ignore_errors=True)
    source = REPRODUCE_TEST_ROOT / "input"
    source.mkdir(parents=True)
    (source / "model.md").write_text("model")
    args = PipelineArguments(
        target_dir=source, output_dir=REPRODUCE_TEST_ROOT / "output", only_steps="0"
    )
    summary = orchestrator._initialize_pipeline_summary(
        args, [("0_template.py", "Template")], {}
    )
    index_run(
        summary["run_hash"],
        REPRODUCE_TEST_ROOT / "summary.json",
        history_dir=REPRODUCE_TEST_ROOT / ".history",
        config={
            "args": args.to_dict(),
            "pipeline": {},
            "identity_config": summary["identity_config"],
            "run_hash_schema": summary["run_hash_schema"],
        },
        file_hashes=summary["file_hashes"],
    )
    return {
        "run_hash": summary["run_hash"],
        "history_dir": str(REPRODUCE_TEST_ROOT / ".history"),
        "selected_steps": list(summary["identity_config"]["selected_steps"]),
    }


@pytest.mark.unit
def test_reproduce_happy_path_returns_verified_identity(
    client: TestClient,
) -> None:
    """A matching indexed run resolves and verifies: 200 with its identity."""
    fixture = _index_reproduce_fixture()
    try:
        response = client.post(
            "/api/v1/reproduce",
            json={
                "run_hash": fixture["run_hash"],
                "history_dir": fixture["history_dir"],
            },
        )
        assert response.status_code == 200
        envelope = _assert_envelope(response.json())
        data = envelope["data"]
        assert data["run_hash"] == fixture["run_hash"]
        assert data["verified"] is True
        assert data["problems"] == []
        assert data["selected_steps"] == fixture["selected_steps"]
        assert data["args"]["only_steps"] == "0"
        assert data["input_config"]["pipeline"] == {}
    finally:
        shutil.rmtree(REPRODUCE_TEST_ROOT, ignore_errors=True)


@pytest.mark.unit
def test_reproduce_unknown_hash_maps_to_404(client: TestClient) -> None:
    """An unknown run hash mirrors the CLI EXIT_ERROR as 404 not_found."""
    fixture = _index_reproduce_fixture()
    try:
        response = client.post(
            "/api/v1/reproduce",
            json={
                "run_hash": "deadbeef0000",
                "history_dir": fixture["history_dir"],
            },
        )
        assert response.status_code == 404
        envelope = _assert_envelope(response.json(), status="error")
        assert envelope["error"]["code"] == "not_found"
        assert "not found" in envelope["error"]["message"]
    finally:
        shutil.rmtree(REPRODUCE_TEST_ROOT, ignore_errors=True)


@pytest.mark.unit
def test_reproduce_drifted_input_maps_to_400(client: TestClient) -> None:
    """Rewriting an indexed input file surfaces the drift problem as 400."""
    fixture = _index_reproduce_fixture()
    try:
        (REPRODUCE_TEST_ROOT / "input" / "model.md").write_text("changed")
        response = client.post(
            "/api/v1/reproduce",
            json={
                "run_hash": fixture["run_hash"],
                "history_dir": fixture["history_dir"],
            },
        )
        assert response.status_code == 400
        envelope = _assert_envelope(response.json(), status="error")
        assert envelope["error"]["code"] == "bad_request"
        assert "Cannot reproduce run" in envelope["error"]["message"]
        assert "differs from the indexed run" in envelope["error"]["message"]
    finally:
        shutil.rmtree(REPRODUCE_TEST_ROOT, ignore_errors=True)


@pytest.mark.unit
def test_reproduce_rejects_unknown_request_fields(client: TestClient) -> None:
    """Request models forbid extra fields explicitly (no silent ignoring)."""
    response = client.post(
        "/api/v1/reproduce",
        json={"run_hash": "abc123", "history_dir": None, "dry_run": False},
    )
    assert response.status_code == 422
    envelope = _assert_envelope(response.json(), status="error")
    assert envelope["error"]["code"] == "validation_error"
