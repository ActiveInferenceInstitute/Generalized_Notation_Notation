"""Backend options and genuine Python/CLI/REST/MCP rendering boundaries."""

from __future__ import annotations

import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gnn import parse_gnn_file, render_gnn_spec
from gnn.api import path_utils
from gnn.api.app import create_app
from gnn.frameworks import RENDER_FRAMEWORKS
from gnn.mcp.mcp import MCP
from gnn.render import process_render
from gnn.render.mcp import register_tools
from gnn.render.admission import render_options_inventory

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "input/gnn_files/discrete/two_state_bistable.md"


def test_real_openapi_and_registered_mcp_expose_the_same_typed_backend_inventory() -> (
    None
):
    inventory = render_options_inventory()
    assert set(inventory) == set(RENDER_FRAMEWORKS)
    options = create_app().openapi()["components"]["schemas"]["RenderRequest"][
        "properties"
    ]["options"]
    assert options["x-framework-options"] == inventory
    registry = _registry()
    tool = registry.get_tool_info("render_gnn_to_format")
    assert tool["schema"]["properties"]["options"]["x-framework-options"] == inventory
    info = registry.execute_tool("get_render_module_info", {})
    assert info["backend_options"] == inventory
    assert info["native_execution_ready"] is None
    assert inventory["thrml"]["properties"]["seed"]["maximum"] == 2**32 - 1
    assert inventory["cpomdp"]["properties"]["max_policies"] == {
        "type": "integer",
        "minimum": 1,
    }


def _registry() -> MCP:
    registry = MCP(
        enable_caching=False, enable_rate_limiting=False, strict_validation=True
    )
    register_tools(registry)
    return registry


def test_generic_jax_does_not_claim_to_honor_execution_contract_run_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    options = {"num_fast_transitions": 200}
    result = render_gnn_spec(
        parse_gnn_file(source), "jax", tmp_path / "python", options
    )
    assert result[0] is False and result[2] == []
    assert "unsupported" in result[1].lower()
    monkeypatch.setattr(path_utils, "get_repo_root", lambda: tmp_path)
    response = TestClient(create_app()).post(
        "/api/v1/render",
        json={
            "file_path": str(source),
            "framework": "jax",
            "output_dir": "rest",
            "options": options,
        },
    )
    assert response.status_code == 400, response.json()
    for name in ("render_spec_to_format", "render_gnn_to_format"):
        result = _registry().execute_tool(
            name,
            {
                "gnn_file_path": str(source),
                "framework": "jax",
                "output_directory": str(tmp_path / name),
                "options": options,
            },
        )
        assert result["success"] is False and not result.get("output_files"), result
    assert not list(tmp_path.rglob("*_jax.py"))


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source/model.md"
    source.parent.mkdir()
    shutil.copy2(MODEL, source)
    return source


def test_public_entrypoints_refuse_incompatible_gaussian_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source/model.md"
    source.parent.mkdir()
    shutil.copy2(ROOT / "input/gnn_files/continuous/continuous_navigation.md", source)
    direct = render_gnn_spec(parse_gnn_file(source), "pymdp", tmp_path / "python")
    assert direct[0] is False and direct[2] == []
    assert "continuous" in direct[1].lower()
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "gnn.cli",
            "render",
            str(source),
            "--framework",
            "pymdp",
            "--output",
            str(tmp_path / "cli.py"),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert cli.returncode != 0
    assert not (tmp_path / "cli.py").exists()
    monkeypatch.setattr(path_utils, "get_repo_root", lambda: tmp_path)
    response = TestClient(create_app()).post(
        "/api/v1/render",
        json={
            "file_path": str(source),
            "framework": "pymdp",
            "output_dir": "rest",
        },
    )
    assert response.status_code == 400, response.json()
    for name in ("render_spec_to_format", "render_gnn_to_format"):
        result = _registry().execute_tool(
            name,
            {
                "gnn_file_path": str(source),
                "framework": "pymdp",
                "output_directory": str(tmp_path / name),
            },
        )
        assert result["success"] is False and not result.get("output_files"), result
    assert not list(tmp_path.rglob("*_pymdp.py"))


@pytest.mark.parametrize("flag", ["strict", "compact"])
def test_parity_json_flags_reject_string_coercion(flag: str) -> None:
    endpoint = "validate" if flag == "strict" else "extract"
    response = TestClient(create_app()).post(
        f"/api/v1/{endpoint}",
        json={
            "file_path": "input/gnn_files/discrete/two_state_bistable.md",
            flag: "false",
        },
    )
    assert response.status_code == 422


def test_backend_error_sanitization_does_not_require_a_valid_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gnn.api import parity
    from gnn.api.path_utils import PathValidationError

    secret = "/private/operator/workspace"
    monkeypatch.setenv("GNN_API_ROOT", secret)

    def rejected_root() -> Path:
        raise PathValidationError("Configured API root is unavailable")

    monkeypatch.setattr(parity, "get_repo_root", rejected_root)
    message = parity._sanitize_detail(ValueError(f"Cannot render {secret}/model.md"))
    assert message == "Cannot render <redacted>/model.md"


@pytest.mark.parametrize(
    "options",
    [
        {"unknown": 1},
        {"num_samples": True},
        {"seed": "3"},
        {"thin": 0},
        {"seed": 2**32},
    ],
)
def test_invalid_backend_options_are_rejected_through_every_entrypoint(
    tmp_path: Path, options: dict
) -> None:
    source = _source(tmp_path)
    direct = render_gnn_spec(
        parse_gnn_file(source), "thrml", tmp_path / "python", options
    )
    assert direct[0] is False and direct[2] == []
    assert not (tmp_path / "python").exists()
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "gnn.cli",
            "render",
            str(source),
            "--framework",
            "thrml",
            "--options",
            json.dumps(options),
            "--output",
            str(tmp_path / "cli.py"),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert cli.returncode != 0
    assert not (tmp_path / "cli.py").exists()
    response = TestClient(create_app()).post(
        "/api/v1/render",
        json={"file_path": str(source), "framework": "thrml", "options": options},
    )
    assert response.status_code == 400
    assert response.json()["status"] == "error"
    assert "thrml" in str(response.json()["error"]).lower()
    for name in ("render_spec_to_format", "render_gnn_to_format"):
        result = _registry().execute_tool(
            name,
            {
                "gnn_file_path": str(source),
                "output_directory": str(tmp_path / name),
                "framework": "thrml",
                "options": options,
            },
        )
        assert result["success"] is False
        assert not result.get("output_files")
        assert not (tmp_path / name).exists()


def test_single_framework_mcp_does_not_render_other_backends_or_list_inherited_files(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    output = tmp_path / "out"
    stale = output / "jax/model_old_jax.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("# inherited\n")
    result = _registry().execute_tool(
        "render_gnn_to_format",
        {
            "gnn_file_path": str(source),
            "output_directory": str(output),
            "framework": "pymdp",
        },
    )
    assert result["success"] is True, result
    assert len(result["output_files"]) == 1
    assert "pymdp" in str(result["output_files"][0])
    assert str(stale) not in result["output_files"]
    assert stale.read_text() == "# inherited\n"
    receipt = json.loads((output / "render_processing_summary.json").read_text())
    assert receipt["total_framework_attempts"] == 1
    assert all(
        set(record["framework_results"]) == {"pymdp"}
        for record in receipt["file_results"].values()
    )
    discovery = _registry().execute_tool("list_render_frameworks", {})
    assert set(discovery["frameworks"]) == set(RENDER_FRAMEWORKS)
    assert discovery["availability_scope"] == "renderer_registration"
    assert discovery["native_execution_ready"] is None


def test_configured_and_explicit_backend_keys_preserve_unoverridden_options(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    output = tmp_path / "out"
    result = process_render(
        source.parent,
        output,
        frameworks=["thrml"],
        resolved_config={
            "render": {
                "backend_options": {
                    "thrml": {
                        "num_timesteps": 3,
                        "num_samples": 11,
                        "burn_in": 5,
                        "seed": 4,
                    }
                }
            }
        },
        backend_options={"thrml": {"seed": 6}},
    )
    assert result is True
    payload = runpy.run_path(str(next(output.rglob("*_thrml.py"))))["PAYLOAD"]
    assert payload["sampling"] == {
        "num_timesteps": 3,
        "num_samples": 11,
        "burn_in": 5,
        "thin": 2,
        "seed": 6,
    }


def test_actual_public_renderers_preserve_the_same_explicit_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    options = {
        "num_timesteps": 3,
        "num_samples": 11,
        "burn_in": 5,
        "thin": 2,
        "seed": 6,
    }
    direct = render_gnn_spec(
        parse_gnn_file(source), "thrml", tmp_path / "python", options
    )
    assert direct[0] is True, direct[1]
    artifacts = [Path(direct[2][0])]
    cli_path = tmp_path / "cli.py"
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "gnn.cli",
            "render",
            str(source),
            "--framework",
            "thrml",
            "--options",
            json.dumps(options),
            "--output",
            str(cli_path),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert cli.returncode == 0, cli.stderr
    artifacts.append(cli_path)
    # Configure the legacy path boundary only; the real HTTP parser, route,
    # admission and backend run unchanged. Installed GNN_API_ROOT acceptance
    # is provisioned separately with the filesystem/platform owners.
    monkeypatch.setattr(path_utils, "get_repo_root", lambda: tmp_path)
    response = TestClient(create_app()).post(
        "/api/v1/render",
        json={
            "file_path": str(source),
            "framework": "thrml",
            "options": options,
            "output_dir": "rest",
        },
    )
    assert response.status_code == 200, response.text
    artifacts.append(Path(response.json()["data"]["artifact"]))
    for name in ("render_spec_to_format", "render_gnn_to_format"):
        result = _registry().execute_tool(
            name,
            {
                "gnn_file_path": str(source),
                "output_directory": str(tmp_path / name),
                "framework": "thrml",
                "options": options,
            },
        )
        assert result["success"] is True, result
        artifacts.append(Path(result["output_files"][0]))
    payloads = [runpy.run_path(str(artifact))["PAYLOAD"] for artifact in artifacts]
    assert all(payload["sampling"] == options for payload in payloads)
    assert all(
        payload["source_identity"]["semantic_sha256"]
        == payloads[0]["source_identity"]["semantic_sha256"]
        for payload in payloads
    )


@pytest.mark.parametrize(
    "frameworks", [[], ["pymdp", "unknown"], ["pymdp", "pymdp"], [True]]
)
def test_backend_selection_never_silently_widens_or_reduces(
    tmp_path: Path, frameworks: object
) -> None:
    source = _source(tmp_path)
    output = tmp_path / "out"
    result = process_render(source.parent, output, frameworks=frameworks)
    assert result is False
    assert not output.exists()
