"""Real render configuration across pipeline, CLI, REST and MCP surfaces."""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gnn.api.app import create_app
from gnn.api.server import create_app as create_job_app
from gnn.cli import main as cli_main
from gnn.render import process_render
from gnn.render.mcp import render_spec_to_format_mcp


def _source(tmp_path: Path) -> Path:
    source = (
        Path(__file__).parents[2] / "input/gnn_files/discrete/two_state_bistable.md"
    )
    destination = tmp_path / "source" / source.name
    destination.parent.mkdir()
    shutil.copy2(source, destination)
    return destination


def _payload(path: Path) -> dict:
    # Loading generated constants exercises serialization; native execution has
    # separate toolchain acceptance and is deliberately not claimed here.
    return runpy.run_path(str(path))["PAYLOAD"]


def test_pipeline_configuration_and_explicit_override(tmp_path: Path) -> None:
    source = _source(tmp_path)
    out = tmp_path / "render"
    result = process_render(
        source.parent,
        out,
        frameworks=["thrml"],
        resolved_config={
            "render": {"backend_options": {"thrml": {"num_samples": 17, "seed": 8}}}
        },
        simulation_params=json.dumps({"thrml": {"num_samples": 23}}),
        timesteps=3,
    )
    assert result is True
    payload = _payload(next(out.rglob("*_thrml.py")))
    assert payload["sampling"] == {
        "num_timesteps": 3,
        "num_samples": 23,
        "burn_in": 256,
        "thin": 2,
        "seed": 8,
    }


def test_cli_and_mcp_apply_exact_options(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "cli_thrml.py"
    options = {
        "num_timesteps": 3,
        "num_samples": 19,
        "burn_in": 7,
        "thin": 1,
        "seed": 4,
    }
    assert (
        cli_main(
            [
                "render",
                str(source),
                "--framework",
                "thrml",
                "--options",
                json.dumps(options),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert _payload(output)["sampling"] == options
    result = render_spec_to_format_mcp(
        str(source), str(tmp_path / "mcp"), "thrml", options=options
    )
    assert result["success"], result
    assert _payload(Path(result["output_files"][0]))["sampling"] == options


@pytest.mark.parametrize("factory", [create_app, create_job_app])
def test_rest_schema_and_options_are_composable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, factory: object
) -> None:
    from gnn.api import path_utils

    source = _source(tmp_path)
    monkeypatch.setattr(path_utils, "get_repo_root", lambda: tmp_path)
    monkeypatch.setenv("GNN_RATE_LIMIT", "0")
    client = TestClient(factory())
    options = {"num_timesteps": 3, "num_samples": 11, "seed": 6}
    response = client.post(
        "/api/v1/render",
        json={
            "file_path": str(source),
            "framework": "thrml",
            "output_dir": "rest",
            "options": options,
        },
    )
    assert response.status_code == 200, response.text
    payload = _payload(Path(response.json()["data"]["artifact"]))
    assert all(payload["sampling"][key] == value for key, value in options.items())
    invalid = client.post(
        "/api/v1/render",
        json={
            "file_path": str(source),
            "framework": "thrml",
            "output_dir": "invalid",
            "options": {"num_samples": True},
        },
    )
    assert invalid.status_code == 400
    assert not list((tmp_path / "invalid").rglob("*_thrml.py"))


def test_unsupported_mapping_is_not_a_failed_render(tmp_path: Path) -> None:
    source = _source(tmp_path)
    # A structural zero is scientifically valid but outside this Gibbs mapping.
    source.write_text(
        source.read_text().replace(
            "(0.8, 0.2),\n  (0.2, 0.8)", "(1.0, 0.2),\n  (0.0, 0.8)"
        )
    )
    out = tmp_path / "render"
    process_render(source.parent, out, frameworks=["thrml"])
    summary = json.loads((out / "render_processing_summary.json").read_text())
    assert len(summary["unsupported_framework_renderings"]) == 1
    assert summary["failed_framework_renderings"] == []
    assert not list(out.rglob("*_thrml.py"))


def test_render_failure_and_stale_artifacts_cannot_become_success(
    tmp_path: Path,
) -> None:
    from gnn.cli.commands import find_render_artifact, render_processing_succeeded

    assert not render_processing_succeeded(False)
    assert render_processing_succeeded(0)
    assert render_processing_succeeded(True)
    old = tmp_path / "old" / "thrml" / "old_thrml.py"
    old.parent.mkdir(parents=True)
    old.write_text("historical artifact")
    (tmp_path / "render_processing_summary.json").write_text(
        json.dumps(
            {
                "file_results": {
                    "current": {
                        "framework_results": {
                            "thrml": {
                                "success": False,
                                "unsupported": True,
                                "output_files": [],
                            }
                        }
                    }
                },
            }
        )
    )
    assert find_render_artifact(tmp_path, "thrml", require_current=True) is None
    assert find_render_artifact(tmp_path, "thrml") == old
