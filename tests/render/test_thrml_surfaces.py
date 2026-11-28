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
from gnn.render import process_render, render_gnn_spec
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


@pytest.mark.parametrize(
    ("supplied", "message"),
    [
        ("{oops", "simulation_params must be valid JSON"),
        ("null", "simulation_params must be a JSON object"),
        ("[]", "simulation_params must be a JSON object"),
        ("7", "simulation_params must be a JSON object"),
        ('{"seed":7}', "Use --timesteps"),
        ('{"thrml":{"seed":7}}', "requires selecting --frameworks thrml"),
        ('{"thrml":[]}', "simulation_params.thrml must be an object"),
        ('{"thrml":{"seed":"7"}}', "thrml.seed must be"),
        ('{"thrml":{"observations":[NaN]}}', "Invalid constant NaN"),
        ('{"thrml":{"seed":2,"seed":7}}', "contains duplicate key"),
    ],
)
def test_simulation_parameters_refuse_before_rendering(
    tmp_path: Path, caplog, supplied: str, message: str
) -> None:
    source = _source(tmp_path)
    output = tmp_path / "render"
    output.mkdir()
    existing = output / "previous_model.py"
    existing.write_text("previous model\n", encoding="utf-8")
    assert (
        process_render(
            source.parent, output, frameworks=["jax"], simulation_params=supplied
        )
        is False
    )
    assert message in caplog.text
    assert list(output.iterdir()) == [existing]
    assert existing.read_text(encoding="utf-8") == "previous model\n"


def test_simulation_parameters_bind_only_to_selected_thrml(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "render"
    assert (
        process_render(
            source.parent,
            output,
            frameworks=["jax", "thrml"],
            simulation_params='{"thrml":{"num_samples":23,"seed":7}}',
        )
        is True
    )
    assert list(output.rglob("*_jax.py"))
    payload = _payload(next(output.rglob("*_thrml.py")))
    assert payload["sampling"]["num_samples"] == 23
    assert payload["sampling"]["seed"] == 7


def test_direct_spec_cannot_silently_drop_simulation_parameters(tmp_path: Path) -> None:
    from gnn.extract.pomdp_extractor import extract_pomdp_from_file
    from gnn.render.pomdp_processor import pomdp_to_gnn_spec

    source = _source(tmp_path)
    parsed = extract_pomdp_from_file(source, on_error="raise")
    assert parsed is not None
    spec = pomdp_to_gnn_spec(parsed)
    spec["model_parameters"]["simulation_params"] = {"thrml": {"num_samples": 23}}
    output = tmp_path / "direct"
    ok, message, files = render_gnn_spec(spec, "jax", output)
    assert not ok and "requires selecting --frameworks thrml" in message and not files
    assert not output.exists()
    ok, message, files = render_gnn_spec(spec, "thrml", output)
    assert ok, message
    assert _payload(Path(files[0]))["sampling"]["num_samples"] == 23


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
    from gnn.utils.runtime_safety import workspace_paths as path_utils

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
