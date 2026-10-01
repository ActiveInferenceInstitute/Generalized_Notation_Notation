"""Public continuous dispatch and execution output contracts."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import pytest

from gnn.render.processor import render_gnn_spec


def continuous_spec() -> dict[str, Any]:
    return {
        "model_name": "OU",
        "model_kind": "continuous",
        "initialparameterization": {
            "F": [[0.5]],
            "H": [[1.0]],
            "Q": [[0.75]],
            "R": [[1.0]],
            "prior_mean": [1.0],
            "prior_cov": [[2.0]],
        },
        "model_parameters": {"num_timesteps": 3, "dt": 1.0, "random_seed": 7},
    }


@pytest.mark.parametrize(
    "target", ["jax", "numpyro", "pytorch", "rxinfer", "stan", "ngclearn", "cpomdp"]
)
def test_public_continuous_dispatch(tmp_path: Path, target: str) -> None:
    ok, message, artifacts = render_gnn_spec(continuous_spec(), target, tmp_path)
    assert ok, message
    assert artifacts and all(Path(path).is_file() for path in artifacts)
    contents = "\n".join(Path(path).read_text() for path in artifacts)
    if target == "rxinfer":
        match = re.search(r'const GNN_SPEC_JSON_B64 = "([A-Za-z0-9+/=]+)"', contents)
        assert match is not None
        payload = json.loads(base64.b64decode(match.group(1), validate=True))
        assert payload["initialparameterization"]["Q"] == [[0.75]]
    else:
        assert "0.75" in contents
    if target == "stan":
        assert {Path(path).suffix for path in artifacts} >= {".stan", ".py"}


@pytest.mark.parametrize(
    "target", ["pymdp", "activeinference_jl", "jax_pomdp", "discopy"]
)
def test_unsupported_continuous_target_rejects(tmp_path: Path, target: str) -> None:
    ok, message, artifacts = render_gnn_spec(continuous_spec(), target, tmp_path)
    assert not ok and not artifacts
    assert "continuous" in message.lower()


def test_jax_runner_routes_results_to_requested_directory(tmp_path: Path) -> None:
    from gnn.execute.jax.jax_runner import execute_jax_script

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "output_jax.py"
    script.write_text(
        "import json, os\nfrom pathlib import Path\n"
        "out = Path(os.environ.get('GNN_OUTPUT_DIR', 'fallback'))\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'simulation_results.json').write_text(json.dumps({'routed': True}))\n"
    )
    output = tmp_path / "requested"
    assert execute_jax_script(script, output_dir=output, timeout=30)
    assert json.loads((output / "simulation_results.json").read_text()) == {
        "routed": True
    }
    assert not (scripts / "fallback").exists()


def test_public_jax_runner_executes_nonstationary_prediction(tmp_path: Path) -> None:
    """Three actual steps distinguish transition/update from repeated prior updates."""
    from gnn.execute.jax.jax_runner import execute_jax_script

    ok, message, artifacts = render_gnn_spec(
        continuous_spec(), "jax", tmp_path / "render"
    )
    assert ok, message
    output = tmp_path / "execution"
    assert execute_jax_script(Path(artifacts[0]), output_dir=output, timeout=60)
    result = json.loads((output / "simulation_results.json").read_text())
    assert result["validation"]["all_valid"] is True
    assert result["model_kind"] == "continuous"
    variances = [2 / 3, 11 / 23, 20 / 43]
    assert [row[0][0] for row in result["posterior_cov"]] == pytest.approx(variances)
    mean = 1.0
    for index, (observation, variance) in enumerate(
        zip(result["observations_continuous"], variances, strict=True)
    ):
        predicted_mean = mean if index == 0 else 0.5 * mean
        mean = predicted_mean + variance * (observation[0] - predicted_mean)
        assert result["beliefs"][index][0] == pytest.approx(mean)
    assert result["control_mode"] == "passive"
    assert result["actions"] == []
