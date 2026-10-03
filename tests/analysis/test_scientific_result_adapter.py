"""Native marginal and Gaussian result semantics are never flattened away."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from gnn.analysis.result_adapter import (
    continuous_result_metrics,
    result_views,
    structured_result_data,
)
from gnn.analysis.rxinfer.analyzer import create_rxinfer_visualizations
from gnn.analysis.rxinfer.animator import generate_animated_html
from gnn.analysis.rxinfer.gif_animator import generate_gif_animation


def _agents() -> dict:
    return {
        "model_kind": "multi_agent",
        "agents": ["agent1", "agent2"],
        "num_timesteps": 2,
        "beliefs_by_agent": {
            "agent1": [[0.9, 0.1], [0.8, 0.2]],
            "agent2": [[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]],
        },
        "actions_by_agent": {"agent1": [0, 1], "agent2": [1, 0]},
        "true_states_by_agent": {"agent1": [0, 0], "agent2": [2, 2]},
    }


def _gaussian() -> dict:
    return {
        "model_kind": "continuous",
        "num_timesteps": 2,
        "beliefs": [[-2.0, 3.0], [-1.0, 2.0]],
        "posterior_cov": [[[0.4, 0.1], [0.1, 0.3]], [[0.2, 0.0], [0.0, 0.1]]],
        "true_states_continuous": [[-2.0, 2.0], [-1.0, 1.0]],
        "controls": [[0.1, 0.2], [0.2, 0.3]],
    }


def test_native_agents_have_identity_bound_shapes_and_traces() -> None:
    data = _agents()
    views = result_views(data)
    assert [len(view["beliefs"][0]) for view in views.values()] == [2, 3]
    assert views["agent2"]["actions"] == [1, 0]
    assert views["agent1"]["observations"] == []
    assert structured_result_data(data)["beliefs_by_agent"] == data["beliefs_by_agent"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("beliefs_by_agent", {"agent1": [[0.9, 0.1], [0.8]]}),
        ("agents", ["agent1", "missing"]),
        ("num_timesteps", 3),
    ],
)
def test_invalid_identity_or_ragged_trace_rejected(key: str, value: object) -> None:
    data = _agents()
    data[key] = value
    with pytest.raises(ValueError):
        result_views(data)


def test_gaussian_covariance_is_required_and_validated() -> None:
    data = _gaussian()
    assert continuous_result_metrics(data)["rmse_vs_true"] > 0
    assert result_views(data)["continuous"]["beliefs"][0][0] == -2
    data["posterior_cov"][0][0][0] = -1
    with pytest.raises(ValueError, match="semidefinite"):
        continuous_result_metrics(data)


@pytest.mark.parametrize("family", ["agents", "gaussian"])
def test_all_visual_formats_accept_the_same_native_family(
    tmp_path: Path, family: str
) -> None:
    data = _agents() if family == "agents" else _gaussian()
    pngs = create_rxinfer_visualizations(data, tmp_path, family)
    html = generate_animated_html(data, tmp_path / "animation.html", family)
    gif = generate_gif_animation(data, tmp_path / "animation.gif", family, dpi=35)
    assert pngs and Path(html).is_file()
    with Image.open(gif) as image:
        assert image.n_frames == 2
    manifest = json.loads((tmp_path / "animation.manifest.json").read_text())
    assert manifest["views"] == (
        ["agent1", "agent2"] if family == "agents" else ["continuous"]
    )
    text = Path(html).read_text()
    assert ("agent2" in text) if family == "agents" else ("Gaussian intervals" in text)


def test_structured_extraction_retains_extension_scientific_fields() -> None:
    from gnn.analysis.result_adapter import structured_result_data

    data = {
        "beliefs": [[0.3, 0.7]],
        "num_timesteps": 1,
        "env_signal_trace": [[0.1, 0.2]],
        "selected_efe_by_agent": {"agent1": [-2.0]},
        "source_identity": {"source_sha256": "abc"},
    }
    assert structured_result_data(data) == data


@pytest.mark.parametrize("timesteps", [True, 1.5, 0, 2])
def test_flat_categorical_declared_timesteps_are_not_truncated(timesteps) -> None:
    with pytest.raises(ValueError, match="num_timesteps"):
        result_views({"beliefs": [[0.2, 0.8]], "num_timesteps": timesteps})
