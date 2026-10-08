"""Public JAX saved-source artifacts retain authored Unicode as valid UTF-8."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable

import pytest

from gnn.render.jax import (
    render_gnn_to_jax,
    render_gnn_to_jax_combined,
    render_gnn_to_jax_pomdp,
)


@pytest.mark.parametrize(
    "renderer",
    [render_gnn_to_jax, render_gnn_to_jax_pomdp, render_gnn_to_jax_combined],
)
def test_saved_jax_source_preserves_authored_unicode(
    tmp_path: Path, renderer: Callable
) -> None:
    model_name = "sensor_\u03bc_\u221d_observation"
    spec = {
        "model_name": model_name,
        "model_parameters": {
            "num_states": 2,
            "num_obs": 2,
            "num_actions": 1,
            "num_timesteps": 3,
            "b_tensor_order": "next_state_previous_state_action",
        },
        "initialparameterization": {
            "A": [[0.8, 0.3], [0.2, 0.7]],
            "B": [[[0.8], [0.3]], [[0.2], [0.7]]],
            "C": [0.2, -0.1],
            "D": [0.6, 0.4],
        },
    }
    script = tmp_path / "saved" / "model.py"
    success, reason, artifacts = renderer(spec, script)
    assert success, reason
    assert artifacts == [str(script)]
    saved = script.read_bytes()
    assert model_name.encode("utf-8") in saved
    decoded = saved.decode("utf-8", errors="strict")
    assert model_name in decoded
    assert isinstance(ast.parse(decoded, filename=str(script)), ast.Module)
    assert spec["model_name"] == model_name
