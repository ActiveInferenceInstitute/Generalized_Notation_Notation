"""Regression tests for canonical POMDP rendering edge cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gnn.extract.pomdp_extractor import extract_pomdp_from_file
from gnn.render.pomdp_processor import POMDPRenderProcessor
from gnn.render.processor import render_gnn_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
DISCRETE_DIR = REPO_ROOT / "input" / "gnn_files" / "discrete"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("filename", "expected_shape"),
    [
        ("actinf_pomdp_agent.md", (3, 3, 3)),
    ],
)
@pytest.mark.parametrize("framework", ["rxinfer", "activeinference_jl"])
def test_discrete_pomdp_models_render_to_julia_frameworks(
    filename: str, expected_shape: tuple[int, int, int], framework: str, tmp_path: Path
) -> None:
    pomdp = extract_pomdp_from_file(DISCRETE_DIR / filename, strict_validation=True)
    assert pomdp is not None
    spec = POMDPRenderProcessor(tmp_path)._pomdp_to_gnn_spec(pomdp)

    b_matrix = np.asarray(spec["initialparameterization"]["B"], dtype=float)
    assert b_matrix.shape == expected_shape
    assert spec["model_parameters"]["num_hidden_states"] == expected_shape[0]
    assert spec["model_parameters"]["num_actions"] == expected_shape[2]
    assert (
        spec["model_parameters"]["b_tensor_order"] == "next_state_previous_state_action"
    )

    ok, message, artifacts = render_gnn_spec(spec, framework, tmp_path / framework)

    assert ok, message
    assert artifacts, "render should produce artifacts"
    rendered = Path(artifacts[0])
    assert rendered.exists()
    text = rendered.read_text(encoding="utf-8")
    assert "using Base64" in text
    assert "base64decode" in text
    assert "JSON.parse(raw" not in text


def test_pomdp_bnlearn_renderer_sanitizes_output_filename(tmp_path: Path) -> None:
    output_dir = tmp_path / "bnlearn"
    output_dir.mkdir()
    processor = POMDPRenderProcessor(tmp_path)

    result = processor._call_bnlearn_renderer(
        {
            "name": "../../escape",
            "model_name": "../../escape",
            "model_parameters": {},
        },
        output_dir,
    )

    assert result["success"] is True
    artifact_path = Path(result["artifacts"][0]).resolve()
    artifact_path.relative_to(output_dir.resolve())
    assert artifact_path.name == "escape_bnlearn.py"
    assert not (tmp_path.parent / "escape_bnlearn.py").exists()


@pytest.mark.parametrize("framework", ["rxinfer", "activeinference_jl"])
def test_episodic_objective_requires_explicit_backend_support(framework, tmp_path):
    pomdp = extract_pomdp_from_file(
        DISCRETE_DIR / "tmaze_epistemic.md", strict_validation=True
    )
    assert pomdp is not None
    processor = POMDPRenderProcessor(tmp_path)
    spec = processor._pomdp_to_gnn_spec(pomdp)
    assert spec["canonical_pomdp_schema"] == "raw_discrete_components_v1"
    result = processor._process_single_framework(pomdp, framework)
    assert result["status"] == "unsupported"
    assert "episodic_contingent_v1" in result["message"]
    ok, message, artifacts = render_gnn_spec(spec, framework, tmp_path / framework)
    assert not ok and not artifacts
    assert message.startswith("unsupported-execution-contract:")
    assert not list(tmp_path.rglob("*.jl"))
