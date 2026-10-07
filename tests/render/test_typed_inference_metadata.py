"""Typed public rendering preserves the source inference contract."""

import importlib.util
from pathlib import Path

import pytest

from gnn.parsers.markdown_parser import MarkdownGNNParser
from gnn.render.processor import render_gnn_spec
from gnn.render.processor.parsing import _internal_representation_to_mapping


@pytest.mark.parametrize(
    "source, static",
    [
        ("basics/static_perception", True),
        ("basics/dynamic_perception", False),
        ("discrete/hmm_baseline", False),
    ],
)
def test_typed_public_render_preserves_inference_metadata(tmp_path, source, static):
    root = Path(__file__).resolve().parents[2]
    parsed = MarkdownGNNParser().parse_file(
        str(root / "input/gnn_files" / f"{source}.md")
    )
    assert parsed.success
    model = parsed.model
    before = model.to_dict()
    mapping = _internal_representation_to_mapping(model)
    assert mapping["time_specification"] == before["time_specification"]
    assert mapping["equations"] == before["equations"]
    if not static:
        assert mapping["model_parameters"]["passive_model"] is True
        assert mapping["matrix_provenance"]["C"]["derived"] is True
        assert mapping["matrix_provenance"]["C"]["source"] == "passive_model_adapter"
        assert all(value == 0 for value in mapping["initialparameterization"]["C"])
        assert "passive_model_zero_preferences" in mapping["adapter_notes"]
    assert model.to_dict() == before
    success, message, artifacts = render_gnn_spec(model, "jax", tmp_path)
    assert success, message
    script = next(Path(path) for path in artifacts if path.endswith(".py"))
    loader = importlib.util.spec_from_file_location("typed_inference", script)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    assert module.STATIC_MODEL is static
    assert module.PASSIVE_MODEL is (not static)
    assert module.INFERENCE_ESTIMAND == (
        "static_conditioning" if static else "filtering"
    )
    trajectory = module.run_simulation(module.create_params(), 1 if static else 2)
    assert len(trajectory["actions"]) == 0
    assert model.to_dict() == before
