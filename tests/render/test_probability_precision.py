"""Scientific values survive emission; invalid models never become demos."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gnn.render.numpyro.numpyro_renderer import _generate_numpyro_code
from gnn.render.pomdp_contract import (
    build_canonical_pomdp_spec,
    normalise_matrix_columns,
    normalise_vector,
)
from gnn.render.spec_matrices import extract_abcd_matrices, format_array_literal


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 2, 3)])
def test_literal_round_trip_preserves_tiny_and_precise_values(
    shape: tuple[int, ...],
) -> None:
    values = np.resize([1 / 3, 1e-12, 1 - 1e-12], np.prod(shape)).reshape(shape)
    literal = format_array_literal(values, prefix="array")
    actual = np.asarray(ast.literal_eval(literal[6:-1]))
    assert np.array_equal(actual, values)


@pytest.mark.parametrize(
    "value", [[-0.1, 1.1], [np.nan, 1], [np.inf, 1], [0, 0], [0.4, 0.4]]
)
def test_invalid_probability_vectors_are_rejected(value: list[float]) -> None:
    with pytest.raises(ValueError):
        normalise_vector(value, name="D")


def test_invalid_columns_and_missing_declared_matrices_are_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        normalise_matrix_columns([[-0.1], [1.1]], name="A")
    with pytest.raises(ValueError, match="Missing declared"):
        extract_abcd_matrices({})
    with pytest.raises(ValueError, match="finite"):
        format_array_literal(np.array([np.nan]), prefix="array")


def test_small_mass_correction_has_source_custody() -> None:
    source = {
        "initialparameterization": {
            "A": [[1, 0], [0, 1]],
            "B": [[1, 0], [0, 1]],
            "C": [0, 0],
            "D": [0.49999, 0.5],
        }
    }
    result = build_canonical_pomdp_spec(source)
    assert result["matrix_provenance"]["D"]["derived"] is True
    assert result["matrix_provenance"]["D"]["source_values"] == [0.49999, 0.5]
    assert source["initialparameterization"]["D"] == [0.49999, 0.5]


@pytest.mark.parametrize("states", [3, 256])
def test_emitted_prior_passes_real_numpyro_simplex(states: int) -> None:
    import jax.numpy as jnp
    import numpyro.distributions as dist

    prior = np.ones(states) / states
    literal = format_array_literal(prior, prefix="array")
    emitted = jnp.asarray(ast.literal_eval(literal[6:-1]))
    dist.Categorical(probs=emitted, validate_args=True)


@pytest.mark.parametrize("states", [3, 256])
def test_generated_numpyro_script_executes_real_probability_model(
    tmp_path: Path, states: int
) -> None:
    script = tmp_path / "precise.py"
    script.write_text(
        _generate_numpyro_code(
            "precision",
            np.eye(states),
            np.eye(states),
            np.zeros(states),
            np.ones(states) / states,
            {"num_timesteps": 2},
        )
    )
    env = {**os.environ, "NUMPYRO_OUTPUT_DIR": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "simulation_results.json").read_text())
    assert payload["num_timesteps"] == 2
    assert payload["validation"]["all_valid"] is True


def test_bistable_axis_migration_preserves_asymmetric_transition_values() -> None:
    from gnn.extract.pomdp_extractor import extract_pomdp_from_file
    from gnn.render.pomdp_processor import pomdp_to_gnn_spec

    source = extract_pomdp_from_file(
        Path("input/gnn_files/discrete/two_state_bistable.md"), strict_validation=True
    )
    spec = pomdp_to_gnn_spec(source)
    raw = np.asarray(source.matrices["B"])
    np.testing.assert_array_equal(
        spec["initialparameterization"]["B"], raw.transpose(1, 2, 0)
    )
    assert (
        spec["matrix_provenance"]["B"]["source_order"]
        == "action_next_state_previous_state"
    )
    assert spec["matrix_provenance"]["B"]["normalized"] is False


def test_materially_invalid_source_likelihood_is_not_normalized() -> None:
    with pytest.raises(ValueError, match="mass must be one"):
        normalise_matrix_columns([[0.9, 0.1], [0.2, 0.8]], name="A")


@pytest.mark.parametrize("name", ["static_perception.md", "dynamic_perception.md"])
def test_maintained_invalid_likelihood_sources_fail_required_render_without_repair(
    tmp_path: Path, name: str
) -> None:
    from gnn.extract.pomdp_extractor import extract_pomdp_from_file
    from gnn.render.pomdp_processor import POMDPRenderProcessor

    source = Path("input/gnn_files/basics") / name
    space = extract_pomdp_from_file(source)
    raw = np.asarray(space.matrices["A"]).copy()
    np.testing.assert_allclose(raw.sum(axis=0), [1.1, 0.9])
    processor = POMDPRenderProcessor(tmp_path)
    with pytest.raises(
        ValueError, match="A column 0 probability mass must be one, got 1.1"
    ):
        processor._pomdp_to_gnn_spec(space)
    rendered = processor.process_pomdp_for_all_frameworks(
        space, source, frameworks=["numpyro"]
    )
    receipt = rendered["framework_results"]["numpyro"]
    assert receipt["success"] is False and receipt["status"] == "failed"
    assert not receipt["output_files"]
    np.testing.assert_array_equal(space.matrices["A"], raw)


def test_uniform_precision_source_correction_is_explicit_and_original_rejected() -> (
    None
):
    from gnn.extract.pomdp_extractor import extract_pomdp_from_file

    source = extract_pomdp_from_file(
        Path("input/gnn_files/precision/precision_weighted.md")
    )
    for key in ("D", "E"):
        np.testing.assert_array_equal(source.matrices[key], [1 / 3] * 3)
        with pytest.raises(ValueError, match="mass must be one"):
            normalise_vector([0.333] * 3, name=key)


@pytest.mark.parametrize(
    "order", ["action_next_state_previous_state", "action_previous_state_next_state"]
)
@pytest.mark.parametrize(
    "declaration_key", ["b_tensor_order", "B_tensor_order", "transition_tensor_order"]
)
@pytest.mark.parametrize("spelling", ["canonical", "uppercase_hyphens"])
def test_direct_extractors_honor_asymmetric_declared_axes(
    order: str, declaration_key: str, spelling: str
) -> None:
    from copy import deepcopy

    actions = np.array(
        [
            [[0.8, 0.15, 0.1], [0.12, 0.7, 0.25], [0.08, 0.15, 0.65]],
            [[0.15, 0.2, 0.05], [0.25, 0.6, 0.25], [0.6, 0.2, 0.7]],
        ]
    )
    declared = (
        actions
        if order == "action_next_state_previous_state"
        else actions.transpose(0, 2, 1)
    )
    spec = {
        "model_parameters": {
            declaration_key: order
            if spelling == "canonical"
            else order.upper().replace("_", "-"),
        },
        "initialparameterization": {
            "A": np.eye(3).tolist(),
            "B": declared.tolist(),
            "C": [0, 0, 0],
            "D": [0.2, 0.3, 0.5],
        },
    }
    before = deepcopy(spec)
    provenance = {}
    _, actual, _, _ = extract_abcd_matrices(spec, transformation_provenance=provenance)
    np.testing.assert_allclose(actual, actions.transpose(1, 2, 0), rtol=0, atol=1e-15)
    assert actual.shape == (3, 3, 2)
    assert provenance["B"]["source_values"] == declared.tolist()
    assert provenance["B"]["orientation_transformed"] is True
    assert provenance["B"]["source_order"] == order
    assert spec == before


@pytest.mark.parametrize("framework", ["numpyro", "pytorch"])
def test_direct_renderer_records_accepted_rounding_without_mutating_source(
    framework: str,
    tmp_path: Path,
) -> None:
    from copy import deepcopy
    import importlib

    spec = {
        "initialparameterization": {
            "A": [[0.79999, 0], [0.2, 1]],
            "B": [[1, 0], [0, 1]],
            "C": [0, 1],
            "D": [0.49999, 0.5],
        }
    }
    before = deepcopy(spec)
    renderer = importlib.import_module(f"gnn.render.{framework}.{framework}_renderer")
    output = tmp_path / f"{framework}.py"
    ok, message, _ = getattr(renderer, f"render_gnn_to_{framework}")(spec, output)
    assert ok, message
    tree = ast.parse(output.read_text())
    emitted = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and any(
            isinstance(key, ast.Constant) and key.value == "matrix_provenance"
            for key in node.keys
        )
    )
    provenance = ast.literal_eval(
        next(
            value
            for key, value in zip(emitted.keys, emitted.values)
            if isinstance(key, ast.Constant) and key.value == "matrix_provenance"
        )
    )
    for name in ("A", "D"):
        assert (
            provenance[name]["source_values"] == before["initialparameterization"][name]
        )
        assert provenance[name]["derived"] is True
        assert provenance[name]["canonical_values"] != provenance[name]["source_values"]
    assert spec == before


def test_explicit_demo_defaults_have_transformation_provenance() -> None:
    provenance = {}
    extract_abcd_matrices({}, allow_adapters=True, transformation_provenance=provenance)
    assert set(provenance) == {"A", "B", "C", "D"}
    assert all(
        value["default_supplied"]
        and value["source_values"] is None
        and value["policy"] == "explicit_demo_adapter"
        for value in provenance.values()
    )


def test_unknown_declared_axes_are_rejected_by_direct_extractors() -> None:
    with pytest.raises(ValueError, match="Unsupported declared B tensor order"):
        extract_abcd_matrices(
            {
                "model_parameters": {"b_tensor_order": "unrecognized"},
                "initialparameterization": {
                    "A": [[1, 0], [0, 1]],
                    "B": np.repeat(np.eye(2)[:, :, None], 2, axis=2).tolist(),
                    "C": [0, 0],
                    "D": [0.5, 0.5],
                },
            }
        )


@pytest.mark.parametrize("name", ["num_hidden_states", "num_obs"])
@pytest.mark.parametrize("value", [True, 1.5, "2", 0, -1])
def test_direct_extractor_rejects_invalid_declared_dimensions(
    name: str, value: object
) -> None:
    spec = {
        "model_parameters": {name: value},
        "initialparameterization": {
            "A": [[1, 0], [0, 1]],
            "B": [[1, 0], [0, 1]],
            "C": [0, 0],
            "D": [0.5, 0.5],
        },
    }
    with pytest.raises(ValueError, match="positive integer"):
        extract_abcd_matrices(spec)


@pytest.mark.parametrize("name", ["num_hidden_states", "num_obs"])
def test_direct_extractor_rejects_matrix_dimension_declaration_mismatch(
    name: str,
) -> None:
    spec = {
        "model_parameters": {name: 3},
        "initialparameterization": {
            "A": [[1, 0], [0, 1]],
            "B": [[1, 0], [0, 1]],
            "C": [0, 0],
            "D": [0.5, 0.5],
        },
    }
    with pytest.raises(ValueError, match="do not match declared"):
        extract_abcd_matrices(spec)


def test_direct_extractor_rejects_conflicting_dimension_declarations() -> None:
    with pytest.raises(ValueError, match="Conflicting declarations"):
        extract_abcd_matrices(
            {"stateSpace": {"size": 3}, "model_parameters": {"num_hidden_states": 2}}
        )


def test_strict_extraction_never_allocates_declared_demo_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("strict extraction allocated unused demo defaults")

    monkeypatch.setattr(np, "eye", forbidden)
    monkeypatch.setattr(np, "zeros", forbidden)
    monkeypatch.setattr(np, "ones", forbidden)
    spec = {
        "model_parameters": {"num_hidden_states": 10**12},
        "initialparameterization": {
            "A": [[1, 0], [0, 1]],
            "B": [[1, 0], [0, 1]],
            "C": [0, 0],
            "D": [0.5, 0.5],
        },
    }
    with pytest.raises(ValueError, match="do not match declared"):
        extract_abcd_matrices(spec)
    spec["model_parameters"]["num_hidden_states"] = 2
    _, _, _, d = extract_abcd_matrices(spec)
    np.testing.assert_array_equal(d, [0.5, 0.5])
