"""Public JAX renderers preserve authored parameters in executable artifacts."""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pytest

from gnn.render.jax import (
    render_gnn_to_jax,
    render_gnn_to_jax_combined,
    render_gnn_to_jax_pomdp,
)


def scientific_tables() -> dict[str, np.ndarray]:
    """Unequal observation/state/action counts expose tensor-axis mistakes."""
    return {
        "A": np.array([[0.7, 0.1], [0.2, 0.3], [0.1, 0.6]]),
        "B": np.array(
            [
                [[0.9, 0.4, 0.2, 0.75], [0.2, 0.7, 0.1, 0.55]],
                [[0.1, 0.6, 0.8, 0.25], [0.8, 0.3, 0.9, 0.45]],
            ]
        ),
        "C": np.array([-2.0, 0.75, 1.25]),
        "D": np.array([0.8, 0.2]),
    }


def canonical_spec() -> dict:
    return {
        "model_name": "Asymmetric Scientific Parameters",
        "model_parameters": {
            "num_hidden_states": 2,
            "num_obs": 3,
            "num_actions": 4,
            "b_tensor_order": "next_state_previous_state_action",
        },
        "initialparameterization": {
            name: value.tolist() for name, value in scientific_tables().items()
        },
    }


def declared_variables() -> list[dict[str, str]]:
    return [
        {"id": "A", "dimensions": "3,2,type=float"},
        {"id": "B", "dimensions": "2,2,4,type=float"},
        {"id": "C", "dimensions": "3,type=float"},
        {"id": "D", "dimensions": "2,type=float"},
    ]


def as_tuple(value):
    return tuple(as_tuple(item) for item in value) if isinstance(value, list) else value


def braced_literal(value):
    authored_tuple = repr(as_tuple(value.tolist()))
    return "{" + authored_tuple[1:-1] + "}"


def generated_parameters(tmp_path: Path, spec: dict, renderer: str) -> dict:
    path = tmp_path / "nested" / f"{renderer}.py"
    before = pickle.dumps(spec, protocol=5)
    render = render_gnn_to_jax if renderer == "general" else render_gnn_to_jax_pomdp
    success, message, artifacts = render(spec, path)
    assert success, message
    assert artifacts == [str(path)]
    assert pickle.dumps(spec, protocol=5) == before
    module_spec = importlib.util.spec_from_file_location(f"consumer_{renderer}", path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if renderer == "general":
        params = module.create_params()
        return {
            name: np.asarray(params[f"{name}_{'matrix' if name in 'AB' else 'vector'}"])
            for name in "ABCD"
        }
    solver = module.create_pomdp_solver()
    assert (solver.num_observations, solver.num_states, solver.num_actions) == (3, 2, 4)
    # Independent hand calculation for action 0 and prior [0.3, 0.7]:
    # next-state mass [0.41, 0.59], then each authored likelihood row.
    np.testing.assert_allclose(
        np.asarray(solver.compute_observation_probability(np.array([0.3, 0.7]), 0)),
        [0.346, 0.259, 0.395],
        rtol=0,
        atol=2e-7,
    )
    return {name: np.asarray(getattr(solver.models, name)) for name in "ABCD"}


@pytest.mark.parametrize("renderer", ["general", "pomdp"])
def test_canonical_parameters_reach_generated_native_consumers(tmp_path, renderer):
    actual = generated_parameters(tmp_path, canonical_spec(), renderer)
    for name, expected in scientific_tables().items():
        np.testing.assert_allclose(actual[name], expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("renderer", ["general", "pomdp"])
@pytest.mark.parametrize("representation", ["tuple", "ndarray"])
def test_canonical_numeric_representations_keep_authored_values(
    tmp_path, renderer, representation
):
    spec = canonical_spec()
    if representation == "ndarray":
        spec["initialparameterization"] = scientific_tables()
    else:
        spec["initialparameterization"] = {
            name: as_tuple(value.tolist())
            for name, value in scientific_tables().items()
        }
    actual = generated_parameters(tmp_path, spec, renderer)
    for name, expected in scientific_tables().items():
        np.testing.assert_allclose(actual[name], expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("renderer", ["general", "pomdp"])
@pytest.mark.parametrize(
    "order", ["action_next_state_previous_state", "action_previous_state_next_state"]
)
def test_declared_transition_axes_reach_native_parameter_consumers(
    tmp_path, renderer, order
):
    spec = canonical_spec()
    tensor = scientific_tables()["B"]
    permutation = (
        (2, 0, 1) if order == "action_next_state_previous_state" else (2, 1, 0)
    )
    spec["initialparameterization"]["B"] = tensor.transpose(permutation).tolist()
    spec["model_parameters"]["b_tensor_order"] = order
    actual = generated_parameters(tmp_path, spec, renderer)
    for name, expected in scientific_tables().items():
        np.testing.assert_allclose(actual[name], expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("renderer", ["general", "pomdp"])
@pytest.mark.parametrize("representation", ["list", "tuple", "ndarray", "text"])
def test_legacy_typed_parameters_reach_generated_consumers(
    tmp_path, renderer, representation
):
    values = scientific_tables()
    if representation == "text":
        values = {name: braced_literal(value) for name, value in values.items()}
    elif representation != "ndarray":
        values = {name: value.tolist() for name, value in values.items()}
        if representation == "tuple":
            values = {name: as_tuple(value) for name, value in values.items()}
    spec = {
        "ModelName": "Legacy Scientific Parameters",
        "variables": declared_variables(),
        "parameters": [
            {"name": name, "value": value} for name, value in values.items()
        ],
    }
    actual = generated_parameters(tmp_path, spec, renderer)
    for name, expected in scientific_tables().items():
        np.testing.assert_allclose(actual[name], expected, rtol=0, atol=1e-7)


def assert_refused_without_artifact(tmp_path: Path, spec: dict, renderer: str):
    render = render_gnn_to_jax if renderer == "general" else render_gnn_to_jax_pomdp
    path = tmp_path / "invalid.py"
    # Refusal must preserve an existing artifact as well as avoid new artifacts.
    original = b"previous independently valid artifact\n"
    path.write_bytes(original)
    before = pickle.dumps(spec, protocol=5)
    success, message, artifacts = render(spec, path)
    assert not success, message
    assert artifacts == []
    assert "failed" in message.lower()
    assert path.read_bytes() == original
    assert pickle.dumps(spec, protocol=5) == before
    absent = tmp_path / "not_created" / "invalid.py"
    success, message, artifacts = render(spec, absent)
    assert not success, message
    assert artifacts == []
    assert not absent.parent.exists()


@pytest.mark.parametrize("renderer", ["general", "pomdp"])
def test_negative_likelihood_is_refused_before_artifact_write(tmp_path, renderer):
    spec = canonical_spec()
    spec["initialparameterization"]["A"][0][0] = -0.7
    assert_refused_without_artifact(tmp_path, spec, renderer)


def text_parameters() -> str:
    return """
    # Values written in declared [next_state, previous_state, action] order.
    A = {(0.7, 0.1), (0.2, 0.3), (0.1, 0.6)}
    B = {((0.9,0.4,0.2,0.75),(0.2,0.7,0.1,0.55)),
         ((0.1,0.6,0.8,0.25),(0.8,0.3,0.9,0.45))}
    C = {-2.0, 0.75, 1.25}
    D = {0.8, 0.2}
    """


@pytest.mark.parametrize("renderer", ["general", "pomdp"])
@pytest.mark.parametrize("representation", ["json_sections", "older_sections"])
def test_documented_text_sections_preserve_tensor_values(
    tmp_path, renderer, representation
):
    spec = {"ModelName": "Text Scientific Parameters"}
    if representation == "json_sections":
        spec.update(
            statespaceblock=declared_variables(),
            raw_sections={"InitialParameterization": text_parameters()},
        )
    else:
        spec.update(
            variables=declared_variables(), InitialParameterization=text_parameters()
        )
    actual = generated_parameters(tmp_path, spec, renderer)
    for name, expected in scientific_tables().items():
        np.testing.assert_allclose(actual[name], expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("renderer", ["general", "pomdp"])
@pytest.mark.parametrize(
    "invalid",
    [
        "negative_B",
        "negative_D",
        "zero_A",
        "zero_B",
        "zero_D",
        "nonunit_A",
        "nonunit_B",
        "nonunit_D",
        "rank_A",
        "empty_A",
        "state_axes_B",
        "empty_actions_B",
        "length_C",
        "length_D",
        "nonfinite_A",
        "nonfinite_B",
        "nonfinite_C",
        "nonfinite_D",
        "unknown_B_order",
        "declared_states",
        "declared_observations",
        "declared_actions",
    ],
)
def test_invalid_scientific_tables_are_refused_without_replacement(
    tmp_path, renderer, invalid
):
    spec = canonical_spec()
    values = scientific_tables()
    if invalid.startswith("negative_"):
        values[invalid[-1]].flat[0] = -0.1
    elif invalid == "zero_A":
        values["A"][:, 0] = 0
    elif invalid == "zero_B":
        values["B"][:, 0, 0] = 0
    elif invalid == "zero_D":
        values["D"][:] = 0
    elif invalid == "nonunit_A":
        values["A"][:, 0] *= 2
    elif invalid == "nonunit_B":
        values["B"][:, 0, 0] *= 2
    elif invalid == "nonunit_D":
        values["D"] *= 2
    elif invalid == "rank_A":
        values["A"] = values["A"].reshape(-1)
    elif invalid == "empty_A":
        values["A"] = np.empty((0, 2))
    elif invalid == "state_axes_B":
        values["B"] = np.ones((3, 2, 4)) / 3
    elif invalid == "empty_actions_B":
        values["B"] = np.empty((2, 2, 0))
    elif invalid == "length_C":
        values["C"] = np.array([-2.0, 0.75])
    elif invalid == "length_D":
        values["D"] = np.array([0.8, 0.1, 0.1])
    elif invalid.startswith("nonfinite_"):
        values[invalid[-1]].flat[0] = np.nan if invalid[-1] in "AD" else np.inf
    elif invalid == "declared_states":
        spec["model_parameters"]["num_hidden_states"] = 3
    elif invalid == "declared_observations":
        spec["model_parameters"]["num_obs"] = 4
    elif invalid == "declared_actions":
        spec["model_parameters"]["num_actions"] = 3
    else:
        spec["model_parameters"]["b_tensor_order"] = "unknown_axis_order"
    spec["initialparameterization"] = {
        name: value.tolist() for name, value in values.items()
    }
    assert_refused_without_artifact(tmp_path, spec, renderer)


@pytest.mark.parametrize("composition", ["continuous", "hierarchical continuous"])
def test_combined_flax_standin_refuses_gaussian_model_and_composition(
    tmp_path, composition
):
    spec = {
        "ModelName": "Gaussian Parameters Require Gaussian Semantics",
        "gnn_section": composition,
        "initialparameterization": {
            "F": [[0.7]],
            "H": [[1.3]],
            "Q": [[0.2]],
            "R": [[0.4]],
            "prior_mean": [0.6],
            "prior_cov": [[0.5]],
        },
    }
    before = pickle.dumps(spec, protocol=5)
    path = tmp_path / "uncreated" / "combined.py"
    success, message, artifacts = render_gnn_to_jax_combined(spec, path)
    assert not success, message
    assert "continuous" in message.lower()
    assert artifacts == []
    assert not path.parent.exists()
    assert pickle.dumps(spec, protocol=5) == before


@pytest.mark.parametrize(
    "representation", ["json_sections", "older_sections", "typed_text"]
)
@pytest.mark.parametrize(
    "invalid", ["bad_number", "ragged_rows", "ragged_tensor", "unclosed_prior"]
)
def test_malformed_authored_text_never_becomes_recovery_parameters(
    tmp_path, representation, invalid
):
    literals = {
        name: braced_literal(value) for name, value in scientific_tables().items()
    }
    if invalid == "bad_number":
        literals["A"] = "{(0.7, invalid), (0.2, 0.3), (0.1, 0.6)}"
    elif invalid == "ragged_rows":
        literals["A"] = "{(0.7, 0.1), (0.2), (0.1, 0.6)}"
    elif invalid == "ragged_tensor":
        literals["B"] = (
            "{((0.9, 0.4), (0.2, 0.7, 0.1, 0.55)), ((0.1, 0.6, 0.8, 0.25), (0.8, 0.3, 0.9, 0.45))}"
        )
    else:
        literals["D"] = "{0.8, 0.2"
    text = "\n".join(f"{name} = {value}" for name, value in literals.items())
    spec = {"ModelName": "Malformed Authored Parameters"}
    if representation == "json_sections":
        spec.update(
            statespaceblock=declared_variables(),
            raw_sections={"InitialParameterization": text},
        )
    elif representation == "older_sections":
        spec.update(variables=declared_variables(), InitialParameterization=text)
    else:
        spec.update(
            variables=declared_variables(),
            parameters=[
                {"name": name, "value": value} for name, value in literals.items()
            ],
        )
    assert_refused_without_artifact(tmp_path, spec, "general")


@pytest.mark.parametrize("renderer", ["general", "pomdp"])
def test_optional_action_prior_array_preserves_dense_parameter_contract(
    tmp_path, renderer
):
    spec = canonical_spec()
    spec["initialparameterization"]["E"] = np.array([0.1, 0.2, 0.3, 0.4])
    actual = generated_parameters(tmp_path, spec, renderer)
    for name, expected in scientific_tables().items():
        np.testing.assert_allclose(actual[name], expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("renderer", ["general", "pomdp"])
@pytest.mark.parametrize(
    "prior", [[0.1, 0.2], [-0.1, 0.2, 0.3, 0.6], [0, 0, 0, 0], [np.inf, 0.2, 0.3, 0.4]]
)
def test_invalid_optional_action_prior_does_not_authorize_flat_render(
    tmp_path, renderer, prior
):
    spec = canonical_spec()
    spec["initialparameterization"]["E"] = prior
    assert_refused_without_artifact(tmp_path, spec, renderer)


@pytest.mark.parametrize(
    "invalid",
    [
        "negative_likelihood",
        "zero_transition_mass",
        "nonunit_prior",
        "nonfinite_payoff",
        "declared_transition_axes",
    ],
)
def test_legacy_supplied_science_and_declarations_are_authoritative(tmp_path, invalid):
    values = scientific_tables()
    declarations = declared_variables()
    if invalid == "negative_likelihood":
        values["A"][0, 0] = -0.7
    elif invalid == "zero_transition_mass":
        values["B"][:, 0, 0] = 0
    elif invalid == "nonunit_prior":
        values["D"] *= 2
    elif invalid == "nonfinite_payoff":
        values["C"][0] = np.nan
    else:
        declarations[1]["dimensions"] = "4,2,2,type=float"
    spec = {
        "ModelName": "Legacy Authored Science",
        "variables": declarations,
        "parameters": [
            {"name": name, "value": value} for name, value in values.items()
        ],
    }
    assert_refused_without_artifact(tmp_path, spec, "general")


def test_complex_legacy_payoffs_cannot_be_coerced_to_different_real_values(tmp_path):
    values = scientific_tables()
    values["C"] = np.array([-2.0 + 4.0j, 0.75, 1.25])
    spec = {
        "ModelName": "Complex Payoffs Are Not Dense Real POMDP Payoffs",
        "variables": declared_variables(),
        "parameters": [
            {"name": name, "value": value} for name, value in values.items()
        ],
    }
    assert_refused_without_artifact(tmp_path, spec, "general")
