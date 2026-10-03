"""Dependency-light scientific/admission contracts for experimental THRML."""

from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from gnn.render.thrml import (
    UnsupportedTHRMLModel,
    build_thrml_payload,
    render_gnn_to_thrml,
)
from gnn.render.thrml.adapter import contract_digest
from gnn.render.thrml.runtime import _validate_payload
from gnn.render.thrml.thrml_renderer import resolve_render_options


def model() -> dict:
    return {
        "model_name": "asymmetric three-state/two-action witness",
        "initialparameterization": {
            "A": [[0.75, 0.15, 0.35], [0.25, 0.85, 0.65]],
            "B": [
                [[0.7, 0.2], [0.1, 0.6], [0.3, 0.1]],
                [[0.2, 0.3], [0.6, 0.1], [0.2, 0.2]],
                [[0.1, 0.5], [0.3, 0.3], [0.5, 0.7]],
            ],
            "C": [0.7, -0.2],
            "D": [0.2, 0.3, 0.5],
            "E": [0.3, 0.7],
        },
        "model_parameters": {
            "num_states": 3,
            "num_hidden_states": 3,
            "num_obs": 2,
            "num_actions": 2,
            "num_timesteps": 3,
            "b_tensor_order": "next_state_previous_state_action",
        },
    }


def options() -> dict:
    return {
        "observations": [0, 1, 0],
        "transition_actions": [1, 0],
        "num_samples": 4096,
        "burn_in": 128,
        "thin": 2,
        "seed": 42,
    }


def factored_model() -> dict:
    base = model()
    a0 = [[[0.8, 0.6, 0.4], [0.3, 0.5, 0.7]], [[0.2, 0.4, 0.6], [0.7, 0.5, 0.3]]]
    matrices = {
        "A_m0": a0,
        "A_m1": [[0.7, 0.2, 0.1], [0.2, 0.6, 0.3], [0.1, 0.2, 0.6]],
        "B_f0": [[0.8, 0.3], [0.2, 0.7]],
        "B_f1": base["initialparameterization"]["B"],
        "C_m0": [0.2, -0.1],
        "C_m1": [0.1, 0.4, -0.2],
        "D_f0": [0.6, 0.4],
        "D_f1": [0.2, 0.3, 0.5],
        "E": [0.3, 0.7],
    }
    return {
        "model_name": "dependent factored witness",
        "model_parameters": {
            "num_actions": 2,
            "num_timesteps": 3,
            "num_factors": 2,
            "b_tensor_order": "next_state_previous_state_action",
        },
        "initialparameterization": matrices,
        "structured_pomdp": {
            "matrices": deepcopy(matrices),
            "state_factors": [{"name": "s_f0", "size": 2}, {"name": "s_f1", "size": 3}],
            "observation_modalities": [
                {"name": "o_m0", "size": 2},
                {"name": "o_m1", "size": 3},
            ],
            "control_factors": [],
        },
    }


def independent_model() -> dict:
    base = model()
    matrices = {
        f"{key}_agent{i}": deepcopy(value)
        for i in (1, 2)
        for key, value in base["initialparameterization"].items()
    }

    return {
        "model_name": "independent agents",
        "model_parameters": {
            "nr_agents": 2,
            "num_timesteps": 3,
            "b_tensor_order": "next_state_previous_state_action",
        },
        "initialparameterization": matrices,
        "structured_pomdp": {"matrices": deepcopy(matrices)},
        "connections": [],
    }


def modality_model() -> dict:
    source = model()
    raw = source["initialparameterization"]
    raw["A_m0"] = raw.pop("A")
    raw["C_m0"] = raw.pop("C")
    raw["A_m1"] = [[0.6, 0.1, 0.3], [0.3, 0.7, 0.4], [0.1, 0.2, 0.3]]
    raw["C_m1"] = [0.1, 0.4, -0.2]
    return source


def test_roundtrip_preserves_values_all_axes_preferences_and_habits(
    tmp_path: Path,
) -> None:
    source = model()
    source["initialparameterization"]["A"][0][0] = 1e-250
    source["initialparameterization"]["A"][1][0] = 1.0
    original = deepcopy(source)
    output = tmp_path / "tiny_thrml.py"
    assert render_gnn_to_thrml(source, output, options())[0]
    module = ast.parse(output.read_text())
    assignment = next(node for node in module.body if isinstance(node, ast.Assign))
    serialized = ast.literal_eval(assignment.value.args[0])
    payload = json.loads(serialized)
    assert (
        payload["components"][0]["source_parameters"]
        == source["initialparameterization"]
    )
    for key in ("A", "B", "C", "D", "E"):
        np.testing.assert_allclose(
            payload["components"][0]["parameters"][key],
            source["initialparameterization"][key],
            rtol=2e-15,
            atol=0,
        )
    assert payload["components"][0]["parameters"]["A"][0][0] == 1e-250
    assert source == original
    assert payload["sampling_dtype"] == "float64"
    _validate_payload(payload)


def test_same_shape_corruption_changes_digest_and_serialized_tampering_fails() -> None:
    first = build_thrml_payload(model(), options())
    source = model()
    source["initialparameterization"]["D"] = [0.3, 0.2, 0.5]
    second = build_thrml_payload(source, options())
    assert (
        first["source_identity"]["semantic_sha256"]
        != second["source_identity"]["semantic_sha256"]
    )
    first["components"][0]["parameters"]["D"] = [0.3, 0.2, 0.5]
    with pytest.raises(ValueError, match="identity mismatch"):
        _validate_payload(first)


def test_asymmetric_action_first_is_bound_to_canonical_axes() -> None:
    expected = build_thrml_payload(model(), options())
    source = model()
    source["initialparameterization"]["B"] = (
        np.asarray(source["initialparameterization"]["B"]).transpose(2, 0, 1).tolist()
    )
    source["model_parameters"]["b_tensor_order"] = "action_next_state_previous_state"
    actual = build_thrml_payload(source, options())
    assert (
        actual["components"][0]["parameters"]["B"]
        == expected["components"][0]["parameters"]["B"]
    )
    assert actual["components"][0]["matrix_provenance"]["B"]["orientation_transformed"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_samples", True),
        ("thin", 1.5),
        ("burn_in", -1),
        ("seed", 2**32),
        ("num_timesteps", 0),
    ],
)
def test_schedule_rejects_invalid_values(key: str, value: object) -> None:
    with pytest.raises(ValueError):
        build_thrml_payload(model(), {**options(), key: value})


@pytest.mark.parametrize("key", ["A", "B", "D"])
def test_structural_zero_support_is_explicit_unsupported_without_artifact(
    key: str, tmp_path: Path
) -> None:
    source = model()
    values = np.asarray(source["initialparameterization"][key])
    if key == "A":
        values[:, 0] = [0, 1]
    elif key == "B":
        values[:, 0, 0] = [0, 0.4, 0.6]
    else:
        values[:] = [0, 0.4, 0.6]
    source["initialparameterization"][key] = values.tolist()
    output = tmp_path / "zero_thrml.py"
    success, message, paths = render_gnn_to_thrml(source, output, options())
    assert (
        not success
        and message.startswith("unsupported-thrml:")
        and not paths
        and not output.exists()
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 0.2])
def test_malformed_probability_not_repaired(value: float) -> None:
    source = model()
    source["initialparameterization"]["A"][0][0] = value
    with pytest.raises(ValueError):
        build_thrml_payload(source, options())


def test_resource_admission_precedes_canonical_and_native_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("canonical materialization must not happen after failed admission")

    monkeypatch.setattr(
        "gnn.render.thrml.adapter.build_canonical_pomdp_spec", forbidden
    )
    with pytest.raises(ValueError, match="resource admission"):
        build_thrml_payload(model(), {"num_timesteps": 100_000})


def test_joint_factor_admission_precedes_common_cartesian_composer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = factored_model()
    source["structured_pomdp"]["state_factors"][0]["size"] = 4096

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail(
            "common Cartesian composition must not happen after failed admission"
        )

    monkeypatch.setattr("gnn.render.pomdp_processor.pomdp_to_gnn_spec", forbidden)
    with pytest.raises(ValueError, match="admission"):
        build_thrml_payload(source, options())


def test_dependent_factored_composition_preserves_every_component_and_axes() -> None:
    source = factored_model()
    payload = build_thrml_payload(source, {**options(), "observations": [0, 4, 1]})
    component = payload["components"][0]
    assert payload["composition"] == "canonical_joint"
    assert component["num_states"] == 6 and component["num_observations"] == 6
    parameters = component["parameters"]
    raw = source["initialparameterization"]
    expected_b = np.stack(
        [
            np.kron(raw["B_f0"], np.asarray(raw["B_f1"])[:, :, action])
            for action in range(2)
        ],
        axis=2,
    )
    assert np.allclose(parameters["B"], expected_b)
    assert np.allclose(parameters["D"], np.kron(raw["D_f0"], raw["D_f1"]))
    for obs0 in range(2):
        for obs1 in range(3):
            for state0 in range(2):
                for state1 in range(3):
                    assert parameters["A"][obs0 * 3 + obs1][
                        state0 * 3 + state1
                    ] == pytest.approx(
                        raw["A_m0"][obs0][state0][state1] * raw["A_m1"][obs1][state1]
                    )
    assert component["composition_metadata"]["source_matrices"] == raw
    assert component["source_model_parameters"] == source["model_parameters"]
    assert parameters["E"] == raw["E"]
    _validate_payload(payload)


def test_verified_composed_entry_does_not_recompose_structured_factor_sources() -> None:
    first = build_thrml_payload(
        factored_model(), {**options(), "observations": [0, 4, 1]}
    )
    component = first["components"][0]
    spec = {
        "model_name": "canonical entry",
        "model_parameters": {
            **component["source_model_parameters"],
            "num_hidden_states": 6,
            "num_obs": 6,
        },
        "initialparameterization": component["parameters"],
        "matrix_provenance": component["matrix_provenance"],
        "structured_pomdp": component["source_structure"],
    }
    second = build_thrml_payload(spec, {**options(), "observations": [0, 4, 1]})
    for key in ("A", "B", "C", "D", "E"):
        np.testing.assert_allclose(
            second["components"][0]["parameters"][key],
            component["parameters"][key],
            rtol=2e-15,
            atol=0,
        )
    assert (
        second["components"][0]["composition_metadata"]["source_matrices"]
        == factored_model()["initialparameterization"]
    )


def test_independent_agents_keep_distinct_parameters_and_declared_habits() -> None:
    source = independent_model()
    source["initialparameterization"]["D_agent2"] = [0.3, 0.5, 0.2]
    payload = build_thrml_payload(
        source,
        {
            **options(),
            "observations": {"agent1": [0, 1, 0], "agent2": [1, 0, 1]},
            "transition_actions": {"agent1": [1, 0], "agent2": [0, 1]},
        },
    )
    assert payload["composition"] == "independent_components"
    assert [item["component_id"] for item in payload["components"]] == [
        "agent1",
        "agent2",
    ]
    assert payload["components"][1]["parameters"]["D"] == [0.3, 0.5, 0.2]
    assert all(item["parameters"]["E"] == [0.3, 0.7] for item in payload["components"])
    _validate_payload(payload)


def test_coupled_agents_are_unsupported() -> None:
    source = independent_model()
    source["connections"] = [{"source": "s_agent1", "target": "o_agent2"}]
    with pytest.raises(UnsupportedTHRMLModel, match="coupling"):
        build_thrml_payload(source)


def test_missing_E_is_not_synthesized() -> None:
    source = model()
    del source["initialparameterization"]["E"]
    payload = build_thrml_payload(source, options())
    assert "E" not in payload["components"][0]["parameters"]


def test_config_precedence_and_unknown_options() -> None:
    actual = resolve_render_options(
        {
            "backend_options": {"thrml": {"num_samples": 128, "seed": 1}},
            "simulation_params": '{"thrml":{"num_samples":256}}',
            "num_samples": 512,
            "recursive": True,
            "strict_validation": True,
        }
    )
    assert actual == {"num_samples": 512, "seed": 1}
    for opts in (
        {"backend_options": {"thrml": {"sampls": 128}}},
        {"simulation_params": '{"thrml":{"sampls":128}}'},
        {"sampls": 128},
    ):
        with pytest.raises(ValueError, match="unknown"):
            resolve_render_options(opts)


def test_continuous_is_explicit_unsupported() -> None:
    source = {
        "model_kind": "continuous",
        "initialparameterization": {
            "F": [[1.0]],
            "H": [[1.0]],
            "Q": [[0.1]],
            "R": [[0.1]],
            "prior_mean": [0.0],
            "prior_cov": [[1.0]],
        },
    }
    with pytest.raises(UnsupportedTHRMLModel, match="continuous"):
        build_thrml_payload(source)


@pytest.mark.parametrize(
    "key", ["num_states", "num_hidden_states", "num_obs", "num_actions"]
)
@pytest.mark.parametrize("value", [True, 2.5, "3", -1])
def test_dimensions_are_not_lossily_coerced(key: str, value: object) -> None:
    source = model()
    source["model_parameters"][key] = value
    with pytest.raises(ValueError, match=key):
        build_thrml_payload(source, options())


def test_complete_independent_factor_groups_refuse_declared_coupling() -> None:
    source = independent_model()
    source["initialparameterization"] = {
        key.replace("agent1", "f0").replace("agent2", "f1"): value
        for key, value in source["initialparameterization"].items()
    }
    source["structured_pomdp"]["matrices"] = deepcopy(source["initialparameterization"])
    source["model_parameters"].pop("nr_agents")
    source["model_parameters"]["num_factors"] = 2
    source["connections"] = [{"source": "s_f0", "target": "s_f1"}]
    with pytest.raises(UnsupportedTHRMLModel, match="coupling"):
        build_thrml_payload(source)


@pytest.mark.parametrize(
    "alias", ["b_tensor_order", "B_tensor_order", "transition_tensor_order"]
)
def test_native_component_orientation_aliases_are_canonical(alias: str) -> None:
    source = independent_model()
    source["model_parameters"].pop("b_tensor_order")
    source["model_parameters"][alias] = "ACTIONS-NEXT-PREVIOUS"
    for name in ("agent1", "agent2"):
        source["initialparameterization"][f"B_{name}"] = (
            np.asarray(source["initialparameterization"][f"B_{name}"])
            .transpose(2, 0, 1)
            .tolist()
        )
    source["structured_pomdp"]["matrices"] = deepcopy(source["initialparameterization"])
    payload = build_thrml_payload(source)
    for component in payload["components"]:
        np.testing.assert_allclose(
            component["parameters"]["B"], model()["initialparameterization"]["B"]
        )
        assert len(component["parameters"]["E"]) == 2


def test_aggregate_admission_happens_before_any_component_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("no component arrays may allocate after failed aggregate admission")

    monkeypatch.setattr(
        "gnn.render.thrml.adapter.build_canonical_pomdp_spec", forbidden
    )
    with pytest.raises(ValueError, match="aggregate"):
        build_thrml_payload(independent_model(), {"num_samples": 400_000})


def test_one_step_cannot_hide_retained_all_action_matrices_from_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("retained source storage must be admitted before array copies")

    monkeypatch.setattr("gnn.render.thrml.adapter.MAX_FACTOR_ENTRIES", 20)
    monkeypatch.setattr(
        "gnn.render.thrml.adapter.build_canonical_pomdp_spec", forbidden
    )
    with pytest.raises(ValueError, match="retained source matrix admission"):
        build_thrml_payload(model(), {"num_timesteps": 1, "num_samples": 1})


def test_serialized_aggregate_admission_precedes_canonical_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = build_thrml_payload(independent_model(), {"num_samples": 8})

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("runtime must admit all serialized components before allocating")

    monkeypatch.setattr(
        "gnn.render.pomdp_contract.build_canonical_pomdp_spec", forbidden
    )
    monkeypatch.setattr(
        "gnn.render.thrml.runtime.MAX_SITE_UPDATES",
        payload["resource_admission"]["site_updates"] - 1,
    )
    with pytest.raises(ValueError, match="aggregate"):
        _validate_payload(payload)


def test_flat_contract_cannot_silently_omit_additional_declared_matrices() -> None:
    source = model()
    source["initialparameterization"]["A_m0"] = source["initialparameterization"]["A"]
    with pytest.raises(UnsupportedTHRMLModel, match="additional declared"):
        build_thrml_payload(source)


def test_multi_modal_product_keeps_source_values_preferences_and_order() -> None:
    source = modality_model()
    opts = {**options(), "observations": [0, 4, 1]}
    payload = build_thrml_payload(source, opts)
    component = payload["components"][0]
    raw = source["initialparameterization"]
    for obs0 in range(2):
        for obs1 in range(3):
            row = obs0 * 3 + obs1
            np.testing.assert_allclose(
                component["parameters"]["A"][row],
                np.asarray(raw["A_m0"][obs0]) * raw["A_m1"][obs1],
            )
            assert component["parameters"]["C"][row] == pytest.approx(
                raw["C_m0"][obs0] + raw["C_m1"][obs1]
            )
    assert component["composition_metadata"]["source_matrices"] == raw
    assert component["composition_metadata"]["joint_observation_order"] == [
        (i, j) for i in range(2) for j in range(3)
    ]
    _validate_payload(payload)


def test_optional_native_packages_are_cold_during_rendering(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    code = """
import importlib.abc, json, sys
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'thrml', 'jax', 'jaxlib', 'equinox'}:
            raise AssertionError('native import during renderer operation: ' + fullname)
sys.meta_path.insert(0, BlockNative())
from pathlib import Path
from gnn.render.thrml import render_gnn_to_thrml
source = json.loads(sys.argv[1])
success, message, paths = render_gnn_to_thrml(source, Path(sys.argv[2]))
assert success, message
assert paths
"""
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            json.dumps(model()),
            str(tmp_path / "cold_thrml.py"),
        ],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert child.returncode == 0, child.stderr


def test_runtime_does_not_repair_uncanonical_serialized_values() -> None:
    payload = build_thrml_payload(model(), options())
    payload["components"][0]["parameters"]["A"][0][0] *= 0.99999
    payload["source_identity"]["semantic_sha256"] = contract_digest(
        {
            "components": payload["components"],
            "sampling": payload["sampling"],
            "posterior_semantics": "fixed_actions_full_sequence_smoothing",
        }
    )
    with pytest.raises(ValueError, match="already be canonical"):
        _validate_payload(payload)


@pytest.mark.parametrize("value", [[], False, 0, "invalid"])
def test_malformed_empty_configuration_is_not_treated_as_absent(value: object) -> None:
    for incoming in (
        {"backend_options": value},
        {"backend_options": {"thrml": value}},
        {"simulation_params": {"thrml": value}},
    ):
        with pytest.raises((ValueError, TypeError)):
            resolve_render_options(incoming)
    source = model()
    source["model_parameters"] = value
    with pytest.raises(ValueError, match="mapping"):
        build_thrml_payload(source)


def test_direct_payload_helper_rejects_unknown_configuration() -> None:
    with pytest.raises(ValueError, match="unknown THRML options"):
        build_thrml_payload(model(), {"sampls": 1024})


def test_original_dimensions_are_retained_separately_from_composed_dimensions() -> None:
    source = modality_model()
    payload = build_thrml_payload(source, {**options(), "observations": [0, 4, 1]})
    component = payload["components"][0]
    assert component["source_model_parameters"] == source["model_parameters"]
    assert component["source_model_parameters"]["num_obs"] == 2
    assert component["composed_model_parameters"]["num_obs"] == 6
    _validate_payload(payload)


@pytest.mark.parametrize("factory", [modality_model, factored_model])
def test_retained_source_is_admitted_before_joint_materialization(
    monkeypatch: pytest.MonkeyPatch,
    factory: object,
) -> None:
    source = factory()
    source["initialparameterization"]["E"] = [1 / 100] * 100
    if source.get("structured_pomdp"):
        source["structured_pomdp"]["matrices"] = deepcopy(
            source["initialparameterization"]
        )
    monkeypatch.setattr("gnn.render.thrml.adapter.MAX_FACTOR_ENTRIES", 100)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail(
            "oversized retained source must fail before Cartesian/native copies"
        )

    monkeypatch.setattr("gnn.render.thrml.composition.itertools.product", forbidden)
    monkeypatch.setattr("gnn.render.thrml.composition.deepcopy", forbidden)
    monkeypatch.setattr("gnn.render.thrml.composition.np.asarray", forbidden)
    with pytest.raises(ValueError, match="retained source matrix admission"):
        build_thrml_payload(source, {"num_timesteps": 1, "num_samples": 1})


@pytest.mark.parametrize(
    "factory", [model, modality_model, factored_model, independent_model]
)
@pytest.mark.parametrize(
    "location",
    ["initialparameterization", "model_parameters", "top_level", "structured_pomdp"],
)
@pytest.mark.parametrize("key", ["coupling_gain", "env_feedback", "signal_precision"])
def test_no_model_family_silently_omits_declared_coupling_semantics(
    factory: object,
    location: str,
    key: str,
    tmp_path: Path,
) -> None:
    source = factory()
    if location == "top_level":
        source[key] = 0.7
    else:
        source.setdefault(location, {})[key] = 0.7
    output = tmp_path / "unsupported_thrml.py"
    success, reason, paths = render_gnn_to_thrml(source, output)
    assert not success and reason.startswith("unsupported-thrml:")
    assert "coupling/environment/signal" in reason
    assert not paths and not output.exists()


def test_current_run_payload_keeps_artifact_stem_and_stable_model_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib
    import runpy

    from gnn.pipeline.run_context import build_run_context

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source_path = input_dir / "unique_model.md"
    exemplar = (
        Path(__file__).parents[2] / "input/gnn_files/thrml/categorical_smoothing.md"
    )
    source_path.write_bytes(exemplar.read_bytes())
    output_dir = tmp_path / "output"
    context = build_run_context(
        input_dir,
        output_dir,
        "thrml-context-witness",
        (11,),
        {"render": {}},
        frameworks=("thrml",),
    )
    monkeypatch.setattr("gnn.pipeline.run_context.current_run_context", lambda: context)
    selected = context.selected_models(11)[0]
    output = output_dir / selected.artifact_stem / "thrml" / "unique_thrml.py"
    success, reason, _ = render_gnn_to_thrml(model(), output, options())
    assert success, reason
    identity = runpy.run_path(str(output))["PAYLOAD"]["source_identity"]
    assert identity["model_id"] == selected.model_id
    assert identity["artifact_stem"] == selected.artifact_stem == "unique_model"
    assert identity["source_relative_path"] == "unique_model.md"
    assert (
        identity["source_sha256"]
        == hashlib.sha256(source_path.read_bytes()).hexdigest()
    )
