"""Public joint conversion must not relabel retained per-agent transition axes."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from gnn.extract.pomdp_extractor import POMDPStateSpace, extract_pomdp_from_file
from gnn.render import process_render
from gnn.render.activeinference_jl import render_gnn_to_activeinference_jl
from gnn.render.multi_agent_common import validate_native_agent_groups
from gnn.render.pomdp_contract import build_canonical_pomdp_spec
from gnn.render.pomdp_processor import POMDPRenderProcessor, pomdp_to_gnn_spec
from gnn.render.rxinfer import render_gnn_to_rxinfer

CANONICAL_ORDER = "next_state_previous_state_action"
SOURCE_ORDERS = (
    CANONICAL_ORDER,
    "action_next_state_previous_state",
    "action_previous_state_next_state",
)
EXEMPLARS = Path(__file__).parents[2] / "input/gnn_files/multiagent"


def _space(order: str, actions: int) -> tuple[POMDPStateSpace, dict[str, np.ndarray]]:
    # Dyadic entries have exact mass. Asymmetry distinguishes next/previous;
    # actions=states=3 also makes every wrong axis permutation shape-compatible.
    canonical = np.stack(
        [
            [[0.5, 0.125, 0.375], [0.375, 0.5, 0.125], [0.125, 0.375, 0.5]],
            [[0.25, 0.625, 0.125], [0.125, 0.25, 0.625], [0.625, 0.125, 0.25]],
            [[0.375, 0.5, 0.125], [0.125, 0.375, 0.5], [0.5, 0.125, 0.375]],
        ][:actions],
        axis=2,
    )
    expected = {"agent1": canonical, "agent2": np.roll(canonical, 1, axis=1)}
    matrices = {}
    for agent, tensor in expected.items():
        raw = (
            tensor
            if order == CANONICAL_ORDER
            else tensor.transpose(2, 0, 1)
            if order == "action_next_state_previous_state"
            else tensor.transpose(2, 1, 0)
        )
        matrices.update(
            {
                f"A_{agent}": [
                    [0.75, 0.125, 0.125],
                    [0.125, 0.75, 0.125],
                    [0.125, 0.125, 0.75],
                ],
                f"B_{agent}": raw.tolist(),
                f"C_{agent}": [0.0, 0.5, 1.0],
                f"D_{agent}": [0.5, 0.375, 0.125],
            }
        )
    return (
        POMDPStateSpace(
            num_states=9,
            num_observations=9,
            num_actions=actions,
            model_name="Asymmetric_Agents",
            gnn_section="ActInfPOMDP",
            model_parameters={
                "num_agents": 2,
                "num_actions": actions,
                "b_tensor_order": order,
            },
            matrices=matrices,
            matrix_provenance={
                key: {
                    "source": "InitialParameterization",
                    "shape": list(np.shape(value)),
                    "derived": False,
                }
                for key, value in matrices.items()
            },
            state_factors=[{"name": f"s_agent{i}", "size": 3} for i in (1, 2)],
            observation_modalities=[{"name": f"o_agent{i}", "size": 3} for i in (1, 2)],
            initial_parameterization=deepcopy(matrices),
        ),
        expected,
    )


@pytest.mark.parametrize("order", SOURCE_ORDERS)
@pytest.mark.parametrize("actions", [2, 3])
def test_raw_joint_and_recanonicalized_public_paths_preserve_component_values(
    order: str, actions: int
) -> None:
    space, expected = _space(order, actions)
    before = deepcopy(space.to_dict())
    raw = pomdp_to_gnn_spec(space, native_agents=True)
    joint = pomdp_to_gnn_spec(space)
    canonical = build_canonical_pomdp_spec(joint)
    assert joint["model_parameters"]["b_tensor_order"] == CANONICAL_ORDER
    for spec in (raw, joint, canonical, build_canonical_pomdp_spec(canonical)):
        native = validate_native_agent_groups(spec)
        for agent, values in expected.items():
            np.testing.assert_array_equal(native[agent]["B"], values)
            assert native[agent]["matrix_provenance"]["B"]["source_order"] == order
            assert (
                native[agent]["matrix_provenance"]["B"]["source_values"]
                == space.matrices[f"B_{agent}"]
            )
        assert spec["structured_pomdp"]["matrices"] == before["matrices"]
    for agent in expected:
        provenance = canonical["matrix_provenance"][f"B_{agent}"]
        assert provenance["source_order"] == order
        assert provenance["source_shape"] == list(
            np.shape(space.matrices[f"B_{agent}"])
        )
        assert provenance["canonicalization"]["canonical_order"] == CANONICAL_ORDER
        assert provenance["canonicalization"]["shape"] == [3, 3, actions]
        assert provenance["canonicalization"]["normalized"] is False
        assert provenance["declared_order_explicit"] is True
        assert provenance["canonicalization"]["declared_order_explicit"] is True
        assert (
            provenance["canonicalization"]["orientation_resolution"]
            == "declared_model_parameter"
        )
    assert space.to_dict() == before


def test_component_specific_order_remains_authoritative_after_joint_conversion() -> (
    None
):
    space, expected = _space("action_next_state_previous_state", 3)
    space.model_parameters["b_tensor_order_agent2"] = "action_previous_state_next_state"
    space.matrices["B_agent2"] = expected["agent2"].transpose(2, 1, 0).tolist()
    spec = build_canonical_pomdp_spec(pomdp_to_gnn_spec(space))
    assert spec["model_parameters"]["b_tensor_order"] == CANONICAL_ORDER
    native = validate_native_agent_groups(spec)
    for agent, values in expected.items():
        np.testing.assert_array_equal(native[agent]["B"], values)
    assert (
        spec["matrix_provenance"]["B_agent1"]["source_order"]
        == "action_next_state_previous_state"
    )
    assert (
        spec["matrix_provenance"]["B_agent2"]["source_order"]
        == "action_previous_state_next_state"
    )


def test_original_agent_default_is_recorded_before_joint_global_order_changes() -> None:
    space, expected = _space("action_next_state_previous_state", 3)
    del space.model_parameters["b_tensor_order"]
    spec = build_canonical_pomdp_spec(pomdp_to_gnn_spec(space))
    native = validate_native_agent_groups(spec)
    for agent, values in expected.items():
        np.testing.assert_array_equal(native[agent]["B"], values)
        assert (
            spec["matrix_provenance"][f"B_{agent}"]["source_order"]
            == "action_next_state_previous_state"
        )
        assert (
            spec["matrix_provenance"][f"B_{agent}"]["canonicalization"][
                "orientation_resolution"
            ]
            == "native_agent_action_first_default"
        )
        assert (
            spec["matrix_provenance"][f"B_{agent}"]["declared_order_explicit"] is False
        )
        assert (
            spec["matrix_provenance"][f"B_{agent}"]["canonicalization"][
                "declared_order_explicit"
            ]
            is False
        )
        assert (
            native[agent]["matrix_provenance"]["B"]["declared_order_explicit"] is False
        )


def test_inferred_component_metadata_does_not_override_explicit_source_axes() -> None:
    space, expected = _space("action_previous_state_next_state", 3)
    for agent in expected:
        space.matrix_provenance[f"B_{agent}"]["source_order"] = "inferred"
    raw = pomdp_to_gnn_spec(space, native_agents=True)
    native = validate_native_agent_groups(raw)
    for agent, values in expected.items():
        np.testing.assert_array_equal(native[agent]["B"], values)


def test_declared_action_count_conflict_is_not_repaired_from_component_shapes() -> None:
    space, _ = _space("action_next_state_previous_state", 3)
    space.num_actions = 2
    space.model_parameters["num_actions"] = 2
    before = deepcopy(space.to_dict())
    for kwargs in ({}, {"native_agents": True}):
        with pytest.raises(ValueError, match="action"):
            pomdp_to_gnn_spec(space, **kwargs)
    assert space.to_dict() == before


def test_heterogeneous_action_spaces_are_refused_instead_of_dropping_source_actions() -> (
    None
):
    space, _ = _space("action_next_state_previous_state", 3)
    space.matrices["B_agent1"] = space.matrices["B_agent1"][:2]
    before = deepcopy(space.to_dict())
    with pytest.raises(ValueError, match="unsupported-agent-action-composition"):
        pomdp_to_gnn_spec(space)
    with pytest.raises(ValueError, match="action dimension"):
        pomdp_to_gnn_spec(space, native_agents=True)
    assert space.to_dict() == before


def test_derived_joint_action_count_cannot_silently_relabel_native_component_sizes() -> (
    None
):
    space, _ = _space("action_next_state_previous_state", 3)
    canonical = build_canonical_pomdp_spec(pomdp_to_gnn_spec(space))
    canonical["model_parameters"]["num_actions"] = 9
    before = deepcopy(canonical)
    with pytest.raises(ValueError, match="action dimension.*does not match 9"):
        validate_native_agent_groups(canonical)
    assert canonical == before


def test_passive_agent_remains_compatible_with_declared_shared_action_space() -> None:
    space, expected = _space("action_next_state_previous_state", 3)
    space.matrices["B_agent1"] = space.matrices["B_agent1"][:1]
    for spec in (
        pomdp_to_gnn_spec(space, native_agents=True),
        build_canonical_pomdp_spec(pomdp_to_gnn_spec(space)),
    ):
        native = validate_native_agent_groups(spec)
        np.testing.assert_array_equal(
            native["agent1"]["B"], expected["agent1"][:, :, :1]
        )
        np.testing.assert_array_equal(native["agent2"]["B"], expected["agent2"])


def test_passive_two_dimensional_source_axes_survive_public_joint_conversion() -> None:
    space, expected = _space("action_next_state_previous_state", 3)
    space.matrices["B_agent1"] = expected["agent1"][:, :, 0].tolist()
    before = deepcopy(space.to_dict())
    raw = pomdp_to_gnn_spec(space, native_agents=True)
    canonical = build_canonical_pomdp_spec(pomdp_to_gnn_spec(space))
    assert (
        canonical["matrix_provenance"]["B_agent1"]["source_order"]
        == "next_state_previous_state"
    )
    for spec in (raw, canonical, build_canonical_pomdp_spec(canonical)):
        native = validate_native_agent_groups(spec)
        np.testing.assert_array_equal(
            native["agent1"]["B"], expected["agent1"][:, :, :1]
        )
        np.testing.assert_array_equal(native["agent2"]["B"], expected["agent2"])
        assert (
            native["agent1"]["matrix_provenance"]["B"]["source_order"]
            == "next_state_previous_state"
        )
    assert space.to_dict() == before


def test_two_dimensional_custody_cannot_relabel_a_three_dimensional_source() -> None:
    space, _ = _space("action_next_state_previous_state", 3)
    raw = pomdp_to_gnn_spec(space, native_agents=True)
    raw["matrix_provenance"]["B_agent1"]["source_order"] = "next_state_previous_state"
    with pytest.raises(ValueError, match="Unsupported declared B tensor order"):
        validate_native_agent_groups(raw)


@pytest.mark.parametrize("framework", ["rxinfer", "activeinference_jl"])
def test_native_prevalidation_uses_strict_component_contract(
    tmp_path: Path, framework: str
) -> None:
    space, _ = _space("action_next_state_previous_state", 3)
    spec = pomdp_to_gnn_spec(space, native_agents=True)
    assert "A" not in spec["initialparameterization"]
    processor = POMDPRenderProcessor(tmp_path)
    assert processor._validate_state_spaces_in_spec(spec, framework)["valid"] is True
    flat = processor._validate_state_spaces_in_spec(spec, "jax")
    assert flat["valid"] is False
    assert "Missing required matrices" in flat["reason"]
    malformed = deepcopy(spec)
    malformed["structured_pomdp"]["matrices"]["A_agent2"][0][0] = 0.5
    failed = processor._validate_state_spaces_in_spec(malformed, framework)
    assert failed["valid"] is False
    assert failed["critical"] is True
    assert "mass" in failed["reason"]


@pytest.mark.parametrize("framework", ["rxinfer", "activeinference_jl"])
@pytest.mark.parametrize(
    "source_name", ["multi_agent_coordination.md", "stigmergic_swarm.md"]
)
def test_reported_exemplars_render_raw_and_canonical_native_values(
    tmp_path: Path, framework: str, source_name: str
) -> None:
    space = extract_pomdp_from_file(EXEMPLARS / source_name, strict_validation=True)
    before = deepcopy(space.to_dict())
    raw = pomdp_to_gnn_spec(space, native_agents=True, timesteps=2)
    canonical = build_canonical_pomdp_spec(pomdp_to_gnn_spec(space, timesteps=2))
    expected = validate_native_agent_groups(raw)
    actual = validate_native_agent_groups(canonical)
    render = (
        render_gnn_to_rxinfer
        if framework == "rxinfer"
        else render_gnn_to_activeinference_jl
    )
    for agent in expected:
        np.testing.assert_array_equal(actual[agent]["B"], expected[agent]["B"])
    for name, spec in (("raw", raw), ("canonical", canonical)):
        script = tmp_path / f"{name}_{framework}.jl"
        ok, message, _ = render(spec, script)
        assert ok, message
        assert script.is_file()
        assert "agent1" in script.read_text()
    assert space.to_dict() == before


@pytest.mark.parametrize("framework", ["rxinfer", "activeinference_jl"])
def test_public_step11_native_routing_avoids_unused_joint_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, framework: str
) -> None:
    import json
    import shutil

    sources = tmp_path / "sources"
    sources.mkdir()
    for filename in ("multi_agent_coordination.md", "stigmergic_swarm.md"):
        shutil.copy2(EXEMPLARS / filename, sources / filename)

    def forbidden(*args, **kwargs):
        raise AssertionError("native Step 11 allocated an unused joint tensor")

    monkeypatch.setattr(
        POMDPRenderProcessor, "_build_canonical_initialparameterization", forbidden
    )
    out = tmp_path / "render"
    assert process_render(sources, out, frameworks=[framework], timesteps=2) is True
    summary = json.loads((out / "render_processing_summary.json").read_text())
    assert not summary["failed_framework_renderings"]
    assert not summary["unsupported_framework_renderings"]
    scripts = list(out.rglob("*.jl"))
    assert len(scripts) == 2
    assert all(script.stat().st_size < 250_000 for script in scripts)


@pytest.mark.parametrize("key", ["A_agent1", "B_agent2", "D_agent1"])
def test_joint_and_native_paths_continue_rejecting_malformed_mass(key: str) -> None:
    space, _ = _space("action_next_state_previous_state", 3)
    if key.startswith("A"):
        space.matrices[key][0][0] = 0.5
    elif key.startswith("B"):
        space.matrices[key][0][0][0] = 0.25
    else:
        space.matrices[key][0] = 0.25
    before = deepcopy(space.to_dict())
    for kwargs in ({}, {"native_agents": True}):
        with pytest.raises(ValueError, match="mass"):
            pomdp_to_gnn_spec(space, **kwargs)
    assert space.to_dict() == before
