"""Native projection is lazy, axis-bound, and cannot repair invalid source factors."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gnn.render.multi_agent_common import validate_native_agent_groups
from gnn.render.pomdp_processor import POMDPRenderProcessor


def _source(order: str = "next_state_previous_state_action") -> dict:
    # Equal axis sizes make shape-based inference insufficient; matrices are
    # asymmetric and doubly stochastic, so both accidental orientations pass mass.
    canonical = np.stack(
        [
            [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]],
            [[0.1, 0.6, 0.3], [0.3, 0.1, 0.6], [0.6, 0.3, 0.1]],
            [[0.4, 0.1, 0.5], [0.5, 0.4, 0.1], [0.1, 0.5, 0.4]],
        ],
        axis=2,
    )
    raw = (
        canonical
        if order == "next_state_previous_state_action"
        else canonical.transpose(2, 0, 1)
        if order == "action_next_state_previous_state"
        else canonical.transpose(2, 1, 0)
    )
    matrices = {
        f"{key}_agent{i}": value
        for i in (1, 2)
        for key, value in {
            "A": np.eye(3).tolist(),
            "B": raw.tolist(),
            "C": [0.0, 1.0, 2.0],
            "D": [1.0, 0.0, 0.0],
        }.items()
    }
    return {
        "structured_pomdp": {"matrices": matrices},
        "model_parameters": {"nr_agents": 2, "num_actions": 3, "b_tensor_order": order},
    }


@pytest.mark.parametrize(
    "order",
    [
        "next_state_previous_state_action",
        "action_next_state_previous_state",
        "action_previous_state_next_state",
    ],
)
def test_native_and_joint_declared_axes_agree_even_when_axes_equal(order: str) -> None:
    from gnn.render.pomdp_contract import canonicalise_b_matrix

    source = _source(order)
    raw = deepcopy(source)
    native = validate_native_agent_groups(source)
    expected, meta = canonicalise_b_matrix(
        source["structured_pomdp"]["matrices"]["B_agent1"],
        num_states=3,
        num_actions=3,
        model_parameters=source["model_parameters"],
    )
    np.testing.assert_array_equal(native["agent1"]["B"], expected)
    assert (
        native["agent1"]["matrix_provenance"]["B"]["source_order"]
        == meta["source_order"]
    )
    assert source == raw


def test_per_agent_axis_declaration_overrides_global() -> None:
    source = _source("action_next_state_previous_state")
    source["model_parameters"]["b_tensor_order_agent2"] = (
        "action_previous_state_next_state"
    )
    source["structured_pomdp"]["matrices"]["B_agent2"] = (
        np.asarray(source["structured_pomdp"]["matrices"]["B_agent2"])
        .transpose(0, 2, 1)
        .tolist()
    )
    native = validate_native_agent_groups(source)
    np.testing.assert_array_equal(native["agent1"]["B"], native["agent2"]["B"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("A_agent1", [[0.9, 0.1, 0.1], [0.2, 0.8, 0.1], [0.0, 0.1, 0.8]]),
        ("D_agent2", [0.3, 0.3, 0.3]),
        ("E_agent1", [0.2, 0.3, 0.5]),
    ],
)
def test_invalid_or_unsupported_native_source_is_not_coerced(
    key: str, value: list
) -> None:
    source = _source()
    source["structured_pomdp"]["matrices"][key] = value
    before = deepcopy(source)
    with pytest.raises(ValueError):
        validate_native_agent_groups(source)
    assert source == before


def test_real_native_swarm_projection_never_builds_joint_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gnn.extract.pomdp_extractor import extract_pomdp_from_file
    from gnn.render.pomdp_processor import pomdp_to_gnn_spec

    source = extract_pomdp_from_file(
        Path("input/gnn_files/multiagent/stigmergic_swarm.md")
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("native projection allocated an unused joint model")

    monkeypatch.setattr(
        POMDPRenderProcessor, "_build_canonical_initialparameterization", forbidden
    )
    spec = pomdp_to_gnn_spec(source, native_agents=True, timesteps=2)
    assert spec["canonical_pomdp_schema"] == "native_agent_pomdp_v1"
    assert "A" not in spec["initialparameterization"]
    assert spec["structured_pomdp"]["matrices"] == source.matrices
    assert set(validate_native_agent_groups(spec)) == {"agent1", "agent2", "agent3"}


def test_joint_normalization_is_derived_from_validated_sources_only(
    tmp_path: Path,
) -> None:
    source = _source()
    # An unreported descriptor makes the computed product contain repeated
    # identical likelihood columns/rows. This is derived projection mass.
    space = SimpleNamespace(
        matrices=source["structured_pomdp"]["matrices"],
        model_parameters=source["model_parameters"],
        state_factors=[
            {"name": "s_agent1", "size": 3},
            {"name": "s_agent2", "size": 3},
        ],
        observation_modalities=[
            {"name": "o_agent1", "size": 3},
            {"name": "o_agent2", "size": 3},
            {"name": "unreported", "size": 2},
        ],
        num_actions=3,
    )
    before = deepcopy(space.matrices)
    joint, provenance = POMDPRenderProcessor(tmp_path)._compose_factored_pomdp(space)
    np.testing.assert_allclose(np.asarray(joint["A"]).sum(axis=0), 1)
    assert provenance["A"]["derived"] is True
    assert (
        provenance["A"]["normalization_scope"]
        == "derived_product_of_validated_source_conditionals"
    )
    assert space.matrices == before
    space.matrices["D_agent1"] = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match="mass"):
        POMDPRenderProcessor(tmp_path)._compose_factored_pomdp(space)


def test_unknown_declared_axes_are_not_guessed() -> None:
    source = _source("next_state_previous_state_action")
    source["model_parameters"]["b_tensor_order"] = "typo_action_order"
    with pytest.raises(ValueError, match="Unsupported declared"):
        validate_native_agent_groups(source)
