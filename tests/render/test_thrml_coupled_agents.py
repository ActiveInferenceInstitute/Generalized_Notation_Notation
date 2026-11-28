"""Source binding and native witnesses for two-agent coupled smoothing."""

from __future__ import annotations

import itertools
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from gnn.render.processor import render_gnn_spec
from gnn.render.thrml import build_thrml_payload, render_gnn_to_thrml
from tests.render.test_thrml_native import (
    execute_generated,
    logspace_smoothing,
    verify_empirical,
)


def _source(*, independent: bool = False) -> dict:
    b0 = np.empty((2, 2, 3, 2))
    b1 = np.empty((3, 2, 3, 2))
    for left, right, action in itertools.product(range(2), range(3), range(2)):
        p = 0.18 + 0.25 * left + 0.10 * action + (0 if independent else 0.14 * right)
        q = 0.16 + 0.22 * (right == 0) + (0 if independent else 0.09 * left)
        r = 0.25 + 0.11 * action + 0.07 * (right == 1)
        b0[:, left, right, action] = [1 - p, p]
        b1[:, left, right, action] = [q, r, 1 - q - r]
    prior = (
        np.outer([0.51, 0.49], [0.34, 0.36, 0.30]).tolist()
        if independent
        else [
            [0.28, 0.12, 0.11],
            [0.06, 0.24, 0.19],
        ]
    )
    return {
        "model_name": "Asymmetric two-agent coupled witness",
        "model_parameters": {"nr_agents": 2},
        "initialparameterization": {
            "thrml_coupled_agents": {
                "contract": "gnn-thrml-two-agent-positive-v1",
                "agent_ids": ["left", "right"],
                "action_ids": ["drift", "route"],
                "A": [
                    [[0.8, 0.21], [0.2, 0.79]],
                    [[0.72, 0.39, 0.19], [0.28, 0.61, 0.81]],
                ],
                "B": [b0.tolist(), b1.tolist()],
                "D": prior,
                "observations": [[0, 1], [1, 0], [0, 1], [1, 1], [1, 0]],
                "transition_actions": ["route", "drift", "route", "drift"],
            },
        },
    }


def _declaration(source: dict) -> dict:
    return source["initialparameterization"]["thrml_coupled_agents"]


def _exact_joint_path_posterior(source: dict) -> np.ndarray:
    """Enumerate physical joint paths directly from authored local tables."""
    raw = _declaration(source)
    readings = raw["observations"]
    actions = [raw["action_ids"].index(action) for action in raw["transition_actions"]]
    states = list(itertools.product(range(2), range(3)))
    posterior = np.zeros((len(readings), len(states)))
    for path in itertools.product(range(len(states)), repeat=len(readings)):
        first, second = states[path[0]]
        weight = raw["D"][first][second]
        for t, identifier in enumerate(path):
            left, right = states[identifier]
            weight *= raw["A"][0][readings[t][0]][left]
            weight *= raw["A"][1][readings[t][1]][right]
            if t:
                action = actions[t - 1]
                previous_left, previous_right = states[path[t - 1]]
                weight *= raw["B"][0][left][previous_left][previous_right][action]
                weight *= raw["B"][1][right][previous_left][previous_right][action]
        for t, identifier in enumerate(path):
            posterior[t, identifier] += weight
    return posterior / posterior.sum(axis=1, keepdims=True)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"contract": "future"}, "contract"),
        ({"agent_ids": ["left", "left"]}, "distinct IDs"),
        ({"agent_ids": ["left", "right", "third"]}, "exactly 2 IDs"),
        ({"action_ids": ["same", "same"]}, "distinct IDs"),
        ({"transition_actions": ["route"] * 3}, "T - 1"),
        ({"transition_actions": ["unknown"] * 4}, "declared action_ids"),
        ({"observations": []}, "authored nonempty sequence"),
        ({"observations": [[False, 1]] * 5}, "must be an integer"),
        ({"observations": [[0, 2]] * 5}, "admitted range"),
        ({"D": [[0.0, 0.2, 0.2], [0.2, 0.2, 0.2]]}, "structural-zero"),
        ({"D": [[0.1, 0.2, 0.2], [0.2, 0.2, 0.2]]}, "mass must be one"),
    ],
)
def test_coupled_source_refuses_invalid_declarations(
    tmp_path: Path, changes: dict, message: str
) -> None:
    source = _source()
    _declaration(source).update(changes)
    artifact = tmp_path / "existing.py"
    artifact.write_text("existing artifact")
    success, reason, paths = render_gnn_to_thrml(source, artifact)
    assert not success and message in reason and not paths
    assert artifact.read_text() == "existing artifact"


def test_coupled_source_caps_and_requires_its_own_conditioning(tmp_path: Path) -> None:
    source = _source()
    for opts, reason in (
        ({"observations": [0] * 5}, "from the source"),
        ({"transition_actions": [0] * 4}, "from the source"),
        ({"num_timesteps": 3}, "match the authored observations"),
        ({"num_samples": 10_000_000}, "resource admission rejected"),
    ):
        with pytest.raises(ValueError, match=reason):
            build_thrml_payload(source, opts)
    success, reason, paths = render_gnn_spec(source, "jax", tmp_path)
    assert not success and "source-declared two-agent smoothing" in reason and not paths

    source["initial_parameterization"] = source["initialparameterization"]
    source["initialparameterization"] = {"other": "nonempty precedence branch"}
    with pytest.raises(ValueError, match="conflicting thrml_coupled_agents"):
        build_thrml_payload(source)


def test_coupled_source_axis_products_and_authored_conditioning() -> None:
    source = _source()
    raw = _declaration(source)
    payload = build_thrml_payload(source)
    (component,) = payload["components"]
    assert payload["composition"] == "two_agent_coupled"
    assert component["composition_metadata"]["source_declaration"] == raw
    assert component["observations"] == [1, 2, 1, 3, 2]
    assert component["transition_actions"] == [1, 0, 1, 0]
    assert component["action_origin"] == "source_fixed_transition_actions"
    a = np.asarray(component["parameters"]["A"]).reshape(2, 2, 2, 3)
    b = np.asarray(component["parameters"]["B"]).reshape(2, 3, 2, 3, 2)
    for i, j, k, l in itertools.product(range(2), range(2), range(2), range(3)):
        assert a[i, j, k, l] == pytest.approx(raw["A"][0][i][k] * raw["A"][1][j][l])
    for i, j, k, l, action in itertools.product(
        range(2), range(3), range(2), range(3), range(2)
    ):
        assert b[i, j, k, l, action] == pytest.approx(
            raw["B"][0][i][k][l][action] * raw["B"][1][j][k][l][action]
        )
    assert payload["resource_admission"]["retained_matrix_entries"] == 182


def test_coupled_source_survives_markdown_cli(tmp_path: Path) -> None:
    from gnn.cli import main

    source = tmp_path / "source.md"
    source.write_text(
        "## GNNSection\nActInfPOMDP\n\n## ModelName\nTHRML joint source\n\n"
        "## StateSpaceBlock\ns[6,type=int]\no[4,type=int]\nu[2,type=int]\n\n"
        f"## InitialParameterization\nthrml_coupled_agents = {_declaration(_source())!r}\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "model.py"
    assert (
        main(["render", str(source), "--framework", "thrml", "--output", str(artifact)])
        == 0
    )
    assert "gnn-thrml-two-agent-positive-v1" in artifact.read_text()


def test_serialized_coupled_payload_checks_source_even_if_rehashed() -> None:
    from gnn.render.thrml.adapter import contract_digest
    from gnn.render.thrml.runtime import _validate_payload

    payload = build_thrml_payload(_source())
    _validate_payload(payload)
    payload["components"][0]["transition_actions"][0] = 0
    payload["source_identity"]["semantic_sha256"] = contract_digest(
        {key: payload[key] for key in ("components", "sampling", "posterior_semantics")}
    )
    with pytest.raises(ValueError, match="disagree with their source declaration"):
        _validate_payload(payload)


@pytest.mark.env_heavy
@pytest.mark.needs_thrml
def test_native_coupled_cpu_posterior_matches_all_source_paths(tmp_path: Path) -> None:
    from gnn.analysis.thrml import adapt_result

    source = _source()
    opts = {"num_samples": 8192, "burn_in": 512, "thin": 2, "seed": 361}
    result = execute_generated(tmp_path, source, opts, "coupled")
    assert result["runtime_metadata"]["device_platform"] == "cpu"
    assert len(result["runtime_metadata"]["devices"]) == 1
    assert result["composition_metadata"]["source_declaration"] == _declaration(source)
    expected = _exact_joint_path_posterior(source)
    verify_empirical(result, expected, tolerance=0.035)
    joint = np.asarray(result["beliefs"]).reshape(5, 2, 3)
    np.testing.assert_allclose(
        result["state_factor_marginals"]["left"], joint.sum(axis=2)
    )
    np.testing.assert_allclose(
        result["state_factor_marginals"]["right"], joint.sum(axis=1)
    )
    adapted = adapt_result(result)
    assert adapted["composition"] == "two_agent_coupled"
    assert adapted["runtime_metadata"]["device_platform"] == "cpu"
    wrong = deepcopy(source)
    b0 = np.asarray(_declaration(source)["B"][0])
    _declaration(wrong)["B"][0] = np.roll(b0, 1, axis=2).tolist()
    assert np.max(np.abs(expected - _exact_joint_path_posterior(wrong))) > 0.05

    replay = execute_generated(tmp_path, source, opts, "replay")
    assert replay["sample_states"] == result["sample_states"]
    assert replay["source_identity"] == result["source_identity"]


@pytest.mark.env_heavy
@pytest.mark.needs_thrml
def test_native_coupled_contract_reduces_under_independent_transitions(
    tmp_path: Path,
) -> None:
    source = _source(independent=True)
    raw = _declaration(source)
    opts = {"num_samples": 8192, "burn_in": 512, "thin": 2, "seed": 361}
    result = execute_generated(tmp_path, source, opts, "independent-limit")
    actions = [raw["action_ids"].index(action) for action in raw["transition_actions"]]
    prior = np.asarray(raw["D"])
    independent = []
    for i, (transition, state_prior) in enumerate(
        (
            (np.asarray(raw["B"][0])[:, :, 0, :], prior.sum(axis=1)),
            (np.asarray(raw["B"][1])[:, 0, :, :], prior.sum(axis=0)),
        )
    ):
        independent.append(
            logspace_smoothing(
                {"A": raw["A"][i], "B": transition, "D": state_prior},
                [reading[i] for reading in raw["observations"]],
                actions,
            )
        )
    expected = np.einsum("ti,tj->tij", *independent).reshape(5, 6)
    np.testing.assert_allclose(
        expected, _exact_joint_path_posterior(source), atol=1e-12
    )
    verify_empirical(result, expected, tolerance=0.035)
