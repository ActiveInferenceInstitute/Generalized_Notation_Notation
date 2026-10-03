"""Real released-wheel Gibbs witnesses against independent finite HMM math."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gnn.render.thrml import build_thrml_payload, render_gnn_to_thrml
from tests.render.test_thrml_contracts import (
    factored_model,
    independent_model,
    modality_model,
    model,
    options,
)

pytestmark = [pytest.mark.env_heavy, pytest.mark.needs_thrml]


def logspace_smoothing(
    parameters: dict, observed: list[int], actions: list[int]
) -> np.ndarray:
    """Independent log-space forward/backward; never calls renderer/runtime."""

    def lse(value: np.ndarray, axis: int | None = None) -> np.ndarray:
        maximum = np.max(value, axis=axis, keepdims=True)
        result = maximum + np.log(
            np.sum(np.exp(value - maximum), axis=axis, keepdims=True)
        )
        return np.squeeze(result, axis=axis) if axis is not None else result.squeeze()

    likelihood = np.log(np.asarray(parameters["A"]))
    transition = np.log(np.asarray(parameters["B"]))
    alpha = [np.log(np.asarray(parameters["D"])) + likelihood[observed[0]]]
    for step, action in enumerate(actions):
        alpha.append(
            likelihood[observed[step + 1]]
            + lse(transition[:, :, action] + alpha[-1][None, :], axis=1)
        )
    beta = [np.zeros_like(alpha[0]) for _ in observed]
    for step in range(len(observed) - 2, -1, -1):
        beta[step] = lse(
            transition[:, :, actions[step]]
            + (likelihood[observed[step + 1]] + beta[step + 1])[:, None],
            axis=0,
        )
    return np.stack(
        [np.exp(left + right - lse(left + right)) for left, right in zip(alpha, beta)]
    )


def execute_generated(
    tmp_path: Path,
    source: dict,
    opts: dict,
    name: str = "witness",
    *,
    managed: bool = False,
) -> dict:
    script = tmp_path / f"{name}_thrml.py"
    destination = tmp_path / f"{name}_results.json"
    success, reason, artifacts = render_gnn_to_thrml(source, script, opts)
    assert success, reason
    assert artifacts == [str(script)]
    managed_output = tmp_path / f"{name}_managed"
    env = dict(
        os.environ,
        GNN_THRML_EXECUTION_ID="native-contract-witness",
        THRML_OUTPUT_DIR=str(managed_output),
    )
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.pathsep.join(
            str(Path(entry).resolve()) for entry in env["PYTHONPATH"].split(os.pathsep)
        )
    command = [sys.executable, str(script)]
    if managed:
        destination = managed_output / "simulation_data" / "simulation_results.json"
    else:
        command.extend(["--output-json", str(destination)])
    child = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    if not managed:
        assert not managed_output.exists(), (
            "explicit --output-json must override managed output"
        )
    result = json.loads(destination.read_text())
    assert "THRML_RESULTS " in child.stdout
    assert result["runtime_metadata"]["execution_id"] == "native-contract-witness"
    assert (
        result["runtime_metadata"]["script_sha256"]
        == hashlib.sha256(script.read_bytes()).hexdigest()
    )
    assert result["dependencies"]["thrml"] == "0.1.4"
    assert (
        result["sampling_dtype"] == "float64"
        and result["runtime_metadata"]["jax_enable_x64"]
    )
    assert (
        result["inference_mode"] == "gibbs_monte_carlo" and "free_energy" not in result
    )
    return result


def verify_empirical(
    result: dict, expected: np.ndarray, tolerance: float = 0.04
) -> None:
    samples = np.asarray(result["sample_states"])
    beliefs = np.asarray(result["beliefs"])
    assert samples.shape == (result["sampling"]["num_samples"], result["num_timesteps"])
    empirical = np.stack(
        [
            np.bincount(samples[:, step], minlength=beliefs.shape[1]) / len(samples)
            for step in range(samples.shape[1])
        ]
    )
    np.testing.assert_array_equal(beliefs, empirical)
    np.testing.assert_allclose(beliefs, expected, rtol=0, atol=tolerance)
    np.testing.assert_allclose(
        result["predictive_observations"],
        beliefs @ np.asarray(result["parameters"]["A"]).T,
        rtol=2e-15,
        atol=2e-15,
    )
    assert "actions" not in result


def test_native_asymmetric_three_state_two_action_smoothing(tmp_path: Path) -> None:
    source, opts = model(), options()
    result = execute_generated(tmp_path, source, opts)
    expected = logspace_smoothing(
        source["initialparameterization"],
        opts["observations"],
        opts["transition_actions"],
    )
    verify_empirical(result, expected)
    assert result["source_parameters"] == source["initialparameterization"]
    assert result["parameters"]["C"] == [0.7, -0.2] and result["parameters"]["E"] == [
        0.3,
        0.7,
    ]
    assert result["transition_actions"] == [1, 0]


def test_native_seed_replay_and_one_timestep_synthetic_draw(tmp_path: Path) -> None:
    source = model()
    opts = {
        "num_timesteps": 1,
        "num_samples": 512,
        "burn_in": 64,
        "thin": 1,
        "seed": 123,
    }
    first = execute_generated(tmp_path, source, opts, "first")
    second = execute_generated(tmp_path, source, opts, "second")
    assert first["sample_states"] == second["sample_states"]
    assert first["observations"] == second["observations"]
    assert first["transition_actions"] == []
    assert first["native_sampling"]["data_origin"] == "synthetic_gibbs_joint_draw"
    assert len(first["native_sampling"]["generated_state_trajectory"]) == 1
    verify_empirical(
        first,
        logspace_smoothing(
            source["initialparameterization"], first["observations"], []
        ),
        tolerance=0.08,
    )


def test_native_dependent_factors_have_distinct_state_and_modal_marginals(
    tmp_path: Path,
) -> None:
    source = factored_model()
    opts = {**options(), "num_samples": 8192, "observations": [0, 4, 1]}
    payload = build_thrml_payload(source, opts)
    result = execute_generated(tmp_path, source, opts)
    raw = source["initialparameterization"]
    # Enumerate from original factor tables rather than using the renderer's
    # joint parameters as the numerical oracle's input.
    likelihood = np.empty((6, 6))
    transition = np.empty((6, 6, 2))
    prior = np.empty(6)
    for state0 in range(2):
        for state1 in range(3):
            previous = state0 * 3 + state1
            prior[previous] = raw["D_f0"][state0] * raw["D_f1"][state1]
            for obs0 in range(2):
                for obs1 in range(3):
                    likelihood[obs0 * 3 + obs1, previous] = (
                        raw["A_m0"][obs0][state0][state1] * raw["A_m1"][obs1][state1]
                    )
            for next0 in range(2):
                for next1 in range(3):
                    for action in range(2):
                        transition[next0 * 3 + next1, previous, action] = (
                            raw["B_f0"][next0][state0]
                            * raw["B_f1"][next1][state1][action]
                        )
    parameters = {"A": likelihood, "B": transition, "D": prior}
    for key, expected_value in parameters.items():
        np.testing.assert_allclose(
            payload["components"][0]["parameters"][key],
            expected_value,
            rtol=2e-15,
            atol=0,
        )
        np.testing.assert_allclose(
            result["parameters"][key], expected_value, rtol=2e-15, atol=0
        )
    expected = logspace_smoothing(
        parameters,
        opts["observations"],
        opts["transition_actions"],
    )
    verify_empirical(result, expected, tolerance=0.035)
    beliefs = np.asarray(result["beliefs"]).reshape(3, 2, 3)
    np.testing.assert_allclose(
        result["state_factor_marginals"]["s_f0"], beliefs.sum(axis=2)
    )
    np.testing.assert_allclose(
        result["state_factor_marginals"]["s_f1"], beliefs.sum(axis=1)
    )
    predictions = np.asarray(result["predictive_observations"]).reshape(3, 2, 3)
    np.testing.assert_allclose(
        result["observation_modality_marginals"]["o_m0"], predictions.sum(axis=2)
    )
    np.testing.assert_allclose(
        result["observation_modality_marginals"]["o_m1"], predictions.sum(axis=1)
    )
    assert (
        result["composition_metadata"]["source_matrices"]
        == source["initialparameterization"]
    )


def test_native_independent_agents_are_separately_bound_and_sampled(
    tmp_path: Path,
) -> None:
    source = independent_model()
    source["initialparameterization"]["D_agent2"] = [0.3, 0.5, 0.2]
    opts = {
        **options(),
        "observations": {"agent1": [0, 1, 0], "agent2": [1, 0, 1]},
        "transition_actions": {"agent1": [1, 0], "agent2": [0, 1]},
    }
    result = execute_generated(tmp_path, source, opts)
    assert set(result["components"]) == {"agent1", "agent2"}
    assert "beliefs" not in result and "sample_states" not in result
    for name, child in result["components"].items():
        expected = logspace_smoothing(
            child["parameters"],
            opts["observations"][name],
            opts["transition_actions"][name],
        )
        verify_empirical(child, expected)
        assert child["component_id"] == name
    assert (
        result["components"]["agent1"]["source_parameters"]["D"]
        != result["components"]["agent2"]["source_parameters"]["D"]
    )


def test_native_observation_modalities_use_exact_product_then_gibbs_smoothing(
    tmp_path: Path,
) -> None:
    source = modality_model()
    opts = {**options(), "num_samples": 8192, "observations": [0, 4, 1]}
    result = execute_generated(tmp_path, source, opts)
    raw = source["initialparameterization"]
    # Independent construction of the product likelihood is part of the oracle;
    # do not use the renderer's joint table to establish composition correctness.
    parameters = {
        "A": [
            (np.asarray(raw["A_m0"][i]) * raw["A_m1"][j]).tolist()
            for i in range(2)
            for j in range(3)
        ],
        "B": raw["B"],
        "D": raw["D"],
    }
    expected = logspace_smoothing(
        parameters, opts["observations"], opts["transition_actions"]
    )
    verify_empirical(result, expected, tolerance=0.035)
    predictive = np.asarray(result["predictive_observations"]).reshape(3, 2, 3)
    np.testing.assert_allclose(
        result["observation_modality_marginals"]["A_m0"], predictive.sum(axis=2)
    )
    np.testing.assert_allclose(
        result["observation_modality_marginals"]["A_m1"], predictive.sum(axis=1)
    )
    assert result["composition_metadata"]["source_matrices"] == raw
    assert result["source_model_parameters"] == source["model_parameters"]
    assert result["composed_model_parameters"]["num_obs"] == 6


def test_native_independent_factor_groups_keep_separate_posteriors(
    tmp_path: Path,
) -> None:
    source = independent_model()
    source["initialparameterization"] = {
        key.replace("agent1", "f0").replace("agent2", "f1"): value
        for key, value in source["initialparameterization"].items()
    }
    source["initialparameterization"]["D_f1"] = [0.5, 0.4, 0.1]
    source["structured_pomdp"]["matrices"] = source["initialparameterization"]
    source["model_parameters"].pop("nr_agents")
    source["model_parameters"]["num_factors"] = 2
    opts = {
        **options(),
        "observations": {"f0": [0, 1, 0], "f1": [1, 0, 1]},
        "transition_actions": {"f0": [1, 0], "f1": [0, 1]},
    }
    result = execute_generated(tmp_path, source, opts)
    assert set(result["components"]) == {"f0", "f1"}
    assert result["source_model_parameters"]["num_factors"] == 2
    for name, child in result["components"].items():
        parameters = {
            key: source["initialparameterization"][f"{key}_{name}"]
            for key in ("A", "B", "D")
        }
        expected = logspace_smoothing(
            parameters, opts["observations"][name], opts["transition_actions"][name]
        )
        verify_empirical(child, expected)


def test_native_managed_artifact_uses_executor_output_directory(tmp_path: Path) -> None:
    opts = {**options(), "num_samples": 1024}
    result = execute_generated(tmp_path, model(), opts, "managed", managed=True)
    assert not (tmp_path / "thrml_results.json").exists()
    expected = logspace_smoothing(
        model()["initialparameterization"],
        opts["observations"],
        opts["transition_actions"],
    )
    verify_empirical(result, expected, tolerance=0.07)


def test_native_uint16_category_indices_preserve_category_299(tmp_path: Path) -> None:
    width = 300
    likelihood = np.full((2, width), 0.2)
    likelihood[0, -1] = 0.9
    likelihood[1] = 1 - likelihood[0]
    prior = np.full(width, 0.05 / (width - 1))
    prior[-1] = 0.95
    source = {
        "model_name": "300-category uint16 boundary",
        "initialparameterization": {
            "A": likelihood.tolist(),
            "B": np.full((width, width, 1), 1 / width).tolist(),
            "C": [0, 0],
            "D": prior.tolist(),
            "E": [1.0],
        },
        "model_parameters": {
            "num_states": width,
            "num_hidden_states": width,
            "num_obs": 2,
            "num_actions": 1,
            "b_tensor_order": "next_state_previous_state_action",
        },
    }
    opts = {
        "num_timesteps": 1,
        "num_samples": 2048,
        "burn_in": 64,
        "thin": 1,
        "seed": 36,
        "observations": [0],
        "transition_actions": [],
    }
    result = execute_generated(tmp_path, source, opts, "uint16")
    # One-step Bayes' rule directly from original source, independent of adapter.
    expected = prior * likelihood[0]
    expected /= expected.sum()
    verify_empirical(result, expected[None, :], tolerance=0.015)
    assert result["native_sampling"]["categorical_index_dtype"] == "uint16"
    assert 299 in np.asarray(result["sample_states"])
    assert max(np.asarray(result["sample_states"]).ravel()) == 299
