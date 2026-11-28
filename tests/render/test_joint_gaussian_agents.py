"""Source-bound native JAX witnesses for coupled Gaussian agent filtering."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from gnn.analysis.result_adapter import result_views
from gnn.extract.pomdp_extractor import extract_pomdp_from_file
from gnn.render.jax.jax_renderer import render_gnn_to_jax
from gnn.render.pomdp_processor import POMDPRenderProcessor, pomdp_to_gnn_spec
from gnn.render.processor import render_gnn_spec

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "input/gnn_files/continuous/joint_gaussian_agents.md"
)


def _spec():
    model = extract_pomdp_from_file(SOURCE, on_error="raise")
    assert model is not None
    return model, pomdp_to_gnn_spec(model)


def _authored_parameters():
    block = SOURCE.read_text(encoding="utf-8").split("## InitialParameterization")[1]
    block = block.split("## Equations")[0]
    return {
        name.strip(): json.loads(value)
        for line in block.splitlines()
        if line.strip()
        for name, value in [line.split("=", 1)]
    }


def _independent_conditioning(parameters):
    # Construct the complete joint Gaussian from dynamics powers and original
    # measurements, then condition afresh at each prefix. No recursive Kalman
    # routine, renderer data, or posterior arrays enter this computation.
    F, H, Q, R, mu, P, U, controls, readings = (
        np.asarray(parameters[key])
        for key in (
            "F",
            "H",
            "Q",
            "R",
            "prior_mean",
            "prior_cov",
            "control_matrix",
            "control_sequence",
            "observation_sequence",
        )
    )
    steps, m = readings.shape
    n = len(mu)
    powers = [np.linalg.matrix_power(F, t) for t in range(steps)]
    means = np.stack(
        [
            powers[t] @ mu
            + sum(
                (powers[t - k - 1] @ U @ controls[k] for k in range(t)),
                start=np.zeros(n),
            )
            for t in range(steps)
        ]
    )
    state_cov = np.block(
        [
            [
                sum(
                    (
                        powers[t - k] @ (P if k == 0 else Q) @ powers[s - k].T
                        for k in range(min(t, s) + 1)
                    ),
                    start=np.zeros((n, n)),
                )
                for s in range(steps)
            ]
            for t in range(steps)
        ]
    )
    block_H = np.kron(np.eye(steps), H)
    obs_cov = block_H @ state_cov @ block_H.T + np.kron(np.eye(steps), R)
    cross_cov = state_cov @ block_H.T
    residual = (readings - means @ H.T).ravel()
    filtered_means, filtered_cov = [], []
    for step in range(steps):
        state = slice(step * n, (step + 1) * n)
        observed = slice(0, (step + 1) * m)
        match = cross_cov[state, observed]
        gain = np.linalg.solve(obs_cov[observed, observed], match.T).T
        filtered_means.append(means[step] + gain @ residual[observed])
        filtered_cov.append(state_cov[state, state] - gain @ match.T)
    return np.asarray(filtered_means), np.asarray(filtered_cov)


def _native(spec, tmp_path):
    ok, reason, files = render_gnn_spec(spec, "jax", tmp_path)
    assert ok, reason
    env = {**os.environ, "GNN_OUTPUT_DIR": str(tmp_path), "JAX_PLATFORMS": "cpu"}
    subprocess.run([sys.executable, files[0]], env=env, timeout=60, check=True)
    return json.loads((tmp_path / "simulation_results.json").read_text())


def _reorder(parameters, state, observation, control):
    for key, rows, cols in (
        ("F", state, state),
        ("H", observation, state),
        ("Q", state, state),
        ("R", observation, observation),
        ("prior_cov", state, state),
        ("control_matrix", state, control),
    ):
        parameters[key] = np.asarray(parameters[key])[np.ix_(rows, cols)].tolist()
    parameters["prior_mean"] = np.asarray(parameters["prior_mean"])[state].tolist()
    parameters["observation_sequence"] = np.asarray(parameters["observation_sequence"])[
        :, observation
    ].tolist()
    parameters["control_sequence"] = np.asarray(parameters["control_sequence"])[
        :, control
    ].tolist()
    parameters["joint_agents"].reverse()


def test_native_coupled_filter_matches_joint_conditioning_and_source_identity(tmp_path):
    _, spec = _spec()
    runtime = _native(spec, tmp_path / "joint")
    means, covariance = _independent_conditioning(_authored_parameters())
    np.testing.assert_allclose(runtime["beliefs"], means, rtol=0, atol=1e-12)
    np.testing.assert_allclose(runtime["posterior_cov"], covariance, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        runtime["posterior_cross_covariance"]["values"],
        covariance[:, :1, 1:],
        rtol=0,
        atol=1e-12,
    )
    assert np.max(np.abs(covariance[:, :1, 1:])) > 1e-3
    assert runtime["runtime_metadata"]["jax_platform"] == "cpu"
    assert runtime["dtype"] == "float64"
    views = result_views(runtime)
    assert list(views) == ["north", "south"]
    np.testing.assert_array_equal(
        views["north"]["control_inputs"], [[0.4], [-0.3], [0.1]]
    )
    assert runtime["agent_layout"][1]["state_slice"] == [1, 3]

    later = deepcopy(spec)
    later["initialparameterization"]["observation_sequence"][-1] = [9.0, 9.0, 9.0]
    changed = _native(later, tmp_path / "later")
    np.testing.assert_array_equal(changed["beliefs"][:-1], runtime["beliefs"][:-1])
    assert not np.allclose(changed["beliefs"][-1], runtime["beliefs"][-1])
    assert changed["source_spec_sha256"] != runtime["source_spec_sha256"]

    permuted = deepcopy(spec)
    _reorder(permuted["initialparameterization"], [1, 2, 0], [2, 0, 1], [1, 0])
    swapped = _native(permuted, tmp_path / "swapped")
    assert swapped["agents"] == ["south", "north"]
    np.testing.assert_allclose(
        swapped["beliefs"], means[:, [1, 2, 0]], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        swapped["posterior_cov"],
        covariance[:, [1, 2, 0]][:, :, [1, 2, 0]],
        rtol=0,
        atol=1e-12,
    )


def test_zero_coupling_matches_two_independent_filters(tmp_path):
    _, spec = _spec()
    parameters = spec["initialparameterization"]
    splits = {
        "F": (1, 1),
        "H": (2, 1),
        "Q": (1, 1),
        "R": (2, 2),
        "prior_cov": (1, 1),
        "control_matrix": (1, 1),
    }
    for key, (rows, cols) in splits.items():
        matrix = np.asarray(parameters[key])
        matrix[:rows, cols:] = 0
        matrix[rows:, :cols] = 0
        parameters[key] = matrix.tolist()
    runtime = _native(spec, tmp_path / "uncoupled")
    np.testing.assert_array_equal(runtime["posterior_cross_covariance"]["values"], 0.0)
    agents = [
        ("north", slice(0, 1), slice(0, 2), slice(0, 1)),
        ("south", slice(1, 3), slice(2, 3), slice(1, 2)),
    ]
    for name, state, obs, control in agents:
        local = {
            key: np.asarray(parameters[key])[row, column]
            for key, row, column in (
                ("F", state, state),
                ("H", obs, state),
                ("Q", state, state),
                ("R", obs, obs),
                ("prior_cov", state, state),
                ("control_matrix", state, control),
                ("observation_sequence", slice(None), obs),
                ("control_sequence", slice(None), control),
            )
        }
        local["prior_mean"] = np.asarray(parameters["prior_mean"])[state]
        means, cov = _independent_conditioning(local)
        np.testing.assert_allclose(
            runtime["beliefs_by_agent"][name], means, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            runtime["posterior_cov_by_agent"][name], cov, rtol=0, atol=1e-12
        )


@pytest.mark.parametrize(
    "mutation",
    ["control_sequence", "bad_covariance", "goal_mean", "too_many_steps", "boolean"],
)
def test_invalid_coupling_refuses_before_replacing_output(tmp_path, mutation):
    _, spec = _spec()
    source = spec["initialparameterization"]
    if mutation == "control_sequence":
        source.pop(mutation)
    elif mutation == "bad_covariance":
        source["Q"][0][0] = -1.0
    elif mutation == "goal_mean":
        source[mutation] = [0.0] * 3
    elif mutation == "too_many_steps":
        spec["model_parameters"]["num_timesteps"] = 129
    else:
        source["control_matrix"][0][0] = True
    output = tmp_path / "existing.py"
    output.write_text("previous artifact\n")
    ok, _, files = render_gnn_to_jax(spec, output)
    assert not ok and not files
    assert output.read_text() == "previous artifact\n"


def test_step11_declares_joint_backend_support_without_flattening(tmp_path):
    model, spec = _spec()
    processor = POMDPRenderProcessor(tmp_path)
    summary = processor.process_pomdp_for_all_frameworks(
        model, SOURCE, frameworks=["jax", "rxinfer"]
    )
    results = summary["framework_results"]
    assert results["jax"]["success"]
    assert results["rxinfer"]["status"] == "unsupported"
    for backend in ("rxinfer", "pymdp", "numpyro", "cpomdp"):
        ok, reason, files = render_gnn_spec(spec, backend, tmp_path / backend)
        assert not ok and "unsupported-composition" in reason and files == []
