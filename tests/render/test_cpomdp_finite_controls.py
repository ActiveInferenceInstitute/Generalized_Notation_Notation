"""E3-FA1 source control admission and independent released-wheel witnesses."""

from __future__ import annotations

import copy
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from gnn.execute.subprocess_envelope import run_subprocess_envelope
from gnn.render.cpomdp import render_gnn_to_cpomdp
from gnn.render.cpomdp.admission import admit_search
from gnn.render.processor import render_gnn_spec


def _spec():
    return {
        "model_kind": "continuous",
        "model_name": "finite_controls_witness",
        "model_parameters": {"num_timesteps": 4, "random_seed": 19},
        "initialparameterization": {
            "F": [[0.8, 0.1, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.7]],
            "H": [[1.0, 0.2, -0.1], [0.0, 0.3, 1.0]],
            "Q": [[0.1, 0.01, 0.0], [0.01, 0.2, 0.02], [0.0, 0.02, 0.15]],
            "R": [[0.2, 0.03], [0.03, 0.3]],
            "prior_mean": [-1.0, 0.2, 0.8],
            "prior_cov": [[0.8, 0.1, -0.1], [0.1, 0.6, 0.05], [-0.1, 0.05, 0.9]],
            "goal_mean": [1.0, -0.4, 0.2],
            "control_gain": 0.1,
            "cpomdp_control": {
                "contract": "gnn-cpomdp-finite-actions-v1",
                "version": "thruster-v1",
                "action_ids": ["forward", "reverse", "forward-alias"],
                "control_units": ["newton"],
                "state_units": ["m/s", "m/s", "m/s"],
                "actions": [[0.3], [-0.15], [0.3]],
                "control_matrix": [[0.2], [-0.4], [0.7]],
            },
        },
    }


@pytest.mark.parametrize(
    "mutation",
    [
        {"contract": "future"},
        {"version": ""},
        {"typo": 1},
        {"action_ids": ["a", "a", "c"]},
        {"action_ids": ["a", "b"]},
        {"control_units": []},
        {"state_units": ["m/s"]},
        {"actions": [[True], [0], [1]]},
        {"actions": [["0.3"], [0], [1]]},
        {"actions": [[float("nan")], [0], [1]]},
        {"actions": [[float("inf")], [0], [1]]},
        {"actions": [[0.0, 1.0], [0.0], [1.0]]},
        {"actions": [[0.0]]},
        {"control_matrix": [[1.0], [True], [0.0]]},
        {"control_matrix": [[1.0], [float("inf")], [0.0]]},
        {"control_matrix": [[1.0, 0.0], [1.0], [0.0]]},
        {"control_matrix": [[1.0], [0.0]]},
    ],
)
def test_bad_source_controls_preserve_existing_artifact(tmp_path, mutation):
    spec = _spec()
    spec["initialparameterization"]["cpomdp_control"].update(mutation)
    dest = tmp_path / "model.py"
    dest.write_text("previous artifact")
    success, message, artifacts = render_gnn_to_cpomdp(spec, dest)
    assert not success and "cpomdp_control" in message and not artifacts
    assert dest.read_text() == "previous artifact"


@pytest.mark.parametrize("options", [{"control_mode": "parity"}, {"action_scale": 0.5}])
def test_source_control_semantics_cannot_be_overridden(tmp_path, options):
    success, message, _ = render_gnn_to_cpomdp(_spec(), tmp_path / "bad.py", options)
    assert not success and "requires efe mode" in message


def test_source_controls_cannot_be_silently_made_passive(tmp_path):
    spec = _spec()
    for key in ("goal_mean", "control_gain"):
        del spec["initialparameterization"][key]
    success, message, _ = render_gnn_to_cpomdp(spec, tmp_path / "bad.py")
    assert not success and "requires goal_mean" in message


@pytest.mark.parametrize(
    "framework", ["jax", "rxinfer", "numpyro", "pytorch", "ngclearn", "stan"]
)
def test_other_continuous_backends_refuse_source_control_contract(tmp_path, framework):
    success, message, artifacts = render_gnn_spec(
        _spec(), framework, tmp_path / framework
    )
    assert not success and "unsupported-cpomdp-control" in message and not artifacts


def test_custom_action_count_and_total_budget_before_allocation(tmp_path):
    receipt = admit_search(3, 2, 1, 4, 2, True, action_count=3)
    assert receipt["policy_count"] == 9
    assert receipt["step_evaluations_per_cycle"] == 18
    assert receipt["total_step_evaluations"] == 72
    with pytest.raises(ValueError, match="max_policies"):
        admit_search(3, 2, 1, 4, 1000000, True, action_count=3)
    with pytest.raises(ValueError, match="max_total_step_evaluations"):
        admit_search(3, 2, 1, 20000, 2, True, action_count=3)
    with pytest.raises(ValueError, match="max_estimated_bytes"):
        admit_search(3, 2, 1, 4, 2, True, max_estimated_bytes=1, action_count=3)
    spec = _spec()
    success, message, artifacts = render_gnn_to_cpomdp(
        spec, tmp_path / "huge.py", {"horizon": 1000000}
    )
    assert not success and "max_policies" in message and not artifacts
    assert not (tmp_path / "huge.py").exists()


def _reference(initial, result):
    # Source-authored parameters; no generator/extractor or emitted parameter copies.
    F, H, Q, R = (np.asarray(initial[k]) for k in ("F", "H", "Q", "R"))
    M = np.asarray(initial["cpomdp_control"]["control_matrix"])
    mean, cov = np.array(initial["prior_mean"]), np.array(initial["prior_cov"])
    goal = H @ np.array(initial["goal_mean"])
    policies = list(itertools.product(initial["cpomdp_control"]["actions"], repeat=2))
    means, covs, scores = [], [], []
    for t, obs in enumerate(result["observations_continuous"]):
        if t:
            mean = F @ mean + M @ np.array(result["control_inputs"][t - 1])
            cov = F @ cov @ F.T + Q
        K = np.linalg.solve(H @ cov @ H.T + R, H @ cov).T
        mean = mean + K @ (obs - H @ mean)
        residual = np.eye(3) - K @ H
        cov = residual @ cov @ residual.T + K @ R @ K.T
        means.append(mean.copy())
        covs.append(cov.copy())
        row = []
        for policy in policies:
            mu, sigma = mean.copy(), cov.copy()
            pragmatic = epistemic = 0.0
            for action in policy:
                mu = F @ mu + M @ action
                sigma = F @ sigma @ F.T + Q
                innovation = H @ sigma @ H.T + R
                delta = H @ mu - goal
                pragmatic += 0.5 * (delta @ delta + np.trace(innovation))
                epistemic += 0.5 * (
                    np.linalg.slogdet(innovation)[1] - np.linalg.slogdet(R)[1]
                )
                gain = np.linalg.solve(innovation, H @ sigma).T
                residual = np.eye(3) - gain @ H
                sigma = residual @ sigma @ residual.T + gain @ R @ gain.T
            row.append([pragmatic - epistemic, pragmatic, epistemic])
        scores.append(row)
    return np.array(means), np.array(covs), np.array(scores)


@pytest.mark.needs_cpomdp
def test_direct_released_library_matches_independent_finite_control_reference(tmp_path):
    spec = _spec()
    script = tmp_path / "direct_native.py"
    receipt = tmp_path / "direct_native.json"
    script.write_text(
        """import itertools
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import cpomdp
from cpomdp import Belief, KalmanBackend, LinearGaussianModel, Preference
from cpomdp.efe import policy_efe
from cpomdp.enumeration import EnumeratedEFESearch, FiniteActionSet

jax.config.update("jax_enable_x64", True)
spec = json.loads(sys.argv[1])
initial = spec["initialparameterization"]
declared = initial["cpomdp_control"]
F, H, Q, R = (jnp.asarray(initial[k], dtype=jnp.float64) for k in ("F", "H", "Q", "R"))
M = jnp.asarray(declared["control_matrix"], dtype=jnp.float64)
prior = Belief(jnp.asarray(initial["prior_mean"]), jnp.asarray(initial["prior_cov"]))
model = LinearGaussianModel(
    dynamics_matrix=F, observation_matrix=H, dynamics_noise=Q,
    observation_noise=R, prior=prior, control_matrix=M,
)
backend = KalmanBackend(model)
first = KalmanBackend(LinearGaussianModel(
    dynamics_matrix=jnp.eye(3), observation_matrix=H, dynamics_noise=jnp.zeros((3, 3)),
    observation_noise=R, prior=prior,
))
search = EnumeratedEFESearch(
    model, FiniteActionSet(declared["actions"], version=declared["version"]), horizon=2
)
preference = Preference(H @ jnp.asarray(initial["goal_mean"]), jnp.eye(2))
policies = search.policies
policy_ids = list(itertools.product(declared["action_ids"], repeat=2))
rng = np.random.default_rng(spec["model_parameters"]["random_seed"])
state = rng.multivariate_normal(initial["prior_mean"], initial["prior_cov"])
belief = prior
action = None
result = {
    "cpomdp_version": cpomdp.__version__,
    "policies": np.asarray(policies).tolist(),
    "cost_per_cycle": int(search.cost_per_plan),
    "observations_continuous": [], "beliefs": [], "posterior_cov": [],
    "controls": [], "control_inputs": [], "scores": [], "selected_policy_index": [],
    "policy_action_ids": [], "step_evaluations": 0,
}
for t in range(spec["model_parameters"]["num_timesteps"]):
    if t:
        state = np.asarray(F) @ state + np.asarray(M @ action)
        state += rng.multivariate_normal(np.zeros(3), np.asarray(Q))
    reading = np.asarray(H) @ state + rng.multivariate_normal(np.zeros(2), np.asarray(R))
    belief = first.infer_states(reading, prior) if t == 0 else backend.infer_states(
        reading, belief, action
    )
    scores = []
    for policy in policies:
        native = policy_efe(model, belief, policy, preference)
        scores.append([float(native.g), float(native.pragmatic), float(native.epistemic)])
        result["step_evaluations"] += len(policy)
    best = int(np.argmin(np.asarray(scores)[:, 0]))
    action = policies[best][0]
    for key, value in (
        ("observations_continuous", reading), ("beliefs", belief.mean),
        ("posterior_cov", belief.cov), ("controls", M @ action),
        ("control_inputs", action),
    ):
        result[key].append(np.asarray(value).tolist())
    result["scores"].append(scores)
    result["selected_policy_index"].append(best)
    result["policy_action_ids"].append(list(policy_ids[best]))
Path(sys.argv[2]).write_text(json.dumps(result), encoding="utf-8")
""",
        encoding="utf-8",
    )
    envelope = run_subprocess_envelope(
        [sys.executable, str(script), json.dumps(spec), str(receipt)],
        timeout=60,
        env=dict(os.environ),
    )
    assert envelope["success"], envelope
    result = json.loads(receipt.read_text(encoding="utf-8"))
    assert result["cpomdp_version"] == "0.5.1"
    declared = spec["initialparameterization"]["cpomdp_control"]
    expected_policies = np.asarray(
        list(itertools.product(declared["actions"], repeat=2))
    )
    np.testing.assert_array_equal(result["policies"], expected_policies)
    assert expected_policies.shape == (9, 2, 1)
    assert result["cost_per_cycle"] == 18
    assert result["step_evaluations"] == 72
    means, covs, expected_scores = _reference(spec["initialparameterization"], result)
    np.testing.assert_allclose(result["beliefs"], means, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(result["posterior_cov"], covs, rtol=1e-9, atol=1e-10)
    scores = np.asarray(result["scores"])
    np.testing.assert_allclose(scores, expected_scores, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(scores[:, :, 0], scores[:, :, 1] - scores[:, :, 2])
    ids = list(itertools.product(declared["action_ids"], repeat=2))
    for t, index in enumerate(result["selected_policy_index"]):
        assert index == int(np.argmin(expected_scores[t, :, 0]))
        assert result["policy_action_ids"][t] == list(ids[index])
        assert "forward-alias" not in ids[index]
        np.testing.assert_array_equal(
            result["control_inputs"][t], expected_policies[index, 0]
        )
    for group in ((0, 2, 6, 8), (1, 7), (3, 5)):
        for index in group[1:]:
            np.testing.assert_allclose(scores[:, group[0], :], scores[:, index, :])
    np.testing.assert_allclose(
        result["controls"],
        np.asarray(result["control_inputs"]) @ np.asarray(declared["control_matrix"]).T,
    )


@pytest.mark.needs_cpomdp
def test_native_asymmetric_controls_every_policy_and_authored_tie_order(tmp_path):
    spec = _spec()
    script = tmp_path / "model.py"
    success, message, _ = render_gnn_to_cpomdp(spec, script, {"horizon": 2})
    assert success, message
    out = tmp_path / "native"
    envelope = run_subprocess_envelope(
        [sys.executable, str(script)],
        timeout=60,
        env=dict(os.environ, CPOMDP_OUTPUT_DIR=str(out)),
    )
    assert envelope["success"], envelope
    result = json.loads((out / "simulation_results.json").read_text())
    assert result["validation"]["all_valid"]
    assert result["cpomdp_version"] == "0.5.1"
    declared = spec["initialparameterization"]["cpomdp_control"]
    assert result["finite_control_contract"] == declared
    assert result["action_set"] == declared["actions"]
    assert result["n_policies"] == 9
    assert result["action_set_version"] == "thruster-v1"
    assert result["cost_per_cycle"] == 18
    assert result["admission"]["total_step_evaluations"] == 90
    certificate = result["search_certificate"]
    assert certificate["expected"] == certificate["visited"] == 9
    assert certificate["action_set_size"] == 3 and certificate["horizon"] == 2
    assert certificate["action_set_version"] == declared["version"]
    assert certificate["warrant"]
    means, covs, scores = _reference(spec["initialparameterization"], result)
    np.testing.assert_allclose(result["beliefs"], means, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(result["posterior_cov"], covs, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        result["efe_history"], scores[:, :, 0], rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(
        result["policy_score_components"]["pragmatic"],
        scores[:, :, 1],
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result["policy_score_components"]["epistemic"],
        scores[:, :, 2],
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result["efe_history"],
        np.asarray(result["policy_score_components"]["pragmatic"])
        - np.asarray(result["policy_score_components"]["epistemic"]),
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result["controls"],
        np.array(result["control_inputs"]) @ np.array(declared["control_matrix"]).T,
    )
    policies = list(itertools.product(declared["actions"], repeat=2))
    ids = list(itertools.product(declared["action_ids"], repeat=2))
    for t, index in enumerate(result["selected_policy_index"]):
        assert index == int(np.argmin(scores[t, :, 0]))
        np.testing.assert_allclose(result["pragmatic_term"][t], scores[t, index, 1])
        np.testing.assert_allclose(result["epistemic_term"][t], scores[t, index, 2])
        assert "forward-alias" not in result["policy_action_ids"][t]
        assert result["policy_action_ids"][t] == list(ids[index])
        assert result["selected_policy"][t] == list(policies[index])
        assert result["control_inputs"][t] == policies[index][0]
    # These controls must distinguish the map, prior-time and cross-covariance semantics.
    changed = copy.deepcopy(spec["initialparameterization"])
    changed["cpomdp_control"]["control_matrix"] = [[1.0], [0.0], [0.0]]
    wrong_means, _, wrong_scores = _reference(changed, result)
    assert not np.allclose(wrong_scores, scores)
    assert not np.allclose(wrong_means, means)
    assert not np.allclose(covs[0], np.diag(np.diag(covs[0])))
    changed["cpomdp_control"] = declared
    changed["prior_mean"] = (
        np.array(changed["F"]) @ np.array(changed["prior_mean"])
    ).tolist()
    assert not np.allclose(_reference(changed, result)[0], means)


def test_source_metadata_identity_and_markdown_pipeline(tmp_path):
    from gnn.render import process_render

    spec = _spec()
    initial = spec["initialparameterization"]
    content = "## ModelName\nfinite_controls_witness\n\n## GNNSection\nContinuous\n\n## InitialParameterization\n"
    content += "\n".join(f"{k} = {v!r}" for k, v in initial.items())
    content += "\n\n## ModelParameters\nnum_timesteps = 4\nrandom_seed = 19\n"
    source = tmp_path / "input"
    source.mkdir()
    (source / "controls.md").write_text(content)
    out = tmp_path / "rendered"
    assert process_render(
        source, out, frameworks="cpomdp", backend_options={"cpomdp": {"horizon": 2}}
    )
    summary = json.loads((out / "render_processing_summary.json").read_text())
    result = next(iter(summary["file_results"].values()))["framework_results"]["cpomdp"]
    assert result["success"], result
    program = Path(result["output_files"][0]).read_text()
    import ast

    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(program).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"FINITE_CONTROLS", "SOURCE_IDENTITY"}
    }
    assert constants["FINITE_CONTROLS"] == initial["cpomdp_control"]
    direct = tmp_path / "direct.py"
    assert render_gnn_to_cpomdp(spec, direct)[0]
    old = direct.read_text().split("SOURCE_IDENTITY = ")[1].split("\n")[0]
    initial["cpomdp_control"]["action_ids"][0] = "renamed"
    assert render_gnn_to_cpomdp(spec, direct)[0]
    assert old != direct.read_text().split("SOURCE_IDENTITY = ")[1].split("\n")[0]


def test_composed_and_direct_routes_cannot_drop_control_declaration(tmp_path):
    from gnn.render.jax import render_gnn_to_jax

    spec = _spec()
    spec["model_parameters"]["nr_agents"] = 2
    assert not render_gnn_to_jax(spec, tmp_path / "jax.py")[0]
    success, message, _ = render_gnn_spec(spec, "jax", tmp_path / "composed")
    assert not success and "unsupported-cpomdp-control" in message


def test_finite_control_overflow_is_a_refusal_not_an_exception(tmp_path):
    spec = _spec()
    spec["initialparameterization"]["cpomdp_control"]["actions"][0][0] = 10**1000
    success, message, artifacts = render_gnn_to_cpomdp(spec, tmp_path / "overflow.py")
    assert not success and "finite nonboolean" in message and not artifacts


@pytest.mark.parametrize("remove_primary", [False, True])
def test_parameter_alias_cannot_hide_or_change_source_controls(
    tmp_path, remove_primary
):
    spec = _spec()
    spec["initial_parameterization"] = copy.deepcopy(spec["initialparameterization"])
    if remove_primary:
        del spec["initialparameterization"]["cpomdp_control"]
    else:
        spec["initial_parameterization"]["cpomdp_control"]["actions"][0][0] = 0.7
    success, message, artifacts = render_gnn_to_cpomdp(spec, tmp_path / "aliases.py")
    assert not success and "declarations differ" in message and not artifacts
    success, message, _ = render_gnn_spec(spec, "jax", tmp_path / "jax")
    assert not success and "unsupported-cpomdp-control" in message


@pytest.mark.parametrize("declaration", [True, False])
@pytest.mark.parametrize("composition", ["multi_agent", "hybrid", "factored"])
def test_direct_cpomdp_refuses_composed_models_before_replacing_artifact(
    tmp_path, declaration, composition
):
    spec = _spec()
    if not declaration:
        del spec["initialparameterization"]["cpomdp_control"]
    if composition == "multi_agent":
        spec["model_parameters"]["nr_agents"] = 2
    elif composition == "hybrid":
        spec["initialparameterization"]["A"] = [[1.0]]
    else:
        spec["model_parameters"]["num_factors"] = 2
        spec["initialparameterization"]["F_f0"] = spec["initialparameterization"]["F"]
    path = tmp_path / "composed.py"
    path.write_text("previous artifact")
    success, message, artifacts = render_gnn_to_cpomdp(spec, path)
    assert not success and "unsupported-composition:" in message and not artifacts
    assert path.read_text() == "previous artifact"
