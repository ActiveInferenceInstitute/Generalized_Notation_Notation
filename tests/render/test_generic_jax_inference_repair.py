"""Native generic JAX inference: source semantics and independent numerics."""

import importlib.util
import json
import os
import subprocess
import sys
from fractions import Fraction
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gnn.extract.pomdp_extractor import extract_pomdp_from_file
from gnn.parsers.json_serializer import JSONSerializer
from gnn.parsers.markdown_parser import MarkdownGNNParser
from gnn.render.jax import render_gnn_to_jax
from gnn.render.jax.jax_model_generator import _generate_jax_model_code
from gnn.render.pomdp_processor import POMDPRenderProcessor, pomdp_to_gnn_spec

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("basics/static_perception", "basics/dynamic_perception", "discrete/hmm_baseline")


def source_spec(source):
    path = ROOT / "input/gnn_files" / f"{source}.md"
    parsed = MarkdownGNNParser().parse_file(str(path))
    assert parsed.success
    typed = json.loads(JSONSerializer().serialize(parsed.model))
    space = extract_pomdp_from_file(path, on_error="raise")
    spec = pomdp_to_gnn_spec(space)
    return spec, typed, space


def module_for(tmp_path, spec):
    path = tmp_path / "generated.py"
    success, message, _ = render_gnn_to_jax(spec, path)
    assert success, message
    loader = importlib.util.spec_from_file_location("generated_generic", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


@pytest.mark.parametrize("x64", [False, True])
def test_conditioning_and_transition_against_alternate_formulation(tmp_path, x64):
    spec, _, _ = source_spec(SOURCES[0])
    module = module_for(tmp_path, spec)
    with jax.enable_x64(x64):
        params = module.create_params()
        eps = np.finfo(np.float64 if x64 else np.float32).eps
        # Exact rational Bayes for both observations, independently of JAX kernels.
        expected = np.array([[float(Fraction(9, 11)), float(Fraction(2, 11))],
                             [float(Fraction(1, 9)), float(Fraction(8, 9))]])
        actual = np.asarray(module.batched_belief_update(
            params, jnp.stack([params["D_vector"]] * 2), jnp.eye(2)))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2 * eps)
        np.testing.assert_allclose(actual.sum(axis=1), 1, rtol=0, atol=eps)
        # Scale invariance distinguishes exact mass division from epsilon bias.
        scaled = np.asarray(module.belief_update(params, params["D_vector"] * 1e-12, jnp.array([1., 0.])))
        np.testing.assert_allclose(scaled, expected[0], rtol=0, atol=2 * eps)
        for action in (0, 1):
            q = jnp.array([.25, .75])
            actual = np.asarray(module.state_transition(params, q * 1e-12, action))
            reference = np.array([sum(float(params["B_matrix"][i, j, action]) * float(q[j])
                                      for j in range(2)) for i in range(2)])
            np.testing.assert_allclose(actual, reference, rtol=0, atol=2 * eps)
            assert abs(actual.sum() - 1) <= eps


@pytest.mark.parametrize("bad", [[0., 0.], [np.nan, .5], [np.inf, .5], [-.1, 1.1]])
@pytest.mark.parametrize("operation", ["belief_update", "state_transition"])
def test_invalid_mass_fails_explicitly(tmp_path, bad, operation):
    spec, _, _ = source_spec(SOURCES[0])
    module = module_for(tmp_path, spec)
    params = module.create_params()
    operand = jnp.array([1., 0.]) if operation == "belief_update" else 0
    with pytest.raises(Exception, match="positive finite mass"):
        getattr(module, operation)(params, jnp.array(bad), operand).block_until_ready()


@pytest.mark.parametrize("source", SOURCES + (
    "hierarchical/hierarchical_pomdp", "hierarchical/temporal_hierarchy", "discrete/tmaze_epistemic",
))
def test_source_metadata_preservation(source):
    spec, typed, space = source_spec(source)
    content = (ROOT / "input/gnn_files" / f"{source}.md").read_text()
    for key, heading in (("equations", "Equations"), ("time", "Time")):
        expected = content.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0].strip()
        assert spec[key] == space.to_dict()[key] == expected
    assert typed["time_specification"]["time_type"] in spec["time"].splitlines()
    assert spec["model_parameters"]["passive_model"] == space.passive_model
    assert space.passive_model == (source in SOURCES[1:])
    # Preserve metadata on the raw component path too, without changing tables.
    raw = pomdp_to_gnn_spec(space, preserve_discrete_structure=True)
    assert raw["time"] == spec["time"]
    assert raw["equations"] == spec["equations"]
    assert raw["initialparameterization"] == space.initial_parameterization
    assert raw["model_parameters"]["passive_model"] == space.passive_model


def test_static_conditioning_has_no_rollout_or_policy(tmp_path):
    spec, _, _ = source_spec(SOURCES[0])
    module = module_for(tmp_path, spec)
    params = module.create_params()
    result = module.simulate_step(params, params["D_vector"], jnp.array([1., 0.]))
    assert result["action"] is None and result["expected_free_energy"] is None
    np.testing.assert_array_equal(result["belief"], result["predicted_next_state"])
    with pytest.raises(ValueError, match="Time Static"):
        module.run_simulation(params, 5)


def test_passive_multiaction_declaration_rejected():
    spec, _, _ = source_spec(SOURCES[0])
    spec.pop("time")
    spec["model_parameters"]["passive_model"] = True
    with pytest.raises(ValueError, match="action-independent"):
        _generate_jax_model_code(spec, None)


@pytest.mark.parametrize("corruption", ["mass", "negative", "final", "action"])
def test_receipt_validation_is_computed(tmp_path, corruption):
    spec, _, _ = source_spec(SOURCES[1])
    module = module_for(tmp_path, spec)
    params = module.create_params()
    trajectory = module.run_simulation(params, 2)
    if corruption == "mass":
        trajectory["beliefs"] = jnp.array([[.1, .1], [.5, .5]])
    elif corruption == "negative":
        trajectory["beliefs"] = jnp.array([[-.1, 1.1], [.5, .5]])
    elif corruption == "final":
        trajectory["final_belief"] = jnp.array([.1, .1])
    else:
        trajectory["actions"] = jnp.array([0])
    receipt = json.loads(Path(module.save_simulation_results(trajectory, params, "test", str(tmp_path))).read_text())
    assert not receipt["success"]
    if corruption in ("mass", "final"):
        assert not receipt["validation"]["beliefs_sum_to_one"]
    if corruption == "negative":
        assert not receipt["validation"]["all_beliefs_valid"]
    if corruption == "action":
        assert not receipt["validation"]["actions_in_range"]


def enumerated_filter(a, b, d, observations):
    """Enumerate joint paths, then marginalize the last state for every prefix."""
    result = []
    for length in range(1, len(observations) + 1):
        masses = np.zeros(len(d))
        for states in product(range(len(d)), repeat=length):
            mass = d[states[0]]
            for t, state in enumerate(states):
                mass *= a[observations[t], state]
                if t:
                    mass *= b[state, states[t - 1]]
            masses[states[-1]] += mass
        result.append(masses / masses.sum())
    return np.array(result)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("source", SOURCES)
def test_native_generated_subprocess_matches_source_joint_enumeration(tmp_path, source, x64):
    _, typed, space = source_spec(source)
    # Actual registered source route; four observations bound path enumeration.
    rendered = POMDPRenderProcessor(tmp_path).process_pomdp_for_all_frameworks(
        space, frameworks=["jax"], timesteps=4
    )
    result = rendered["framework_results"]["jax"]
    assert result["success"], result
    script = Path(result["output_files"][0])
    env = {**os.environ, "JAX_ENABLE_X64": "1" if x64 else "0", "GNN_OUTPUT_DIR": str(tmp_path)}
    completed = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env,
                               text=True, capture_output=True, timeout=60)
    (tmp_path / "stdout.txt").write_text(completed.stdout)
    (tmp_path / "stderr.txt").write_text(completed.stderr)
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads((tmp_path / "simulation_results.json").read_text())
    source_params = {p["name"]: p["value"] for p in typed["parameters"]}
    a = np.asarray(source_params["A"])
    b = np.asarray(source_params["B"])
    if b.ndim == 3:
        b = b[:, :, 0]  # Never used by the single-observation static reference.
    d = np.asarray(source_params["D"]).flatten()
    expected = enumerated_filter(a, b, d, receipt["observations"])
    eps = np.finfo(np.float64 if x64 else np.float32).eps
    np.testing.assert_allclose(receipt["beliefs"], expected, rtol=0, atol=4 * eps)
    np.testing.assert_allclose(np.sum(receipt["beliefs"], axis=1), 1, rtol=0, atol=2 * eps)
    np.testing.assert_array_equal(receipt["final_belief"], receipt["beliefs"][-1])
    assert receipt["success"] and receipt["validation"]["beliefs_sum_to_one"]
    assert receipt["actions"] == receipt["simulation_trace"]["efe_history"] == []
    assert receipt["metrics"]["average_efe"] is None
    assert receipt["metrics"]["expected_free_energy_convention"] is None
    assert receipt["control_mode"] == "none"
    assert receipt["inference_estimand"] == ("static_conditioning" if source == SOURCES[0] else "filtering")
    assert receipt["num_timesteps"] == (1 if source == SOURCES[0] else 4)
    if source != SOURCES[0]:
        assert "no backward smoothing" in receipt["inference_description"]
    (tmp_path / "independent-comparison.json").write_text(json.dumps({
        "max_abs_error": float(np.max(np.abs(np.asarray(receipt["beliefs"]) - expected))),
        "max_mass_error": float(np.max(np.abs(np.sum(receipt["beliefs"], axis=1) - 1))),
        "x64": x64, "reference": "joint latent-path enumeration from typed source parameters",
        "metadata_boundary": "actual extractor fields; no manual metadata bridge",
    }, indent=2))


@pytest.mark.parametrize("declaration,static", [
    ("Static # independent observation\nDiscreteTime", True),
    ("# Static is only a comment\nDynamic", False),
    ("", False),
])
def test_time_declarations_do_not_depend_on_names(tmp_path, declaration, static):
    spec, _, _ = source_spec(SOURCES[0])
    spec["time"] = declaration
    spec["name"] = spec["model_name"] = "dynamic_perception" if static else "static_perception"
    assert module_for(tmp_path, spec).STATIC_MODEL is static


def test_conflicting_time_declarations_fail():
    spec, _, _ = source_spec(SOURCES[0])
    spec["time"] = "Static\nDynamic"
    with pytest.raises(ValueError, match="both Static and Dynamic"):
        _generate_jax_model_code(spec, None)
