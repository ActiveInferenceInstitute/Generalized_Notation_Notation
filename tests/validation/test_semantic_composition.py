"""Independent composition witnesses use joint conditioning vs recursive filtering."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from gnn.cli import main as cli_main
from gnn.validation.semantic_witness import (
    build_semantic_witness,
    verify_semantic_witness,
)
from gnn.validation.witness_composition_source import composition_equations
from gnn.validation.witness_source import read_source

AGENTS = ("agent1", "agent2")
BLOCKS = {
    "agent1": {
        "F": [[0.85]],
        "H": [[1.25], [-0.4]],
        "Q": [[0.04]],
        "R": [[0.12, 0.015], [0.015, 0.2]],
        "prior_mean": [-1.0],
        "prior_cov": [[0.3]],
    },
    "agent2": {
        "F": [[0.9, 0.15], [0.0, 0.8]],
        "H": [[0.75, -1.1]],
        "Q": [[0.03, 0.005], [0.005, 0.06]],
        "R": [[0.2]],
        "prior_mean": [2.0, -0.5],
        "prior_cov": [[0.4, 0.08], [0.08, 0.5]],
    },
}
OBSERVATIONS = {
    "agent1": [[0.3, -0.4], [0.1, 0.2], [-0.2, 0.7], [0.4, -0.1]],
    "agent2": [[0.2], [-0.1], [0.8], [-0.7]],
}


def authored_composition(blocks=None) -> bytes:
    blocks = deepcopy(BLOCKS if blocks is None else blocks)
    states = {
        agent: [f"x{index}" for index in range(len(blocks[agent]["F"]))]
        for agent in AGENTS
    }
    observations = {
        agent: [f"y{index}" for index in range(len(blocks[agent]["H"]))]
        for agent in AGENTS
    }
    declaration = {
        "schema": "gnn-independent-gaussian-composition-v1",
        "model": "gaussian_product",
        "question": "filtering",
        "dtype": "float64",
        "initial_time": "first_observation_uses_prior",
        "axes": {"F": ["next_state", "previous_state"], "H": ["observation", "state"]},
        "agents": list(AGENTS),
        "state_ids": states,
        "observation_ids": observations,
    }
    state_text = "\n".join(
        f"x_{agent}[{len(states[agent])},1,type=float]\ny_{agent}[{len(observations[agent])},1,type=float]"
        for agent in AGENTS
    )
    parameters = "\n".join(
        f"{key}_{agent} = {json.dumps(value)}"
        for agent in AGENTS
        for key, value in blocks[agent].items()
    )
    sections = {
        "ModelName": "Asymmetric independent Gaussian witness",
        "GNNSection": "ActInfContinuousMultiAgent",
        "StateSpaceBlock": state_text,
        "InitialParameterization": parameters,
        "ModelParameters": "nr_agents: 2\nagent_coupling: independent\nnum_timesteps: 4\nrandom_seed_agent1: 17\nrandom_seed_agent2: 93",
        "Equations": composition_equations(),
        "NumericalWitness": json.dumps(declaration),
    }
    return "\n\n".join(f"## {key}\n{value}" for key, value in sections.items()).encode()


def recursive_oracle(parameters, observations):
    """Separate recursive Kalman/Joseph oracle; never imports witness references."""
    p = {key: np.asarray(value, dtype=np.float64) for key, value in parameters.items()}
    f, h, q, r = (p[key] for key in ("F", "H", "Q", "R"))
    mean, covariance = p["prior_mean"].copy(), p["prior_cov"].copy()
    means, covariances = [], []
    for index, observation in enumerate(observations):
        if index:
            mean, covariance = f @ mean, f @ covariance @ f.T + q
        gain = np.linalg.solve(h @ covariance @ h.T + r, h @ covariance).T
        mean = mean + gain @ (np.asarray(observation) - h @ mean)
        residual = np.eye(len(mean)) - gain @ h
        covariance = residual @ covariance @ residual.T + gain @ r @ gain.T
        means.append(mean.tolist())
        covariances.append(covariance.tolist())
    return means, covariances


def reported_result(
    blocks=None, observations=None, program=b"reported fixture program"
):
    blocks = deepcopy(BLOCKS if blocks is None else blocks)
    observations = deepcopy(OBSERVATIONS if observations is None else observations)
    children = {}
    for index, agent in enumerate(AGENTS):
        means, covariance = recursive_oracle(blocks[agent], observations[agent])
        children[agent] = {
            "schema_version": "gnn_lgssm_filter_v1",
            "model_kind": "continuous",
            "framework": "jax",
            "dtype": "float64",
            "inference_contract": "gnn_lgssm_filter_v1",
            "inference_mode": "gaussian_filtering",
            "initial_time": "first_observation_uses_prior",
            "conditioning_horizon": [1, 2, 3, 4],
            "axes": {
                "F": ["next_state", "previous_state"],
                "H": ["observation", "state"],
            },
            "num_timesteps": 4,
            "num_states": len(blocks[agent]["F"]),
            "num_observations": len(blocks[agent]["H"]),
            "random_seed": (17, 93)[index],
            "control_mode": "passive",
            "parameters": blocks[agent],
            "beliefs": means,
            "posterior_cov": covariance,
            "observations_continuous": observations[agent],
            "controls": np.zeros((4, len(blocks[agent]["F"]))).tolist(),
            "generated_script_sha256": hashlib.sha256(program).hexdigest(),
            "validation": {"all_valid": True, "means_finite": True},
        }
    result = {
        "schema_version": "gnn_multi_agent_continuous_v1",
        "model_kind": "multi_agent_continuous",
        "framework": "jax",
        "agents": list(AGENTS),
        "agent_coupling": "independent",
        "num_timesteps": 4,
        "agent_results": children,
        "validation": {"all_valid": True},
    }
    for key in ("beliefs", "posterior_cov", "controls", "observations_continuous"):
        result[key + "_by_agent"] = {
            agent: deepcopy(children[agent][key]) for agent in AGENTS
        }
    return result


def build(source=None, result=None):
    return build_semantic_witness(
        authored_composition() if source is None else source,
        json.dumps(reported_result() if result is None else result).encode(),
        b"reported fixture program",
    )


def test_joint_conditioning_matches_separate_recursive_oracle_and_replays():
    receipt = build()
    assert receipt["schema"] == "gnn-semantic-witness-v3"
    assert receipt["status"] == "passed", receipt["comparisons"]
    assert verify_semantic_witness(receipt) == []
    assert receipt["formal_evidence"] == {
        "status": "unavailable",
        "universal_equivalence": False,
    }
    assert receipt["execution_evidence"]["execution_source_verified"] is False
    covariance = np.asarray(receipt["reference"]["quantities"]["posterior_cov"])
    np.testing.assert_array_equal(covariance[:, :1, 1:], 0)
    assert np.any(covariance[:, 1, 2] != 0)
    for agent in AGENTS:
        assert receipt["comparisons"][agent + ":beliefs"]["max_abs_error"] < 1e-12
        assert receipt["comparisons"][agent + ":posterior_cov"]["max_abs_error"] < 1e-12


@pytest.mark.parametrize(
    "target", ["reference", "source", "result", "candidate", "verdict"]
)
def test_raw_replay_does_not_trust_edited_arrays_or_verdict(target):
    receipt = build()
    if target == "reference":
        receipt["reference"]["components"]["agent1"]["quantities"]["beliefs"][0][0] += (
            0.1
        )
    elif target in {"source", "result"}:
        receipt["artifacts"][target]["utf8"] += " "
    elif target == "candidate":
        receipt["candidate"]["components"]["agent2"]["beliefs"][0][0] += 0.1
    else:
        receipt["status"] = "failed"
    assert verify_semantic_witness(receipt)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "float32"),
        ("initial_time", "predict_before_first_observation"),
        ("conditioning_horizon", [4, 4, 4, 4]),
        ("random_seed", True),
        ("control_mode", "closed_loop_proportional"),
        ("framework", "rxinfer"),
        ("inference_mode", "gaussian_smoothing"),
        ("generated_script_sha256", "0" * 64),
        (
            "axes",
            {"F": ["previous_state", "next_state"], "H": ["observation", "state"]},
        ),
        ("num_states", 2.0),
    ],
)
def test_child_semantic_mismatch_cannot_borrow_matching_numbers(field, value):
    result = reported_result()
    result["agent_results"]["agent2"][field] = value
    receipt = build(result=result)
    assert receipt["status"] == "refused"
    assert receipt["comparisons"]["agent2:beliefs"]["status"] == "refused"
    assert receipt["comparisons"]["agent2:beliefs"]["numeric_status"] == "passed"


@pytest.mark.parametrize(
    "field", ["dtype", "parameters", "random_seed", "conditioning_horizon"]
)
def test_missing_child_metadata_keeps_agreement_unavailable(field):
    result = reported_result()
    del result["agent_results"]["agent2"][field]
    assert build(result=result)["status"] == "unavailable"


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "projection",
        "observations",
        "control",
        "validation",
        "covariance",
        "parameter",
    ],
)
def test_child_records_projections_and_parameters_are_bound(change):
    result = reported_result()
    if change == "missing":
        del result["agent_results"]["agent2"]
    elif change == "extra":
        result["agent_results"]["agent3"] = deepcopy(result["agent_results"]["agent2"])
    elif change == "projection":
        result["beliefs_by_agent"]["agent2"][0][0] += 0.2
    elif change == "observations":
        result["agent_results"]["agent2"]["observations_continuous"][0][0] += 0.2
    elif change == "control":
        result["agent_results"]["agent2"]["controls"][0][0] = 0.1
        result["controls_by_agent"]["agent2"][0][0] = 0.1
    elif change == "validation":
        result["agent_results"]["agent2"]["validation"]["means_finite"] = False
    elif change == "covariance":
        result["agent_results"]["agent2"]["posterior_cov"][0][0][1] += 0.1
        result["posterior_cov_by_agent"]["agent2"][0][0][1] += 0.1
    else:
        result["agent_results"]["agent2"]["parameters"]["F"][0][1] += 0.1
    receipt = build(result=result)
    assert receipt["status"] in {"refused", "failed", "unavailable"}
    assert receipt["status"] != "passed"


@pytest.mark.parametrize(
    "mutation",
    [
        "coupled",
        "controls",
        "duplicate",
        "missing",
        "extra",
        "smoothing",
        "nonfinite",
        "boolean",
        "covariance",
        "dimensions",
        "time",
        "seed",
        "unknown_heading",
        "equations",
        "shared",
    ],
)
def test_source_composition_admission_rejects_ambiguity_and_unsupported_contracts(
    mutation,
):
    source = authored_composition()
    replacements = {
        "coupled": (b"agent_coupling: independent", b"agent_coupling: coupled"),
        "controls": (b"F_agent1 = ", b"goal_mean_agent1 = [1.0]\nF_agent1 = "),
        "duplicate": (b"F_agent1 = ", b"F_agent1 = [[0.85]]\nF_agent1 = "),
        "missing": (b"Q_agent1 = [[0.04]]\n", b""),
        "extra": (b"F_agent1 = ", b"F_agent3 = [[1.0]]\nF_agent1 = "),
        "smoothing": (b'"question": "filtering"', b'"question": "smoothing"'),
        "nonfinite": (b"F_agent1 = [[0.85]]", b"F_agent1 = [[NaN]]"),
        "boolean": (b"F_agent1 = [[0.85]]", b"F_agent1 = [[true]]"),
        "covariance": (b"Q_agent1 = [[0.04]]", b"Q_agent1 = [[-0.04]]"),
        "dimensions": (b"x_agent1[1,1,type=float]", b"x_agent1[2,1,type=float]"),
        "time": (b"num_timesteps: 4", b"num_timesteps: 9"),
        "seed": (b"random_seed_agent1: 17", b"random_seed_agent1: true"),
        "unknown_heading": (
            b"## Equations",
            b"## Coupling\nF_cross = [[0.1]]\n\n## Equations",
        ),
        "equations": (b"F_agent1*x_agent1_t", b"F_agent1*x_agent2_t"),
        "shared": (b"F_agent1 = ", b"F = [[1.0]]\nF_agent1 = "),
    }
    original, changed = replacements[mutation]
    assert original in source
    with pytest.raises(ValueError):
        read_source(source.replace(original, changed))


def test_future_observations_do_not_leak_into_earlier_component_filter():
    first = build()
    changed_observations = deepcopy(OBSERVATIONS)
    changed_observations["agent2"][-1][0] += 30
    second = build(result=reported_result(observations=changed_observations))
    assert second["status"] == "passed"
    left, right = (
        np.asarray(item["reference"]["quantities"]["beliefs"])
        for item in (first, second)
    )
    np.testing.assert_array_equal(left[:-1], right[:-1])
    np.testing.assert_array_equal(left[:, :1], right[:, :1])
    assert not np.allclose(left[-1, 1:], right[-1, 1:])


def test_agent_permutation_reindexes_whole_component_without_changing_its_filter():
    first = build()
    swapped_blocks = {
        "agent1": deepcopy(BLOCKS["agent2"]),
        "agent2": deepcopy(BLOCKS["agent1"]),
    }
    swapped_observations = {
        "agent1": deepcopy(OBSERVATIONS["agent2"]),
        "agent2": deepcopy(OBSERVATIONS["agent1"]),
    }
    second = build(
        source=authored_composition(swapped_blocks),
        result=reported_result(swapped_blocks, swapped_observations),
    )
    assert second["status"] == "passed"
    for original, permuted in zip(AGENTS, reversed(AGENTS), strict=True):
        np.testing.assert_allclose(
            first["reference"]["components"][original]["quantities"]["beliefs"],
            second["reference"]["components"][permuted]["quantities"]["beliefs"],
            atol=1e-14,
            rtol=1e-14,
        )


def test_cli_builds_and_replays_composition_without_rewriting(tmp_path, capsys):
    source, result, program, receipt = (
        tmp_path / name
        for name in ("source.md", "native.json", "program.py", "witness.json")
    )
    source.write_bytes(authored_composition())
    result.write_text(json.dumps(reported_result()))
    program.write_bytes(b"reported fixture program")
    assert (
        cli_main(
            [
                "witness",
                str(source),
                "--results",
                str(result),
                "--program",
                str(program),
                "--output",
                str(receipt),
                "--json",
            ]
        )
        == 0
    )
    before = receipt.read_bytes()
    assert cli_main(["witness", "--verify", str(receipt), "--json"]) == 0
    assert receipt.read_bytes() == before
    assert "passed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failed"),
        ("success", False),
        ("source_sha256", "0" * 64),
        ("source_identity", {"source_sha256": "0" * 64}),
        ("inference_mode", "gaussian_smoothing"),
        ("control_mode", "coupled_control"),
        ("dtype", "float32"),
        ("validation", {"all_valid": True, "per_agent": {"agent1": False}}),
    ],
)
def test_explicit_outer_contradictions_refuse_matching_child_quantities(field, value):
    result = reported_result()
    result[field] = value
    receipt = build(result=result)
    assert receipt["status"] == "refused"
    assert receipt["comparisons"]["agent1:beliefs"]["status"] == "refused"


@pytest.mark.parametrize("target", ["parameters", "projection"])
def test_extra_native_inventory_cannot_hide_an_unmodelled_agent_or_control(target):
    result = reported_result()
    if target == "parameters":
        result["agent_results"]["agent2"]["parameters"]["goal_mean"] = [0.0, 0.0]
    else:
        result["beliefs_by_agent"]["agent3"] = [[0.0]]
    assert build(result=result)["status"] == "refused"


def test_returned_contract_mutation_cannot_change_later_source_admission():
    source = authored_composition()
    changed = read_source(source)
    changed["components"]["agent1"]["axes"]["F"].reverse()
    fresh = read_source(source)
    assert fresh["components"]["agent1"]["axes"]["F"] == [
        "next_state",
        "previous_state",
    ]
    assert build()["status"] == "passed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "float32"),
        ("agents", ["agent2", "agent1"]),
        ("state_ids", {"agent1": [], "agent2": ["x0", "x1"]}),
        ("state_ids", {"agent1": ["x0"], "agent2": ["x0", "x0"]}),
        ("observation_ids", {"agent1": ["x0"]}),
        ("state_ids", {"agent1": ["x0", "x1", "x2"], "agent2": ["x0", "x1"]}),
    ],
)
def test_authored_agent_axes_and_precision_are_explicit(field, value):
    source = authored_composition().decode()
    prefix, declaration = source.split("## NumericalWitness\n")
    metadata = json.loads(declaration)
    metadata[field] = value
    with pytest.raises(ValueError):
        read_source((prefix + "## NumericalWitness\n" + json.dumps(metadata)).encode())


@pytest.mark.parametrize(
    "replacement",
    [
        ("nr_agents: 2", "nr_agents: true"),
        ("random_seed_agent1: 17", "random_seed_agent1: 4294967296"),
        ("num_timesteps: 4", "num_timesteps: true"),
        (
            "Q_agent2 = [[0.03, 0.005], [0.005, 0.06]]",
            "Q_agent2 = [[0.03, 0.01], [0.005, 0.06]]",
        ),
        ("R_agent2 = [[0.2]]", "R_agent2 = [[0.0]]"),
        ("F_agent2 = [[0.9, 0.15], [0.0, 0.8]]", "F_agent2 = [[0.9], [0.0, 0.8]]"),
    ],
)
def test_integer_budgets_and_covariance_contracts_refuse_invalid_sources(replacement):
    source = authored_composition()
    original, changed = (value.encode() for value in replacement)
    assert original in source
    with pytest.raises(ValueError):
        read_source(source.replace(original, changed))


@pytest.mark.parametrize(
    "mutation", ["missing_agent", "horizon", "boolean", "nonfinite"]
)
def test_reference_observation_identity_and_numerics_are_required(mutation):
    result = reported_result()
    if mutation == "missing_agent":
        del result["observations_continuous_by_agent"]["agent2"]
    elif mutation == "horizon":
        result["observations_continuous_by_agent"]["agent2"].pop()
    elif mutation == "boolean":
        result["observations_continuous_by_agent"]["agent2"][0][0] = True
    else:
        result["observations_continuous_by_agent"]["agent2"][0][0] = float("nan")
    with pytest.raises(ValueError):
        build(result=result)


def test_optional_program_and_missing_native_children_keep_evidence_planes_explicit():
    receipt = build_semantic_witness(
        authored_composition(), json.dumps(reported_result()).encode()
    )
    assert receipt["status"] == "passed"
    assert receipt["artifacts"]["program"] is None
    assert verify_semantic_witness(receipt) == []
    assert receipt["execution_evidence"]["execution_source_verified"] is False
    result = reported_result()
    del result["agent_results"]
    assert build(result=result)["status"] == "unavailable"
    with pytest.raises(ValueError, match="JSON object"):
        build_semantic_witness(authored_composition(), b"[]")


@pytest.mark.env_heavy
def test_native_jax_composition_retains_children_and_matches_independent_reference(
    tmp_path,
):
    from gnn.extract.pomdp_extractor import extract_pomdp_from_file
    from gnn.render.pomdp_processor import pomdp_to_gnn_spec
    from gnn.render.processor import render_gnn_spec

    source = tmp_path / "source.md"
    source.write_bytes(authored_composition())
    space = extract_pomdp_from_file(source, strict_validation=True)
    assert space is not None
    spec = pomdp_to_gnn_spec(space)
    success, message, artifacts = render_gnn_spec(spec, "jax", tmp_path / "rendered")
    assert success, message
    program = Path(artifacts[0])
    output = tmp_path / "results"
    child = subprocess.run(
        [sys.executable, str(program)],
        cwd=tmp_path,
        env={**os.environ, "GNN_OUTPUT_DIR": str(output)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stderr[-4000:]
    result = (output / "simulation_results.json").read_bytes()
    receipt = build_semantic_witness(source.read_bytes(), result, program.read_bytes())
    assert receipt["status"] == "passed", receipt["comparisons"]
    assert verify_semantic_witness(receipt) == []
    destination = os.environ.get("GNN_NUMERICAL_WITNESS_DIR")
    if destination:
        retained = Path(destination)
        retained.mkdir(parents=True, exist_ok=True)
        (retained / "gaussian-product-semantic-witness-v3.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        )
    for agent in AGENTS:
        assert receipt["candidate"]["components"][agent]["parameters"] == BLOCKS[agent]
        assert receipt["candidate"]["components"][agent]["dtype"] == "float64"


def test_documented_source_is_an_admitted_asymmetric_native_fixture():
    guide = (
        Path(__file__).resolve().parents[2]
        / "docs/development/e4_independent_gaussian_composition.md"
    )
    text = guide.read_text()
    source = text.split("```markdown\n", 1)[1].split("\n```", 1)[0].encode()
    contract = read_source(source)
    assert contract["parameters"] == read_source(authored_composition())["parameters"]
    assert contract["components"]["agent1"]["state_ids"] == ["x0"]
    assert contract["components"]["agent2"]["state_ids"] == ["x0", "x1"]
