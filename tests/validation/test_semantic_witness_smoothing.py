"""E4 v4: replayable finite full-sequence smoothing and its negative controls.

The counterexample fixtures provide controlled values. The native tests run a
generated bnlearn program and replay against independent latent-path enumeration.
"""

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

from gnn.cli import main
from gnn.render.bnlearn.source_program import generate_source_program
from gnn.validation.semantic_witness import (
    SCHEMA,
    SMOOTHING_CONTRACT,
    SMOOTHING_SCHEMA,
    build_semantic_witness,
    verify_semantic_witness,
)
from tests.helpers.numerical_witness import enumerate_finite, read_source
from tests.render.test_source_numerical_witnesses import authored_source, extracted_spec

OBSERVATIONS = [2, 0, 1, 2]
ACTIONS = [1, 0, 1]


def smoothing_source() -> bytes:
    return authored_source("finite", question="smoothing")


def oracle(
    question: str = "smoothing",
    *,
    actions: list[int] | None = None,
    parameters: dict | None = None,
) -> np.ndarray:
    contract = read_source(smoothing_source())
    contract = {**contract, "question": question}
    if parameters is not None:
        contract["parameters"] = parameters
    return enumerate_finite(contract, OBSERVATIONS, actions or ACTIONS)


def reported(**overrides) -> dict:
    contract = read_source(smoothing_source())
    result = {
        "framework": "pgmpy",
        "question": "smoothing",
        "dtype": "float64",
        "initial_time": contract["initial_time"],
        "axes": contract["axes"],
        "conditioning_horizon": [4, 4, 4, 4],
        "num_timesteps": 4,
        "num_states": 2,
        "num_observations": 3,
        "random_seed": 19,
        "parameters": contract["parameters"],
        "inference_contract": SMOOTHING_CONTRACT,
        "inference_mode": "exact_categorical_smoothing",
        "control_mode": "fixed_transition_actions",
        "action_semantics": "transition_into_next_state",
        "observations": OBSERVATIONS,
        "transition_actions": ACTIONS,
        "smoothed_beliefs": oracle().tolist(),
    }
    result.update(overrides)
    return result


def witness(result: dict, source: bytes | None = None) -> dict:
    return build_semantic_witness(
        smoothing_source() if source is None else source,
        json.dumps(result).encode(),
    )


def test_smoothing_receipt_matches_separate_oracle_and_replays():
    receipt = witness(reported())
    assert receipt["schema"] == SMOOTHING_SCHEMA
    assert receipt["status"] == "passed", {
        key: row
        for key, row in receipt["comparisons"].items()
        if row["status"] != "passed"
    }
    assert verify_semantic_witness(receipt) == []
    row = receipt["comparisons"]["smoothed_beliefs"]
    assert row["max_abs_error"] < 1e-14 and row["diagnosis"] == []
    assert (row["atol"], row["rtol"]) == (1e-12, 1e-10)
    assert receipt["reference"]["algorithm"] == (
        "finite_complete_path_enumeration_full_sequence"
    )
    assert receipt["assumptions"]["conditioning_horizon"] == [4, 4, 4, 4]
    assert receipt["execution_evidence"]["execution_source_verified"] is False
    assert receipt["formal_evidence"] == {
        "status": "unavailable",
        "universal_equivalence": False,
    }
    assert (
        receipt["artifacts"]["source"]["sha256"]
        == hashlib.sha256(smoothing_source()).hexdigest()
    )


def test_smoothing_and_filtering_differ_on_fixture_but_agree_at_final_time():
    smoothed, filtered = oracle(), oracle("filtering")
    assert np.abs(smoothed[:-1] - filtered[:-1]).max() > 1e-3
    np.testing.assert_allclose(smoothed[-1], filtered[-1], rtol=0, atol=1e-15)
    receipt = witness(reported())
    control = receipt["reference"]["negative_control"]
    assert control["question"] == "filtering"
    assert control["final_time_max_abs_difference"] < 1e-15
    np.testing.assert_allclose(control["beliefs"], filtered, rtol=0, atol=1e-15)


@pytest.mark.needs_bnlearn
@pytest.mark.parametrize(
    ("observations", "actions", "left_posteriors"),
    [
        (OBSERVATIONS, ACTIONS, [371 / 3305, 616 / 661, 19437 / 23135, 325 / 4627]),
        ([OBSERVATIONS[0]], [], [7 / 43]),
    ],
    ids=["four-steps", "one-step"],
)
def test_native_bnlearn_smoothing_matches_independent_witness(
    tmp_path: Path,
    observations: list[int],
    actions: list[int],
    left_posteriors: list[float],
) -> None:
    steps = len(observations)
    source = smoothing_source().replace(
        b"num_timesteps: 4", f"num_timesteps: {steps}".encode()
    )
    source_path = tmp_path / "source.md"
    contract = read_source(source)
    spec = extracted_spec(tmp_path, source)
    source_identity = {"source_sha256": hashlib.sha256(source).hexdigest()}
    script = tmp_path / "model.py"
    script.write_text(generate_source_program(spec, source_identity), encoding="utf-8")
    observations_path = tmp_path / "observations.json"
    observations_path.write_text(json.dumps(observations), encoding="utf-8")
    actions_path = tmp_path / "actions.json"
    actions_path.write_text(json.dumps(actions), encoding="utf-8")
    result_dir = tmp_path / "result"
    result_path = result_dir / "simulation_results.json"
    command = [
        sys.executable,
        str(script),
        "--inference",
        "smooth",
        "--observations-json",
        str(observations_path),
    ]
    environment = dict(os.environ, BNLEARN_OUTPUT_DIR=str(result_dir))
    if steps == 1:
        missing_actions = subprocess.run(
            command, env=environment, capture_output=True, text=True, timeout=90
        )
        assert missing_actions.returncode == 2
        assert "smoothing requires --observations-json" in missing_actions.stderr
        assert not result_path.exists()
    completed = subprocess.run(
        [*command, "--transition-actions-json", str(actions_path)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(result_path.read_bytes())
    assert result["framework"] == "bnlearn"
    assert all(result.get(name) for name in ("bnlearn_version", "pgmpy_version"))
    assert result["observations"] == observations
    assert result["transition_actions"] == actions
    assert result["source_identity"] == source_identity
    expected = enumerate_finite(contract, observations, actions)
    np.testing.assert_allclose(expected[:, 0], left_posteriors, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        result["smoothed_beliefs"], expected, rtol=1e-10, atol=1e-12
    )

    receipt_path = tmp_path / "semantic-witness-v4.json"
    assert (
        main(
            [
                "witness",
                str(source_path),
                "--results",
                str(result_path),
                "--program",
                str(script),
                "--output",
                str(receipt_path),
            ]
        )
        == 0
    )
    assert main(["witness", "--verify", str(receipt_path)]) == 0
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt["schema"] == SMOOTHING_SCHEMA
    assert receipt["status"] == "passed"
    assert receipt["assumptions"]["conditioning_horizon"] == [steps] * steps
    assert receipt["execution_evidence"]["execution_source_verified"] is False
    assert receipt["formal_evidence"]["status"] == "unavailable"
    if steps > 1:
        filtering = receipt["reference"]["negative_control"]["beliefs"]
        assert np.max(np.abs(expected[:-1] - np.asarray(filtering[:-1]))) > 0.09
        np.testing.assert_allclose(expected[-1], filtering[-1], rtol=0, atol=1e-15)

    if destination := os.environ.get("GNN_NUMERICAL_WITNESS_DIR"):
        out = Path(destination)
        out.mkdir(parents=True, exist_ok=True)
        prefix = out / f"e4-finite-smoothing-{steps}-step"
        prefix.with_suffix(".json").write_bytes(receipt_bytes)
        repository = Path(__file__).resolve().parents[2]
        custody = {
            "checkout_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository, text=True
            ).strip(),
            "checkout_dirty_paths": subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=repository,
                text=True,
            ).splitlines(),
            "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "return_code": completed.returncode,
            "observations": observations,
            "transition_actions": actions,
        }
        prefix.with_suffix(".custody.json").write_text(
            json.dumps(custody, indent=2) + "\n", encoding="utf-8"
        )


def test_filtering_values_substituted_for_smoothing_fail_with_diagnosis():
    receipt = witness(reported(smoothed_beliefs=oracle("filtering").tolist()))
    row = receipt["comparisons"]["smoothed_beliefs"]
    assert receipt["status"] == "failed"
    assert row["status"] == "failed"
    assert row["diagnosis"] == ["filtering_posteriors_substituted_for_smoothing"]
    errors = row["per_time_max_abs_error"]
    assert errors[-1] < 1e-15 and max(errors[:-1]) > 1e-3
    assert verify_semantic_witness(receipt) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"inference_contract": "bnlearn_fixed_action_filter_v1"},
        {"inference_mode": "exact_categorical_filtering"},
        {"question": "filtering"},
    ],
)
def test_filtering_declarations_refuse_the_smoothing_question(overrides):
    receipt = witness(reported(**overrides))
    assert receipt["status"] == "refused"
    assert receipt["comparisons"]["identity:question"]["status"] == "refused"
    row = receipt["comparisons"]["smoothed_beliefs"]
    assert row["status"] == "refused" and row["numeric_status"] == "passed"


def test_missing_question_and_quantity_remain_unavailable():
    result = reported()
    del result["question"], result["smoothed_beliefs"]
    receipt = witness(result)
    assert receipt["status"] == "unavailable"
    assert receipt["comparisons"]["identity:question"]["status"] == "unavailable"
    assert receipt["comparisons"]["smoothed_beliefs"]["status"] == "unavailable"
    assert receipt["comparisons"]["validity:smoothed_beliefs"]["status"] == (
        "unavailable"
    )


def test_horizon_mismatch_refuses_or_raises():
    receipt = witness(reported(conditioning_horizon=[1, 2, 3, 4]))
    assert receipt["comparisons"]["identity:conditioning_horizon"]["status"] == (
        "refused"
    )
    assert receipt["status"] == "refused"
    with pytest.raises(ValueError, match="conditioning_horizon"):
        witness(reported(observations=OBSERVATIONS[:3], transition_actions=ACTIONS[:2]))


def test_transposed_rectangular_a_is_refused_in_result_and_source():
    contract = read_source(smoothing_source())
    parameters = deepcopy(contract["parameters"])
    parameters["A"] = np.asarray(parameters["A"]).T.tolist()
    receipt = witness(reported(parameters=parameters))
    row = receipt["comparisons"]["parameter:A"]
    assert row["status"] == "refused" and row["reason"] == "quantity shape mismatch"
    assert (row["expected_shape"], row["actual_shape"]) == ([3, 2], [2, 3])
    source = smoothing_source().decode()
    original = json.dumps(contract["parameters"]["A"])
    transposed = source.replace(original, json.dumps(parameters["A"]))
    assert transposed != source
    with pytest.raises(ValueError, match="dimensions"):
        witness(reported(), transposed.encode())


def test_perturbed_transition_fails_parameter_and_quantity():
    contract = read_source(smoothing_source())
    perturbed = deepcopy(contract["parameters"])
    perturbed["B"][0][0][1] = 0.65
    perturbed["B"][1][0][1] = 0.35
    values = oracle(parameters=perturbed).tolist()
    honest = witness(reported(parameters=perturbed, smoothed_beliefs=values))
    assert honest["comparisons"]["parameter:B"]["status"] == "failed"
    assert honest["comparisons"]["smoothed_beliefs"]["numeric_status"] == "failed"
    hidden = witness(reported(smoothed_beliefs=values))
    assert hidden["comparisons"]["parameter:B"]["status"] == "passed"
    assert hidden["comparisons"]["smoothed_beliefs"]["status"] == "failed"
    assert hidden["comparisons"]["smoothed_beliefs"]["diagnosis"] == []


@pytest.mark.parametrize(
    "shifted,name",
    [
        ([0, 1, 1], "action_timing_shifted_earlier"),
        ([1, 1, 0], "action_timing_shifted_later"),
    ],
)
def test_shifted_exogenous_action_timing_is_diagnosed(shifted, name):
    values = oracle(actions=shifted)
    assert np.abs(values - oracle()).max() > 1e-6
    receipt = witness(reported(smoothed_beliefs=values.tolist()))
    row = receipt["comparisons"]["smoothed_beliefs"]
    assert row["status"] == "failed"
    assert name in row["diagnosis"]


def test_float32_precision_mismatch_refuses_and_is_diagnosed():
    rounded = oracle().astype(np.float32).astype(np.float64).tolist()
    receipt = witness(reported(dtype="float32", smoothed_beliefs=rounded))
    assert receipt["comparisons"]["identity:dtype"]["status"] == "refused"
    row = receipt["comparisons"]["smoothed_beliefs"]
    assert row["status"] == "refused" and row["numeric_status"] == "failed"
    assert row["diagnosis"] == ["float32_precision_rounding"]
    silent = witness(reported(smoothed_beliefs=rounded))
    # float32 rows lose unit mass beyond 1e-12 before any numerical comparison.
    assert silent["comparisons"]["validity:smoothed_beliefs"]["status"] == "refused"
    row = silent["comparisons"]["smoothed_beliefs"]
    assert row["numeric_status"] == "failed"
    assert row["diagnosis"] == ["float32_precision_rounding"]


def test_source_digest_tampering_refuses_and_fails_replay():
    receipt = witness(reported(source_sha256="0" * 64))
    assert receipt["comparisons"]["identity:source_sha256"]["status"] == "refused"
    passed = witness(reported())
    tampered = deepcopy(passed)
    tampered["artifacts"]["source"]["sha256"] = "f" * 64
    assert verify_semantic_witness(tampered) == [
        "semantic witness differs from independent raw-artifact replay"
    ]
    edited = deepcopy(passed)
    edited["reference"]["quantities"]["smoothed_beliefs"][0][0] += 1e-6
    assert verify_semantic_witness(edited) != []
    relabeled = deepcopy(passed)
    relabeled["schema"] = SCHEMA
    assert verify_semantic_witness(relabeled) != []


def test_framework_outside_smoothing_contract_refuses():
    receipt = witness(reported(framework="jax"))
    assert receipt["comparisons"]["identity:framework"]["status"] == "refused"


def test_retained_v2_smoothing_refusals_remain_replay_valid():
    result = reported()
    legacy = build_semantic_witness(
        smoothing_source(), json.dumps(result).encode(), schema=SCHEMA
    )
    assert legacy["schema"] == SCHEMA and legacy["status"] == "refused"
    assert legacy["comparisons"]["identity:question"]["status"] == "refused"
    assert verify_semantic_witness(legacy) == []
    with pytest.raises(ValueError, match="unsupported semantic witness schema"):
        build_semantic_witness(
            smoothing_source(), json.dumps(result).encode(), schema="v9"
        )


def test_cli_writes_and_verifies_smoothing_receipt(tmp_path, capsys):
    model, native, output = (
        tmp_path / name for name in ("model.md", "result.json", "receipt.json")
    )
    model.write_bytes(smoothing_source())
    native.write_text(json.dumps(reported()))
    arguments = ["witness", str(model), "--results", str(native)]
    assert main([*arguments, "--output", str(output), "--json"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["data"]["schema"] == SMOOTHING_SCHEMA
    assert envelope["data"]["status"] == "passed"
    assert main(["witness", "--verify", str(output), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "success"
    native.write_text(
        json.dumps(reported(smoothed_beliefs=oracle("filtering").tolist()))
    )
    failed = tmp_path / "failed.json"
    main([*arguments, "--output", str(failed), "--json"])
    capsys.readouterr()
    assert main(["witness", "--verify", str(failed), "--json"]) == 1
