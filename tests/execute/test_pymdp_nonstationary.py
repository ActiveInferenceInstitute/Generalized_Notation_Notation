#!/usr/bin/env python3
"""Nonstationary pymdp rollout: regime-switched and time-varying B (GEN-4).

Exercises the per-step rebuild semantics end to end: a ``B_regime`` tensor
with ``b_regime_schedule`` (and a ``B_t`` phase tensor) must produce a
successful rollout whose results record the applied schedule, the distinct
transitions used, and hold-last truncation beyond the declared span.

Zero-skip contract: pymdp/JAX presence is gated by the sanctioned dynamic
``needs_pymdp`` probe marker, never by ``pytest.skip``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gnn.execute.pymdp import run_pymdp_simulation

pytestmark = [pytest.mark.needs_pymdp]

_A = [
    [0.85, 0.10, 0.05],
    [0.10, 0.80, 0.10],
    [0.05, 0.10, 0.85],
]
_C = [0.0, 0.0, 1.0]
_D = [0.34, 0.33, 0.33]

# (next, prev, action) slices — one per regime / phase (exemplar values).
_SLICE_CALM = [
    [[0.7, 0.1], [0.2, 0.1], [0.1, 0.1]],
    [[0.2, 0.1], [0.7, 0.1], [0.1, 0.1]],
    [[0.1, 0.8], [0.1, 0.8], [0.8, 0.8]],
]
_SLICE_STORM = [
    [[0.2, 0.4], [0.4, 0.2], [0.3, 0.1]],
    [[0.6, 0.2], [0.2, 0.4], [0.3, 0.1]],
    [[0.2, 0.4], [0.4, 0.4], [0.4, 0.8]],
]


def _base_params() -> dict:
    return {
        "num_hidden_states": 3,
        "num_obs": 3,
        "num_actions": 2,
        "batch_size": 1,
        "policy_len": 1,
        "random_seed": 7,
    }


def _regime_spec(schedule: list[int] | None = None, horizon: int = 6) -> dict:
    model_parameters = _base_params()
    model_parameters["num_timesteps"] = horizon
    if schedule is not None:
        model_parameters["b_regime_schedule"] = schedule
    return {
        "model_name": "regime-switched-test",
        "gnn_section": "ActInfPOMDP",
        "initialparameterization": {
            "A": _A,
            "B_regime": [_SLICE_CALM, _SLICE_STORM],
            "C": _C,
            "D": _D,
        },
        "model_parameters": model_parameters,
    }


def _time_varying_spec(horizon: int) -> dict:
    model_parameters = _base_params()
    model_parameters["num_timesteps"] = horizon
    return {
        "model_name": "time-varying-test",
        "gnn_section": "ActInfPOMDP",
        "initialparameterization": {
            "A": _A,
            "B_t": [_SLICE_CALM, _SLICE_STORM],
            "C": _C,
            "D": _D,
        },
        "model_parameters": model_parameters,
    }


def _assert_valid_rollout(results: dict, horizon: int) -> None:
    assert results.get("framework") == "PyMDP"
    assert results.get("schema_version") == "pymdp_simulation_v1"
    assert len(results["observations"]) == horizon
    assert len(results["actions"]) == horizon
    validation = results.get("validation", {})
    assert validation.get("all_beliefs_valid")
    assert validation.get("beliefs_sum_to_one")
    assert validation.get("actions_in_range")
    assert validation.get("nonstationary_schedule_applied") is True
    for belief in results.get("beliefs", []):
        assert all(0.0 <= v <= 1.0 for v in belief)


def test_regime_switched_rollout_applies_schedule(tmp_path: Path) -> None:
    success, results = run_pymdp_simulation(
        _regime_spec(schedule=[0, 0, 0, 1, 1, 1], horizon=6), tmp_path / "regime"
    )
    assert success, results.get("error", results)
    _assert_valid_rollout(results, 6)

    nonstationary = results["nonstationary"]
    assert nonstationary["kind"] == "regime_switched"
    assert nonstationary["transition_key"] == "B_regime"
    assert nonstationary["schedule"] == [0, 0, 0, 1, 1, 1]
    assert nonstationary["schedule_truncated"] is False
    assert nonstationary["distinct_transitions"] == 2


def test_time_varying_rollout_holds_last_phase(tmp_path: Path) -> None:
    success, results = run_pymdp_simulation(
        _time_varying_spec(horizon=4), tmp_path / "time_varying"
    )
    assert success, results.get("error", results)
    _assert_valid_rollout(results, 4)

    nonstationary = results["nonstationary"]
    assert nonstationary["kind"] == "time_varying"
    assert nonstationary["transition_key"] == "B_t"
    assert nonstationary["declared_span"] == 2
    assert nonstationary["horizon"] == 4
    assert nonstationary["schedule_truncated"] is True
    assert nonstationary["distinct_transitions"] == 2


def test_regime_without_schedule_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="b_regime_schedule"):
        run_pymdp_simulation(_regime_spec(schedule=None, horizon=4), tmp_path / "bad")


def test_regime_schedule_out_of_range_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside 0..1"):
        run_pymdp_simulation(
            _regime_spec(schedule=[0, 0, 2, 1, 1, 1], horizon=6), tmp_path / "range"
        )


@pytest.mark.parametrize("schedule", [[0, 0.9], [0, True], "0,,1"])
def test_regime_schedule_keeps_authored_integer_indices(tmp_path, schedule):
    spec = _regime_spec(horizon=3)
    spec["model_parameters"]["b_regime_schedule"] = schedule
    with pytest.raises(ValueError, match="nonboolean integer indices"):
        run_pymdp_simulation(spec, tmp_path)
    assert not (tmp_path / "simulation_results.json").exists()


def test_native_asymmetric_phases_apply_the_declared_action_and_hold_last(tmp_path):
    spec = _time_varying_spec(horizon=4)
    spec["initialparameterization"]["A"] = np.eye(3).tolist()
    spec["initialparameterization"]["D"] = [1.0, 0.0, 0.0]
    # Authored phase/action targets are independent of the prior state.
    targets = [[1, 2], [2, 0]]
    stored = np.zeros((2, 2, 3, 3))
    for phase, actions in enumerate(targets):
        for action, next_state in enumerate(actions):
            stored[phase, action, next_state, :] = 1.0
    spec["initialparameterization"]["B_t"] = stored.tolist()
    spec["model_parameters"]["b_tensor_order_t"] = "action_next_state_previous_state"
    spec["model_parameters"]["b_tensor_order"] = "next_state_previous_state_action"

    success, result = run_pymdp_simulation(spec, tmp_path)
    assert success, result
    states = [0] + [
        targets[min(step, 1)][action] for step, action in enumerate(result["actions"])
    ]
    assert result["true_states"] == states
    assert result["observations"] == states[:-1]
    assert result["true_states"][2] != targets[0][result["actions"][1]]
