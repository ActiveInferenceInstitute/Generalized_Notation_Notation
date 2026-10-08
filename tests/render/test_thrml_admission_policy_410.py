"""Installed THRML admission uses its canonical owner across first-import order.

Fresh children isolate policy changes without resetting module caches or native
packages. The real public payload builder performs only bounded composition;
these controls never run a sampler or construct a THRML model.
"""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest

CAPS = [
    ("MAX_FACTOR_ENTRIES", 100, 200),
    ("MAX_ESTIMATED_BYTES", 1, 64 * 1024 * 1024),
    ("MAX_CATEGORIES", 2, 6),
]

# Execute the installed public boundary in a fresh ordinary interpreter. The
# lower-policy sentinels observe forbidden allocations, not a simulated builder.
CHILD = r"""
import importlib
import json
import sys

from gnn.render.thrml import adapter

request = json.load(sys.stdin)
name = request['cap']
original = getattr(adapter, name)
assert 'gnn.render.thrml.composition' not in sys.modules
if request['prime'] is not None:
    setattr(adapter, name, request['prime'])
try:
    composition = importlib.import_module('gnn.render.thrml.composition')
finally:
    setattr(adapter, name, original)
if request['active'] is not None:
    setattr(adapter, name, request['active'])
active = getattr(adapter, name)
materializations = []
replacements = []
if request['reject']:
    import gnn.render.pomdp_processor as processor
    def forbidden(*args, **kwargs):
        materializations.append('Cartesian/native materialization')
        raise AssertionError('materialization occurred before admission refusal')
    for module, attribute in [(composition.itertools, 'product'),
                              (composition, 'deepcopy'),
                              (composition.np, 'asarray'),
                              (processor, 'pomdp_to_gnn_spec')]:
        replacements.append((module, attribute, getattr(module, attribute)))
        setattr(module, attribute, forbidden)
try:
    try:
        payload = adapter.build_thrml_payload(request['source'], request['options'])
        from gnn.render.thrml.runtime import _validate_payload
        _validate_payload(payload)
        result = {'success': True, 'payload': payload}
    except ValueError as error:
        result = {'success': False, 'error': str(error)}
finally:
    setattr(adapter, name, original)
    for module, attribute, value in replacements:
        setattr(module, attribute, value)
result.update(active=active, original=original,
              source_after=request['source'], materializations=materializations,
              native_imports=[n for n in ('jax', 'thrml', 'equinox') if n in sys.modules],
              origin=adapter.__file__)
assert not result['native_imports'], result['native_imports']
assert getattr(adapter, name) == original
print(json.dumps(result, allow_nan=False))
"""


def authored_factors() -> dict:
    """Dyadic probabilities avoid tolerance hiding in the independent oracle."""
    matrices = {
        "A_m0": [
            [[0.75, 0.5, 0.25], [0.125, 0.375, 0.875]],
            [[0.25, 0.5, 0.75], [0.875, 0.625, 0.125]],
        ],
        "A_m1": [[0.5, 0.25, 0.125], [0.25, 0.5, 0.375], [0.25, 0.25, 0.5]],
        "B_f0": [[0.75, 0.25], [0.25, 0.75]],
        "B_f1": [
            [[0.5, 0.25], [0.25, 0.5], [0.125, 0.25]],
            [[0.25, 0.5], [0.5, 0.25], [0.375, 0.25]],
            [[0.25, 0.25], [0.25, 0.25], [0.5, 0.5]],
        ],
        "C_m0": [0.5, -0.25],
        "C_m1": [-0.5, 0.25, 0.75],
        "D_f0": [0.25, 0.75],
        "D_f1": [0.25, 0.25, 0.5],
        "E": [0.25, 0.75],
    }
    return {
        "model_name": "authored six-state six-observation admission witness",
        "model_parameters": {
            "num_actions": 2,
            "num_timesteps": 3,
            "num_factors": 2,
            "b_tensor_order": "next_state_previous_state_action",
        },
        "initialparameterization": matrices,
        "structured_pomdp": {
            "matrices": deepcopy(matrices),
            "state_factors": [{"name": "s_f0", "size": 2}, {"name": "s_f1", "size": 3}],
            "observation_modalities": [
                {"name": "o_m0", "size": 2},
                {"name": "o_m1", "size": 3},
            ],
            "control_factors": [],
        },
    }


def public_payload(tmp_path, cap, prime, active, *, reject=False, one_step=False):
    source = authored_factors()
    options = {
        "num_timesteps": 1 if one_step else 3,
        "num_samples": 1 if one_step else 8,
        "burn_in": 0 if one_step else 2,
        "thin": 1,
        "observations": [0] if one_step else [0, 4, 1],
        "transition_actions": [] if one_step else [1, 0],
        "seed": 42,
    }
    request = {
        "cap": cap,
        "prime": prime,
        "active": active,
        "reject": reject,
        "source": source,
        "options": options,
    }
    authored = tmp_path / "authored.json"
    authored.write_text(json.dumps(source), encoding="utf-8")
    original_bytes = authored.read_bytes()
    process = subprocess.run(
        [sys.executable, "-I", "-c", CHILD],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads(process.stdout)
    receipt = tmp_path / "payload-result.json"
    receipt.write_text(json.dumps(result), encoding="utf-8")
    reopened = json.loads(receipt.read_text(encoding="utf-8"))
    assert reopened == result
    assert result["source_after"] == source
    assert authored.read_bytes() == original_bytes
    assert result["native_imports"] == []
    assert result["materializations"] == []
    return source, result


def assert_authored_joint(source, result):
    assert result["success"], result
    payload = result["payload"]
    component = payload["components"][0]
    assert payload["composition"] == "canonical_joint"
    assert component["num_states"] == component["num_observations"] == 6
    raw = source["initialparameterization"]
    parameters = component["parameters"]
    expected_b = np.stack(
        [
            np.kron(raw["B_f0"], np.asarray(raw["B_f1"])[:, :, action])
            for action in range(2)
        ],
        axis=2,
    )
    expected_a = [
        [
            raw["A_m0"][o0][s0][s1] * raw["A_m1"][o1][s1]
            for s0 in range(2)
            for s1 in range(3)
        ]
        for o0 in range(2)
        for o1 in range(3)
    ]
    np.testing.assert_array_equal(parameters["A"], expected_a)
    np.testing.assert_array_equal(parameters["B"], expected_b)
    np.testing.assert_array_equal(parameters["D"], np.kron(raw["D_f0"], raw["D_f1"]))
    np.testing.assert_array_equal(
        parameters["C"], [a + b for a in raw["C_m0"] for b in raw["C_m1"]]
    )
    assert parameters["E"] == raw["E"]
    assert component["composition_metadata"]["source_matrices"] == raw
    assert component["source_model_parameters"] == source["model_parameters"]


@pytest.mark.parametrize("cap,low,safe", CAPS)
def test_restored_canonical_caps_admit_authored_payload_after_first_import(
    tmp_path, cap, low, safe
):
    source, result = public_payload(tmp_path, cap, low, None)
    assert result["active"] == result["original"]
    assert_authored_joint(source, result)


@pytest.mark.parametrize("cap,low,safe", CAPS)
def test_current_safe_budget_admits_after_composition_was_imported_under_lower_cap(
    tmp_path, cap, low, safe
):
    source, result = public_payload(tmp_path, cap, low, safe)
    assert result["active"] == safe
    assert_authored_joint(source, result)


@pytest.mark.parametrize("first_import_low", [False, True])
@pytest.mark.parametrize("cap,low,safe", CAPS)
def test_current_lower_budget_refuses_before_cartesian_or_array_materialization(
    tmp_path, cap, low, safe, first_import_low
):
    _, result = public_payload(
        tmp_path, cap, low if first_import_low else None, low, reject=True
    )
    assert result["success"] is False
    assert result["active"] == low
    assert "admission" in result["error"] or "admitted range" in result["error"]


def test_all_action_joint_storage_is_refused_before_materialization_at_one_step(
    tmp_path,
):
    _, result = public_payload(
        tmp_path, "MAX_FACTOR_ENTRIES", None, 100, reject=True, one_step=True
    )
    assert result["success"] is False
    assert (
        "joint composition admission rejected before Cartesian materialization"
        in result["error"]
    )
