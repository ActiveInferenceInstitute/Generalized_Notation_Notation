"""Source-bound finite Bayesian filtering and smoothing with bnlearn.

The action sequence is exogenous. Preferences and policy optimization are outside
this contract; the declared A/B/D conditionals define every inference update.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from gnn.render.pomdp_contract import (
    _declared_b_order,
    build_canonical_pomdp_spec,
    nested_shape,
)


def source_payload(model: dict[str, Any]) -> dict[str, Any]:
    """Validate a complete finite model without substituting synthetic CPTs."""
    from gnn.render.pomdp_contract import ModelKind, detect_model_kinds

    kinds = detect_model_kinds(model)
    if kinds - {ModelKind.FLAT, ModelKind.FACTORED}:
        raise ValueError(
            "unsupported-bnlearn: source filtering requires a stationary finite model"
        )
    initial = {
        **((model.get("structured_pomdp") or {}).get("matrices") or {}),
        **(
            model.get("initialparameterization")
            or model.get("initial_parameterization")
            or {}
        ),
    }
    if any(str(key).startswith(("A_", "B_", "C_", "D_", "E_")) for key in initial):
        raise ValueError(
            "unsupported-bnlearn: component matrices require a verified joint mapping"
        )
    missing = {"A", "B", "D"} - initial.keys()
    if missing:
        raise ValueError(
            f"bnlearn source filtering requires authored A/B/D: {sorted(missing)}"
        )
    # Bound retained source data before the canonical validator copies arrays.
    cells = 0
    stack = list(initial.values())
    while stack:
        value = stack.pop()
        if isinstance(value, np.ndarray):
            cells += value.size
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        else:
            cells += 1
        if cells > 2_000_000:
            raise ValueError(
                "bnlearn source filtering exceeds the 2,000,000-cell admission limit"
            )
    parameters = model.get("model_parameters") or {}
    if len(nested_shape(initial["B"])) == 3 and not _declared_b_order(parameters):
        raise ValueError(
            "unsupported-bnlearn: three-dimensional B requires declared tensor orientation"
        )
    authored = dict(initial)
    # C has no role in a fixed-action Bayesian filtering question. A neutral
    # neutral vector admits the shared shape validator without inventing a policy.
    if "C" not in initial:
        initial["C"] = np.zeros(np.asarray(initial["A"]).shape[0]).tolist()
    canonical = build_canonical_pomdp_spec(
        {**model, "initialparameterization": initial}
    )
    if "C" not in authored:
        canonical["matrix_provenance"]["C"].update(
            source="fixed_action_filter_adapter",
            derived=True,
            reason="neutral unused preferences; authored A/B/D define conditional filtering",
        )
    matrices = canonical["initialparameterization"]
    a, b, d = (np.asarray(matrices[key], dtype=float) for key in ("A", "B", "D"))
    if a.size + b.size + d.size > 2_000_000:
        raise ValueError(
            "bnlearn source filtering exceeds the 2,000,000-cell admission limit"
        )
    timesteps = parameters.get("num_timesteps", 15)
    seed = parameters.get("random_seed", parameters.get("seed", 42))
    for key, value, minimum in (
        ("num_timesteps", timesteps, 1),
        ("random_seed", seed, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if timesteps > 100_000 or timesteps * (a.size + b.size) > 100_000_000:
        raise ValueError("bnlearn source filtering workload admission limit exceeded")
    payload = {
        "model_name": str(model.get("model_name") or model.get("name") or "GNN Model"),
        "A": a.tolist(),
        "B": b.tolist(),
        "D": d.reshape(-1).tolist(),
        "source_parameters": authored,
        "matrix_provenance": canonical["matrix_provenance"],
        "num_timesteps": timesteps,
        "random_seed": seed,
        "inference_contract": "bnlearn_fixed_action_filter_v1",
        "action_policy": "declared external actions, otherwise cyclic actions; no policy optimization",
        "preference_semantics": "C/E are not used by conditional fixed-action filtering",
    }
    payload["model_contract_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return payload


def generate_source_program(
    model: dict[str, Any], source_identity: dict[str, Any] | None = None
) -> str:
    """Emit a standalone native bnlearn program preserving the validated CPTs."""
    payload = source_payload(model)
    payload["source_identity"] = dict(source_identity or {})
    return _PROGRAM.replace(
        "__PAYLOAD_LITERAL__", repr(json.dumps(payload, allow_nan=False))
    )


_PROGRAM = '''#!/usr/bin/env python3
"""Native bnlearn fixed-action inference for the declared finite A/B/D model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from importlib.metadata import version

import bnlearn as bn
import numpy as np
from pgmpy.factors.discrete import TabularCPD

PAYLOAD = json.loads(__PAYLOAD_LITERAL__)


def run_simulation(observations=None, transition_actions=None):
    started = time.monotonic()
    A, B, D = (np.asarray(PAYLOAD[key], dtype=float) for key in ("A", "B", "D"))
    n, _, actions = B.shape
    m, steps = len(A), PAYLOAD["num_timesteps"]
    supplied = observations is not None
    if observations is not None and (
        not isinstance(observations, list) or len(observations) != steps or
        any(type(o) is not int or not 0 <= o < m for o in observations)
    ):
        raise ValueError("observations must supply one valid integer index per authored timestep")
    if transition_actions is None:
        transition_actions = [step % actions for step in range(steps - 1)]
    if (not isinstance(transition_actions, list) or len(transition_actions) != steps - 1 or
        any(type(a) is not int or not 0 <= a < actions for a in transition_actions)):
        raise ValueError("transition_actions must supply one valid integer index per transition")
    rng = np.random.default_rng(PAYLOAD["random_seed"])
    state = int(rng.choice(n, p=D)) if not supplied else None
    readings, truth, beliefs = [], [], []
    prior = D
    for step in range(steps):
        if step and not supplied:
            state = int(rng.choice(n, p=B[:, state, transition_actions[step - 1]]))
        reading = observations[step] if supplied else int(rng.choice(m, p=A[:, state]))
        if step == 0:
            cpds = [TabularCPD("S", n, prior[:, None]),
                    TabularCPD("O", m, A, evidence=["S"], evidence_card=[n])]
            network = bn.make_DAG([("S", "O")], CPD=cpds)
            evidence = {"O": reading}
        else:
            # pgmpy orders the last evidence axis fastest: (previous, action).
            cpds = [TabularCPD("S_prev", n, prior[:, None]),
                    TabularCPD("Action", actions, np.full((actions, 1), 1 / actions)),
                    TabularCPD("S", n, B.reshape(n, n * actions),
                               evidence=["S_prev", "Action"], evidence_card=[n, actions]),
                    TabularCPD("O", m, A, evidence=["S"], evidence_card=[n])]
            network = bn.make_DAG([("S_prev", "S"), ("Action", "S"), ("S", "O")], CPD=cpds)
            evidence = {"Action": transition_actions[step - 1], "O": reading}
        query = bn.inference.fit(network, variables=["S"], evidence=evidence)
        posterior = np.asarray(query.values, dtype=float).reshape(-1)
        if posterior.shape != (n,) or not np.isfinite(posterior).all() or np.any(posterior < 0) or not np.isclose(posterior.sum(), 1):
            raise ValueError("native bnlearn returned an invalid posterior (possibly impossible evidence)")
        prior = posterior
        beliefs.append(posterior.tolist())
        readings.append(reading)
        if state is not None:
            truth.append(state)
    results = {
        "schema_version": "bnlearn_fixed_action_filter_v1", "framework": "bnlearn",
        "inference_contract": "bnlearn_fixed_action_filter_v1",
        "dtype": str(A.dtype), "initial_time": "first_observation_uses_prior",
        "conditioning_horizon": list(range(1, steps + 1)),
        "axes": {"A": ["observation", "state"], "B": ["next_state", "previous_state", "action"]},
        "belief_axes": ["timestep", "state"],
        "python_version": sys.version, "numpy_version": np.__version__,
        "model_name": PAYLOAD["model_name"], "model_kind": "discrete",
        "inference_mode": "exact_categorical_filtering", "control_mode": "fixed_transition_actions",
        "num_timesteps": steps, "num_states": n, "num_observations": m,
        "beliefs": beliefs, "observations": readings, "transition_actions": transition_actions,
        "action_semantics": "transition_into_next_state",
        "true_states": truth, "parameters": {"A": A.tolist(), "B": B.tolist(), "D": D.tolist()},
        "model_parameters": {"num_states": n, "num_obs": m, "num_actions": actions,
                             "num_timesteps": steps, "b_tensor_order": "next_state_previous_state_action"},
        "model_contract_sha256": PAYLOAD["model_contract_sha256"],
        "source_parameter_fidelity": True,
        "source_parameters": PAYLOAD["source_parameters"],
        "matrix_provenance": PAYLOAD["matrix_provenance"],
        "source_identity": PAYLOAD["source_identity"],
        "scientific_limits": [PAYLOAD["action_policy"], PAYLOAD["preference_semantics"]],
        "bnlearn_version": bn.__version__, "random_seed": PAYLOAD["random_seed"],
        "validation": {"posterior_finite": True, "posterior_normalized": True, "all_valid": True},
        "execution_time_seconds": time.monotonic() - started,
        "generated_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return write_result(results)


def write_result(results):
    output = Path(os.environ.get("BNLEARN_OUTPUT_DIR", "."))
    output.mkdir(parents=True, exist_ok=True)
    (output / "simulation_results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print("BNLEARN_RESULTS " + json.dumps(results, allow_nan=False))
    return results


def run_smoothing(observations, transition_actions):
    """Condition every hidden state on the same complete observation sequence."""
    started = time.monotonic()
    A, B, D = (np.asarray(PAYLOAD[key], dtype=np.float64) for key in ("A", "B", "D"))
    n, _, actions = B.shape
    m, steps = len(A), PAYLOAD["num_timesteps"]
    if max(n, m, actions) > 4 or steps > 8 or n ** steps > 4096:
        raise ValueError("bnlearn smoothing exceeds the finite witness scope")
    if (not isinstance(observations, list) or len(observations) != steps or
        any(type(value) is not int or not 0 <= value < m for value in observations)):
        raise ValueError("smoothing requires one observed integer index per timestep")
    if (not isinstance(transition_actions, list) or len(transition_actions) != steps - 1 or
        any(type(value) is not int or not 0 <= value < actions for value in transition_actions)):
        raise ValueError("smoothing requires one fixed integer action per transition")

    states = [f"S{step}" for step in range(steps)]
    outputs = [f"O{step}" for step in range(steps)]
    edges = list(zip(states, outputs, strict=True))
    edges.extend(zip(states[:-1], states[1:], strict=True))
    cpds = [TabularCPD(states[0], n, D[:, None])]
    for step, (state, observation) in enumerate(zip(states, outputs, strict=True)):
        cpds.append(TabularCPD(observation, m, A, evidence=[state], evidence_card=[n]))
        if step:
            action = transition_actions[step - 1]
            cpds.append(TabularCPD(state, n, B[:, :, action], evidence=[states[step - 1]], evidence_card=[n]))
    network = bn.make_DAG(edges, CPD=cpds)
    evidence = dict(zip(outputs, observations, strict=True))
    beliefs = []
    for state in states:
        query = bn.inference.fit(network, variables=[state], evidence=evidence, to_df=False)
        posterior = np.asarray(query.values)
        if (posterior.dtype != np.dtype("float64") or posterior.shape != (n,) or
            not np.isfinite(posterior).all() or np.any(posterior < 0) or
            np.any(posterior > 1) or not np.isclose(posterior.sum(), 1, atol=1e-12, rtol=0)):
            raise ValueError("native bnlearn returned an invalid smoothing posterior")
        beliefs.append(posterior.tolist())

    contract = "gnn_finite_fixed_action_smoother_v1"
    return write_result({
        "schema_version": contract, "inference_contract": contract, "framework": "bnlearn",
        "question": "smoothing", "inference_mode": "exact_categorical_smoothing",
        "control_mode": "fixed_transition_actions", "action_semantics": "transition_into_next_state",
        "dtype": str(posterior.dtype), "initial_time": "first_observation_uses_prior",
        "conditioning_horizon": [steps] * steps,
        "axes": {"A": ["observation", "state"], "B": ["next_state", "previous_state", "action"]},
        "belief_axes": ["timestep", "state"], "num_timesteps": steps, "num_states": n,
        "num_observations": m, "num_actions": actions, "model_name": PAYLOAD["model_name"],
        "random_seed": PAYLOAD["random_seed"], "parameters": {"A": A.tolist(), "B": B.tolist(), "D": D.tolist()},
        "smoothed_beliefs": beliefs, "observations": observations, "transition_actions": transition_actions,
        "model_contract_sha256": PAYLOAD["model_contract_sha256"], "source_identity": PAYLOAD["source_identity"],
        "generated_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bnlearn_version": bn.__version__, "pgmpy_version": version("pgmpy"), "numpy_version": np.__version__,
        "python_version": sys.version, "execution_time_seconds": time.monotonic() - started,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", choices=("filter", "smooth"), default="filter")
    parser.add_argument("--observations-json", type=Path)
    parser.add_argument("--transition-actions-json", type=Path)
    args = parser.parse_args()
    if args.inference == "smooth" and (
        args.observations_json is None or args.transition_actions_json is None
    ):
        parser.error("smoothing requires --observations-json and --transition-actions-json")
    observations = json.loads(args.observations_json.read_text()) if args.observations_json else None
    actions = json.loads(args.transition_actions_json.read_text()) if args.transition_actions_json else None
    if args.inference == "smooth":
        run_smoothing(observations, actions)
    else:
        run_simulation(observations, actions)
'''
