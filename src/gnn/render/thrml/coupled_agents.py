"""Compile a bounded positive two-agent categorical source to a joint HMM."""

from __future__ import annotations

from copy import deepcopy
from numbers import Real
from typing import Any

import numpy as np

from gnn.render.pomdp_contract import nested_shape
from gnn.render.thrml.adapter import (
    MAX_CATEGORIES,
    UnsupportedTHRMLModel,
    _admission,
    _integer,
)

SOURCE_KEY = "thrml_coupled_agents"
COUPLED_CONTRACT = "gnn-thrml-two-agent-positive-v1"
COUPLED_COMPOSITION = "two_agent_coupled"
_SOURCE_KEYS = {
    "contract",
    "agent_ids",
    "action_ids",
    "A",
    "B",
    "D",
    "observations",
    "transition_actions",
}


def _labels(values: object, name: str, *, count: int | None = None) -> list[str]:
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= MAX_CATEGORIES:
        raise ValueError(f"{SOURCE_KEY} {name} must be a bounded ordered list")
    if count is not None and len(values) != count:
        raise ValueError(f"{SOURCE_KEY} {name} must contain exactly {count} IDs")
    if any(
        not isinstance(value, str) or not value.strip() or len(value) > 128
        for value in values
    ):
        raise ValueError(
            f"{SOURCE_KEY} {name} must contain nonempty string IDs of at most 128 characters"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"{SOURCE_KEY} {name} must contain distinct IDs")
    return list(values)


def _positive_probabilities(
    raw: object, shape: tuple[int, ...], name: str, *, axis: int | None
) -> np.ndarray:
    entries = np.asarray(raw, dtype=object)
    if entries.shape != shape:
        raise ValueError(f"{SOURCE_KEY} {name} must have shape {shape}")
    if any(
        isinstance(value, bool) or not isinstance(value, Real) for value in entries.flat
    ):
        raise ValueError(
            f"{SOURCE_KEY} {name} requires finite nonboolean probabilities"
        )
    try:
        values = entries.astype(np.float64)
    except OverflowError as exc:
        raise ValueError(f"{SOURCE_KEY} {name} requires finite probabilities") from exc
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(
            f"{SOURCE_KEY} {name} requires finite nonnegative probabilities"
        )
    if np.any(values == 0):
        raise UnsupportedTHRMLModel(
            f"structural-zero {SOURCE_KEY}/{name} support is unsupported"
        )
    if not np.allclose(values.sum(axis=axis), 1.0, rtol=0, atol=1e-12):
        raise ValueError(f"{SOURCE_KEY} {name} probability mass must be one")
    return values


def authored_schedule(raw: object) -> int:
    """Read the versioned source header and length of its observed sequence."""
    if not isinstance(raw, dict) or set(raw) != _SOURCE_KEYS:
        raise ValueError(f"{SOURCE_KEY} requires exactly {sorted(_SOURCE_KEYS)}")
    if raw["contract"] != COUPLED_CONTRACT:
        raise UnsupportedTHRMLModel(f"unknown {SOURCE_KEY} contract")
    sequence = raw["observations"]
    if not isinstance(sequence, (list, tuple)) or not sequence:
        raise ValueError(
            f"{SOURCE_KEY} observations must be an authored nonempty sequence"
        )
    return len(sequence)


def compile_coupled_agents(
    source: dict[str, Any], raw: dict[str, Any], schedule: dict[str, int]
) -> tuple[dict[str, Any], list[int], list[int]]:
    """Bind the source axes and materialize exactly one admitted joint chain."""
    agents = _labels(raw["agent_ids"], "agent_ids", count=2)
    actions = _labels(raw["action_ids"], "action_ids")
    a_raw, b_raw = raw["A"], raw["B"]
    if any(
        not isinstance(value, (list, tuple)) or len(value) != 2
        for value in (a_raw, b_raw)
    ):
        raise ValueError(f"{SOURCE_KEY} A and B require one ordered table per agent")
    a_shapes = [nested_shape(table) for table in a_raw]
    if any(len(shape) != 2 or min(shape) < 1 for shape in a_shapes):
        raise ValueError(f"{SOURCE_KEY} A tables have axes (observation, own_state)")
    (o0, s0), (o1, s1) = a_shapes
    observations, states, controls = o0 * o1, s0 * s1, len(actions)
    dimensions = {
        "num_states": states,
        "num_hidden_states": states,
        "num_obs": observations,
        "num_observations": observations,
        "num_actions": controls,
        "num_controls": controls,
    }
    for mapping in (source, source.get("model_parameters") or {}):
        for key, expected in dimensions.items():
            if key in mapping and _integer(mapping[key], key) != expected:
                raise ValueError(
                    f"{SOURCE_KEY} {key} disagrees with the joint dimensions"
                )
    joint_entries = (
        observations * states + states * states * controls + states + observations
    )
    _admission(
        states,
        observations,
        schedule,
        synthetic=False,
        parameters=raw,
        additional_retained_entries=joint_entries,
    )

    readings = []
    for reading in raw["observations"]:
        if not isinstance(reading, (list, tuple)) or len(reading) != 2:
            raise ValueError(f"{SOURCE_KEY} observations require one index per agent")
        left = _integer(
            reading[0], f"{SOURCE_KEY} first observation", minimum=0, maximum=o0 - 1
        )
        right = _integer(
            reading[1], f"{SOURCE_KEY} second observation", minimum=0, maximum=o1 - 1
        )
        readings.append(left * o1 + right)
    fixed = raw["transition_actions"]
    if not isinstance(fixed, (list, tuple)) or len(fixed) != len(readings) - 1:
        raise ValueError(f"{SOURCE_KEY} requires exactly T - 1 transition_actions")
    action_indices = {action: index for index, action in enumerate(actions)}
    if any(
        not isinstance(action, str) or action not in action_indices for action in fixed
    ):
        raise ValueError(
            f"{SOURCE_KEY} transition_actions must reference declared action_ids"
        )

    a0 = _positive_probabilities(a_raw[0], (o0, s0), "A[0]", axis=0)
    a1 = _positive_probabilities(a_raw[1], (o1, s1), "A[1]", axis=0)
    b0 = _positive_probabilities(b_raw[0], (s0, s0, s1, controls), "B[0]", axis=0)
    b1 = _positive_probabilities(b_raw[1], (s1, s0, s1, controls), "B[1]", axis=0)
    prior = _positive_probabilities(raw["D"], (s0, s1), "D", axis=None)
    a = np.einsum("ik,jl->ijkl", a0, a1).reshape(observations, states)
    b = np.einsum("ikla,jkla->ijkla", b0, b1).reshape(states, states, controls)
    parameters = {
        "A": a.tolist(),
        "B": b.tolist(),
        "C": [0.0] * observations,
        "D": prior.ravel().tolist(),
    }
    derived = {
        "model_name": source.get("model_name", "GNN model"),
        "initialparameterization": parameters,
        "model_parameters": {
            "num_states": states,
            "num_obs": observations,
            "num_actions": controls,
            "b_tensor_order": "next_state_previous_state_action",
        },
        "matrix_provenance": {
            key: {"source": COUPLED_COMPOSITION, "derived": True} for key in parameters
        },
        "thrml_composition": {
            "kind": COUPLED_COMPOSITION,
            "source_declaration": deepcopy(raw),
            "state_factors": [
                {"name": name, "size": size}
                for name, size in zip(agents, (s0, s1), strict=True)
            ],
            "observation_modalities": [
                {"name": name, "size": size}
                for name, size in zip(agents, (o0, o1), strict=True)
            ],
            "joint_index_order": "Cartesian in declared agent order, last agent varies fastest",
            "preference_origin": "derived neutral vector; fixed-action smoothing does not use preferences",
        },
    }
    return derived, readings, [action_indices[action] for action in fixed]
