#!/usr/bin/env python3
"""
Schema constants and payload/path helpers for GNN Step 16 analysis visualizations.

Extracted from ``analysis.visualizations``.
"""

from pathlib import Path
from typing import (
    Any,
    Dict,
)

from .viz_base import np

CURRENT_VISUALIZATION_SCHEMAS = {
    "pymdp_simulation_v1",
    "rxinfer_simulation_v1",
    "activeinference_jl_simulation_v1",
    # Ungated backends (pytorch/numpyro) also stamp these schema ids.
    "pytorch_simulation_v1",
    "numpyro_simulation_v1",
}

VISUALIZATION_FRAMEWORK_DIRS = {
    "pymdp",
    "rxinfer",
    "activeinference_jl",
    "jax",
    "discopy",
    "pytorch",
    "numpyro",
    "cpomdp",
    "bnlearn",
    # stan renders + executes; its results were previously attributed to
    # "unknown" by _framework_from_path_or_payload.
    "stan",
}


def _current_schema_visualization_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Map a current-schema payload into visualization fields.

    By-factor / by-modality maps remain the primary source; the flat
    top-level arrays (as written by the pytorch and numpyro runner schemas)
    act as the fallback for each field.
    """
    if data.get("schema_version") not in CURRENT_VISUALIZATION_SCHEMAS:
        return {}
    return {
        "beliefs": (data.get("beliefs_by_factor", {}) or {}).get(
            "joint_state", data.get("beliefs", [])
        ),
        "actions": (data.get("actions_by_control_factor", {}) or {}).get(
            "joint_action", data.get("actions", [])
        ),
        "observations": (data.get("observations_by_modality", {}) or {}).get(
            "joint_observation", data.get("observations", [])
        ),
        "expected_free_energy": data.get("expected_free_energy", []),
        "variational_free_energy": data.get("variational_free_energy", []),
        "metrics": data.get("metrics", {}),
        "model_parameters": data.get("model_parameters", {}),
        "schema_version": data.get("schema_version"),
    }


def _state_count_from_payload(payload: Dict[str, Any]) -> int:
    """Handle state count from payload for internal callers."""
    model_parameters = payload.get("model_parameters", {})
    if isinstance(model_parameters, dict):
        for key in ("num_states", "num_hidden_states"):
            value = model_parameters.get(key)
            if isinstance(value, int) and value > 0:
                return value

        shape = model_parameters.get("B_shape") or model_parameters.get("A_shape")
        if isinstance(shape, list) and shape:
            first = shape[0]
            if isinstance(first, int) and first > 0:
                return first

    beliefs = payload.get("beliefs") or payload.get("beliefs_by_factor", {}).get(
        "joint_state", []
    )
    if isinstance(beliefs, list) and beliefs and isinstance(beliefs[0], list):
        return len(beliefs[0])
    return 0


def _is_gridworld_payload(payload: Dict[str, Any]) -> bool:
    """Return whether gridworld payload."""
    if payload.get("schema_version") not in CURRENT_VISUALIZATION_SCHEMAS:
        return False

    model_parameters = payload.get("model_parameters", {})
    b_shape = (
        model_parameters.get("B_shape") if isinstance(model_parameters, dict) else None
    )
    if b_shape == [9, 9, 5]:
        return True

    state_count = _state_count_from_payload(payload)
    actions = payload.get("actions") or payload.get(
        "actions_by_control_factor", {}
    ).get("joint_action", [])
    return state_count == 9 and isinstance(actions, list) and len(set(actions)) <= 5


def _series_from_payload(
    payload: Dict[str, Any],
    current_data: Dict[str, Any],
    plain_key: str,
    grouped_key: str,
    grouped_name: str,
) -> list[Any]:
    """Handle series from payload for internal callers."""
    value = payload.get(plain_key)
    if isinstance(value, list) and value:
        return value
    grouped = payload.get(grouped_key, {})
    if isinstance(grouped, dict):
        grouped_value = grouped.get(grouped_name)
        if isinstance(grouped_value, list) and grouped_value:
            return grouped_value
    current_value = current_data.get(plain_key)
    if isinstance(current_value, list):
        return current_value
    return []


def _belief_map_states(beliefs: list[Any]) -> list[int]:
    """Handle belief map states for internal callers."""
    states: list[int] = []
    for belief in beliefs:
        if not isinstance(belief, list) or not belief:
            continue
        try:
            states.append(int(np.argmax(np.asarray(belief, dtype=float))))
        except (TypeError, ValueError):
            continue
    return states


def _gridworld_state_sequence(
    payload: Dict[str, Any], current_data: Dict[str, Any]
) -> list[int]:
    """Handle gridworld state sequence for internal callers."""
    hidden_states = _series_from_payload(
        payload,
        current_data,
        "hidden_states",
        "hidden_states_by_factor",
        "joint_state",
    )
    states: list[int] = []
    for state in hidden_states:
        if isinstance(state, list) and state:
            state = state[0]
        try:
            states.append(int(state))
        except (TypeError, ValueError):
            continue

    beliefs = _series_from_payload(
        payload, current_data, "beliefs", "beliefs_by_factor", "joint_state"
    )
    if not states:
        states = _belief_map_states(beliefs)

    step_counts = [
        len(series)
        for series in [
            beliefs,
            _series_from_payload(
                payload,
                current_data,
                "actions",
                "actions_by_control_factor",
                "joint_action",
            ),
            _series_from_payload(
                payload,
                current_data,
                "observations",
                "observations_by_modality",
                "joint_observation",
            ),
        ]
        if series
    ]
    max_steps = max(step_counts) if step_counts else len(states)
    return states[:max_steps]


def _grid_side_for_states(state_count: int) -> int:
    """Handle grid side for states for internal callers."""
    side = int(np.sqrt(state_count))
    if side * side != state_count:
        raise ValueError(
            f"GridWorld animation requires square state count: {state_count}"
        )
    return side


def _normalize_framework_name(framework: str) -> str:
    """
    Normalize framework names to canonical form.

    Consolidates variants like PyMDP, pymdp -> pymdp
    """
    if not framework:
        return "unknown"

    fw_lower = framework.lower()

    # Consolidate pymdp variants
    if fw_lower == "pymdp" or fw_lower.startswith("pymdp_"):
        return "pymdp"

    # Consolidate rxinfer variants
    if fw_lower in ["rxinfer", "rxinfer_jl"]:
        return "rxinfer"

    # Consolidate activeinference variants
    if fw_lower in ["activeinference_jl", "activeinference"]:
        return "activeinference_jl"

    return fw_lower


def _model_name_from_path(path: Path) -> str:
    """Handle model name from path for internal callers."""
    for ancestor in path.parents:
        candidate = ancestor.name
        if (
            candidate
            and candidate not in VISUALIZATION_FRAMEWORK_DIRS
            and candidate
            not in {
                "simulation_data",
                "execution_logs",
                "individual_outputs",
                "12_execute_output",
                "output",
            }
        ):
            return candidate
    return "unknown"


def _framework_from_path_or_payload(path: Path, payload: Dict[str, Any]) -> str:
    """Handle framework from path or payload for internal callers."""
    for part in path.parts:
        if part in VISUALIZATION_FRAMEWORK_DIRS:
            return _normalize_framework_name(part)
    return _normalize_framework_name(str(payload.get("framework", "unknown")))
