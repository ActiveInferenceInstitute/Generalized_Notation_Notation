"""Bounded independent and exact joint categorical source composition."""

from __future__ import annotations

import itertools
import json
import math
import re
from copy import deepcopy
from typing import Any

import numpy as np

from gnn.render.pomdp_contract import _declared_b_order, nested_shape
from gnn.render.thrml import adapter as _adapter
from gnn.render.thrml.adapter import (
    _GROUP,
    _MODALITY,
    UnsupportedTHRMLModel,
    _admission,
    _integer,
)


def _independent_specs(
    spec: dict[str, Any], matrices: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for key, value in matrices.items():
        match = _GROUP.fullmatch(str(key))
        if match:
            groups.setdefault(match[2], {})[match[1]] = value
    if not groups:
        return {}
    if any(not {"A", "B", "C", "D"} <= set(group) for group in groups.values()):
        if any(name.startswith("agent") for name in groups):
            raise UnsupportedTHRMLModel(
                "agents require complete independent A/B/C/D groups"
            )
        return {}  # Dependent factor/modality tables use the bounded joint composer.
    if any(str(key).startswith(("env_", "coupling", "signal_")) for key in matrices):
        raise UnsupportedTHRMLModel(
            "environment/coupling semantics are not represented by independent THRML components"
        )
    for connection in spec.get("connections", []):
        component_names = set(re.findall(r"(?:agent|f)\d+", json.dumps(connection)))
        if len(component_names & set(groups)) > 1:
            raise UnsupportedTHRMLModel(
                "cross-component connections require an explicit joint coupling contract"
            )
    consumed = {
        f"{letter}_{name}" for name, group in groups.items() for letter in group
    }
    extra = [
        key
        for key in matrices
        if re.match(r"^[ABCDE](?:_|$)", str(key)) and key not in consumed
    ]
    if extra:
        raise UnsupportedTHRMLModel(
            f"unrepresented active-inference matrices: {sorted(extra)}"
        )
    params = spec.get("model_parameters") or {}
    for key, prefix in (
        ("nr_agents", "agent"),
        ("num_agents", "agent"),
        ("num_factors", "f"),
        ("num_state_factors", "f"),
    ):
        declared = params.get(key)
        matching = [name for name in groups if name.startswith(prefix)]
        if (
            declared is not None
            and matching
            and _integer(declared, key) != len(matching)
        ):
            raise ValueError(
                f"{key} does not match the represented independent components"
            )
    provenance = (
        spec.get("matrix_provenance")
        or (spec.get("structured_pomdp") or {}).get("matrix_provenance")
        or {}
    )
    result = {}
    for name, initial in sorted(groups.items()):
        a_shape = nested_shape(initial["A"])
        b_shape = nested_shape(initial["B"])
        if len(a_shape) != 2 or len(b_shape) not in {2, 3}:
            raise UnsupportedTHRMLModel(
                "independent components require 2D likelihoods and 2D/3D transition tables"
            )
        order = _declared_b_order(
            {
                **params,
                **(
                    {"b_tensor_order": params[f"b_tensor_order_{name}"]}
                    if params.get(f"b_tensor_order_{name}")
                    else {}
                ),
            }
        ) or _declared_b_order(
            {"b_tensor_order": (provenance.get(f"B_{name}") or {}).get("source_order")}
        )
        if len(b_shape) == 3 and not order:
            raise UnsupportedTHRMLModel(
                "native component transitions require an explicit B tensor orientation"
            )
        if len(b_shape) == 2:
            actions = 1
        elif order.startswith(("action_", "actions_")):
            actions = b_shape[0]
        else:
            actions = b_shape[-1]
        component_params = {
            **params,
            "num_hidden_states": a_shape[1],
            "num_states": a_shape[1],
            "num_obs": a_shape[0],
            "num_actions": actions,
            "num_factors": 1,
            "nr_agents": 1,
            "num_agents": 1,
            "b_tensor_order": order or "next_state_previous_state_action",
        }
        for descriptor_key in (
            "state_factors",
            "observation_modalities",
            "control_factors",
        ):
            component_params.pop(descriptor_key, None)
        result[name] = {
            "model_name": f"{spec.get('model_name', 'model')}/{name}",
            "initialparameterization": initial,
            "model_parameters": component_params,
            "matrix_provenance": {
                letter: deepcopy(provenance.get(f"{letter}_{name}") or {})
                for letter in initial
            },
        }
    return result


def _compose_modalities(
    spec: dict[str, Any], matrices: dict[str, Any], schedule: dict[str, int]
) -> dict[str, Any]:
    keys = sorted(key for key in matrices if _MODALITY.fullmatch(str(key)))
    if not keys:
        return spec
    if not {"B", "D"} <= set(matrices) or any(
        len(nested_shape(matrices[key])) != 2 for key in keys
    ):
        raise UnsupportedTHRMLModel(
            "multi-modality mapping requires one shared state, B/D and 2D A_m tables"
        )
    allowed = set(keys) | {key.replace("A_", "C_", 1) for key in keys} | {"B", "D", "E"}
    if any(
        re.match(r"^[ABCDE](?:_|$)", str(key)) and key not in allowed
        for key in matrices
    ):
        raise UnsupportedTHRMLModel(
            "multi-modality product has unrepresented declared matrices"
        )
    sizes = [nested_shape(matrices[key])[0] for key in keys]
    states = nested_shape(matrices[keys[0]])[1]
    observations = math.prod(sizes)
    _admission(
        states, observations, schedule, synthetic=True, parameters=matrices
    )  # Before Cartesian materialization.
    if any(nested_shape(matrices[key])[1] != states for key in keys):
        raise ValueError("all observation modalities must bind the same hidden state")
    rows = list(itertools.product(*(range(size) for size in sizes)))
    arrays = [np.asarray(matrices[key], dtype=np.float64) for key in keys]
    from gnn.render.pomdp_contract import normalise_matrix_columns

    arrays = [
        np.asarray(normalise_matrix_columns(value.tolist(), name=key))
        for key, value in zip(keys, arrays)
    ]
    joint_a = np.array(
        [
            np.prod([array[index] for array, index in zip(arrays, row)], axis=0)
            for row in rows
        ]
    )
    c_keys = [key.replace("A_", "C_", 1) for key in keys]
    if any(key not in matrices for key in c_keys):
        raise ValueError("each modality requires its declared C_m preferences")
    c_values = [np.asarray(matrices[key], dtype=float).reshape(-1) for key in c_keys]
    if any(
        len(value) != size or not np.isfinite(value).all()
        for value, size in zip(c_values, sizes)
    ):
        raise ValueError("C_m preferences do not match modality dimensions")
    joint_c = [
        sum(float(value[index]) for value, index in zip(c_values, row)) for row in rows
    ]
    initial = {
        "A": joint_a.tolist(),
        "B": matrices["B"],
        "C": joint_c,
        "D": matrices["D"],
    }
    if "E" in matrices:
        initial["E"] = matrices["E"]
    derived = deepcopy(spec)
    derived["initialparameterization"] = initial
    derived["model_parameters"] = {
        **(spec.get("model_parameters") or {}),
        "num_obs": observations,
        "num_observations": observations,
        "num_hidden_states": states,
    }
    derived["thrml_composition"] = {
        "kind": "conditionally_independent_observation_product",
        "modalities": [{"name": key, "size": size} for key, size in zip(keys, sizes)],
        "joint_observation_order": rows,
        "source_matrices": {key: deepcopy(value) for key, value in matrices.items()},
        "claim": "exact likelihood product under declared conditional independence; sampling is separate",
    }
    return derived


def _compose_factors(
    spec: dict[str, Any], matrices: dict[str, Any], schedule: dict[str, int]
) -> dict[str, Any]:
    """Admit dimensions before delegating to the repository's exact composer."""
    from gnn.extract.pomdp_state import POMDPStateSpace
    from gnn.render.pomdp_processor import pomdp_to_gnn_spec

    structured = spec.get("structured_pomdp") or {}
    params = spec.get("model_parameters") or {}
    factors = structured.get("state_factors") or params.get("state_factors") or []
    modalities = (
        structured.get("observation_modalities")
        or params.get("observation_modalities")
        or []
    )
    controls = structured.get("control_factors") or params.get("control_factors") or []
    if not factors or not modalities:
        raise UnsupportedTHRMLModel(
            "dependent factor composition requires every state/modal descriptor"
        )
    states = math.prod(
        _integer(item.get("size"), "factor size", maximum=_adapter.MAX_CATEGORIES)
        for item in factors
    )
    observations = math.prod(
        _integer(item.get("size"), "modality size", maximum=_adapter.MAX_CATEGORIES)
        for item in modalities
    )
    _admission(states, observations, schedule, synthetic=True, parameters=matrices)
    b_keys = sorted(key for key in matrices if re.fullmatch(r"B_f\d+", str(key)))
    if len(b_keys) != len(factors):
        raise UnsupportedTHRMLModel(
            "each state factor requires exactly one represented B_f table"
        )
    action_counts = []
    for key, factor in zip(b_keys, factors):
        shape = nested_shape(matrices[key])
        order = _declared_b_order(
            {
                **params,
                **(
                    {"b_tensor_order": params[f"b_tensor_order_{key[2:]}"]}
                    if params.get(f"b_tensor_order_{key[2:]}")
                    else {}
                ),
            }
        )
        if len(shape) not in {2, 3}:
            raise UnsupportedTHRMLModel(
                "conditional/coupled factor transitions require a verified joint mapping"
            )
        if len(shape) == 3 and not order:
            raise UnsupportedTHRMLModel(
                "factored B tensors require declared source axes before composition"
            )
        action_counts.append(
            1
            if len(shape) == 2
            else shape[0]
            if order.startswith(("action_", "actions_"))
            else shape[-1]
        )
    actions = (
        math.prod(action_counts)
        if len(b_keys) > 1 and not controls
        else _integer(params.get("num_actions", max(action_counts)), "num_actions")
    )
    # The common composer materializes B for ALL actions, not only fixed edges.
    materialized = (
        observations * states + states * states * actions + states + observations
    )
    if (
        materialized > _adapter.MAX_FACTOR_ENTRIES
        or 64 * materialized > _adapter.MAX_ESTIMATED_BYTES
    ):
        raise ValueError(
            "THRML joint composition admission rejected before Cartesian materialization"
        )
    # The composer consumes categorical A_*/B_*/C_*/D_* and the optional global E.
    # Reject other declared semantics before its output could omit them.
    if any(
        str(key).startswith(("env_", "coupling", "signal_", "E_")) for key in matrices
    ):
        raise UnsupportedTHRMLModel(
            "joint factor composition cannot omit coupling/environment/component habit semantics"
        )
    space = POMDPStateSpace(
        num_states=states,
        num_observations=observations,
        num_actions=actions,
        state_factors=deepcopy(factors),
        observation_modalities=deepcopy(modalities),
        control_factors=deepcopy(controls),
        matrices=deepcopy(matrices),
        model_parameters=deepcopy(params),
        model_name=spec.get("model_name", "GNN model"),
        gnn_section=spec.get("gnn_section"),
        num_timesteps=schedule["num_timesteps"],
        initial_parameterization=deepcopy(spec.get("initialparameterization") or {}),
        matrix_provenance=deepcopy(
            spec.get("matrix_provenance") or structured.get("matrix_provenance") or {}
        ),
    )
    result = pomdp_to_gnn_spec(space, timesteps=schedule["num_timesteps"])
    result["thrml_composition"] = {
        "kind": "bounded_repository_factored_joint_composition",
        "state_factors": deepcopy(factors),
        "observation_modalities": deepcopy(modalities),
        "source_matrices": deepcopy(matrices),
        "materialized_entries": materialized,
        "claim": "repository exact categorical composition; Gibbs sampling accuracy is separate",
    }
    return result


def _joint_metadata(spec: dict[str, Any]) -> dict[str, Any]:
    structured = spec.get("structured_pomdp") or {}
    params = spec.get("model_parameters") or {}
    metadata = deepcopy(spec.get("thrml_composition") or {})
    provenance = (
        spec.get("matrix_provenance") or structured.get("matrix_provenance") or {}
    )
    if (
        not metadata
        and (provenance.get("A") or {}).get("source") != "factored_joint_composition"
    ):
        # Flat extractor descriptors also contain next-state bookkeeping aliases.
        # Retain them in source_structure without claiming joint marginals.
        return {}
    for key in ("state_factors", "observation_modalities", "control_factors"):
        if structured.get(key) or params.get(key):
            metadata[key] = deepcopy(structured.get(key) or params.get(key))
    if structured.get("matrices"):
        metadata["source_matrices"] = deepcopy(structured["matrices"])
    return metadata
