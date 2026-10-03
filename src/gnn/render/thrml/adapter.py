"""Strict, bounded categorical model mapping for released THRML 0.1.4.

All validation/admission is independent of JAX/THRML imports. The native
program samples a fixed-action, full-observation-sequence posterior; it does
not optimize preferences or claim variational/exact inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from typing import Any

import numpy as np

from gnn.render.pomdp_contract import (
    ModelKind,
    _declared_b_order,
    build_canonical_pomdp_spec,
    detect_model_kinds,
    nested_shape,
)

THRML_VERSION = "0.1.4"
SCHEMA_VERSION = 1
MAX_COMPONENTS = 16
MAX_FACTOR_ENTRIES = 1_048_576
MAX_SITE_UPDATES = 4_194_304
MAX_ESTIMATED_BYTES = 128 * 1024 * 1024
MAX_CATEGORIES = 4096
_GROUP = re.compile(r"^([ABCDE])_(agent\d+|f\d+)$")
_MODALITY = re.compile(r"^A_m\d+$")
OPTION_KEYS = {
    "num_timesteps",
    "num_samples",
    "burn_in",
    "thin",
    "seed",
    "observations",
    "transition_actions",
}


class UnsupportedTHRMLModel(ValueError):
    """Declared semantics are outside this experimental mapping."""


def _integer(
    value: Any, name: str, *, minimum: int = 1, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its admitted range")
    return result


def contract_digest(value: Any) -> str:
    """Bind numeric values and mapping semantics, independently of source bytes."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _retained_matrix_entries(parameters: dict[str, Any] | None) -> int:
    """Count existing source leaves without an array copy or Cartesian list."""
    count = 0
    exhausted = object()
    for key, value in (parameters or {}).items():
        if not re.match(r"^[ABCDE](?:_|$)", str(key)):
            continue
        stack = [iter((value,))]
        while stack:
            item = next(stack[-1], exhausted)
            if item is exhausted:
                stack.pop()
                continue
            if isinstance(item, np.ndarray):
                count += int(item.size)
            elif isinstance(item, (list, tuple)):
                stack.append(iter(item))
            else:
                count += 1
            if count > MAX_FACTOR_ENTRIES:
                raise ValueError(
                    "THRML retained source matrix admission limit exceeded"
                )
    return count


def _schedule(spec: dict[str, Any], options: dict[str, Any]) -> dict[str, int]:
    params = spec.get("model_parameters") or {}
    return {
        "num_timesteps": _integer(
            options.get("num_timesteps", params.get("num_timesteps", 10)),
            "num_timesteps",
        ),
        "num_samples": _integer(options.get("num_samples", 2048), "num_samples"),
        "burn_in": _integer(options.get("burn_in", 256), "burn_in", minimum=0),
        "thin": _integer(options.get("thin", 2), "thin"),
        "seed": _integer(options.get("seed", 0), "seed", minimum=0, maximum=2**32 - 1),
    }


def _admission(
    states: int,
    observations: int,
    schedule: dict[str, int],
    *,
    synthetic: bool,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not 1 <= states <= MAX_CATEGORIES or not 1 <= observations <= MAX_CATEGORIES:
        raise ValueError("THRML category admission limit exceeded")
    timesteps = schedule["num_timesteps"]
    entries = (
        states + max(timesteps - 1, 0) * states**2 + timesteps * observations * states
    )
    updates = (
        schedule["burn_in"] + schedule["num_samples"] * schedule["thin"]
    ) * timesteps
    if synthetic:
        updates += (schedule["burn_in"] + schedule["thin"]) * 2 * timesteps
    retained = _retained_matrix_entries(parameters)
    estimated = (
        64 * entries
        + 64 * retained
        + 32 * schedule["num_samples"] * timesteps
        + 256 * timesteps * (states + observations)
    )
    if (
        entries > MAX_FACTOR_ENTRIES
        or retained > MAX_FACTOR_ENTRIES
        or updates > MAX_SITE_UPDATES
        or estimated > MAX_ESTIMATED_BYTES
    ):
        raise ValueError(
            f"THRML resource admission rejected: factor_entries={entries}, site_updates={updates}, estimated_bytes={estimated}"
        )
    return {
        "factor_entries": entries,
        "site_updates": updates,
        "estimated_bytes": estimated,
        "retained_matrix_entries": retained,
        "estimate_is_upper_bound": False,
        "caps": {
            "factor_entries": MAX_FACTOR_ENTRIES,
            "site_updates": MAX_SITE_UPDATES,
            "estimated_bytes": MAX_ESTIMATED_BYTES,
        },
        "authoritative_limit": "execution_process_watchdog",
    }


def _declared_dimensions(spec: dict[str, Any]) -> None:
    """Reject lossy dimension coercions before shared canonicalization."""
    keys = (
        "num_hidden_states",
        "num_states",
        "num_obs",
        "num_observations",
        "num_actions",
        "num_controls",
    )
    if not isinstance(spec, dict):
        raise ValueError("THRML source must be a mapping")
    parameters = spec.get("model_parameters", {})
    if parameters is None:
        parameters = {}
    for values in (spec, parameters):
        if not isinstance(values, dict):
            raise ValueError("model_parameters must be a mapping")
        for key in keys:
            if key in values:
                _integer(values[key], key)


def _component_admission(
    spec: dict[str, Any], schedule: dict[str, int], *, synthetic: bool
) -> dict[str, Any]:
    """Bound source retention and native factors before creating arrays."""
    _declared_dimensions(spec)
    raw = spec.get("initialparameterization") or spec.get("initial_parameterization")
    if not isinstance(raw, dict):
        raise ValueError("THRML requires declared categorical matrices")
    shape = nested_shape(raw.get("A", []))
    if len(shape) != 2:
        raise ValueError("THRML requires a two-dimensional canonical likelihood")
    b_shape = nested_shape(raw.get("B", []))
    if len(b_shape) == 3 and not _declared_b_order(spec.get("model_parameters") or {}):
        raise UnsupportedTHRMLModel(
            "three-dimensional B requires explicit source tensor orientation"
        )
    return _admission(shape[1], shape[0], schedule, synthetic=synthetic, parameters=raw)


def _matrices(spec: dict[str, Any]) -> dict[str, Any]:
    initial = (
        spec.get("initialparameterization")
        or spec.get("initial_parameterization")
        or {}
    )
    structured = (spec.get("structured_pomdp") or {}).get("matrices") or {}
    return {**structured, **initial}


def _component_option(
    options: dict[str, Any], key: str, name: str, names: list[str]
) -> Any:
    value = options.get(key)
    if isinstance(value, dict):
        if set(value) != set(names):
            raise ValueError(f"{key} must bind every selected component exactly")
        return value[name]
    if len(names) > 1 and value is not None:
        raise ValueError(f"{key} for multiple components requires a component mapping")
    return value


def _indices(value: Any, length: int, categories: int, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} integer indices")
    return [_integer(item, name, minimum=0, maximum=categories - 1) for item in value]


def build_thrml_payload(
    gnn_spec: dict[str, Any], options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate all represented components before allocating a native program."""
    from gnn.render.thrml.composition import (
        _compose_factors,
        _compose_modalities,
        _independent_specs,
        _joint_metadata,
    )

    options = dict(options or {})
    if set(options) - OPTION_KEYS:
        raise ValueError(f"unknown THRML options: {sorted(set(options) - OPTION_KEYS)}")
    _declared_dimensions(gnn_spec)
    kinds = detect_model_kinds(gnn_spec)
    excluded = {
        ModelKind.CONTINUOUS,
        ModelKind.HYBRID,
        ModelKind.NONSTATIONARY,
        ModelKind.HIERARCHICAL,
        ModelKind.LEARNING,
        ModelKind.STRUCTURAL,
    }
    if kinds & excluded:
        raise UnsupportedTHRMLModel(
            f"unsupported model kinds for THRML: {sorted(kind.value for kind in kinds & excluded)}"
        )
    if "control_mode" in options or "horizon" in options:
        raise UnsupportedTHRMLModel(
            "THRML fixed-action sampling does not optimize policies/EFE"
        )
    matrices = _matrices(gnn_spec)
    semantics = {
        **gnn_spec,
        **(gnn_spec.get("structured_pomdp") or {}),
        **(gnn_spec.get("model_parameters") or {}),
        **matrices,
    }
    if any(
        str(key)
        .lower()
        .startswith(
            (
                "env_",
                "environment",
                "coupling",
                "signal_",
                "shared_environment",
                "communication",
            )
        )
        for key in semantics
    ):
        raise UnsupportedTHRMLModel(
            "coupling/environment/signal semantics require a verified native joint mapping"
        )
    schedule = _schedule(gnn_spec, options)
    provenance = (
        gnn_spec.get("matrix_provenance")
        or (gnn_spec.get("structured_pomdp") or {}).get("matrix_provenance")
        or {}
    )
    canonical_joint = (
        all(key in matrices for key in ("A", "B", "C", "D"))
        and (provenance.get("A") or {}).get("source") == "factored_joint_composition"
    )
    independent = {} if canonical_joint else _independent_specs(gnn_spec, matrices)
    if independent:
        component_specs = independent
        composition = "independent_components"
    else:
        if ModelKind.MULTI_AGENT in kinds and not canonical_joint:
            raise UnsupportedTHRMLModel(
                "multi-agent models require native independent groups or a bound canonical joint composition"
            )
        if canonical_joint or all(key in matrices for key in ("A", "B", "C", "D")):
            if not canonical_joint and any(
                re.match(r"^[ABCDE]_", str(key)) for key in matrices
            ):
                raise UnsupportedTHRMLModel(
                    "flat categorical mapping cannot omit additional declared matrices"
                )
            selected = gnn_spec
        elif any(re.fullmatch(r"B_f\d+", str(key)) for key in matrices):
            selected = _compose_factors(gnn_spec, matrices, schedule)
        else:
            selected = _compose_modalities(gnn_spec, matrices, schedule)
        component_specs = {"model": selected}
        composition = (
            "canonical_joint"
            if canonical_joint or selected.get("thrml_composition")
            else "flat"
        )
    if len(component_specs) > MAX_COMPONENTS:
        raise ValueError("THRML component admission limit exceeded")
    names = list(component_specs)
    # Admit the complete corpus of components before any canonical array copy.
    admissions = {
        name: _component_admission(
            spec,
            schedule,
            synthetic=_component_option(options, "observations", name, names) is None,
        )
        for name, spec in component_specs.items()
    }
    total = {
        key: sum(item[key] for item in admissions.values())
        for key in (
            "factor_entries",
            "site_updates",
            "estimated_bytes",
            "retained_matrix_entries",
        )
    }
    if (
        total["factor_entries"] > MAX_FACTOR_ENTRIES
        or total["retained_matrix_entries"] > MAX_FACTOR_ENTRIES
        or total["site_updates"] > MAX_SITE_UPDATES
        or total["estimated_bytes"] > MAX_ESTIMATED_BYTES
    ):
        raise ValueError("THRML aggregate component admission limit exceeded")
    components = []
    for name, spec in component_specs.items():
        raw = (
            spec.get("initialparameterization")
            or spec.get("initial_parameterization")
            or {}
        )
        shape = nested_shape(raw.get("A", []))
        if len(shape) != 2:
            raise ValueError("THRML requires a two-dimensional canonical likelihood")
        requested_observations = _component_option(options, "observations", name, names)
        admission = admissions[name]
        canonical = build_canonical_pomdp_spec(spec)
        initial = canonical["initialparameterization"]
        for key in ("A", "B", "D"):
            values = np.asarray(initial[key], dtype=np.float64)
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise UnsupportedTHRMLModel(
                    f"structural-zero {name}/{key} support is unsupported by the verified stock Gibbs mapping; no clipping or normalization repair"
                )
        states, observations = shape[1], shape[0]
        controls = canonical["model_parameters"]["num_actions"]
        transitions = _component_option(options, "transition_actions", name, names)
        if transitions is None:
            transitions = [0] * max(schedule["num_timesteps"] - 1, 0)
            action_origin = "explicit_default_fixed_action_0"
        else:
            action_origin = "provided_fixed_transition_actions"
        transitions = _indices(
            transitions,
            max(schedule["num_timesteps"] - 1, 0),
            controls,
            "transition_actions",
        )
        observed = (
            None
            if requested_observations is None
            else _indices(
                requested_observations,
                schedule["num_timesteps"],
                observations,
                "observations",
            )
        )
        metadata = _joint_metadata(spec)
        for key, expected in (
            ("state_factors", states),
            ("observation_modalities", observations),
        ):
            descriptors = metadata.get(key) or []
            if (
                descriptors
                and math.prod(
                    _integer(item.get("size"), f"{key} size") for item in descriptors
                )
                != expected
            ):
                raise ValueError(
                    f"{key} do not bind the composed categorical dimension"
                )
        components.append(
            {
                "component_id": name,
                "parameters": initial,
                "source_parameters": deepcopy(raw),
                "matrix_provenance": canonical["matrix_provenance"],
                "source_model_parameters": deepcopy(
                    gnn_spec.get("model_parameters") or {}
                ),
                "composed_model_parameters": deepcopy(canonical["model_parameters"]),
                "num_states": states,
                "num_observations": observations,
                "observations": observed,
                "transition_actions": transitions,
                "action_origin": action_origin,
                "admission": admission,
                "composition_metadata": metadata,
                "source_structure": deepcopy(gnn_spec.get("structured_pomdp") or {}),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "framework": "thrml",
        "experimental": True,
        "model_kind": "discrete",
        "source_model_kinds": sorted(kind.value for kind in kinds),
        "model_name": gnn_spec.get("model_name", "GNN model"),
        "source_model_parameters": deepcopy(gnn_spec.get("model_parameters") or {}),
        "composition": composition,
        "components": components,
        "sampling": schedule,
        "sampling_dtype": "float64",
        "source_identity": {
            "semantic_sha256": contract_digest(
                {
                    "components": components,
                    "sampling": schedule,
                    "posterior_semantics": "fixed_actions_full_sequence_smoothing",
                }
            ),
            "claim": "parameter/mapping identity; extraction and Monte Carlo accuracy require separate witnesses",
        },
        "resource_admission": total,
        "inference_mode": "gibbs_monte_carlo",
        "posterior_semantics": "fixed_actions_full_sequence_smoothing",
        "predictive_semantics": "replicate_observation_given_smoothed_state",
        "action_semantics": "transition_into_next_state",
        "limitations": {
            "C_E": "preserved preferences/habits; conditioned fixed actions, no EFE/policy optimizer",
            "structural_zeros": "unsupported without verified zero-safe, ergodic mapping",
            "continuous": "no verified continuous THRML adapter",
            "hardware": "local JAX simulation; no hardware energy/speed claim",
        },
    }
    return payload
