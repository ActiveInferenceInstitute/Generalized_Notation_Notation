"""Released THRML categorical factor programs for finite-sequence smoothing.

Only aggregation uses NumPy. Every categorical draw is made by THRML's
``sample_states`` over a ``FactorSamplingProgram``. Hardware execution,
convergence, exact inference and action optimization are separate claims.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from gnn.render.thrml.adapter import (
    MAX_COMPONENTS,
    MAX_ESTIMATED_BYTES,
    MAX_FACTOR_ENTRIES,
    MAX_SITE_UPDATES,
    THRML_VERSION,
    _component_admission,
    _indices,
    _integer,
    contract_digest,
)


def _validate_payload(payload: dict[str, Any]) -> None:
    """Recheck serialized admission and value binding before native imports."""
    from gnn.render.pomdp_contract import build_canonical_pomdp_spec

    required_semantics = {
        "schema_version": 1,
        "framework": "thrml",
        "model_kind": "discrete",
        "inference_mode": "gibbs_monte_carlo",
        "posterior_semantics": "fixed_actions_full_sequence_smoothing",
        "predictive_semantics": "replicate_observation_given_smoothed_state",
        "action_semantics": "transition_into_next_state",
        "sampling_dtype": "float64",
        "experimental": True,
    }
    if any(payload.get(key) != value for key, value in required_semantics.items()):
        raise ValueError("THRML serialized mapping semantics mismatch")

    components = payload.get("components")
    if not isinstance(components, list) or not 1 <= len(components) <= MAX_COMPONENTS:
        raise ValueError("invalid THRML component list")
    schedule = payload.get("sampling") or {}
    for key in ("num_timesteps", "num_samples", "thin"):
        _integer(schedule.get(key), key)
    _integer(schedule.get("burn_in"), "burn_in", minimum=0)
    _integer(schedule.get("seed"), "seed", minimum=0, maximum=2**32 - 1)
    expected = contract_digest(
        {
            "components": components,
            "sampling": schedule,
            "posterior_semantics": "fixed_actions_full_sequence_smoothing",
        }
    )
    if (payload.get("source_identity") or {}).get("semantic_sha256") != expected:
        raise ValueError("THRML serialized semantic identity mismatch")
    totals = {
        "factor_entries": 0,
        "site_updates": 0,
        "estimated_bytes": 0,
        "retained_matrix_entries": 0,
    }
    names = set()
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("THRML components must be mappings")
        name = component.get("component_id")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("THRML component identities must be unique")
        names.add(name)
        params = component.get("parameters") or {}
        from gnn.render.pomdp_contract import nested_shape

        b_shape = nested_shape(params.get("B", []))
        if len(b_shape) != 3:
            raise ValueError("THRML serialized B must have canonical rank three")
        admission = _component_admission(
            {
                "initialparameterization": params,
                "model_parameters": {
                    "num_states": component.get("num_states"),
                    "num_obs": component.get("num_observations"),
                    "num_actions": b_shape[-1],
                    "b_tensor_order": "next_state_previous_state_action",
                },
            },
            schedule,
            synthetic=component.get("observations") is None,
        )
        if component.get("admission") != admission:
            raise ValueError("THRML serialized component admission mismatch")
        for key in totals:
            totals[key] += admission[key]
    if (
        totals["factor_entries"] > MAX_FACTOR_ENTRIES
        or totals["retained_matrix_entries"] > MAX_FACTOR_ENTRIES
        or totals["site_updates"] > MAX_SITE_UPDATES
        or totals["estimated_bytes"] > MAX_ESTIMATED_BYTES
    ):
        raise ValueError("THRML aggregate resource admission rejected")
    if payload.get("resource_admission") != totals:
        raise ValueError("THRML serialized resource admission mismatch")
    # Only after aggregate admission may canonicalization allocate arrays.
    for component in components:
        params = component["parameters"]
        canonical = build_canonical_pomdp_spec(
            {
                "initialparameterization": params,
                "model_parameters": {
                    "num_states": component.get("num_states"),
                    "num_obs": component.get("num_observations"),
                    "num_actions": np.asarray(params.get("B")).shape[-1],
                    "b_tensor_order": "next_state_previous_state_action",
                },
            }
        )
        states = len(canonical["initialparameterization"]["D"])
        observations = len(canonical["initialparameterization"]["A"])
        if states != component.get("num_states") or observations != component.get(
            "num_observations"
        ):
            raise ValueError("THRML serialized category dimensions disagree")
        for key in ("A", "B", "C", "D", "E"):
            if key in params and not np.allclose(
                np.asarray(params[key]),
                np.asarray(canonical["initialparameterization"][key]),
                rtol=2e-15,
                atol=0,
            ):
                raise ValueError(
                    "THRML serialized parameters must already be canonical"
                )
        for key in ("A", "B", "D"):
            if not np.isfinite(np.asarray(params[key])).all() or np.any(
                np.asarray(params[key]) <= 0
            ):
                raise ValueError(
                    "THRML verified mapping requires strictly positive finite probabilities"
                )
        _indices(
            component.get("transition_actions"),
            schedule["num_timesteps"] - 1,
            np.asarray(params["B"]).shape[-1],
            "transition_actions",
        )
        if component.get("observations") is not None:
            _indices(
                component["observations"],
                schedule["num_timesteps"],
                observations,
                "observations",
            )


def _native_samples(
    component: dict[str, Any],
    schedule: dict[str, int],
    key: Any,
    native: tuple[Any, Any, Any, Any],
) -> tuple[np.ndarray, list[int], dict[str, Any]]:
    jax, jnp, thrml, models = native
    timesteps, states, observations = (
        schedule["num_timesteps"],
        component["num_states"],
        component["num_observations"],
    )
    parameters = component["parameters"]
    a = jnp.asarray(parameters["A"], dtype=jnp.float64)
    b = jnp.asarray(parameters["B"], dtype=jnp.float64)
    d = jnp.asarray(parameters["D"], dtype=jnp.float64)
    state_nodes = [thrml.CategoricalNode() for _ in range(timesteps)]
    observation_nodes = [thrml.CategoricalNode() for _ in range(timesteps)]
    all_states, all_observations = (
        thrml.Block(state_nodes),
        thrml.Block(observation_nodes),
    )
    free = [thrml.Block(state_nodes[::2])]
    if state_nodes[1::2]:
        free.append(thrml.Block(state_nodes[1::2]))
    factors = [
        models.CategoricalEBMFactor(
            [thrml.Block([state_nodes[0]])], jnp.log(d)[None, :]
        )
    ]
    if timesteps > 1:
        transition_weights = jnp.stack(
            [jnp.log(b[:, :, action]) for action in component["transition_actions"]]
        )
        factors.append(
            models.CategoricalEBMFactor(
                [thrml.Block(state_nodes[1:]), thrml.Block(state_nodes[:-1])],
                transition_weights,
            )
        )
    factors.append(
        models.CategoricalEBMFactor(
            [all_observations, all_states],
            jnp.broadcast_to(jnp.log(a), (timesteps, observations, states)),
        )
    )
    node_types = {thrml.CategoricalNode: jax.ShapeDtypeStruct((), jnp.uint16)}
    initial = [jnp.zeros((len(block.nodes),), dtype=jnp.uint16) for block in free]
    sampling_metadata: dict[str, Any] = {
        "factor_count": len(factors),
        "free_state_blocks": len(free),
        "categorical_index_dtype": "uint16",
        "backend_api": "thrml.FactorSamplingProgram/sample_states",
    }
    supplied = component.get("observations")
    synthetic_key, posterior_key = jax.random.split(key)
    if supplied is None:
        generative_blocks = free + [all_observations]
        generative = thrml.FactorSamplingProgram(
            thrml.BlockGibbsSpec(generative_blocks, [], node_shape_dtypes=node_types),
            [models.CategoricalGibbsConditional(states) for _ in free]
            + [models.CategoricalGibbsConditional(observations)],
            factors,
            [],
        )
        joint_draw = thrml.sample_states(
            synthetic_key,
            generative,
            thrml.SamplingSchedule(schedule["burn_in"], 1, schedule["thin"]),
            initial + [jnp.zeros((timesteps,), dtype=jnp.uint16)],
            [],
            [all_states, all_observations],
        )
        generated_states = _indices(
            np.asarray(joint_draw[0])[0].tolist(), timesteps, states, "synthetic_states"
        )
        supplied = _indices(
            np.asarray(joint_draw[1])[0].tolist(),
            timesteps,
            observations,
            "synthetic_observations",
        )
        sampling_metadata.update(
            data_origin="synthetic_gibbs_joint_draw",
            generated_state_trajectory=generated_states,
            synthetic_draw_claim="finite-burn-in THRML joint sample; no exact independent simulator claim",
        )
    else:
        sampling_metadata["data_origin"] = "provided_observation_sequence"
    posterior = thrml.FactorSamplingProgram(
        thrml.BlockGibbsSpec(free, [all_observations], node_shape_dtypes=node_types),
        [models.CategoricalGibbsConditional(states) for _ in free],
        factors,
        [],
    )
    sampled = thrml.sample_states(
        posterior_key,
        posterior,
        thrml.SamplingSchedule(
            schedule["burn_in"], schedule["num_samples"], schedule["thin"]
        ),
        initial,
        [jnp.asarray(supplied, dtype=jnp.uint16)],
        [all_states],
    )
    trajectories = np.asarray(
        sampled[0]
    )  # Synchronizes native sampling before reporting completion.
    if (
        trajectories.shape != (schedule["num_samples"], timesteps)
        or not np.issubdtype(trajectories.dtype, np.integer)
        or np.any(trajectories < 0)
        or np.any(trajectories >= states)
    ):
        raise ValueError(
            "THRML native sampler returned malformed categorical trajectories"
        )
    return trajectories, list(supplied), sampling_metadata


def _marginals(
    values: np.ndarray, descriptors: list[dict[str, Any]]
) -> dict[str, list[list[float]]]:
    if not descriptors:
        return {}
    sizes = [int(item["size"]) for item in descriptors]
    if np.prod(sizes) != values.shape[1]:
        raise ValueError("source descriptor cardinalities do not bind joint result")
    tensor = values.reshape((values.shape[0], *sizes))
    output = {}
    for index, descriptor in enumerate(descriptors):
        name = str(descriptor.get("name") or f"factor_{index}")
        if name in output:
            raise ValueError("source marginal names must be unique")
        axes = tuple(axis + 1 for axis in range(len(sizes)) if axis != index)
        output[name] = (tensor.sum(axis=axes) if axes else tensor).tolist()
    return output


def run_payload(
    payload: dict[str, Any],
    *,
    execution_id: str | None = None,
    script_sha256: str | None = None,
) -> dict[str, Any]:
    """Execute a validated payload using the released native THRML program."""
    started = time.monotonic()
    _validate_payload(payload)
    version = importlib.metadata.version("thrml")
    if version != THRML_VERSION:
        raise RuntimeError(f"THRML {THRML_VERSION} required; installed {version}")
    jax = importlib.import_module("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = importlib.import_module("jax.numpy")
    thrml = importlib.import_module("thrml")
    models = importlib.import_module("thrml.models")
    dependencies = {
        name: importlib.metadata.version(name)
        for name in ("thrml", "jax", "jaxlib", "equinox", "jaxtyping", "numpy")
    }
    key = jax.random.key(payload["sampling"]["seed"])
    shared = {
        name: deepcopy(payload[name])
        for name in (
            "schema_version",
            "framework",
            "experimental",
            "model_name",
            "model_kind",
            "source_model_kinds",
            "source_model_parameters",
            "composition",
            "source_identity",
            "sampling",
            "sampling_dtype",
            "resource_admission",
            "inference_mode",
            "posterior_semantics",
            "predictive_semantics",
            "action_semantics",
            "limitations",
        )
    }
    shared.update(
        success=True,
        num_timesteps=payload["sampling"]["num_timesteps"],
        dependencies=dependencies,
        validation={
            "mapping_supported": True,
            "all_conditionals_valid": True,
            "basis": "strictly positive finite A/B/D yields finite full-support categorical conditionals",
        },
        runtime_metadata={
            "execution_id": execution_id,
            "script_sha256": script_sha256,
            "execution_plane": "local_jax_simulation",
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "devices": [str(device) for device in jax.devices()],
            "python": sys.version.split()[0],
        },
    )
    results = {}
    for index, component in enumerate(payload["components"]):
        component_started = time.monotonic()
        trajectories, observed, native_metadata = _native_samples(
            component,
            payload["sampling"],
            jax.random.fold_in(key, index),
            (jax, jnp, thrml, models),
        )
        beliefs = np.stack(
            [
                np.bincount(trajectories[:, step], minlength=component["num_states"])
                / len(trajectories)
                for step in range(shared["num_timesteps"])
            ]
        )
        predictive = (
            beliefs @ np.asarray(component["parameters"]["A"], dtype=np.float64).T
        )
        metadata = component["composition_metadata"]
        child = {
            **deepcopy(shared),
            **{
                name: deepcopy(component[name])
                for name in (
                    "component_id",
                    "parameters",
                    "source_parameters",
                    "matrix_provenance",
                    "source_model_parameters",
                    "composed_model_parameters",
                    "transition_actions",
                    "action_origin",
                    "composition_metadata",
                    "source_structure",
                )
            },
            "observations": observed,
            "sample_states": trajectories.astype(int).tolist(),
            "beliefs": beliefs.tolist(),
            "predictive_observations": predictive.tolist(),
            "native_sampling": native_metadata,
            "state_factor_marginals": _marginals(
                beliefs, metadata.get("state_factors") or []
            ),
            "observation_modality_marginals": _marginals(
                predictive,
                metadata.get("observation_modalities")
                or metadata.get("modalities")
                or [],
            ),
            "elapsed_seconds": time.monotonic() - component_started,
        }
        results[component["component_id"]] = child
    result = (
        {**shared, "components": results}
        if payload["composition"] == "independent_components"
        else results[next(iter(results))]
    )
    result["elapsed_seconds"] = time.monotonic() - started
    usage = {
        "admission_estimate_bytes": payload["resource_admission"]["estimated_bytes"]
    }
    try:
        resource = importlib.import_module("resource")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        usage.update(
            process_peak_rss_bytes=int(
                peak * (1 if sys.platform == "darwin" else 1024)
            ),
            measurement="process lifetime ru_maxrss; includes imports/JIT, not an allocation estimate",
        )
    except ImportError:
        usage["measurement_unavailable_reason"] = (
            "platform does not provide resource.ru_maxrss; executor watchdog measures process resources separately"
        )
    result["resource_usage"] = usage
    return result


def write_results(result: dict[str, Any], destination: Path) -> None:
    """Publish complete JSON atomically; failed inference never writes success."""
    from gnn.render.naming import atomic_write_text

    atomic_write_text(
        destination,
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def run_generated_script(payload: dict[str, Any], script_path: str) -> None:
    """Small generated-script entry point with fresh execution identity."""
    import argparse
    import hashlib

    parser = argparse.ArgumentParser(
        description="Experimental THRML fixed-action categorical smoothing"
    )
    managed_output = os.environ.get("THRML_OUTPUT_DIR")
    default_destination = (
        Path(managed_output) / "simulation_data" / "simulation_results.json"
        if managed_output
        else Path("thrml_results.json")
    )
    parser.add_argument("--output-json", type=Path, default=default_destination)
    args = parser.parse_args()
    result = run_payload(
        payload,
        execution_id=os.environ.get("GNN_THRML_EXECUTION_ID"),
        script_sha256=hashlib.sha256(Path(script_path).read_bytes()).hexdigest(),
    )
    write_results(result, args.output_json)
    print("THRML_RESULTS " + json.dumps(result, separators=(",", ":"), allow_nan=False))
