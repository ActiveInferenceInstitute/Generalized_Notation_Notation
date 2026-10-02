"""Validate finite discrete Gibbs traces without inventing inference metrics."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from gnn.analysis.result_adapter import categorical_trace

THRML_VERSION = "0.1.4"


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _indices(value: Any, name: str, shape: tuple[int, ...], width: int) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != shape
        or (array.size and array.dtype.kind not in "iu")
        or np.any(array < 0)
        or np.any(array >= width)
    ):
        raise ValueError(f"{name} must contain integer indices with shape {shape}")
    # A mixed boolean/integer JSON array otherwise erases boolean provenance.
    flattened = [value] if len(shape) == 1 else value
    if any(isinstance(item, (bool, np.bool_)) for row in flattened for item in row):
        raise ValueError(f"{name} must not contain Boolean indices")
    return array


def _marginal_views(
    values: np.ndarray,
    descriptors: Any,
    retained: Any,
    name: str,
) -> dict[str, list[list[float]]]:
    """Verify each named projection against the retained joint sample witness."""
    if not isinstance(descriptors, list) or not isinstance(retained, dict):
        raise ValueError(f"THRML {name} requires descriptor and marginal mappings")
    if not descriptors:
        if retained:
            raise ValueError(f"THRML {name} lacks source descriptors")
        return {}
    sizes: list[int] = []
    names: list[str] = []
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ValueError(f"THRML {name} descriptors must be objects")
        label = descriptor.get("name")
        if not isinstance(label, str) or not label or label in names:
            raise ValueError(f"THRML {name} requires unique source names")
        names.append(label)
        sizes.append(_integer(descriptor.get("size"), f"{name} size", 1))
    # Python integer multiplication cannot overflow a NumPy scalar here.
    cardinality = 1
    for size in sizes:
        cardinality *= size
    if cardinality != values.shape[1] or set(retained) != set(names):
        raise ValueError(f"THRML {name} source dimensions or names disagree")
    tensor = values.reshape((len(values), *sizes))
    verified = {}
    for index, label in enumerate(names):
        axes = tuple(axis + 1 for axis in range(len(sizes)) if axis != index)
        expected = tensor.sum(axis=axes) if axes else tensor
        marginal = categorical_trace(retained[label], name=f"THRML {name}:{label}")
        if marginal.shape != expected.shape or not np.allclose(
            marginal, expected, rtol=1e-6, atol=1e-8
        ):
            raise ValueError(f"THRML {name}:{label} differs from its joint projection")
        verified[label] = marginal.tolist()
    return verified


def adapt_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the declared smoothing interpretation and retained sample witness.

    Gibbs samples describe an approximate posterior conditional on the entire
    observation sequence and fixed actions. Entropy describes the empirical
    posterior; neither calibration nor convergence follows from that entropy.
    """
    if not isinstance(payload, dict):
        raise ValueError("THRML results must be a JSON object")
    if payload.get("model_kind") != "discrete":
        raise ValueError("THRML continuous/structured mappings are unsupported")
    if (
        _integer(payload.get("schema_version"), "schema_version", 1) != 1
        or payload.get("framework") != "thrml"
        or payload.get("success") is not True
        or payload.get("inference_mode") != "gibbs_monte_carlo"
        or payload.get("posterior_semantics") != "fixed_actions_full_sequence_smoothing"
        or payload.get("predictive_semantics")
        != "replicate_observation_given_smoothed_state"
        or payload.get("action_semantics") != "transition_into_next_state"
        or payload.get("sampling_dtype") != "float64"
    ):
        raise ValueError("THRML results require the declared Gibbs smoothing contract")
    validation = payload.get("validation")
    if (
        not isinstance(validation, dict)
        or validation.get("mapping_supported") is not True
        or validation.get("all_conditionals_valid") is not True
    ):
        raise ValueError("THRML mapping/conditional validity was not established")
    identity = payload.get("source_identity")
    if not isinstance(identity, dict):
        raise ValueError("THRML semantic source identity must be an object")
    semantic_digest = identity.get("semantic_sha256")
    if (
        not isinstance(semantic_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", semantic_digest) is None
    ):
        raise ValueError(
            "THRML semantic source identity requires a lowercase SHA256 digest"
        )
    if (
        identity.get("claim")
        != "parameter/mapping identity; extraction and Monte Carlo accuracy require separate witnesses"
    ):
        raise ValueError(
            "THRML semantic identity must preserve its bounded evidence claim"
        )
    runtime = payload.get("runtime_metadata")
    if not isinstance(runtime, dict) or runtime.get("jax_enable_x64") is not True:
        raise ValueError(
            "THRML runtime identity must affirm the float64/x64 sampling policy"
        )
    # This applies to an enclosing component receipt as well as each marginal.
    # Dispatch must not make unsupported global numerical claims disappear.
    for name in (
        "variational_free_energy",
        "vfe_per_iteration",
        "vfe_history",
        "expected_free_energy",
        "efe_per_action",
        "efe_history",
        "posterior_cov",
        "calibration",
        "effective_sample_size",
        "converged",
    ):
        if name in payload:
            raise ValueError(f"THRML {name} is not part of the verified mapping")
    if "components" in payload:
        components = payload["components"]
        if (
            payload.get("composition") != "independent_components"
            or not isinstance(components, dict)
            or not components
            or "beliefs" in payload
        ):
            raise ValueError(
                "THRML components require declared independent composition without a fabricated joint posterior"
            )
        inherited = {
            key: payload[key]
            for key in (
                "schema_version",
                "framework",
                "success",
                "model_kind",
                "inference_mode",
                "posterior_semantics",
                "predictive_semantics",
                "action_semantics",
                "sampling_dtype",
                "sampling",
                "validation",
                "source_identity",
                "dependencies",
                "runtime_metadata",
            )
            if key in payload
        }
        summaries = {}
        timesteps = _integer(payload.get("num_timesteps"), "num_timesteps", 1)
        sampling = payload.get("sampling")
        if (
            not isinstance(sampling, dict)
            or _integer(sampling.get("num_timesteps"), "sampling num_timesteps", 1)
            != timesteps
        ):
            raise ValueError(
                "THRML component sampling schedule must match its timestep count"
            )
        for name, component in components.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(component, dict)
                or component.get("component_id") != name
                or "components" in component
            ):
                raise ValueError(
                    "THRML component identities must match their flat native results"
                )
            if any(
                key in component and component[key] != value
                for key, value in inherited.items()
            ):
                raise ValueError(
                    "THRML component contract metadata differs from its enclosing receipt"
                )
            summary = adapt_result({**inherited, **component})
            if summary["metrics"]["num_timesteps"] != timesteps:
                raise ValueError(
                    "THRML independent components must share their timestep count"
                )
            summaries[name] = summary
        return {
            **inherited,
            "experimental": True,
            "composition": "independent_components",
            "model_name": payload.get("model_name"),
            "metrics": {"num_timesteps": timesteps, "num_components": len(summaries)},
            "components": summaries,
            "unavailable_metrics": {
                "joint_posterior": "Independent native component marginals remain separate",
                "coupled_inference": "No verified coupled component mapping was reported",
            },
            "measured_resources": payload.get("measured_resources"),
            "resource_usage": payload.get("resource_usage"),
            "resource_admission": payload.get("resource_admission"),
            "source_model_kinds": payload.get("source_model_kinds"),
            "source_model_parameters": payload.get("source_model_parameters"),
            "composed_model_parameters": payload.get("composed_model_parameters"),
            "sampling": payload.get("sampling"),
            "sampling_dtype": payload.get("sampling_dtype"),
            "limitations": payload.get("limitations"),
        }
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("THRML results must retain their model parameters")
    likelihood = np.asarray(parameters.get("A"), dtype=float)
    transition = np.asarray(parameters.get("B"), dtype=float)
    prior = np.asarray(parameters.get("D"), dtype=float)
    preferences = np.asarray(parameters.get("C"), dtype=float)
    habits = np.asarray(parameters["E"], dtype=float) if "E" in parameters else None
    if likelihood.ndim != 2 or min(likelihood.shape) < 1:
        raise ValueError("THRML A must have shape (observations, states)")
    observations, states = likelihood.shape
    if (
        transition.ndim != 3
        or transition.shape[:2] != (states, states)
        or transition.shape[2] < 1
        or prior.shape != (states,)
        or preferences.shape != (observations,)
        or (habits is not None and habits.shape != (transition.shape[2],))
    ):
        raise ValueError("THRML parameter dimensions or canonical B axes disagree")
    probabilities = [
        ("A", likelihood),
        ("B", transition),
        ("D", prior),
    ]
    if habits is not None:
        probabilities.append(("E", habits))
    for name, values in probabilities:
        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or (name != "E" and np.any(values == 0))
        ):
            raise ValueError(
                f"THRML {name} requires positive finite probabilities; structural zeros unsupported"
            )
        if not np.allclose(values.sum(axis=0), 1, rtol=1e-6, atol=1e-8):
            raise ValueError(f"THRML {name} probability mass is invalid")
    if not np.isfinite(preferences).all():
        raise ValueError("THRML preferences must be finite")
    beliefs = categorical_trace(payload.get("beliefs"), name="THRML beliefs")
    prediction = categorical_trace(
        payload.get("predictive_observations"), name="THRML predictive observations"
    )
    timesteps = len(beliefs)
    if beliefs.shape[1] != states or prediction.shape != (timesteps, observations):
        raise ValueError("THRML posterior and predictive dimensions disagree")
    if _integer(payload.get("num_timesteps"), "num_timesteps", 1) != timesteps:
        raise ValueError("THRML num_timesteps must match posterior length")
    if not np.allclose(prediction, beliefs @ likelihood.T, rtol=1e-6, atol=1e-8):
        raise ValueError(
            "THRML predictive trace does not match its declared replicate semantics"
        )
    sampling = payload.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("THRML sampling provenance is required")
    count = _integer(sampling.get("num_samples"), "num_samples", 1)
    if (
        _integer(sampling.get("num_timesteps"), "sampling num_timesteps", 1)
        != timesteps
    ):
        raise ValueError("THRML sampling num_timesteps must match posterior length")
    _integer(sampling.get("burn_in"), "burn_in")
    _integer(sampling.get("thin"), "thin", 1)
    if _integer(sampling.get("seed"), "seed") > 2**32 - 1:
        raise ValueError("THRML seed must fit its uint32 native key")
    samples = _indices(
        payload.get("sample_states"), "sample_states", (count, timesteps), states
    )
    _indices(payload.get("observations"), "observations", (timesteps,), observations)
    _indices(
        payload.get("transition_actions"),
        "transition_actions",
        (timesteps - 1,),
        transition.shape[2],
    )
    empirical = np.stack(
        [np.bincount(samples[:, t], minlength=states) / count for t in range(timesteps)]
    )
    if not np.allclose(beliefs, empirical, rtol=1e-6, atol=1e-8):
        raise ValueError("THRML beliefs do not match the retained sample witness")
    composition_metadata = payload.get("composition_metadata") or {}
    if not isinstance(composition_metadata, dict):
        raise ValueError("THRML composition metadata must be an object")
    state_marginals = _marginal_views(
        beliefs,
        composition_metadata.get("state_factors") or [],
        payload.get("state_factor_marginals", {}),
        "state factor marginals",
    )
    observation_marginals = _marginal_views(
        prediction,
        composition_metadata.get("observation_modalities")
        or composition_metadata.get("modalities")
        or [],
        payload.get("observation_modality_marginals", {}),
        "observation modality marginals",
    )
    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, dict) or dependencies.get("thrml") != THRML_VERSION:
        raise ValueError(
            "THRML released dependency identity is missing or incompatible"
        )
    unavailable = {
        "variational_free_energy": "Gibbs sampling reports no variational objective",
        "expected_free_energy": "Fixed transition actions perform no policy search",
        "calibration": "No independent calibration experiment was reported",
        "effective_sample_size": "No validated autocorrelation diagnostic was reported",
        "convergence": "A finite Gibbs sample count alone does not establish convergence",
        "continuous_covariance": "Only the verified discrete mapping is supported",
    }
    entropy = -(beliefs * np.log(np.maximum(beliefs, np.finfo(float).tiny))).sum(axis=1)
    return {
        "schema_version": 1,
        "framework": "thrml",
        "experimental": True,
        "model_kind": "discrete",
        "inference_mode": "gibbs_monte_carlo",
        "posterior_semantics": payload["posterior_semantics"],
        "predictive_semantics": payload["predictive_semantics"],
        "action_semantics": payload["action_semantics"],
        "model_name": payload.get("model_name"),
        "metrics": {
            "num_timesteps": timesteps,
            "num_states": states,
            "num_observations": observations,
            "retained_samples": count,
            "mean_empirical_entropy_nats": float(entropy.mean()),
            "mean_empirical_max_probability": float(beliefs.max(axis=1).mean()),
        },
        "unavailable_metrics": unavailable,
        "source_identity": payload["source_identity"],
        "dependencies": dependencies,
        "sampling": sampling,
        "runtime_metadata": runtime,
        "matrix_provenance": payload.get("matrix_provenance"),
        "measured_resources": payload.get("measured_resources"),
        "resource_usage": payload.get("resource_usage"),
        "limitations": payload.get("limitations"),
        "composition": payload.get("composition"),
        "component_id": payload.get("component_id"),
        "composition_metadata": composition_metadata,
        "state_factor_marginals": state_marginals,
        "observation_modality_marginals": observation_marginals,
        "resource_admission": payload.get("resource_admission"),
        "sampling_dtype": payload.get("sampling_dtype"),
        "native_sampling": payload.get("native_sampling"),
        "source_structure": payload.get("source_structure"),
        "source_model_kinds": payload.get("source_model_kinds"),
        "source_model_parameters": payload.get("source_model_parameters"),
        "composed_model_parameters": payload.get("composed_model_parameters"),
    }
