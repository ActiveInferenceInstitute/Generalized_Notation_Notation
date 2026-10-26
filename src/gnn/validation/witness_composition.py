"""Replayable numerical witnesses for two independent passive Gaussian agents."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .semantic_witness import (
    _artifact,
    _candidate_validity,
    _declared_checks,
    _identity,
    _outcome,
    _quantity,
    _row,
)
from .witness_composition_source import AGENTS, joint_gaussian_contract
from .witness_source import _array, _json, condition_joint_gaussian

SCHEMA = "gnn-semantic-witness-v3"


def _child_checks(
    child: dict[str, Any],
    component: dict[str, Any],
    observations: Any,
    program: dict[str, str] | None,
) -> dict[str, dict[str, Any]]:
    count = component["model_parameters"]["num_timesteps"]
    identity = {
        "dtype": "float64",
        "initial_time": "first_observation_uses_prior",
        "axes": component["axes"],
        "conditioning_horizon": list(range(1, count + 1)),
        "num_timesteps": count,
        "num_states": len(component["state_ids"]),
        "num_observations": len(component["observation_ids"]),
        "random_seed": component["model_parameters"]["random_seed"],
        "inference_contract": "gnn_lgssm_filter_v1",
        "inference_mode": "gaussian_filtering",
        "control_mode": "passive",
        "framework": "jax",
        "schema_version": "gnn_lgssm_filter_v1",
        "model_kind": "continuous",
    }
    checks = {
        f"identity:{key}": _identity(value, child.get(key))
        for key, value in identity.items()
    }
    checks.update(_declared_checks(child, component))
    checks["identity:framework"] = _identity("jax", child.get("framework"))
    checks.update(_candidate_validity(child, component))
    validation = child.get("validation")
    checks["identity:validation.all_valid"] = _identity(
        True, validation.get("all_valid") if isinstance(validation, dict) else None
    )
    checks["observations"] = _quantity(
        observations, child.get("observations_continuous"), exact=True
    )
    checks["controls"] = _quantity(
        np.zeros((count, len(component["state_ids"]))).tolist(),
        child.get("controls"),
        exact=True,
    )
    parameters = child.get("parameters")
    checks["identity:parameter_keys"] = _identity(
        sorted(component["parameters"]),
        sorted(parameters) if isinstance(parameters, dict) else None,
    )
    parameters = parameters if isinstance(parameters, dict) else {}
    for key, expected in component["parameters"].items():
        checks[f"parameter:{key}"] = _quantity(
            expected, parameters.get(key), exact=True
        )
    if program is not None:
        checks["identity:generated_script_sha256"] = _identity(
            program["sha256"], child.get("generated_script_sha256")
        )
    return checks


def build_composition_witness(
    source: bytes, result: bytes, program: bytes | None, contract: dict[str, Any]
) -> dict[str, Any]:
    """Compare actual child results against independent joint conditioning.

    Zero cross-agent covariance is a declared reference-model consequence, not
    a native measurement. Neither supplied program bytes nor agreeing arrays
    authenticate execution or establish a formal translation proof.
    """
    result_artifact = _artifact(result)
    artifacts = {
        "source": _artifact(source),
        "result": result_artifact,
        "program": _artifact(program) if program is not None else None,
    }
    payload = _json(result_artifact["utf8"])
    if not isinstance(payload, dict):
        raise ValueError("composition native result must be a JSON object")
    observations = payload.get("observations_continuous_by_agent")
    if not isinstance(observations, dict) or set(observations) != set(AGENTS):
        raise ValueError(
            "composition requires explicit observations for exactly both agents"
        )
    count = contract["model_parameters"]["num_timesteps"]
    observed = []
    for agent in AGENTS:
        values = _array(observations[agent], f"observations.{agent}")
        expected_shape = (count, len(contract["components"][agent]["observation_ids"]))
        if values.shape != expected_shape:
            raise ValueError(
                f"observations.{agent}: source horizon/dimensions disagree"
            )
        observed.append(values)
    joint_observations = np.concatenate(observed, axis=1).tolist()
    joint = joint_gaussian_contract(contract)
    means, covariance = condition_joint_gaussian(joint, joint_observations)
    outer_identity = {
        "schema_version": "gnn_multi_agent_continuous_v1",
        "model_kind": "multi_agent_continuous",
        "framework": "jax",
        "agents": list(AGENTS),
        "agent_coupling": "independent",
        "num_timesteps": count,
    }
    comparisons = {
        f"identity:{key}": _identity(expected, payload.get(key))
        for key, expected in outer_identity.items()
    }
    common = {
        key: payload[key]
        for key in (
            "validation",
            "status",
            "success",
            "source_sha256",
            "source_identity",
        )
        if key in payload
    }
    common["framework"] = payload.get("framework")
    comparisons.update(_declared_checks(common, contract["components"]["agent1"]))
    comparisons["identity:framework"] = _identity("jax", payload.get("framework"))
    for key, expected in {
        "dtype": "float64",
        "inference_mode": "independent_gaussian_filtering",
        "inference_contract": "gnn_independent_gaussian_filter_v1",
        "control_mode": "passive",
        "state_ids": contract["state_ids"],
        "observation_ids": contract["observation_ids"],
    }.items():
        if key in payload:
            comparisons[f"identity:{key}"] = _identity(expected, payload[key])
    for key in ("beliefs", "posterior_cov", "controls", "observations_continuous"):
        projection = payload.get(key + "_by_agent")
        comparisons[f"identity:{key}_projection_agents"] = _identity(
            list(AGENTS), sorted(projection) if isinstance(projection, dict) else None
        )
    validation = payload.get("validation")
    comparisons["identity:validation.all_valid"] = _identity(
        True, validation.get("all_valid") if isinstance(validation, dict) else None
    )
    records = payload.get("agent_results")
    records_valid = isinstance(records, dict) and set(records) == set(AGENTS)
    comparisons["identity:agent_results"] = _row(
        "passed" if records_valid else "unavailable" if records is None else "refused",
        ""
        if records_valid
        else "complete native child records for exactly both agents required",
    )
    records = records if isinstance(records, dict) else {}
    reference_components = {}
    candidate_components = {}
    for agent in AGENTS:
        child = records.get(agent)
        child = child if isinstance(child, dict) else {}
        component = contract["components"][agent]
        checks = _child_checks(
            child, component, observations[agent], artifacts["program"]
        )
        for key in ("beliefs", "posterior_cov", "controls", "observations_continuous"):
            projected = payload.get(key + "_by_agent")
            projected = projected.get(agent) if isinstance(projected, dict) else None
            if child.get(key) is None:
                checks[f"projection:{key}"] = _row(
                    "unavailable", "native child quantity is missing"
                )
            else:
                checks[f"projection:{key}"] = _quantity(
                    child[key], projected, exact=True
                )
        comparisons.update({f"{agent}:{key}": row for key, row in checks.items()})
    admission = _outcome(comparisons)
    start = 0
    for agent in AGENTS:
        component = contract["components"][agent]
        end = start + len(component["state_ids"])
        quantities = {
            "beliefs": means[:, start:end].tolist(),
            "posterior_cov": covariance[:, start:end, start:end].tolist(),
        }
        child = records.get(agent)
        child = child if isinstance(child, dict) else {}
        candidate_components[agent] = deepcopy(child)
        reference_components[agent] = {
            "observations": observations[agent],
            "quantities": quantities,
            "state_ids": component["state_ids"],
            "observation_ids": component["observation_ids"],
        }
        for key, expected in quantities.items():
            row = _quantity(expected, child.get(key))
            if admission != "passed":
                row["numeric_status"] = row["status"]
                row["status"] = (
                    "refused" if admission in {"refused", "failed"} else "unavailable"
                )
                row["reason"] = "composition identity/parameter admission did not pass"
            comparisons[f"{agent}:{key}"] = row
        start = end
    return {
        "schema": SCHEMA,
        "status": _outcome(comparisons),
        "source_contract": contract,
        "reference": {
            "algorithm": "joint_block_diagonal_gaussian_prefix_conditioning",
            "conditioning_horizon": list(range(1, count + 1)),
            "observations": joint_observations,
            "quantities": {
                "beliefs": means.tolist(),
                "posterior_cov": covariance.tolist(),
            },
            "components": reference_components,
            "cross_agent_covariance": "zero_by_declared_product_model; not a measured native quantity",
        },
        "candidate": {"agents": list(AGENTS), "components": candidate_components},
        "comparisons": comparisons,
        "assumptions": {
            "question": "filtering",
            "agent_coupling": "independent",
            "control_mode": "passive",
            "initial_time": "first_observation_uses_prior",
            "conditioning_horizon": list(range(1, count + 1)),
            "state_ids": contract["state_ids"],
            "observation_ids": contract["observation_ids"],
            "axes": contract["axes"],
            "label_mapping": "Source labels name ordered numeric axes; native label provenance is not authenticated.",
            "scope": "Exactly two independent passive stationary Gaussian agents; float64; at most eight times and four total state/observation axes; no coupling, shared noise, control, smoothing or universal equivalence.",
        },
        "artifacts": artifacts,
        "formal_evidence": {"status": "unavailable", "universal_equivalence": False},
        "execution_evidence": {
            "status": "artifact_only",
            "execution_source_verified": False,
            "reported_framework": payload.get("framework"),
        },
    }
