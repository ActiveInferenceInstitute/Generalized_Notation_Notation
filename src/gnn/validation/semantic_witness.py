"""Replayable source-bound numerical comparisons, separate from execution/proof.

Only explicitly declared bounded authored source subsets are admitted. Candidate metadata
comes from the result artifact; absent declarations are never copied from the
source. Verification rebuilds the complete receipt from its retained bytes.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

import numpy as np

from .witness_source import (
    _array,
    _json,
    condition_joint_gaussian,
    enumerate_finite,
    read_source,
)

SCHEMA = "gnn-semantic-witness-v2"
ATOL = 1e-12
RTOL = 1e-10
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
ROW_MASS_ATOL = 1e-12
COVARIANCE_SYMMETRY_FACTOR = 64
MAX_VALIDATION_NODES = 4096
MAX_VALIDATION_DEPTH = 32
MAX_VALIDATION_PATH_CHARS = 2048


def _artifact(value: bytes) -> dict[str, str]:
    if len(value) > MAX_ARTIFACT_BYTES:
        raise ValueError("artifact byte budget exceeded")
    return {"sha256": hashlib.sha256(value).hexdigest(), "utf8": value.decode("utf-8")}


def _row(status: str, reason: str = "", **fields: Any) -> dict[str, Any]:
    return {"status": status, "reason": reason, **fields}


def _quantity(expected: Any, actual: Any, *, exact: bool = False) -> dict[str, Any]:
    if actual is None:
        return _row("unavailable", "required native quantity is missing")
    try:
        reference, measured = _array(expected, "reference"), _array(actual, "candidate")
    except (ValueError, TypeError, OverflowError) as exc:
        return _row("refused", str(exc))
    if measured.shape != reference.shape:
        return _row(
            "refused",
            "quantity shape mismatch",
            expected_shape=list(reference.shape),
            actual_shape=list(measured.shape),
        )
    atol, rtol = (0.0, 0.0) if exact else (ATOL, RTOL)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        difference = np.abs(measured - reference)
        scale = np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
        relative = difference / scale
    maximum = float(difference.max())
    maximum_relative = float(relative.max())
    # Valid finite operands can overflow subtraction; JSON never stores NaN/Inf.
    if not np.isfinite(maximum) or not np.isfinite(maximum_relative):
        return _row(
            "failed", "difference exceeds finite diagnostic range", atol=atol, rtol=rtol
        )
    passed = bool(np.all(difference <= atol + rtol * np.abs(reference)))
    return _row(
        "passed" if passed else "failed",
        "" if passed else "outside declared tolerance",
        max_abs_error=maximum,
        max_rel_error=maximum_relative,
        atol=atol,
        rtol=rtol,
        shape=list(reference.shape),
    )


def _identity(expected: Any, actual: Any) -> dict[str, Any]:
    if actual is None:
        return _row(
            "unavailable",
            "required native semantic declaration is missing",
            expected=expected,
            actual=None,
        )
    # JSON equality alone conflates booleans, integer indices and floats.
    same = type(actual) is type(expected) and json.dumps(
        actual, sort_keys=True
    ) == json.dumps(expected, sort_keys=True)
    return _row(
        "passed" if same else "refused",
        "" if same else "incompatible scientific identity",
        expected=expected,
        actual=actual,
    )


def _outcome(comparisons: dict[str, dict[str, Any]]) -> str:
    states = {item["status"] for item in comparisons.values()}
    return next(
        (state for state in ("refused", "failed", "unavailable") if state in states),
        "passed",
    )


def _declared_checks(
    payload: dict[str, Any], contract: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Refuse contradictory native declarations without inventing absent labels."""
    finite = contract["model"] == "finite"
    expected = {
        "source_sha256": contract["source_sha256"],
        "state_ids": contract["state_ids"],
        "observation_ids": contract["observation_ids"],
        "model_kind": "discrete" if finite else "continuous",
        "schema_version": "bnlearn_fixed_action_filter_v1"
        if finite
        else "gnn_lgssm_filter_v1",
        "success": True,
    }
    checks = {
        f"identity:{key}": _identity(value, payload[key])
        for key, value in expected.items()
        if key in payload
    }
    if "source_identity" in payload:
        identity = payload["source_identity"]
        if not isinstance(identity, dict):
            checks["identity:source_identity"] = _row(
                "refused", "native source_identity must be a mapping"
            )
        elif "source_sha256" in identity:
            checks["identity:source_identity.source_sha256"] = _identity(
                contract["source_sha256"], identity["source_sha256"]
            )
    frameworks = ("bnlearn",) if finite else ("jax", "numpyro", "pytorch", "ngclearn")
    framework = payload.get("framework")
    checks["identity:framework"] = _row(
        "unavailable"
        if framework is None
        else "passed"
        if framework in frameworks
        else "refused",
        ""
        if framework in frameworks
        else "native framework is missing or outside this result contract",
        expected=list(frameworks),
        actual=framework,
    )
    if "status" in payload:
        status = payload["status"]
        passed = isinstance(status, str) and status.lower() in {"success", "completed"}
        checks["identity:status"] = _row(
            "passed" if passed else "refused",
            "" if passed else "native result did not declare terminal success",
            expected=["success", "completed"],
            actual=status,
        )
    if "validation" in payload:
        validation = payload["validation"]
        if not isinstance(validation, dict):
            checks["identity:validation"] = _row(
                "refused", "native validation must be a mapping"
            )
        else:
            if "all_valid" in validation:
                checks["identity:validation.all_valid"] = _identity(
                    True, validation["all_valid"]
                )
            pending: list[tuple[str, Any, int]] = [("validation", validation, 0)]
            queued = 1
            exhausted = False
            while pending:
                path, value, depth = pending.pop()
                if isinstance(value, dict):
                    children = ((f".{key}", child) for key, child in value.items())
                elif isinstance(value, list):
                    children = (
                        (f"[{index}]", child) for index, child in enumerate(value)
                    )
                else:
                    if value is False:
                        checks[f"identity:{path}"] = _row(
                            "refused",
                            "native result declares a validation failure",
                            expected=True,
                            actual=False,
                        )
                    continue
                for suffix, child in children:
                    queued += 1
                    if (
                        queued > MAX_VALIDATION_NODES
                        or depth + 1 > MAX_VALIDATION_DEPTH
                        or len(path) + len(suffix) > MAX_VALIDATION_PATH_CHARS
                    ):
                        checks["identity:validation_budget"] = _row(
                            "refused",
                            "native validation exceeds node, depth or path budget",
                        )
                        exhausted = True
                        break
                    pending.append((path + suffix, child, depth + 1))
                if exhausted:
                    break
    return checks


def _candidate_validity(
    payload: dict[str, Any], contract: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Validate native distributions before approximate numerical agreement.

    The admitted source prior/Q/R are positive definite, so every native
    posterior covariance must remain strictly positive definite. Covariance
    symmetry admits only 64 float64 eps times max(1, largest absolute entry);
    values are never symmetrized, clipped, normalized or otherwise repaired.
    """
    t = contract["model_parameters"]["num_timesteps"]
    n = len(contract["state_ids"])
    checks = {}
    for name, shape in (("beliefs", (t, n)), ("posterior_cov", (t, n, n))):
        if name == "posterior_cov" and contract["model"] == "finite":
            continue
        value = payload.get(name)
        if value is None:
            checks[f"validity:{name}"] = _row(
                "unavailable", "required native quantity is missing"
            )
            continue
        try:
            values = _array(value, name)
            if values.shape != shape:
                raise ValueError(f"{name}: expected shape {shape}")
            if name == "beliefs" and contract["model"] == "finite":
                if (
                    np.any(values < 0)
                    or np.any(values > 1)
                    or not np.allclose(
                        values.sum(axis=1), 1, rtol=0, atol=ROW_MASS_ATOL
                    )
                ):
                    raise ValueError(
                        "beliefs: probabilities in [0,1] and categorical rows with unit mass required"
                    )
            elif name == "posterior_cov":
                tolerance = (
                    COVARIANCE_SYMMETRY_FACTOR
                    * np.finfo(np.float64).eps
                    * np.maximum(
                        1.0, np.max(np.abs(values), axis=(1, 2), keepdims=True)
                    )
                )
                if np.any(np.abs(values - np.swapaxes(values, 1, 2)) > tolerance):
                    raise ValueError(
                        "posterior_cov: symmetric covariance required within float64 machine tolerance"
                    )
                if np.any(np.diagonal(values, axis1=1, axis2=2) <= 0):
                    raise ValueError(
                        "posterior_cov: strictly positive variances required"
                    )
                try:
                    np.linalg.cholesky(values)
                except np.linalg.LinAlgError as exc:
                    raise ValueError(
                        "posterior_cov: strictly positive definite covariance required"
                    ) from exc
            checks[f"validity:{name}"] = _row("passed")
        except (ValueError, TypeError, OverflowError) as exc:
            checks[f"validity:{name}"] = _row("refused", str(exc))
    return checks


def build_semantic_witness(
    source: bytes, result: bytes, program: bytes | None = None
) -> dict[str, Any]:
    """Independently read a source and compare one identified native result.

    This process performs no native execution or Lean compilation. Hash binding
    of supplied bytes cannot authenticate how a result file was produced.
    """
    source_artifact, result_artifact = _artifact(source), _artifact(result)
    program_artifact = _artifact(program) if program is not None else None
    contract = read_source(source)
    if contract["model"] == "gaussian_product":
        from .witness_composition import build_composition_witness

        return build_composition_witness(source, result, program, contract)
    payload = _json(result_artifact["utf8"])
    if not isinstance(payload, dict):
        raise ValueError("native result must be a JSON object")
    finite = contract["model"] == "finite"
    observations = payload.get("observations" if finite else "observations_continuous")
    actions = payload.get("transition_actions") if finite else []
    if not isinstance(observations, list) or not isinstance(actions, list):
        raise ValueError(
            "native observations and transition actions must be explicit lists"
        )
    if finite:
        quantities = {
            "beliefs": enumerate_finite(contract, observations, actions).tolist()
        }
    else:
        means, covariances = condition_joint_gaussian(contract, observations)
        quantities = {"beliefs": means.tolist(), "posterior_cov": covariances.tolist()}
    t = contract["model_parameters"]["num_timesteps"]
    identity = {
        "dtype": "float64",
        "initial_time": contract["initial_time"],
        "axes": contract["axes"],
        "conditioning_horizon": [t] * t
        if contract["question"] == "smoothing"
        else list(range(1, t + 1)),
        "num_timesteps": t,
        "num_states": len(contract["state_ids"]),
        "num_observations": len(contract["observation_ids"]),
        "random_seed": contract["model_parameters"]["random_seed"],
        "inference_contract": "bnlearn_fixed_action_filter_v1"
        if finite
        else "gnn_lgssm_filter_v1",
        "inference_mode": "exact_categorical_filtering"
        if finite
        else "gaussian_filtering",
        "control_mode": "fixed_transition_actions" if finite else "passive",
    }
    comparisons = {
        f"identity:{key}": _identity(value, payload.get(key))
        for key, value in identity.items()
    }
    comparisons.update(_declared_checks(payload, contract))
    comparisons.update(_candidate_validity(payload, contract))
    if contract["question"] != "filtering":
        comparisons["identity:question"] = _row(
            "refused",
            "native adapter admits filtering only; smoothing requires a separate result contract",
        )
    if finite:
        comparisons["identity:action_semantics"] = _identity(
            "transition_into_next_state", payload.get("action_semantics")
        )
    else:
        comparisons["controls"] = _quantity(
            np.zeros((t, len(contract["state_ids"]))).tolist(),
            payload.get("controls"),
            exact=True,
        )
    native_parameters = payload.get("parameters")
    native_parameters = native_parameters if isinstance(native_parameters, dict) else {}
    for key, expected in contract["parameters"].items():
        if finite and key == "C":
            continue  # fixed exogenous-action filtering has no preference query
        comparisons[f"parameter:{key}"] = _quantity(
            expected, native_parameters.get(key), exact=True
        )
    if program_artifact is not None:
        comparisons["identity:generated_script_sha256"] = _identity(
            program_artifact["sha256"], payload.get("generated_script_sha256")
        )
    admission = _outcome(comparisons)
    for key, expected in quantities.items():
        row = _quantity(expected, payload.get(key))
        if admission != "passed":
            # Retain a diagnostic difference without calling incompatible values agreement.
            row["numeric_status"] = row["status"]
            row["status"] = (
                "refused" if admission in ("refused", "failed") else "unavailable"
            )
            row["reason"] = (
                "semantic/parameter admission did not pass; see identity and parameter rows"
            )
        comparisons[key] = row
    candidate = {key: deepcopy(payload.get(key)) for key in identity}
    candidate.update(
        observations=observations,
        transition_actions=actions,
        quantities={key: deepcopy(payload.get(key)) for key in quantities},
    )
    reference = {
        "algorithm": "finite_complete_path_enumeration"
        if finite
        else "joint_gaussian_prefix_conditioning",
        "quantities": quantities,
        "observations": observations,
        "transition_actions": actions,
        "conditioning_horizon": identity["conditioning_horizon"],
    }
    return {
        "schema": SCHEMA,
        "status": _outcome(comparisons),
        "source_contract": contract,
        "reference": reference,
        "candidate": candidate,
        "comparisons": comparisons,
        "assumptions": {
            "question": contract["question"],
            "initial_time": contract["initial_time"],
            "state_ids": contract["state_ids"],
            "observation_ids": contract["observation_ids"],
            "axes": contract["axes"],
            "conditioning_horizon": identity["conditioning_horizon"],
            "label_mapping": "Ordered source labels name numeric axes; native label provenance is not authenticated.",
            "scope": "One stationary factor, fixed exogenous actions or passive LGSSM; float64; at most eight times/four dimensions; no planning, coupled/hybrid semantics or universal equivalence.",
        },
        "artifacts": {
            "source": source_artifact,
            "result": result_artifact,
            "program": program_artifact,
        },
        "formal_evidence": {"status": "unavailable", "universal_equivalence": False},
        "execution_evidence": {
            "status": "artifact_only",
            "execution_source_verified": False,
            "reported_framework": payload.get("framework"),
            "reported_versions": {
                key: payload[key]
                for key in (
                    "python_version",
                    "numpy_version",
                    "jax_version",
                    "bnlearn_version",
                    "torch_version",
                )
                if key in payload
            },
        },
    }


def verify_semantic_witness(receipt: object) -> list[str]:
    """Recompute the entire receipt, including references, rather than trust flags.

    An intact failed/refused record can be replay-valid. Callers must additionally
    require ``status == 'passed'`` when accepting numerical agreement.
    """
    try:
        if not isinstance(receipt, dict):
            return ["semantic witness receipt must be a mapping"]
        if receipt.get("schema") not in (SCHEMA, "gnn-semantic-witness-v3"):
            return ["unsupported semantic witness schema"]
        artifacts = receipt["artifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != {
            "source",
            "result",
            "program",
        }:
            return [
                "semantic witness artifacts must contain source, result and program"
            ]
        for name in ("source", "result", "program"):
            artifact = artifacts[name]
            if name == "program" and artifact is None:
                continue
            if (
                not isinstance(artifact, dict)
                or set(artifact) != {"utf8", "sha256"}
                or not isinstance(artifact["utf8"], str)
                or not isinstance(artifact["sha256"], str)
            ):
                return [
                    f"semantic witness {name} artifact requires UTF-8 text and digest strings"
                ]
        source = artifacts["source"]["utf8"].encode("utf-8")
        result = artifacts["result"]["utf8"].encode("utf-8")
        program = (
            artifacts["program"]["utf8"].encode("utf-8")
            if artifacts["program"] is not None
            else None
        )
        expected = build_semantic_witness(source, result, program)
        if json.dumps(expected, sort_keys=True, allow_nan=False) != json.dumps(
            receipt, sort_keys=True, allow_nan=False
        ):
            return ["semantic witness differs from independent raw-artifact replay"]
        return []
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        RecursionError,
    ) as exc:
        return [f"semantic witness replay refused: {exc}"]
