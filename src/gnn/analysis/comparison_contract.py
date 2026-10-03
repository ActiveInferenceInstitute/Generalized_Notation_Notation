"""Source-bound numerical comparisons, separate from operational aggregates.

Equal shapes and display names do not establish model or inference identity.
The JSON payload must declare its floating point precision; comparisons use
the least precise declared dtype (float32: rtol=1e-5/atol=1e-6, float64:
rtol=1e-8/atol=1e-10). These are numerical witnesses, not equivalence proofs.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np

from .framework_common import CURRENT_SIMULATION_SCHEMAS
from .result_adapter import (
    categorical_trace,
    continuous_result_metrics,
    model_family,
    result_views,
)

_FIELDS = {
    "model_id": ("model_id",),
    "source_sha256": ("source_sha256", "source_hash"),
    "model_kind": ("model_kind",),
    "inference_mode": ("inference_mode",),
    "dtype": ("dtype", "sampling_dtype"),
    "source_relative_path": ("source_relative_path",),
    "input_sha256": ("input_sha256", "input_hash"),
    "run_id": ("run_id",),
    "posterior_semantics": ("posterior_semantics",),
    "action_semantics": ("action_semantics",),
    "expected_free_energy_convention": ("expected_free_energy_convention",),
    "free_energy_kind": ("free_energy_kind",),
    "free_energy_convention": ("free_energy_convention",),
    "variational_free_energy_convention": ("variational_free_energy_convention",),
    "input_semantics": ("input_semantics",),
}
_ALIGNMENT = (
    "observations",
    "observations_by_agent",
    "observations_by_factor",
    "observations_by_modality",
    "observations_continuous",
    "observations_continuous_by_agent",
    "actions",
    "actions_by_agent",
    "actions_by_factor",
    "actions_by_control_factor",
    "controls",
    "controls_by_agent",
    "true_states",
    "true_states_by_agent",
    "true_states_by_factor",
    "hidden_states_by_factor",
    "true_states_continuous",
    "true_states_continuous_by_agent",
    "timesteps",
    "time_index",
    "seed",
    "axes",
    "belief_axes",
    "covariance_axes",
    "state_labels",
    "matrix_provenance",
    "source_model_parameters",
    "composed_model_parameters",
    "model_parameters",
    "simulation_parameters",
    "configuration",
    "agent_coupling",
    "factor_order",
    "state_factors",
    "factor_cardinalities",
    "free_energy_index",
    "expected_free_energy_index",
    "variational_free_energy_index",
    "free_energy_axes",
    "expected_free_energy_axes",
    "variational_free_energy_axes",
)


def _equal(left: Any, right: Any) -> bool:
    """Compare structured identity/alignment data without flattening it."""
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_equal(left[key], right[key]) for key in left)
        )
    try:
        a, b = np.asarray(left), np.asarray(right)
        return a.shape == b.shape and bool(np.array_equal(a, b))
    except (TypeError, ValueError):
        return False


def _present(value: Any) -> bool:
    return value is not None and not (
        isinstance(value, (list, dict, tuple, str)) and len(value) == 0
    )


def _family(payload: dict[str, Any]) -> str:
    runtime = payload.get("runtime_metadata", {})
    return model_family(
        {**payload, "runtime_metadata": runtime if isinstance(runtime, dict) else {}}
    )


def _validate_source_parameters(parameters: Any, orientation: Any = None) -> None:
    """Check supplied source probabilities in their declared axes, unchanged."""
    if not isinstance(parameters, dict):
        raise ValueError("source parameters must be a mapping")
    for key, value in parameters.items():
        if isinstance(value, dict):
            _validate_source_parameters(value, orientation)
            continue
        base = key.split("_", 1)[0]
        if base in {"F", "H", "Q", "R", "C"} or key in {
            "prior_mean",
            "prior_cov",
            "goal_mean",
            "control_gain",
        }:
            array = np.asarray(value, dtype=float)
            if not np.isfinite(array).all():
                raise ValueError(f"source {key} must contain finite values")
            if base in {"Q", "R"} or key == "prior_cov":
                if (
                    array.ndim != 2
                    or array.shape[0] != array.shape[1]
                    or not array.size
                    or not np.allclose(array, array.T, rtol=0, atol=1e-8)
                    or np.any(np.linalg.eigvalsh(array) < -1e-8)
                ):
                    raise ValueError(
                        f"source {key} must be a symmetric positive semidefinite covariance"
                    )
            continue
        if base not in {"A", "B", "D", "E"}:
            continue
        array = np.asarray(value, dtype=float)
        if base in {"D", "E"}:
            if array.ndim != 1:
                raise ValueError(f"source {key} must be a probability vector")
            categorical_trace(array[None, :], name=f"source {key}")
        elif base == "A":
            if array.ndim < 2:
                raise ValueError(f"source {key} must have a leading observation axis")
            rows = np.moveaxis(array, 0, -1)
            categorical_trace(rows.reshape(-1, rows.shape[-1]), name=f"source {key}")
        else:
            if (
                array.ndim not in {2, 3}
                or not np.isfinite(array).all()
                or np.any(array < 0)
            ):
                raise ValueError(
                    f"source {key} must contain finite nonnegative transition probabilities"
                )
            if array.ndim == 2:
                categorical_trace(array.T, name=f"source {key}")
            else:
                order = parameters.get("b_tensor_order") or parameters.get(
                    "B_tensor_order"
                )
                if order is None and isinstance(orientation, dict):
                    provenance = orientation.get("matrix_provenance", {}).get(key, {})
                    order = provenance.get("source_order") or orientation.get(
                        "model_parameters", {}
                    ).get("b_tensor_order")
                axes = {
                    "next_state_previous_state_action": 0,
                    "action_next_state_previous_state": 1,
                    "action_previous_state_next_state": 2,
                }
                if order not in axes:
                    raise ValueError(f"source {key} requires declared transition axes")
                rows = np.moveaxis(array, axes[order], -1)
                categorical_trace(
                    rows.reshape(-1, rows.shape[-1]), name=f"source {key}"
                )


def _failed_validation(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            (
                item is False
                and (
                    key
                    in {
                        "all_valid",
                        "valid",
                        "success",
                        "covariance_psd",
                        "posterior_cov_psd",
                        "symmetric_covariance",
                    }
                    or key.endswith(
                        ("_valid", "_finite", "_sum_to_one", "_in_range", "_normalized")
                    )
                )
            )
            or (isinstance(item, dict) and _failed_validation(item))
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_failed_validation(item) for item in value)
    return False


def _validate_view_dimensions(
    payload: dict[str, Any], views: dict[str, dict[str, Any]]
) -> None:
    params = payload.get("model_parameters", {})
    source = payload.get("source_model_parameters", {})
    if not isinstance(params, dict) or not isinstance(source, dict):
        raise ValueError("model/source parameters must be mappings")
    for name, view in views.items():
        width = np.asarray(view["beliefs"]).shape[1]
        for key in ("time_index", "timesteps"):
            value = payload.get(key)
            if (
                key == "timesteps"
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                if value != len(view["beliefs"]):
                    raise ValueError(
                        "timesteps count must align with posterior trace length"
                    )
            elif _present(value):
                index = np.asarray(value, dtype=float)
                if (
                    index.ndim != 1
                    or len(index) != len(view["beliefs"])
                    or not np.isfinite(index).all()
                ):
                    raise ValueError(f"{key} must align with posterior trace length")
        dimensions: list[Any] = []
        if len(views) == 1:
            dimensions.extend(
                params.get(key)
                for key in ("num_states", "num_hidden_states")
                if key in params
            )
            for key in ("D", "prior_mean"):
                if key in source:
                    dimensions.append(len(source[key]))
        for key in ("num_states_by_agent", "num_states_by_factor"):
            table = params.get(key, {})
            if isinstance(table, dict) and name in table:
                dimensions.append(table[name])
        if f"D_{name}" in source:
            dimensions.append(len(source[f"D_{name}"]))
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value != width
            for value in dimensions
        ):
            raise ValueError(
                f"posterior width for {name} conflicts with source/declared dimensions"
            )


def _validate_posterior_axes(payload: dict[str, Any]) -> None:
    axes = payload.get("belief_axes")
    grouped_axes = payload.get("axes", {})
    if not isinstance(grouped_axes, dict):
        raise ValueError("axes must be a mapping of reported quantities")
    if isinstance(grouped_axes, dict) and "beliefs" in grouped_axes:
        if axes is not None and not _equal(axes, grouped_axes["beliefs"]):
            raise ValueError("conflicting posterior axes declarations")
        axes = grouped_axes["beliefs"]
    if axes is not None and not (
        _equal(axes, ["timestep", "state"])
        or (
            _family(payload) in {"continuous", "multi_agent_continuous"}
            and _equal(axes, ["timestep", "dimension"])
        )
    ):
        raise ValueError("posterior axes must declare timestep then state/dimension")
    covariance_axes = payload.get("covariance_axes", grouped_axes.get("posterior_cov"))
    if covariance_axes is not None and not any(
        _equal(covariance_axes, candidate)
        for candidate in (
            ["timestep", "state", "state"],
            ["timestep", "state_row", "state_column"],
            ["timestep", "dimension", "dimension"],
        )
    ):
        raise ValueError(
            "covariance axes must declare timestep then state rows/columns"
        )
    for quantity in ("free_energy", "expected_free_energy", "variational_free_energy"):
        quantity_axes = payload.get(f"{quantity}_axes")
        if quantity_axes is not None and not any(
            _equal(quantity_axes, [dimension])
            for dimension in ("timestep", "iteration", "policy")
        ):
            raise ValueError(
                f"{quantity}_axes must declare one scalar series dimension"
            )


def prepare_scientific_record(
    result: dict[str, Any], binding: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Retain one result's native views and reject conflicting identity claims.

    Step 12 bindings can supply source identity, but a payload's contradictory
    declaration is never overwritten. Missing custody prevents comparison;
    individually validated statistics remain available without that claim.
    """
    layers = [result]
    for key in ("simulation_data", "simulation_trace"):
        if isinstance(result.get(key), dict):
            layers.append(result[key])
    if binding is not None:
        layers.append(binding)
    metadata_layers = list(layers)
    for layer in layers:
        for key in (
            "source_identity",
            "runtime_metadata",
            "execution_metadata",
            "model_parameters",
        ):
            if isinstance(layer.get(key), dict):
                metadata_layers.append(layer[key])
    metadata: dict[str, Any] = {}
    reasons: list[str] = []
    for name, aliases in _FIELDS.items():
        values = [
            layer[key]
            for layer in metadata_layers
            for key in aliases
            if _present(layer.get(key))
        ]
        if values:
            if any(not _equal(values[0], value) for value in values[1:]):
                reasons.append(f"conflicting {name} declarations")
            metadata[name] = values[0]
    payload: dict[str, Any] = {}
    for layer in reversed(layers[: len(layers) - (binding is not None)]):
        payload.update(layer)
    for layer in layers:
        for key in ("source_identity", "runtime_metadata", "execution_metadata"):
            if key in layer and not isinstance(layer[key], dict):
                reasons.append(f"invalid {key}: metadata must be a mapping")
    for key in _ALIGNMENT:
        values = [layer[key] for layer in metadata_layers if _present(layer.get(key))]
        if values:
            if any(not _equal(values[0], value) for value in values[1:]):
                reasons.append(f"conflicting {key} declarations")
            payload[key] = values[0]
    for name in (
        "model_kind",
        "inference_mode",
        "dtype",
        "posterior_semantics",
        "action_semantics",
    ):
        if name in metadata:
            payload[name] = metadata[name]
    if not _present(payload.get("beliefs")):
        joint = payload.get("beliefs_by_factor", {})
        if isinstance(joint, dict) and list(joint) == ["joint_state"]:
            payload["beliefs"] = joint["joint_state"]
    for target, table, entry in (
        ("actions", "actions_by_control_factor", "joint_action"),
        ("observations", "observations_by_modality", "joint_observation"),
    ):
        if not _present(payload.get(target)) and isinstance(payload.get(table), dict):
            payload[target] = payload[table].get(entry, [])
    record: dict[str, Any] = {
        "metadata": metadata,
        "payload": payload,
        "reasons": reasons,
        "statistics": {},
        "binding": {
            key: value
            for key, value in (binding or {}).items()
            if key in _ALIGNMENT
            or any(key in aliases for aliases in _FIELDS.values())
            or key
            in {
                "source_identity",
                "runtime_metadata",
                "execution_metadata",
                "model_parameters",
            }
        },
    }
    try:
        _validate_posterior_axes(payload)
        if _failed_validation(payload.get("validation", {})):
            reasons.append("reported scientific validation failure")
        for key in ("source_model_parameters", "composed_model_parameters"):
            if _present(payload.get(key)):
                _validate_source_parameters(payload[key], payload)
        declared_kind = metadata.get("model_kind")
        family = _family(payload)
        known = {
            "flat",
            "discrete",
            "continuous",
            "multi_agent_continuous",
            "multi_agent",
            "factored",
            "hierarchical",
            "nonstationary",
            "learning",
        }
        if declared_kind is not None and declared_kind not in known:
            reasons.append("unsupported declared model_kind")
        if (
            declared_kind in known
            and family != declared_kind
            and not (
                declared_kind == "discrete"
                and family
                in {
                    "flat",
                    "multi_agent",
                    "factored",
                    "hierarchical",
                    "learning",
                    "nonstationary",
                }
            )
        ):
            reasons.append("model_kind conflicts with native posterior family")
        # Transition actions can legitimately have T-1 entries. The adapter's
        # categorical per-timestep check otherwise expects T; retain the actual
        # actions for pairwise alignment after validating this explicit contract.
        view_payload = dict(payload)
        if not isinstance(view_payload.get("runtime_metadata", {}), dict):
            view_payload.pop("runtime_metadata", None)
        actions = payload.get("actions")
        if (
            _present(actions)
            and payload.get("action_semantics") == "transition_into_next_state"
        ):
            assert actions is not None
            beliefs = payload.get("beliefs", [])
            if len(actions) != len(beliefs) - 1:
                raise ValueError("transition actions must have T-1 entries")
            view_payload.pop("actions", None)
        views = result_views(view_payload)
        _validate_view_dimensions(payload, views)
        statistics: dict[str, Any] = {}
        for name, view in views.items():
            beliefs = np.asarray(view["beliefs"], dtype=float)
            if model_family(view) == "continuous":
                stats = continuous_result_metrics(view)
                stats["final_posterior_mean"] = beliefs[-1].tolist()
                stats["unavailable_metrics"] = {
                    "mean_confidence": "Gaussian means are not categorical probabilities"
                }
            else:
                stats = {
                    "belief_dims": list(beliefs.shape),
                    "mean_confidence": float(np.max(beliefs, axis=1).mean()),
                    "final_belief": beliefs[-1].tolist(),
                }
            statistics[name] = stats
        record["views"] = views
        record["statistics"] = (
            next(iter(statistics.values()))
            if len(statistics) == 1
            else {"views": statistics}
        )
        if not views:
            reasons.append("no validated posterior views")
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
        reasons.append(f"invalid scientific result: {exc}")
        record["views"] = {}
    for name in ("model_id", "source_sha256", "model_kind", "inference_mode", "dtype"):
        value = metadata.get(name)
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"missing declared {name}")
    for key in ("source_sha256", "input_sha256"):
        digest = metadata.get(key)
        if key in metadata and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest.lower())
        ):
            reasons.append(f"{key} must be a SHA-256 hex digest")
    if not isinstance(metadata.get("dtype"), str) or metadata["dtype"] not in {
        "float32",
        "float64",
    }:
        reasons.append("dtype must declare float32 or float64")
    if (
        not any(
            _present(payload.get(key))
            for key in _ALIGNMENT
            if key.startswith("observations")
        )
        and not _present(metadata.get("input_sha256"))
        and metadata.get("input_semantics") != "input_free"
    ):
        reasons.append(
            "missing bound observation inputs or explicit input-free semantics"
        )
    if (
        not _present(payload.get("belief_axes"))
        and not _present(
            payload.get("axes", {}).get("beliefs")
            if isinstance(payload.get("axes", {}), dict)
            else None
        )
        and payload.get("schema_version") not in CURRENT_SIMULATION_SCHEMAS
    ):
        reasons.append("missing declared posterior axes or canonical result schema")
    record["statistics"]["model_id"] = metadata.get("model_id")
    record["statistics"]["model_kind"] = _family(payload)
    return record


def revalidate_scientific_record(record: dict[str, Any]) -> dict[str, Any]:
    """Revalidate cached records; mutable views cannot bypass source checks."""
    payload, metadata = record.get("payload"), record.get("metadata")
    if not isinstance(payload, dict) or not isinstance(metadata, dict):
        return prepare_scientific_record({})
    binding = record.get("binding", metadata)
    fresh = prepare_scientific_record(
        payload, binding if isinstance(binding, dict) else metadata
    )
    fresh["reasons"] = list(
        dict.fromkeys(fresh["reasons"] + list(record.get("reasons", [])))
    )
    if "views" in record and not _equal(record["views"], fresh["views"]):
        fresh["reasons"].append("cached native views differ from validated payload")
    return fresh


def compare_scientific_pair(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    """Admit two source-bound results and compare each named native view."""
    left, right = (
        revalidate_scientific_record(left),
        revalidate_scientific_record(right),
    )
    reasons = list(dict.fromkeys(left["reasons"] + right["reasons"]))
    a, b = left["metadata"], right["metadata"]
    for key in _FIELDS:
        if key == "dtype":
            continue
        if not _equal(a.get(key), b.get(key)):
            reasons.append(f"incompatible {key}")
    for key in _ALIGNMENT:
        av, bv = left["payload"].get(key), right["payload"].get(key)
        if _present(av) or _present(bv):
            if not _equal(av, bv):
                reasons.append(f"unaligned {key}")
    if left.get("views", {}).keys() != right.get("views", {}).keys():
        reasons.append("unaligned agent/factor view identities")
    else:
        for name, view in left.get("views", {}).items():
            other = right["views"][name]
            for key in _ALIGNMENT:
                av, bv = view.get(key), other.get(key)
                if (_present(av) or _present(bv)) and not _equal(av, bv):
                    reasons.append(f"unaligned {key} for view {name}")
    if reasons:
        return {"status": "not_comparable", "reasons": list(dict.fromkeys(reasons))}
    dtype = "float32" if "float32" in (a["dtype"], b["dtype"]) else "float64"
    rtol, atol = (1e-5, 1e-6) if dtype == "float32" else (1e-8, 1e-10)
    witness: dict[str, Any] = {
        "status": "comparable",
        "model_id": a["model_id"],
        "source_sha256": a["source_sha256"],
        "model_kind": a["model_kind"],
        "inference_mode": a["inference_mode"],
        "dtypes": [a["dtype"], b["dtype"]],
        "tolerance": {"rtol": rtol, "atol": atol},
        "views": {},
        "claim": "Reported source-bound numerical witness; does not establish extraction correctness, inference convergence or universal equivalence",
    }
    witness["unavailable_metrics"] = {}
    for quantity, convention in (
        ("free_energy", "free_energy_convention"),
        ("expected_free_energy", "expected_free_energy_convention"),
        ("variational_free_energy", "variational_free_energy_convention"),
    ):
        x_value, y_value = left["payload"].get(quantity), right["payload"].get(quantity)
        if not _present(x_value) or not _present(y_value):
            witness["unavailable_metrics"][quantity] = (
                "quantity absent from one or both results"
            )
            continue
        if not _present(a.get(convention)) or (
            quantity == "free_energy" and not _present(a.get("free_energy_kind"))
        ):
            witness["unavailable_metrics"][quantity] = (
                "missing declared scalar free energy kind/convention"
            )
            continue
        try:
            x_fe, y_fe = (
                np.asarray(x_value, dtype=float),
                np.asarray(y_value, dtype=float),
            )
            if (
                x_fe.ndim != 1
                or x_fe.shape != y_fe.shape
                or not np.isfinite(x_fe).all()
                or not np.isfinite(y_fe).all()
            ):
                raise ValueError("finite aligned scalar series required")
            posterior_steps = len(next(iter(left["views"].values()))["beliefs"])
            index = left["payload"].get(f"{quantity}_index")
            axes = left["payload"].get(f"{quantity}_axes")
            if len(x_fe) != posterior_steps and (
                not _present(index) or not _present(axes)
            ):
                raise ValueError(
                    "scalar series must align with posterior timesteps or declare its own axes and index"
                )
            if _present(index):
                index_array = np.asarray(index, dtype=float)
                if (
                    index_array.ndim != 1
                    or len(index_array) != len(x_fe)
                    or not np.isfinite(index_array).all()
                ):
                    raise ValueError(
                        "quantity-specific index must align with scalar series"
                    )
            witness[quantity] = {
                "convention": a[convention],
                "max_absolute_difference": float(np.abs(x_fe - y_fe).max()),
                "within_tolerance": bool(np.allclose(x_fe, y_fe, rtol=rtol, atol=atol)),
            }
        except (ValueError, TypeError) as exc:
            witness["unavailable_metrics"][quantity] = str(exc)
    for name, av in left["views"].items():
        bv = right["views"][name]
        x, y = (
            np.asarray(av["beliefs"], dtype=float),
            np.asarray(bv["beliefs"], dtype=float),
        )
        if x.shape != y.shape:
            return {
                "status": "not_comparable",
                "reasons": [f"unaligned posterior shape for {name}"],
                "dims_a": list(x.shape),
                "dims_b": list(y.shape),
            }
        metrics: dict[str, Any] = {
            "same_dimensions": True,
            "max_absolute_difference": float(np.abs(x - y).max()),
            "within_tolerance": bool(np.allclose(x, y, rtol=rtol, atol=atol)),
        }
        if model_family(av) == "continuous":
            ca, cb = np.asarray(av["posterior_cov"]), np.asarray(bv["posterior_cov"])
            metrics["covariance_max_absolute_difference"] = float(np.abs(ca - cb).max())
            metrics["covariance_within_tolerance"] = bool(
                np.allclose(ca, cb, rtol=rtol, atol=atol)
            )
        else:
            cx, cy = x.max(axis=1), y.max(axis=1)
            if len(cx) > 1 and cx.std() > 0 and cy.std() > 0:
                metrics["confidence_correlation"] = float(np.corrcoef(cx, cy)[0, 1])
            else:
                metrics["unavailable_metrics"] = {
                    "confidence_correlation": "constant confidence or fewer than two timesteps"
                }
        witness["views"][name] = metrics
    return witness


def performance_admission(records: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Require identical attempted identity sets, config and environment for ranks."""
    reasons: list[str] = []
    rows = [
        [revalidate_scientific_record(row) for row in group]
        for group in records.values()
    ]
    if not rows or any(not row for row in rows):
        return ["missing source-bound attempted results"]
    identities = [
        Counter(
            tuple(
                value if isinstance(value, str) else None
                for value in (
                    row["metadata"].get("model_id"),
                    row["metadata"].get("source_sha256"),
                )
            )
            for row in group
        )
        for group in rows
    ]
    if any(group != identities[0] for group in identities[1:]):
        reasons.append("different attempted model identity sets")
    all_rows = [row for group in rows for row in group]
    for row in all_rows:
        reasons.extend(row["reasons"])
        status = row["payload"].get("status", "success")
        if (
            row["payload"].get("success") is not True
            or not isinstance(status, str)
            or status.lower() not in {"success", "completed"}
        ):
            reasons.append("unsuccessful or unfinished attempted result")
    for key in ("execution_configuration", "execution_environment"):
        values = [row["payload"].get(key) for row in all_rows]
        if any(not _present(value) for value in values):
            reasons.append(f"missing declared {key}")
        elif any(not _equal(values[0], value) for value in values[1:]):
            reasons.append(f"different {key}")
    return list(dict.fromkeys(reasons))


def scientific_comparisons(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Group by stable model ID; never compare unrelated corpus members."""
    comparisons: dict[str, Any] = {
        "metric_agreement": {},
        "simulation_statistics": {},
        "scientific_exclusions": {},
    }
    records = {
        framework: [revalidate_scientific_record(row) for row in rows]
        for framework, rows in records.items()
    }
    groups: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
    for framework, rows in records.items():
        stats = {
            str(index): row["statistics"]
            for index, row in enumerate(rows)
            if row.get("views")
        }
        if stats:
            comparisons["simulation_statistics"][framework] = (
                next(iter(stats.values())) if len(rows) == 1 else {"models": stats}
            )
        for index, row in enumerate(rows):
            label = f"{framework}[{index}]"
            if row["reasons"]:
                comparisons["scientific_exclusions"][label] = row["reasons"]
            model_id = row["metadata"].get("model_id")
            if isinstance(model_id, str) and model_id:
                groups.setdefault(model_id, []).append((framework, label, row))
            else:
                comparisons["scientific_exclusions"].setdefault(label, []).append(
                    "no source-bound counterpart"
                )
    for model_id, group in groups.items():
        if len({fw for fw, _, _ in group}) < 2:
            for _, label, _ in group:
                comparisons["scientific_exclusions"].setdefault(label, []).append(
                    "no source-bound counterpart"
                )
            continue
        for (fw_a, label_a, a), (fw_b, label_b, b) in combinations(group, 2):
            if fw_a == fw_b:
                continue
            key = (
                f"{fw_a}_vs_{fw_b}"
                if all(len(rows) == 1 for rows in records.values())
                else f"{model_id}:{label_a}_vs_{label_b}"
            )
            comparisons["metric_agreement"][key] = compare_scientific_pair(a, b)
    return comparisons
