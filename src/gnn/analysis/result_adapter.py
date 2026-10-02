"""Validated scientific result views shared by analyzers and execution receipts.

Native agent/factor marginals remain separate. Gaussian means are never
interpreted as categorical probabilities or reconstructed joint distributions.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def model_family(data: dict[str, Any]) -> str:
    """Resolve existing root/runtime discriminants and continuous payloads."""
    kind = data.get("model_kind") or data.get("runtime_metadata", {}).get("model_kind")
    if kind == "multi_agent_continuous" or "posterior_cov_by_agent" in data:
        return "multi_agent_continuous"
    if kind == "continuous" or "posterior_cov" in data:
        return "continuous"
    if data.get("beliefs_by_agent"):
        return "multi_agent"
    return str(kind or "flat")


def numeric_trace(value: Any, *, name: str, width: int | None = None) -> np.ndarray:
    """Require a regular finite timestep-by-dimension trace."""
    try:
        trace = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a regular numeric trace") from exc
    if trace.ndim != 2 or not trace.shape[0] or not trace.shape[1]:
        raise ValueError(f"{name} must have shape (timesteps, dimensions)")
    if not np.isfinite(trace).all():
        raise ValueError(f"{name} must contain finite values")
    if width is not None and trace.shape[1] != width:
        raise ValueError(f"{name} width {trace.shape[1]} does not match {width}")
    return trace


def categorical_trace(value: Any, *, name: str) -> np.ndarray:
    """Validate probabilities without changing their values."""
    trace = numeric_trace(value, name=name)
    if (
        np.any(trace < 0)
        or np.any(trace > 1)
        or not np.allclose(trace.sum(axis=1), 1, rtol=1e-6, atol=1e-8)
    ):
        raise ValueError(f"{name} must contain categorical probability rows")
    return trace


def _check_declared_timesteps(data: dict[str, Any], actual: int) -> None:
    declared = data.get("num_timesteps")
    if declared is not None and (
        isinstance(declared, bool)
        or not isinstance(declared, (int, float))
        or not np.isfinite(declared)
        or declared < 1
        or declared != actual
    ):
        raise ValueError(
            "num_timesteps must match the positive integer posterior length"
        )


def _check_bound_series(view: dict[str, Any], actual: int) -> None:
    for key in (
        "actions",
        "observations",
        "true_states",
        "efe_per_action",
        "expected_free_energy",
        "policy_posterior",
    ):
        value = view.get(key)
        if value is not None and len(value) and len(value) != actual:
            raise ValueError(f"{key} must align with posterior timesteps")


def continuous_result_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """Validate Gaussian means/covariances and return directly bound metrics."""
    means = numeric_trace(data.get("beliefs"), name="posterior means")
    cov = np.asarray(data.get("posterior_cov"), dtype=float)
    expected = (means.shape[0], means.shape[1], means.shape[1])
    if cov.shape != expected or not np.isfinite(cov).all():
        raise ValueError(f"posterior_cov must be finite with shape {expected}")
    if not np.allclose(cov, cov.transpose(0, 2, 1), rtol=0, atol=1e-8):
        raise ValueError("posterior_cov must be symmetric")
    if np.any(np.linalg.eigvalsh(cov) < -1e-8):
        raise ValueError("posterior_cov must be positive semidefinite")
    _check_declared_timesteps(data, len(means))
    metrics: dict[str, Any] = {
        "num_timesteps": len(means),
        "num_states": means.shape[1],
        "mean_posterior_std": float(
            np.mean(np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0)))
        ),
        "covariance_psd": True,
    }
    true = data.get("true_states_continuous")
    if true is not None and len(true):
        states = numeric_trace(
            true, name="true continuous states", width=means.shape[1]
        )
        if states.shape != means.shape:
            raise ValueError("True continuous states and posterior means must align")
        metrics["rmse_vs_true"] = float(np.sqrt(np.mean((means - states) ** 2)))
    for key in ("controls", "observations_continuous"):
        reported = data.get(key)
        if reported is not None and len(reported):
            trace = numeric_trace(
                reported, name=key, width=means.shape[1] if key == "controls" else None
            )
            if len(trace) != len(means):
                raise ValueError(f"{key} must align with posterior timesteps")
    return metrics


def result_views(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return named coherent result views without inventing unreported traces."""
    family = model_family(data)
    if family == "continuous":
        continuous_result_metrics(data)
        return {"continuous": data}
    agents = data.get("beliefs_by_agent")
    factors = data.get("beliefs_by_factor")
    if isinstance(agents, dict) and agents:
        declared = data.get("agents", [])
        names = [str(name) for name in declared] if declared else sorted(agents)
        if set(names) != set(agents) or len(names) != len(set(names)):
            raise ValueError("Declared agents must match belief trace identities")
        source, suffix = agents, "by_agent"
    elif (
        isinstance(factors, dict)
        and len(factors) > 1
        and family in {"hierarchical", "factored"}
    ):
        names, source, suffix = list(factors), factors, "by_factor"
    else:
        beliefs = data.get("beliefs")
        if isinstance(beliefs, dict):
            if len(beliefs) != 1:
                raise ValueError(
                    "Multiple belief traces require declared agent/factor semantics"
                )
            beliefs = next(iter(beliefs.values()))
        if beliefs is None and isinstance(factors, dict) and len(factors) == 1:
            beliefs = next(iter(factors.values()))
        if beliefs is None or not len(beliefs):
            return {}
        categorical_trace(beliefs, name="beliefs")
        _check_declared_timesteps(data, len(beliefs))
        _check_bound_series(data, len(beliefs))
        return {"joint_state": {**data, "beliefs": beliefs}}
    views = {}
    expected_steps: int | None = None
    for name in names:
        if family == "multi_agent_continuous":
            view = {**data, "model_kind": "continuous", "beliefs": source[name]}
            for key in (
                "posterior_cov",
                "true_states_continuous",
                "observations_continuous",
                "controls",
                "vfe_per_iteration",
                "variational_free_energy",
            ):
                table = data.get(f"{key}_by_agent")
                if key == "posterior_cov" and (
                    not isinstance(table, dict) or name not in table
                ):
                    raise ValueError(f"Missing Gaussian covariance for agent {name}")
                view[key] = table.get(name, []) if isinstance(table, dict) else []
            for key in ("beliefs_by_agent", "posterior_cov_by_agent"):
                view.pop(key, None)
            metrics = continuous_result_metrics(view)
            if (
                expected_steps is not None
                and metrics["num_timesteps"] != expected_steps
            ):
                raise ValueError(
                    "Agent posterior traces must share their timestep count"
                )
            expected_steps = metrics["num_timesteps"]
            views[name] = view
            continue
        trace = categorical_trace(source[name], name=f"beliefs/{name}")
        if expected_steps is not None and len(trace) != expected_steps:
            raise ValueError(
                "Agent/factor belief traces must share their timestep count"
            )
        expected_steps = len(trace)
        view = {
            **data,
            "beliefs": trace.tolist(),
            "num_timesteps": len(trace),
            "model_kind": "flat",
            "runtime_metadata": {
                **data.get("runtime_metadata", {}),
                "model_kind": "flat",
                "source_model_kind": family,
            },
            "model_parameters": {
                **data.get("model_parameters", {}),
                "state_factors": [],
                "num_states": trace.shape[1],
            },
        }
        for target, source_name in (
            ("actions", "actions"),
            ("observations", "observations"),
            ("true_states", "true_states"),
            ("efe_per_action", "efe_per_action"),
            ("expected_free_energy", "selected_efe"),
            ("policy_posterior", "policy_posterior"),
            ("vfe_per_iteration", "vfe_per_iteration"),
        ):
            table = data.get(f"{source_name}_{suffix}")
            if isinstance(table, dict):
                view[target] = table.get(name, [])
            elif suffix in {"by_agent", "by_factor"}:
                view[target] = []
        if suffix == "by_factor":
            states = data.get("hidden_states_by_factor", {})
            if isinstance(states, dict):
                view["true_states"] = states.get(name, [])
        for key in ("beliefs_by_agent", "beliefs_by_factor"):
            view.pop(key, None)
        _check_bound_series(view, len(trace))
        views[name] = view
    if expected_steps is not None:
        _check_declared_timesteps(data, expected_steps)
    return views


def structured_result_data(data: dict[str, Any]) -> dict[str, Any]:
    """Retain native scientific fields in an execution result after validation."""
    result_views(data)
    keys = (
        "schema_version",
        "model_kind",
        "agents",
        "agent_coupling",
        "beliefs",
        "beliefs_by_agent",
        "beliefs_by_factor",
        "posterior_cov",
        "posterior_cov_by_agent",
        "true_states",
        "true_states_by_agent",
        "hidden_states_by_factor",
        "true_states_continuous",
        "true_states_continuous_by_agent",
        "observations",
        "observations_by_agent",
        "observations_by_modality",
        "observations_continuous",
        "observations_continuous_by_agent",
        "actions",
        "actions_by_agent",
        "actions_by_control_factor",
        "controls",
        "controls_by_agent",
        "expected_free_energy",
        "expected_free_energy_convention",
        "efe_per_action",
        "efe_per_action_by_agent",
        "policy_posterior",
        "policy_posterior_by_agent",
        "variational_free_energy",
        "variational_free_energy_by_agent",
        "vfe_per_iteration",
        "vfe_per_iteration_by_agent",
        "validation",
        "model_parameters",
        "runtime_metadata",
        "matrix_provenance",
        "num_timesteps",
    )
    # Keep extension scientific fields too. Named views select semantics for
    # visualization; extraction must preserve the complete reported payload.
    return {**data, **{key: data[key] for key in keys if key in data}}
