"""Admission for the two-agent joint linear-Gaussian filtering contract."""

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from .continuous_spec import (
    JOINT_GAUSSIAN_CONTRACT,
    REQUIRED_KEYS,
    ContinuousSpec,
    extract_continuous_spec,
)

MAX_AXES = 16
MAX_TIMESTEPS = 128
_INPUT_KEYS = set(REQUIRED_KEYS) | {
    "joint_agents",
    "control_matrix",
    "control_sequence",
    "observation_sequence",
}


@dataclass(frozen=True)
class JointGaussianSpec:
    system: ContinuousSpec
    agents: list[dict[str, Any]]
    control_matrix: np.ndarray
    controls: np.ndarray
    observations: np.ndarray


def _label(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError(
            "Joint Gaussian axis identities must be nonempty strings of at most 128 characters"
        )
    return value


def _axis_values(value: Any, shape: tuple[int, ...], name: str) -> Any:
    """Validate types and exact dimensions before allocating NumPy arrays."""
    if not shape:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} requires finite nonboolean numeric entries")
        try:
            number = float(value)
        except OverflowError as error:
            raise ValueError(f"{name} requires finite numeric entries") from error
        if not np.isfinite(number):
            raise ValueError(f"{name} requires finite numeric entries")
        return number
    if isinstance(value, np.ndarray) and value.shape != shape:
        raise ValueError(f"{name} does not match its declared axes")
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != shape[0]:
        raise ValueError(f"{name} does not match its declared axes")
    return [_axis_values(item, shape[1:], name) for item in value]


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    return np.asarray(_axis_values(value, shape, name), dtype=np.float64).reshape(shape)


def extract_joint_gaussian_spec(spec: dict[str, Any]) -> JointGaussianSpec:
    """Admit complete, source-declared matrices and fixed sequences without synthesis."""
    params = spec.get("model_parameters") or {}
    if params.get("agent_coupling") != JOINT_GAUSSIAN_CONTRACT:
        raise ValueError(
            "unsupported-composition: expected the joint Gaussian filtering declaration"
        )
    count = params.get("nr_agents", params.get("num_agents"))
    if not isinstance(count, int) or count != 2:
        raise ValueError("Joint Gaussian filtering requires exactly two agents")
    if "num_agents" in params and (
        type(params["num_agents"]) is not int or params["num_agents"] != count
    ):
        raise ValueError("Joint Gaussian agent declarations disagree")
    steps = params.get("num_timesteps")
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or not 1 <= steps <= MAX_TIMESTEPS
    ):
        raise ValueError(
            f"Joint Gaussian num_timesteps must be an integer in 1..{MAX_TIMESTEPS}"
        )

    initial = spec.get("initialparameterization") or spec.get(
        "initial_parameterization"
    )
    if not isinstance(initial, dict) or set(initial) != _INPUT_KEYS:
        raise ValueError(
            f"Joint Gaussian InitialParameterization requires exactly {sorted(_INPUT_KEYS)}"
        )
    declarations = initial["joint_agents"]
    if not isinstance(declarations, (list, tuple)) or len(declarations) != 2:
        raise ValueError("joint_agents must declare two ordered agent layouts")

    axes = ("state", "observation", "control")
    sizes = dict.fromkeys(axes, 0)
    agents = []
    for declaration in declarations:
        if not isinstance(declaration, dict) or set(declaration) != {
            "id",
            *(f"{axis}_ids" for axis in axes),
        }:
            raise ValueError(
                "Each joint agent needs an id and ordered state_ids, observation_ids and control_ids"
            )
        agent: dict[str, Any] = {"id": _label(declaration["id"])}
        for axis in axes:
            key = f"{axis}_ids"
            values = declaration[key]
            minimum = 0 if axis == "control" else 1
            if (
                not isinstance(values, (list, tuple))
                or not minimum <= len(values) <= MAX_AXES
            ):
                raise ValueError(
                    f"Joint agent {key} needs {minimum}..{MAX_AXES} labels"
                )
            labels = [_label(value) for value in values]
            if len(set(labels)) != len(labels):
                raise ValueError(f"Duplicate {key} in joint agent {agent['id']}")
            end = sizes[axis] + len(labels)
            if end > MAX_AXES:
                raise ValueError(f"Joint {axis} dimension exceeds {MAX_AXES}")
            agent[key] = labels
            agent[f"{axis}_slice"] = [sizes[axis], end]
            sizes[axis] = end
        agents.append(agent)
    if agents[0]["id"] == agents[1]["id"]:
        raise ValueError("joint_agents requires unique agent identities")

    n, m, p = (sizes[axis] for axis in axes)
    for key, expected in (
        ("num_states", n),
        ("num_hidden_states", n),
        ("num_obs", m),
        ("num_observations", m),
        ("num_actions", p),
    ):
        if key in params and (type(params[key]) is not int or params[key] != expected):
            raise ValueError(f"{key} does not match ordered joint Gaussian axes")
    shape = {
        "F": (n, n),
        "H": (m, n),
        "Q": (n, n),
        "R": (m, m),
        "prior_mean": (n,),
        "prior_cov": (n, n),
    }
    system = extract_continuous_spec(
        {
            "initialparameterization": {
                key: _array(initial[key], expected_shape, key)
                for key, expected_shape in shape.items()
            },
            "model_parameters": {
                key: value for key, value in params.items() if key != "agent_coupling"
            },
            "model_name": spec.get("model_name") or spec.get("name"),
        }
    )
    return JointGaussianSpec(
        system=system,
        agents=agents,
        control_matrix=_array(initial["control_matrix"], (n, p), "control_matrix"),
        controls=_array(
            initial["control_sequence"], (steps - 1, p), "control_sequence"
        ),
        observations=_array(
            initial["observation_sequence"], (steps, m), "observation_sequence"
        ),
    )
