"""Independent source reader for bounded passive Gaussian-agent composition.

No production parser, extractor, renderer or backend is imported. Independence
is an explicit authored assumption, never inferred from compatible dimensions.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

from .witness_source import GAUSSIAN_AXES, _array, _json, _source_sections

SOURCE_SCHEMA = "gnn-independent-gaussian-composition-v1"
AGENTS = ("agent1", "agent2")
PARAMETERS = ("F", "H", "Q", "R", "prior_mean", "prior_cov")


def composition_equations() -> str:
    """Return the exact admitted passive product-model declaration."""
    parts = [
        "p(x_agent1,x_agent2,y_agent1,y_agent2)=p(x_agent1,y_agent1)*p(x_agent2,y_agent2)"
    ]
    for agent in AGENTS:
        parts.extend(
            [
                f"x_{agent}_0~N(prior_mean_{agent},prior_cov_{agent})",
                f"x_{agent}_(t+1)=F_{agent}*x_{agent}_t+N(0,Q_{agent})",
                f"y_{agent}_t=H_{agent}*x_{agent}_t+N(0,R_{agent})",
            ]
        )
    return "; ".join(parts)


def _assignments(text: str, *, options: bool = False) -> dict[str, Any]:
    values: dict[str, Any] = {}
    separator = ": " if options else " = "
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*)" + separator + r"(.+)", line)
        if match is None:
            raise ValueError(
                "composition grammar requires one JSON assignment per line"
            )
        key, literal = match.groups()
        if key in values:
            raise ValueError(f"duplicate composition parameter: {key}")
        values[key] = literal if key == "agent_coupling" and options else _json(literal)
    return values


def _labels(value: Any, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 3
        or any(
            not isinstance(label, str) or not 1 <= len(label) <= 256 for label in value
        )
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name}: unique nonempty bounded ordered labels required")
    return value


def read_composition_source(
    source: bytes, sections: dict[str, str] | None = None
) -> dict[str, Any]:
    """Independently admit exactly two stationary passive Gaussian agents."""
    sections = _source_sections(source) if sections is None else sections
    declaration = _json(sections["NumericalWitness"])
    keys = {
        "schema",
        "model",
        "question",
        "dtype",
        "initial_time",
        "axes",
        "agents",
        "state_ids",
        "observation_ids",
    }
    if (
        not isinstance(declaration, dict)
        or set(declaration) != keys
        or declaration.get("schema") != SOURCE_SCHEMA
        or declaration.get("model") != "gaussian_product"
    ):
        raise ValueError("composition: unsupported witness schema or fields")
    if declaration["question"] != "filtering":
        raise ValueError(
            "composition: only filtering is admitted; smoothing is unsupported"
        )
    if (
        declaration["dtype"] != "float64"
        or declaration["initial_time"] != "first_observation_uses_prior"
        or declaration["axes"] != GAUSSIAN_AXES
    ):
        raise ValueError("composition: incompatible dtype, timing or axes")
    if declaration["agents"] != list(AGENTS):
        raise ValueError("composition: ordered agent1/agent2 identities required")
    labels = {}
    for key in ("state_ids", "observation_ids"):
        mapping = declaration[key]
        if not isinstance(mapping, dict) or set(mapping) != set(AGENTS):
            raise ValueError(f"composition {key}: exact agent identity map required")
        labels[key] = {
            agent: _labels(mapping[agent], f"{key}.{agent}") for agent in AGENTS
        }
        if sum(map(len, labels[key].values())) > 4:
            raise ValueError("composition: total dimension budget exceeded")
    options = _assignments(sections["ModelParameters"], options=True)
    if set(options) != {
        "nr_agents",
        "agent_coupling",
        "num_timesteps",
        "random_seed_agent1",
        "random_seed_agent2",
    }:
        raise ValueError("composition model parameters: missing or unsupported fields")
    if type(options["nr_agents"]) is not int or options["nr_agents"] != 2:
        raise ValueError("composition: exactly two agents required")
    if options["agent_coupling"] != "independent":
        raise ValueError("composition: coupled or shared models are unsupported")
    count = options["num_timesteps"]
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError("composition: timestep budget exceeded")
    for agent in AGENTS:
        seed = options[f"random_seed_{agent}"]
        if type(seed) is not int or seed < 0 or seed > 2**32 - 1:
            raise ValueError(f"{agent}: bounded nonnegative integer seed required")
    parameters = _assignments(sections["InitialParameterization"])
    expected = {f"{key}_{agent}" for agent in AGENTS for key in PARAMETERS}
    if set(parameters) != expected:
        raise ValueError(
            "composition: complete passive agent blocks required; controls/shared keys unsupported"
        )
    if sections["GNNSection"] != "ActInfContinuousMultiAgent":
        raise ValueError(
            "composition: explicit continuous multi-agent section required"
        )
    declarations = []
    components = {}
    for agent in AGENTS:
        states, observations = (
            labels["state_ids"][agent],
            labels["observation_ids"][agent],
        )
        n, m = len(states), len(observations)
        declarations.extend(
            [f"x_{agent}[{n},1,type=float]", f"y_{agent}[{m},1,type=float]"]
        )
        shapes = {
            "F": (n, n),
            "H": (m, n),
            "Q": (n, n),
            "R": (m, m),
            "prior_mean": (n,),
            "prior_cov": (n, n),
        }
        component_parameters = {}
        for key, shape in shapes.items():
            raw = parameters[f"{key}_{agent}"]
            array = _array(raw, f"{key}_{agent}")
            if array.shape != shape:
                raise ValueError(
                    f"{key}_{agent}: dimensions disagree with declared identities"
                )
            if key in {"Q", "R", "prior_cov"}:
                if not np.allclose(array, array.T, rtol=0, atol=1e-14):
                    raise ValueError(f"{key}_{agent}: symmetric covariance required")
                try:
                    np.linalg.cholesky(array)
                except np.linalg.LinAlgError as error:
                    raise ValueError(
                        f"{key}_{agent}: positive definite covariance required"
                    ) from error
            component_parameters[key] = raw
        components[agent] = {
            "schema": SOURCE_SCHEMA,
            "model": "gaussian",
            "question": "filtering",
            "dtype": "float64",
            "initial_time": declaration["initial_time"],
            "axes": {key: list(value) for key, value in GAUSSIAN_AXES.items()},
            "state_ids": states,
            "observation_ids": observations,
            "parameters": component_parameters,
            "model_parameters": {
                "num_timesteps": count,
                "random_seed": options[f"random_seed_{agent}"],
            },
            "source_sha256": hashlib.sha256(source).hexdigest(),
        }
    if sections["StateSpaceBlock"] != "\n".join(declarations):
        raise ValueError(
            "composition: exact agent state/observation declarations required"
        )
    if sections["Equations"] != composition_equations():
        raise ValueError("composition: explicit passive product equations required")
    return {
        **declaration,
        "state_ids": [
            f"{agent}:{label}"
            for agent in AGENTS
            for label in labels["state_ids"][agent]
        ],
        "observation_ids": [
            f"{agent}:{label}"
            for agent in AGENTS
            for label in labels["observation_ids"][agent]
        ],
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "components": components,
        "parameters": parameters,
        "model_parameters": options,
        "equations": sections["Equations"],
    }


def joint_gaussian_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Assemble the authored independent model, without using renderer code."""
    components = [contract["components"][agent] for agent in AGENTS]
    parameters = {}
    for key in PARAMETERS:
        arrays = [
            np.asarray(component["parameters"][key], dtype=np.float64)
            for component in components
        ]
        if key == "prior_mean":
            value = np.concatenate(arrays)
        else:
            rows = sum(array.shape[0] for array in arrays)
            columns = sum(array.shape[1] for array in arrays)
            value = np.zeros((rows, columns), dtype=np.float64)
            row = column = 0
            for array in arrays:
                nr, nc = array.shape
                value[row : row + nr, column : column + nc] = array
                row += nr
                column += nc
        parameters[key] = value.tolist()
    return {
        "model": "gaussian",
        "question": "filtering",
        "parameters": parameters,
        "model_parameters": {
            "num_timesteps": contract["model_parameters"]["num_timesteps"]
        },
        "state_ids": contract["state_ids"],
        "observation_ids": contract["observation_ids"],
    }
