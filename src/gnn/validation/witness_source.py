"""Restricted source reader and bounded independent numerical references.

This module deliberately imports no GNN parser, extractor, renderer or backend.
It accepts only explicit single-factor and passive independent-agent source
contracts. Finite path sums and joint-Gaussian conditioning are numerical
witnesses, not translation proofs.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray

SCHEMA = "gnn-independent-numerical-witness-v1"
FINITE_AXES = {
    "A": ["observation", "state"],
    "B": ["next_state", "previous_state", "action"],
}
GAUSSIAN_AXES = {"F": ["next_state", "previous_state"], "H": ["observation", "state"]}


def _json(text: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value: str) -> NoReturn:
        raise ValueError(f"nonfinite literal: {value}")

    return json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite)


def _assignments(text: str, *, model_options: bool = False) -> dict[str, Any]:
    result = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        separator = ": " if model_options else " = "
        match = re.fullmatch(r"([A-Za-z_]+)" + separator + r"(.+)", line)
        if not match:
            raise ValueError("reference grammar requires one JSON assignment per line")
        name, literal = match.groups()
        if name in result:
            raise ValueError(f"duplicate parameter: {name}")
        result[name] = (
            literal if model_options and name == "b_tensor_order" else _json(literal)
        )
    return result


def _array(value: Any, name: str) -> np.ndarray:
    def numeric(item: Any) -> bool:
        if isinstance(item, list):
            return bool(item) and all(numeric(child) for child in item)
        return type(item) in (int, float) and np.isfinite(item)

    if not numeric(value):
        raise ValueError(f"{name}: finite nonboolean numeric literals required")
    try:
        result = np.asarray(value, dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"{name}: rectangular array required") from error
    return result


def _source_sections(source: bytes) -> dict[str, str]:
    """Read the bounded exact section envelope shared by witness contracts."""
    if len(source) > 1024 * 1024:
        raise ValueError("source byte budget exceeded")
    text = source.decode("utf-8").replace("\r\n", "\n")
    required = {
        "ModelName",
        "GNNSection",
        "StateSpaceBlock",
        "InitialParameterization",
        "ModelParameters",
        "Equations",
        "NumericalWitness",
    }
    for heading in re.findall(r"(?m)^[ \t]*##(?!#)[^\n]*", text):
        match = re.fullmatch(r"## ([A-Za-z]+)[ \t]*", heading)
        if match is None or match.group(1) not in required:
            raise ValueError(
                "reference grammar: malformed or unsupported source heading"
            )
    sections = {}
    parts = re.split(r"(?m)^## ([A-Za-z]+)\s*\n", text)
    for index in range(1, len(parts), 2):
        name = parts[index]
        if name in sections:
            raise ValueError(f"duplicate section: {name}")
        sections[name] = parts[index + 1].strip()
    if set(sections) != required:
        raise ValueError("reference grammar: missing or unsupported source section")
    return sections


def read_source(source: bytes) -> dict[str, Any]:
    """Read only declared witness subsets, independently of GNN extraction."""
    sections = _source_sections(source)
    declaration = _json(sections["NumericalWitness"])
    if (
        isinstance(declaration, dict)
        and declaration.get("schema") == "gnn-independent-gaussian-composition-v1"
    ):
        from .witness_composition_source import read_composition_source

        return read_composition_source(source, sections)
    keys = {
        "schema",
        "model",
        "question",
        "dtype",
        "initial_time",
        "axes",
        "state_ids",
        "observation_ids",
    }
    if (
        not isinstance(declaration, dict)
        or set(declaration) != keys
        or declaration["schema"] != SCHEMA
    ):
        raise ValueError("unsupported witness schema or fields")
    if declaration["dtype"] != "float64":
        raise ValueError("dtype: this witness admits float64 only")
    if declaration["initial_time"] != "first_observation_uses_prior":
        raise ValueError("initial_time: first observation must use the declared prior")
    model = declaration["model"]
    if model not in ("finite", "gaussian"):
        raise ValueError("model: unsupported scientific contract")
    questions = ("filtering", "smoothing") if model == "finite" else ("filtering",)
    if declaration["question"] not in questions:
        raise ValueError("question: unsupported inference or EFE/sign contract")
    axes = FINITE_AXES if model == "finite" else GAUSSIAN_AXES
    if declaration["axes"] != axes:
        raise ValueError("axes: incompatible variable/axis identity")
    for key in ("state_ids", "observation_ids"):
        ids = declaration[key]
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(item, str) or not item for item in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(f"{key}: unique nonempty ordered identities required")
    n, m = len(declaration["state_ids"]), len(declaration["observation_ids"])
    if n > 4 or m > 4:
        raise ValueError("reference dimension budget exceeded")
    parameters = _assignments(sections["InitialParameterization"])
    options = _assignments(sections["ModelParameters"], model_options=True)
    allowed_options = {"num_timesteps", "random_seed"}
    if model == "finite":
        allowed_options.add("b_tensor_order")
    if set(options) != allowed_options:
        raise ValueError("reference model parameters: unsupported fields")
    t = options["num_timesteps"]
    if (
        type(t) is not int
        or not 1 <= t <= 8
        or type(options["random_seed"]) is not int
        or options["random_seed"] < 0
    ):
        raise ValueError("reference time/seed budget or integer identity")
    if model == "finite":
        if sections["GNNSection"] != "ActInfPOMDP" or set(parameters) != {
            "A",
            "B",
            "C",
            "D",
        }:
            raise ValueError(
                "finite: only a single stationary A/B/C/D model is admitted"
            )
        if options["b_tensor_order"] != "next_state_previous_state_action":
            raise ValueError("B axes: explicit canonical orientation required")
        a, b, c, d = (_array(parameters[key], key) for key in ("A", "B", "C", "D"))
        if (
            a.shape != (m, n)
            or b.ndim != 3
            or b.shape[:2] != (n, n)
            or d.shape != (n,)
            or c.shape != (m,)
            or np.any(c != 0)
        ):
            raise ValueError("finite: dimensions or passive preference contract")
        if not 1 <= b.shape[2] <= 4 or n**t > 4096:
            raise ValueError("finite path/action budget exceeded")
        for key, values in (("A", a), ("B", b), ("D", d)):
            if np.any(values < 0) or not np.allclose(
                values.sum(axis=0), 1, rtol=0, atol=1e-14
            ):
                raise ValueError(f"{key}: normalized probabilities required")
        equations = (
            "p(s_0)=D; p(o_t|s_t)=A[o_t,s_t]; p(s_(t+1)|s_t,a_t)=B[s_(t+1),s_t,a_t]"
        )
        declarations = (
            f"s[{n},1,type=int]\no[{m},1,type=int]\nu[{b.shape[2]},1,type=int]"
        )
    else:
        if sections["GNNSection"] != "ActInfContinuous" or set(parameters) != {
            "F",
            "H",
            "Q",
            "R",
            "prior_mean",
            "prior_cov",
        }:
            raise ValueError("gaussian: only a passive stationary LGSSM is admitted")
        shapes = {
            "F": (n, n),
            "H": (m, n),
            "Q": (n, n),
            "R": (m, m),
            "prior_mean": (n,),
            "prior_cov": (n, n),
        }
        for key, shape in shapes.items():
            values = _array(parameters[key], key)
            if values.shape != shape:
                raise ValueError(f"{key}: declared dimensions disagree")
            if key in ("Q", "R", "prior_cov"):
                if not np.allclose(values, values.T, rtol=0, atol=1e-14):
                    raise ValueError(f"{key}: symmetric covariance required")
                try:
                    np.linalg.cholesky(values)
                except np.linalg.LinAlgError as error:
                    raise ValueError(
                        f"{key}: positive definite covariance required"
                    ) from error
        equations = (
            "x_0~N(prior_mean,prior_cov); x_(t+1)=F*x_t+N(0,Q); y_t=H*x_t+N(0,R)"
        )
        declarations = f"x[{n},1,type=float]\ny[{m},1,type=float]"
    if (
        sections["Equations"] != equations
        or sections["StateSpaceBlock"] != declarations
    ):
        raise ValueError(
            "equations/state identities: source is outside the admitted subset"
        )
    return {
        **declaration,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "parameters": parameters,
        "model_parameters": options,
        "equations": equations,
    }


def enumerate_finite(
    contract: dict, observations: list[int], actions: list[int]
) -> np.ndarray:
    """Sum complete latent paths; filtering uses one independent prefix at a time."""
    p = contract["parameters"]
    a, b, d = (np.asarray(p[key], dtype=np.float64) for key in ("A", "B", "D"))
    t, n = len(observations), len(d)
    if (
        contract["model"] != "finite"
        or t != contract["model_parameters"]["num_timesteps"]
        or len(actions) != t - 1
    ):
        raise ValueError(
            "conditioning_horizon: observations/actions do not match source"
        )
    if any(type(o) is not int or not 0 <= o < a.shape[0] for o in observations) or any(
        type(u) is not int or not 0 <= u < b.shape[2] for u in actions
    ):
        raise ValueError("observation/action identity is out of range")

    def prefix(length: int) -> NDArray[np.float64]:
        mass = np.zeros((length, n), dtype=np.float64)
        total = 0.0
        for path in itertools.product(range(n), repeat=length):
            weight = d[path[0]] * a[observations[0], path[0]]
            for time in range(1, length):
                weight *= (
                    b[path[time], path[time - 1], actions[time - 1]]
                    * a[observations[time], path[time]]
                )
            total += weight
            for time, state in enumerate(path):
                mass[time, state] += weight
        if total <= 0 or not np.isfinite(total):
            raise ValueError("conditioning event has zero or nonfinite probability")
        return mass / total

    if contract["question"] == "smoothing":
        return prefix(t)
    return np.stack([prefix(time + 1)[-1] for time in range(t)])


def condition_joint_gaussian(
    contract: dict, observations: list[list[float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Condition joint Gaussians, refusing invalid float64 intermediate results."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            means, covariances = _condition_joint_gaussian(contract, observations)
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError(
            "Gaussian reference numerical failure: overflow, invalid arithmetic or unsolvable conditioning system"
        ) from exc
    if not np.isfinite(means).all() or not np.isfinite(covariances).all():
        raise ValueError(
            "Gaussian reference numerical failure: nonfinite means or covariance"
        )
    return means, covariances


def _condition_joint_gaussian(
    contract: dict, observations: list[list[float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Condition full joint state/observation Gaussians, without Kalman recursion."""
    if contract["model"] != "gaussian" or contract["question"] != "filtering":
        raise ValueError("question: this reference admits Gaussian filtering only")
    p = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in contract["parameters"].items()
    }
    f, h, q, r, mu, prior = (
        p[key] for key in ("F", "H", "Q", "R", "prior_mean", "prior_cov")
    )
    t, n, m = contract["model_parameters"]["num_timesteps"], len(mu), h.shape[0]
    y = _array(observations, "observations")
    if y.shape != (t, m):
        raise ValueError(
            "conditioning_horizon: observation shape does not match source"
        )
    # Independent generative construction: x = L [x_0, epsilon_1, ...].
    loading = np.zeros((t * n, t * n))
    noise = np.zeros_like(loading)
    means = []
    for time in range(t):
        loading[time * n : (time + 1) * n, :n] = np.linalg.matrix_power(f, time)
        means.append(np.linalg.matrix_power(f, time) @ mu)
        noise[time * n : (time + 1) * n, time * n : (time + 1) * n] = (
            prior if time == 0 else q
        )
        for innovation in range(1, time + 1):
            loading[
                time * n : (time + 1) * n, innovation * n : (innovation + 1) * n
            ] = np.linalg.matrix_power(f, time - innovation)
    mean_array = np.asarray(means)
    if not all(np.isfinite(value).all() for value in (loading, noise, mean_array)):
        raise ValueError(
            "Gaussian reference numerical failure: nonfinite generative construction"
        )
    covariance = loading @ noise @ loading.T
    if not np.isfinite(covariance).all():
        raise ValueError(
            "Gaussian reference numerical failure: nonfinite joint covariance"
        )
    posterior_means, posterior_covariances = [], []
    for time in range(t):
        count = time + 1
        observation_map = np.kron(np.eye(count), h)
        observed_covariance = observation_map @ covariance[
            : count * n, : count * n
        ] @ observation_map.T + np.kron(np.eye(count), r)
        cross = covariance[time * n : (time + 1) * n, : count * n] @ observation_map.T
        if not np.isfinite(observed_covariance).all() or not np.isfinite(cross).all():
            raise ValueError(
                "Gaussian reference numerical failure: nonfinite observation covariance or cross covariance"
            )
        np.linalg.cholesky(observed_covariance)
        gain = np.linalg.solve(observed_covariance, cross.T).T
        predicted_observations = observation_map @ mean_array[:count].reshape(-1)
        residual = y[:count].reshape(-1) - predicted_observations
        if not all(
            np.isfinite(value).all()
            for value in (gain, predicted_observations, residual)
        ):
            raise ValueError(
                "Gaussian reference numerical failure: nonfinite gain, observation mean or residual"
            )
        posterior_mean = mean_array[time] + gain @ residual
        posterior_covariance = (
            covariance[time * n : (time + 1) * n, time * n : (time + 1) * n]
            - gain @ cross.T
        )
        if (
            not np.isfinite(posterior_mean).all()
            or not np.isfinite(posterior_covariance).all()
        ):
            raise ValueError(
                "Gaussian reference numerical failure: nonfinite conditioned mean or covariance"
            )
        posterior_means.append(posterior_mean)
        posterior_covariances.append(posterior_covariance)
    return np.asarray(posterior_means), np.asarray(posterior_covariances)
