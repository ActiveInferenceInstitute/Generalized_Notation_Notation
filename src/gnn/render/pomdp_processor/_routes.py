#!/usr/bin/env python3
"""Renderer routes and dispatch entries for the POMDP render processor."""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal

from gnn.render.naming import safe_output_stem


def _continuous_shape(value: Any) -> List[int]:
    """Best-effort nested shape for continuous parameter values (scalars → [])."""
    shape: List[int] = []
    current = value
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return shape


def _safe_output_stem(value: Any, fallback: str = "pomdp_model") -> str:
    return safe_output_stem(value, fallback)


@dataclass(frozen=True)
class RendererRoute:
    """Declarative dispatch entry for a framework renderer."""

    module: str
    function: str
    suffix: str
    label: str
    validate: bool
    options_mode: Literal["kwargs", "timesteps", "kwargs_or_none"]
    result_mode: Literal["warnings", "artifacts"]
    artifacts_mode: Literal["primary_plus_metadata", "primary", "returned"]


RENDERER_ROUTES: Dict[str, RendererRoute] = {
    "pymdp": RendererRoute(
        module=".pymdp.pymdp_renderer",
        function="render_gnn_to_pymdp",
        suffix="_pymdp.py",
        label="PyMDP",
        validate=True,
        options_mode="kwargs",
        result_mode="warnings",
        artifacts_mode="primary_plus_metadata",
    ),
    "rxinfer": RendererRoute(
        module=".rxinfer.rxinfer_renderer",
        function="render_gnn_to_rxinfer",
        suffix="_rxinfer.jl",
        label="RxInfer",
        validate=True,
        options_mode="kwargs",
        result_mode="warnings",
        artifacts_mode="primary",
    ),
    "activeinference_jl": RendererRoute(
        module=".activeinference_jl.activeinference_renderer",
        function="render_gnn_to_activeinference_jl",
        suffix="_activeinference.jl",
        label="ActiveInference.jl",
        validate=True,
        options_mode="kwargs",
        result_mode="artifacts",
        artifacts_mode="returned",
    ),
    "jax": RendererRoute(
        module=".jax.jax_renderer",
        function="render_gnn_to_jax",
        suffix="_jax.py",
        label="JAX",
        validate=True,
        options_mode="kwargs",
        result_mode="artifacts",
        artifacts_mode="returned",
    ),
    "discopy": RendererRoute(
        module=".discopy.discopy_renderer",
        function="render_gnn_to_discopy",
        suffix="_discopy.py",
        label="DisCoPy",
        validate=False,
        options_mode="kwargs",
        result_mode="warnings",
        artifacts_mode="primary",
    ),
    "pytorch": RendererRoute(
        module=".pytorch.pytorch_renderer",
        function="render_gnn_to_pytorch",
        suffix="_pytorch.py",
        label="PyTorch",
        validate=False,
        options_mode="timesteps",
        result_mode="artifacts",
        artifacts_mode="returned",
    ),
    "numpyro": RendererRoute(
        module=".numpyro.numpyro_renderer",
        function="render_gnn_to_numpyro",
        suffix="_numpyro.py",
        label="NumPyro",
        validate=False,
        options_mode="timesteps",
        result_mode="artifacts",
        artifacts_mode="returned",
    ),
    "stan": RendererRoute(
        module=".stan.stan_renderer",
        function="render_gnn_to_stan",
        suffix="_stan.py",
        label="Stan",
        validate=False,
        options_mode="kwargs_or_none",
        result_mode="artifacts",
        artifacts_mode="returned",
    ),
    "cpomdp": RendererRoute(
        module=".cpomdp.cpomdp_renderer",
        function="render_gnn_to_cpomdp",
        suffix="_cpomdp.py",
        label="cpomdp (experimental)",
        validate=False,
        options_mode="kwargs",
        result_mode="artifacts",
        artifacts_mode="returned",
    ),
    "thrml": RendererRoute(
        module=".thrml.thrml_renderer",
        function="render_gnn_to_thrml",
        suffix="_thrml.py",
        label="THRML (experimental)",
        validate=False,
        options_mode="kwargs",
        result_mode="artifacts",
        artifacts_mode="returned",
    ),
    "ngclearn": RendererRoute(
        module=".ngclearn.ngclearn_renderer",
        function="render_gnn_to_ngclearn",
        suffix="_ngclearn.py",
        label="ngc-learn",
        validate=True,
        options_mode="kwargs",
        result_mode="artifacts",
        artifacts_mode="returned",
    ),
}
