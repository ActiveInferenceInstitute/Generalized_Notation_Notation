"""Strict categorical renderer and resolved backend-option boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gnn.render.naming import atomic_write_text
from gnn.render.thrml.adapter import (
    OPTION_KEYS,
    UnsupportedTHRMLModel,
    build_thrml_payload,
)
from gnn.render.thrml.script_template import generate_thrml_script

COORDINATOR_KEYS = {
    "run_id",
    "logger",
    "verbose",
    "framework",
    "frameworks",
    "simulation_params",
    "backend_options",
    "timesteps",
    "deadline",
    "timeout",
    "resolved_config",
    "execution_context",
    "model_id",
    "source_sha256",
    "source_relative_path",
    "config",
    "input_config",
    "recursive",
    "strict_validation",
    "strict_framework_success",
}


def resolve_render_options(options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge configured and CLI THRML options, with direct values last."""
    incoming = dict(options or {})
    unknown = set(incoming) - OPTION_KEYS - COORDINATOR_KEYS
    if unknown:
        raise ValueError(f"unknown THRML options: {sorted(unknown)}")
    result: dict[str, Any] = {}
    backend = incoming.get("backend_options")
    if backend is None:
        backend = {}
    if not isinstance(backend, dict):
        raise ValueError("backend_options must be a mapping")
    configured = backend.get("thrml", {})
    simulation = incoming.get("simulation_params")
    if simulation is None:
        simulation = {}
    if isinstance(simulation, str):
        simulation = json.loads(simulation)
    if not isinstance(simulation, dict):
        raise ValueError("simulation_params must be a mapping")
    command = simulation.get("thrml", {})
    for layer in (configured, command):
        if not isinstance(layer, dict) or set(layer) - OPTION_KEYS:
            raise ValueError(
                "THRML backend configuration contains unknown or malformed options"
            )
        result.update(layer)
    if incoming.get("timesteps") is not None:
        result["num_timesteps"] = incoming["timesteps"]
    result.update({key: value for key, value in incoming.items() if key in OPTION_KEYS})
    return result


def render_gnn_to_thrml(
    gnn_spec: dict[str, Any], output_path: Path, options: dict[str, Any] | None = None
) -> tuple[bool, str, list[str]]:
    """Render admitted models; unsupported semantics produce no script."""
    from gnn.render.execution_contracts import unsupported_contract

    refusal = unsupported_contract(gnn_spec, "thrml")
    if refusal:
        return False, refusal, []

    try:
        if not isinstance(gnn_spec, dict):
            raise ValueError("THRML source must be a mapping")
        parameters = gnn_spec.get("model_parameters", {})
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise ValueError("model_parameters must be a mapping")
        # Model-side simulation options are explicit configuration, not samples.
        base = {"simulation_params": parameters.get("simulation_params", {})}
        if options:
            base.update(options)
        payload = build_thrml_payload(gnn_spec, resolve_render_options(base))
        from gnn.pipeline.run_context import current_run_context

        context = current_run_context()
        if context and Path(output_path).resolve().is_relative_to(
            Path(context.output_root)
        ):
            for model in context.selected_models(11):
                if model.artifact_stem in Path(output_path).parts:
                    payload["source_identity"].update(
                        model_id=model.model_id,
                        artifact_stem=model.artifact_stem,
                        source_relative_path=model.relative_path,
                        source_sha256=model.sha256,
                    )
                    break
        atomic_write_text(output_path, generate_thrml_script(payload))
        return (
            True,
            f"experimental THRML categorical Gibbs script generated: {output_path}",
            [str(output_path)],
        )
    except UnsupportedTHRMLModel as error:
        return False, f"unsupported-thrml: {error}", []
    except (ValueError, TypeError, OSError, KeyError) as error:
        return False, f"THRML rendering failed: {error}", []
