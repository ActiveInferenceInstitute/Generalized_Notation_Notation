"""Public backend option admission, independent of native execution readiness.

Only options consumed by maintained renderers are admitted. Scientific and
allocation validation stays with each adapter; registration permits code
generation and does not certify an installed execution runtime.
"""

from __future__ import annotations

import math
from typing import Any

from gnn.frameworks import RENDER_FRAMEWORKS

# These public spec targets select distinct generators on the same backends.
SPEC_TARGET_BACKENDS = {"jax_pomdp": "jax", "discopy_combined": "discopy"}
SPEC_RENDER_TARGETS = (*RENDER_FRAMEWORKS, *SPEC_TARGET_BACKENDS)

_OPTION_SCHEMAS: dict[str, dict[str, Any]] = {
    "pymdp": {"mode": {"type": "string", "enum": ["pipeline", "standalone"]}},
    "rxinfer": {"inference_mode": {"type": "string", "enum": ["batch", "online"]}},
    "pytorch": {"num_timesteps": {"type": "integer", "minimum": 1}},
    "numpyro": {"num_timesteps": {"type": "integer", "minimum": 1}},
    "jax": {
        "seed": {
            "type": "integer",
            "minimum": 0,
            "description": "Execution-contract or discrete factorized models only",
        },
        "num_timesteps": {
            "type": "integer",
            "minimum": 1,
            "description": "Declared execution-contract models only",
        },
        "num_fast_transitions": {
            "type": "integer",
            "minimum": 1,
            "description": "Declared execution-contract models only",
        },
        "action_precision": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "Discrete factorized models only",
        },
    },
    "discopy": {"matrix_permutations": {"type": "object"}},
    "cpomdp": {
        "control_mode": {"type": "string", "enum": ["efe", "parity"]},
        "action_scale": {"type": "number", "exclusiveMinimum": 0},
        "horizon": {"type": "integer", "minimum": 1},
        "goal_precision": {"type": "number", "exclusiveMinimum": 0},
        "max_policies": {"type": "integer", "minimum": 1},
        "max_step_evaluations": {"type": "integer", "minimum": 1},
        "max_estimated_bytes": {"type": "integer", "minimum": 1},
    },
    "thrml": {
        "num_timesteps": {"type": "integer", "minimum": 1},
        "num_samples": {"type": "integer", "minimum": 1},
        "burn_in": {"type": "integer", "minimum": 0},
        "thin": {"type": "integer", "minimum": 1},
        "seed": {"type": "integer", "minimum": 0, "maximum": 2**32 - 1},
        "observations": {"type": ["array", "object"]},
        "transition_actions": {"type": ["array", "object"]},
    },
}


def render_options_schema(framework: str) -> dict[str, Any]:
    """Return an isolated public option schema for one registered backend."""
    from copy import deepcopy

    if framework not in RENDER_FRAMEWORKS:
        raise ValueError(f"Unknown render framework: {framework!r}")
    return {
        "type": "object",
        "properties": deepcopy(_OPTION_SCHEMAS.get(framework, {})),
        "additionalProperties": False,
    }


def render_options_inventory(
    *, spec_targets: bool = False
) -> dict[str, dict[str, Any]]:
    """Expose live typed options for schema/discovery consumers of each interface."""
    names = SPEC_RENDER_TARGETS if spec_targets else RENDER_FRAMEWORKS
    inventory = {
        name: render_options_schema(SPEC_TARGET_BACKENDS.get(name, name))
        for name in names
    }
    if spec_targets:
        # Distinct generator APIs do not consume Step 11 matrix permutations.
        for name in ("discopy", "discopy_combined", "jax_pomdp"):
            inventory[name] = {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }
    return inventory


def validate_render_options(framework: str, options: Any) -> dict[str, Any]:
    """Validate user options without silently coercing or discarding values."""
    schema = render_options_schema(framework)
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise ValueError(f"{framework} options must be a mapping")
    properties = schema["properties"]
    unknown = set(options) - properties.keys()
    if unknown:
        raise ValueError(f"Unknown {framework} options: {sorted(unknown)}")
    for key, value in options.items():
        rule = properties[key]
        kind = rule["type"]
        if kind == "string":
            valid = isinstance(value, str)
        elif kind == "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif kind == "number":
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )
        elif kind == "array":
            valid = isinstance(value, list)
        elif kind == ["array", "object"]:
            valid = isinstance(value, (list, dict))
        else:
            valid = isinstance(value, dict)
        if not valid:
            raise ValueError(f"{framework}.{key} must be a {kind}")
        if "enum" in rule and value not in rule["enum"]:
            raise ValueError(f"{framework}.{key} must be one of {rule['enum']}")
        if (
            ("minimum" in rule and value < rule["minimum"])
            or ("maximum" in rule and value > rule["maximum"])
            or ("exclusiveMinimum" in rule and value <= rule["exclusiveMinimum"])
        ):
            raise ValueError(f"{framework}.{key} is outside its admitted range")
    return dict(options)


def validate_model_render_options(
    framework: str, options: dict[str, Any], spec: dict[str, Any]
) -> None:
    """Reject options that the selected JAX model generator would ignore."""
    if framework != "jax" or not options:
        return
    from gnn.render.execution_contracts import execution_contract

    if execution_contract(spec) is not None:
        admitted = {"seed", "num_timesteps", "num_fast_transitions"}
    else:
        from gnn.render.continuous_common import is_continuous_spec
        from gnn.render.jax.jax_factorized_generator import _is_factorized_spec

        admitted = (
            {"seed", "action_precision"}
            if not is_continuous_spec(spec) and _is_factorized_spec(spec)
            else set()
        )
    ignored = set(options) - admitted
    if ignored:
        raise ValueError(
            f"JAX options are unsupported for this model generator: {sorted(ignored)}"
        )


def merge_backend_options(configured: Any, explicit: Any) -> dict[str, dict[str, Any]]:
    """Merge option keys per backend, explicit keys overriding configured keys."""
    result: dict[str, dict[str, Any]] = {}
    for label, layer in (("configured", configured), ("explicit", explicit)):
        if not isinstance(layer, dict):
            raise ValueError(f"{label} backend_options must be a mapping")
        for framework, options in layer.items():
            checked = validate_render_options(framework, options)
            result[framework] = {**result.get(framework, {}), **checked}
    return result
