#!/usr/bin/env python3
"""Render continuous GNN models to standalone cpomdp simulation scripts.

The script filters with cpomdp's exact Kalman backend and scores finite
action sequences. Discrete POMDP specs get the ``unsupported``
status: cpomdp is the continuous-state sibling of pymdp.

@Web: https://github.com/inferogenesis/cpomdp
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from gnn.render.continuous_common import extract_continuous_spec, is_continuous_spec
from gnn.render.cpomdp.script_template import (
    generate_cpomdp_script,
    resolve_render_options,
)
from gnn.render.naming import atomic_write_text

logger = logging.getLogger(__name__)

UNSUPPORTED_MESSAGE = "cpomdp renders continuous (linear-Gaussian) models only"


def _source_declarations(gnn_spec: dict[str, Any], key: str) -> list[Any]:
    canonical = gnn_spec.get("initialparameterization")
    alternate = gnn_spec.get("initial_parameterization")
    declarations = [
        mapping[key]
        for mapping in (canonical, alternate)
        if isinstance(mapping, dict) and key in mapping
    ]
    initial = canonical or alternate
    if declarations and (not isinstance(initial, dict) or key not in initial):
        raise ValueError(f"{key} declarations differ between parameter mappings")
    return declarations


def render_gnn_to_cpomdp(
    gnn_spec: dict[str, Any],
    output_path: Path,
    options: dict[str, Any] | None = None,
) -> tuple[bool, str, list[str]]:
    """Render a continuous GNN specification to a cpomdp simulation script.

    Args:
        gnn_spec: Parsed GNN model specification (continuous branch).
        output_path: Path to write the generated script.
        options: ``control_mode`` (``"efe"`` or ``"parity"``), ``action_scale``,
            ``horizon`` and ``goal_precision``. See ``resolve_render_options``.

    Returns:
        ``(success, message, artifact_paths)``. A discrete spec returns
        ``(False, UNSUPPORTED_MESSAGE, [])`` so the processor records it under
        ``unsupported_framework_renderings``. An invalid spec, a bad option or
        a failed write returns ``(False, reason, [])``.
    """
    from gnn.render.execution_contracts import unsupported_contract

    refusal = unsupported_contract(gnn_spec, "cpomdp")
    if refusal:
        return False, refusal, []

    if not is_continuous_spec(gnn_spec):
        return False, UNSUPPORTED_MESSAGE, []

    try:
        from gnn.render.cpomdp.finite_controls import candidate_search, finite_controls
        from gnn.render.pomdp_contract import (
            detect_model_kinds,
            unsupported_composition_reason,
        )

        kinds = detect_model_kinds(gnn_spec)
        if len(kinds) > 1:
            return False, unsupported_composition_reason(kinds), []
        spec = extract_continuous_spec(gnn_spec)
        resolved = resolve_render_options(options)
        declarations = _source_declarations(gnn_spec, "cpomdp_control")
        if declarations:
            control = finite_controls(declarations[0], spec.n)
            if any(
                finite_controls(declared, spec.n) != control
                for declared in declarations[1:]
            ):
                raise ValueError(
                    "cpomdp_control declarations differ between parameter mappings"
                )
            if not spec.has_control:
                raise ValueError("cpomdp_control requires goal_mean and control_gain")
            if resolved["control_mode"] != "efe" or (
                options and "action_scale" in options
            ):
                raise ValueError(
                    "cpomdp_control requires efe mode and no action_scale override"
                )
            resolved["finite_controls"] = control
        searches = _source_declarations(gnn_spec, "cpomdp_search")
        if searches:
            if "finite_controls" not in resolved:
                raise ValueError(
                    "cpomdp_search requires source-declared cpomdp_control"
                )
            action_ids = resolved["finite_controls"]["action_ids"]
            search = candidate_search(searches[0], action_ids)
            if any(candidate_search(raw, action_ids) != search for raw in searches[1:]):
                raise ValueError(
                    "cpomdp_search declarations differ between parameter mappings"
                )
            if (
                options
                and "horizon" in options
                and resolved["horizon"] != search["horizon"]
            ):
                raise ValueError(
                    "cpomdp_search horizon conflicts with the renderer option"
                )
            resolved["horizon"] = search["horizon"]
            resolved["candidate_search"] = search
        contract = {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in vars(spec).items()
        }
        if "finite_controls" in resolved:
            contract["cpomdp_control"] = resolved["finite_controls"]
        if "candidate_search" in resolved:
            contract["cpomdp_search"] = resolved["candidate_search"]
        resolved["source_identity"] = {
            "model_contract_sha256": hashlib.sha256(
                json.dumps(contract, sort_keys=True).encode()
            ).hexdigest(),
            "claim": "identity of the validated numeric model; extraction correctness is separate",
        }
        from gnn.pipeline.run_context import current_run_context

        context = current_run_context()
        if context and Path(output_path).resolve().is_relative_to(
            Path(context.output_root)
        ):
            for model in context.selected_models(11):
                if model.artifact_stem in Path(output_path).parts:
                    resolved["source_identity"].update(
                        model_id=model.model_id,
                        source_relative_path=model.relative_path,
                        source_sha256=model.sha256,
                    )
                    break
        code = generate_cpomdp_script(spec, resolved)
        output_path = Path(output_path)
        atomic_write_text(output_path, code)
        mode = resolved["control_mode"] if spec.has_control else "passive"
        if searches:
            mode = "efe with declared candidates"
        logger.info(f"✅ cpomdp continuous script written to: {output_path}")
        return (
            True,
            f"cpomdp continuous script generated ({mode}): {output_path}",
            [str(output_path)],
        )
    except (ValueError, TypeError, OSError) as e:
        logger.error(f"❌ cpomdp rendering failed: {e}")
        return False, f"cpomdp rendering failed: {e}", []
