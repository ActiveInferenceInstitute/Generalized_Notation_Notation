#!/usr/bin/env python3
"""Render continuous GNN models to standalone cpomdp simulation scripts.

The script filters with cpomdp's exact Kalman backend and scores every policy
over a declared finite action set. Discrete POMDP specs get the ``unsupported``
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
    if not is_continuous_spec(gnn_spec):
        return False, UNSUPPORTED_MESSAGE, []

    try:
        spec = extract_continuous_spec(gnn_spec)
        resolved = resolve_render_options(options)
        contract = {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in vars(spec).items()
        }
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
        logger.info(f"✅ cpomdp continuous script written to: {output_path}")
        return (
            True,
            f"cpomdp continuous script generated ({mode}): {output_path}",
            [str(output_path)],
        )
    except (ValueError, TypeError, OSError) as e:
        logger.error(f"❌ cpomdp rendering failed: {e}")
        return False, f"cpomdp rendering failed: {e}", []
