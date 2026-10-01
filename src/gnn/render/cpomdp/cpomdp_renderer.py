#!/usr/bin/env python3
"""Render continuous GNN models to standalone cpomdp simulation scripts.

The script filters with cpomdp's exact Kalman backend and scores every policy
over a declared finite action set. Discrete POMDP specs get the ``unsupported``
status: cpomdp is the continuous-state sibling of pymdp.

@Web: https://github.com/inferogenesis/cpomdp
"""

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
