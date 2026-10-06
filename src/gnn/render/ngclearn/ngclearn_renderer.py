#!/usr/bin/env python3
"""
ngc-learn Renderer for GNN Specifications

Renders continuous (linear-Gaussian) GNN models to standalone ngc-learn
simulation scripts. Codegen-only (pytorch precedent): the renderer never
imports ``ngclearn`` — the generated script does, behind an import-or-exit
guard. Discrete POMDPs have no ngc-learn representation and are reported
first-class unsupported.

@Web: https://github.com/NACLab/ngc-learn
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gnn.render.continuous_common import extract_continuous_spec, is_continuous_spec
from gnn.render.continuous_script import generate_continuous_script
from gnn.render.naming import atomic_write_text

logger = logging.getLogger(__name__)

#: First-class unsupported refusal for every non-continuous model kind.
_DISCRETE_MESSAGE = (
    "discrete POMDP: ngclearn supports continuous linear-Gaussian models only"
)


def render_gnn_to_ngclearn(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]:
    """Render a GNN specification to a standalone ngc-learn simulation script.

    Continuous linear-Gaussian models are emitted through the shared
    continuous-script generator with ``backend="ngclearn"``: the Kalman
    numerics stay byte-identical to the ``jax`` backend so ``rmse_vs_true``
    remains apples-to-apples across backends. Discrete POMDPs (and other
    non-continuous kinds) are first-class unsupported — ngc-learn has no
    categorical A/B/C-D machinery.

    Args:
        gnn_spec: GNN specification dictionary
        output_path: File path for the generated script
        options: Additional options (unused; shared generator contract)

    Returns:
        Tuple of (success, message, generated_files)
    """
    from gnn.render.execution_contracts import unsupported_contract

    refusal = unsupported_contract(gnn_spec, "ngclearn")
    if refusal:
        return False, refusal, []

    try:
        if not is_continuous_spec(gnn_spec):
            return False, _DISCRETE_MESSAGE, []
        from gnn.render.pomdp_contract import ModelKind, detect_model_kinds

        if detect_model_kinds(gnn_spec) == frozenset(
            {ModelKind.FACTORED, ModelKind.CONTINUOUS}
        ):
            return (
                False,
                "unsupported-factored-continuous: ngclearn renders the flat "
                "linear-Gaussian family only; per-factor compositions are "
                "refused rather than silently rendered flat",
                [],
            )
        spec = extract_continuous_spec(gnn_spec)
        code = generate_continuous_script(spec, "ngclearn")
        output_file = atomic_write_text(Path(output_path), code)
        logger.info(f"✅ ngc-learn continuous script written to: {output_file}")
        return (
            True,
            f"ngc-learn continuous LGSSM script generated: {output_file}",
            [str(output_file)],
        )
    except Exception as e:
        logger.error(f"❌ ngc-learn rendering failed: {e}")
        return False, f"ngc-learn rendering failed: {e}", []
