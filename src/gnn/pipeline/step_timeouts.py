"""Step timeout configuration for the GNN pipeline.

Environment overrides:
  - GNN_STEP_TIMEOUT_{N}: absolute timeout for step N (wins over scale).
  - GNN_STEP_TIMEOUT_SCALE: multiplier applied to every configured timeout
    (for slow-storage checkouts where process startup dominates; invalid or
    non-positive values are ignored).
  - GNN_PIPELINE_DEADLINE: epoch seconds by which the whole run must finish,
    armed by ``main`` from ``pipeline.timeout.total`` in input/config.yaml.
    Every step timeout is capped at the remaining budget.
"""

import logging
import os
import time
from typing import Any, Optional, cast

logger = logging.getLogger(__name__)

# Timeout configuration in seconds
STEP_TIMEOUTS: dict[str, Any] = {
    "2_tests.py": {"default": 900, "comprehensive": 1200},
    "3_gnn.py": 300,  # multi-format serialization incl. dense scaling-study tensors
    "7_export.py": 300,  # multi-format export incl. dense scaling-study tensors
    "8_visualization.py": 600,  # matrix heatmaps incl. dense scaling-study tensors (~6.5 min)
    "9_advanced_viz.py": 600,
    "13_llm.py": 900,  # 72 LLM calls (9 prompts × 8 files), ~12s each
    "16_analysis.py": 900,  # per-model analysis incl. dense scaling-study tensors
    "17_integration.py": 300,  # Dependency graph + system checks
    "22_gui.py": 600,
    "11_render.py": 300,
    "12_execute.py": 7200,  # every model executed across all frameworks (scaling study is heavy)
}

DEFAULT_TIMEOUT = 180

PIPELINE_DEADLINE_ENV = "GNN_PIPELINE_DEADLINE"


def pipeline_budget_remaining() -> Optional[float]:
    """Seconds left before ``GNN_PIPELINE_DEADLINE``, or None when no deadline is armed."""
    raw = os.environ.get(PIPELINE_DEADLINE_ENV)
    if not raw:
        return None
    try:
        return float(raw) - time.time()
    except ValueError:
        logger.debug("Invalid %s value: %s", PIPELINE_DEADLINE_ENV, raw)
        return None


def _cap_to_pipeline_budget(seconds: int) -> int:
    remaining = pipeline_budget_remaining()
    if remaining is None:
        return seconds
    return max(1, min(seconds, int(remaining)))


def get_step_timeout(script_name: str, comprehensive: bool = False) -> int:
    """Get timeout for a pipeline step.

    Supports environment variable override: GNN_STEP_TIMEOUT_{STEP_NUMBER}
    e.g., GNN_STEP_TIMEOUT_2=1800 overrides 2_tests.py timeout. The result is
    capped at the remaining ``GNN_PIPELINE_DEADLINE`` budget when one is armed.
    """
    return _cap_to_pipeline_budget(_configured_step_timeout(script_name, comprehensive))


def _configured_step_timeout(script_name: str, comprehensive: bool) -> int:
    # Check env var override first
    step_num = script_name.split("_")[0] if "_" in script_name else ""
    env_key = f"GNN_STEP_TIMEOUT_{step_num}"
    env_val = os.environ.get(env_key)
    if env_val:
        try:
            return int(env_val)
        except ValueError as e:
            logger.debug("Invalid timeout value in %s: %s", env_key, e)

    timeout_config = STEP_TIMEOUTS.get(script_name)
    if timeout_config is None:
        return _scale_timeout(DEFAULT_TIMEOUT)
    if isinstance(timeout_config, dict):
        return _scale_timeout(
            cast(
                "int",
                timeout_config.get(
                    "comprehensive" if comprehensive else "default", DEFAULT_TIMEOUT
                ),
            )
        )
    return _scale_timeout(cast("int", timeout_config))


def _scale_timeout(seconds: int) -> int:
    """Apply the GNN_STEP_TIMEOUT_SCALE multiplier.

    On slow storage (e.g. external USB drives where every ``uv run`` re-reads
    the virtualenv), a single global multiplier is simpler than overriding each
    step individually. ``GNN_STEP_TIMEOUT_SCALE=3`` triples every configured
    timeout; invalid values are ignored so a typo cannot zero out a step.
    """
    raw = os.environ.get("GNN_STEP_TIMEOUT_SCALE")
    if not raw:
        return seconds
    try:
        scale = float(raw)
    except ValueError as e:
        logger.debug("Invalid GNN_STEP_TIMEOUT_SCALE %r: %s", raw, e)
        return seconds
    if scale <= 0:
        logger.debug("Ignoring non-positive GNN_STEP_TIMEOUT_SCALE %r", raw)
        return seconds
    return max(1, int(seconds * scale))
