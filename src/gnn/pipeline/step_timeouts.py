"""Step timeout configuration for the GNN pipeline.

Every registered step declares its budget here. A step without a declared
entry fails closed (:class:`StepTimeoutNotDeclared`) instead of inheriting a
silent default: during the 2026-11-02 full-corpus run an undeclared
``20_website.py`` was killed at the old implicit 180-second default after
202.6 s of legitimate work, and the receipt was indistinguishable from a
hang. The import below also refuses a registry step that has no budget, so a
new step cannot be added without naming the time it is allowed to take.

Each comment names the cost measured in that same run — 38 selected models,
serial ``main.py``, macOS arm64 (Apple M4 Pro), from
``pipeline_execution_summary.json`` — so every budget is auditable against a
measurement rather than a guess.

Environment overrides:
  - GNN_STEP_TIMEOUT_{N}: absolute timeout for step N (wins over scale).
  - GNN_STEP_TIMEOUT_SCALE: multiplier applied to every configured timeout
    (for slow-storage checkouts where process startup dominates; invalid or
    non-positive values are ignored).
"""

import logging
import os
from typing import Any, cast

from gnn.pipeline.step_registry import STEPS

logger = logging.getLogger(__name__)


class StepTimeoutNotDeclared(ValueError):
    """A step asked for a wall-clock budget that it never declared.

    Raised instead of returning a default. A step's budget is a design
    decision backed by a measurement; a silent fallback hides both.
    """


# Timeout configuration in seconds. Every entry is required: the keys are
# checked against ``step_registry.STEPS`` at import, and an unknown name is an
# error rather than a default.
STEP_TIMEOUTS: dict[str, Any] = {
    "0_template.py": 180,
    "1_setup.py": 900,  # `uv sync`; measured 5.9 s warm, network-bound cold
    "2_tests.py": {
        "default": 900,
        "comprehensive": 1200,
    },  # measured 65.7 s (fast subset)
    "3_gnn.py": 300,  # multi-format serialization incl. dense scaling-study tensors; 7.8 s
    "4_model_registry.py": 180,  # measured 0.2 s
    "5_type_checker.py": 180,  # measured 6.1 s
    "6_validation.py": 180,  # measured 1.8 s
    "7_export.py": 300,  # multi-format export incl. dense scaling-study tensors; 1.6 s
    "8_visualization.py": 600,  # matrix heatmaps incl. dense tensors; measured 215.2 s
    "9_advanced_viz.py": 600,  # interactive plots incl. dense tensors; measured 222.8 s
    "10_ontology.py": 180,  # measured 0.7 s
    "11_render.py": 600,  # raised from 300: measured 153.4 s was 51% of the old budget
    "12_execute.py": 7200,  # every model executed across all frameworks; measured 1381.5 s
    # Fallback only: with a run context, summary_wiring replaces step 13's
    # budget with `llm.timeout_seconds` else 600 s per selected model (22800 s
    # for the 38-model corpus), and run_context.py does the same for the total
    # deadline. This value applies when no run context exists.
    "13_llm.py": 900,
    "14_ml_integration.py": 180,  # measured 0.4 s
    "15_audio.py": 600,  # measured 5.1 s with no audio backend; synthesis is backend-bound
    "16_analysis.py": 900,  # per-model analysis incl. dense tensors; measured 358.6 s
    "17_integration.py": 300,  # dependency graph + system checks; measured 6.5 s
    "18_security.py": 180,  # measured 0.7 s
    "19_research.py": 180,  # measured 0.6 s
    # Measured 271.0 s standalone over the full-corpus output; the 180 s
    # implicit default killed the in-run instance at 202.6 s, mid-asset-copy.
    # The cost is dominated by copying producer visualization trees into
    # assets/ (also copied once per HTML artifact by the website module), so
    # this is a safety net, not a target.
    "20_website.py": 900,
    "21_mcp.py": 180,  # measured 4.2 s
    "22_gui.py": 600,  # measured 2.9 s (headless default)
    "23_report.py": 180,  # measured 4.4 s
    "24_intelligent_analysis.py": 600,  # LLM-backed; measured 11.5 s with the rule-based fallback
}


def _undeclared_registry_steps() -> list[str]:
    """Registered pipeline steps without a declared timeout."""
    return [step.script_name for step in STEPS if step.script_name not in STEP_TIMEOUTS]


_UNDECLARED_STEPS = _undeclared_registry_steps()
if _UNDECLARED_STEPS:  # pragma: no cover - import-time contract failure
    raise StepTimeoutNotDeclared(
        "pipeline steps without a declared timeout: "
        + ", ".join(_UNDECLARED_STEPS)
        + "; declare each budget in gnn.pipeline.step_timeouts.STEP_TIMEOUTS"
    )


def get_step_timeout(script_name: str, comprehensive: bool = False) -> int:
    """Get timeout for a pipeline step.

    Supports environment variable override: GNN_STEP_TIMEOUT_{STEP_NUMBER}
    e.g., GNN_STEP_TIMEOUT_2=1800 overrides 2_tests.py timeout.

    Raises:
        StepTimeoutNotDeclared: when *script_name* (or the requested
            ``comprehensive``/``default`` variant) has no declared budget.
            An override cannot invent one: the declared entry is what the
            pipeline is designed around.
    """
    timeout_config = STEP_TIMEOUTS.get(script_name)
    if timeout_config is None:
        raise StepTimeoutNotDeclared(
            f"{script_name!r} has no declared timeout; add it to "
            "gnn.pipeline.step_timeouts.STEP_TIMEOUTS (a missing entry is not a default)"
        )

    # Check env var override first
    step_num = script_name.split("_")[0] if "_" in script_name else ""
    env_key = f"GNN_STEP_TIMEOUT_{step_num}"
    env_val = os.environ.get(env_key)
    if env_val:
        try:
            return int(env_val)
        except ValueError as e:
            logger.debug("Invalid timeout value in %s: %s", env_key, e)

    if isinstance(timeout_config, dict):
        variant = "comprehensive" if comprehensive else "default"
        value = timeout_config.get(variant)
        if value is None:
            raise StepTimeoutNotDeclared(
                f"{script_name!r} declares no {variant!r} timeout; "
                "both variants must be named explicitly"
            )
        return _scale_timeout(cast("int", value))
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
