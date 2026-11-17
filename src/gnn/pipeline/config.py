#!/usr/bin/env python3
"""
Pipeline configuration module.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, cast

import yaml

logger = logging.getLogger(__name__)

# PyYAML is a core dependency (``pyyaml>=6.0`` in pyproject.toml). Import it
# unconditionally so a broken environment fails loud at import time instead
# of silently discarding the user's YAML configuration at load time.
_CONFIG_PARSE_ERRORS: tuple[type[Exception], ...] = (
    json.JSONDecodeError,
    OSError,
    ValueError,
    yaml.YAMLError,
)

# Canonical default paths used across the pipeline package (single source).
DEFAULT_TARGET_DIR = "input/gnn_files"
DEFAULT_OUTPUT_DIR = "output"


# Pipeline configuration derived from canonical step registry
from gnn.pipeline.step_registry import (
    STEP_METADATA_DICT as STEP_METADATA,  # noqa: E402,F401
)


class StepConfig:
    """Configuration for a pipeline step."""

    def __init__(self, step_name: str, **kwargs: Any) -> None:
        """Initialize the instance."""
        self.step_name = step_name
        self.enabled = kwargs.get("enabled", True)
        self.timeout = kwargs.get("timeout", 3600)
        self.retries = kwargs.get("retries", 3)
        self.dependencies = kwargs.get("dependencies", [])
        self.parameters = kwargs.get("parameters", {})
        self.required = kwargs.get("required", True)
        self.output_subdir = kwargs.get(
            "output_subdir", f"{step_name.removesuffix('.py')}_output"
        )
        self.performance_tracking = kwargs.get("performance_tracking", True)


class PipelineConfig:
    """Pipeline configuration manager."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize the instance."""
        self.config_path = config_path or Path("pipeline_config.yaml")
        self.config = self._load_settings_from_path()

    def _load_settings_from_path(self) -> Dict[str, Any]:
        """Load pipeline settings dict from ``config_path`` (YAML or JSON)."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    if self.config_path.suffix in (".yaml", ".yml"):
                        return yaml.safe_load(f) or {}
                    else:
                        return cast("dict[str, Any]", json.load(f))
            except _CONFIG_PARSE_ERRORS as e:
                logger.error("Could not parse config file %s: %s", self.config_path, e)
                return {}
        return {}

    @property
    def steps(self) -> Dict[str, StepConfig]:
        """Return configured step objects keyed by canonical step script name."""
        config_steps = self.config.get("steps", {})
        if isinstance(config_steps, dict):
            return {
                step_name: StepConfig(step_name, **step_data)
                for step_name, step_data in config_steps.items()
                if isinstance(step_data, dict)
            }
        return {
            f"{step_name}.py": self.get_step_config(f"{step_name}.py")
            for step_name in STEP_METADATA
        }

    def get_step_config(self, step_name: str) -> StepConfig:
        """Get configuration for a specific step."""
        step_data = self.config.get(step_name, {})
        return StepConfig(step_name, **step_data)

    def save_config(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_path, "w") as f:
                if self.config_path.suffix in (".yaml", ".yml"):
                    yaml.dump(self.config, f)
                else:
                    json.dump(self.config, f, indent=2)
        except (OSError, ValueError) as e:
            logger.debug("Config save failed (%s), attempting JSON recovery", e)
            try:
                json_path = self.config_path.with_suffix(".json")
                with open(json_path, "w") as jf:
                    json.dump(self.config, jf, indent=2)
            except (OSError, TypeError, ValueError) as e:
                logger.debug("JSON recovery config save also failed: %s", e)


def get_pipeline_config() -> dict:
    """Get pipeline configuration as a plain dict for compatibility with tests."""
    cfg = PipelineConfig()
    data = cfg.config if isinstance(cfg.config, dict) else {}
    # Ensure required keys exist for tests with sensible defaults
    if "steps" not in data:
        data["steps"] = list(STEP_METADATA.keys())
    if "timeout" not in data:
        data["timeout"] = 3600
    if "parallel" not in data:
        data["parallel"] = True
    return data


def get_pipeline_config_dict() -> Dict[str, Any]:
    """Get the pipeline configuration as a plain dict (compatibility helper)."""
    cfg = PipelineConfig()
    return cfg.config if isinstance(cfg.config, dict) else {}


def set_pipeline_config(config: PipelineConfig) -> None:
    """Set the pipeline configuration."""
    config.save_config()


def get_output_dir_for_script(script_name: str, base_output_dir: Path) -> Path:
    """Get output directory for a specific script.

    Use a consistent numbered '<N_name>_output' directory for every step to
    keep the pipeline coherent and simple.

    Args:
        script_name: Name of the pipeline script (e.g., "3_gnn.py" or "3_gnn")
        base_output_dir: Base output directory (usually "output/")

    Returns:
        Path to step-specific output directory

    Note:
        This function prevents nested directories by detecting if base_output_dir
        already ends with the expected output directory name.
    """
    from gnn.pipeline.step_registry import output_dir_for_stem

    script_stem = Path(script_name).stem
    normalized = script_stem  # e.g., '7_export'

    # Try exact match from step registry
    mapped = output_dir_for_stem(normalized)
    if mapped is not None:
        result = base_output_dir / mapped
        # Prevent nesting
        if base_output_dir.name == mapped:
            return base_output_dir
        return result

    # Get expected output directory name for this script
    expected_dir_name = f"{normalized}_output"
    # Check if base_output_dir already ends with the expected directory name
    # This prevents nested directories like "10_ontology_output/10_ontology_output"
    if base_output_dir.name == expected_dir_name:
        # Already at the correct output directory - return as is
        return base_output_dir

    # Check if base_output_dir is already inside a step output directory
    # (e.g., "output/10_ontology_output/subdir" should not create another layer)
    if "_output" in base_output_dir.name and base_output_dir.parent.name != "output":
        # We're inside a step output directory - use the parent's parent as base
        # This handles cases like passing "output/10_ontology_output/results" as base
        actual_base = base_output_dir
        while (
            actual_base.name.endswith("_output") or "_output" in actual_base.parts[-2:]
        ):
            if actual_base.parent.name == "output":
                break
            actual_base = actual_base.parent
        base_output_dir = (
            actual_base.parent if actual_base.name.endswith("_output") else actual_base
        )

    # Default recovery (unregistered script name): warn loudly so a typo like
    # "5_typecheck.py" (vs the registered "5_type_checker") cannot silently
    # make producer and consumer disagree about the output directory.
    logger.warning(
        "Unregistered script name %r; using fallback output dir %r",
        script_name,
        expected_dir_name,
    )
    return base_output_dir / expected_dir_name


def resolve_step_output_dir(step_stem: str, output_dir: Path) -> Path:
    """Resolve a step's output directory from an arbitrary caller view.

    Consolidates the three independent base-output-dir reconstruction
    heuristics that consumers used to reach the same answer:

    - export/processor.py looked up Step 3 results from inside an
      ``7_export_output`` view (name-prefix parent climb);
    - analysis/framework_common.py looked up Step 12 execution from inside a
      ``16_analysis_output`` view (passing ``output_dir.parent``); and
    - gui/runner.py resolved its own orchestrator output from a base dir.

    All three want the same thing: when ``output_dir`` is already inside a
    step-output subdirectory (``output/<step>_output`` or deeper), resolve
    ``step_stem``'s output from the shared pipeline root; otherwise resolve
    from ``output_dir`` directly. The shared ``_output`` parent is identified
    by name (the registry-derivable ``<stem>_output`` convention), so no
    caller-specific prefix list is needed.

    Fallback policy: this helper is the single source of output-directory
    resolution. Wrappers and callers delegate here unconditionally and must
    not keep their own ``except ImportError`` fallbacks — an unimportable
    ``gnn`` package is a broken environment and must fail loud. For
    genuinely standalone use outside an importable ``gnn`` package, the
    agreed recovery is to return the caller-supplied directory unchanged
    (least surprising), never a reconstructed sibling path.

    Args:
        step_stem: Registry step stem, e.g. ``"3_gnn"`` or ``"12_execute"``.
        output_dir: Caller's view of an output directory (base or nested).

    Returns:
        The resolved ``output/<step_stem>_output`` path.
    """
    candidate = output_dir
    # Find the nearest ``*_output`` ancestor (the step-output subdir the
    # caller is nested inside); its parent is the shared pipeline root.
    # If no such ancestor exists, resolve directly from ``output_dir``.
    step_output_ancestor: Path | None = None
    walk = output_dir
    while walk != walk.parent:  # stop at filesystem root
        if walk.name.endswith("_output"):
            step_output_ancestor = walk
            break
        walk = walk.parent
    if (
        step_output_ancestor is not None
        and step_output_ancestor.parent != step_output_ancestor
    ):
        candidate = step_output_ancestor.parent
    return get_output_dir_for_script(step_stem, candidate)
