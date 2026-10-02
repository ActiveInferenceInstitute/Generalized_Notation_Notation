"""Shared framework-name and path-inference helpers for the analysis module.

Single source of truth for:
- the set of framework directory names the pipeline uses under
  ``output/12_execute_output/<model>/<framework>/``,
- framework-name normalization (``"ActiveInference.jl"`` → ``"activeinference_jl"``),
- model-name inference from a path segment that precedes a framework segment,
- discovery of current-schema ``simulation_results.json`` payloads.

Extracted from processor.py and visualizations.py (both previously kept their
own copies of these constants with silent drift — e.g. bnlearn missing from
the visualization-local set).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Framework directory names under the Step 12 execution output tree.
# NOTE: ``bnlearn`` is a rendered/executed framework but has no analyzer
# subpackage; it is included here for path inference and dashboard grouping.
FRAMEWORK_DIR_NAMES: frozenset[str] = frozenset(
    {
        "activeinference_jl",
        "bnlearn",
        "cpomdp",
        "thrml",
        "ngclearn",
        "discopy",
        "jax",
        "numpyro",
        "pymdp",
        "pytorch",
        "rxinfer",
        # stan renders + executes (F12: its results were previously dropped by
        # the path-inference helpers), but has no analyzer subpackage.
        "stan",
    }
)

# Frameworks whose current payloads are schema-gated
# (``*_simulation_v1``). Payloads outside this set are accepted as-is.
SCHEMA_GATED_FRAMEWORKS: frozenset[str] = frozenset(
    {"activeinference_jl", "pymdp", "rxinfer"}
)

# Canonical schema_version strings accepted by the schema-gated consumers.
CURRENT_SIMULATION_SCHEMAS: frozenset[str] = frozenset(
    {
        "pymdp_simulation_v1",
        "rxinfer_simulation_v1",
        "activeinference_jl_simulation_v1",
        # Ungated backends (pytorch/numpyro) also stamp these schema ids;
        # their payloads are accepted with or without them.
        "pytorch_simulation_v1",
        "numpyro_simulation_v1",
    }
)


def normalize_framework_name(framework: Any) -> str:
    """Normalize a framework name to its directory/registry form.

    ``"ActiveInference.jl"`` → ``"activeinference_jl"``,
    ``"Rx Infer"`` → ``"rx_infer"`` (existing contract: lowercase, ``.`` and
    whitespace both mapped to ``_``).
    """
    return str(framework).lower().replace(".", "_").replace(" ", "_")


def model_name_from_path(path: Path, default: str = "unknown") -> str:
    """Infer the model name from a path segment preceding a framework segment.

    Given ``.../<model>/<framework>/...`` the model name is the segment
    directly before the framework name. Returns ``default`` when no framework
    segment is present (or it is the first segment).
    """
    parts = Path(path).parts
    for index, part in enumerate(parts):
        if part in FRAMEWORK_DIR_NAMES and index >= 1:
            return parts[index - 1]
    return default


def framework_from_path(path: Path) -> Optional[str]:
    """Return the framework directory name found in ``path``, if any."""
    for part in Path(path).parts:
        if part in FRAMEWORK_DIR_NAMES:
            return part
    return None


def iter_current_schema_results(
    execution_dir: Path,
    pattern: str = "*simulation_results.json",
) -> list[tuple[Path, Dict[str, Any]]]:
    """Yield ``(path, payload)`` pairs for current-schema simulation results.

    A file qualifies when it parses as a JSON object, its framework (inferred
    from the path) is schema-gated, and its ``schema_version`` is one of
    ``CURRENT_SIMULATION_SCHEMAS``. Non-schema-gated frameworks are returned
    as-is (their consumers accept heterogeneous payloads).
    """
    results: list[tuple[Path, Dict[str, Any]]] = []
    for sim_file in sorted(Path(execution_dir).rglob(pattern)):
        try:
            with open(sim_file, encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            logger.debug("Skipping unreadable %s: %s", sim_file.name, e)
            continue
        if not isinstance(payload, dict):
            continue
        fw = framework_from_path(sim_file)
        if fw is None:
            continue
        if fw in SCHEMA_GATED_FRAMEWORKS:
            if payload.get("schema_version") not in CURRENT_SIMULATION_SCHEMAS:
                continue
        results.append((sim_file, payload))
    return results


def resolve_execution_dir(output_dir: Path) -> Path:
    """Resolve the Step 12 execution output directory for a given output dir.

    Thin delegate to ``pipeline.config.resolve_step_output_dir``; the shared
    fallback policy (standalone recovery = caller-supplied directory, fail
    loud on an unimportable package) is documented there. This wrapper keeps
    no private ``except ImportError`` fallback of its own.
    """
    from gnn.pipeline.config import resolve_step_output_dir

    return Path(resolve_step_output_dir("12_execute", output_dir))


def load_execution_summary(
    execution_dir: Path,
) -> Tuple[Path, Optional[Dict[str, Any]]]:
    """Load the Step 12 execution summary for ``execution_dir``.

    Prefers ``summaries/execution_summary.json`` then the root
    ``execution_summary.json``.

    Returns ``(resolved_path, payload_or_None)``.
    """
    candidate = Path(execution_dir) / "summaries" / "execution_summary.json"
    if not candidate.exists():
        candidate = Path(execution_dir) / "execution_summary.json"
    if not candidate.exists():
        return (candidate, None)
    try:
        with open(candidate, encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load execution summary %s: %s", candidate, e)
        return (candidate, None)
    return (candidate, payload if isinstance(payload, dict) else None)


def filter_paths_by_scope(
    path: Path,
    payload_framework: Any,
    allowed_frameworks: Optional[Set[str]],
    allowed_model_names: Optional[Set[str]],
) -> bool:
    """Return True when a result file passes the current-run scope filters.

    ``payload_framework`` is the framework declared inside the payload (may be
    None — the path is then used for inference). Model names are matched
    against the path segment preceding the framework segment.
    """
    fw = (
        normalize_framework_name(payload_framework)
        if payload_framework
        else framework_from_path(path)
    )
    if allowed_frameworks and (fw is None or fw not in allowed_frameworks):
        return False
    if allowed_model_names:
        model = model_name_from_path(path)
        if model and model not in allowed_model_names:
            return False
    return True


__all__ = [
    "CURRENT_SIMULATION_SCHEMAS",
    "FRAMEWORK_DIR_NAMES",
    "SCHEMA_GATED_FRAMEWORKS",
    "filter_paths_by_scope",
    "framework_from_path",
    "iter_current_schema_results",
    "load_execution_summary",
    "model_name_from_path",
    "normalize_framework_name",
    "resolve_execution_dir",
]
