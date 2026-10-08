"""RxInfer execution-result discovery and JSON ingestion.

Filesystem and decoding failures belong here, separate from numerical analysis
and presentation. This module does not import the plotting stack.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RxInferResultReadError(ValueError):
    """An execution evidence file cannot supply a JSON object."""

    def __init__(self, path: Path, cause: Exception) -> None:
        self.path = path
        self.reason = f"{type(cause).__name__}: {cause}"
        super().__init__(f"{path}: {self.reason}")


def read_result_object(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object, retaining the exact decoding/I/O cause."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RxInferResultReadError(path, exc) from exc
    if not isinstance(data, dict):
        cause = ValueError(f"expected a JSON object, got {type(data).__name__}")
        raise RxInferResultReadError(path, cause) from cause
    return data


def _current_rxinfer_models(execution_dir: Path) -> set[str] | None:
    """Return current model names; only an absent summary permits standalone use.

    An authoritative empty summary admits no model folders. An unreadable or
    malformed summary raises a contextual error rather than widening selection.
    """
    summary_file = execution_dir / "summaries" / "execution_summary.json"
    if not summary_file.exists():
        summary_file = execution_dir / "execution_summary.json"
    if not summary_file.exists():
        return None
    data = read_result_object(summary_file)
    details = data.get("execution_details")
    if not isinstance(details, list):
        details = data.get("execution_results")
    if not isinstance(details, list):
        cause = ValueError("execution_details or execution_results must be a list")
        raise RxInferResultReadError(summary_file, cause) from cause
    models = {
        str(detail.get("model_name"))
        for detail in details
        if isinstance(detail, dict)
        and str(detail.get("framework", "")).lower() == "rxinfer"
        and detail.get("model_name")
    }
    return models


def _latest_current_results_file(sim_data_dir: Path) -> Path | None:
    """Select one current RxInfer result file from a simulation_data directory."""
    candidates = sorted(
        sim_data_dir.glob("*simulation_results.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            data = read_result_object(candidate)
        except RxInferResultReadError as exc:
            logger.warning("Skipping invalid RxInfer result: %s", exc)
            continue
        if data.get("schema_version") == "rxinfer_simulation_v1":
            return candidate
    return candidates[0] if candidates else None


def extract_simulation_data(
    execution_dir: Path, logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Extract RxInfer simulation data from execution outputs.

    Args:
        execution_dir: Directory containing execution results
        logger: Logger instance

    Returns:
        Dictionary with extracted simulation data
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    data: dict[str, Any] = {
        "beliefs": [],
        "observations": [],
        "true_states": [],
        "time_steps": 0,
        "model_name": "",
        "framework": "rxinfer",
    }

    try:
        sim_data_dir = execution_dir / "simulation_data"
        if sim_data_dir.exists():
            results_files = list(sim_data_dir.glob("*simulation_results.json"))
            if results_files:
                results = read_result_object(results_files[0])
                data.update(results)

    except (RxInferResultReadError, OSError) as exc:
        logger.warning(
            "Failed to extract RxInfer data (%s): %s", type(exc).__name__, exc
        )

    return data
