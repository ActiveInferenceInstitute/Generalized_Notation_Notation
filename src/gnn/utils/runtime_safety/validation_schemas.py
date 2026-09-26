#!/usr/bin/env python3
"""Shared input-validation helpers for the GNN pipeline.

Public entry points in ``render``, ``execute``, ``gnn``, and ``analysis`` processors
previously accepted ``Dict`` / ``Path`` / ``str`` arguments without type or existence
checks. When callers passed None or an empty mapping, the processor would either
crash with an opaque AttributeError or silently produce no artifacts while
reporting success. This module centralizes the checks so every processor can
validate the same way.

Typical usage::

    from gnn.utils.runtime_safety.validation_schemas import validate_model_data, validate_target_dir

    def generate_pymdp_code(model_data, output_path=None):
        model_data = validate_model_data(model_data, context="generate_pymdp_code")
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from gnn.frameworks import ALL_FRAMEWORKS, LITE_FRAMEWORKS

__all__: list[Any] = [
    "validate_model_data",
    "validate_target_dir",
    "validate_frameworks_arg",
    "normalize_pomdp_columns",
    "KNOWN_FRAMEWORKS",
    "FRAMEWORK_PRESETS",
]


# ``KNOWN_FRAMEWORKS`` derives from the canonical name enumeration
# (``gnn.frameworks.ALL_FRAMEWORKS``); it must never be re-listed here.
# ``FRAMEWORK_IMPORT_CHECK`` in
# ``src/gnn/utils/runtime_safety/framework_availability.py`` maps toolchain
# probes per framework; ``KNOWN_FRAMEWORKS`` covers every name that mapping
# can address, plus the execution-only ``lean`` backend.
KNOWN_FRAMEWORKS: tuple[str, ...] = ALL_FRAMEWORKS

FRAMEWORK_PRESETS: dict[str, Any] = {
    "all": list(ALL_FRAMEWORKS),
    "lite": list(LITE_FRAMEWORKS),
}


def validate_model_data(
    data: Optional[Dict[str, Any]],
    *,
    required_keys: Iterable[str] = ("model_name",),
    context: str = "",
) -> Dict[str, Any]:
    """Validate a ``model_data`` mapping passed to a code generator or processor.

    Raises ``ValueError`` when ``data`` is None, not a mapping, or missing any key
    in ``required_keys``. Returns the input dict unchanged on success so callers
    can use it in an expression: ``model_data = validate_model_data(model_data)``.
    """
    prefix = f"[{context}] " if context else ""
    if data is None:
        raise ValueError(f"{prefix}model_data is None")
    if not isinstance(data, dict):
        raise ValueError(
            f"{prefix}model_data must be a dict, got {type(data).__name__}"
        )
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"{prefix}model_data missing required keys: {missing}")
    return data


def validate_target_dir(
    path: Any, *, must_exist: bool = True, context: str = ""
) -> Path:
    """Validate a target-directory argument.

    Accepts str or Path; coerces to Path. When ``must_exist`` is True (default),
    raises FileNotFoundError if the path does not exist. When the path exists but
    is a file rather than a directory, raises NotADirectoryError. Returns the
    coerced Path.
    """
    prefix = f"[{context}] " if context else ""
    if path is None:
        raise ValueError(f"{prefix}target_dir is None")
    p = Path(path)
    if must_exist and not p.exists():
        raise FileNotFoundError(f"{prefix}target_dir does not exist: {p}")
    if p.exists() and not p.is_dir():
        raise NotADirectoryError(f"{prefix}target_dir is not a directory: {p}")
    return p


def validate_frameworks_arg(arg: Any, *, context: str = "") -> str:
    """Validate the ``frameworks`` CLI/API argument.

    Accepts a preset name (``"all"`` or ``"lite"``), a comma-separated list of
    known framework names, or an empty string (treated as ``"all"``).
    Raises ValueError on non-string input or when the comma list contains no
    recognized framework names.

    Returns the normalized string for use with parse_frameworks_parameter.
    """
    prefix = f"[{context}] " if context else ""
    if arg is None or arg == "":
        return "all"
    if not isinstance(arg, str):
        raise ValueError(
            f"{prefix}frameworks must be a string, got {type(arg).__name__}"
        )
    normalized = arg.strip().lower()
    if normalized in FRAMEWORK_PRESETS:
        return normalized
    parts = [p.strip() for p in normalized.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{prefix}frameworks argument parsed to empty list: {arg!r}")
    if not any(p in KNOWN_FRAMEWORKS for p in parts):
        raise ValueError(
            f"{prefix}frameworks argument contains no known frameworks. "
            f"Got {parts!r}; known: {list(KNOWN_FRAMEWORKS)}"
        )
    return normalized


def normalize_pomdp_columns(matrix: Any) -> Any:
    """Column-normalize a 2D probability matrix so each column sums to 1.

    Zero-sum columns are replaced with a uniform distribution (1/n_rows in every
    cell) — the same recovery policy used in
    ``src/gnn/render/processor/parsing.py::normalize_matrices``. Non-2D input is returned
    unchanged so callers can apply this to the leaves of 3D or list-of-arrays
    structures without special-casing.
    """
    try:
        import numpy as np
    except ImportError as err:
        raise RuntimeError("numpy is required for normalize_pomdp_columns") from err

    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2:
        return m
    col_sums = m.sum(axis=0, keepdims=True)
    zero_mask = col_sums == 0
    col_sums[zero_mask] = 1.0
    m = m / col_sums
    if bool(np.any(zero_mask)):
        uniform = np.ones(m.shape[0], dtype=np.float64) / m.shape[0]
        for col_idx in np.where(zero_mask.flatten())[0]:
            m[:, col_idx] = uniform
    return m
