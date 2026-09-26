#!/usr/bin/env python3
"""Shared framework-availability helpers for the GNN pipeline.

Single source of truth for "is this Python ML/AI framework importable?" — used by
``src/gnn/execute/processor.py`` (Step 12) and ``src/gnn/render/processor/`` (Step 11) to
decide between running, skipping, or warning on framework-specific code paths.

Previously this logic lived duplicated inside ``execute/processor.py`` and a parallel
block in ``render/processor/``. Keeping it here lets both steps stay in sync when
new frameworks are added and avoids the hazard of one step reporting a framework
available while the other believes it missing.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess  # nosec B404
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

__all__: list[Any] = [
    "FRAMEWORK_IMPORT_CHECK",
    "FrameworkStatus",
    "is_framework_available",
    "check_framework",
]


# Canonical mapping: framework-name → (importable-module, install-hint).
# Keep in sync with src/gnn/execute/executor.py framework runners.
FRAMEWORK_IMPORT_CHECK: Dict[str, Tuple[str, str]] = {
    "jax": ("jax", "uv sync"),
    "numpyro": ("numpyro", "uv sync"),
    "pytorch": ("torch", "uv sync"),
    "stan": ("cmdstanpy", "uv sync --extra stan"),
    "discopy": ("discopy", "uv sync"),
    "bnlearn": ("bnlearn", "uv sync --extra bnlearn"),
    "pymdp": ("pymdp", "uv sync"),
    "ngclearn": ("ngclearn", "uv sync --extra ngclearn"),
}

# Frameworks whose Python module alone is not enough: the probe must also
# find the external toolchain, otherwise Step 12 would record a failed run
# instead of the documented dependency skip.
FRAMEWORK_PROBE_STATEMENT: Dict[str, str] = {
    "stan": "import cmdstanpy; cmdstanpy.cmdstan_path()",
}


@dataclass(frozen=True)
class FrameworkStatus:
    """Structured answer to "is this framework available?"."""

    name: str
    available: bool
    missing_module: Optional[str] = None
    install_hint: Optional[str] = None


def is_framework_available(
    framework: str,
    executor: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Return True if the framework's required Python module is importable.

    Unknown frameworks return True so callers do not over-skip external tools.

    When ``executor`` is None (default), checks via ``importlib.util.find_spec`` —
    cheap, in-process. When ``executor`` is a Python interpreter path, shells
    out so the check reflects the
    target interpreter's environment rather than the caller's.
    """
    if framework not in FRAMEWORK_IMPORT_CHECK:
        return True

    module_name, _hint = FRAMEWORK_IMPORT_CHECK[framework]
    probe = FRAMEWORK_PROBE_STATEMENT.get(framework, f"import {module_name}")

    if executor is None:
        if importlib.util.find_spec(module_name) is None:
            return False
        if framework in FRAMEWORK_PROBE_STATEMENT:
            try:
                exec(probe, {})  # nosec B102 — fixed, non-user statement
            except Exception:
                return False
        return True

    try:
        result = subprocess.run(  # nosec B603
            [executor, "-c", probe],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError) as err:
        if logger is not None:
            logger.debug(
                f"Availability check for {framework} via {executor} failed: {err}"
            )
        return False


def check_framework(
    framework: str,
    executor: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> FrameworkStatus:
    """Return a structured FrameworkStatus for the given framework."""
    if framework not in FRAMEWORK_IMPORT_CHECK:
        return FrameworkStatus(name=framework, available=True)

    module_name, install_hint = FRAMEWORK_IMPORT_CHECK[framework]
    available = is_framework_available(framework, executor=executor, logger=logger)
    return FrameworkStatus(
        name=framework,
        available=available,
        missing_module=None if available else module_name,
        install_hint=None if available else install_hint,
    )
