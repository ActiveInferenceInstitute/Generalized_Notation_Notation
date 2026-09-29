#!/usr/bin/env python3
"""Shared framework-availability helpers for the GNN pipeline.

Single source of truth for "is this Python ML/AI framework importable?" — used by
``src/gnn/execute/processor/`` (Step 12) and ``src/gnn/render/processor/`` (Step 11) to
decide between running, skipping, or warning on framework-specific code paths.

Previously this logic lived duplicated inside ``execute/processor.py`` (now the
``gnn.execute.processor`` package) and a parallel
block in ``render/processor/``. Keeping it here lets both steps stay in sync when
new frameworks are added and avoids the hazard of one step reporting a framework
available while the other believes it missing.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess  # nosec B404
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

__all__: list[Any] = [
    "FRAMEWORK_IMPORT_CHECK",
    "FrameworkStatus",
    "diagnose_framework",
    "is_framework_available",
    "check_framework",
    "last_unavailable_status",
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

# How to install the external toolchain a probe statement looks for.
FRAMEWORK_TOOLCHAIN_HINT: Dict[str, str] = {
    "stan": "python -c 'import cmdstanpy; cmdstanpy.install_cmdstan()'",
}

# Frameworks whose extra is gated on a newer interpreter (see pyproject.toml
# environment markers); on older interpreters the module can never install.
FRAMEWORK_MIN_PYTHON: Dict[str, Tuple[int, int]] = {
    "ngclearn": (3, 12),
}

# Heavy imports (bnlearn pulls pgmpy/torch) can take well over 10 s on a loaded
# machine; a timeout is reported as its own skip category, never as "missing".
PROBE_TIMEOUT_SECONDS = 60

# Skip categories carried on FrameworkStatus.skip_category.
SKIP_MISSING_MODULE = "missing_module"
SKIP_PYTHON_TOO_OLD = "python_too_old"
SKIP_PROBE_FAILED = "probe_failed"
SKIP_PROBE_TIMEOUT = "probe_timeout"

_EXIT_MISSING_MODULE = 3
_EXIT_PYTHON_TOO_OLD = 4


@dataclass(frozen=True)
class FrameworkStatus:
    """Structured answer to "is this framework available?"."""

    name: str
    available: bool
    missing_module: Optional[str] = None
    install_hint: Optional[str] = None
    reason: Optional[str] = None
    skip_category: Optional[str] = None


# Last negative diagnosis per (framework, executor), so Step 12 can put the
# real cause on a skipped-script envelope without re-running the probe.
_LAST_UNAVAILABLE: Dict[Tuple[str, Optional[str]], FrameworkStatus] = {}


def _probe_program(framework: str) -> str:
    module_name, _hint = FRAMEWORK_IMPORT_CHECK[framework]
    min_py = FRAMEWORK_MIN_PYTHON.get(framework)
    lines = ["import importlib.util, sys"]
    if min_py is not None:
        lines.append(
            f"if sys.version_info[:2] < {min_py!r}: "
            f"print('.'.join(map(str, sys.version_info[:2]))); "
            f"sys.exit({_EXIT_PYTHON_TOO_OLD})"
        )
    lines.append(
        f"if importlib.util.find_spec({module_name!r}) is None: "
        f"sys.exit({_EXIT_MISSING_MODULE})"
    )
    lines.append(FRAMEWORK_PROBE_STATEMENT.get(framework, f"import {module_name}"))
    return "\n".join(lines)


def _unavailable(framework: str, category: str, detail: str = "") -> FrameworkStatus:
    module_name, install_hint = FRAMEWORK_IMPORT_CHECK[framework]
    if category == SKIP_PYTHON_TOO_OLD:
        need = ".".join(map(str, FRAMEWORK_MIN_PYTHON[framework]))
        reason = (
            f"Dependency not installable: {module_name} requires Python >= {need}"
            f" (interpreter is {detail or 'older'})"
        )
    elif category == SKIP_PROBE_FAILED:
        toolchain = FRAMEWORK_TOOLCHAIN_HINT.get(framework)
        reason = f"Dependency probe failed for {module_name}: {detail or 'error'}"
        if toolchain:
            install_hint = toolchain
    elif category == SKIP_PROBE_TIMEOUT:
        reason = (
            f"Dependency probe for {module_name} timed out after "
            f"{PROBE_TIMEOUT_SECONDS}s (module may be installed; machine overloaded?)"
        )
    else:
        reason = f"Dependency not installed: {module_name}"
    return FrameworkStatus(
        name=framework,
        available=False,
        missing_module=module_name,
        install_hint=install_hint,
        reason=reason,
        skip_category=category,
    )


def _diagnose_in_process(framework: str) -> FrameworkStatus:
    module_name, _hint = FRAMEWORK_IMPORT_CHECK[framework]
    min_py = FRAMEWORK_MIN_PYTHON.get(framework)
    if min_py is not None and sys.version_info[:2] < min_py:
        return _unavailable(
            framework,
            SKIP_PYTHON_TOO_OLD,
            ".".join(map(str, sys.version_info[:2])),
        )
    if importlib.util.find_spec(module_name) is None:
        return _unavailable(framework, SKIP_MISSING_MODULE)
    if framework in FRAMEWORK_PROBE_STATEMENT:
        try:
            exec(FRAMEWORK_PROBE_STATEMENT[framework], {})  # nosec B102 — fixed, non-user statement
        except Exception as err:
            return _unavailable(
                framework, SKIP_PROBE_FAILED, f"{type(err).__name__}: {err}"
            )
    return FrameworkStatus(name=framework, available=True)


def _diagnose_via_executor(
    framework: str, executor: str, logger: Optional[logging.Logger]
) -> FrameworkStatus:
    try:
        result = subprocess.run(  # nosec B603
            [executor, "-c", _probe_program(framework)],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as err:
        if logger is not None:
            logger.debug(
                f"Availability check for {framework} via {executor} failed: {err}"
            )
        return _unavailable(framework, SKIP_PROBE_TIMEOUT)
    except FileNotFoundError as err:
        if logger is not None:
            logger.debug(
                f"Availability check for {framework} via {executor} failed: {err}"
            )
        return _unavailable(
            framework, SKIP_PROBE_FAILED, f"executor not found: {executor}"
        )
    if result.returncode == 0:
        return FrameworkStatus(name=framework, available=True)
    if result.returncode == _EXIT_MISSING_MODULE:
        return _unavailable(framework, SKIP_MISSING_MODULE)
    if result.returncode == _EXIT_PYTHON_TOO_OLD:
        return _unavailable(framework, SKIP_PYTHON_TOO_OLD, result.stdout.strip())
    stderr_lines = [ln for ln in (result.stderr or "").splitlines() if ln.strip()]
    detail = stderr_lines[-1].strip() if stderr_lines else f"exit {result.returncode}"
    return _unavailable(framework, SKIP_PROBE_FAILED, detail)


def diagnose_framework(
    framework: str,
    executor: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> FrameworkStatus:
    """Return availability plus, when unavailable, the precise reason.

    Unknown frameworks are reported available so callers do not over-skip
    external tools. When ``executor`` is a Python interpreter path the probe
    runs in that interpreter, so it reflects the target environment. A
    negative answer is remembered for :func:`last_unavailable_status`.
    """
    if framework not in FRAMEWORK_IMPORT_CHECK:
        return FrameworkStatus(name=framework, available=True)
    status = (
        _diagnose_in_process(framework)
        if executor is None
        else _diagnose_via_executor(framework, executor, logger)
    )
    key = (framework, executor)
    if status.available:
        _LAST_UNAVAILABLE.pop(key, None)
    else:
        _LAST_UNAVAILABLE[key] = status
    return status


def last_unavailable_status(
    framework: str, executor: Optional[str] = None
) -> Optional[FrameworkStatus]:
    """Return the most recent negative diagnosis for ``(framework, executor)``."""
    return _LAST_UNAVAILABLE.get((framework, executor))


def is_framework_available(
    framework: str,
    executor: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Return True if the framework's required Python module is importable.

    Thin boolean view of :func:`diagnose_framework`.
    """
    return diagnose_framework(framework, executor=executor, logger=logger).available


def check_framework(
    framework: str,
    executor: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> FrameworkStatus:
    """Return a structured FrameworkStatus for the given framework."""
    return diagnose_framework(framework, executor=executor, logger=logger)
