#!/usr/bin/env python3
"""
Execution type definitions for GNN Step 12.

Shared types, literals, and constants for the execute module. This module is
a leaf: it must not import from ``execute.processor`` or any other execute
sibling, to keep the facade import graph acyclic.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, TypedDict, Union

# File suffixes Step 12 can execute; companion artifacts (``.stan``, ``.toml``,
# ``.json``) listed in a render summary are not scripts.
_EXECUTABLE_SUFFIXES = frozenset({".py", ".jl"})

# Shared default wall-clock timeout (seconds) for rendered-script execution.
# Runners whose historical default differs (rxinfer/jax/numpyro/pytorch/discopy
# 300, bnlearn 1800, lean 1800) keep their own values; this constant replaces
# the literal 600 defaults (pymdp, activeinference_jl) and the executor
# dispatch helpers.
DEFAULT_RUNNER_TIMEOUT_SECONDS: int = 600

# Literal list mirrors ``gnn.frameworks.ALL_FRAMEWORKS`` (the runtime source).
# mypy requires a literal list; cannot derive at runtime. Drift is guarded by
# ``tests/render/test_framework_availability.py`` (TestCanonicalConsistency).
ExecutionFrameworkName = Literal[
    "pymdp",
    "rxinfer",
    "activeinference_jl",
    "jax",
    "discopy",
    "pytorch",
    "numpyro",
    "stan",
    "bnlearn",
    "cpomdp",
    "thrml",
    "ngclearn",
    "lean",
]


@dataclass(frozen=True)
class ScriptExecutionContext:
    """Normalized execution metadata for one rendered script."""

    script_path: Path
    script_name: str
    framework: str
    model_name: str
    executor: str


@dataclass(frozen=True)
class ExecutionOutcome:
    """Result of the pure Step 12 outcome classification.

    ``outcome`` follows the historical Step 12 contract: ``True`` on success,
    ``False`` on failure, and ``2`` when a non-strict requested-framework run
    had nothing to execute. ``attempted`` is ``total_found - skipped``.
    """

    outcome: Union[bool, int]
    status: str
    reason: str
    exit_code: int
    attempted: int


class ExecutionPlan(TypedDict):
    """Dry-run plan returned by :func:`execute.planning.plan_execute`.

    Keys are populated progressively; every key is optional so partial plans
    (for example when the render output directory is missing) remain
    representable without ``None``-sentinel gymnastics.

    ``unsupported_render_receipts`` carries receipts for frameworks the
    renderer declared unable to represent the model — refused, not failed.
    """

    requested_frameworks: List[str]
    target_directory: str
    output_directory: str
    render_output_dir: Optional[str]
    render_contract_found: bool
    status: str
    total_scripts: int
    would_execute: List[Dict[str, str]]
    would_skip_dependency: List[Dict[str, str]]
    unknown_framework_scripts: List[Dict[str, str]]
    missing_render_scripts: List[str]
    render_failures: List[Dict[str, str]]
    unsupported_render_receipts: List[Dict[str, str]]
