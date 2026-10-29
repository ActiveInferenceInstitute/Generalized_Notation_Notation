"""Lean execution backend: FEP_Formal bridge document verification (v0.6)."""

from .lean_runner import (
    FEP_FORMAL_ROOT_ENV,
    FEP_LEAN_ROOT_ENV,
    fep_bridge_executable,
    lean_toolchain_available,
    resolve_fep_formal_root,
    resolve_fep_lean_root,
    run_lean_scripts,
    verify_document,
)

__all__ = [
    "FEP_FORMAL_ROOT_ENV",
    "FEP_LEAN_ROOT_ENV",
    "fep_bridge_executable",
    "resolve_fep_formal_root",
    "lean_toolchain_available",
    "resolve_fep_lean_root",
    "run_lean_scripts",
    "verify_document",
]
