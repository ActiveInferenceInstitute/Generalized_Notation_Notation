#!/usr/bin/env python3
"""Fail when a numbered orchestrator lacks the stdlib-shadowing guard.

``src/gnn/types/`` shadows the standard-library ``types`` module whenever a
numbered script is executed directly (``python src/gnn/N_*.py`` puts
``src/gnn/`` first on ``sys.path``). Every directly runnable script must
therefore pop the script directory before the first import that could resolve
inside ``src/gnn/``:

    if globals().get("__package__") is None and not sys.flags.safe_path:
        sys.path.pop(0)

The guard is the accepted, gate-enforced resolution of composability item
COMP-010 (docs/standards/composability.md): the hoist into
``create_standardized_pipeline_script`` is unreachable because that helper is
itself imported after the guard, so the guard is a required pre-import step of
the thin-orchestrator contract.

Mechanics: for every ``src/gnn/[0-9]*_*.py``, parse the module AST and require
(a) a module-level guard whose test is exactly
``globals().get("__package__") is None and not sys.flags.safe_path`` and whose
body calls ``sys.path.pop(0)``, and (b) no import that executes at module
import time, other than ``import sys`` or a ``from __future__`` directive,
ahead of it. Fully deterministic: file reads only; no network, no clock, no
randomness.

Usage: uv run --extra dev python scripts/check_stdlib_shadow_guard.py
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR_GLOB = "[0-9]*_*.py"

# ``ast.unparse`` normalization of the canonical guard test (single quotes,
# parenthesized ``not`` clause).
GUARD_TEST = "globals().get('__package__') is None and (not sys.flags.safe_path)"


def _is_shadow_guard(node: ast.stmt) -> bool:
    """Recognize the canonical module-level guard statement."""
    if not isinstance(node, ast.If) or ast.unparse(node.test) != GUARD_TEST:
        return False
    return any(
        isinstance(statement, ast.Expr)
        and ast.unparse(statement.value) == "sys.path.pop(0)"
        for statement in node.body
    )


def _is_allowed_pre_guard_import(node: ast.stmt) -> bool:
    """Report whether an import is safe ahead of the guard.

    ``import sys`` cannot resolve inside ``src/gnn/`` (no ``sys`` entry
    exists there) and a ``from __future__`` directive is a compiler directive.
    """
    if isinstance(node, ast.Import):
        return all(alias.name == "sys" for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return node.module == "__future__"
    return False


def _module_imports(tree: ast.Module) -> list[ast.stmt]:
    """Collect the imports that execute while the module is imported.

    Walks module-level statements, class bodies and control flow; imports
    inside function or lambda bodies execute only when called, after the guard
    has already run, so they are excluded.
    """
    found: list[ast.stmt] = []

    def walk(node: ast.AST) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        else:
            for child in ast.iter_child_nodes(node):
                walk(child)

    walk(tree)
    return found


def check_file(path: Path, base: Path) -> list[str]:
    """Return the failure messages for one numbered script (empty when compliant)."""
    rel = path.relative_to(base).as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        return [f"{rel}: unreadable or unparseable ({exc})"]

    guard = next((node for node in tree.body if _is_shadow_guard(node)), None)
    if guard is None:
        return [f"{rel}: missing the canonical stdlib-shadowing guard"]

    return [
        f"{rel}: import at line {node.lineno} precedes the guard "
        f"(lines {guard.lineno}-{guard.end_lineno})"
        for node in _module_imports(tree)
        if not _is_allowed_pre_guard_import(node) and node.lineno <= guard.end_lineno
    ]


def collect_failures(scan_root: Path, base: Path) -> list[str]:
    """Check every numbered orchestrator under ``scan_root``."""
    files = sorted(scan_root.glob(ORCHESTRATOR_GLOB))
    if not files:
        return [
            f"no files match {scan_root / ORCHESTRATOR_GLOB} - the orchestrator "
            "layout changed; update this gate."
        ]
    failures: list[str] = []
    for path in files:
        failures.extend(check_file(path, base))
    return failures


def main(
    argv: list[str] | None = None,
    *,
    scan_root: Path | None = None,
    base: Path | None = None,
) -> int:
    """Report guard failures; exit 1 when any numbered script is non-compliant."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "stdlib-shadowing guard gate").splitlines()[0]
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Compatibility flag; guard failures fail by default.",
    )
    args = parser.parse_args(argv)
    del args  # the gate fails on any breach, with or without --strict

    root = base if base is not None else ROOT
    scanned = scan_root if scan_root is not None else ROOT / "src" / "gnn"
    failures = collect_failures(scanned, root)
    if failures:
        print(f"check_stdlib_shadow_guard: {len(failures)} failure(s):")
        for failure in failures:
            print(f"  {failure}")
        print(
            "The numbered scripts run directly (python src/gnn/N_*.py) and "
            "src/gnn/types/ would otherwise shadow the stdlib types module. "
            "Add the guard before the first import:\n"
            '    if globals().get("__package__") is None and not sys.flags.safe_path:\n'
            "        sys.path.pop(0)"
        )
        return 1

    count = len(sorted(scanned.glob(ORCHESTRATOR_GLOB)))
    print(
        f"check_stdlib_shadow_guard: {count} numbered orchestrators carry the "
        "guard before the first import."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
