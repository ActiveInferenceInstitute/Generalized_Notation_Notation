#!/usr/bin/env python3
"""Audit maintained repository text for stale API and generated-artifact terms."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAINTAINED_ROOTS = [
    ROOT / ".agent_rules",
    ROOT / ".github",
    ROOT / "docs",
    ROOT / "input",
    ROOT / "scripts",
    ROOT / "src",
    ROOT / "AGENTS.md",
    ROOT / "README.md",
    ROOT / "SPEC.md",
    ROOT / "pyproject.toml",
    ROOT / "justfile",
]

SKIP_PARTS = {
    "fleet-logs",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "other",
    "archive",
    "build",
    "dist",
    "node_modules",
    "output",
}

TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".csv",
    ".html",
    ".jl",
    ".json",
    ".md",
    ".py",
    ".scala",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

AUDIT_TOOL_FILES = {
    ROOT / "scripts" / "check_maintained_doc_terms.py",
    ROOT / "tests" / "docs" / "test_docs_audit.py",
}

PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"zero-mock", re.IGNORECASE), "test-double wording"),
    (re.compile(r"\bPyMDPConverter\b"), "stale PyMDP converter name"),
    (re.compile(r"\bmock_services\b"), "test-double service wording"),
    (re.compile(r"\bmock_data\b"), "test-double data wording"),
    (re.compile(r"\bmock_llm\b"), "test-double LLM wording"),
    (re.compile(r"\bdummy_input\b"), "test-double input wording"),
    (re.compile(r"\bdummy_observations\b"), "test-double observation wording"),
    (re.compile(r"\bRoot shims\b"), "compatibility-layer wording"),
    (re.compile(r"\btest shims\b"), "test shim wording"),
    (re.compile(r"\btool shims\b"), "MCP shim wording"),
    (
        re.compile(r"\bfallback shims removed\b", re.IGNORECASE),
        "stale fallback shim wording",
    ),
    (re.compile(r"no mocks", re.IGNORECASE), "test-double wording"),
    (re.compile(r"unittest\.mock", re.IGNORECASE), "test-double import/reference"),
    (re.compile(r"\bmock(s)?\b", re.IGNORECASE), "test-double wording"),
    (re.compile(r"\bstub(s)?\b", re.IGNORECASE), "incomplete-surface wording"),
    (re.compile(r"\bfake(s)?\b", re.IGNORECASE), "test-double wording"),
    (re.compile(r"\blegacy\b", re.IGNORECASE), "stale-version wording"),
    (re.compile(r"(?<!::)placeholder", re.IGNORECASE), "incomplete-surface wording"),
    (re.compile(r"\bplaceholder API\b", re.IGNORECASE), "incomplete-surface API"),
    (re.compile(r"\bfake default\b", re.IGNORECASE), "invented-default wording"),
    (re.compile(r"\bfallback agent\b", re.IGNORECASE), "stale fallback wording"),
    (re.compile(r"\bfallback circuit\b", re.IGNORECASE), "stale fallback wording"),
    (
        re.compile(r"\bfallback shim\b", re.IGNORECASE),
        "compatibility-only layer wording",
    ),
    (re.compile(r"\bsimple fallback\b", re.IGNORECASE), "hidden-recovery wording"),
    (re.compile(r"\bhidden fallback\b", re.IGNORECASE), "hidden-recovery wording"),
    (
        re.compile(r"backwards?-compat(?:ible|ibility)?", re.IGNORECASE),
        "compatibility promise",
    ),
    (
        re.compile(r"\bcompat(?:ibility)?-only\b", re.IGNORECASE),
        "compatibility-only API",
    ),
    (re.compile(r"\blegacy API\b", re.IGNORECASE), "stale-version API"),
    (re.compile(r"compatibility alias", re.IGNORECASE), "compatibility-only API"),
    (re.compile(r"\bshim\b", re.IGNORECASE), "compatibility-only layer wording"),
    (re.compile(r"\bdummy\b", re.IGNORECASE), "test-double wording"),
    (re.compile(r"\bdeprecated\b", re.IGNORECASE), "stale-version wording"),
    (re.compile(r"NotImplementedError"), "unimplemented public surface"),
    (re.compile(r"not implemented", re.IGNORECASE), "unimplemented wording"),
]


# Exact migration statements and the two reviewed transport-test boundary lines.
# Each exception requires its path, diagnostic and complete stripped line.
# Existing paired tests use a controlled read boundary to prove accepted()
# consumes frozen metadata without consulting a live companion checkout.
# Copies, changed statements and unrelated test-double terms still fail.
_REVIEWED_FEP_MIGRATION_TERMS = frozenset(
    {
        (
            ".github/test_fep_paired_runtime.py",
            "test-double wording",
            "from unittest import mock",
        ),
        (
            ".github/test_fep_paired_runtime.py",
            "test-double wording",
            "with mock.patch.object(",
        ),
        (
            "src/gnn/execute/executor_specs.py",
            "stale-version wording",
            '"(set FEP_FORMAL_ROOT; deprecated alias FEP_LEAN_ROOT)"',
        ),
        (
            "src/gnn/execute/lean/AGENTS.md",
            "stale-version wording",
            "Canonical checkout selection uses `FEP_FORMAL_ROOT` before the deprecated",
        ),
        (
            "src/gnn/execute/lean/README.md",
            "stale-version wording",
            "Canonical checkout selection uses `FEP_FORMAL_ROOT` before the deprecated",
        ),
    }
)


def _is_generated_path(path: Path) -> bool:
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        return True
    parts = rel.parts
    if any(part in SKIP_PARTS for part in parts):
        return True
    if any(part.startswith("activeinference_outputs_") for part in parts):
        return True
    if any(part.endswith("_outputs") or "_outputs_" in part for part in parts):
        return True
    return "pomdp_gridworld_outputs" in parts


def _iter_files() -> list[Path]:
    files: set[Path] = set()
    this_file = Path(__file__).resolve()
    for root in MAINTAINED_ROOTS:
        if not root.exists() or _is_generated_path(root):
            continue
        if root.is_file():
            if root.resolve() != this_file and root.resolve() not in AUDIT_TOOL_FILES:
                files.add(root)
            continue
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.resolve() != this_file
                and path.resolve() not in AUDIT_TOOL_FILES
                and path.suffix in TEXT_SUFFIXES
                and not _is_generated_path(path)
            ):
                files.add(path)
    return sorted(files)


def scan() -> list[tuple[Path, int, str, str]]:
    violations: list[tuple[Path, int, str, str]] = []
    for path in _iter_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern, description in PATTERNS:
                if pattern.search(line):
                    identity = (
                        path.relative_to(ROOT).as_posix(),
                        description,
                        line.strip(),
                    )
                    if identity in _REVIEWED_FEP_MIGRATION_TERMS:
                        continue
                    violations.append(
                        (path.relative_to(ROOT), line_no, description, line.strip())
                    )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict", action="store_true", help="Exit 1 when violations are found."
    )
    args = parser.parse_args()

    violations = scan()
    if not violations:
        print("check_repo_terminology: maintained tree clean.")
        return 0

    print(f"check_repo_terminology: {len(violations)} violation(s):\n")
    for rel, line_no, description, line in violations:
        print(f"  {rel}:{line_no}: {description}")
        print(f"    {line[:220]}")

    if args.strict:
        return 1
    print("\n(non-strict: exit 0; use --strict to fail)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
