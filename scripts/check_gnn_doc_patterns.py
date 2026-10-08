#!/usr/bin/env python3
"""
Spot-check markdown under docs/ and src/gnn/ for known-stale GNN documentation
patterns. Also pins an occurrence-count ratchet for ``except Exception`` under
src/gnn (see docs/standards/exceptions.md).

Does not fail the build by default; run in CI with --strict to exit non-zero on hits.

Usage:
  uv run --extra dev python scripts/check_gnn_doc_patterns.py
  uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Footer-staleness ratchet for src/gnn: bold colon-outside stamps only
# (`**Last Updated**: <token>`; colon-inside `**Last Updated:**` is a different
# convention and not part of this check). ISO stamps must be >= the ratchet;
# a non-ISO token is itself a violation. One known month-format stamp is
# exempted by repo-relative path.
FOOTER_STAMP = re.compile(r"\*\*Last Updated\*\*: (\S+)")
FOOTER_RATCHET = "2026-09-10"
KNOWN_OUTSTANDING = {"src/gnn/doc/QUICK_REFERENCE.md"}

# except-Exception occurrence ratchet for src/gnn, per
# docs/standards/exceptions.md: one occurrence = one ast.ExceptHandler whose
# handler type is exactly the name "Exception" (tuple handlers containing
# Exception, bare ``except:``, and ``except BaseException`` are outside this
# count). Fails on growth under --strict; prints the count on success. Lower
# the baseline as sites are removed; raise it only with a decision note in the
# doctrine page.
EXCEPT_EXCEPTION_BASELINE = 1151

# (regex, description) — tune as docs evolve
PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"from gnn import GNNParser\b"),
        "obsolete import (use exports from src/gnn/__init__.py)",
    ),
    (
        re.compile(r"from gnn\.parser import GNNParser\b"),
        "GNNParser not exported from gnn.parser",
    ),
    (
        re.compile(r"from gnn\.parsing import"),
        "gnn.parsing is not the public package layout",
    ),
    (
        re.compile(r"parsers/serializers\.py"),
        "serializers live in parsers/*_serializer.py",
    ),
    (re.compile(r"lark_parser\.py"), "stale filename unless reintroduced"),
    (
        re.compile(r"- Python 3\.12\+"),
        "use Python >= 3.11 to match pyproject.toml in SPEC examples",
    ),
    # Stale doc paths (canonical: docs/gnn/reference/gnn_file_structure_doc.md; punctuation: src/gnn/documentation/punctuation.md)
    (
        re.compile(r"docs/gnn/gnn_file_structure\.md"),
        "use docs/gnn/reference/gnn_file_structure_doc.md",
    ),
    (
        re.compile(r"docs/gnn/gnn_punctuation\.md"),
        "use src/gnn/documentation/punctuation.md or docs/gnn/gnn_syntax.md",
    ),
    (
        re.compile(r"src/gnn/gnn_file_structure\.md"),
        "use src/gnn/documentation/file_structure.md",
    ),
    (
        re.compile(r"src/gnn/gnn_punctuation\.md"),
        "use src/gnn/documentation/punctuation.md",
    ),
]


def scan(paths: list[Path]) -> list[tuple[Path, int, str, str]]:
    violations: list[tuple[Path, int, str, str]] = []
    for base in paths:
        if not base.exists():
            continue
        for md in sorted(base.rglob("*.md")):
            if "node_modules" in md.parts or ".git" in md.parts:
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                for pat, desc in PATTERNS:
                    if pat.search(line):
                        violations.append((md, i, line.strip(), desc))
    return violations


def footer_violations(root: Path) -> list[tuple[Path, int, str, str]]:
    """Staleness for src/gnn bold colon-outside footer stamps only."""
    violations: list[tuple[Path, int, str, str]] = []
    for md in sorted((root / "src" / "gnn").rglob("*.md")):
        if "node_modules" in md.parts or ".git" in md.parts:
            continue
        if str(md.relative_to(root)) in KNOWN_OUTSTANDING:
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            m = FOOTER_STAMP.search(line)
            if not m:
                continue
            token = m.group(1)
            iso = re.fullmatch(r"20\d{2}-\d{2}-\d{2}", token)
            if iso is None:
                violations.append(
                    (
                        md,
                        i,
                        line.strip(),
                        "non-ISO footer stamp (want YYYY-MM-DD >= ratchet)",
                    )
                )
            elif token < FOOTER_RATCHET:
                violations.append(
                    (
                        md,
                        i,
                        line.strip(),
                        f"footer stamp predates ratchet {FOOTER_RATCHET}",
                    )
                )
    return violations


def except_exception_occurrences(root: Path) -> int:
    """Count ``except Exception`` handlers under src/gnn (doctrine ratchet).

    Definition (pinned, docs/standards/exceptions.md): one occurrence per
    ``ast.ExceptHandler`` node whose handler type is exactly
    ``ast.Name(id="Exception")`` in every ``.py`` file under ``src/gnn``.
    Files that fail to parse are skipped; a non-parsing file fails other CI
    legs, so the ratchet never depends on an unparseable file.
    """
    total = 0
    for py in sorted((root / "src" / "gnn").rglob("*.py")):
        if "__pycache__" in py.parts or ".git" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ExceptHandler)
                and isinstance(node.type, ast.Name)
                and node.type.id == "Exception"
            ):
                total += 1
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any pattern matches or the except-Exception ratchet grows.",
    )
    args = parser.parse_args()

    targets = [ROOT / "docs", ROOT / "src" / "gnn"]
    violations = scan(targets)
    violations += footer_violations(ROOT)
    except_count = except_exception_occurrences(ROOT)
    ratchet_over = except_count > EXCEPT_EXCEPTION_BASELINE

    if not violations and not ratchet_over:
        print(
            "check_gnn_doc_patterns: no banned patterns in docs/ and src/gnn/ (markdown)."
        )
        print(
            f"check_gnn_doc_patterns: except-Exception ratchet: {except_count} occurrences"
            f" under src/gnn (baseline {EXCEPT_EXCEPTION_BASELINE})."
        )
        return 0

    if violations:
        print(f"check_gnn_doc_patterns: {len(violations)} match(es):\n")
        for path, line_no, line, desc in violations:
            rel = path.relative_to(ROOT)
            print(f"  {rel}:{line_no}: {desc}")
            print(f"    {line[:200]}")

    print(
        f"check_gnn_doc_patterns: except-Exception ratchet: {except_count} occurrences"
        f" under src/gnn (baseline {EXCEPT_EXCEPTION_BASELINE})."
    )
    if ratchet_over:
        print(
            f"check_gnn_doc_patterns: except-Exception ratchet exceeded: {except_count}"
            f" > {EXCEPT_EXCEPTION_BASELINE}; see docs/standards/exceptions.md before"
            " raising the baseline."
        )

    if args.strict and (violations or ratchet_over):
        return 1
    print("\n(non-strict: exit 0; use --strict to fail)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
