#!/usr/bin/env python3
"""Fail when maintained docs cite a file that does not exist.

A path-shaped scan cannot tell a wrong citation from a correct statement that
something is gone, so this gate **classifies before it complains**. Only one of
the five classes is a finding:

1. ``citation`` — the line implies the target exists. **A finding when the target
   is missing.** This is the only failing class.
2. ``negative`` — the line records absence or retirement (``does not exist``,
   ``no longer exists``, ``absent``, ``removed``, ``former``, ``never created``,
   ``deleted``). Correct as written: the missing target is the point.
3. ``illustrative`` — the line labels itself an example (``illustrative``), or the
   path is a registered schematic path (:data:`SCHEMATIC_PATHS`). Correct
   as written.
4. ``external-sibling`` — a document under :data:`EXTERNAL_DOC_PREFIXES` cites
   *another repository's* path. ``docs/downstream/fundamentals.md`` describing
   ``src/active_inference/gnn_spec.py`` is not a GNN path, so resolving it against
   this tree is a category error. Measured at 64 of 155 unresolved citations, which
   is why this class is first-class rather than a catch-all.
5. ``history`` — a changelog, a historical doc tree, or a document carrying
   :data:`HISTORY_MARKER`. Correct as written: those paths existed at the source
   revision the document records.

Codes 2-5 are not defects and the gate accepts them. It fails on a new ``citation``
finding, and it also fails when a recorded finding **disappears**, because a
baseline that does not shrink is an obsolete set that the next reader will mistake
for a specification.

The baseline (:data:`BASELINE_PATH`) carries its own provenance — the commit it was
measured at, the date, the document and citation population, and every class with
its count — so no count in this gate ships without the population it was counted
over. Regenerate with ``--write-baseline`` only when the residue genuinely changed,
and lower the entries as docs are fixed.

Usage:
    uv run --extra dev python scripts/check_doc_path_references.py
    uv run --extra dev python scripts/check_doc_path_references.py --write-baseline
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "scripts" / "doc_path_references_baseline.json"
BASELINE_SCHEMA = "doc-path-references-baseline/v1"

#: Path-shaped citations in backticks. Restricted to trees whose members are code
#: or build inputs: a stale ``src/`` import misleads a reader into writing a broken
#: import, and it is the class this repository has drifted in.
CITED_PATH_RE = re.compile(
    r"`((?:src|tests|scripts)/[\w./-]+\.(?:py|md|json|toml|ya?ml|j2|txt))`"
)

#: Generated output and third-party mirrors are not maintained documentation.
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "output",
    "build",
    "dist",
    "__pycache__",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    "site-packages",
    ".tox",
    ".eggs",
}

#: Documents that are records by name.
HISTORICAL_DOC_FILES = {"CHANGELOG.md", "VERSION_MAP.md"}
HISTORICAL_DOC_PREFIXES = (
    "docs/development/fleet-logs/",
    "docs/other/",
    "docs/sympy/",
)

#: Documents that cite other repositories' trees.
EXTERNAL_DOC_PREFIXES = ("docs/downstream/",)

#: Opt-in marker for any other dated record. Explicit and greppable, so a document
#: states that it is a record instead of the gate inferring it from prose.
HISTORY_MARKER = "<!-- doc-path-references: historical-record -->"

#: Lines that record absence rather than asserting existence.
NEGATIVE_RE = re.compile(
    r"(?i)\b(?:"
    r"does not exist|do not exist|did not exist|"
    r"no longer exists?|no longer exist|"
    r"absent|never (?:existed|created)|not in the tree|"
    r"(?:has|have|was|were|had) been removed|"
    r"(?:was|were|is|are) removed|"
    r"retired|deleted|former"
    r")\b"
)

#: Lines that label themselves as examples rather than citations.
ILLUSTRATIVE_RE = re.compile(r"(?i)\billustrative\b")

#: Registered schematic paths. A schematic path names a *pattern*, not a file,
#: so it is correct that no such file exists.
SCHEMATIC_PATHS = {
    "src/gnn/N_module.py": "numbered-driver naming pattern",
    "src/gnn/X_module.py": "schematic module name in the pipeline architecture diagram",
    "src/gnn/23_new_step.py": "instruction to create a new numbered driver",
}

#: Order matters only for reporting. ``citation`` is the only failing class.
CLASSES = ("citation", "negative", "illustrative", "external-sibling", "history")
FAILING_CLASS = "citation"


@dataclass(frozen=True)
class Finding:
    """One path-shaped citation and the class it was placed in."""

    doc: str
    line: int
    cited: str
    kind: str

    @property
    def key(self) -> str:
        """Baseline identity: line numbers move, so they are not part of the key."""
        return f"{self.doc}::{self.cited}"


def iter_doc_files(root: Path = ROOT) -> Iterator[Path]:
    """Maintained Markdown files: root guides, ``docs/**``, ``src/gnn/**``, ``input/``."""
    for name in ("README.md", *sorted(HISTORICAL_DOC_FILES)):
        path = root / name
        if path.exists():
            yield path
    for base in (root / "docs", root / "src" / "gnn", root / "input"):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.md")):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            yield path


def _is_historical(rel: str, text: str) -> bool:
    if Path(rel).name in HISTORICAL_DOC_FILES:
        return True
    return rel.startswith(HISTORICAL_DOC_PREFIXES) or HISTORY_MARKER in text


def classify(cited: str, line: str) -> str:
    """Classify one citation by its line. Only ``"citation"`` is a finding."""
    if cited in SCHEMATIC_PATHS:
        return "illustrative"
    if ILLUSTRATIVE_RE.search(line):
        return "illustrative"
    if NEGATIVE_RE.search(line):
        return "negative"
    return "citation"


def scan(root: Path = ROOT) -> tuple[list[Finding], int, int]:
    """Return ``(findings, documents_scanned, unresolved_citations)``."""
    findings: list[Finding] = []
    documents = citations = 0
    for path in iter_doc_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        documents += 1
        rel = path.relative_to(root).as_posix()
        historical = _is_historical(rel, text)
        external = rel.startswith(EXTERNAL_DOC_PREFIXES)
        for line_no, line in enumerate(text.splitlines(), start=1):
            for match in CITED_PATH_RE.finditer(line):
                cited = match.group(1)
                if (root / cited).exists():
                    continue
                citations += 1
                if external:
                    kind = "external-sibling"
                elif historical:
                    kind = "history"
                else:
                    kind = classify(cited, line)
                findings.append(Finding(rel, line_no, cited, kind))
    return findings, documents, citations


def class_counts(findings: list[Finding]) -> dict[str, int]:
    return {kind: sum(1 for f in findings if f.kind == kind) for kind in CLASSES}


def diff_baseline(
    findings: list[Finding], baseline: dict
) -> tuple[list[Finding], list[dict]]:
    """Return ``(new, vanished)`` for the failing class only."""
    current = {f.key: f for f in findings if f.kind == FAILING_CLASS}
    recorded = {
        f"{entry['doc']}::{entry['cited']}": entry
        for entry in baseline.get("entries", [])
    }
    new = [current[key] for key in sorted(current.keys() - recorded.keys())]
    vanished = [recorded[key] for key in sorted(recorded.keys() - current.keys())]
    return new, vanished


def build_baseline(findings: list[Finding], documents: int, citations: int) -> dict:
    entries = [
        {"doc": f.doc, "line": f.line, "cited": f.cited}
        for f in findings
        if f.kind == FAILING_CLASS
    ]
    entries.sort(key=lambda entry: (entry["doc"], entry["cited"], entry["line"]))
    return {
        "schema": BASELINE_SCHEMA,
        "measured_at_commit": _head_commit(),
        "measured_at_date": date.today().isoformat(),
        "population": {
            "documents_scanned": documents,
            "unresolved_citations": citations,
        },
        "classes": class_counts(findings),
        "entries": entries,
    }


def _head_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="record the current residue as the baseline",
    )
    args = parser.parse_args(argv)

    findings, documents, citations = scan()
    counts = class_counts(findings)

    if args.write_baseline:
        BASELINE_PATH.write_text(
            json.dumps(build_baseline(findings, documents, citations), indent=2) + "\n"
        )
        print(
            f"check_doc_path_references: wrote {BASELINE_PATH.relative_to(ROOT)} — "
            f"{documents} docs, {citations} unresolved citations, "
            + ", ".join(f"{kind}={count}" for kind, count in counts.items())
        )
        return 0

    print(
        "check_doc_path_references: "
        f"{documents} maintained docs scanned; {citations} citations of paths that "
        f"do not resolve ({', '.join(f'{k}={v}' for k, v in counts.items())})"
    )

    if not BASELINE_PATH.exists():
        print(
            f"  FAIL no baseline at {BASELINE_PATH.relative_to(ROOT)}; "
            "run with --write-baseline to record the current residue"
        )
        return 1

    baseline = json.loads(BASELINE_PATH.read_text())
    if baseline.get("schema") != BASELINE_SCHEMA:
        print(f"  FAIL baseline schema is not {BASELINE_SCHEMA}")
        return 1

    new, vanished = diff_baseline(findings, baseline)
    status = 0
    for finding in new:
        print(f"  FAIL new citation {finding.doc}:{finding.line}: {finding.cited}")
        status = 1
    for entry in vanished:
        print(
            f"  FAIL baseline entry no longer present: {entry['doc']} -> "
            f"{entry['cited']} (fixed? regenerate the baseline so the set shrinks)"
        )
        status = 1

    recorded = len(baseline.get("entries", []))
    if status == 0:
        print(
            f"  ok no new or vanished citations; {recorded} recorded residue "
            f"entries remain (baseline measured at "
            f"{baseline.get('measured_at_commit', '?')[:10]})"
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
