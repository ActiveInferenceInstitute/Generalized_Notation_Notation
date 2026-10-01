#!/usr/bin/env python3
"""Check documentation contracts that link and scaffold audits cannot verify.

This intentionally checks a small set of high-value invariants:
- the primary quickstart contains every enforced GNN section;
- maintained command examples use current pipeline flag spellings;
- the configuration guide names the automatic ``input/config.yaml`` path;
- documentation distinguishes ten render targets from eleven Step-12 executor
  families and names the bnlearn executor (``src/gnn/execute/bnlearn/``);
- the primary hub does not claim generated counts or production readiness.

Run from the repository root::

    uv run --extra dev python scripts/check_doc_contracts.py --strict
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_QUICKSTART_SECTIONS = (
    "GNNSection",
    "GNNVersionAndFlags",
    "ModelName",
    "StateSpaceBlock",
    "Connections",
)

# These are pipeline flags that were repeatedly documented but are not exposed by
# the current main/numbered-script parsers. Explanatory prose is allowed when the
# same line explicitly marks a spelling as unsupported/obsolete.
STALE_COMMAND_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"src/gnn/main\.py[^\n]*--config(?:-file)?\b"),
        "main.py config override",
    ),
    (re.compile(r"src/gnn/main\.py[^\n]*--skip\s+(?!steps)"), "main.py --skip"),
    (re.compile(r"src/gnn/main\.py[^\n]*--debug\b"), "main.py --debug"),
    (re.compile(r"src/gnn/main\.py[^\n]*--dry-run\b"), "main.py --dry-run"),
    (
        re.compile(r"src/gnn/main\.py[^\n]*--memory-efficient\b"),
        "main.py --memory-efficient",
    ),
    (re.compile(r"src/gnn/12_execute\.py[^\n]*--dry-run\b"), "Step 12 --dry-run"),
    (
        re.compile(r"src/gnn/11_render\.py[^\n]*--force-regenerate\b"),
        "Step 11 --force-regenerate",
    ),
    (
        re.compile(r"src/gnn/1_setup\.py[^\n]*--install_optional\b"),
        "underscore setup flag",
    ),
    (
        re.compile(r"src/gnn/1_setup\.py[^\n]*--optional_groups\b"),
        "underscore setup flag",
    ),
    (re.compile(r"src/gnn/1_setup\.py[^\n]*--recreate-venv\b"), "obsolete venv flag"),
)


def _is_explanatory_line(line: str) -> bool:
    lowered = line.lower()
    return any(
        marker in lowered
        for marker in (
            "not supported",
            "unsupported",
            "obsolete",
            "old spelling",
            "no ",
            "there is no",
        )
    )


def scan() -> list[str]:
    """Return human-readable contract violations."""
    issues: list[str] = []
    quickstart = ROOT / "docs" / "gnn" / "tutorials" / "quickstart_tutorial.md"
    text = quickstart.read_text(encoding="utf-8")
    positions = [text.find(f"## {section}") for section in REQUIRED_QUICKSTART_SECTIONS]
    if any(position < 0 for position in positions):
        missing = [
            section
            for section, position in zip(REQUIRED_QUICKSTART_SECTIONS, positions)
            if position < 0
        ]
        issues.append(f"quickstart missing enforced sections: {', '.join(missing)}")
    elif positions != sorted(positions):
        issues.append("quickstart enforced sections are out of order")

    config = ROOT / "docs" / "configuration" / "README.md"
    config_text = config.read_text(encoding="utf-8")
    if "input/config.yaml" not in config_text:
        issues.append("configuration guide does not name input/config.yaml")

    hub = (ROOT / "docs" / "README.md").read_text(encoding="utf-8").lower()
    for forbidden in ("doc_markdown_files", "production_ready", "600+", "610 markdown"):
        if forbidden in hub:
            issues.append(
                f"primary documentation hub contains generated/stale metadata: {forbidden}"
            )

    pipeline = (ROOT / "docs" / "pipeline" / "README.md").read_text(encoding="utf-8")
    if "10 render" not in pipeline.lower() or "11 executor" not in pipeline.lower():
        issues.append(
            "pipeline guide does not distinguish ten render targets and eleven "
            "executor families"
        )
    if "src/gnn/execute/bnlearn/" not in pipeline:
        issues.append(
            "pipeline guide does not name the bnlearn executor (src/gnn/execute/bnlearn/)"
        )

    for path in sorted((ROOT / "docs").rglob("*.md")):
        if any(
            part in {"output", "other", ".git", "__pycache__", "fleet-logs"}
            for part in path.parts
        ):
            continue
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            for pattern, description in STALE_COMMAND_PATTERNS:
                if pattern.search(line) and not _is_explanatory_line(line):
                    relative = path.relative_to(ROOT)
                    issues.append(f"{relative}:{line_no}: unsupported {description}")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict", action="store_true", help="Exit 1 when a contract fails"
    )
    args = parser.parse_args()
    issues = scan()
    if not issues:
        print("check_doc_contracts: documentation contracts pass.")
        return 0
    print(f"check_doc_contracts: {len(issues)} issue(s):")
    for issue in issues:
        print(f"  - {issue}")
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
