"""Record actual JUnit execution counts and the tested Git identity for CI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from defusedxml import ElementTree


def build_receipt(path: Path) -> dict[str, object]:
    """Count test cases rather than double-counting nested suite aggregates."""
    root = ElementTree.parse(path).getroot()
    cases = list(root.iter("testcase"))
    skipped = sum(case.find("skipped") is not None for case in cases)
    failed = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "schema_version": 1,
        "commit": revision,
        "working_tree_dirty": bool(dirty),
        "identity_claim": "commit plus working tree" if dirty else "committed checkout",
        "python": sys.version,
        "junit_path": path.as_posix(),
        "executed": len(cases) - skipped,
        "passed": len(cases) - skipped - failed - errors,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "complete": bool(len(cases) - skipped) and not (failed or errors),
    }


def main() -> int:
    """Write adjacent JSON plus the GitHub summary; missing evidence fails."""
    path = Path(sys.argv[1])
    receipt = build_receipt(path)
    path.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as summary:
            summary.write(f"\nTested commit: `{receipt['commit']}`\n\n")
            summary.write(
                f"Executed: {receipt['executed']}; passed: {receipt['passed']}; "
                f"failed: {receipt['failed']}; errors: {receipt['errors']}; "
                f"skipped: {receipt['skipped']}.\n"
            )
    print(json.dumps(receipt))
    return 0 if receipt["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
