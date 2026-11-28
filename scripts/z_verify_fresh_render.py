#!/usr/bin/env python3
"""Check the committed publication evidence or verify a validated fresh render.

Run --committed-only before rendering. After template Stage 04, the default
check validates the fresh deliveries and compares reproducible intermediates
with the committed custody manifest. Exit 1 signals drift or invalid evidence,
exit 3 signals a stale template revision, and exit 4 signals an unmeasured
comparison environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from gnn.manuscript.render_custody import (  # noqa: E402
    STALE_CHAIN,
    UNMEASURED,
    custody_issues,
    environment_report,
    load_render_manifest,
    manifest_path,
    publication_validation_issues,
    verify_fresh_render,
)


def _report_environment() -> None:
    """Print what the comparison can see, before it reports a verdict."""
    try:
        manifest = load_render_manifest(_PROJECT_ROOT)
    except RuntimeError as exc:
        print(f"[render-custody] [ENV] unavailable: {exc}", file=sys.stderr)
        return
    for line in environment_report(_PROJECT_ROOT, manifest):
        print(f"[render-custody] [ENV] {line}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--committed-only", action="store_true")
    if parser.parse_args().committed_only:
        issues = custody_issues(_PROJECT_ROOT)
        for issue in issues:
            print(f"[render-custody] [FAIL] {issue}", file=sys.stderr)
        if not issues:
            print(f"committed evidence matches {manifest_path(_PROJECT_ROOT)}")
        return int(bool(issues))

    _report_environment()
    issues = verify_fresh_render(_PROJECT_ROOT)
    if not any(issue.startswith((STALE_CHAIN, UNMEASURED)) for issue in issues):
        issues.extend(
            f"[FAIL] {issue}" for issue in publication_validation_issues(_PROJECT_ROOT)
        )
    for issue in issues:
        print(f"[render-custody] {issue}", file=sys.stderr)
    if any(issue.startswith(STALE_CHAIN) for issue in issues):
        return 3
    if any(issue.startswith(UNMEASURED) for issue in issues):
        return 4
    if any(issue.startswith("[FAIL]") for issue in issues):
        return 1
    if not issues:
        print(f"fresh render matches {manifest_path(_PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
