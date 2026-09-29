#!/usr/bin/env python3
"""Fail when the committed hydrated prose is stale for the committed token map.

The PR-time custody drift guard (SC-22 — see
``scripts/z_generate_manuscript_variables.py``'s docstring). A count-changing
PR regenerates ``output/data/manuscript_variables.json`` and re-records the
custody manifest, so every PR-time gate passes; but the hydrated sections
under ``output/manuscript/`` and the PDF evidence rendered from them only
change when the full ritual runs, and the scheduled
``custody-re-render.yml`` then goes red after the merge. This check catches
that before merge, with no LaTeX: it re-substitutes ``manuscript/*.md``
from the committed token map and compares the result to the committed
hydrated tree byte for byte. A failure means: run the full ritual (figures,
the template's ``stage_03_render``, then
``scripts/z_record_manuscript_render_manifest.py``) and commit ``output/``.

Thin orchestrator: the logic is
:func:`gnn.manuscript.render_custody.hydration_issues`. Exits 1 on any issue.

Usage:
    uv run python scripts/check_hydrated_prose.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (_PROJECT_ROOT / "src", _PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from gnn.manuscript.render_custody import (  # noqa: E402
    HYDRATION_FIX,
    hydration_issues,
)
from scripts.lib.manuscript_exclusions import EXCLUDED_DOC_FILENAMES  # noqa: E402


def main() -> int:
    issues = hydration_issues(_PROJECT_ROOT, EXCLUDED_DOC_FILENAMES)
    for issue in issues:
        print(f"[hydrated-prose] {issue}", file=sys.stderr)
    if issues:
        print(f"[hydrated-prose] fix: {HYDRATION_FIX}", file=sys.stderr)
        return 1
    print("output/manuscript/ matches the committed token map")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
