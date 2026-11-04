#!/usr/bin/env python3
"""Verify the fresh render against the committed custody manifest.

The step before recording in the scheduled re-render ritual (SC-22 — see
``scripts/z_generate_manuscript_variables.py``'s docstring): after the
template's ``stage_03_render`` has produced fresh PDF evidence and hydrated
prose, compare them to ``output/data/manuscript_render_manifest.json`` as
committed, BEFORE the record step overwrites it — so a red run certifies the
committed chain rather than just render success. ``[STALE-CHAIN]`` means the committed chain describes a different template
revision than the fresh render observed — that finding is returned alone,
no artifact comparison is attempted, and the exit code is **3** (a distinct
status, not a generic failure: the chain is out of date, not the render);
``[FAIL]`` means an artifact the comparison covers disagrees with the
committed chain (the manifest names the profile it compared under, and a
permitted-but-different profile narrows what is covered — see the
``[ENV]`` scope line); ``[UNMEASURED]`` means the gate cannot say: the chain
declares no usable render profile, or the template revision this render used
is unknown. There is no warning level for an artifact comparison: bytes that
disagree with a chain from a permitted profile are drift.

Thin orchestrator, same shape as ``z_record_manuscript_render_manifest.py``:
all logic lives in :mod:`gnn.manuscript.render_custody`. Exits 3 iff any
``[STALE-CHAIN]``, 4 iff any ``[UNMEASURED]``, 1 iff any ``[FAIL]``, and
prints the environment report (``[ENV]`` lines, including the comparison's
scope) before the verdict on every run, so what the comparison could and
could not see is always on the record — and a green run is never read as
having compared an artifact it did not.

Usage:
    uv run python scripts/z_verify_fresh_render.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from gnn.manuscript.render_custody import (  # noqa: E402
    STALE_CHAIN,
    UNMEASURED,
    environment_report,
    load_render_manifest,
    manifest_path,
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
    _report_environment()
    issues = verify_fresh_render(_PROJECT_ROOT)
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
