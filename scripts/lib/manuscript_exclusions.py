#!/usr/bin/env python3
"""Single source for the manuscript filename exclusion sets.

Four call sites used to hand-copy these sets — the token gate's standalone
fallback, the figure build's label scan, the published-commands test and the
figure-build test — and they drifted independently (adding a fifth authoring
guide meant editing four places; missing one silently scanned or skipped the
wrong files). They live here now, with the contract stated once:

``EXCLUDED_DOC_FILENAMES``
    ``manuscript/*.md`` files the render pipeline does NOT token-substitute,
    so no scanner may treat their text as published prose. Imported from the
    shared ``gnn.manuscript.substitution`` contract, which prefers the template's
    actual value when importable and otherwise uses the headless fallback.

``AUTHORING_GUIDE_FILENAMES``
    The subset that is a mere authoring guide (examples, commands, embedded
    figures that are not the manuscript's own). Label scans and command scans
    skip these so a worked example in SYNTAX.md is never read as a declared
    figure or a published command. A superset would hide real content, so
    each consumer imports exactly this set rather than re-deciding.
"""

from __future__ import annotations

import sys
from pathlib import Path

# These script helpers also run from an uninstalled source checkout.
_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))


from gnn.manuscript.substitution import EXCLUDED_DOC_FILENAMES  # noqa: E402

__all__ = ["EXCLUDED_DOC_FILENAMES", "AUTHORING_GUIDE_FILENAMES"]

AUTHORING_GUIDE_FILENAMES = frozenset({"SYNTAX.md", "README.md", "AGENTS.md"})
