#!/usr/bin/env python3
"""Prepare render evidence after Stage 03 or record it after Stage 04.

The default command intentionally renews the custody manifest. The scheduled
verifier checks the existing manifest and never calls the default command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from gnn.manuscript.render_custody import (  # noqa: E402
    manifest_path,
    prepare_render_artifacts,
    record_render_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="remove personal path prefixes before Stage 04",
    )
    if parser.parse_args().prepare:
        for rel in prepare_render_artifacts(_PROJECT_ROOT):
            print(f"[render-custody] sanitized home path prefixes in {rel}")
        return 0
    try:
        record_render_manifest(_PROJECT_ROOT)
    except RuntimeError as exc:
        print(f"[render-custody] {exc}", file=sys.stderr)
        return 1
    print(manifest_path(_PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
