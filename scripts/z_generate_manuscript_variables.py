#!/usr/bin/env python3
"""Hydrate GeneralizedNotationNotation manuscript variables before rendering.

Thin orchestrator (docxology/template contract). The render pipeline
(``infrastructure.rendering._manuscript_source.run_manuscript_variable_script``)
invokes this exact script name automatically before PDF render.

Responsibilities (delegated to :mod:`gnn.manuscript`):

1. Compute the deterministic ``{{TOKEN}}`` map from the repository at ``HEAD``.
2. Write the producer-owned ``manuscript/config.yaml`` fields (``version:``,
   ``date:``) from that map — config.yaml is never token-substituted, so those
   title-page literals can only stay true by being written.
3. Persist the map to ``output/data/manuscript_variables.json`` for audit/debug.
4. Hydrate ``manuscript/*.md`` into ``output/manuscript/*.md`` with tokens
   resolved, via the template injector when available.

Every count in the map describes the commit reported as ``GNN_GIT_COMMIT``. A
dirty working tree is reported on stderr because the rendered numbers will then
describe the last commit rather than what is on disk.

Runs standalone too (``python scripts/z_generate_manuscript_variables.py``);
the template hydration step is skipped gracefully there when the sibling
template ``infrastructure`` package is not importable.

Two modes, told apart mechanically:

* **Standalone** (a developer ran this script; ``TEMPLATE_REPO_ROOT`` unset):
  hydration is a best effort — when the injector cannot be imported the script
  writes the variables and exits 0.
* **Render-invoked** (the template launcher exported ``TEMPLATE_REPO_ROOT`` —
  that export is the chosen detection mechanism, documented in
  ``infrastructure.rendering._manuscript_source.run_manuscript_variable_script``):
  hydration is the contract. If the template root / injector cannot be found
  the render would ship an unsubstituted manuscript, so the script FAILS
  (exit 1) instead of degrading to variables-only.

A machine-readable provenance receipt is always written to
``output/data/manuscript_variables_receipt.json`` (which commit the counts
describe, whether the working tree is dirty, and the dirty paths), so the
dirty-tree caveat is a file the gates can read, not just a stderr line.

Scheduled re-render ritual (SC-22): ``.github/workflows/local-gates.yml``
runs ``check_manuscript_tokens.py --strict`` on every push/PR to main (and
``workflow_dispatch`` — event-driven per commit, not a cron schedule) and
proves the *committed* artifacts describe ``HEAD``: the token map's
commit-stable checksum matches a fresh ``generate_variables``, every figure
registry ``png_sha256`` matches its PNG, and every ``source_sha256`` matches
the bytes the prose describes. What it cannot do is render. So when HEAD
moves with count-affecting changes, that gate fails until this manual
ritual has run, in order:

1. regenerate — this script, invoked by the template render pipeline (or
   standalone, when only the token map is wanted);
2. rebuild the figures (``python -m scripts.manuscript_build_figures``);
3. run the template's ``stage_03_render``;
4. record the render custody manifest
   (``uv run python scripts/z_record_manuscript_render_manifest.py``);
5. commit the regenerated ``output/`` artifacts (including
   ``output/data/manuscript_render_manifest.json``).

PR-time drift guard: ``scripts/check_hydrated_prose.py`` (``local-gates.yml``
``repo-gates``) re-substitutes ``manuscript/*.md`` from the committed token map
and fails when the committed ``output/manuscript/`` differs. So a PR that
regenerates the token map (every count-changing PR) must run this whole
ritual. Regenerating and re-recording alone now fails before merge, where it
used to fail only in the scheduled ``custody-re-render.yml`` after merge.

What the gates verify afterwards: the strict token checks above, plus —
via the custody manifest read by ``tests/test_manuscript_latex_log.py`` —
that the committed ``.log``/``.tex``/``.md``, the hydrated sections under
``output/manuscript/``, the token map, and the receipt are artifacts of one
render invocation. What remains manual: running the render in the template
checkout and committing the artifacts — no gate can run it for you, and
since the custody manifest is recorded from the files on disk, the
"render actually happened before the record" ordering is the one part of
the chain the operator performs on trust.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from gnn.manuscript import (  # noqa: E402
    generate_variables,
    save_variables,
    sync_config_metadata,
    sync_preamble_metadata,
)

_RENDER_INVOCATION_ENV = "TEMPLATE_REPO_ROOT"
_RECEIPT_REL = Path("output") / "data" / "manuscript_variables_receipt.json"
_DIRTY_PATH_LIMIT = 200

_INJECTION_REL = Path("infrastructure") / "rendering" / "manuscript_injection.py"


def _discover_template_root() -> Path | None:
    """Locate the sibling docxology/template checkout that owns the injector.

    The render launcher (``run_manuscript_variable_script``) runs this script with
    ``cwd=project_root`` and exports ``TEMPLATE_REPO_ROOT``. For a symlinked
    ``working/`` project the in-tree ``parents[2]`` trick does not reach the template
    root, so we resolve it from the env var first, then by walking up from both the
    invocation cwd (which may be the symlink path) and the project root.
    """
    candidates: list[Path] = []
    env_root = os.environ.get("TEMPLATE_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    for base in (Path.cwd(), Path(__file__).resolve().parent, _PROJECT_ROOT):
        candidates.append(base)
        candidates.extend(base.parents)
    for cand in candidates:
        if (cand / _INJECTION_REL).is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    return None


def _render_invoked() -> bool:
    """True when the docxology render pipeline invoked this script.

    ``run_manuscript_variable_script`` exports ``TEMPLATE_REPO_ROOT`` before
    calling this exact script name; a developer running the script by hand
    does not set it. Detection deliberately keys on the exported variable
    rather than a cwd heuristic: the launcher is the only caller that both
    knows the template root and requires hydration to have happened.
    """
    return bool(os.environ.get(_RENDER_INVOCATION_ENV))


def _dirty_state() -> dict:
    """``git status --porcelain`` as data: ``{"git_available", "dirty_paths"}``."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        return {"git_available": False, "dirty_paths": []}
    if result.returncode != 0:
        return {"git_available": False, "dirty_paths": []}
    paths = sorted(line[3:] for line in result.stdout.splitlines() if line.strip())
    return {"git_available": True, "dirty_paths": paths}


def _dirty_receipt(variables: Mapping[str, str], state: dict) -> dict:
    """Assemble the machine-readable provenance receipt (pure; no I/O)."""
    paths: list[str] = state["dirty_paths"]
    return {
        "receipt_version": "gnn_manuscript_variables_receipt_v1",
        "generator": "scripts/z_generate_manuscript_variables.py",
        "counts_describe_commit": variables.get("GNN_GIT_COMMIT", "unknown"),
        "git_available": state["git_available"],
        "working_tree_clean": state["git_available"] and not paths,
        "dirty_path_count": len(paths),
        "dirty_paths": paths[:_DIRTY_PATH_LIMIT],
    }


def _report_dirty_tree(variables: Mapping[str, str]) -> dict:
    """Write the receipt under ``output/data/``; keep the stderr warning."""
    receipt = _dirty_receipt(variables, _dirty_state())
    receipt_path = (
        _PROJECT_ROOT / "output" / "data" / ("manuscript_variables_receipt.json")
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not receipt["git_available"]:
        print(
            "[manuscript-variables] git unavailable; counts describe the working "
            "tree with no commit provenance (receipt: "
            f"{receipt_path.relative_to(_PROJECT_ROOT)})",
            file=sys.stderr,
        )
    elif receipt["dirty_path_count"]:
        print(
            f"[manuscript-variables] working tree has "
            f"{receipt['dirty_path_count']} uncommitted change(s); every count "
            "describes the last commit, not the files on disk "
            f"(receipt: {receipt_path.relative_to(_PROJECT_ROOT)})",
            file=sys.stderr,
        )
    return receipt


def main() -> int:
    variables = generate_variables(_PROJECT_ROOT)
    for change in sync_config_metadata(_PROJECT_ROOT, variables):
        print(f"[manuscript-variables] config.yaml {change}", file=sys.stderr)
    for change in sync_preamble_metadata(_PROJECT_ROOT, variables):
        print(f"[manuscript-variables] preamble.md {change}", file=sys.stderr)
    _report_dirty_tree(variables)
    out_path = _PROJECT_ROOT / "output" / "data" / "manuscript_variables.json"
    save_variables(variables, out_path)

    _discover_template_root()
    try:
        from infrastructure.rendering.manuscript_injection import (
            write_resolved_manuscript_tree,
        )
    except ModuleNotFoundError:
        if _render_invoked():
            print(
                "[manuscript-variables] render invocation could not load the "
                f"template injector (looked for {_INJECTION_REL.as_posix()}; "
                f"{_RENDER_INVOCATION_ENV} is set). The render would ship an "
                "unsubstituted manuscript — failing instead of writing "
                "variables only",
                file=sys.stderr,
            )
            return 1
        print(
            "[manuscript-variables] template injector unavailable; "
            "wrote variables only (standalone mode)",
            file=sys.stderr,
        )
        print(str(out_path))
        return 0

    write_resolved_manuscript_tree(_PROJECT_ROOT, variables)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
