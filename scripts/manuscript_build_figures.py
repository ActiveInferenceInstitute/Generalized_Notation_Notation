#!/usr/bin/env python3
"""Regenerate every manuscript figure deterministically.

Thin orchestrator that runs each ``scripts/manuscript_fig_*.py`` generator in turn,
so the manuscript's figures under ``output/figures/`` can be rebuilt from source with
a single command before rendering (the render pipeline does not regenerate figures).

It also writes ``output/figures/figure_registry.json``, the accessibility registry
the template's ``validate_figure_registry`` reads. The registry is *generated here*
rather than hand-authored precisely because it must not drift from the figures that
are actually built: the label, the produced filename and the generator that made it
all come from the same table below. Only ``alt_text`` is authored — it must describe
what a reader who cannot see the figure needs, in different words from the caption
(the caption is a title; the alt text says what the image shows).

Each entry also carries build provenance, which is what stops a stale figure from
shipping. ``png_sha256`` is the digest of the PNG this build produced, and
``consumed_tokens`` is the ``{key: value}`` map the generator actually read out of
``output/data/manuscript_variables.json`` (observed by
``scripts/lib/manuscript_figure_tokens.py``, not declared).
``tests/test_manuscript_figure_freshness.py`` re-checks both against the
committed PNG and the live token map, so a figure built before a count moved fails
the suite. Without it, ``fig:repo_metrics`` shipped "365 test files" on the same PDF
page as prose reading 367.

Usage:
    python scripts/manuscript_build_figures.py
The token map is regenerated unconditionally first (``z_generate``) and the
build fails unless its ``GNN_GIT_COMMIT`` equals this checkout's HEAD, so a
stale-but-present ``manuscript_variables.json`` can never be baked into fresh
PNGs. Exit code is non-zero if the regeneration or the commit check fails, any
generator fails, any expected PNG is missing, or the registry does not cover
every ``{#fig:...}`` label the manuscript declares.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.lib.manuscript_exclusions import AUTHORING_GUIDE_FILENAMES  # noqa: E402

_FIG_DIR = _PROJECT_ROOT / "output" / "figures"
_MANUSCRIPT_DIR = _PROJECT_ROOT / "manuscript"
_REGISTRY_PATH = _FIG_DIR / "figure_registry.json"
_PROVENANCE_ENV = "GNN_FIGURE_TOKEN_PROVENANCE"
_Z_GENERATE = "z_generate_manuscript_variables.py"

# (crossref label, generator script, expected PNG, alt text).
#
# The label is what the manuscript writes as {#fig:...}; a mismatch fails the
# build below rather than shipping an unregistered figure. The alt text is the
# only authored field, and is deliberately not the caption: a screen-reader user
# needs the figure's content, not its title.
_FIGURES = [
    (
        "fig:graphical_abstract",
        "manuscript_fig_graphical_abstract.py",
        "gnn_graphical_abstract.png",
        "End-to-end summary panel of the GNN pipeline in seven left-to-right stage "
        "cards: a plain-text GNN document declaring A, B, C, D and E; parsing; "
        "validation and type checking with the B-tensor orientation check; code "
        "rendering to the registered backends; execution of the backends including "
        "the bnlearn lane; analysis; and cross-repository interchange with "
        "GEO-INFER and fep_lean. Step pills name the pipeline step behind each "
        "stage, a callout states the two newest capabilities, and a footer strip "
        "carries the producer's scale counts for steps, families, backends and "
        "MCP tools.",
    ),
    (
        "fig:pipeline",
        "manuscript_fig_pipeline_dag.py",
        "gnn_pipeline_dag.png",
        "Directed acyclic graph of the numbered GNN pipeline steps. Nodes are the "
        "step modules under src/gnn/, arranged in rowwise topological order across "
        "five columns. Each node shows the complete step number and module name; "
        "the legend above the nodes identifies execution phases by color. Arrows "
        "show the required orchestration prerequisites declared in "
        "src/gnn/STEP_INDEX.md, linking each prerequisite step to its consumers. "
        "Optional artifact enrichments are documented separately.",
    ),
    (
        "fig:family_matrix",
        "manuscript_fig_family_framework.py",
        "gnn_family_framework_matrix.png",
        "Grid of model families (rows) against rendering backends (columns). A "
        "filled cell means the family declares that backend in "
        "input/model_family_manifest.json. The grid is sparse: most families "
        "declare a single backend, and only the continuous, hierarchical and "
        "gridworld families declare several. Cells record declared coverage, not "
        "profiled outcomes.",
    ),
    (
        "fig:backend_matrix",
        "manuscript_fig_backend_matrix.py",
        "gnn_backend_capability_matrix.png",
        "Table of the registered rendering backends, one row each, with columns "
        "for registry key, display name, implementation language, whether a "
        "render-output subdirectory exists under src/gnn/render/, and the backend's "
        "role in the cross-framework reference comparison. Highlighted rows are "
        "the backends the reliability gate profiles; a backend the family declares "
        "but the gate never profiles is marked separately.",
    ),
    (
        "fig:repo_metrics",
        "manuscript_fig_repo_metrics.py",
        "gnn_repo_metrics.png",
        "Horizontal bar chart on a logarithmic axis of repository-scale counts: "
        "pipeline steps, model families, registered backends, execution backends, "
        "MCP tools, source packages, test files, example models and documentation "
        "files. Each bar is annotated with its exact value; all values come from "
        "output/data/manuscript_variables.json.",
    ),
    (
        "fig:triple_play",
        "manuscript_fig_triple_play.py",
        "gnn_triple_play.png",
        "The Triple Play as one central GNN specification with three projections "
        "radiating from it: the readable text form, graphical visualizations of "
        "state-space and factor structure, and executable model code. All three "
        "point back to the single specification at the center.",
    ),
    (
        "fig:orchestration",
        "manuscript_fig_orchestration.py",
        "gnn_orchestration.png",
        "The three long-running orchestration contracts side by side — durable "
        "observation streams, resumable run sessions, and auditable container "
        "plans — each shown with the artifacts it generates and validates, and "
        "each stopping short of any live mutation.",
    ),
]

# Repo-relative data surfaces each generator reads DIRECTLY (not through the
# token map): the step index for the DAG, the family manifest and framework
# registry for the two matrices, plus the selection helper's module for
# ``fig:backend_matrix``. These are the SC-21 risk: prose counts come from the
# HEAD ``RepositorySnapshot`` while these generators read the working tree, so
# the build digests them into ``source_sha256`` in figure_registry.json and
# the gates compare those digests against the files (freshness suite) and
# against HEAD (``check_manuscript_tokens --strict``) — a figure built from a
# source the prose does not describe fails instead of shipping.
_FIGURE_SOURCES: dict[str, tuple[str, ...]] = {
    "fig:pipeline": ("src/gnn/STEP_INDEX.md",),
    "fig:family_matrix": (
        "input/model_family_manifest.json",
        "src/gnn/render/framework_registry.py",
    ),
    "fig:backend_matrix": (
        "src/gnn/render/framework_registry.py",
        "input/model_family_manifest.json",
        "src/gnn/pipeline/cross_framework_reliability.py",
        "src/gnn/manuscript/variables.py",
    ),
}

_FIG_LABEL_RE = re.compile(r"\{#(fig:[\w:-]+)")
# manuscript/SYNTAX.md is the authoring guide; its example embeds are not figures.
# Shared with the token gate and the published-commands test — see
# scripts/lib/manuscript_exclusions.py.
_LABEL_SCAN_SKIP = AUTHORING_GUIDE_FILENAMES


def _declared_labels() -> set[str]:
    """Every ``{#fig:...}`` label the manuscript sections actually declare."""
    labels: set[str] = set()
    for md in sorted(_MANUSCRIPT_DIR.glob("*.md")):
        if md.name in _LABEL_SCAN_SKIP:
            continue
        labels.update(_FIG_LABEL_RE.findall(md.read_text(encoding="utf-8")))
    return labels


def _sha256(path: Path) -> str:
    """Digest of a built artifact, so a swapped or stale PNG is detectable.

    Returns the empty string when the figure was not produced: this build has
    already recorded that as a failure and exits non-zero, and an empty digest
    can never match a real PNG in the freshness gate.
    """
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_registry(provenance: dict[str, dict[str, str]]) -> list[str]:
    """Write figure_registry.json; return coverage problems (empty when clean)."""
    declared = _declared_labels()
    registered = {label for label, _, _, _ in _FIGURES}
    problems = [
        f"manuscript declares {label} with no registry entry"
        for label in sorted(declared - registered)
    ]
    problems += [
        f"registry entry {label} is declared by no manuscript section"
        for label in sorted(registered - declared)
    ]
    payload = {
        "schema_version": "gnn_figure_registry_v2",
        "figures": [
            {
                "label": label,
                "filename": png,
                "alt_text": alt,
                "generated_by": f"scripts/{script}",
                "png_sha256": _sha256(_FIG_DIR / png),
                "source_sha256": {
                    rel: _sha256(_PROJECT_ROOT / rel)
                    for rel in _FIGURE_SOURCES.get(label, ())
                },
                "consumed_tokens": provenance.get(label, {}),
            }
            for label, script, png, alt in _FIGURES
        ],
    }
    _REGISTRY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return problems


def _generated_commit_mismatch(variables_json: Path) -> str:
    """Failure message when the regenerated map is not this checkout's HEAD.

    Every figure is built from counts that describe one commit; if the map
    names another commit (or reports ``unknown`` because git was unavailable)
    the PNGs and the provenance record would ship numbers the checkout cannot
    reproduce.
    """
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    head_commit = head.stdout.strip() if head.returncode == 0 else "unknown"
    try:
        generated = json.loads(variables_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (
            f"{variables_json.relative_to(_PROJECT_ROOT)} is unreadable after "
            f"regeneration: {exc}"
        )
    commit = str(generated.get("GNN_GIT_COMMIT", "unknown"))
    if commit == "unknown":
        return (
            "regenerated manuscript variables report GNN_GIT_COMMIT=unknown "
            "(git unavailable): figures cannot be built from counts with no "
            "commit provenance"
        )
    if commit != head_commit:
        return (
            f"regenerated manuscript variables describe commit {commit} but "
            f"this checkout is at {head_commit}"
        )
    return ""


def main() -> int:
    # Repo metrics reads output/data/manuscript_variables.json — regenerate it
    # UNCONDITIONALLY. A stale-but-present map used to be baked into fresh
    # PNGs and re-recorded as consumed_tokens provenance; the only thing the
    # build may consume is what the producer emits right now.
    variables_json = _PROJECT_ROOT / "output" / "data" / "manuscript_variables.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "scripts" / _Z_GENERATE),
        ],
        cwd=str(_PROJECT_ROOT),
        check=False,
    )
    if result.returncode != 0:
        print(
            f"\n❌ {_Z_GENERATE} exited {result.returncode} — the variables "
            "JSON was not regenerated; refusing to build figures from a map "
            "of unknown freshness"
        )
        return 1
    commit_mismatch = _generated_commit_mismatch(variables_json)
    if commit_mismatch:
        print("\n❌ Figure build FAILED:")
        print(f"  ✗ {commit_mismatch}")
        return 1

    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    provenance: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for label, script, png, _alt in _FIGURES:
            script_path = _PROJECT_ROOT / "scripts" / script
            # Each generator writes the token keys it actually read here.
            token_record = Path(tmp) / f"{label.replace(':', '_')}.json"
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(_PROJECT_ROOT),
                env={
                    "MPLBACKEND": "Agg",
                    _PROVENANCE_ENV: str(token_record),
                    **_environ(),
                },
                check=False,
            )
            out = _FIG_DIR / png
            if result.returncode != 0:
                failures.append(f"{script} exited {result.returncode}")
                continue
            if not (out.is_file() and out.stat().st_size > 5_000):
                failures.append(f"{script} did not produce {png} (>5KB)")
                continue
            # Absent record = the generator reads no tokens (e.g. fig:orchestration).
            provenance[label] = (
                json.loads(token_record.read_text(encoding="utf-8"))
                if token_record.is_file()
                else {}
            )
            print(f"  ✓ {png} ({out.stat().st_size // 1024} KB)")

    failures.extend(_write_registry(provenance))

    if failures:
        print("\n❌ Figure build FAILED:")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print(f"\n✅ Built {len(_FIGURES)} manuscript figures in {_FIG_DIR}")
    print(f"✅ Wrote {_REGISTRY_PATH.relative_to(_PROJECT_ROOT)}")
    return 0


def _environ() -> dict[str, str]:
    import os

    return os.environ.copy()


if __name__ == "__main__":
    raise SystemExit(main())
