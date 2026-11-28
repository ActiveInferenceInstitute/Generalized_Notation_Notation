"""Regression tests for ``scripts/manuscript_build_figures.py``.

The build orchestrator declares a ``(label, generator, expected PNG, alt text)``
table that is the single source for three things at once: what gets built, what
``output/figures/figure_registry.json`` records, and what alt text ships with
each figure. These tests lock that contract mechanically, so a new
``manuscript_fig_*.py`` generator that is not registered, a stale entry pointing
at a removed generator, a figure the manuscript references without a registry
entry, or a registry entry with no alt text all fail immediately instead of
silently producing an incomplete or inaccessible figure set.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Single-source the exclusion set (SC-20): identical semantics to the
# hand-copied set this file previously duplicated.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from lib.manuscript_exclusions import (
    AUTHORING_GUIDE_FILENAMES as _LABEL_SCAN_SKIP,  # noqa: E402
)

SCRIPT = REPO_ROOT / "scripts" / "manuscript_build_figures.py"
REGISTRY = REPO_ROOT / "output" / "figures" / "figure_registry.json"
MANUSCRIPT = REPO_ROOT / "docs" / "manuscript"
_FIG_LABEL_RE = re.compile(r"\{#(fig:[\w:-]+)")


def _figures_constant() -> list[tuple[str, str, str, str]]:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_FIGURES" for t in node.targets
        ):
            entries: list[tuple[str, str, str, str]] = []
            assert isinstance(node.value, (ast.List, ast.Tuple))
            for item in node.value.elts:
                assert isinstance(item, ast.Tuple)
                label, gen, png, alt = item.elts
                entries.append(
                    (
                        ast.literal_eval(label),
                        ast.literal_eval(gen),
                        ast.literal_eval(png),
                        ast.literal_eval(alt),
                    )
                )
            return entries
    raise AssertionError("_FIGURES constant not found in manuscript_build_figures.py")


def _declared_labels() -> set[str]:
    labels: set[str] = set()
    for md in sorted(MANUSCRIPT.glob("*.md")):
        if md.name in _LABEL_SCAN_SKIP:
            continue
        labels.update(_FIG_LABEL_RE.findall(md.read_text(encoding="utf-8")))
    return labels


def test_every_generator_is_registered() -> None:
    """Every manuscript_fig_*.py generator appears in _FIGURES."""
    registered = {gen for _, gen, _, _ in _figures_constant()}
    on_disk = {p.name for p in (REPO_ROOT / "scripts").glob("manuscript_fig_*.py")}
    assert on_disk, "no manuscript figure generators found"
    missing = on_disk - registered
    assert not missing, f"generators missing from _FIGURES: {sorted(missing)}"


def test_no_stale_generator_entries() -> None:
    """Every registered generator and expected PNG name exists on disk."""
    for _label, gen, png, _alt in _figures_constant():
        assert (REPO_ROOT / "scripts" / gen).is_file(), f"stale generator entry: {gen}"
        assert png.endswith(".png"), f"expected PNG must be .png: {png}"


def test_registered_generators_are_unique() -> None:
    """No duplicate generator registrations."""
    gens = [gen for _, gen, _, _ in _figures_constant()]
    assert len(gens) == len(set(gens))


def test_table_covers_exactly_the_labels_the_manuscript_declares() -> None:
    """A figure the manuscript embeds must be built here, and vice versa."""
    registered = {label for label, _, _, _ in _figures_constant()}
    declared = _declared_labels()
    assert declared, "no {#fig:...} labels found in manuscript sections"
    assert registered == declared, (
        f"declared-not-built: {sorted(declared - registered)}; "
        f"built-not-declared: {sorted(registered - declared)}"
    )


def test_every_figure_carries_alt_text_distinct_from_its_caption() -> None:
    """Alt text must describe the image, not repeat the caption.

    The accessibility validator only requires a non-empty ``alt_text``; a caption
    copied into that field satisfies it while telling a screen-reader user
    nothing new.
    """
    captions = "\n".join(
        md.read_text(encoding="utf-8")
        for md in sorted(MANUSCRIPT.glob("*.md"))
        if md.name not in _LABEL_SCAN_SKIP
    )
    for label, _gen, _png, alt in _figures_constant():
        assert alt.strip(), f"{label} has empty alt_text"
        assert len(alt.split()) >= 15, f"{label} alt_text is too thin to be useful"
        assert alt not in captions, f"{label} alt_text duplicates its caption"


def test_generated_registry_matches_the_table() -> None:
    """output/figures/figure_registry.json is generated, never hand-edited."""
    assert REGISTRY.is_file(), (
        "figure_registry.json missing — run python -m scripts.manuscript_build_figures"
    )
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    records = {rec["label"]: rec for rec in payload["figures"]}
    for label, gen, png, alt in _figures_constant():
        assert label in records, f"{label} missing from generated registry"
        record = records[label]
        assert record["filename"] == png
        assert record["alt_text"] == alt
        assert record["generated_by"] == f"scripts/{gen}"
        assert (REGISTRY.parent / png).is_file(), f"{png} not built"
    assert set(records) == {label for label, _, _, _ in _figures_constant()}
