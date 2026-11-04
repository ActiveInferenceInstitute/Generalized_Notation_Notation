"""The headless manuscript-injection mirror, pinned to a recorded template copy.

``gnn.manuscript.substitution`` prefers the template injector when it is
importable and falls back to a local mirror when it is not — and CI is the
environment with no template checkout, so the mirror is the implementation every
push gate measures. That is also the path nothing else can contradict, so the
mirror is pinned to a **recorded artifact** rather than to a live differential
test that would skip exactly where it matters.

Golden fixture: ``tests/fixtures/manuscript_hydration_9ce91f62a/``, generated
2026-11-04 by ``write_resolved_manuscript_tree`` from ``docxology/template`` at
``9ce91f62af3c6dbd8c031b6f80e2253f6c59f964`` — the revision the 2026-11-04
custody chain was recorded against — over its own ``project/manuscript/`` input.
A future template move that changes the destination copy gets a new fixture
directory, so this one keeps measuring the behaviour it recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gnn.manuscript.render_custody as render_custody
import gnn.manuscript.substitution as substitution

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "manuscript_hydration_9ce91f62a"
)


def _mirrored_copy() -> str:
    """The destination copy the fallback mirror produces for the fixture input."""
    tokens = json.loads((_FIXTURE / "tokens.json").read_text(encoding="utf-8"))
    source = (_FIXTURE / "project" / "manuscript" / "00_fixture.md").read_text(
        encoding="utf-8"
    )
    return substitution.rebase_output_links_for_destination(
        substitution.hydrate_text(source, tokens)
    )


def test_the_mirror_reproduces_the_recorded_template_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback's copy equals the template's own write, byte for byte.

    The mirror is forced (``_INJECTOR`` absent) so the assertion measures the
    implementation the CI gates actually run, not the template it prefers when
    one happens to be importable.
    """
    monkeypatch.setattr(substitution, "_INJECTOR", None)
    expected_path = _FIXTURE / "expected" / "output" / "manuscript" / "00_fixture.md"
    assert _mirrored_copy() == expected_path.read_text(encoding="utf-8")


def test_the_mirror_rebases_only_output_relative_targets() -> None:
    """Targets into ``output/`` are rebased; every other relative form is left alone."""
    text = (
        "[a](../output/data/x.json) "
        "[b](../elsewhere/x.md) "
        "![c](../output/figures/c.png){#fig:c} "
        '<img src="../output/figures/i.png">'
    )
    assert substitution._fallback_rebase_output_links(text) == (
        "[a](../data/x.json) "
        "[b](../elsewhere/x.md) "
        "![c](../figures/c.png){#fig:c} "
        '<img src="../figures/i.png">'
    )


def test_the_hydration_expectation_applies_the_destination_rebase(
    tmp_path: Path,
) -> None:
    """The PR-time guard compares the copy the renderer reads, not the source form."""
    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript" / "00_fixture.md").write_text(
        "{{T}} and [x](../output/data/y.json)\n", encoding="utf-8"
    )
    tree = render_custody._expected_hydrated_tree(tmp_path, {"T": "value"}, frozenset())
    assert tree["output/manuscript/00_fixture.md"] == b"value and [x](../data/y.json)\n"
