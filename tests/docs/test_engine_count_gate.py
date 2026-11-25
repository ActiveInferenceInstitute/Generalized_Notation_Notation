"""Contract tests for the render-engine count gate.

The gate used to exempt any count preceded by "other", and it never matched a
qualifier at all, so "the other nine categorical backends" was invisible twice over.
These tests pin the replacement rules: qualifiers are visible, declared qualifiers are
judged against a population derived from the registry, ``other N`` is judged as N
peers plus the subject, and an undeclared qualifier is refused rather than guessed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "docs" / "development"))

import docs_audit as audit  # noqa: E402  (script, not a package)


def _issues(tmp_path: Path, monkeypatch, line: str) -> list[tuple]:
    # The registry is parsed from REPO_ROOT, which is relocated to the synthetic
    # tree here, so pin the two derived values to the real registry and test the
    # judgement logic against them. Population derivation has its own test.
    populations = audit._engine_count_populations()
    registry_count = audit._framework_registry_count()
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(audit, "_framework_registry_count", lambda: registry_count)
    monkeypatch.setattr(audit, "_engine_count_populations", lambda: populations)
    doc = tmp_path / "docs" / "claim.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(line + "\n")
    return audit.audit_engine_count_claims([doc])


def test_populations_are_derived_from_the_registry() -> None:
    populations = audit._engine_count_populations()

    assert populations["total"] == audit._framework_registry_count()
    assert populations["total"] == (
        populations["non_experimental"] + populations["experimental"]
    )
    # The frames that refuse continuous models in the default set.
    assert 0 < populations["categorical_only"] <= populations["non_experimental"]
    assert populations["categorical"] >= populations["categorical_only"]


@pytest.mark.parametrize(
    "line",
    [
        "`framework_registry.py` declares the twelve frameworks in total.",
        "The registry is the single declaration of twelve frameworks.",
    ],
)
def test_bound_total_matching_the_registry_passes(tmp_path, monkeypatch, line) -> None:
    assert _issues(tmp_path, monkeypatch, line) == []


def test_bound_total_contradicting_the_registry_fails(tmp_path, monkeypatch) -> None:
    issues = _issues(
        tmp_path, monkeypatch, "`framework_registry.py` declares eleven frameworks."
    )
    assert len(issues) == 1
    assert "11" in issues[0][3] or "eleven" in issues[0][3]


def test_unbound_count_is_not_a_claim(tmp_path, monkeypatch) -> None:
    # No binding phrase: a subset or hypothetical count, not an inventory claim.
    assert _issues(tmp_path, monkeypatch, "We would use nine frameworks here.") == []


def test_other_n_counts_peers_plus_the_subject(tmp_path, monkeypatch) -> None:
    # Ten categorical frames, so the peers of one of them number nine.
    assert (
        _issues(
            tmp_path,
            monkeypatch,
            "It is registered in the registry alongside the other nine categorical "
            "backends.",
        )
        == []
    )


def test_other_n_with_the_wrong_peer_count_fails(tmp_path, monkeypatch) -> None:
    issues = _issues(
        tmp_path,
        monkeypatch,
        "It is registered in the registry alongside the other eight categorical "
        "backends.",
    )
    assert len(issues) == 1
    assert "other eight" in issues[0][3]


def test_categorical_only_qualifier_is_judged_against_its_own_population(
    tmp_path, monkeypatch
) -> None:
    assert (
        _issues(
            tmp_path,
            monkeypatch,
            "The four categorical-only backends are registered in the registry.",
        )
        == []
    )
    issues = _issues(
        tmp_path,
        monkeypatch,
        "The five categorical-only backends are registered in the registry.",
    )
    assert len(issues) == 1


def test_step_idiom_stays_exempt(tmp_path, monkeypatch) -> None:
    # "Step 11" is a step number, not a count of frameworks.
    assert (
        _issues(tmp_path, monkeypatch, "Step 11 covers nine backends in the registry.")
        == []
    )


def test_render_backends_phrasing_is_visible(tmp_path, monkeypatch) -> None:
    # "ten render backends" used to slip past the noun pattern entirely.
    issues = _issues(
        tmp_path,
        monkeypatch,
        "GNN compiles down to the ten render backends registered in the registry.",
    )
    assert len(issues) == 1, "the 'render backends' phrasing must not be invisible"


def test_undeclared_qualifier_is_refused(tmp_path, monkeypatch) -> None:
    issues = _issues(
        tmp_path,
        monkeypatch,
        "The nine made-up-kind backends are registered in the registry.",
    )
    assert len(issues) == 1
    assert "no derivable population" in issues[0][3]
