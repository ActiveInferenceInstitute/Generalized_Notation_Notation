"""Contract tests for the path-reference gate's classification.

The gate's whole value is that it classifies before it complains. If a refactor
collapses ``external-sibling`` or ``negative`` back into ``citation``, the gate goes
back to demanding edits to correct documentation — eleven documents in the
sibling-repository case alone. Each class therefore gets its own synthetic case, and
the ratchet is exercised in both directions: a new finding fails, and a recorded
finding that disappeared also fails, because a baseline that never shrinks becomes
an obsolete specification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_doc_path_references as gate  # noqa: E402  (script, not a package)


def _tree(tmp_path: Path, docs: dict[str, str], existing: tuple[str, ...] = ()) -> Path:
    for rel, text in docs.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for rel in existing:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# exists\n")
    return tmp_path


def _kinds(root: Path) -> dict[str, str]:
    findings, _, _ = gate.scan(root)
    return {f.cited: f.kind for f in findings}


def test_citation_implying_existence_is_the_finding(tmp_path: Path) -> None:
    root = _tree(
        tmp_path, {"docs/a.md": "See `src/gnn/missing_thing.py` for details.\n"}
    )
    assert _kinds(root) == {"src/gnn/missing_thing.py": "citation"}


def test_negative_retirement_record_is_accepted(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        {"docs/a.md": "`src/gnn/gone.py` no longer exists; it moved elsewhere.\n"},
    )
    assert _kinds(root) == {"src/gnn/gone.py": "negative"}


def test_labelled_illustrative_example_is_accepted(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        {
            "docs/a.md": "# (illustrative) `src/gnn/example_only.py` — path does not exist\n"
        },
    )
    assert _kinds(root) == {"src/gnn/example_only.py": "illustrative"}


def test_registered_schematic_placeholder_is_illustrative(tmp_path: Path) -> None:
    # No negative wording anywhere: the registry alone has to carry this class.
    doc = (
        "4. **Create thin orchestrator** `src/gnn/23_new_step.py`: "
        "handle orchestration only\n"
    )
    findings, _, _ = gate.scan(_tree(tmp_path, {"docs/pattern.md": doc}))
    assert [(f.cited, f.kind) for f in findings] == [
        ("src/gnn/23_new_step.py", "illustrative")
    ]


def test_sibling_repository_citation_is_external_not_a_finding(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        {
            "docs/downstream/fundamentals.md": (
                "The repository declares `src/active_inference/gnn_spec.py`.\n"
            )
        },
    )
    assert _kinds(root) == {"src/active_inference/gnn_spec.py": "external-sibling"}


def test_changelog_is_history_by_name(tmp_path: Path) -> None:
    root = _tree(
        tmp_path, {"CHANGELOG.md": "`src/gnn/render/renderer.py` split in two.\n"}
    )
    assert _kinds(root) == {"src/gnn/render/renderer.py": "history"}


def test_marker_turns_a_document_into_a_record(tmp_path: Path) -> None:
    marker = "<!-- doc-path-references: historical-record -->"
    root = _tree(
        tmp_path,
        {"docs/security/receipt.md": f"{marker}\n\n- `src/gnn/execute/processor.py`\n"},
    )
    assert _kinds(root) == {"src/gnn/execute/processor.py": "history"}


def test_existing_paths_are_never_findings(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        {"docs/a.md": "See `src/gnn/present.py`.\n"},
        existing=("src/gnn/present.py",),
    )
    findings, documents, citations = gate.scan(root)
    assert (findings, citations) == ([], 0)
    assert documents == 1


def test_all_classes_exist_in_the_schema() -> None:
    # Guards the reporting contract: no class may silently disappear.
    assert gate.CLASSES == (
        "citation",
        "negative",
        "illustrative",
        "external-sibling",
        "history",
    )
    assert gate.FAILING_CLASS == "citation"


def test_ratchet_fails_on_a_new_citation(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"docs/a.md": "See `src/gnn/fresh_breakage.py`.\n"})
    findings, documents, citations = gate.scan(root)
    baseline = gate.build_baseline([], documents, citations)

    new, vanished = gate.diff_baseline(findings, baseline)

    assert [f.cited for f in new] == ["src/gnn/fresh_breakage.py"]
    assert vanished == []


def test_ratchet_fails_when_a_recorded_citation_disappears(tmp_path: Path) -> None:
    # The document was fixed; the baseline still lists it. The gate must say so,
    # otherwise the recorded set is enforced as if it were the specification.
    recorded = {
        "entries": [
            {"doc": "docs/a.md", "line": 1, "cited": "src/gnn/now_fixed.py"},
        ]
    }
    new, vanished = gate.diff_baseline([], recorded)

    assert new == []
    assert [entry["cited"] for entry in vanished] == ["src/gnn/now_fixed.py"]


def test_baseline_carries_its_population() -> None:
    baseline = gate.build_baseline([], documents=3, citations=0)

    assert baseline["schema"] == gate.BASELINE_SCHEMA
    assert baseline["population"] == {
        "documents_scanned": 3,
        "unresolved_citations": 0,
    }
    assert set(baseline["classes"]) == set(gate.CLASSES)
    assert "measured_at_commit" in baseline and "measured_at_date" in baseline


def test_committed_baseline_parses_and_matches_its_schema() -> None:
    baseline = json.loads(gate.BASELINE_PATH.read_text())

    assert baseline["schema"] == gate.BASELINE_SCHEMA
    assert set(baseline["classes"]) == set(gate.CLASSES)
    # Provenance, not a bare list: a baseline without its population is a number
    # nobody can re-derive.
    assert baseline["measured_at_commit"] != "unknown"
    assert baseline["measured_at_date"]
    assert baseline["population"]["documents_scanned"] > 0
