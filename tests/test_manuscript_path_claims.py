"""Regression tests for the manuscript's *path* claims.

Every other manuscript check compares a **number** against the producer. A path
is not a number, which is how this defect shipped clean: a commit repointed the
``multiagent`` family's ``target_dir`` from ``input/multi_agent_models`` into
``input/gnn_files/multiagent``, and three prose sites went on describing the
pre-change layout. The counts beside them were all still correct, so
``check_manuscript_tokens.py --strict`` reported "clean" over the contradiction.

Two mechanisms close it, and these tests pin both:

* :func:`manuscript_variables.corpus_coverage_notes` generates the two sentences
  that state the family-to-directory relationship, so no manuscript file types
  it. Both branches are exercised here — the all-inside branch the repository is
  in today, and the some-outside branch it was in before.
* ``check_manuscript_tokens._path_claim_issues`` cross-checks any ``input/...``
  literal that prose does type against the manifest.

The live-repository test at the end is the one that would have failed on the
shipped PDF.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn.manuscript import (  # noqa: E402
    RepositorySnapshot,
    generate_variables,
)
from gnn.manuscript.variables import (  # noqa: E402
    _families,
    _outside_corpus_dirs,
    corpus_coverage_notes,
    outside_corpus_note,
)


def _load_gate() -> ModuleType:
    """Import the gate script by path; ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location(
        "_check_manuscript_tokens", REPO_ROOT / "scripts" / "check_manuscript_tokens.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = _load_gate()

_MANIFEST_FAMILIES = [
    {"name": "multiagent", "target_dir": "input/gnn_files/multiagent"},
    {"name": "basics", "target_dir": "input/gnn_files/basics"},
]


_FIXTURE_GIT = (
    "git",
    "-c",
    "user.email=fixture@example.invalid",
    "-c",
    "user.name=fixture",
    "-c",
    "commit.gpgsign=false",
)


def _commit_fixture(root: Path, message: str = "fixture") -> None:
    """Make ``root`` its own committed repository.

    ``RepositorySnapshot`` resolves the **enclosing** repository's commit when there is
    one, so a fixture that is only a temporary directory inherits whatever repository
    contains ``TMPDIR``. On this machine ``TMPDIR`` is ``/Volumes/GitVault/Local/tmp/...``
    — inside the vault repository — so the snapshot read the vault's HEAD, saw none of
    the fixture's files, and this test failed; on the hosted runner ``TMPDIR`` is inside
    no repository, the working-tree fallback ran, and the same test passed (measured
    2026-11-04). Committing the fixture makes the snapshot's source the fixture itself,
    on either host.
    """
    for argv in (("init", "-q"), ("add", "-A"), ("commit", "-qm", message)):
        subprocess.run(
            [*_FIXTURE_GIT, *argv],
            cwd=root,
            check=True,
            capture_output=True,
        )


# --- the generated sentences ------------------------------------------------


def test_all_inside_branch_states_full_coverage() -> None:
    note, coverage = corpus_coverage_notes(
        {"input/gnn_files/a", "input/gnn_files/b"}, [], 29
    )
    assert "none" in note
    assert "all 2 target directories" in note
    assert coverage.startswith("All 2 registered family target directories")
    assert "a single invocation reaches every registered family" in coverage


def test_some_outside_branch_names_the_directories_and_pluralizes() -> None:
    note, coverage = corpus_coverage_notes(
        {"input/gnn_files/a", "input/multi_agent_models"},
        ["input/multi_agent_models"],
        29,
    )
    assert "`input/multi_agent_models` (1 of the 2 target directories)" in note
    assert "29 models under `input/gnn_files`" in note
    # singular subject takes a singular verb
    assert "1 registered family target directory" in coverage
    assert "lies outside that tree and needs a separate run" in coverage


def test_some_outside_branch_pluralizes_for_more_than_one() -> None:
    _, coverage = corpus_coverage_notes(
        {"input/gnn_files/a", "input/x", "input/y"}, ["input/x", "input/y"], 29
    )
    assert "2 registered family target directories" in coverage
    assert "lie outside that tree and need a separate run" in coverage


def test_the_two_branches_make_opposite_claims() -> None:
    """The point of generating the sentence is that it flips with the manifest."""
    inside, _ = corpus_coverage_notes({"input/gnn_files/a"}, [], 29)
    outside, _ = corpus_coverage_notes({"input/x"}, ["input/x"], 29)
    assert inside != outside
    assert "none" in inside and "none" not in outside


def test_both_notes_are_emitted_as_tokens() -> None:
    variables = generate_variables(REPO_ROOT)
    for key in ("GNN_UNSCANNED_CORPUS_NOTE", "GNN_TARGET_DIR_COVERAGE_NOTE"):
        assert variables[key].strip(), f"{key} is empty"
        assert "{{" not in variables[key], f"{key} contains an unresolved token"


# --- the gate rule ----------------------------------------------------------


def _section(tmp_path: Path, body: str) -> list[Path]:
    path = tmp_path / "0X_section.md"
    path.write_text(body, encoding="utf-8")
    return [path]


def test_gate_flags_a_family_named_beside_the_wrong_directory(
    tmp_path: Path,
) -> None:
    sections = _section(
        tmp_path,
        "Coverage reaches every family but not the `multiagent` family, whose "
        "target directory is `input/multi_agent_models`.\n",
    )
    issues = GATE._path_claim_issues(sections, _MANIFEST_FAMILIES)
    assert any("multiagent" in i and "input/multi_agent_models" in i for i in issues)


def test_gate_accepts_a_family_named_beside_its_declared_directory(
    tmp_path: Path,
) -> None:
    sections = _section(
        tmp_path,
        "The `multiagent` family lives in `input/gnn_files/multiagent`.\n",
    )
    assert GATE._path_claim_issues(sections, _MANIFEST_FAMILIES) == []


def test_gate_accepts_an_ancestor_directory_that_covers_the_family(
    tmp_path: Path,
) -> None:
    """`input/gnn_files` legitimately *contains* the family's target_dir."""
    sections = _section(
        tmp_path,
        "Pointing --target-dir at `input/gnn_files` reaches the `multiagent` family.\n",
    )
    assert GATE._path_claim_issues(sections, _MANIFEST_FAMILIES) == []


def test_gate_ignores_a_family_name_used_as_an_ordinary_word(
    tmp_path: Path,
) -> None:
    """Without "family" nearby, `basics` is just a backticked word."""
    sections = _section(
        tmp_path,
        "The `basics` fixtures are documented under `input/gnn_files/recursive`.\n",
    )
    assert GATE._path_claim_issues(sections, _MANIFEST_FAMILIES) == []


def test_gate_does_not_sweep_in_an_unrelated_sibling_directory(
    tmp_path: Path,
) -> None:
    """Only the nearest literal is read as the family's location."""
    sections = _section(
        tmp_path,
        "The `multiagent` family lives in `input/gnn_files/multiagent`, and "
        "`input/gnn_files/recursive/` is a reserved directory that ships "
        "documentation only.\n",
    )
    assert GATE._path_claim_issues(sections, _MANIFEST_FAMILIES) == []


def test_gate_flags_an_input_path_that_does_not_exist(tmp_path: Path) -> None:
    sections = _section(tmp_path, "See `input/nonexistent_models/` for details.\n")
    issues = GATE._path_claim_issues(sections, _MANIFEST_FAMILIES)
    assert issues == [
        "0X_section.md: `input/nonexistent_models` does not exist in the repository"
    ]


def test_gate_ignores_paths_inside_fenced_commands(tmp_path: Path) -> None:
    """A path in a reproduction command is an argument, not a location claim."""
    sections = _section(
        tmp_path,
        "Run the gate:\n\n```bash\npython run.py --target-dir input/no_such_dir\n```\n",
    )
    assert GATE._path_claim_issues(sections, _MANIFEST_FAMILIES) == []


# --- the live repository ----------------------------------------------------


def test_live_manuscript_has_no_contradicted_path_claims() -> None:
    """The assertion that would have failed on the shipped PDF."""
    manuscript_dir = REPO_ROOT / "manuscript"
    sections = GATE._section_files(manuscript_dir)
    assert sections, "no manuscript sections found"
    families = _families(RepositorySnapshot(REPO_ROOT))
    assert families, "manifest declares no families"
    assert GATE._path_claim_issues(sections, families) == []


def test_live_manuscript_types_no_family_target_directory_by_hand() -> None:
    """Every declared target_dir must reach prose through the manifest.

    A section may still name ``input/gnn_files`` (the ancestor the reproduction
    command takes); what it may not do is hand-type a *family's own* directory,
    because that is the literal that silently staled.
    """
    families = _families(RepositorySnapshot(REPO_ROOT))
    declared = {str(f["target_dir"]).rstrip("/") for f in families}
    offenders: list[str] = []
    for path in GATE._section_files(REPO_ROOT / "manuscript"):
        raw = path.read_text(encoding="utf-8")
        for _, content in GATE._code_spans(raw):
            if content.rstrip("/") in declared:
                offenders.append(f"{path.name}: `{content}`")
    assert offenders == [], (
        "family target directories are typed into prose instead of arriving "
        f"through the manifest: {offenders}"
    )


# --- model files that live outside input/gnn_files --------------------------
#
# GNN_EXAMPLE_COUNT counts ``## GNNSection``-bearing files under
# ``input/gnn_files`` only. A remediation pass gave
# ``input/multi_agent_models/multi_agent_coordination.md`` the ``## GNNSection``
# header the manuscript calls Required, which made it a model file that no
# count, table, figure or gate could see; the only thing that described it was a
# typed sentence in S01. These tests pin the generated replacement.


def test_outside_corpus_note_names_each_directory_and_its_model_count() -> None:
    note = outside_corpus_note(
        [("input/multi_agent_models", 1), ("input/recursive_models", 0)],
        {"input/gnn_files/multiagent"},
        29,
    )
    assert "`input/multi_agent_models/` holds 1 model file" in note
    assert "`input/recursive_models/` holds no model files" in note
    assert "registered by no manifest family" in note
    assert "That model file is outside the 29-file count above." in note


def test_outside_corpus_note_flips_when_a_family_claims_the_directory() -> None:
    """Registering the fixture must change the sentence, not just a number."""
    note = outside_corpus_note(
        [("input/multi_agent_models", 1)], {"input/multi_agent_models"}, 29
    )
    assert "is registered as a manifest family target directory" in note
    assert "registered by no manifest family" not in note


def test_outside_corpus_note_flips_when_a_model_is_added() -> None:
    single = outside_corpus_note([("input/multi_agent_models", 1)], set(), 29)
    double = outside_corpus_note([("input/multi_agent_models", 2)], set(), 29)
    assert "holds 1 model file" in single
    assert "holds 2 model files" in double
    assert "Those 2 model files are outside the 29-file count above." in double


def test_outside_corpus_note_states_full_coverage_when_nothing_is_outside() -> None:
    assert outside_corpus_note([], set(), 29) == (
        "Every model file under `input/` lives in that subtree, so the "
        "29-file count covers the whole tree."
    )
    docs_only = outside_corpus_note([("input/recursive_models", 0)], set(), 29)
    assert "No model file lies outside `input/gnn_files`" in docs_only


def test_live_outside_corpus_dirs_are_empty_after_corpus_closure() -> None:
    """The 3.3.0 corpus closure folded the two legacy dirs into gnn_files."""
    pairs = dict(_outside_corpus_dirs(RepositorySnapshot(REPO_ROOT)))
    assert pairs == {}, pairs


def test_live_s01_states_the_outside_corpus_relationship_through_the_token() -> None:
    """S01 must not re-type the sentence the producer now owns."""
    raw = (REPO_ROOT / "manuscript" / "S01_source_surface.md").read_text(
        encoding="utf-8"
    )
    assert "{{GNN_OUTSIDE_CORPUS_NOTE}}" in raw
    assert "multi_agent_models" not in raw
    assert "recursive_models" not in raw


def test_live_variables_carry_the_outside_corpus_tokens() -> None:
    variables = generate_variables(REPO_ROOT)
    assert variables["GNN_OUTSIDE_CORPUS_MODEL_COUNT"] == "0"
    note = variables["GNN_OUTSIDE_CORPUS_NOTE"]
    assert "Every model file under `input/` lives in that subtree" in note
    assert (
        f"so the {variables['GNN_EXAMPLE_COUNT']}-file count covers the whole tree"
        in note
    )


def test_producer_model_census_matches_pipeline_discovery() -> None:
    """The counts the manuscript prints must be the files the pipeline sees.

    ``_example_models`` recognizes a model by its ``## GNNSection`` header;
    ``src/main.py`` discovers one with ``gnn.processing.discovery.is_model_source_path``.
    Two independent predicates over the same tree is how a corpus count can be
    true of the producer and false of the pipeline, so this pins them equal.
    """
    from gnn.processing.discovery import is_model_source_path  # noqa: PLC0415

    variables = generate_variables(REPO_ROOT)
    discovered = {
        root: sum(
            1 for md in (REPO_ROOT / root).rglob("*.md") if is_model_source_path(md)
        )
        for root in (
            "input/gnn_files",
            "input/multi_agent_models",
            "input/recursive_models",
        )
    }
    assert discovered["input/gnn_files"] == int(variables["GNN_EXAMPLE_COUNT"])
    outside = (
        discovered["input/multi_agent_models"] + discovered["input/recursive_models"]
    )
    assert outside == int(variables["GNN_OUTSIDE_CORPUS_MODEL_COUNT"])


def test_a_doc_that_only_names_the_header_is_not_counted_as_a_model(
    tmp_path: Path,
) -> None:
    """Prose about ``## GNNSection`` must not make a README a model file.

    Caught live: rewriting ``input/multi_agent_models/README.md`` to explain
    which headers the fixture carries took the directory's model count from 1
    to 2 on the very next run.
    """
    corpus = tmp_path / "input" / "gnn_files" / "basics"
    corpus.mkdir(parents=True)
    (corpus / "model.md").write_text("# M\n\n## GNNSection\nM\n", encoding="utf-8")
    fixtures = tmp_path / "input" / "fixtures"
    fixtures.mkdir(parents=True)
    (fixtures / "README.md").write_text(
        "It carries the `## GNNSection` header the reference marks Required.\n",
        encoding="utf-8",
    )
    (fixtures / "real_model.md").write_text("## GNNSection\nReal\n", encoding="utf-8")
    _commit_fixture(tmp_path)
    snapshot = RepositorySnapshot(tmp_path)
    assert dict(_outside_corpus_dirs(snapshot)) == {"input/fixtures": 1}

    # The snapshot's source is the fixture's own commit, not the disk and not an
    # enclosing repository: with the model rewritten so it is no longer a model on
    # disk, the committed view must still count it. Without this direction the
    # assertion above would also pass if the source were ignored entirely.
    (fixtures / "real_model.md").write_text("Prose only now.\n", encoding="utf-8")
    assert dict(_outside_corpus_dirs(RepositorySnapshot(tmp_path))) == {
        "input/fixtures": 1
    }


if __name__ == "__main__":  # pragma: no cover - convenience
    raise SystemExit(pytest.main([__file__, "-q"]))
