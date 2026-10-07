"""The LaTeX log's ``Infinite glue shrinkage`` warnings, counted and bounded.

Four of these warnings ship in every render of this manuscript, and they have
been re-reported as an open defect twice. This module ends that: it pins the
*count* (against a probe that does not miss wrapped messages) and the *bound*
that explains them (at most one per paragraph-column ``longtable``).

The finding itself, with the experiments that established it, is written up in
``manuscript/AGENTS.md`` under "Known benign LaTeX diagnostics".

Both artifacts these tests read are tracked, so they are asserted present rather
than skipped over; ``.gitignore`` carries a named exception for the log.

*Which render* they came from is pinned separately, by the render custody
manifest (``output/data/manuscript_render_manifest.json``): the committed
``.log``/``.tex``/``.md``, the hydrated sections under ``output/manuscript/``,
and the token map must be artifacts of one render invocation, not three
renders passing together. ``record_render_manifest`` writes that manifest as
the last step of the SC-22 re-render ritual; ``custody_issues`` (shared with
``scripts/z_record_manuscript_render_manifest.py``) checks the whole chain
in the tests at the bottom of this module.

Two facts these tests exist to protect:

* TeX breaks its own log lines and will split the message mid-word, so
  ``grep -c 'Infinite glue'`` under-counts. On the log at the time of writing it
  returned 3 against a true count of 4. Every probe here joins lines first.
* ``\\LT@start`` (``longtable.sty``) runs once per ``longtable``, and its
  ``\\vsplit`` is the only ``\\vsplit`` reachable from ``\\endlongtable``, so one
  table can contribute at most one message. A count above the number of
  paragraph-column tables means something new is emitting them and the write-up
  no longer explains the log.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn.manuscript.render_custody import (  # noqa: E402
    custody_issues,
    hydration_issues,
    mask_log_timestamp,
    record_render_manifest,
    verify_fresh_render,
)
from gnn.manuscript.substitution import active_preamble  # noqa: E402

LOG_PATH = REPO_ROOT / "output" / "pdf" / "_combined_manuscript.log"
TEX_PATH = REPO_ROOT / "output" / "pdf" / "_combined_manuscript.tex"
PREAMBLE_PATH = REPO_ROOT / "manuscript" / "preamble.md"

# The declarations docxology/template's ``_pdf_combined_preamble.py`` checks
# for, with its own patterns: a preamble missing one gets a fallback
# injected at render time, so the PDF renders anyway while the committed
# preamble no longer describes it.
REQUIRED_PREAMBLE_DECLARATIONS = {
    r"\usepackage{listings}": re.compile(r"\\usepackage(?:\[[^\]]*\])?\{listings\}"),
    r"\newtheorem{theorem}": re.compile(r"\\newtheorem\*?\s*\{theorem\}"),
    r"\newtheorem{remark}": re.compile(r"\\newtheorem\*?\s*\{remark\}"),
    r"\newtheorem{example}": re.compile(r"\\newtheorem\*?\s*\{example\}"),
}

MESSAGE = "Infinite glue shrinkage found in box being split"

# A `longtable` preamble runs from \begin{longtable} to whichever of \caption or
# \toprule opens the table body; a `p{...}` there is a paragraph column.
_LONGTABLE_PREAMBLE_RE = re.compile(
    r"\\begin\{longtable\}(.*?)(?:\\caption|\\toprule)", re.DOTALL
)


def count_message(log_text: str) -> int:
    """Occurrences of *MESSAGE*, counting ones TeX wrapped across a line break."""
    return log_text.replace("\n", "").count(MESSAGE)


def count_paragraph_column_longtables(tex_text: str) -> int:
    """``longtable`` environments whose column preamble declares a ``p{...}``."""
    return sum(
        1 for preamble in _LONGTABLE_PREAMBLE_RE.findall(tex_text) if "p{" in preamble
    )


# --- the counting probe -----------------------------------------------------


def test_counting_survives_a_message_tex_wrapped_mid_word() -> None:
    """Verbatim from ``_combined_manuscript.log``: TeX split this one in two."""
    wrapped = "ignored: In\nfinite glue shrinkage found in box being split [25]\n"
    assert MESSAGE not in wrapped, "the naive scan must miss this line"
    assert count_message(wrapped) == 1


def test_counting_finds_two_messages_on_one_line() -> None:
    line = f"ignored: {MESSAGE} [7] [8]\nignored: {MESSAGE} [14]\n"
    assert count_message(line) == 2


def test_counting_reports_zero_on_a_clean_log() -> None:
    assert count_message("[1] [2] [3]\nOutput written on x.pdf (3 pages).\n") == 0


# --- the column-spec probe --------------------------------------------------


def test_paragraph_column_longtables_are_told_apart_from_plain_ones() -> None:
    tex = (
        "\\begin{longtable}[]{@{}lll@{}}\n\\caption{plain}\n\\end{longtable}\n"
        "\\begin{longtable}[]{@{}\n"
        "  >{\\raggedright\\arraybackslash}p{(\\linewidth) * \\real{0.5}}@{}}\n"
        "\\caption{wide}\n\\end{longtable}\n"
    )
    assert count_paragraph_column_longtables(tex) == 1


# --- the shipped artifact ---------------------------------------------------


def _shipped() -> tuple[str, str]:
    """The committed render artifacts, or an explicit failure.

    No skip guard: both files are tracked (``.gitignore`` carries an explicit
    exception for the log), so their absence means the committed render is
    incomplete, and the repo's zero-skip contract
    (``tests/test_zero_skip_contracts.py``) forbids hiding that behind a
    skip. A skip here also silently disarmed the only check on the shipped
    LaTeX diagnostics.
    """
    missing = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in (LOG_PATH, TEX_PATH)
        if not path.exists()
    ]
    assert not missing, (
        f"committed render artifacts missing: {missing}. These are tracked "
        "files; regenerate them with the template's stage_03_render and commit "
        "the result."
    )
    return (
        LOG_PATH.read_text(encoding="utf-8", errors="replace"),
        TEX_PATH.read_text(encoding="utf-8", errors="replace"),
    )


def test_every_infinite_glue_warning_is_accounted_for_by_a_wide_table() -> None:
    """At most one message per paragraph-column ``longtable``.

    ``\\LT@start`` is reached once per table, so this bound is structural, not a
    tally. If it ever fails, a *new* emitter has appeared and the AGENTS.md
    write-up has stopped explaining the log.
    """
    log_text, tex_text = _shipped()
    seen = count_message(log_text)
    wide = count_paragraph_column_longtables(tex_text)
    assert seen <= wide, (
        f"{seen} '{MESSAGE}' messages against {wide} paragraph-column "
        "longtables; something other than longtable's \\LT@start is emitting them"
    )


def test_a_manuscript_with_no_wide_table_would_carry_no_such_warning() -> None:
    """The bound has teeth: zero wide tables must mean zero messages."""
    log_text, tex_text = _shipped()
    if count_paragraph_column_longtables(tex_text) == 0:
        assert count_message(log_text) == 0


def test_the_log_carries_no_overfull_boxes() -> None:
    """Corroborates that nothing is clipped or run into the margin."""
    log_text, _ = _shipped()
    assert log_text.replace("\n", "").count("Overfull \\hbox") == 0
    assert log_text.replace("\n", "").count("Overfull \\vbox") == 0


def test_the_log_carries_no_personal_machine_paths() -> None:
    """Home-directory prefixes from the render host must never ship."""

    log_text, _ = _shipped()
    assert "/Users/" not in log_text
    assert "/home/" not in log_text


def test_the_preamble_declares_what_the_template_requires() -> None:
    """``listings`` and the three theorem environments are declared in-repo."""
    preamble = active_preamble(PREAMBLE_PATH.read_text(encoding="utf-8"))
    missing = [
        name
        for name, pattern in REQUIRED_PREAMBLE_DECLARATIONS.items()
        if pattern.search(preamble) is None
    ]
    assert not missing, f"manuscript/preamble.md lacks {missing}"


def test_the_preamble_patterns_reject_a_preamble_without_them() -> None:
    """The patterns have teeth: a commented-out or renamed declaration is absent."""
    stripped = active_preamble(
        "prose \\usepackage{listings}\n```latex\n% \\usepackage{listings}\n% \\newtheorem{theorem}{Theorem}\n\\usepackage{xcolor}\n\\newtheorem{lemma}{Lemma}\n```\n"
    )
    assert all(
        pattern.search(stripped) is None
        for pattern in REQUIRED_PREAMBLE_DECLARATIONS.values()
    )


# --- the render custody chain ------------------------------------------------


def _custody_fixture(tmp_path: Path) -> Path:
    """A miniature committed tree: token map, receipt, hydrated prose, artifacts."""
    root = tmp_path / "repo"
    (root / "output" / "data").mkdir(parents=True)
    (root / "output" / "manuscript").mkdir(parents=True)
    (root / "output" / "pdf").mkdir(parents=True)
    (root / "output" / "data" / "manuscript_variables.json").write_text(
        json.dumps({"GNN_GIT_COMMIT": "abc1234", "GNN_STEP_COUNT": "25"}),
        encoding="utf-8",
    )
    (root / "output" / "data" / "manuscript_variables_receipt.json").write_text(
        json.dumps({"counts_describe_commit": "abc1234"}), encoding="utf-8"
    )
    (root / "output" / "manuscript" / "05_reproducibility.md").write_text(
        "a 25-step pipeline\n", encoding="utf-8"
    )
    (root / "output" / "pdf" / "_combined_manuscript.md").write_text(
        "a 25-step pipeline\n", encoding="utf-8"
    )
    (root / "output" / "pdf" / "_combined_manuscript.tex").write_text(
        "a 25-step pipeline\n", encoding="utf-8"
    )
    (root / "output" / "pdf" / "_combined_manuscript.log").write_text(
        "[1] [2] Output written on x.pdf (2 pages).\n", encoding="utf-8"
    )
    record_render_manifest(root)
    return root


def test_the_committed_pdf_evidence_is_the_render_the_manifest_records() -> None:
    """Log, tex, md, hydrated prose, token map and receipt: one render's chain."""
    assert custody_issues(REPO_ROOT) == []


def test_a_missing_custody_manifest_is_a_failure_not_a_skip() -> None:
    """A checkout without the manifest cannot pass silently."""
    issues = custody_issues(Path("/nonexistent-checkout"))
    assert issues and "z_record_manuscript_render_manifest" in issues[0]


def test_a_log_swapped_after_the_record_fails(tmp_path: Path) -> None:
    """Re-rendering and committing a new log without re-recording fails."""
    root = _custody_fixture(tmp_path)
    (root / "output" / "pdf" / "_combined_manuscript.log").write_text(
        "[1] [2] [3] Output written on x.pdf (3 pages).\n", encoding="utf-8"
    )
    issues = custody_issues(root)
    assert any("_combined_manuscript.log" in issue for issue in issues), issues


def test_prose_edited_after_the_render_fails(tmp_path: Path) -> None:
    """A hydrated section the committed PDF never saw fails the chain."""
    root = _custody_fixture(tmp_path)
    (root / "output" / "manuscript" / "05_reproducibility.md").write_text(
        "a 26-step pipeline\n", encoding="utf-8"
    )
    issues = custody_issues(root)
    assert any("05_reproducibility.md" in issue for issue in issues), issues


def test_a_token_map_regenerated_after_the_render_fails(tmp_path: Path) -> None:
    """The render is pinned to the token map the strict gate pins to HEAD."""
    root = _custody_fixture(tmp_path)
    (root / "output" / "data" / "manuscript_variables.json").write_text(
        json.dumps({"GNN_GIT_COMMIT": "def5678", "GNN_STEP_COUNT": "26"}),
        encoding="utf-8",
    )
    issues = custody_issues(root)
    assert any("token map" in issue for issue in issues), issues


def test_a_fresh_render_matching_the_manifest_has_no_issues(tmp_path: Path) -> None:
    """Bytes straight from the fixture's render agree with the committed manifest."""
    root = _custody_fixture(tmp_path)
    assert verify_fresh_render(root) == []


def test_an_artifact_swap_with_unchanged_prose_warns(tmp_path: Path) -> None:
    """Artifact-only drift with the recorded inputs fully matching: toolchain."""
    root = _custody_fixture(tmp_path)
    (root / "output" / "pdf" / "_combined_manuscript.log").write_text(
        "[1] [2] [3] Output written on x.pdf (3 pages).\n", encoding="utf-8"
    )
    issues = verify_fresh_render(root)
    assert len(issues) == 1, issues
    assert issues[0].startswith("[WARN] output/pdf/_combined_manuscript.log"), issues


def test_a_re_render_at_a_new_commit_with_identical_prose_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-rendering identical prose at a new commit is not custody drift.

    Regression for the hosted false-[FAIL] (custody runs 34853088569 and
    34862542718): the ritual records the manifest one commit before the
    output commit exists, so CI's fresh render stamps the same prose with
    ITS head while the committed chain carries the parent's stamp. Both
    sides must mask both stamps — artifacts and inputs alike — or every
    push after a ritual reads as joint drift.
    """
    root = tmp_path / "repo"
    (root / "output" / "data").mkdir(parents=True)
    (root / "output" / "manuscript").mkdir(parents=True)
    (root / "output" / "pdf").mkdir(parents=True)
    (root / "output" / "data" / "manuscript_variables.json").write_text(
        json.dumps({"GNN_GIT_COMMIT": "abc1234", "GNN_STEP_COUNT": "25"}),
        encoding="utf-8",
    )
    (root / "output" / "data" / "manuscript_variables_receipt.json").write_text(
        json.dumps({"counts_describe_commit": "abc1234"}), encoding="utf-8"
    )
    parent = {
        "output/manuscript/05_reproducibility.md": b"rendered at abc1234\n",
        "output/pdf/_combined_manuscript.md": b"rendered at abc1234\n",
        "output/pdf/_combined_manuscript.tex": b"rendered at abc1234\n",
        "output/pdf/_combined_manuscript.log": b"[1] written at abc1234.\n",
    }
    for rel, data in parent.items():
        (root / rel).write_bytes(data)
    record_render_manifest(root)

    child = {
        "output/manuscript/05_reproducibility.md": b"rendered at bbb2222\n",
        "output/pdf/_combined_manuscript.md": b"rendered at bbb2222\n",
        "output/pdf/_combined_manuscript.tex": b"rendered at bbb2222\n",
        "output/pdf/_combined_manuscript.log": b"[1] written at bbb2222.\n",
    }
    for rel, data in child.items():
        (root / rel).write_bytes(data)
    monkeypatch.setattr(
        "gnn.manuscript.render_custody._git_show",
        lambda _root, rel: parent[rel],
    )
    monkeypatch.setattr(
        "gnn.manuscript.render_custody._head_stamp",
        lambda _root: "bbb2222cafebabe0000000000000000000000000",
    )
    assert verify_fresh_render(root) == []


def test_an_artifact_swap_alongside_prose_drift_fails(tmp_path: Path) -> None:
    """Artifact and input drift together: the committed chain is stale for HEAD."""
    root = _custody_fixture(tmp_path)
    (root / "output" / "manuscript" / "05_reproducibility.md").write_text(
        "a 26-step pipeline\n", encoding="utf-8"
    )
    (root / "output" / "pdf" / "_combined_manuscript.md").write_text(
        "a 26-step pipeline\n", encoding="utf-8"
    )
    issues = verify_fresh_render(root)
    assert len(issues) == 1, issues
    assert issues[0].startswith("[FAIL] output/pdf/_combined_manuscript.md"), issues


def test_a_missing_artifact_fails(tmp_path: Path) -> None:
    """A render that did not produce a committed artifact cannot be certified."""
    root = _custody_fixture(tmp_path)
    (root / "output" / "pdf" / "_combined_manuscript.log").unlink()
    issues = verify_fresh_render(root)
    assert len(issues) == 1, issues
    assert issues[0].startswith("[FAIL] output/pdf/_combined_manuscript.log"), issues


_XETEX_BANNER = (
    "This is XeTeX, Version 3.141592653-2.6-0.999998 (TeX Live 2026) "
    "(preloaded format=xelatex 2026.9.18)  {stamp}\n"
    "entering extended mode\n[1] [2] Output written on x.pdf (2 pages).\n"
)


def test_mask_log_timestamp_masks_only_the_banner_start_time() -> None:
    """The run start time goes; the engine version and format date stay."""
    masked = mask_log_timestamp(_XETEX_BANNER.format(stamp="28 SEP 2026 21:42"))
    assert "28 SEP 2026 21:42" not in masked
    assert "(preloaded format=xelatex 2026.9.18)  <timestamp>" in masked
    assert masked == mask_log_timestamp(_XETEX_BANNER.format(stamp="3 OCT 2026 07:05"))
    other_engine = _XETEX_BANNER.replace("TeX Live 2026", "TeX Live 2027")
    assert mask_log_timestamp(other_engine.format(stamp="3 OCT 2026 07:05")) != masked


def test_a_re_render_differing_only_in_the_log_timestamp_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for the PR #225 review: an identical render is not a [WARN].

    The committed log and a fresh one differ only in XeTeX's banner start
    time; ``verify_fresh_render`` must mask it on both sides.
    """
    root = _custody_fixture(tmp_path)
    log_rel = "output/pdf/_combined_manuscript.log"
    committed = _XETEX_BANNER.format(stamp="28 SEP 2026 21:42").encode("utf-8")
    (root / log_rel).write_bytes(committed)
    record_render_manifest(root)
    committed_tree = {
        rel: (root / rel).read_bytes()
        for rel in (
            log_rel,
            "output/manuscript/05_reproducibility.md",
            "output/pdf/_combined_manuscript.md",
            "output/pdf/_combined_manuscript.tex",
        )
    }
    monkeypatch.setattr(
        "gnn.manuscript.render_custody._git_show",
        lambda _root, rel: committed_tree[rel],
    )
    (root / log_rel).write_text(
        _XETEX_BANNER.format(stamp="29 SEP 2026 07:14"), encoding="utf-8"
    )
    assert verify_fresh_render(root) == []

    (root / log_rel).write_text(
        _XETEX_BANNER.format(stamp="29 SEP 2026 07:14").replace("[2] ", ""),
        encoding="utf-8",
    )
    issues = verify_fresh_render(root)
    assert len(issues) == 1 and issues[0].startswith(f"[WARN] {log_rel}"), issues


# --- the PR-time hydration guard ----------------------------------------------


def _hydration_fixture(tmp_path: Path) -> Path:
    """Sources, a token map, and the tree the injector writes from them."""
    root = tmp_path / "repo"
    (root / "manuscript").mkdir(parents=True)
    (root / "output" / "data").mkdir(parents=True)
    (root / "output" / "manuscript").mkdir(parents=True)
    (root / "manuscript" / "05_reproducibility.md").write_text(
        "a {{GNN_STEP_COUNT}}-step pipeline at {{GNN_GIT_COMMIT}}\n",
        encoding="utf-8",
    )
    (root / "manuscript" / "AGENTS.md").write_text(
        "example: {{GNN_STEP_COUNT}}\n", encoding="utf-8"
    )
    (root / "manuscript" / "preamble.md").write_text(
        "\\usepackage{listings} {{NOT_SUBSTITUTED}}\n", encoding="utf-8"
    )
    (root / "manuscript" / "references.bib").write_text(
        "@misc{k, title={t}}\n", encoding="utf-8"
    )
    (root / "output" / "data" / "manuscript_variables.json").write_text(
        json.dumps({"GNN_GIT_COMMIT": "abc1234", "GNN_STEP_COUNT": "25"}),
        encoding="utf-8",
    )
    out = root / "output" / "manuscript"
    (out / "05_reproducibility.md").write_text(
        "a 25-step pipeline at abc1234\n", encoding="utf-8"
    )
    for name in ("preamble.md", "references.bib"):
        (out / name).write_bytes((root / "manuscript" / name).read_bytes())
    pdf = root / "output/pdf"
    pdf.mkdir()
    for suffix in ("md", "tex"):
        (pdf / f"_combined_manuscript.{suffix}").write_text(
            "rendered at abc1234\n", encoding="utf-8"
        )
    return root


_EXCLUDED = frozenset({"AGENTS.md", "README.md", "SYNTAX.md"})


def test_hydrated_prose_matching_the_token_map_passes(tmp_path: Path) -> None:
    """The injector's own output, from the committed map, is not drift."""
    assert hydration_issues(_hydration_fixture(tmp_path), _EXCLUDED) == []


def test_a_token_map_regenerated_without_rehydrating_fails(tmp_path: Path) -> None:
    """The drift a count-changing PR leaves behind when the ritual is skipped.

    The map moved (a new count at a new commit) and the manifest could be
    re-recorded over it, but the prose still carries the old values; the
    scheduled custody re-render would go red after merge.
    """
    root = _hydration_fixture(tmp_path)
    (root / "output" / "data" / "manuscript_variables.json").write_text(
        json.dumps({"GNN_GIT_COMMIT": "def5678", "GNN_STEP_COUNT": "26"}),
        encoding="utf-8",
    )
    issues = hydration_issues(root, _EXCLUDED)
    assert len(issues) == 3, issues
    assert "output/manuscript/05_reproducibility.md" in issues[0], issues
    assert "def5678" in issues[0], issues


def test_a_commit_only_regeneration_fails_too(tmp_path: Path) -> None:
    """Same counts, new commit: the cron's fresh comparison fails on the stamp."""
    root = _hydration_fixture(tmp_path)
    (root / "output" / "data" / "manuscript_variables.json").write_text(
        json.dumps({"GNN_GIT_COMMIT": "def5678", "GNN_STEP_COUNT": "25"}),
        encoding="utf-8",
    )
    assert len(hydration_issues(root, _EXCLUDED)) == 3


def test_a_source_edit_never_hydrated_fails(tmp_path: Path) -> None:
    """Prose edited in manuscript/ but never rendered is stale evidence."""
    root = _hydration_fixture(tmp_path)
    (root / "manuscript" / "preamble.md").write_text(
        "\\usepackage{listings}\n", encoding="utf-8"
    )
    (root / "manuscript" / "07_new.md").write_text("new\n", encoding="utf-8")
    (root / "output" / "manuscript" / "08_gone.md").write_text("x\n", encoding="utf-8")
    issues = hydration_issues(root, _EXCLUDED)
    assert [issue.split()[0] for issue in issues] == [
        "output/manuscript/07_new.md",
        "output/manuscript/preamble.md",
        "output/manuscript/08_gone.md",
    ], issues


def test_a_missing_token_map_is_a_failure_not_a_skip(tmp_path: Path) -> None:
    root = _hydration_fixture(tmp_path)
    (root / "output" / "data" / "manuscript_variables.json").unlink()
    issues = hydration_issues(root, _EXCLUDED)
    assert issues and "z_generate_manuscript_variables" in issues[0], issues


if __name__ == "__main__":  # pragma: no cover - convenience
    raise SystemExit(pytest.main([__file__, "-q"]))
