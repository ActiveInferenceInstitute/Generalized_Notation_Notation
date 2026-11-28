#!/usr/bin/env python3
"""Negative tests for the manuscript integrity gate's strict checks.

Each test is the SC-3 / SC-20 / SC-23 failure mode the gate was built for,
reproduced on a fixture: a hand-typed step count, a crossref to a label
nothing declares, an unresolved token in the rendered copy, a committed token
map that has drifted from the producer, and a producer with no commit
provenance. Every one of these shipped green through every gate once; the
fixture versions must FAIL.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))


def _load_by_path(name: str, relative: str):
    """Import a repo script by path; ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = _load_by_path("gate", "scripts/check_manuscript_tokens.py")
Z_GENERATE = _load_by_path("z_generate", "scripts/z_generate_manuscript_variables.py")

# A stand-in producer output: one structure count, one step token, one big
# count. Values chosen to trip the gate's own floors.
_PRODUCER = {
    "GNN_GIT_COMMIT": "abc1234",
    "GNN_STEP_COUNT": "25",
    "GNN_STEP_3": "3",
    "GNN_TEST_FILE_COUNT": "397",
    "GNN_MANUSCRIPT_FIGURE_COUNT": "6",
}
_HARDCODE_TARGETS = {
    "25": "GNN_STEP_COUNT",
    "397": "GNN_TEST_FILE_COUNT",
}
_STEP_LITERALS = {"3": "GNN_STEP_3"}


def _section(tmp_path: Path, body: str) -> list[Path]:
    path = tmp_path / "0X_fixture_section.md"
    path.write_text(body, encoding="utf-8")
    return [path]


# --- SC-20: step phrases ------------------------------------------------------


def test_a_hard_typed_step_count_is_flagged() -> None:
    """'25 steps' must be {{GNN_STEP_COUNT}}, not a literal."""
    issues = GATE._section_hardcode_issues(
        "fixture.md", "The pipeline runs in 25 steps.", _HARDCODE_TARGETS, {}
    )
    assert any("GNN_STEP_COUNT" in issue for issue in issues), issues


def test_a_hard_typed_hyphenated_step_count_is_flagged() -> None:
    """'25-step pipeline' — the plural form with the number leading."""
    issues = GATE._section_hardcode_issues(
        "fixture.md", "a 25-step pipeline", _HARDCODE_TARGETS, {}
    )
    assert any("GNN_STEP_COUNT" in issue for issue in issues), issues


def test_a_singular_step_literal_is_still_flagged() -> None:
    """'step 3' keeps its pre-existing rule while the plural forms arrive."""
    issues = GATE._section_hardcode_issues(
        "fixture.md", "step 3 begins parsing", _HARDCODE_TARGETS, _STEP_LITERALS
    )
    assert any("GNN_STEP_3" in issue for issue in issues), issues


def test_a_step_count_spelled_as_a_token_is_not_flagged() -> None:
    issues = GATE._section_hardcode_issues(
        "fixture.md", "The pipeline runs in phases.", _HARDCODE_TARGETS, {}
    )
    assert issues == []


# --- SC-20: cross-reference labels --------------------------------------------


def test_a_crossref_to_an_undeclared_label_is_flagged(tmp_path) -> None:
    """'[@fig:typo]' renders as 'fig. ???' — the gate must catch it."""
    sections = _section(tmp_path, "See [@fig:typo] for the overview.\n")
    issues = GATE._crossref_issues(sections, _PRODUCER)
    assert any("@fig:typo" in issue for issue in issues), issues


def test_an_undeclared_table_crossref_is_flagged(tmp_path) -> None:
    sections = _section(tmp_path, "Summarized in [@tbl:no_such_table].\n")
    issues = GATE._crossref_issues(sections, _PRODUCER)
    assert any("@tbl:no_such_table" in issue for issue in issues), issues


def test_a_crossref_to_a_declared_label_passes(tmp_path) -> None:
    sections = _section(
        tmp_path, "![Alt.](x.png){#fig:ok width=80%}\n\nSee [@fig:ok].\n"
    )
    assert GATE._crossref_issues(sections, _PRODUCER) == []


def test_a_table_token_carries_its_own_label(tmp_path) -> None:
    """{{GNN_STEP_TABLE}} arrives with ``{#tbl:...}`` attached — declared."""
    sections = _section(tmp_path, "The steps are listed in [@tbl:pipeline_steps].\n")
    producer = {
        **_PRODUCER,
        "GNN_STEP_TABLE": "| Step |\n|---|\n| 0 | : steps {#tbl:pipeline_steps}",
    }
    assert GATE._crossref_issues(sections, producer) == []


# --- SC-3: the shipped copy ---------------------------------------------------


def test_an_unresolved_token_in_output_manuscript_is_flagged(tmp_path) -> None:
    rendered = tmp_path / "docs" / "manuscript"
    rendered.mkdir(parents=True)
    (rendered / "0X_section.md").write_text(
        "The value {{GNN_OOPS}} reaches the PDF verbatim.\n", encoding="utf-8"
    )
    issues = GATE._unresolved_output_tokens(rendered)
    assert any("GNN_OOPS" in issue for issue in issues), issues


def test_tokens_inside_code_spans_are_not_output_findings(tmp_path) -> None:
    """A backticked mention is documentation about tokens, not a live one."""
    rendered = tmp_path / "docs" / "manuscript"
    rendered.mkdir(parents=True)
    (rendered / "0X_ok.md").write_text(
        "Write `{{GNN_NOT_A_TOKEN}}` only inside backticks.\n", encoding="utf-8"
    )
    assert GATE._unresolved_output_tokens(rendered) == []


def test_a_missing_output_tree_is_not_an_output_finding(tmp_path) -> None:
    assert GATE._unresolved_output_tokens(tmp_path / "does_not_exist") == []


# --- SC-3: committed token map vs the producer --------------------------------


def _write_committed(tmp_path: Path, variables: dict[str, str]) -> Path:
    path = tmp_path / "manuscript_variables.json"
    path.write_text(json.dumps(variables, indent=2) + "\n", encoding="utf-8")
    return path


def test_a_stale_committed_token_map_is_detected(tmp_path) -> None:
    """The SC-3 regression: consistently-stale JSON passed every gate together."""
    committed = _write_committed(tmp_path, {**_PRODUCER, "GNN_TEST_FILE_COUNT": "373"})
    issue = GATE._committed_variables_issue(_PRODUCER, committed)
    assert issue.startswith("stale:"), issue
    assert "token_checksum" in issue


def test_a_fresh_committed_token_map_passes(tmp_path) -> None:
    committed = _write_committed(tmp_path, _PRODUCER)
    assert GATE._committed_variables_issue(_PRODUCER, committed) == ""


def test_the_checksum_ignores_the_commit_token(tmp_path) -> None:
    """Pin: GNN_GIT_COMMIT is normalized out of the checksum comparison.

    A committed map can never record the hash of the commit that carries
    it, so maps differing ONLY in ``GNN_GIT_COMMIT`` must both pass — the
    gate compares commit-stable counts and pins the commit via the receipt's
    ``counts_describe_commit`` instead. Without the normalization the gate
    fails on every commit that moves HEAD.
    """
    committed = _write_committed(tmp_path, {**_PRODUCER, "GNN_GIT_COMMIT": "deadbeef"})
    producer_now = {**_PRODUCER, "GNN_GIT_COMMIT": "cafef00d"}
    assert GATE._committed_variables_issue(producer_now, committed) == ""


def test_a_missing_committed_token_map_is_named(tmp_path) -> None:
    issue = GATE._committed_variables_issue(_PRODUCER, tmp_path / "absent.json")
    assert "missing" in issue and "z_generate_manuscript_variables" in issue


# --- SC-23: commit provenance --------------------------------------------------


def test_the_unknown_commit_sentinel_fails_the_gate() -> None:
    """A token map with no commit behind it is not publishable."""
    assert GATE._provenance_issue({**_PRODUCER, "GNN_GIT_COMMIT": "unknown"}) != ""
    assert GATE._provenance_issue(_PRODUCER) == ""
    assert GATE._provenance_issue({}) != ""


def test_the_dirty_tree_writes_a_machine_readable_receipt() -> None:
    """The dirty-tree caveat is a file the gates can read, not only stderr."""
    receipt = Z_GENERATE._dirty_receipt(
        _PRODUCER,
        {"git_available": True, "dirty_paths": ["src/gnn/main.py", "justfile"]},
    )
    assert receipt["counts_describe_commit"] == "abc1234"
    assert receipt["working_tree_clean"] is False
    assert receipt["dirty_path_count"] == 2
    assert receipt["generator"] == "scripts/z_generate_manuscript_variables.py"
    clean = Z_GENERATE._dirty_receipt(
        _PRODUCER, {"git_available": True, "dirty_paths": []}
    )
    assert clean["working_tree_clean"] is True
    without_git = Z_GENERATE._dirty_receipt(
        {"GNN_GIT_COMMIT": "unknown"}, {"git_available": False, "dirty_paths": []}
    )
    assert without_git["git_available"] is False
    assert without_git["working_tree_clean"] is False


_STUB = '''"""Canned producer double for the token-gate tests.

The real ``gnn.manuscript`` producer introspects the live repository; these
tests exercise the generator script's control flow, so the stub returns the
fixed token map stored in the sibling ``variables.json``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def generate_variables(project_root: Path) -> dict[str, str]:
    data = json.loads(
        (Path(__file__).parent / "variables.json").read_text(encoding="utf-8")
    )
    return dict(data)


def sync_config_metadata(project_root: Path, variables: Mapping[str, str]) -> list[str]:
    return []


def sync_preamble_metadata(project_root: Path, variables: Mapping[str, str]) -> list[str]:
    return []


def save_variables(variables: Mapping[str, str], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(variables, indent=2, sort_keys=True, ensure_ascii=False) + "\\n",
        encoding="utf-8",
    )
    return out_path
'''


def _write_tmp_project(tmp_path: Path, variables: dict[str, str]) -> Path:
    """Scaffold a minimal project root under *tmp_path* around the real script.

    ``scripts/z_generate_manuscript_variables.py`` resolves every path it
    writes from its own location (``_PROJECT_ROOT = Path(__file__).resolve()
    .parents[1]``), so the real script is copied into the tmp project and the
    producer package is replaced by the canned stub above. The generator's
    writes — ``output/data/manuscript_variables{,_receipt}.json`` — land
    under *tmp_path* and the committed artifacts are never touched.
    """
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(
        REPO_ROOT / "scripts" / "z_generate_manuscript_variables.py",
        scripts / "z_generate_manuscript_variables.py",
    )
    package = tmp_path / "src" / "gnn" / "manuscript"
    package.mkdir(parents=True)
    (tmp_path / "src" / "gnn" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text(_STUB, encoding="utf-8")
    (package / "variables.json").write_text(
        json.dumps(variables, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return tmp_path


def test_the_receipt_lands_under_output_data_after_generation(tmp_path) -> None:
    """SC-23: after generation the receipt lives at ``<root>/output/data/``.

    The generation runs in a tmp project (``_write_tmp_project``) and the
    assertions read the freshly written tmp copy, so a test run never
    regenerates the committed ``output/data`` artifacts against the live
    tree (the test-hygiene regression recorded in TO-DO.md, 2026-09-17).
    """
    project = _write_tmp_project(tmp_path, _PRODUCER)
    env = dict(os.environ)
    env.pop("TEMPLATE_REPO_ROOT", None)
    result = subprocess.run(
        [
            sys.executable,
            str(project / "scripts" / "z_generate_manuscript_variables.py"),
        ],
        cwd=str(project),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    receipt = project / "output" / "data" / "manuscript_variables_receipt.json"
    assert receipt.is_file(), result.stderr
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["generator"] == "scripts/z_generate_manuscript_variables.py"
    assert payload["counts_describe_commit"] == _PRODUCER["GNN_GIT_COMMIT"]
    assert (project / "output" / "data" / "manuscript_variables.json").is_file()


def test_the_render_invocation_contract_is_documented_in_repo() -> None:
    """SC-23: the script name and invocation expectations are pinned in-repo.

    The render contract ("invokes this exact script name automatically") lives
    in the external template repo; these are the in-repo traces that must
    survive: the manuscript's reproducibility section publishes the command,
    the producer module names its orchestrator, and the orchestrator documents
    the detection mechanism and the launcher that exports it.
    """
    repro = (REPO_ROOT / "docs" / "manuscript" / "05_reproducibility.md").read_text(
        encoding="utf-8"
    )
    assert "scripts/z_generate_manuscript_variables.py" in repro
    z_generate_text = (
        REPO_ROOT / "scripts" / "z_generate_manuscript_variables.py"
    ).read_text(encoding="utf-8")
    assert "run_manuscript_variable_script" in z_generate_text, (
        "the template launcher that invokes this script is not named"
    )
    assert "TEMPLATE_REPO_ROOT" in z_generate_text, (
        "the render-invoked detection mechanism is not documented"
    )
    producer_text = (
        REPO_ROOT / "src" / "gnn" / "manuscript" / "variables.py"
    ).read_text(encoding="utf-8")
    assert "z_generate_manuscript_variables" in producer_text, (
        "the producer module no longer names the thin orchestrator script"
    )


def test_render_invocation_is_detected_by_the_template_root_env() -> None:
    """The documented mechanism: TEMPLATE_REPO_ROOT set ⇒ render-invoked."""
    sentinel = "GNN_TEST_TEMPLATE_REPO_ROOT_SENTINEL"
    old = os.environ.get("TEMPLATE_REPO_ROOT")
    try:
        os.environ["TEMPLATE_REPO_ROOT"] = sentinel
        assert Z_GENERATE._render_invoked() is True
        del os.environ["TEMPLATE_REPO_ROOT"]
        assert Z_GENERATE._render_invoked() is False
    finally:
        if old is not None:
            os.environ["TEMPLATE_REPO_ROOT"] = old


def test_the_render_invoked_mode_fails_without_the_injector(tmp_path) -> None:
    """SC-3-4: a render-invoked run that cannot hydrate exits non-zero.

    Standalone keeps its graceful exit; the render would ship an
    unsubstituted manuscript, so it must fail instead. The generator writes
    the variables map and receipt BEFORE the injector check, so against the
    live tree every invocation rewrote the committed
    ``output/data/manuscript_variables{,_receipt}.json`` (the hygiene
    regression recorded in TO-DO.md, 2026-09-17). Here the run happens in a tmp
    project: those writes land under tmp_path and a digest guard pins the
    committed artifacts as byte-identical across the run.
    """
    project = _write_tmp_project(tmp_path, _PRODUCER)

    def _digest(path: Path) -> str:
        return (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file()
            else "absent"
        )

    committed = (
        REPO_ROOT / "output" / "data" / "manuscript_variables.json",
        REPO_ROOT / "output" / "data" / "manuscript_variables_receipt.json",
    )
    before = tuple(_digest(path) for path in committed)

    env = dict(os.environ)
    env["TEMPLATE_REPO_ROOT"] = "/nonexistent-template-root"
    result = subprocess.run(
        [
            sys.executable,
            str(project / "scripts" / "z_generate_manuscript_variables.py"),
        ],
        cwd=str(project),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 1, result.stderr
    assert "render invocation could not load the template injector" in result.stderr
    assert tuple(_digest(path) for path in committed) == before
    assert (project / "output" / "data" / "manuscript_variables.json").is_file()
    assert (project / "output" / "data" / "manuscript_variables_receipt.json").is_file()


if __name__ == "__main__":  # pragma: no cover - convenience
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))

# --- SC-20: the recorder observes every read path ------------------------------


def test_the_recorder_observes_membership_and_iteration() -> None:
    """SC-20: reads through `in`, iteration and views are recorded too."""
    from scripts.lib import manuscript_figure_tokens as mft

    mft._CONSUMED.clear()
    tokens = mft.RecordingTokens({"GNN_A": "1", "GNN_B": "2"})
    assert "GNN_A" in tokens
    assert "GNN_NOPE" not in tokens
    assert sorted(tokens) == ["GNN_A", "GNN_B"]
    assert sorted(tokens.items()) == [("GNN_A", "1"), ("GNN_B", "2")]
    assert sorted(tokens.values()) == ["1", "2"]
    assert mft._CONSUMED == {"GNN_A": "1", "GNN_B": "2"}
