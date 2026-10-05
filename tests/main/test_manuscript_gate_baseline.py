"""Rendered evidence and immutable PR-baseline acceptance contracts."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gnn.manuscript import substitution
from gnn.manuscript.gate_baseline import compare_findings, resolve_base
from gnn.manuscript.render_custody import (
    hydration_issues,
    record_render_manifest,
    verify_fresh_render,
)

EXCLUDED = frozenset({"AGENTS.md"})


@pytest.mark.parametrize("configured_template", [False, True])
@pytest.mark.parametrize("entrypoint", ["exclusions", "hydration", "stale-hydration"])
def test_script_exclusion_contract_reexports_active_names_from_uninstalled_checkout(
    tmp_path: Path, configured_template: bool, entrypoint: str
) -> None:
    import sys

    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env.pop("GNN_MANUSCRIPT_BASE_REF", None)
    env.pop("TEMPLATE_REPO_ROOT", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if configured_template:
        template = tmp_path / "template"
        for package in ("scripts", "infrastructure", "infrastructure/rendering"):
            directory = template / package
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "__init__.py").write_text("")
        (template / "infrastructure/rendering/manuscript_injection.py").write_text(
            "import re\n"
            "_TOKEN_RE = re.compile(r'\\{\\{([A-Z][A-Z0-9_]*)\\}\\}')\n"
            "EXCLUDED_DOC_FILENAMES = frozenset({'AGENTS.md', 'MANUSCRIPT_STATUS.md', 'README.md', 'SYNTAX.md', 'TEMPLATE_GUIDE.md'})\n"
            "def substitute_manuscript_text(text, variables):\n"
            "    missing = []\n"
            "    def replace(match):\n"
            "        key = match.group(1)\n"
            "        if key in variables: return variables[key]\n"
            "        missing.append(key)\n"
            "        return match.group(0)\n"
            "    return _TOKEN_RE.sub(replace, text), missing\n"
        )
        env["TEMPLATE_REPO_ROOT"] = str(template)
    if entrypoint == "exclusions":
        helper = repo / "scripts/lib/manuscript_exclusions.py"
        command = [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import runpy,sys; contract=runpy.run_path(sys.argv[1]); assert 'AGENTS.md' in contract['EXCLUDED_DOC_FILENAMES']; assert 'SYNTAX.md' in contract['AUTHORING_GUIDE_FILENAMES']",
            str(helper),
        ]
    else:
        fixture = _tree(tmp_path / "project")
        (fixture / "src").symlink_to(repo / "src", target_is_directory=True)
        script = fixture / "scripts/check_hydrated_prose.py"
        script.parent.mkdir()
        script.write_bytes((repo / "scripts/check_hydrated_prose.py").read_bytes())
        if configured_template:
            (fixture / "manuscript/TEMPLATE_GUIDE.md").write_text("{{UNKNOWN}}\n")
        if entrypoint == "stale-hydration":
            (fixture / "output/manuscript/00_section.md").write_text("stale science\n")
        command = [sys.executable, "-I", "-B", str(script)]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=15, env=env
    )
    assert "ModuleNotFoundError" not in result.stderr, result.stderr
    if entrypoint == "stale-hydration":
        assert result.returncode == 1, result.stderr
        assert "[hydrated-prose]" in result.stderr
        assert "00_section.md" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        if entrypoint == "hydration":
            assert "Hydrated prose and rendered commit evidence match" in result.stdout


def _tree(root: Path) -> Path:
    for directory in ("manuscript", "output/data", "output/manuscript", "output/pdf"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "manuscript/00_section.md").write_text(
        "count {{COUNT}} at {{GNN_GIT_COMMIT}}\n"
    )
    (root / "output/manuscript/00_section.md").write_text("count 25 at abc1234\n")
    (root / "output/data/manuscript_variables.json").write_text(
        json.dumps({"COUNT": "25", "GNN_GIT_COMMIT": "abc1234"})
    )
    (root / "output/data/manuscript_variables_receipt.json").write_text(
        json.dumps({"counts_describe_commit": "abc1234"})
    )
    for suffix in ("md", "tex", "log"):
        (root / f"output/pdf/_combined_manuscript.{suffix}").write_text(
            "rendered at abc1234\n"
        )
    return root


@pytest.mark.parametrize("suffix", ["md", "tex"])
def test_hydrate_and_rerecord_cannot_bless_stale_render(
    tmp_path: Path, suffix: str
) -> None:
    root = _tree(tmp_path)
    (root / f"output/pdf/_combined_manuscript.{suffix}").write_text(
        "rendered at def5678\n"
    )
    record_render_manifest(root)
    assert any(
        f"_combined_manuscript.{suffix}" in issue
        for issue in hydration_issues(root, EXCLUDED)
    )


def test_missing_and_invalid_commit_evidence_fail(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "output/pdf/_combined_manuscript.md").unlink()
    assert any(
        "missing rendered commit evidence" in issue
        for issue in hydration_issues(root, EXCLUDED)
    )
    (root / "output/data/manuscript_variables.json").write_text(
        '{"GNN_GIT_COMMIT": null}'
    )
    assert "unreadable" in hydration_issues(root, EXCLUDED)[0]


def test_fresh_file_is_never_its_own_committed_baseline(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    record_render_manifest(root)
    (root / "output/pdf/_combined_manuscript.tex").write_text(
        "different science at abc1234\n"
    )
    assert any(
        "_combined_manuscript.tex" in issue for issue in verify_fresh_render(root)
    )


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _commit(root: Path) -> str:
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return _git(root, "rev-parse", "HEAD")


def _base_audit(root: Path, _sha: str) -> list[str]:
    return hydration_issues(root, EXCLUDED)


def test_unchanged_stale_base_warns_but_strict_main_fails(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "output/pdf/_combined_manuscript.tex").write_text("old render at def5678\n")
    _git(root, "init", "-q")
    base = _commit(root)
    (root / "README.md").write_text("docs only\n")
    _commit(root)
    findings = hydration_issues(root, EXCLUDED)
    result = compare_findings(root, findings, base, _base_audit)
    assert result.base_sha == base
    assert result.failures == ()
    assert result.inherited == tuple(findings)
    assert compare_findings(root, findings, None, _base_audit).failures == tuple(
        findings
    )


def test_same_diagnostic_with_changed_evidence_is_new_drift(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    artifact = root / "output/pdf/_combined_manuscript.tex"
    artifact.write_text("old render at def5678\n")
    _git(root, "init", "-q")
    base = _commit(root)
    artifact.write_text("even older science at def5678\n")
    _commit(root)
    result = compare_findings(root, hydration_issues(root, EXCLUDED), base, _base_audit)
    assert result.failures and not result.inherited


def test_new_missing_artifact_and_bad_base_are_failures(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _git(root, "init", "-q")
    base = _commit(root)
    (root / "output/pdf/_combined_manuscript.md").unlink()
    _commit(root)
    issues = hydration_issues(root, EXCLUDED)
    assert compare_findings(root, issues, base, _base_audit).failures
    assert any(
        "base comparison failed" in issue
        for issue in compare_findings(root, [], "missing", _base_audit).failures
    )


def test_base_is_merge_base_not_moving_target_tip(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _git(root, "init", "-q")
    base = _commit(root)
    _git(root, "checkout", "-qb", "feature")
    (root / "README.md").write_text("feature\n")
    feature = _commit(root)
    _git(root, "checkout", "-q", "--detach", base)
    (root / "README.md").write_text("target moved\n")
    target = _commit(root)
    _git(root, "checkout", "-q", "--detach", feature)
    assert resolve_base(root, target) == base


def test_template_substitution_is_preferred_and_unknown_tokens_survive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def inject(text: str, variables: dict[str, str]) -> tuple[str, list[str]]:
        calls.append((text, variables))
        return "template", []

    monkeypatch.setattr(
        substitution, "_INJECTOR", SimpleNamespace(substitute_manuscript_text=inject)
    )
    assert substitution.hydrate_text("{{COUNT}}", {"COUNT": "25"}) == "template"
    assert calls
    monkeypatch.setattr(substitution, "_INJECTOR", None)
    assert substitution.substitute_manuscript_text(
        "{{COUNT}} {{UNKNOWN}} {{lower}}", {"COUNT": "25"}
    ) == ("25 {{UNKNOWN}} {{lower}}", ["UNKNOWN"])


def test_preamble_comments_and_nonlatex_fences_cannot_declare_packages() -> None:
    text = "prose \\usepackage{listings}\n```latex\n% \\usepackage{listings}\n\\usepackage{xcolor} % comment\n\\newcommand{\\percent}{\\%}\n```\n"
    active = substitution.active_preamble(text)
    assert "listings" not in active
    assert "xcolor" in active and "\\%" in active
    assert "listings" not in substitution.active_preamble(
        "```python\n\\usepackage{listings}\n```"
    )


@pytest.mark.parametrize("head_count, succeeds", [("25", True), ("26", False)])
def test_strict_token_cli_waives_only_unchanged_base_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
    head_count: str,
    succeeds: bool,
) -> None:
    from scripts import check_manuscript_tokens as gate

    root = _tree(tmp_path)
    (root / "manuscript/references.bib").write_text("")
    (root / "manuscript/config.yaml").write_text("{}")
    (root / "output/data/manuscript_variables.json").write_text(
        json.dumps({"COUNT": "24", "GNN_GIT_COMMIT": "abc1234"})
    )
    _git(root, "init", "-q")
    base = _commit(root)
    (root / "README.md").write_text("docs only\n")
    _commit(root)
    monkeypatch.setattr(gate, "_PROJECT_ROOT", root)
    monkeypatch.setattr(
        gate,
        "generate_variables",
        lambda _root, snapshot=None: {
            "COUNT": "25" if snapshot else head_count,
            "GNN_GIT_COMMIT": "abc1234",
        },
    )
    monkeypatch.setattr(gate, "config_metadata_drift", lambda *_args: [])
    monkeypatch.setattr(gate, "_figure_artifact_issues", lambda *_args: ([], []))
    monkeypatch.setattr("sys.argv", ["tokens", "--strict", "--base-ref", base])
    assert (gate.main() == 0) == succeeds
    output = capsys.readouterr().out
    if succeeds:
        assert base in output and "unchanged inherited drift" in output
    else:
        assert "FAILED" in output
