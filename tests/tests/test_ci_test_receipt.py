"""JUnit evidence counts testcase leaves and reports source identity honestly."""

from pathlib import Path
from subprocess import CompletedProcess, run

import pytest

from scripts.test_receipt import build_receipt


@pytest.mark.parametrize(
    ("body", "executed", "passed", "failed", "errors", "skipped", "complete"),
    [
        (
            '<testsuite tests="99"><testsuite><testcase/><testcase><skipped/></testcase></testsuite><testsuite><testcase><failure/></testcase><testcase><error/></testcase></testsuite></testsuite>',
            3,
            1,
            1,
            1,
            1,
            False,
        ),
        ("<testsuite/>", 0, 0, 0, 0, 0, False),
        (
            "<testsuite><testcase><skipped/></testcase></testsuite>",
            0,
            0,
            0,
            0,
            1,
            False,
        ),
        ("<testsuite><testcase/><testcase/></testsuite>", 2, 2, 0, 0, 0, True),
    ],
)
def test_junit_leaf_counts_and_empty_failure_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: str,
    executed: int,
    passed: int,
    failed: int,
    errors: int,
    skipped: int,
    complete: bool,
) -> None:
    def git_run(command, **kwargs):
        return CompletedProcess(
            command,
            0,
            stdout="a" * 40 + "\n" if command[1] == "rev-parse" else "",
            stderr="",
        )

    monkeypatch.setattr("scripts.test_receipt.subprocess.run", git_run)
    path = tmp_path / "junit.xml"
    path.write_text(body)
    receipt = build_receipt(path)
    assert [
        receipt[key]
        for key in ("executed", "passed", "failed", "errors", "skipped", "complete")
    ] == [executed, passed, failed, errors, skipped, complete]
    assert receipt["identity_claim"] == "committed checkout"


def test_new_untracked_source_cannot_claim_a_committed_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def git_run(command, **kwargs):
        if command[1] == "status":
            assert "--untracked-files=no" not in command
            return CompletedProcess(
                command, 0, stdout="?? src/gnn/new_module.py\n", stderr=""
            )
        return CompletedProcess(command, 0, stdout="b" * 40 + "\n", stderr="")

    monkeypatch.setattr("scripts.test_receipt.subprocess.run", git_run)
    path = tmp_path / "junit.xml"
    path.write_text("<testsuite><testcase/></testsuite>")
    receipt = build_receipt(path)
    assert receipt["working_tree_dirty"] is True
    assert receipt["identity_claim"] == "commit plus working tree"


def test_generated_junit_files_preserve_clean_source_identity_but_source_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the maintained ignore rules against a real isolated Git index."""
    ignore = (Path(__file__).resolve().parents[2] / ".gitignore").read_bytes()
    (tmp_path / ".gitignore").write_bytes(ignore)
    run(["git", "init", "-q", str(tmp_path)], check=True)
    run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    run(
        [
            "git",
            "-c",
            "user.name=Receipt Test",
            "-c",
            "user.email=receipt@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Source baseline",
        ],
        cwd=tmp_path,
        check=True,
    )
    monkeypatch.chdir(tmp_path)
    junit = tmp_path / "junit"
    junit.mkdir()
    xml = junit / "pipeline-contracts.xml"
    xml.write_text("<testsuite><testcase/></testsuite>")
    (junit / "coverage.json").write_text("{}")
    receipt = build_receipt(xml)
    assert receipt["working_tree_dirty"] is False
    assert receipt["identity_claim"] == "committed checkout"
    (junit / "untracked_source.py").write_text("source = True\n")
    assert build_receipt(xml)["working_tree_dirty"] is True
    (junit / "untracked_source.py").unlink()
    (tmp_path / "src").mkdir()
    (tmp_path / "src/new_module.py").write_text("source = True\n")
    assert build_receipt(xml)["identity_claim"] == "commit plus working tree"
