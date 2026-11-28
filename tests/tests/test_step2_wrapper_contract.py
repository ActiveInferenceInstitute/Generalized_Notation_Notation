"""Contract for the Step 2 wrapper surface: ``tests.run_tests`` routing args."""

from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast


def test_run_tests_signature_matches_documented_contract() -> None:
    """2_tests.py and docs/gnn/modules/02_tests.md depend on this signature.

    Import from ``tests.runner`` (the canonical source): under pytest the
    conftest registers a minimal ``sys.modules['tests']`` alias, so the
    package-level attribute is only reliable outside pytest — see
    ``test_tests_package_imports.py``.
    """
    from tests.runner import run_tests

    params = inspect.signature(run_tests).parameters
    assert list(params) == [
        "logger",
        "output_dir",
        "verbose",
        "fast_only",
        "comprehensive",
        "generate_coverage",
        "auto_fallback",
    ]
    assert params["fast_only"].default is True
    assert params["comprehensive"].default is False
    assert params["auto_fallback"].default is True


def test_step2_wrapper_exists_and_is_thin() -> None:
    """``src/2_tests.py`` must keep delegating to ``tests.run_tests``."""
    source = (
        Path(__file__).resolve().parents[2] / "src" / "gnn" / "2_tests.py"
    ).read_text(encoding="utf-8")
    assert "from tests import run_tests" in source
    assert "SKIP_TESTS_IN_PIPELINE" in source


@pytest.mark.parametrize("skip", [False, True], ids=["run", "skip"])
def test_step2_script_mode_resolves_harness_when_requested(
    tmp_path: Path, skip: bool
) -> None:
    """Step 2 finds its checkout harness on demand and skips before importing it."""
    repo_root = Path(__file__).resolve().parents[2]
    relative_script = Path("src/gnn/2_tests.py")
    staged_script = tmp_path / relative_script
    staged_script.parent.mkdir(parents=True)
    shutil.copyfile(repo_root / relative_script, staged_script)

    tests_package = tmp_path / "tests"
    tests_package.mkdir()
    harness_source = (
        "raise RuntimeError('Step 2 must not import the harness when skipped')\n"
        if skip
        else "def run_tests(**kwargs):\n    print('staged test harness ran')\n    return True\n"
    )
    (tests_package / "__init__.py").write_text(harness_source, encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(repo_root / "src"), env.get("PYTHONPATH")))
    )
    if skip:
        env["SKIP_TESTS_IN_PIPELINE"] = "1"
    else:
        env.pop("SKIP_TESTS_IN_PIPELINE", None)
    proc = subprocess.run(
        [
            sys.executable,
            str(staged_script),
            "--target-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    expected_log = "tests will be skipped" if skip else "staged test harness ran"
    assert expected_log in proc.stdout + proc.stderr


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_pipeline_reports_step2_skip_without_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parallel: bool
) -> None:
    from gnn import main as pipeline_main
    from gnn.cli import main
    from gnn.pipeline.hasher import verify_indexed_run

    def refuse_dispatch(*args: object, **kwargs: object) -> None:
        pytest.fail("The scheduler dispatched a worker for skipped Step 2")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SKIP_TESTS_IN_PIPELINE", "1")
    monkeypatch.setattr(pipeline_main, "execute_pipeline_step", refuse_dispatch)
    target = tmp_path / "input"
    output = tmp_path / "output"
    target.mkdir()
    argv = [
        "run",
        "--target-dir",
        str(target),
        "--output-dir",
        str(output),
        "--only-steps",
        "2",
        "--skip-steps",
        "1",
    ]
    if parallel:
        argv.append("--parallel")
    assert main(argv) == 0

    summary_dir = output / "00_pipeline_summary"
    summary = json.loads((summary_dir / "pipeline_execution_summary.json").read_text())
    assert summary["overall_status"] == "SUCCESS"
    (step,) = summary["steps"]
    assert step["script_name"] == "2_tests.py"
    assert step["status"] == "SKIPPED"
    assert step["skip_reason"] == "SKIP_TESTS_IN_PIPELINE"
    assert step["artifacts"] == []
    assert not (output / "2_tests_output").exists()
    report = (output / "PIPELINE_REPORT.md").read_text()
    assert "Test suite execution | ⏭️ SKIPPED (SKIP_TESTS_IN_PIPELINE)" in report

    checkpoint = json.loads((summary_dir / "run_session.json").read_text())
    (unit,) = checkpoint["units"]
    assert (unit["unit_id"], unit["status"]) == ("2_tests.py", "SKIPPED")

    index = json.loads((summary_dir / ".history/index.json").read_text())
    entry = index[summary["run_hash"]]
    assert verify_indexed_run(entry) == []
    monkeypatch.delenv("SKIP_TESTS_IN_PIPELINE")
    assert verify_indexed_run(entry) != []
