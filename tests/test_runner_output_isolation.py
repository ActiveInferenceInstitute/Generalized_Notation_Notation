"""Regression tests for pipeline test output isolation."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tests import test_runner_helper, test_runner_modes
from tests.test_runner_modular import _isolated_pipeline_output_dir


def test_runner_helper_default_output_dir_honors_pipeline_env(
    monkeypatch: Any,
) -> None:
    """Nested test helpers should not fall back to tracked ``output/``."""
    monkeypatch.setenv("GNN_PIPELINE_TEST_OUTPUT_DIR", "/tmp/gnn-isolated-output")

    assert test_runner_helper._default_output_dir() == "/tmp/gnn-isolated-output"


def test_fast_pipeline_tests_pass_isolated_output_env(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """Step 2's fast pytest subprocess should inherit an isolated output root."""
    monkeypatch.delenv("GNN_PIPELINE_TEST_OUTPUT_DIR", raising=False)
    monkeypatch.setenv("GNN_RUN_CONTEXT_FILE", str(tmp_path / "parent-context.json"))
    captured: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        captured["cmd"] = args[0]
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="1 passed in 0.01s", stderr="")

    monkeypatch.setattr(test_runner_modes.subprocess, "run", fake_run)

    logger = logging.getLogger("test-fast-pipeline-isolation")
    assert test_runner_modes.run_fast_pipeline_tests(logger, tmp_path)

    expected = str(tmp_path / "isolated_pipeline_outputs")
    assert captured["env"]["GNN_PIPELINE_TEST_OUTPUT_DIR"] == expected
    assert "GNN_RUN_CONTEXT_FILE" not in captured["env"]
    # The fast-test subprocess must run from the project root (the directory
    # containing ``src/`` and ``pyproject.toml``). Assert against the real root
    # rather than a hard-coded checkout basename so the contract holds for any
    # linked worktree or renamed clone.
    project_root = Path(__file__).resolve().parents[1]
    assert Path(captured["cwd"]).resolve() == project_root


def test_modular_runner_uses_isolated_pipeline_output_dir(tmp_path: Any) -> None:
    """Category subprocesses should share the same isolation convention."""
    assert _isolated_pipeline_output_dir(tmp_path) == str(
        tmp_path / "isolated_pipeline_outputs"
    )


def test_predefined_helper_uses_existing_step2_script(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def capture_run(cmd: list[str], **kwargs: Any) -> Any:
        captured["cmd"] = cmd
        captured.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(test_runner_helper.subprocess, "run", capture_run)
    assert test_runner_helper.run_test_configuration(
        "minimal", output_dir=str(tmp_path)
    )
    script = Path(captured["cmd"][1])
    assert script == Path(__file__).resolve().parents[1] / "src" / "gnn" / "2_tests.py"
    assert script.is_file()
    assert "--no-coverage" not in captured["cmd"]


def test_custom_helper_applies_controls_and_preserves_failure(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from gnn.utils.testing.runner import TestRunner as PublicTestRunner

    captured: dict[str, Any] = {}

    def capture_tests(self: Any, paths: Any, output: Any) -> Any:
        captured.update(options=self.config, paths=paths, output=output)
        return {"success": False}

    monkeypatch.setattr(PublicTestRunner, "run_tests", capture_tests)
    assert not test_runner_helper.run_custom_tests(
        markers="unit",
        timeout=7,
        output_dir=str(tmp_path),
    )
    assert captured["options"]["timeout_seconds"] == 7
    assert "(unit)" in captured["options"]["markers"][0]
    assert captured["options"]["parallel"] is False
    assert captured["paths"] == [Path(__file__).resolve().parent]
    captured.clear()
    assert not test_runner_helper.run_custom_tests(markers="unit", timeout=-1)
    assert not captured


def test_helper_bootstrap_resolves_source_package_without_pythonpath() -> None:
    import subprocess
    import sys

    helper = Path(__file__).resolve().parent / "test_runner_helper.py"
    package = helper.parent.parent / "src" / "gnn" / "__init__.py"
    code = (
        f"import runpy; runpy.run_path({str(helper)!r}); import gnn; "
        f"assert gnn.__file__ == {str(package)!r}, gnn.__file__"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
