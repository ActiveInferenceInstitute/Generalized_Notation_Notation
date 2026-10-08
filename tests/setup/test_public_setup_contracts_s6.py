"""Public setup contracts using native commands and caller-owned scratch data.

Path bindings isolate setup's module-level project configuration. Commands,
package metadata queries, filesystem operations and serializers remain real;
these tests never install dependencies or recreate an interpreter environment.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

from gnn.setup import dependency_setup, uv_management


@pytest.fixture
def setup_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Bind the setup data paths to scratch without replacing native operations."""
    root = tmp_path / "project"
    root.mkdir()
    bindings = {
        "PROJECT_ROOT": root,
        "PYPROJECT_PATH": root / "pyproject.toml",
        "LOCK_PATH": root / "uv.lock",
        "VENV_PATH": root / "environment-artifacts",
        "VENV_PYTHON": Path(sys.executable),
    }
    for name, value in bindings.items():
        monkeypatch.setattr(uv_management, name, value)
    monkeypatch.setattr(dependency_setup, "PROJECT_ROOT", root)
    return root


def test_native_command_preserves_cwd_literal_arguments_and_decoding(
    setup_workspace: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Arguments remain literal, output decodes safely, and cwd stays caller-owned."""
    literal = "argument;$(never-execute)"
    script = (
        "import json, os, sys; "
        "print(json.dumps({'cwd': os.getcwd(), 'argument': sys.argv[1]})); "
        "sys.stderr.buffer.write(b'non-utf8: \\xff\\n')"
    )
    with caplog.at_level(logging.DEBUG, logger=uv_management.__name__):
        result = uv_management.run_command(
            [sys.executable, "-c", script, literal],
            cwd=setup_workspace,
            verbose=True,
        )
    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "cwd": str(setup_workspace.resolve()),
        "argument": literal,
    }
    assert result.stderr.startswith("non-utf8: ")
    assert result.stderr.endswith("\n")
    assert "Stdout:" in caplog.text and "Stderr:" in caplog.text
    assert not list(setup_workspace.iterdir())


@pytest.mark.parametrize("check", [False, True])
def test_native_command_failure_keeps_output_and_exit_status(
    setup_workspace: Path, caplog: pytest.LogCaptureFixture, check: bool
) -> None:
    script = "import sys; print('public stdout'); print('public stderr', file=sys.stderr); sys.exit(7)"
    command = [sys.executable, "-c", script]
    if check:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            uv_management.run_command(command, cwd=setup_workspace, check=True)
        assert caught.value.returncode == 7
        assert caught.value.stdout == "public stdout\n"
        assert caught.value.stderr == "public stderr\n"
        assert "Return code: 7" in caplog.text
    else:
        result = uv_management.run_command(command, cwd=setup_workspace, check=False)
        assert result.returncode == 7
        assert result.stdout == "public stdout\n"
        assert result.stderr == "public stderr\n"
        assert "non-zero exit code: 7" in caplog.text
    assert "public stdout" in caplog.text and "public stderr" in caplog.text
    assert not list(setup_workspace.iterdir())


@pytest.mark.parametrize("check", [False, True])
def test_missing_native_command_is_an_explicit_failure(
    setup_workspace: Path, caplog: pytest.LogCaptureFixture, check: bool
) -> None:
    missing = setup_workspace / "absent-executable"
    with pytest.raises(FileNotFoundError):
        uv_management.run_command([str(missing)], cwd=setup_workspace, check=check)
    assert "Command not found" in caplog.text
    assert not list(setup_workspace.iterdir())


def test_dependency_sync_refuses_a_missing_manifest_without_creating_environment(
    setup_workspace: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert uv_management.install_uv_dependencies(verbose=True) is False
    assert "pyproject.toml not found" in caplog.text
    assert not list(setup_workspace.iterdir())


@pytest.mark.parametrize("group", ["unknown-extra", "--version", "graphs;unexpected"])
def test_optional_group_refusal_preserves_the_project(
    setup_workspace: Path, caplog: pytest.LogCaptureFixture, group: str
) -> None:
    manifest = setup_workspace / "pyproject.toml"
    manifest.write_text("this deliberately is not an installable manifest\n")
    before = manifest.read_bytes()
    with caplog.at_level(logging.INFO, logger=dependency_setup.__name__):
        assert (
            dependency_setup.install_optional_package_group(group, verbose=True)
            is False
        )
    assert "Unknown package group" in caplog.text
    assert "Available groups:" in caplog.text
    assert manifest.read_bytes() == before
    assert set(setup_workspace.iterdir()) == {manifest}


@pytest.mark.parametrize("group", ["JAX", "PyMDP", "numpyro", "plotly", "llm"])
def test_core_group_aliases_report_core_ownership_without_sync(
    setup_workspace: Path, caplog: pytest.LogCaptureFixture, group: str
) -> None:
    manifest = setup_workspace / "pyproject.toml"
    manifest.write_text("this deliberately is not an installable manifest\n")
    before = manifest.read_bytes()
    with caplog.at_level(logging.INFO, logger=dependency_setup.__name__):
        assert (
            dependency_setup.install_optional_package_group(group, verbose=True) is True
        )
    assert "installed by the core dependency set" in caplog.text
    assert manifest.read_bytes() == before
    assert set(setup_workspace.iterdir()) == {manifest}


def test_jax_setup_refuses_a_missing_interpreter_before_repair(
    setup_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        dependency_setup, "VENV_PYTHON", setup_workspace / "absent-python"
    )
    assert dependency_setup.install_jax_and_test(verbose=True) is False
    assert "Venv Python not found" in caplog.text
    assert not list(setup_workspace.iterdir())


def test_native_inventory_is_complete_and_preserves_unrelated_artifacts(
    setup_workspace: Path,
) -> None:
    """Query the actual current interpreter and serialize only into owned scratch."""
    artifacts = uv_management.VENV_PATH
    artifacts.mkdir()
    sentinel = artifacts / "caller-owned.txt"
    sentinel.write_text("retain me\n")
    for verbose in (False, True):
        packages = uv_management.get_installed_package_versions(verbose=verbose)
        for name in ("pytest", "numpy", "matplotlib", "scipy"):
            assert packages[name] == version(name)
        assert (
            json.loads((artifacts / "installed_packages_uv.json").read_text())
            == packages
        )
        assert sentinel.read_text() == "retain me\n"
        assert not list(artifacts.glob(".installed_packages_uv.json.*.tmp"))


def test_saved_setup_receipt_keeps_validation_and_configuration_distinct(
    setup_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uv_management.VENV_PATH.mkdir()
    monkeypatch.setattr(uv_management, "VENV_PYTHON", setup_workspace / "absent-python")
    validation = {
        "overall_status": False,
        "caller_reason": "requested-validation-failed",
    }
    output = setup_workspace / "receipt"
    uv_management.save_setup_results(
        output, validation, extras=["api", "thrml"], dev=True, install_all_extras=True
    )
    summary = json.loads((output / "environment_setup_summary.json").read_text())
    inventory = json.loads((output / "installed_packages.json").read_text())
    assert summary["validation"] == validation
    assert summary["configuration"]["extras_installed"] == ["api", "thrml"]
    assert summary["configuration"]["dev_dependencies"] is True
    assert summary["configuration"]["install_all_extras"] is True
    assert summary["system_info"]["python_executable"] == sys.executable
    assert summary["uv_info"]["project_root"] == str(setup_workspace)
    assert summary["uv_info"]["uv_setup_status"]["overall_status"] is False
    assert inventory == summary["uv_info"]["installed_packages"]
    assert inventory == {}
    assert not (setup_workspace / "pyproject.toml").exists()
    assert not (setup_workspace / "uv.lock").exists()


def test_cold_workspace_health_reports_missing_resources_without_repair(
    setup_workspace: Path,
) -> None:
    health = uv_management.check_environment_health(verbose=True)
    assert health["overall_healthy"] is False
    assert health["venv_exists"] is False
    assert health["venv_python_works"] is False
    assert health["pyproject_exists"] is False
    assert health["lock_file_exists"] is False
    assert {
        "pyproject.toml not found",
        "uv.lock not found",
        "Virtual environment not found",
    }.issubset(health["issues"])
    assert health["core_packages"]["pytest"] == version("pytest")
    assert set(health["tools"]) == {"pkl", "d2", "julia", "ruff", "ollama"}
    assert all(isinstance(value, bool) for value in health["tools"].values())
    assert not list(setup_workspace.iterdir())


def test_public_project_structure_emits_documented_artifacts_and_preserves_neighbors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "authored-project"
    root.mkdir()
    sentinel = root / "README.txt"
    sentinel.write_text("caller-authored content\n")
    assert (
        dependency_setup.create_project_structure(root, logging.getLogger(__name__))
        is True
    )
    assert (root / "input/gnn_files").is_dir()
    assert (root / "output/logs").is_dir()
    assert (root / "output/temp").is_dir()
    assert (root / "doc").is_dir()
    assert (root / "input/config.yaml").read_text() == "# GNN Pipeline Configuration\n"
    assert (root / "output/.gitkeep").read_text() == ""
    assert (root / "tests/__init__.py").read_text() == "# Tests package\n"
    assert sentinel.read_text() == "caller-authored content\n"


def test_public_project_structure_returns_failure_for_a_blocked_path(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocked = tmp_path / "caller-file"
    blocked.write_text("must remain a file\n")
    assert (
        dependency_setup.create_project_structure(blocked, logging.getLogger(__name__))
        is False
    )
    assert blocked.read_text() == "must remain a file\n"
    assert "Failed to create project structure" in caplog.text
