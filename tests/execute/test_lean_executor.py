"""Regression tests for the ``lean`` executor framework registration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import gnn.execute.executor as executor_module
from gnn.execute.executor import (
    ExecutorFrameworkSpec,
    GNNExecutor,
    _framework_specs,
    _RunnerState,
    list_frameworks,
)


def test_executor_registers_lean() -> None:
    frameworks = {entry["framework"]: entry for entry in list_frameworks()}
    assert "lean" in frameworks
    assert frameworks["lean"]["result_key"] == "lean_executions"
    assert frameworks["lean"]["operation"] == "execute_lean_verification"


def test_lean_spec_present() -> None:
    specs = [spec for spec in _framework_specs() if spec.framework_dir_key == "lean"]
    assert len(specs) == 1
    spec = specs[0]
    assert isinstance(spec, ExecutorFrameworkSpec)
    assert spec.result_key == "lean_executions"
    assert "FEP_LEAN_ROOT" in spec.unavailable_log
    assert "FEP_LEAN_ROOT" in spec.unavailable_message


def test_executor_lean_unavailable_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no resolvable fep_lean checkout, the executor reports unavailable."""
    from gnn.execute.lean.lean_runner import FEP_LEAN_ROOT_ENV

    monkeypatch.delenv("FEP_FORMAL_ROOT", raising=False)
    monkeypatch.setenv(FEP_LEAN_ROOT_ENV, "/nonexistent/fep_lean")
    from gnn.execute.lean.lean_runner import resolve_fep_lean_root

    assert resolve_fep_lean_root() is None

    monkeypatch.setattr(
        executor_module,
        "_runner_state",
        lambda key: _RunnerState(available=False, runner=None),
    )
    result = GNNExecutor()._execute_lean_verification("model.md")
    assert result == {"success": False, "error": "fep_lean unavailable"}


def _source_checkout(root: Path, package: str = "fep_formal") -> None:
    """Write real path/metadata inputs for the planner; no tool is simulated."""
    project = "fep-formal" if package == "fep_formal" else "fep_lean"
    executable = "fep-formal" if package == "fep_formal" else "fep-lean"
    source = root / "src" / package
    source.mkdir(parents=True)
    (source / "cli.py").write_text("def main():\n    return 0\n")
    (root / "pyproject.toml").write_text(
        f"[project]\nname = {project!r}\n[project.scripts]\n"
        f"{executable} = '{package}.cli:main'\n"
        '[tool.uv]\nrequired-version = "==0.12.34"\n'
    )


@pytest.mark.parametrize("package", ["fep_formal", "fep_lean"])
@pytest.mark.parametrize("variable", ["FEP_FORMAL_ROOT", "FEP_LEAN_ROOT"])
def test_checkout_namespace_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package: str, variable: str
) -> None:
    from gnn.execute.lean import (
        fep_bridge_executable,
        resolve_fep_formal_root,
        resolve_fep_lean_root,
    )

    _source_checkout(tmp_path, package)
    monkeypatch.delenv("FEP_FORMAL_ROOT", raising=False)
    monkeypatch.delenv("FEP_LEAN_ROOT", raising=False)
    monkeypatch.setenv(variable, str(tmp_path))
    assert resolve_fep_formal_root() == tmp_path
    assert resolve_fep_lean_root() == tmp_path
    expected = "fep-formal" if package == "fep_formal" else "fep-lean"
    assert fep_bridge_executable(tmp_path) == expected


def test_canonical_override_does_not_fall_back_when_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.execute.lean import resolve_fep_formal_root

    _source_checkout(tmp_path, "fep_lean")
    monkeypatch.setenv("FEP_LEAN_ROOT", str(tmp_path))
    monkeypatch.setenv("FEP_FORMAL_ROOT", str(tmp_path / "missing"))
    assert resolve_fep_formal_root() is None


def test_checkout_requires_package_and_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.execute.lean import resolve_fep_formal_root

    monkeypatch.setenv("FEP_FORMAL_ROOT", str(tmp_path))
    assert resolve_fep_formal_root() is None
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='fep-formal'\n[project.scripts]\n"
        "fep-formal='fep_formal.cli:main'\n"
    )
    assert resolve_fep_formal_root() is None
    (tmp_path / "src" / "fep_formal").mkdir(parents=True)
    assert resolve_fep_formal_root() is None
    (tmp_path / "src/fep_formal/cli.py").write_text("def main():\n    return 0\n")
    assert resolve_fep_formal_root() == tmp_path


def _runtime_checkout(root: Path, package: str = "fep_formal") -> None:
    """Create planning inputs only; these files are never external executions."""
    _source_checkout(root, package)
    (root / ".python-version").write_text("3.14\n")
    (root / ".venv").mkdir()
    (root / ".venv/pyvenv.cfg").write_text("version_info = 3.14.0\n")
    bindir = root / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    bindir.mkdir()
    executable = "fep-formal" if package == "fep_formal" else "fep-lean"
    for name in (
        "python.exe" if os.name == "nt" else "python",
        executable + (".exe" if os.name == "nt" else ""),
    ):
        path = bindir / name
        path.write_text("Planning input only; not an executable tool implementation.\n")
        path.chmod(0o755)


@pytest.mark.parametrize("package", ["fep_formal", "fep_lean"])
def test_runtime_invocation_uses_companion_pin_and_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    package: str,
) -> None:
    from gnn.execute.lean.lean_runner import (
        fep_bridge_executable,
        fep_runtime_invocation,
    )

    _runtime_checkout(tmp_path, package)
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/wrong/environment")
    monkeypatch.setenv("UV_PYTHON", "/wrong/python")
    monkeypatch.setenv("PYTHONPATH", "/caller/source")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-secret-must-not-leak")
    invocation = fep_runtime_invocation(
        tmp_path, [fep_bridge_executable(tmp_path), "--help"]
    )
    assert invocation.command[:7] == (
        "uv",
        "tool",
        "run",
        "--offline",
        "--from",
        "uv==0.12.34",
        "uv",
    )
    assert "--frozen" in invocation.command and "--no-sync" in invocation.command
    assert invocation.command.count("--offline") == 2
    bindir = tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    interpreter = str(bindir / ("python.exe" if os.name == "nt" else "python"))
    assert invocation.command[invocation.command.index("--python") + 1] == interpreter
    assert invocation.command[invocation.command.index("-B") - 1] == interpreter
    assert invocation.command[-1] == "--help"
    bootstrap = invocation.command[invocation.command.index("-c") + 1]
    assert f"{package}.cli" in bootstrap
    assert str(tmp_path / "src" / package / "cli.py") in bootstrap
    assert "sys.executable" in bootstrap and "sys.prefix" in bootstrap
    assert invocation.environment["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / ".venv")
    assert invocation.environment["UV_PYTHON"] == interpreter
    assert invocation.environment["VIRTUAL_ENV"] == str(tmp_path / ".venv")
    assert "synthetic-secret-must-not-leak" not in repr(invocation)
    assert "OPENAI_API_KEY" not in invocation.environment
    assert invocation.environment["UV_PYTHON_DOWNLOADS"] == "never"
    assert invocation.environment["UV_OFFLINE"] == "1"
    assert (
        invocation.environment["PYTHONPATH"]
        == str(tmp_path / "src") + os.pathsep + "/caller/source"
    )
    assert invocation.environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert __import__("os").environ["UV_PROJECT_ENVIRONMENT"] == "/wrong/environment"


@pytest.mark.parametrize("pin", ["", ">=0.13.0", "==synthetic-secret", "==0.13.0rc1"])
def test_invalid_runtime_pin_is_explicit_and_not_disclosed(
    tmp_path: Path, pin: str
) -> None:
    from gnn.execute.lean.lean_runner import (
        FepRuntimeConfigurationError,
        fep_runtime_invocation,
    )

    _runtime_checkout(tmp_path)
    metadata = tmp_path / "pyproject.toml"
    metadata.write_text(metadata.read_text().replace("==0.12.34", pin))
    with pytest.raises(FepRuntimeConfigurationError, match="exact stable uv") as error:
        fep_runtime_invocation(tmp_path, ["fep-formal", "--help"])
    assert "synthetic-secret" not in str(error.value)


@pytest.mark.parametrize(
    "case", ["missing", "malformed", "python", "environment", "conflict", "table"]
)
def test_incomplete_runtime_metadata_fails_closed(tmp_path: Path, case: str) -> None:
    from gnn.execute.lean.lean_runner import (
        FepRuntimeConfigurationError,
        fep_runtime_invocation,
    )

    _runtime_checkout(tmp_path)
    if case == "missing":
        (tmp_path / ".python-version").unlink()
    elif case == "malformed":
        (tmp_path / "pyproject.toml").write_text("not valid toml synthetic-secret")
    elif case == "python":
        (tmp_path / ".python-version").write_text("synthetic-secret")
    elif case == "environment":
        (tmp_path / ".venv/pyvenv.cfg").unlink()
    elif case == "conflict":
        (tmp_path / "uv.toml").write_text('required-version = "==0.99.1"\n')
    else:
        metadata = tmp_path / "pyproject.toml"
        metadata.write_text(
            'tool = "synthetic-secret"\n' + metadata.read_text().split("[tool.uv]")[0]
        )
    with pytest.raises(FepRuntimeConfigurationError) as error:
        fep_runtime_invocation(tmp_path, ["fep-formal", "--help"])
    assert "synthetic-secret" not in str(error.value)


def test_standalone_uv_pin_is_supported(tmp_path: Path) -> None:
    from gnn.execute.lean.lean_runner import fep_runtime_invocation

    _runtime_checkout(tmp_path)
    metadata = tmp_path / "pyproject.toml"
    metadata.write_text(metadata.read_text().split("[tool.uv]")[0])
    (tmp_path / "uv.toml").write_text('required-version = "==0.12.35"\n')
    assert (
        "uv==0.12.35"
        in fep_runtime_invocation(tmp_path, ["fep-formal", "--help"]).command
    )


def test_blank_canonical_root_never_uses_cwd_or_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gnn.execute.lean.lean_runner import resolve_fep_formal_root

    _runtime_checkout(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FEP_FORMAL_ROOT", "")
    monkeypatch.setenv("FEP_LEAN_ROOT", str(tmp_path))
    assert resolve_fep_formal_root() is None


def test_invalid_metadata_returns_failure_without_secret_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gnn.execute.lean.lean_runner import verify_document

    _runtime_checkout(tmp_path)
    metadata = tmp_path / "pyproject.toml"
    metadata.write_text(metadata.read_text().replace("==0.12.34", "synthetic-secret"))
    monkeypatch.setenv("FEP_FORMAL_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-caller-secret")
    result = verify_document(tmp_path / "model.md", tmp_path / "receipt.json")
    assert result["success"] is False
    assert result["error_type"] == "FepRuntimeConfigurationError"
    assert "synthetic-secret" not in repr(result)
    assert "synthetic-caller-secret" not in repr(result)
    assert not (tmp_path / "receipt.json").exists()


def test_mismatched_installed_python_fails_before_no_sync_can_ignore_it(
    tmp_path: Path,
) -> None:
    from gnn.execute.lean.lean_runner import (
        FepRuntimeConfigurationError,
        fep_runtime_invocation,
    )

    _runtime_checkout(tmp_path)
    (tmp_path / ".venv/pyvenv.cfg").write_text("version_info = 3.12.0\n")
    with pytest.raises(
        FepRuntimeConfigurationError, match="does not match declared Python"
    ):
        fep_runtime_invocation(tmp_path, ["fep-formal", "--help"])


def test_default_checkout_prefers_canonical_sibling_then_released_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.execute.lean import lean_runner

    vault = tmp_path / "vault"
    canonical = vault / "fep_formal"
    released = vault / "fep_lean"
    _source_checkout(canonical)
    _source_checkout(released, "fep_lean")
    monkeypatch.setattr(
        lean_runner, "__file__", str(vault / "GNN/src/gnn/execute/lean/lean_runner.py")
    )
    monkeypatch.delenv("FEP_FORMAL_ROOT", raising=False)
    monkeypatch.delenv("FEP_LEAN_ROOT", raising=False)
    assert lean_runner.resolve_fep_formal_root() == canonical
    (canonical / "src/fep_formal/cli.py").unlink()
    assert lean_runner.resolve_fep_formal_root() == released


@pytest.mark.parametrize("defect", ["identity", "script", "target", "physical_source"])
def test_cli_contract_rejects_cross_namespace_or_missing_source(
    tmp_path: Path, defect: str
) -> None:
    from gnn.execute.lean.lean_runner import (
        FepRuntimeConfigurationError,
        fep_bridge_executable,
    )

    _source_checkout(tmp_path)
    metadata = tmp_path / "pyproject.toml"
    if defect == "identity":
        metadata.write_text(
            metadata.read_text().replace("'fep-formal'", "'unknown-project'")
        )
    elif defect == "script":
        metadata.write_text(metadata.read_text().replace("fep-formal =", "fep-lean ="))
    elif defect == "target":
        metadata.write_text(
            metadata.read_text().replace("fep_formal.cli:main", "fep_lean.cli:main")
        )
    else:
        (tmp_path / "src/fep_formal/cli.py").unlink()
    with pytest.raises(FepRuntimeConfigurationError):
        fep_bridge_executable(tmp_path)


@pytest.mark.parametrize("missing", ["python", "console"])
def test_unprovisioned_environment_fails_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    from gnn.execute.lean import lean_runner

    _runtime_checkout(tmp_path)
    bindir = tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    name = (
        ("python.exe" if os.name == "nt" else "python")
        if missing == "python"
        else "fep-formal" + (".exe" if os.name == "nt" else "")
    )
    (bindir / name).unlink()
    monkeypatch.setenv("FEP_FORMAL_ROOT", str(tmp_path))

    def reject_spawn(*args: object, **kwargs: object) -> None:
        raise AssertionError("Invalid planning inputs must never reach the envelope")

    monkeypatch.setattr(lean_runner, "run_subprocess_envelope", reject_spawn)
    result = lean_runner.verify_document(
        tmp_path / "model.md", tmp_path / "receipt.json"
    )
    assert result["success"] is False
    assert result["error_type"] == "FepRuntimeConfigurationError"
    assert result["return_code"] == lean_runner.NEVER_STARTED
    assert not (tmp_path / "receipt.json").exists()


def test_runtime_arguments_cannot_select_another_console(tmp_path: Path) -> None:
    from gnn.execute.lean.lean_runner import (
        FepRuntimeConfigurationError,
        fep_runtime_invocation,
    )

    _runtime_checkout(tmp_path)
    with pytest.raises(FepRuntimeConfigurationError, match="declared CLI"):
        fep_runtime_invocation(tmp_path, ["fep-lean", "--help"])
