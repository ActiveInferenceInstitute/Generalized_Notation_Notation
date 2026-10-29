"""Regression tests for the ``lean`` executor framework registration."""

from __future__ import annotations

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

    (tmp_path / "src" / package).mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
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

    (tmp_path / "src" / "fep_lean").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    monkeypatch.setenv("FEP_LEAN_ROOT", str(tmp_path))
    monkeypatch.setenv("FEP_FORMAL_ROOT", str(tmp_path / "missing"))
    assert resolve_fep_formal_root() is None


def test_checkout_requires_package_and_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.execute.lean import resolve_fep_formal_root

    monkeypatch.setenv("FEP_FORMAL_ROOT", str(tmp_path))
    assert resolve_fep_formal_root() is None
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    assert resolve_fep_formal_root() is None
    (tmp_path / "src" / "fep_formal").mkdir(parents=True)
    assert resolve_fep_formal_root() == tmp_path


def _runtime_checkout(root: Path, package: str = "fep_formal") -> None:
    (root / "src" / package).mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.uv]\nrequired-version = "==0.12.34"\n')
    (root / ".python-version").write_text("3.14\n")
    (root / ".venv").mkdir()
    (root / ".venv/pyvenv.cfg").write_text("version_info = 3.14.0\n")


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
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-secret-must-not-leak")
    invocation = fep_runtime_invocation(
        tmp_path, [fep_bridge_executable(tmp_path), "--help"]
    )
    assert invocation.command[:6] == (
        "uv",
        "tool",
        "run",
        "--from",
        "uv==0.12.34",
        "uv",
    )
    assert "--frozen" in invocation.command and "--no-sync" in invocation.command
    assert invocation.command[-2] == (
        "fep-formal" if package == "fep_formal" else "fep-lean"
    )
    assert invocation.environment["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / ".venv")
    assert invocation.environment["UV_PYTHON"] == "3.14"
    assert invocation.environment["VIRTUAL_ENV"] == str(tmp_path / ".venv")
    assert "synthetic-secret-must-not-leak" not in repr(invocation)
    assert "OPENAI_API_KEY" not in invocation.environment
    assert invocation.environment["UV_PYTHON_DOWNLOADS"] == "never"
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
    (tmp_path / "pyproject.toml").write_text(f'[tool.uv]\nrequired-version = "{pin}"\n')
    with pytest.raises(FepRuntimeConfigurationError, match="exact stable uv") as error:
        fep_runtime_invocation(tmp_path, ["--help"])
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
        (tmp_path / "pyproject.toml").write_text('tool = "synthetic-secret"\n')
    with pytest.raises(FepRuntimeConfigurationError) as error:
        fep_runtime_invocation(tmp_path, ["--help"])
    assert "synthetic-secret" not in str(error.value)


def test_standalone_uv_pin_is_supported(tmp_path: Path) -> None:
    from gnn.execute.lean.lean_runner import fep_runtime_invocation

    _runtime_checkout(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "fixture"\n')
    (tmp_path / "uv.toml").write_text('required-version = "==0.12.35"\n')
    assert "uv==0.12.35" in fep_runtime_invocation(tmp_path, ["--help"]).command


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
    (tmp_path / "pyproject.toml").write_text(
        '[tool.uv]\nrequired-version = "synthetic-secret"\n'
    )
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
        fep_runtime_invocation(tmp_path, ["--help"])
