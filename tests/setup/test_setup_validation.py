#!/usr/bin/env python3
"""Phase 4.2 regression tests for setup (Step 1).

Uses real sys.version_info and real filesystem.
"""

import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_check_python_version_respects_current_interpreter() -> Any:
    """check_python_version applies the shared MIN_PYTHON_VERSION floor, which
    must equal the pyproject.toml requires-python lower bound."""
    import tomllib

    from gnn.setup import check_python_version
    from gnn.setup.constants import MIN_PYTHON_VERSION

    requires = tomllib.loads((SRC / "pyproject.toml").read_text())["project"][
        "requires-python"
    ]
    floor = ".".join(str(part) for part in MIN_PYTHON_VERSION)
    assert requires.startswith(f">={floor},"), (requires, MIN_PYTHON_VERSION)
    result = check_python_version()
    assert result is (sys.version_info[:2] >= MIN_PYTHON_VERSION)


def test_validate_environment_returns_dict() -> Any:
    """validate_environment must return a diagnostic dict, not raise."""
    from gnn.setup import validate_environment

    result = validate_environment()
    assert isinstance(result, dict)
    # Should expose at least a high-level status signal.
    expected_keys: set[Any] = {
        "valid",
        "python_version",
        "errors",
        "warnings",
        "overall_health",
        "status",
        "issues",
    }
    assert expected_keys & set(result.keys()), (
        f"validate_environment returned no recognized status key: {list(result.keys())}"
    )


def test_get_module_info_exposes_version() -> Any:
    from gnn.setup import get_module_info

    info = get_module_info()
    assert isinstance(info, dict)
    assert "version" in info
    # Version follows semver
    v = str(info["version"])
    assert v.count(".") >= 2, f"Unexpected version format: {v!r}"


def test_environment_manager_instantiates_without_side_effects() -> Any:
    """Constructing EnvironmentManager must not fail or mutate state."""
    from gnn.setup import EnvironmentManager

    mgr = EnvironmentManager()
    assert mgr is not None


def test_uv_availability_detection_matches_shutil() -> Any:
    """If setup exposes a uv detection helper, it must agree with shutil.which.

    Covers this by importing uv_management indirectly — if the helper is
    reachable we verify parity; otherwise we skip.
    """
    try:
        from gnn.setup.uv_management import check_uv_availability
    except ImportError:
        raise AssertionError(
            "gnn.setup.uv_management.check_uv_availability not exposed"
        )
    result = check_uv_availability()
    # Result shape varies (bool or dict); just check it matches shutil.which.
    has_uv = shutil.which("uv") is not None
    if isinstance(result, bool):
        assert result == has_uv
    elif isinstance(result, dict):
        reported = result.get("available") or result.get("found")
        if reported is not None:
            assert bool(reported) == has_uv


def test_mcp_package_spec_validation_rejects_malicious_names() -> None:
    """S2-4: install_uv_dependency_mcp must reject injection payloads."""
    from gnn.setup.mcp import install_uv_dependency_mcp

    for malicious in [
        "--version",  # flag injection
        "requests && touch /tmp/pwned",  # shell metacharacters
        "ruff; rm -rf /",  # command chaining
        "pkg$(reboot)",  # command substitution
        "pkg\n--extras",  # newline smuggling
        "requests https://evil.example.com",  # whitespace + url
    ]:
        result = install_uv_dependency_mcp(malicious)
        assert result["success"] is False, f"{malicious!r} was not rejected"
        assert "Invalid package specification" in result["message"]


def test_mcp_package_spec_validation_accepts_normal_names() -> None:
    """Legitimate PEP 508 specs survive validation without invoking uv."""
    from gnn.setup.mcp import _validate_uv_package_spec

    for spec in ["requests", "ruff>=0.5", "gnn-pipeline==1.2.3", "fastapi[dev]"]:
        assert _validate_uv_package_spec(spec)


def test_sync_uv_dependencies_mcp_rejects_out_of_repo_paths(
    tmp_path: Path,
) -> None:
    """S2-4: sync cwd must stay inside the repository boundary."""
    from gnn.setup.mcp import sync_uv_dependencies_mcp

    result = sync_uv_dependencies_mcp(str(tmp_path))
    assert result["success"] is False
    assert "Invalid project directory" in result["message"]


@pytest.mark.needs_pkl
def test_pkl_eval_degrades_to_none_on_hostile_content() -> None:
    """S2-31: hostile pkl content must not raise; it degrades to None."""
    from gnn.parsers.schema_parser import PKLParser

    parser = PKLParser()
    hostile = (
        'amends "Pkl"; import "pkl:base" '
        'output { text = read("file:///etc/passwd") } '
        "while (true) {}"
    )
    outcome = parser._eval_pkl_native(hostile)
    assert outcome is None
