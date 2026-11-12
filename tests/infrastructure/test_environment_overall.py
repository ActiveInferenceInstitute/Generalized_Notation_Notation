"""
Test Environment Overall Tests

This file contains comprehensive tests for the environment module functionality.
"""

import subprocess
import sys
from collections.abc import Callable
from statistics import median
from typing import Any

import pytest


class TestEnvironmentModuleComprehensive:
    """Comprehensive tests for the environment module."""

    @pytest.mark.unit
    def test_environment_module_imports(self) -> None:
        """Test that environment module can be imported."""
        import gnn.setup as setup

        assert hasattr(setup, "__version__")
        assert hasattr(setup, "EnvironmentManager")
        assert hasattr(setup, "VirtualEnvironment")
        assert hasattr(setup, "get_environment_info")

    @pytest.mark.unit
    def test_environment_manager_instantiation(self) -> None:
        """Test EnvironmentManager class instantiation."""
        from gnn.setup import EnvironmentManager

        manager = EnvironmentManager()
        assert manager is not None
        assert hasattr(manager, "setup_environment")
        assert hasattr(manager, "validate_environment")

    @pytest.mark.unit
    def test_virtual_environment_instantiation(self) -> None:
        """Test VirtualEnvironment class instantiation."""
        from gnn.setup import VirtualEnvironment

        venv = VirtualEnvironment("test_env")
        assert venv is not None
        assert hasattr(venv, "create")
        assert hasattr(venv, "activate")

    @pytest.mark.unit
    def test_environment_module_info(self) -> None:
        """Test environment module information retrieval."""
        from gnn.setup import get_module_info

        info = get_module_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "description" in info
        assert "environment_types" in info

    @pytest.mark.unit
    def test_environment_validation(self) -> None:
        """Test environment validation functionality."""
        from gnn.setup import validate_environment

        result = validate_environment()
        assert isinstance(result, dict)
        assert "python_version" in result
        assert "dependencies" in result


class TestEnvironmentFunctionality:
    """Tests for environment functionality."""

    @pytest.mark.unit
    def test_environment_setup(self, isolated_temp_dir: Any) -> None:
        """Test environment setup functionality."""
        from gnn.setup import EnvironmentManager

        manager = EnvironmentManager()
        result = manager.setup_environment(isolated_temp_dir)
        assert result is not None

    @pytest.mark.slow
    @pytest.mark.unit
    def test_dependency_installation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dependency installation must request the maintained dev test extras."""
        from gnn.setup import install_dependencies, uv_management

        commands: list[list[str]] = []

        def fake_run(
            command: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            commands.append([str(part) for part in command])
            return subprocess.CompletedProcess(
                command, 0, stdout="Python 3.12.0", stderr=""
            )

        monkeypatch.setattr(uv_management.subprocess, "run", fake_run)
        monkeypatch.setattr(
            uv_management,
            "get_installed_package_versions",
            lambda verbose=False: {},
        )

        result = install_dependencies(verbose=False, dev=True)
        assert isinstance(result, bool)
        assert result is True
        assert any(command[1:4] == ["sync", "--extra", "dev"] for command in commands)

    @pytest.mark.unit
    def test_python_version_check(self) -> None:
        """Test Python version checking."""
        from gnn.setup import check_python_version

        result = check_python_version()
        assert isinstance(result, bool)


class TestEnvironmentIntegration:
    """Integration tests for environment module."""

    @pytest.mark.integration
    def test_environment_pipeline_integration(self, isolated_temp_dir: Any) -> None:
        """Test environment module integration with pipeline."""
        from gnn.setup import EnvironmentManager

        manager = EnvironmentManager()
        result = manager.validate_environment()
        assert result is not None
        assert isinstance(result, dict)

    @pytest.mark.integration
    def test_environment_mcp_integration(self) -> None:
        """Test environment MCP integration."""
        from gnn.setup.mcp import register_tools

        assert callable(register_tools)


def test_environment_module_completeness() -> None:
    """Test that environment module has all required components."""
    required_components: list[Any] = [
        "EnvironmentManager",
        "VirtualEnvironment",
        "get_module_info",
        "validate_environment",
        "install_dependencies",
        "check_python_version",
    ]
    try:
        import gnn.setup as setup

        for component in required_components:
            assert hasattr(setup, component), f"Missing component: {component}"
    except ImportError as exc:
        raise AssertionError("Environment module must be importable") from exc


@pytest.mark.slow
def test_environment_module_performance(
    record_property: Callable[[str, object], None],
) -> None:
    """Bound validation overhead against the same real probes on this host."""
    import time

    from gnn.setup import EnvironmentManager, uv_management
    from gnn.setup.constants import PROJECT_ROOT, VENV_PATH, VENV_PYTHON
    from gnn.utils.runtime_safety.jax_stack_validation import (
        run_jax_stack_probe_subprocess,
    )

    manager = EnvironmentManager()
    reference_samples: list[float] = []
    validation_samples: list[float] = []
    # Interleave three independent probe passes and full validations so the
    # medians tolerate one cold start or scheduling outlier under xdist.
    for _ in range(3):
        start_time = time.perf_counter()
        assert uv_management.check_system_requirements()
        assert VENV_PATH.exists()
        assert uv_management.get_installed_package_versions()
        ok_jax, diagnostic = run_jax_stack_probe_subprocess(VENV_PYTHON, PROJECT_ROOT)
        assert ok_jax, diagnostic
        reference_samples.append(time.perf_counter() - start_time)

        start_time = time.perf_counter()
        result = manager.validate_environment()
        validation_samples.append(time.perf_counter() - start_time)
        for check in (
            "system_requirements",
            "uv_environment",
            "dependencies",
            "jax_installation",
            "jax_stack_functional",
            "overall_status",
        ):
            assert result[check] is True, f"{check} failed: {result}"
        assert result["python_version"] == ".".join(map(str, sys.version_info[:3]))

    reference_time = median(reference_samples)
    processing_time = median(validation_samples)
    # Independent calibration scales with host/JAX speed without allowing the
    # validation under test to inflate its own budget by repeating work.
    budget = 1.5 * reference_time + 1.0
    record_property("reference_elapsed_seconds", reference_time)
    record_property("validation_elapsed_seconds", processing_time)
    record_property("validation_budget_seconds", budget)
    record_property("reference_samples_seconds", reference_samples)
    record_property("validation_samples_seconds", validation_samples)
    assert processing_time <= budget, (
        f"Full validation median {processing_time:.3f}s exceeds {budget:.3f}s "
        f"(1.5 x independent probe median {reference_time:.3f}s + 1s); "
        f"reference samples={reference_samples}, "
        f"validation samples={validation_samples}"
    )
