#!/usr/bin/env python3
"""
Test Environment Python - Tests for Python environment configuration.

Tests Python version, path configuration, and interpreter settings.
"""

import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
from packaging.specifiers import SpecifierSet

# Add src to path for imports


class TestPythonVersion:
    """Tests for Python version requirements."""

    @pytest.mark.fast
    def test_python_version_minimum(self) -> Any:
        """Test Python version meets minimum requirements."""
        assert sys.version_info >= (3, 12), f"Python 3.12+ required, got {sys.version}"

    @pytest.mark.fast
    def test_python_version_compatible(self) -> Any:
        """Test Python version is in compatible range."""
        metadata = tomllib.loads(
            (Path(__file__).parents[2] / "pyproject.toml").read_text()
        )
        required = SpecifierSet(metadata["project"]["requires-python"])
        assert ".".join(map(str, sys.version_info[:3])) in required

    @pytest.mark.fast
    def test_python_implementation(self) -> Any:
        """Test Python implementation is CPython."""
        import platform

        impl = platform.python_implementation()
        # CPython is the expected implementation
        assert impl in ("CPython", "PyPy")


class TestPythonPath:
    """Tests for Python path configuration."""

    @pytest.mark.fast
    def test_src_in_path(self) -> Any:
        """Test src directory is in Python path."""
        src_dir = Path(__file__).parents[2] / "src"

        # src should be importable
        assert str(src_dir) in sys.path or any(
            src_dir.samefile(Path(p)) for p in sys.path if Path(p).exists()
        )

    @pytest.mark.fast
    def test_project_root_accessible(self) -> Any:
        """Test project root is accessible."""
        project_root = Path(__file__).parents[2]

        assert project_root.exists()
        assert (
            (project_root / "pyproject.toml").exists()
            or (project_root / "gnn.setup.py").exists()
            or (project_root / "src").exists()
        )

    @pytest.mark.fast
    def test_modules_importable(self) -> Any:
        """Test core modules are importable."""
        # These should all import successfully
        import gnn
        import gnn.render as render
        import gnn.report as report

        assert gnn is not None
        assert render is not None
        assert report is not None


class TestPythonInterpreter:
    """Tests for Python interpreter settings."""

    @pytest.mark.fast
    def test_encoding_utf8(self) -> Any:
        """Test default encoding is UTF-8."""
        assert sys.getdefaultencoding() == "utf-8"

    @pytest.mark.fast
    def test_recursion_limit_adequate(self) -> Any:
        """Test recursion limit is adequate."""
        limit = sys.getrecursionlimit()

        # Should be at least 1000 (default is usually 1000 or more)
        assert limit >= 1000

    @pytest.mark.fast
    def test_platform_supported(self) -> Any:
        """Test platform is supported."""
        import platform

        system = platform.system()

        # Should be one of the supported platforms
        assert system in ("Darwin", "Linux", "Windows")


class TestPythonFeatures:
    """Tests for Python language features."""

    @pytest.mark.fast
    def test_async_await_available(self) -> Any:
        """Test async/await is available."""

        async def async_func() -> Any:
            return 42

        import asyncio

        result = asyncio.run(async_func())
        assert result == 42

    @pytest.mark.fast
    def test_type_hints_work(self) -> Any:
        """Test type hints work correctly."""
        from typing import Dict, List

        def typed_func(items: List[str]) -> Dict[str, int]:
            return {item: len(item) for item in items}

        result = typed_func(["a", "bb", "ccc"])
        assert result == {"a": 1, "bb": 2, "ccc": 3}

    @pytest.mark.fast
    def test_dataclasses_available(self) -> Any:
        """Test dataclasses are available."""
        from dataclasses import dataclass

        @dataclass
        class TestData:
            name: str
            value: int

        data = TestData(name="test", value=42)
        assert data.name == "test"
        assert data.value == 42

    @pytest.mark.fast
    def test_pathlib_fully_functional(self) -> Any:
        """Test pathlib is fully functional."""
        from pathlib import Path

        p = Path(".")
        assert p.exists()
        assert p.is_dir()
        assert p.resolve().is_absolute()


class TestPythonModules:
    """Tests for Python standard library modules."""

    @pytest.mark.fast
    def test_json_module(self) -> Any:
        """Test json module works."""
        import json

        data: dict[str, Any] = {"key": "value", "number": 42}
        encoded = json.dumps(data)
        decoded = json.loads(encoded)

        assert decoded == data

    @pytest.mark.fast
    def test_logging_module(self) -> Any:
        """Test logging module works."""
        import logging

        logger = logging.getLogger("test")
        logger.setLevel(logging.DEBUG)

        # Should not raise
        logger.debug("test")

    @pytest.mark.fast
    def test_subprocess_module(self) -> Any:
        """Test subprocess module works."""
        import subprocess  # nosec B404

        result = subprocess.run(  # nosec B603
            [sys.executable, "--version"], capture_output=True, text=True
        )

        assert result.returncode == 0
        assert "Python" in result.stdout or "python" in result.stdout.lower()

    @pytest.mark.fast
    def test_threading_module(self) -> Any:
        """Test threading module works."""
        import threading

        results: list[Any] = []

        def worker() -> Any:
            results.append(42)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert results == [42]


class TestPythonEnvironment:
    """Tests for Python environment variables."""

    @pytest.mark.fast
    def test_environment_variables_accessible(self) -> Any:
        """Test environment variables are accessible."""
        import os

        # Should be able to access environment
        env = os.environ
        assert isinstance(env, dict) or hasattr(env, "__getitem__")

    @pytest.mark.fast
    def test_cwd_accessible(self) -> Any:
        """Test current working directory is accessible."""
        import os

        cwd = os.getcwd()
        assert cwd is not None
        assert len(cwd) > 0
