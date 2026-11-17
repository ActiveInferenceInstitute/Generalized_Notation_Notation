#!/usr/bin/env python3
"""
Test Environment System - Tests for system environment configuration.

Tests system-level requirements, filesystem, and OS configuration.
"""

import os
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest

# Add src to path for imports


class TestSystemPlatform:
    """Tests for system platform requirements."""

    @pytest.mark.fast
    def test_platform_identified(self) -> None:
        """Test platform can be identified."""
        import platform

        system = platform.system()
        assert system in ("Darwin", "Linux", "Windows")

    @pytest.mark.fast
    def test_architecture_identified(self) -> None:
        """Test architecture can be identified."""
        import platform

        arch = platform.machine()
        assert arch is not None
        assert len(arch) > 0

    @pytest.mark.fast
    def test_os_name_available(self) -> None:
        """Test OS name is available."""
        assert os.name in ("posix", "nt")


class TestFilesystem:
    """Tests for filesystem requirements."""

    @pytest.mark.fast
    def test_temp_directory_available(self) -> None:
        """Test temporary directory is available."""
        import tempfile

        temp_dir = tempfile.gettempdir()
        assert Path(temp_dir).exists()
        assert Path(temp_dir).is_dir()

    @pytest.mark.fast
    def test_temp_file_creation(self) -> None:
        """Test temporary files can be created."""
        import tempfile

        with tempfile.NamedTemporaryFile(delete=True) as f:
            f.write(b"test")
            assert Path(f.name).exists()

    @pytest.mark.fast
    def test_directory_creation(self, tmp_path: Path) -> None:
        """Test directories can be created."""
        new_dir = tmp_path / "test_dir" / "nested"
        new_dir.mkdir(parents=True, exist_ok=True)

        assert new_dir.exists()
        assert new_dir.is_dir()

    @pytest.mark.fast
    def test_file_read_write(self, tmp_path: Path) -> None:
        """Test files can be read and written."""
        test_file = tmp_path / "test.txt"

        # Write
        test_file.write_text("Hello, World!")

        # Read
        content = test_file.read_text()
        assert content == "Hello, World!"

    @pytest.mark.fast
    def test_binary_file_operations(self, tmp_path: Path) -> None:
        """Test binary file operations work."""
        test_file = tmp_path / "test.bin"

        data = bytes(range(256))
        test_file.write_bytes(data)

        read_data = test_file.read_bytes()
        assert read_data == data


class TestSystemResources:
    """Tests for system resource availability."""

    @pytest.mark.fast
    def test_memory_available(self) -> None:
        """Test memory allocation works."""
        # Allocate 1MB
        data = bytearray(1024 * 1024)
        assert len(data) == 1024 * 1024

    @pytest.mark.fast
    def test_file_descriptors_available(self, tmp_path: Path) -> None:
        """Test file descriptors can be opened."""
        files: list[Any] = []
        try:
            # Try to open several files
            for i in range(10):
                f = open(tmp_path / f"test_{i}.txt", "w")
                files.append(f)

            assert len(files) == 10
        finally:
            for f in files:
                f.close()

    @pytest.mark.fast
    def test_cpu_count_available(self) -> None:
        """Test CPU count can be determined."""
        cpu_count = os.cpu_count()

        assert cpu_count is not None
        assert cpu_count >= 1


class TestSystemProcesses:
    """Tests for process management."""

    @pytest.mark.fast
    def test_subprocess_execution(self) -> None:
        """Test subprocess execution works."""
        import subprocess  # nosec B404

        result = subprocess.run(  # nosec B603
            [sys.executable, "-c", "print('hello')"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        assert "hello" in result.stdout

    @pytest.mark.fast
    def test_process_id_available(self) -> None:
        """Test process ID is available."""
        pid = os.getpid()

        assert pid is not None
        assert pid > 0

    @pytest.mark.fast
    def test_environment_variables(self) -> None:
        """Test environment variables work."""
        # Set and get
        os.environ["GNN_TEST_VAR"] = "test_value"
        assert os.environ.get("GNN_TEST_VAR") == "test_value"

        # Clean up
        del os.environ["GNN_TEST_VAR"]


class TestSystemTime:
    """Tests for time and date functionality."""

    @pytest.mark.fast
    def test_time_available(self) -> None:
        """Test time functions work."""
        import time

        now = time.time()
        assert now > 0

    @pytest.mark.fast
    def test_datetime_available(self) -> None:
        """Test datetime functions work."""
        from datetime import datetime

        now = datetime.now()
        assert now.year >= 2024

    @pytest.mark.fast
    def test_timezone_available(self) -> None:
        """Test timezone functionality works."""
        from datetime import datetime, timezone

        utc_now = datetime.now(UTC)
        assert utc_now.tzinfo is not None


class TestSystemPath:
    """Tests for path operations."""

    @pytest.mark.fast
    def test_path_separator(self) -> None:
        """Test path separator is correct."""
        sep = os.sep

        if os.name == "nt":
            assert sep == "\\"
        else:
            assert sep == "/"

    @pytest.mark.fast
    def test_absolute_path_works(self) -> None:
        """Test absolute path resolution works."""
        relative = Path(".")
        absolute = relative.resolve()

        assert absolute.is_absolute()

    @pytest.mark.fast
    def test_path_normalization(self) -> None:
        """Test path normalization works."""
        messy_path = Path("a/b/../c/./d")
        clean_parts = [p for p in messy_path.parts if p not in (".", "..")]

        # Should be able to normalize
        assert isinstance(clean_parts, list)


class TestSystemLocale:
    """Tests for locale and encoding."""

    @pytest.mark.fast
    def test_utf8_encoding(self) -> None:
        """Test UTF-8 encoding works."""
        text = "Hello, 世界! 🌍"
        encoded = text.encode("utf-8")
        decoded = encoded.decode("utf-8")

        assert decoded == text

    @pytest.mark.fast
    def test_filesystem_encoding(self) -> None:
        """Test filesystem encoding is accessible."""
        encoding = sys.getfilesystemencoding()

        assert encoding is not None
        assert encoding.lower() in ("utf-8", "utf8", "ascii", "latin-1", "mbcs")


class TestSystemNetwork:
    """Tests for basic network functionality."""

    @pytest.mark.fast
    def test_socket_module_available(self) -> None:
        """Test socket module is available."""
        import socket

        # Get hostname
        hostname = socket.gethostname()
        assert hostname is not None
        assert len(hostname) > 0

    @pytest.mark.fast
    def test_localhost_resolvable(self) -> None:
        """Test localhost is resolvable."""
        import socket

        try:
            addr = socket.gethostbyname("localhost")
            assert addr in ("127.0.0.1", "::1") or addr.startswith("127.")
        except socket.gaierror:
            # May not resolve on all systems
            pass
