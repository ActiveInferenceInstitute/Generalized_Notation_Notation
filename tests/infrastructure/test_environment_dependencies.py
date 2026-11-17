"""
Test Environment Dependencies - Tests for environment dependency management.

Tests verification and management of Python package dependencies.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


class TestCoreDependencies:
    """Tests for core Python dependencies."""

    @pytest.mark.fast
    def test_numpy_available(self) -> Any:
        """Test NumPy is installed and functional."""
        import numpy as np

        arr = np.array([1, 2, 3])
        assert arr.shape == (3,)
        assert np.sum(arr) == 6

    @pytest.mark.fast
    def test_pathlib_available(self) -> Any:
        """Test pathlib is available (stdlib)."""

        p = Path(".")
        assert p.exists()

    @pytest.mark.fast
    def test_json_available(self) -> Any:
        """Test json is available (stdlib)."""
        import json

        data: dict[str, Any] = {"key": "value"}
        encoded = json.dumps(data)
        decoded = json.loads(encoded)
        assert decoded == data

    @pytest.mark.fast
    def test_logging_available(self) -> Any:
        """Test logging is available (stdlib)."""
        import logging

        logger = logging.getLogger("test")
        assert logger is not None


class TestVisualizationDependencies:
    """Tests for visualization dependencies."""

    @pytest.mark.fast
    def test_matplotlib_available(self) -> Any:
        """Test matplotlib is installed."""
        import matplotlib
        import matplotlib.pyplot as plt

        assert hasattr(matplotlib, "__version__"), "matplotlib missing __version__"
        assert hasattr(plt, "figure"), "matplotlib.pyplot missing figure function"

    @pytest.mark.fast
    def test_seaborn_available(self) -> Any:
        """Test seaborn is installed."""
        import seaborn as sns

        assert hasattr(sns, "__version__"), "seaborn missing __version__"
        assert hasattr(sns, "set_theme"), "seaborn missing set_theme function"


class TestAudioDependencies:
    """Tests for audio processing dependencies."""

    @pytest.mark.fast
    def test_audio_backends(self) -> Any:
        """Test audio backend availability."""
        from gnn.audio import check_audio_backends

        backends = check_audio_backends()
        assert isinstance(backends, dict)
        assert "numpy" in backends
        assert backends["numpy"]["available"] is True

    @pytest.mark.fast
    def test_librosa_availability(self) -> Any:
        """Test librosa availability or structured optional-backend reporting."""
        from gnn.audio import check_audio_backends

        backends = check_audio_backends()
        if importlib.util.find_spec("librosa") is None:
            assert backends["librosa"] == {"available": False, "version": None}
            return
        import librosa

        assert hasattr(librosa, "__version__")
        assert backends["librosa"]["available"] is True

    @pytest.mark.fast
    def test_soundfile_availability(self) -> Any:
        """Test soundfile availability or structured optional-backend reporting."""
        from gnn.audio import check_audio_backends

        backends = check_audio_backends()
        if importlib.util.find_spec("soundfile") is None:
            assert backends["soundfile"] == {"available": False, "version": None}
            return
        import soundfile

        assert hasattr(soundfile, "__version__")
        assert backends["soundfile"]["available"] is True


class TestMLDependencies:
    """Tests for machine learning dependencies."""

    @pytest.mark.fast
    def test_scipy_available(self) -> Any:
        """Test scipy is installed."""
        import scipy

        assert hasattr(scipy, "__version__")

    @pytest.mark.fast
    def test_pymdp_available(self) -> Any:
        """Test pymdp is installed for Active Inference."""
        try:
            try:
                from pymdp.agent import Agent

                assert Agent is not None
                return
            except ImportError:
                pass
            import pymdp

            assert (
                hasattr(pymdp, "MDP")
                or hasattr(pymdp, "mdp")
                or hasattr(pymdp, "__version__")
            ), "pymdp missing expected attributes"
        except ImportError:
            pytest.fail("pymdp is a required default execution dependency")


class TestDependencyVersions:
    """Tests for dependency version requirements."""

    @pytest.mark.fast
    def test_python_version(self) -> Any:
        """Test Python version meets requirements."""

        assert sys.version_info >= (3, 12)

    @pytest.mark.fast
    def test_numpy_version(self) -> Any:
        """Test NumPy version."""
        import numpy as np

        version_parts = np.__version__.split(".")
        major = int(version_parts[0])
        assert major >= 1


class TestDependencyConflicts:
    """Tests for dependency conflict detection."""

    @pytest.mark.fast
    def test_no_import_conflicts(self) -> Any:
        """Test core imports don't conflict."""
        import json
        import logging

        import numpy as np

        assert json.dumps({"test": 1}) == '{"test": 1}'
        assert logging.getLogger("test") is not None
        assert Path(".").exists()
        assert np.array([1, 2, 3]).sum() == 6

    @pytest.mark.fast
    def test_visualization_imports_compatible(self) -> Any:
        """Test visualization imports are compatible."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        line_objs = ax.plot([1, 2, 3])
        assert len(line_objs) == 1, "Expected one line object from plot"
        assert fig is not None, "Figure should be created"
        plt.close(fig)


class TestOptionalDependencies:
    """Tests for optional dependency handling."""

    @pytest.mark.fast
    def test_optional_deps_graceful_import(self) -> Any:
        """Test optional dependencies fail gracefully."""
        assert importlib.util.find_spec("nonexistent_package") is None

    @pytest.mark.fast
    def test_feature_flags_reflect_deps(self) -> Any:
        """Test feature flags reflect dependency availability."""
        from gnn.audio import FEATURES, check_audio_backends

        check_audio_backends()
        assert isinstance(FEATURES, dict)


class TestDependencyDiscovery:
    """Tests for dependency discovery functionality."""

    @pytest.mark.fast
    def test_list_installed_packages(self) -> Any:
        """Test we can list installed packages."""
        from importlib.metadata import distributions

        packages = [d.metadata["Name"] for d in distributions()]
        assert len(packages) > 0

    @pytest.mark.fast
    def test_pyproject_toml_exists(self) -> Any:
        """Test pyproject.toml dependency file exists (uv-managed)."""

        pyproject = Path(__file__).parents[2] / "pyproject.toml"
        assert pyproject.exists(), (
            "pyproject.toml must exist for uv-managed dependencies"
        )
