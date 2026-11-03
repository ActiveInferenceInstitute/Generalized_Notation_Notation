"""
Test configuration and fixtures for the GNN Processing Pipeline.

Phase 7: conftest was reduced from 545 lines to a focused set of project-
specific fixtures. Removed:
  - pathlib._local PurePosixPath patch (unused, speculative)
  - safe_to_fail marker auto-apply (Phase 7.1)
  - Unused fixtures: full_pipeline_environment, simulate_failures,
    capture_logs, pipeline_arguments
  - RealRenderModule recovery fallback (in-tree imports always succeed)

S2-17: ``pytest_collection_modifyitems`` also auto-tags ``toolchain`` on
every ``needs_*`` item (default-suite deselection) and skips ``needs_*``
items whose toolchain probe (``tests/helpers/toolchain_probes.py``) fails.
"""

from __future__ import annotations

import sys
import tempfile

# Make "tests.*" an importable alias for the tests/ directory so that
# tests which do `from tests.conftest import X` continue to resolve.
import types as _types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, cast

import pytest

_pkg = _types.ModuleType("tests")
_pkg.__path__ = [str(Path(__file__).parent)]
sys.modules.setdefault("tests", _pkg)
sys.modules["tests.conftest"] = sys.modules[__name__]

from tests.helpers.gnn_samples import (  # noqa: E402 - needs the alias above
    SAMPLE_GNN_CONTENT,
    write_sample_gnn_markdown,
)
from tests.helpers.mcp_stubs import MCPTools  # noqa: E402 - needs the alias above


@pytest.hookimpl(wrapper=True)
def pytest_collection_modifyitems(config: Any, items: list) -> None:
    """Tag slow tests, auto-tag ``toolchain``, and gate ``needs_*`` items.

    Runs as a hook wrapper: the pre-``yield`` phase runs before every other
    implementation, so auto-applied markers are visible to ``-m`` deselection
    (implemented by the core as a ``trylast`` hookimpl); the post-``yield``
    phase runs after deselection, so availability probes below only fire for
    tests that actually survive the default-suite filters.
    """
    from tests.helpers.toolchain_probes import MARKER_IMPLICATIONS, TOOLCHAIN_MARKERS

    for item in items:
        if any(m.name == "slow" for m in item.iter_markers()):
            item.add_marker(pytest.mark.performance)
        needs = {name for name in TOOLCHAIN_MARKERS if item.get_closest_marker(name)}
        if not needs:
            continue
        for name in needs:
            implied = MARKER_IMPLICATIONS.get(name)
            if implied is not None and not item.get_closest_marker(implied):
                item.add_marker(getattr(pytest.mark, implied))
        item.add_marker(pytest.mark.toolchain)

    result = yield

    # Post-deselection: only survivors reach the (cached) availability probes.
    for item in items:
        for name, (probe, reason) in TOOLCHAIN_MARKERS.items():
            if item.get_closest_marker(name) and not probe():
                item.add_marker(pytest.mark.skip(reason=f"{name}: {reason}"))
    return result


# -----------------------------------------------------------------------------
# Session-level fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="function")
def _auto_seed_rng() -> None:
    """Ensure every test function starts with deterministic random state."""
    import numpy as np

    np.random.seed(0)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the project root."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def src_dir() -> Path:
    """Absolute path to the src/ directory."""
    return Path(__file__).parent.parent / "src"


@pytest.fixture(scope="session")
def test_dir() -> Path:
    """Absolute path to the tests/ directory."""
    return Path(__file__).parent


# -----------------------------------------------------------------------------
# Filesystem fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def safe_filesystem() -> Generator[Any, None, None]:
    """A scratch filesystem under a fresh tempdir that cleans itself up."""
    temp_dir = Path(tempfile.mkdtemp())

    class SafeFileSystem:
        def __init__(self, base: Path) -> None:
            self.temp_dir = base
            self.created_files: list[Path] = []
            self.created_dirs: list[Path] = []

        def create_file(self, path: Any, content: str = "") -> Path:
            full = self.temp_dir / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content)
            self.created_files.append(full)
            return cast("Path", full)

        def create_dir(self, path: Any) -> Path:
            full = self.temp_dir / path
            full.mkdir(parents=True, exist_ok=True)
            self.created_dirs.append(full)
            return cast("Path", full)

        def cleanup(self) -> None:
            import shutil

            if self.temp_dir.exists():
                try:
                    shutil.rmtree(self.temp_dir)
                except OSError:
                    pass

    fs = SafeFileSystem(temp_dir)
    yield fs
    fs.cleanup()


@pytest.fixture
def isolated_temp_dir() -> Generator[Path, None, None]:
    """A throwaway temp directory with automatic cleanup."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        yield temp_dir
    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_directories(tmp_path: Path) -> Dict[str, Path]:
    """Standard input/output/temp directory set anchored under tmp_path."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    for d in (input_dir, output_dir, temp_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "temp_dir": temp_dir,
        "root": tmp_path,
    }


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Temporary output directory for visualization tests."""
    base = Path(tempfile.mkdtemp())
    directory = base / "viz_output"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        yield directory
    finally:
        import shutil

        shutil.rmtree(base, ignore_errors=True)


# -----------------------------------------------------------------------------
# GNN sample content fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_gnn_files(safe_filesystem: Any) -> Dict[str, Path]:
    """Pair of on-disk GNN files sharing a minimal coherent POMDP schema."""
    files: dict[str, Any] = {
        "simple": safe_filesystem.create_file("simple.gnn", SAMPLE_GNN_CONTENT),
        "second": safe_filesystem.create_file(
            "second.gnn",
            SAMPLE_GNN_CONTENT.replace("test_model", "second_model"),
        ),
    }
    return files


@contextmanager
def _sample_gnn_markdown_file(relative: str) -> Generator[Path, None, None]:
    """Yield an on-disk canonical sample GNN file at ``relative`` under a fresh tempdir, cleaning up on exit."""
    base = Path(tempfile.mkdtemp())
    path = base / relative
    write_sample_gnn_markdown(path)
    try:
        yield path
    finally:
        import shutil

        shutil.rmtree(base, ignore_errors=True)


# Both fixtures below exist because the pipeline exposes the same canonical
# sample content in two nesting shapes (a samples/ subdirectory vs. a bare
# file), consumed by different modules. They stay separate fixtures because
# merging them into one parameterized fixture would rename what ~20 existing
# consumers request.


@pytest.fixture
def test_data_dir() -> Generator[Path, None, None]:
    """Directory containing a sample GNN file at samples/actinf_pomdp_agent.md."""
    with _sample_gnn_markdown_file("samples/actinf_pomdp_agent.md") as path:
        yield path.parent


@pytest.fixture
def sample_gnn_file() -> Generator[Path, None, None]:
    """Path to a single on-disk sample GNN markdown file."""
    with _sample_gnn_markdown_file("actinf_pomdp_agent.md") as path:
        yield path


@pytest.fixture
def sample_gnn_spec() -> Dict[str, Any]:
    """In-memory GNN spec dict used by render tests."""
    return {
        "model_name": "actinf_pomdp_agent",
        "name": "actinf_pomdp_agent",
        "states": ["s0", "s1"],
        "observations": ["o0", "o1"],
        "actions": ["stay"],
        "num_states": 2,
        "num_observations": 2,
        "num_actions": 1,
        "time_horizon": 4,
        "seed": 42,
        "b_tensor_order": "next_state_previous_state_action",
        "initialparameterization": {
            "A": [[0.9, 0.1], [0.1, 0.9]],
            "B": [[0.9, 0.2], [0.1, 0.8]],
            "C": [0.0, 1.0],
            "D": [1.0, 0.0],
            "E": [1.0],
        },
        "model_parameters": {
            "num_hidden_states": 2,
            "num_obs": 2,
            "num_actions": 1,
            "b_tensor_order": "next_state_previous_state_action",
        },
        "connections": [
            {"source": "D", "relation": ">", "target": "s"},
            {"source": "s", "relation": "-", "target": "A"},
            {"source": "s", "relation": ">", "target": "s_prime"},
            {"source": "A", "relation": "-", "target": "o"},
            {"source": "s", "relation": "-", "target": "B"},
            {"source": "C", "relation": ">", "target": "G"},
            {"source": "E", "relation": ">", "target": "π"},
            {"source": "G", "relation": ">", "target": "π"},
            {"source": "π", "relation": ">", "target": "u"},
            {"source": "B", "relation": ">", "target": "u"},
            {"source": "u", "relation": ">", "target": "s_prime"},
        ],
        "parameters": {
            "A": [[0.9, 0.1], [0.1, 0.9]],
            "B": [[0.9, 0.2], [0.1, 0.8]],
            "C": [0.0, 1.0],
            "D": [1.0, 0.0],
            "E": [1.0],
        },
    }


@pytest.fixture
def sample_markdown() -> str:
    """Minimal GNN markdown source for parser tests."""
    return (
        "# TestModel\n\n"
        "## ModelName\nTestModel\n\n"
        "## StateSpaceBlock\ns[2,1,type=int]\n\n"
        "## Connections\ns->o\n\n"
        "## InitialParameterization\nA={(1,0),(0,1)}\n"
    )


@pytest.fixture
def sample_scala() -> str:
    """Minimal Scala source snippet for parser tests."""
    return (
        "object MyModel {\n"
        "  val a_m: Matrix(Fin 2, Fin 3) = ???\n"
        "  val b_f: Matrix(Fin 2, Fin 2) = ???\n"
        "  // EFE = G + F\n"
        "}\n"
    )


@pytest.fixture
def comprehensive_test_data(isolated_temp_dir: Path) -> Dict[str, Any]:
    """Consolidated test-data bundle used by integration tests."""
    return {
        "temp_dir": isolated_temp_dir,
        "gnn_files": {
            "simple": isolated_temp_dir / "simple.gnn",
            "complex": isolated_temp_dir / "complex.gnn",
        },
        "output_dir": isolated_temp_dir / "output",
        "config": {"test_mode": True, "safe_mode": True},
    }


# -----------------------------------------------------------------------------
# Render + MCP test helpers
# -----------------------------------------------------------------------------


class _RealRenderModule:
    """Thin adapter used by render integration tests — delegates to
    ``render.processor.render_gnn_spec``. Phase 7: fallback chain removed."""

    def render_gnn_spec(self, spec: Any, target: str, outdir: Any) -> Any:
        from gnn.render.processor import render_gnn_spec

        return render_gnn_spec(spec, target, outdir)


@pytest.fixture
def test_render_module() -> _RealRenderModule:
    return _RealRenderModule()


@pytest.fixture
def test_mcp_tools() -> MCPTools:
    return MCPTools()
