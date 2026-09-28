"""Version consistency across the ``gnn`` package surfaces.

``pyproject.toml`` is the single source of truth; every ``__version__``
literal and the FastAPI app metadata must agree with it.
"""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_single_version() -> None:
    import gnn
    import gnn.api as api
    import gnn.cli as cli

    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        expected = tomllib.load(stream)["project"]["version"]
    assert gnn.__version__ == expected
    assert gnn.cli.__version__ == expected
    assert gnn.api.__version__ == expected
    assert gnn.api.MODULE_VERSION == expected
    assert importlib.metadata.version("generalized-notation-notation") == expected
