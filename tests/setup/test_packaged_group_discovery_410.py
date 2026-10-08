"""Actual package metadata drives public discovery without native setup calls."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from importlib.metadata import metadata
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gnn.api.app import create_app
from gnn.cli.mcp import register_tools
from gnn.mcp.mcp import MCP
from gnn.setup import get_module_info, get_setup_options
from gnn.setup.constants import OPTIONAL_GROUPS

ROOT = Path(__file__).resolve().parents[2]


def test_packaged_groups_reach_python_cli_rest_and_registered_mcp() -> None:
    declared = set(metadata("generalized-notation-notation").get_all("Provides-Extra"))
    assert set(OPTIONAL_GROUPS) == declared
    assert "geo-infer" in OPTIONAL_GROUPS
    assert get_module_info()["optional_groups"] == OPTIONAL_GROUPS
    assert get_setup_options()["optional_groups"] == OPTIONAL_GROUPS
    cli = subprocess.run(
        [sys.executable, "-m", "gnn.cli", "preflight", "--json"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert cli.returncode in (0, 1, 2)
    assert json.loads(cli.stdout)["data"]["optional_groups"] == OPTIONAL_GROUPS
    response = TestClient(create_app()).post("/api/v1/preflight", json={})
    payload = response.json()
    assert response.status_code in (200, 400)
    discovery = (
        payload["data"] if response.status_code == 200 else payload["error"]["details"]
    )
    assert discovery["optional_groups"] == OPTIONAL_GROUPS
    registry = MCP(
        enable_caching=False, enable_rate_limiting=False, strict_validation=True
    )
    register_tools(registry)
    result = registry.execute_tool("cli.preflight", {})
    assert result["optional_groups"] == OPTIONAL_GROUPS
    # Report native/environment failures honestly, independently of discovery.
    assert result["success"] is (result["checks_failed"] == 0)


def _isolated_discovery(scratch: Path) -> subprocess.CompletedProcess:
    code = (
        "import json,sys; "
        f"sys.path[:0] = [{str(scratch)!r}, {str(ROOT / 'src')!r}]; "
        "from gnn.setup.constants import OPTIONAL_GROUPS; "
        "print(json.dumps({'groups':OPTIONAL_GROUPS, 'modules':list(sys.modules)}))"
    )
    return subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=scratch,
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_absent_distribution_metadata_uses_the_actual_checkout_manifest(
    tmp_path: Path,
) -> None:
    result = _isolated_discovery(tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert set(payload["groups"]) == set(project["optional-dependencies"])
    assert not {"jax", "torch", "numpyro", "pymdp"} & set(payload["modules"])


@pytest.mark.parametrize(
    "extras",
    ["Provides-Extra: INVALID VALUE\n", "Provides-Extra: api\nProvides-Extra: api\n"],
)
def test_broken_installed_metadata_does_not_fall_back_to_checkout(
    tmp_path: Path, extras: str
) -> None:
    distribution = tmp_path / "generalized_notation_notation-4.0.1.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: generalized-notation-notation\nVersion: 4.0.1\n"
        + extras
    )
    result = _isolated_discovery(tmp_path)
    assert result.returncode != 0
    assert "malformed optional-group metadata" in result.stderr
