"""Registration contract for the MCP meta-tools (finding F1).

The eight meta tools defined in ``src/gnn/mcp/meta_mcp.py`` must be
registered during module discovery (alongside ``sympy_mcp``), so the
documented MCP surface matches the live registry. These tests run the real
full discovery via ``gnn.mcp.mcp.initialize``; the discovery is scoped to
the module so it runs once.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.unit

from gnn.mcp.mcp import MCP, initialize
from gnn.mcp.npx_inspector import GNN_PROJECT_ROOT

META_TOOL_NAMES = (
    "get_mcp_server_capabilities",
    "get_mcp_server_status",
    "get_mcp_server_auth_status",
    "get_mcp_server_encryption_status",
    "get_mcp_module_info",
    "get_mcp_tool_categories",
    "get_mcp_performance_metrics",
    "get_mcp_diagnostics",
)


@pytest.fixture(scope="module")
def mcp_instance() -> MCP:
    # This suite promises a complete module census, independent of scoped callers.
    mcp, _, _ = initialize(
        halt_on_missing_sdk=False, force_proceed_flag=True, force_refresh=True
    )
    return mcp


class TestMetaToolsRegistration:
    """All eight documented meta tools are live on the registry."""

    def test_all_meta_tools_registered(self, mcp_instance: MCP) -> None:
        assert len(META_TOOL_NAMES) == 8
        for name in META_TOOL_NAMES:
            assert name in mcp_instance.tools, f"missing meta tool: {name}"
            assert mcp_instance.tools[name].module == "meta"

    def test_meta_mcp_module_entry_loaded(self, mcp_instance: MCP) -> None:
        module_info = mcp_instance.modules["meta_mcp"]
        assert module_info.status == "loaded"
        assert module_info.tools_count == 8


class TestMetaToolBehavior:
    """The registered meta tools execute against the real instance."""

    def test_get_mcp_server_capabilities(self, mcp_instance: MCP) -> None:
        result: dict[str, Any] = mcp_instance.execute_tool(
            "get_mcp_server_capabilities", {}
        )
        assert isinstance(result, dict) and result
        assert {"tools", "resources", "validation_mode", "server"} <= set(result)
        assert isinstance(result["tools"], list) and result["tools"]

    def test_get_mcp_tool_categories(self, mcp_instance: MCP) -> None:
        result: dict[str, Any] = mcp_instance.execute_tool(
            "get_mcp_tool_categories", {}
        )
        assert result["success"] is True
        assert isinstance(result["categories"], dict)
        assert result["total_tools"] == len(mcp_instance.tools)
        assert result["total_categories"] >= 1

    def test_get_mcp_diagnostics(self, mcp_instance: MCP) -> None:
        result: dict[str, Any] = mcp_instance.execute_tool("get_mcp_diagnostics", {})
        assert "overall_health" in result
        assert result["overall_health"] in {"healthy", "degraded", "unhealthy"}

    def test_get_mcp_module_info_audio(self, mcp_instance: MCP) -> None:
        result: dict[str, Any] = mcp_instance.execute_tool(
            "get_mcp_module_info", {"module_name": "audio"}
        )
        assert result["success"] is True
        assert result["module_name"] == "audio"
        assert result["full_name"] == mcp_instance.modules["audio"].name
        assert result["status"] == mcp_instance.modules["audio"].status
        assert result["tools_count"] == len(result["tools"])


class TestInspectorCliPathPinning:
    """The inspector's project root actually contains the MCP CLI module."""

    def test_cli_module_exists_under_project_root(self) -> None:
        assert (GNN_PROJECT_ROOT / "gnn" / "mcp" / "cli.py").is_file()
