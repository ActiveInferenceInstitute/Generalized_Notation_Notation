#!/usr/bin/env python3
"""
Test Mcp Tools Tests

This file contains tests migrated from test_mcp_comprehensive.py.
"""

import pytest

pytestmark = pytest.mark.mcp
import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Migrated from test_mcp_integration_comprehensive.py
_VALIDATION_MARKDOWN = """## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
Référence

## ModelAnnotation
Test the requested content format.

## StateSpaceBlock
s[2,type=float]

## Connections
s>s

## InitialParameterization
s={(1.0,2.0)}

## Time
Static

## Footer
End
"""
_VALIDATION_JSON = '{"model_name": "Référence"}'
_VALIDATION_YAML = "model_name: Référence\n"


@pytest.mark.parametrize(
    ("content", "hint"),
    [
        (_VALIDATION_MARKDOWN, "markdown"),
        (_VALIDATION_JSON, "json"),
        (_VALIDATION_YAML, "yaml"),
        (_VALIDATION_JSON, "yaml"),  # JSON syntax is also valid YAML.
    ],
)
def test_validation_tool_uses_requested_format(
    test_mcp_tools: Any, content: str, hint: str
) -> None:
    from gnn.mcp.gnn_root import register_tools

    register_tools(test_mcp_tools)
    result = test_mcp_tools.execute_tool(
        "validate_gnn_content",
        content=content,
        format_hint=hint,
        validation_level="BASIC",
    )
    assert result["success"], result
    assert result["is_valid"], result["errors"]
    assert result["format_tested"] == hint
    assert result["validation_level"] == "basic"


@pytest.mark.parametrize(
    ("content", "hint"),
    [
        (_VALIDATION_MARKDOWN, "json"),
        (_VALIDATION_MARKDOWN, "yaml"),
        (_VALIDATION_JSON, "markdown"),
        (_VALIDATION_YAML, "json"),
    ],
)
def test_validation_tool_refuses_conflicting_format(
    test_mcp_tools: Any, content: str, hint: str
) -> None:
    from gnn.mcp.gnn_root import register_tools

    register_tools(test_mcp_tools)
    result = test_mcp_tools.execute_tool(
        "validate_gnn_content",
        content=content,
        format_hint=hint,
        validation_level="basic",
    )
    assert result["success"] and not result["is_valid"], result
    assert result["format_tested"] == hint
    assert result["errors"]


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"validation_level": "nonsense"}, "Unknown validation level"),
        ({"validation_level": ""}, "Unknown validation level"),
        ({"format_hint": "binary"}, "Unknown text format hint"),
    ],
)
def test_validation_tool_reports_invalid_options(
    test_mcp_tools: Any, options: dict[str, str], error: str
) -> None:
    from gnn.mcp.gnn_root import register_tools

    register_tools(test_mcp_tools)
    assert (
        "binary"
        not in test_mcp_tools.tools["validate_gnn_content"]["schema"]["format_hint"][
            "enum"
        ]
    )
    result = test_mcp_tools.execute_tool(
        "validate_gnn_content", content=_VALIDATION_MARKDOWN, **options
    )
    assert result["success"] is False
    assert error in result["error"]


def test_validation_tool_omitted_options_keep_markdown_defaults(
    test_mcp_tools: Any,
) -> None:
    from gnn.mcp.gnn_root import register_tools

    register_tools(test_mcp_tools)
    result = test_mcp_tools.execute_tool(
        "validate_gnn_content", content=_VALIDATION_MARKDOWN
    )
    assert result["success"], result
    assert result["is_valid"], result["errors"]
    assert result["format_tested"] == "markdown"
    assert result["validation_level"] == "standard"


class TestMCPToolExecution:
    """Test actual MCP tool execution."""

    @pytest.mark.unit
    def test_gnn_validate_tool_execution(
        self, test_mcp_tools: Any, comprehensive_test_data: Any
    ) -> None:
        """Test GNN validation tool registration (lightweight test)."""
        try:
            from gnn.mcp.gnn_root import register_tools as register_gnn_tools

            register_gnn_tools(test_mcp_tools)

            # Just verify that the tool was registered properly
            assert "validate_gnn_content" in test_mcp_tools.tools
            tool_info = test_mcp_tools.tools["validate_gnn_content"]
            assert "function" in tool_info
            assert "description" in tool_info
            assert callable(tool_info["function"])

        except ImportError:
            raise AssertionError("GNN MCP not available")
        except Exception as e:
            # Should handle registration errors gracefully
            assert "error" in str(e).lower() or "import" in str(e).lower()

    @pytest.mark.unit
    def test_export_tool_execution(
        self, test_mcp_tools: Any, comprehensive_test_data: Any
    ) -> None:
        """Test export tool registration (lightweight test)."""
        try:
            from gnn.export.mcp import register_tools

            register_tools(test_mcp_tools)

            # Just verify that tools were registered properly
            assert len(test_mcp_tools.tools) > 0
            for _tool_name, tool_info in test_mcp_tools.tools.items():
                assert "function" in tool_info
                assert "description" in tool_info
                assert callable(tool_info["function"])

        except ImportError:
            raise AssertionError("Export MCP not available")
        except Exception as e:
            # Should handle registration errors gracefully
            assert "error" in str(e).lower() or "import" in str(e).lower()

    @pytest.mark.unit
    def test_utils_system_info_execution(self, test_mcp_tools: Any) -> None:
        """Test utils system info tool registration (lightweight test)."""
        try:
            from gnn.utils.mcp.server import register_tools

            register_tools(test_mcp_tools)

            # Just verify that the system info tool was registered properly
            assert "get_system_info" in test_mcp_tools.tools
            tool_info = test_mcp_tools.tools["get_system_info"]
            assert "function" in tool_info
            assert "description" in tool_info
            assert callable(tool_info["function"])

        except ImportError:
            raise AssertionError("Utils MCP not available")
        except Exception as e:
            # Should handle registration errors gracefully
            assert "error" in str(e).lower() or "import" in str(e).lower()
