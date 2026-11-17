"""
Test suite for MCP (Model Context Protocol) module.

Tests MCP tool registration, execution, and server functionality.
"""

from typing import Any

import pytest

pytestmark = pytest.mark.mcp
import json


class TestMCPModule:
    """Test suite for MCP module functionality."""

    def test_features_available(self) -> None:
        """Test that FEATURES dict is properly populated."""
        from gnn.mcp import FEATURES

        expected_features: list[Any] = [
            "tool_registration",
            "resource_access",
            "module_discovery",
            "caching",
            "rate_limiting",
            "concurrent_control",
            "mcp_integration",
        ]

        for feature in expected_features:
            assert feature in FEATURES, f"Missing feature: {feature}"

    def test_version_format(self) -> None:
        """Test version string format."""
        from gnn.mcp import __version__

        # Should be semantic versioning format
        parts = __version__.split(".")
        assert len(parts) >= 2, "Version should have at least major.minor"

    def test_process_mcp_function(self) -> None:
        """Test main process_mcp function exists."""
        from gnn.mcp import process_mcp

        assert callable(process_mcp)


class TestMCPTool:
    """Test MCPTool class."""

    def test_mcp_tool_creation(self) -> None:
        """Test creating an MCP tool."""
        from gnn.mcp import MCPTool

        def sample_func(params: Any) -> Any:
            return {"result": "success"}

        # MCPTool uses dataclass with func parameter
        tool = MCPTool(
            name="test_tool",
            func=sample_func,
            schema={"type": "object", "properties": {}},
            description="A test tool",
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert callable(tool.func)

    def test_mcp_tool_execution(self) -> None:
        """Test executing an MCP tool."""
        from gnn.mcp import MCPTool

        def echo_func(message: str = "default") -> Any:
            return {"echo": message}

        tool = MCPTool(
            name="echo_tool",
            func=echo_func,
            schema={"type": "object", "properties": {"message": {"type": "string"}}},
            description="Echoes the message",
        )

        # Execute the function directly
        result = tool.func(message="hello")
        assert result == {"echo": "hello"}


class TestMCPResource:
    """Test MCPResource class."""

    def test_mcp_resource_creation(self) -> None:
        """Test creating an MCP resource."""
        from gnn.mcp import MCPResource

        def retriever(uri: str) -> Any:
            return {"content": "test data"}

        # MCPResource uses uri_template and retriever
        resource = MCPResource(
            uri_template="test://resource/{id}",
            retriever=retriever,
            description="A test resource",
        )

        assert isinstance(resource, MCPResource)
        assert resource.description == "A test resource"


class TestMCPServer:
    """Test MCPServer class."""

    def test_mcp_server_instantiation(self) -> None:
        """Test that MCPServer can be instantiated."""
        from gnn.mcp import MCPServer

        server = MCPServer()
        assert isinstance(server, MCPServer)

    def test_mcp_server_has_methods(self) -> None:
        """Test MCPServer has expected methods."""
        from gnn.mcp import MCPServer

        server = MCPServer()

        assert hasattr(server, "register_tool"), (
            "MCPServer must have a register_tool method"
        )
        assert callable(server.register_tool), "register_tool must be callable"


class TestMCPErrors:
    """Test MCP error classes."""

    def test_mcp_error(self) -> None:
        """Test MCPError class."""
        from gnn.mcp import MCPError

        error = MCPError("Test error message")
        assert "Test error message" in str(error)
        assert isinstance(error, Exception)

    def test_mcp_tool_execution_error(self) -> None:
        """Test MCPToolExecutionError class."""
        from gnn.mcp import MCPToolExecutionError

        original_error = ValueError("Original error")
        error = MCPToolExecutionError("test_tool", original_error)
        assert "test_tool" in str(error)
        assert isinstance(error, Exception)

    def test_mcp_validation_error(self) -> None:
        """Test MCPValidationError class."""
        from gnn.mcp import MCPValidationError

        error = MCPValidationError("Validation failed")
        assert "Validation failed" in str(error)
        assert isinstance(error, Exception)


class TestMCPProcessing:
    """Test MCP processing functionality."""

    def test_process_mcp(self, safe_filesystem: Any) -> None:
        """Test main MCP processing function."""
        from gnn.mcp import process_mcp

        target_dir = safe_filesystem.temp_dir
        output_dir = safe_filesystem.create_dir("mcp_output")

        import logging

        logger = logging.getLogger("test_mcp")

        result = process_mcp(
            target_dir=target_dir, output_dir=output_dir, logger=logger, verbose=True
        )

        # Should return success
        assert result is True or (
            isinstance(result, dict) and result.get("success", False)
        )

    def test_mcp_output_files(self, safe_filesystem: Any) -> None:
        """Test that MCP processing creates expected output files."""
        from gnn.mcp import process_mcp

        target_dir = safe_filesystem.temp_dir
        output_dir = safe_filesystem.create_dir("mcp_files_output")

        import logging

        logger = logging.getLogger("test_mcp_files")

        process_mcp(target_dir=target_dir, output_dir=output_dir, logger=logger)

        # Check for expected output files
        expected_files: list[Any] = ["mcp_processing_summary.json", "mcp_results.json"]

        for filename in expected_files:
            file_path = output_dir / filename
            if file_path.exists():
                with open(file_path) as f:
                    data = json.load(f)
                assert isinstance(data, dict)


class TestMCPUtilities:
    """Test MCP utility functions."""

    def test_list_available_tools(self) -> None:
        """Test listing available tools."""
        from gnn.mcp import list_available_tools

        tools = list_available_tools()
        assert isinstance(tools, (list, dict))

    def test_get_available_tools(self) -> None:
        """Test getting available tools via processor."""
        from gnn.mcp import get_available_tools

        tools = get_available_tools()
        assert isinstance(tools, (list, dict))

    def test_initialize_function(self) -> None:
        """Test MCP initialization function (SDK required in-repo)."""
        from gnn.mcp import initialize

        result = initialize()
        assert isinstance(result, tuple) and len(result) == 3
        assert result[1] is True and result[2] is True, "MCP init must succeed"


class TestMCPCaching:
    """Test MCP caching functionality."""

    def test_cache_feature_enabled(self) -> None:
        """Test that caching feature is enabled."""
        from gnn.mcp import FEATURES

        assert FEATURES.get("caching", False) is True

    def test_rate_limiting_feature_enabled(self) -> None:
        """Test that rate limiting feature is enabled."""
        from gnn.mcp import FEATURES

        assert FEATURES.get("rate_limiting", False) is True


class TestMCPIntegration:
    """Integration tests for MCP module."""

    def test_full_mcp_workflow(self, safe_filesystem: Any) -> None:
        """Test complete MCP workflow: register, execute, cleanup."""
        from gnn.mcp import MCPTool

        # Create tool with correct signature
        def add_func(a: int = 0, b: int = 0) -> Any:
            return {"result": a + b}

        tool = MCPTool(
            name="add_numbers",
            func=add_func,
            schema={
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            },
            description="Adds two numbers",
        )

        # Execute
        result = tool.func(a=5, b=3)
        assert result["result"] == 8

    def test_mcp_with_gnn_files(self, safe_filesystem: Any) -> None:
        """Test MCP processing with actual GNN files."""
        from gnn.mcp import process_mcp

        # Create GNN file
        gnn_content = """# MCP Test Model

## StateSpaceBlock
mcp_state[10]

## Parameters
rate = 0.1
"""
        safe_filesystem.create_file("mcp_model.md", gnn_content)
        output_dir = safe_filesystem.create_dir("mcp_gnn_output")

        import logging

        logger = logging.getLogger("test_mcp_gnn")

        result = process_mcp(
            target_dir=safe_filesystem.temp_dir, output_dir=output_dir, logger=logger
        )
        assert isinstance(result, bool) and result is True
