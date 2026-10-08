"""Param-validation fidelity contract tests (wave-2 MED-01).

A ``func(**params)`` signature mismatch is an INVALID_PARAMS wire failure
(-32602), not an internal error (-32603); a ``None`` return from a tool with
no declared output contract is legitimate (``MCPTool`` has no ``returns``
schema); and capabilities expose the active validation mode so clients know
whether schema constraints beyond ``required`` are enforced.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.mcp

from gnn.mcp.exceptions import MCPInvalidParamsError, MCPToolExecutionError
from gnn.mcp.mcp import MCP
from gnn.mcp.server_core import MCPServer


def _registry(strict: bool = False) -> MCP:
    return MCP(
        enable_caching=False, enable_rate_limiting=False, strict_validation=strict
    )


class TestSignatureMismatchClassification:
    """func(**params) failures map to -32602, not -32603."""

    @pytest.mark.unit
    def test_extra_kwarg_raises_invalid_params(self) -> None:
        registry = _registry()
        registry.register_tool(
            name="two_args",
            func=lambda a, b: {"r": a + b},
            schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
            description="adds two numbers",
        )
        with pytest.raises(MCPInvalidParamsError) as excinfo:
            registry.execute_tool("two_args", {"a": 1, "b": 2, "c": 3})
        assert excinfo.value.code == -32602
        assert excinfo.value.tool_name == "two_args"

    @pytest.mark.unit
    def test_missing_required_argument_raises_invalid_params(self) -> None:
        registry = _registry()
        registry.register_tool(
            name="no_schema_tool",
            func=lambda a: {"r": a},
            schema={},
            description="declares no schema",
        )
        with pytest.raises(MCPInvalidParamsError):
            registry.execute_tool("no_schema_tool", {})

    @pytest.mark.unit
    def test_type_error_inside_tool_body_stays_execution_error(self) -> None:
        registry = _registry()
        registry.register_tool(
            name="bad_body",
            func=lambda: 1 + "x",  # type: ignore[operator]
            schema={},
            description="raises TypeError internally",
        )
        with pytest.raises(MCPToolExecutionError) as excinfo:
            registry.execute_tool("bad_body", {})
        assert excinfo.value.code == -32603

    @pytest.mark.unit
    def test_signature_mismatch_maps_to_32602_on_the_wire(self) -> None:
        registry = _registry()
        registry.register_tool(
            name="two_args",
            func=lambda a, b: {"r": a + b},
            schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
            description="adds two numbers",
        )
        server = MCPServer(mcp_instance=registry)
        response: dict[str, Any] | None = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {
                    "name": "two_args",
                    "arguments": {"a": 1, "b": 2, "c": 3},
                },
            }
        )
        assert response is not None
        assert response["error"]["code"] == -32602
        assert "two_args" in response["error"]["message"]


class TestStrictSchemaAdmission:
    """Public JSON-RPC validation must run before a registered handler executes."""

    @pytest.mark.parametrize(
        "arguments",
        [
            {"config": {"budget": 1}, "unknown": True},
            {"config": {}},
            {"config": {"budget": float("nan")}},
            {"config": {"budget": float("inf")}},
        ],
    )
    def test_nested_required_unknown_keys_and_nonfinite_numbers_are_invalid_params(
        self, arguments: dict
    ) -> None:
        registry = _registry(strict=True)
        calls = []

        def handler(config: dict) -> dict:
            calls.append(config)
            return {"accepted": config}

        registry.register_tool(
            name="bounded_consumer",
            func=handler,
            schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "config": {
                        "type": "object",
                        "required": ["budget"],
                        "properties": {"budget": {"type": "number", "minimum": 0}},
                    }
                },
                "required": ["config"],
            },
            description="A bounded consumer with an explicit nested contract",
        )
        response = MCPServer(mcp_instance=registry).handle_request(
            {
                "jsonrpc": "2.0",
                "id": 41,
                "method": "tools/call",
                "params": {"name": "bounded_consumer", "arguments": arguments},
            }
        )
        assert response["error"]["code"] == -32602
        assert calls == []
        accepted = registry.execute_tool(
            "bounded_consumer", {"config": {"budget": 1.5}}
        )
        assert accepted == {"accepted": {"budget": 1.5}}
        assert calls == [{"budget": 1.5}]


class TestNoneOutputAllowed:
    """A tool with no declared output contract may return None."""

    @pytest.mark.unit
    def test_none_return_passes_output_validation(self) -> None:
        registry = _registry()
        registry.register_tool(
            name="side_effect_tool",
            func=lambda: None,
            schema={},
            description="returns None by design",
        )
        assert registry.execute_tool("side_effect_tool", {}) is None

    @pytest.mark.unit
    def test_none_return_is_wire_valid(self) -> None:
        registry = _registry()
        registry.register_tool(
            name="side_effect_tool",
            func=lambda: None,
            schema={},
            description="returns None by design",
        )
        server = MCPServer(mcp_instance=registry)
        response = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {"name": "side_effect_tool", "arguments": {}},
            }
        )
        assert response is not None
        assert response["result"]["content"][0]["text"] == "null"


class TestValidationModeCapability:
    """Capabilities expose whether strict schema checks are active."""

    @pytest.mark.unit
    def test_default_mode_is_required_only(self) -> None:
        assert _registry().get_capabilities()["validation_mode"] == "required_only"

    @pytest.mark.unit
    def test_strict_mode_is_advertised(self) -> None:
        assert _registry(strict=True).get_capabilities()["validation_mode"] == (
            "strict"
        )
