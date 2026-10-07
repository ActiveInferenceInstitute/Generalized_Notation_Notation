"""Server-wide tool-timeout knob contract tests (tier-2 row 13).

``initialize(tool_timeout=...)`` / ``MCP(tool_timeout=...)`` set a default
timeout applied to tools that did not register their own ``timeout``; a
tool-level registration wins over the knob; ``None`` (the default) leaves
un-registered tools on the unbounded inline path.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.mcp

from gnn.mcp.exceptions import MCPToolTimeoutError
from gnn.mcp.mcp import MCP, initialize


class TestToolTimeoutKnob:
    """The server default bounds un-registered tools; explicit wins."""

    @pytest.mark.unit
    def test_knob_times_out_slow_untimed_tool(self) -> None:
        registry = MCP(
            enable_caching=False, enable_rate_limiting=False, tool_timeout=0.2
        )
        registry.register_tool(
            name="hung_tool",
            func=lambda: __import__("time").sleep(1.5) or {"never": True},
            schema={},
            description="sleeps past the server default timeout",
        )
        with pytest.raises(MCPToolTimeoutError) as excinfo:
            registry.execute_tool("hung_tool", {})
        assert excinfo.value.code == -32008

    @pytest.mark.unit
    def test_tool_explicit_timeout_overrides_knob(self) -> None:
        # The tool's own 1.5s timeout is the effective bound, so a 0.2s knob
        # must NOT cut it off; the quick body returns normally.
        registry = MCP(
            enable_caching=False, enable_rate_limiting=False, tool_timeout=0.2
        )
        registry.register_tool(
            name="own_timeout_tool",
            func=lambda: {"ok": True},
            schema={},
            description="registers a longer explicit timeout",
            timeout=1.5,
        )
        assert registry.execute_tool("own_timeout_tool", {}) == {"ok": True}

    @pytest.mark.unit
    def test_explicit_timeout_beats_knob_on_timeout_path(self) -> None:
        # The effective timeout is the tool's own 0.2s (not the 1.5s knob),
        # so the hung body trips the bound at the tool's value.
        registry = MCP(
            enable_caching=False, enable_rate_limiting=False, tool_timeout=1.5
        )
        registry.register_tool(
            name="hung_own_timeout_tool",
            func=lambda: __import__("time").sleep(1.0) or {"never": True},
            schema={},
            description="registers a shorter explicit timeout than the knob",
            timeout=0.2,
        )
        with pytest.raises(MCPToolTimeoutError) as excinfo:
            registry.execute_tool("hung_own_timeout_tool", {})
        assert excinfo.value.code == -32008

    @pytest.mark.unit
    def test_default_knob_keeps_untimed_tool_unbounded(self) -> None:
        registry = MCP(enable_caching=False, enable_rate_limiting=False)
        assert registry._tool_timeout is None
        registry.register_tool(
            name="slow_untimed_tool",
            func=lambda: {"finished": True},
            schema={},
            description="no registered timeout, no knob",
        )
        # No pool involvement: the inline path runs the handler on the caller
        # thread; a slow body would block indefinitely rather than surface
        # MCPToolTimeoutError.
        assert registry.execute_tool("slow_untimed_tool", {}) == {"finished": True}

    @pytest.mark.unit
    def test_zero_knob_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError):
            MCP(tool_timeout=0.0)
        with pytest.raises(ValueError):
            MCP(tool_timeout=-1.0)


class TestInitializeToolTimeoutKnob:
    """initialize(tool_timeout=...) validates and applies like the other overrides."""

    @pytest.mark.unit
    def test_initialize_applies_tool_timeout(self) -> None:
        mcp = initialize(
            halt_on_missing_sdk=False,
            force_proceed_flag=True,
            performance_mode="low",
            modules_allowlist=["mcp"],
            per_module_timeout=5.0,
            overall_timeout=20.0,
            tool_timeout=2.5,
        )[0]
        try:
            assert mcp._tool_timeout == 2.5
            assert mcp.config["tool_timeout"] == 2.5
        finally:
            mcp._tool_timeout = None

    @pytest.mark.unit
    def test_initialize_rejects_nonpositive_tool_timeout(self) -> None:
        with pytest.raises(ValueError):
            initialize(
                tool_timeout=0.0, halt_on_missing_sdk=False, force_proceed_flag=True
            )


class TestProcessorAliasMap:
    """The pipeline kwargs path maps tool_timeout through alias_map."""

    @pytest.mark.unit
    def test_alias_map_includes_tool_timeout(self) -> None:
        from gnn.mcp import processor

        src = inspect.getsource(processor.process_mcp)
        assert '"tool_timeout": ("tool_timeout", "mcp_tool_timeout")' in src
