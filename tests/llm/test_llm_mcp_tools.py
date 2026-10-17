"""Deterministic tests for MCP tool functions in llm.mcp (happy paths and
parameter honoring). Path-sandbox security is covered by test_llm_mcp_security.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.llm.defaults import DEFAULT_OLLAMA_MODEL
from gnn.llm.mcp import (
    _ANALYSIS_TYPES,
    analyze_gnn_with_llm_mcp,
    generate_llm_documentation_mcp,
    get_llm_module_info_mcp,
    get_llm_providers_mcp,
    process_llm_mcp,
    register_tools,
)

pytestmark = pytest.mark.unit

SAMPLE_GNN = (
    Path(__file__).resolve().parent.parent.parent
    / "input"
    / "gnn_files"
    / "discrete"
    / "actinf_pomdp_agent.md"
)


class _RecordingMCP:
    def __init__(self) -> None:
        self.registered: list[tuple[str, object, dict, str]] = []

    def register_tool(
        self,
        name: str,
        handler: object,
        schema: dict,
        description: str,
        **kwargs: object,
    ) -> None:
        self.registered.append((name, handler, schema, description))


class TestAnalyzeGnnWithLLMMCP:
    def test_rejects_unknown_analysis_type(self) -> None:
        result = analyze_gnn_with_llm_mcp(str(SAMPLE_GNN), analysis_type="bogus")
        assert result["success"] is False
        assert "bogus" in result["error"]
        assert "comprehensive" in result["error"]

    def test_provider_override_pins_default_ollama_tag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_analyze(path: Path, ollama_model: str | None = None) -> dict:
            captured["model"] = ollama_model
            return {"analysis": "ok"}

        monkeypatch.setattr("gnn.llm.mcp.analyze_gnn_file_with_llm", fake_analyze)

        result = analyze_gnn_with_llm_mcp(str(SAMPLE_GNN), provider="ollama")
        assert result["success"] is True
        assert captured["model"] == DEFAULT_OLLAMA_MODEL
        assert result["provider"] == "ollama"
        assert result["analysis_type"] == "comprehensive"

    def test_no_provider_uses_default_routing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_analyze(path: Path, ollama_model: str | None = None) -> dict:
            captured["model"] = ollama_model
            return {"analysis": "ok"}

        monkeypatch.setattr("gnn.llm.mcp.analyze_gnn_file_with_llm", fake_analyze)
        result = analyze_gnn_with_llm_mcp(str(SAMPLE_GNN), provider=None)
        assert result["success"] is True
        assert captured["model"] is None

    def test_analysis_type_enum_matches_schema(self) -> None:
        assert _ANALYSIS_TYPES == (
            "comprehensive",
            "summary",
            "complexity",
            "connections",
        )


class TestProviderMatrix:
    def test_all_registered_providers_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key in (
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "PERPLEXITY_API_KEY",
            "ANTHROPIC_API_KEY",
        ):
            monkeypatch.setenv(key, "test-key")
        result = get_llm_providers_mcp()
        assert result["success"] is True
        for name in (
            "openai",
            "anthropic",
            "ollama",
            "google",
            "openrouter",
            "perplexity",
        ):
            assert name in result["providers"], name
        for cloud in ("openai", "openrouter", "perplexity", "anthropic"):
            assert result["providers"][cloud]["configured"] is True
        assert result["count"] == len(result["providers"])


class TestModuleInfo:
    def test_lists_all_registered_tools(self) -> None:
        result = get_llm_module_info_mcp()
        assert result["success"] is True
        assert result["tools"] == [
            "process_llm",
            "analyze_gnn_with_llm",
            "generate_llm_documentation",
            "get_llm_providers",
            "get_llm_module_info",
        ]


class TestRegisterTools:
    def test_registers_all_tools_table_driven(self) -> None:
        mcp = _RecordingMCP()
        register_tools(mcp)
        names = [name for name, _, _, _ in mcp.registered]
        assert names == [
            "process_llm",
            "analyze_gnn_with_llm",
            "generate_llm_documentation",
            "get_llm_providers",
            "get_llm_module_info",
        ]
        for _, _, schema, description in mcp.registered:
            assert isinstance(schema, dict)
            assert description


class TestProcessLLMMCP:
    def test_success_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_process_llm(**kwargs: object) -> bool:
            captured.update(kwargs)
            return True

        monkeypatch.setattr("gnn.llm.mcp.process_llm", fake_process_llm)
        result = process_llm_mcp(
            str(SAMPLE_GNN.parent.parent), str(SAMPLE_GNN.parent), verbose=True
        )
        assert result["success"] is True
        assert captured["verbose"] is True


def test_mcp_local_override_uses_configured_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "local-capable:exact")
    captured = {}

    def analyze(path: Path, ollama_model: str | None = None) -> dict:
        captured["model"] = ollama_model
        return {"llm_summary_status": "unavailable"}

    monkeypatch.setattr("gnn.llm.mcp.analyze_gnn_file_with_llm", analyze)
    result = analyze_gnn_with_llm_mcp(str(SAMPLE_GNN), provider="ollama")
    assert captured["model"] == "local-capable:exact"
    assert result["success"] is False


def test_mcp_inventory_reports_http_configuration_not_sdk_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:23001")
    monkeypatch.setenv("OLLAMA_DISABLED", "true")
    ollama = get_llm_providers_mcp()["providers"]["ollama"]
    assert ollama["available"] is True
    assert ollama["configured"] is False
    assert ollama["base_url"] == "http://127.0.0.1:23001"
    assert ollama["readiness"] == "not_probed"


@pytest.mark.asyncio
async def test_mcp_resolves_async_analyzer_in_running_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def analyze(path: Path, ollama_model: str | None = None) -> dict:
        return {"llm_summary_status": "available", "llm_summary": "awaited"}

    monkeypatch.setattr("gnn.llm.mcp.analyze_gnn_file_with_llm", analyze)
    result = analyze_gnn_with_llm_mcp(str(SAMPLE_GNN), provider="ollama")
    assert result["success"] is True
    assert result["llm_summary"] == "awaited"


@pytest.mark.parametrize("format", ["markdown", "text"])
def test_structural_documentation_writes_requested_readable_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, format: str
) -> None:
    output = tmp_path / f"documentation.{format}"
    monkeypatch.setattr(
        "gnn.llm.mcp._resolve_output_file", lambda value, **kwargs: Path(value)
    )
    result = generate_llm_documentation_mcp(str(SAMPLE_GNN), str(output), format)
    assert result["success"] and result["analysis_source"] == "structural"
    text = output.read_text()
    assert text.startswith("# " if format == "markdown" else SAMPLE_GNN.name)
    assert result["variable_documentation"][0]["name"] in text
    assert "Variables" in text and "Connections" in text and "Usage Examples" in text
    assert result["usage_examples"][0]["code"] in text
    if format == "text":
        assert "## " not in text and "**" not in text and "```" not in text


def test_documentation_refuses_unknown_format_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unknown format must fail before generation")

    monkeypatch.setattr("gnn.llm.mcp.generate_documentation", unexpected)
    result = generate_llm_documentation_mcp(str(SAMPLE_GNN), format="json")
    assert not result["success"] and "format" in result["error"]
