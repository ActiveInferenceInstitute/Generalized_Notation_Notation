"""Tests for the gnn.parsers registries and GNNParsingSystem surface.

Covers: PARSER_REGISTRY, SERIALIZER_REGISTRY entries and
GNNParsingSystem.parse_string / get_supported_formats / get_available_parsers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

GNN_SNIPPET = "## ModelName\nTiny Agent\n\n## StateSpaceBlock\ns_f0[2,1,type=int]\n"


class TestRegistries:
    """Test format registries exported by gnn.parsers."""

    def test_registry_entries_map_formats_to_classes(self) -> None:
        from gnn import parsers
        from gnn.parsers import GNNFormat, MarkdownGNNParser, MarkdownSerializer

        assert parsers.PARSER_REGISTRY[GNNFormat.MARKDOWN] is MarkdownGNNParser
        assert parsers.SERIALIZER_REGISTRY[GNNFormat.MARKDOWN] is MarkdownSerializer
        assert len(parsers.PARSER_REGISTRY) >= 20
        assert len(parsers.SERIALIZER_REGISTRY) >= 20
        assert all(
            entry.__name__.endswith(("Parser", "GNNParser"))
            for entry in parsers.PARSER_REGISTRY.values()
        )


class TestGNNParsingSystem:
    """Test the registry-driven parsing system."""

    def test_system_lists_and_uses_markdown_parser(self) -> None:
        from gnn.parsers import GNNFormat, GNNParsingSystem

        system = GNNParsingSystem()
        supported = system.get_supported_formats()
        assert GNNFormat.MARKDOWN in supported
        assert system.get_available_parsers()[GNNFormat.MARKDOWN] == "MarkdownGNNParser"

        result = system.parse_string(GNN_SNIPPET, GNNFormat.MARKDOWN)
        assert result.success is True
        assert result.model.model_name == "Tiny Agent"
        assert [variable.name for variable in result.model.variables] == ["s_f0"]
