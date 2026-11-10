#!/usr/bin/env python3
"""
Tests for GNN format detection and file conversion.

Pins:
- public ``detect_gnn_format_from_content`` heuristic (unified_parser)
- ``GNNParsingSystem.convert_file`` success and error contracts
- ``_PARSER_CLASS_PATHS`` coverage parity with ``PARSER_REGISTRY``
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.parsers import GNNFormat, GNNParsingSystem, convert_gnn_format  # noqa: E402
from gnn.parsers.common import ParseError  # noqa: E402
from gnn.parsers.system import PARSER_REGISTRY  # noqa: E402
from gnn.parsers.unified_parser import (  # noqa: E402
    _PARSER_CLASS_PATHS,
    UnifiedGNNParser,
    detect_gnn_format_from_content,
)

REPO = Path(__file__).resolve().parents[2]
EXEMPLAR = REPO / "input" / "gnn_files" / "discrete" / "actinf_pomdp_agent.md"


class TestDetectGNNFormatFromContent:
    """Content-based format sniffing never raises and detects by markers."""

    @pytest.mark.unit
    def test_xml_detected(self) -> None:
        assert (
            detect_gnn_format_from_content('<?xml version="1.0"?><model/>')
            == GNNFormat.XML
        )

    @pytest.mark.unit
    def test_pnml_detected_from_xml_marker(self) -> None:
        content = '<?xml version="1.0"?><pnml><net/></pnml>'
        assert detect_gnn_format_from_content(content) == GNNFormat.PNML

    @pytest.mark.unit
    def test_json_detected(self) -> None:
        assert detect_gnn_format_from_content('{"model_name": "m"}') == GNNFormat.JSON

    @pytest.mark.unit
    def test_markdown_gnn_sections_detected(self) -> None:
        content = "## GNNSection\n## ModelName: m\n## StateSpaceBlock\n"
        assert detect_gnn_format_from_content(content) == GNNFormat.MARKDOWN

    @pytest.mark.unit
    def test_unknown_content_falls_back_to_markdown(self) -> None:
        assert detect_gnn_format_from_content("lorem ipsum dolor") == GNNFormat.MARKDOWN

    @pytest.mark.unit
    def test_empty_content_falls_back_to_markdown(self) -> None:
        assert detect_gnn_format_from_content("") == GNNFormat.MARKDOWN

    @pytest.mark.unit
    def test_coq_comment_marker_detected(self) -> None:
        assert (
            detect_gnn_format_from_content("Require Import List.\n(* model *)")
            == GNNFormat.COQ
        )


class TestGNNParsingSystemConvertFile:
    """convert_file: parse input, serialize to target, never silently succeed."""

    @pytest.fixture
    def system(self) -> GNNParsingSystem:
        return GNNParsingSystem(strict_validation=False)

    @pytest.mark.unit
    def test_markdown_to_json_conversion(
        self, system: GNNParsingSystem, tmp_path: Path
    ) -> None:
        out = tmp_path / "conv.json"
        result = system.convert_file(EXEMPLAR, out)
        assert result == out
        payload: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
        # The JSON serializer embeds model data that parses back cleanly.
        assert payload

    @pytest.mark.unit
    def test_converted_output_reparses_with_matching_model_name(
        self, system: GNNParsingSystem, tmp_path: Path
    ) -> None:
        out = tmp_path / "conv.json"
        system.convert_file(EXEMPLAR, out)
        reparsed = system.parse_file(out)
        assert reparsed.success
        source = system.parse_file(EXEMPLAR)
        assert reparsed.model.model_name == source.model.model_name

        public_out = tmp_path / "public.json"
        assert convert_gnn_format(str(EXEMPLAR), str(public_out)) is None
        public_result = system.parse_file(public_out)
        assert public_result.success
        assert public_result.model.model_name == source.model.model_name

    @pytest.mark.unit
    def test_explicit_target_format_overrides_extension(
        self, system: GNNParsingSystem, tmp_path: Path
    ) -> None:
        out = tmp_path / "conv.txt"
        result = system.convert_file(EXEMPLAR, out, to_format=GNNFormat.YAML)
        assert result == out
        assert out.stat().st_size > 0

        public_out = tmp_path / "public.txt"
        assert convert_gnn_format(EXEMPLAR, public_out, GNNFormat.YAML) is None
        public_result = system.parse_file(public_out, GNNFormat.YAML)
        assert public_result.success
        assert public_result.model.model_name == system.parse_file(EXEMPLAR).model.model_name

    @pytest.mark.unit
    def test_unknown_output_extension_raises_value_error(
        self, system: GNNParsingSystem, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="Unknown file extension"):
            system.convert_file(EXEMPLAR, tmp_path / "conv.xyz")

    @pytest.mark.unit
    def test_missing_input_raises_file_not_found(
        self, system: GNNParsingSystem, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            system.convert_file(tmp_path / "missing.md", tmp_path / "out.json")

    @pytest.mark.unit
    def test_failed_parse_raises_parse_error(
        self, system: GNNParsingSystem, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all", encoding="utf-8")
        with pytest.raises(ParseError):
            system.convert_file(bad, tmp_path / "out.md")

        existing = tmp_path / "previous.md"
        original = b"caller-owned output\n"
        existing.write_bytes(original)
        with pytest.raises(ParseError, match="bad[.]json"):
            convert_gnn_format(bad, existing)
        assert existing.read_bytes() == original

        fresh = tmp_path / "absent" / "model.json"
        with pytest.raises(ParseError, match="bad[.]json"):
            convert_gnn_format(bad, fresh)
        assert not fresh.parent.exists()

    @pytest.mark.unit
    def test_parse_only_target_format_raises_value_error(
        self, system: GNNParsingSystem, tmp_path: Path
    ) -> None:
        # PNML is registered as a parser but has no serializer.
        assert GNNFormat.PNML not in system.serializers
        with pytest.raises(ValueError, match="Unsupported target format"):
            system.convert_file(EXEMPLAR, tmp_path / "conv.pnml")

    @pytest.mark.unit
    def test_creates_missing_parent_directories(
        self, system: GNNParsingSystem, tmp_path: Path
    ) -> None:
        out = tmp_path / "nested" / "deeper" / "conv.json"
        result = system.convert_file(EXEMPLAR, out)
        assert result.exists()


class TestParserClassPathParity:
    """Unified parser dispatch covers every registry parser format."""

    @pytest.mark.unit
    def test_table_covers_registry(self) -> None:
        assert set(_PARSER_CLASS_PATHS) == set(PARSER_REGISTRY)

    @pytest.mark.unit
    def test_every_entry_resolves_to_registry_class(self) -> None:
        parser = UnifiedGNNParser()
        for fmt, expected in PARSER_REGISTRY.items():
            assert parser._get_parser_class(fmt) is expected, (
                f"format {fmt} resolves to unexpected class"
            )
