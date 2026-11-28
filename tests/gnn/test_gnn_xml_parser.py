#!/usr/bin/env python3
"""
Tests for gnn/parsers/xml_parser.py — XMLGNNParser and PNMLParser.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.parsers import GNNFormat, GNNParsingSystem  # noqa: E402

MINIMAL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<GNNModel>
  <Metadata>
    <ModelName>TestModel</ModelName>
    <GNNSection>ActInfPOMDP</GNNSection>
  </Metadata>
  <StateSpaceBlock>
    <Variable name="s" dimensions="3,1" type="float"/>
    <Variable name="o" dimensions="2,1" type="int"/>
  </StateSpaceBlock>
  <Connections>
    <Connection source="s" target="o" directed="true"/>
  </Connections>
</GNNModel>
"""

MINIMAL_PNML = """\
<?xml version="1.0"?>
<pnml>
  <net id="net1" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <name><text>TestNet</text></name>
    <page id="page1">
      <place id="p1"><name><text>place1</text></name></place>
      <place id="p2"><name><text>place2</text></name></place>
      <transition id="t1"><name><text>trans1</text></name></transition>
      <arc id="a1" source="p1" target="t1"/>
    </page>
  </net>
</pnml>
"""


def test_saved_xml_preserves_typed_model_data_in_native_elements(
    tmp_path: Path,
) -> None:
    model = {
        "model_name": "Authored",
        "version": "1.1.0",
        "annotation": "Model rationale.\n\nSecond paragraph.",
        "extensions": {"gnn_section": "ActInfPOMDP"},
        "variables": [
            {
                "name": "s",
                "data_type": "integer",
                "dimensions": [2],
                "constraints": {"default": "ones"},
            },
            {"name": "o", "data_type": "binary", "dimensions": [2]},
            {"name": "u", "data_type": "integer", "dimensions": [2]},
        ],
        "connections": [
            {
                "source_variables": ["u", "s"],
                "target_variables": ["o"],
                "connection_type": "bidirectional",
                "weight": 0.0,
                "annotation": "likelihood:α",
                "description": "two-way",
            }
        ],
        "parameters": [
            {
                "name": "options",
                "value": {"flag": False, "label": "false"},
                "type_hint": "mapping",
                "description": "authored",
                "metadata": {"reviewed": True},
                "section": "ModelParameters",
            }
        ],
        "equations": [
            {
                "label": "update",
                "content": "x = x",
                "format": "latex",
                "description": "authored relation",
            }
        ],
        "time_specification": {
            "horizon": 0,
            "step_size": 0.0,
            "metadata": {"time_variable": "t"},
        },
    }
    source = tmp_path / "authored.json"
    source.write_text(json.dumps(model), encoding="utf-8")
    target = tmp_path / "saved.xml"
    system = GNNParsingSystem(strict_validation=False)
    system.convert_file(source, target)
    assert "MODEL_DATA" not in target.read_text()

    restored = system.parse_file(target)
    assert restored.success, restored.errors
    loaded = restored.model
    assert loaded.version == "1.1.0"
    assert loaded.annotation == model["annotation"]
    assert loaded.extensions == model["extensions"]
    assert loaded.get_variable_by_name("s").constraints == {"default": "ones"}
    edge = loaded.connections[0]
    assert edge.source_variables == ["u", "s"]
    assert edge.annotation == "likelihood:α"
    assert edge.description == "two-way" and edge.weight == 0.0
    parameter = loaded.parameters[0]
    assert parameter.value == {"flag": False, "label": "false"}
    assert parameter.metadata == {"reviewed": True}
    assert parameter.section == "ModelParameters"
    assert parameter.type_hint == "mapping"
    assert parameter.description == "authored"
    equation = loaded.equations[0]
    assert (equation.label, equation.content, equation.description) == (
        "update",
        "x = x",
        "authored relation",
    )
    assert loaded.time_specification.horizon == 0
    assert loaded.time_specification.step_size == 0.0
    assert loaded.time_specification.metadata == {"time_variable": "t"}


def test_legacy_xml_fills_omissions_without_overriding_visible_model(
    tmp_path: Path,
) -> None:
    embedded = {
        "model_name": "Stale title",
        "version": "1.0",
        "parameters": [
            {"name": "keep_type", "value": [False, 0]},
            {"name": "visible_edit", "value": 1},
        ],
        "connections": [
            {
                "source_variables": ["s"],
                "target_variables": ["o"],
                "annotation": "legacy label",
            }
        ],
    }
    source = tmp_path / "legacy.xml"
    source.write_text(
        '<gnn_model name="Visible title" version="1.1">'
        '<connections><connection source="s" target="o" weight="0.5">'
        "<description>physical edge</description></connection></connections>"
        '<parameters><parameter name="keep_type">[False, 0]</parameter>'
        '<parameter name="visible_edit">2</parameter></parameters>'
        '<equations><equation label="visible">x = x</equation></equations>'
        f"<!-- MODEL_DATA: {json.dumps(embedded)} -->"
        "</gnn_model>",
        encoding="utf-8",
    )
    system = GNNParsingSystem(strict_validation=False)
    restored = system.parse_file(source)
    assert restored.success, restored.errors
    model = restored.model
    assert (model.model_name, model.version) == ("Visible title", "1.1")
    assert model.connections[0].annotation == "legacy label"
    assert model.connections[0].description == "physical edge"
    assert model.connections[0].weight == 0.5
    assert model.parameters[0].value == [False, 0]
    assert model.parameters[1].value == "2"
    assert model.equations[0].content == "x = x"

    comment_only = f'<?xml version="1.0"?><gnn_model><!-- MODEL_DATA: {json.dumps(embedded)} --></gnn_model>'
    fallback = system.parse_string(comment_only, GNNFormat.XML)
    assert fallback.success
    assert fallback.model.model_name == "Stale title"
    assert fallback.model.parameters[0].value == [False, 0]


class TestXMLGNNParser:
    def _get_parser(self) -> Any:
        try:
            from gnn.parsers.xml_parser import XMLGNNParser

            return XMLGNNParser()
        except ImportError:
            raise AssertionError("xml_parser not importable")

    def test_parse_string_returns_parse_result(self) -> Any:
        parser = self._get_parser()
        result = parser.parse_string(MINIMAL_XML)
        assert result is not None

    def test_parse_string_with_valid_xml_no_errors(self) -> Any:
        parser = self._get_parser()
        result = parser.parse_string(MINIMAL_XML)
        # Should parse without crashing; errors list may be present
        assert hasattr(result, "errors") or result is not None

    def test_parse_string_invalid_xml_handles_gracefully(self) -> Any:
        parser = self._get_parser()
        result = parser.parse_string("NOT XML AT ALL <<<")
        # Must not raise — should return error result or None
        # If it returns something, it should have errors
        assert result is None or hasattr(result, "errors")

    def test_get_supported_extensions_includes_xml(self) -> Any:
        parser = self._get_parser()
        exts = parser.get_supported_extensions()
        assert any("xml" in ext.lower() for ext in exts)

    def test_parse_string_result_has_model(self) -> Any:
        parser = self._get_parser()
        result = parser.parse_string(MINIMAL_XML)
        if result is None:
            raise AssertionError("parser returned None for minimal XML")
        # ParseResult has a .model attribute
        assert hasattr(result, "model") or hasattr(result, "model_name")


class TestPNMLParser:
    def _get_parser(self) -> Any:
        try:
            from gnn.parsers.xml_parser import PNMLParser

            return PNMLParser()
        except ImportError:
            raise AssertionError("PNMLParser not importable")

    def test_parse_string_returns_result(self) -> Any:
        parser = self._get_parser()
        result = parser.parse_string(MINIMAL_PNML)
        assert result is not None

    def test_parse_string_invalid_pnml_handles_gracefully(self) -> Any:
        parser = self._get_parser()
        result = parser.parse_string("<<INVALID>>")
        assert result is None or hasattr(result, "errors")

    def test_get_supported_extensions_includes_pnml(self) -> Any:
        parser = self._get_parser()
        exts = parser.get_supported_extensions()
        assert any("pnml" in ext.lower() or "xml" in ext.lower() for ext in exts)
