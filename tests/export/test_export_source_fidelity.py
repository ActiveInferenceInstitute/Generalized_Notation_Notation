"""Source sections and topology through the registered export writers."""

from pathlib import Path

import networkx as nx
import pytest
from defusedxml.ElementTree import parse

from gnn.export import export_model, parse_gnn_content
from gnn.parsers.markdown_parser import MarkdownGNNParser

SAMPLE = (
    Path(__file__).resolve().parents[2]
    / "input/gnn_files/discrete/two_state_bistable.md"
)


def test_export_preserves_typed_sections_and_grouped_connections() -> None:
    source = """## ModelName
Groups
## StateSpaceBlock
s[2,type=int]
o[2,type=int]
a[1,type=int]
## Connections
(s,o)→a:forecast # grouped relation
s|o
## Initial Parameterization
# State prior stays in this section
D={(0.5, 0.5)}
"""
    exported = parse_gnn_content(source)
    assert exported["sections"]["InitialParameterization"] == [
        "# State prior stays in this section",
        "D={(0.5, 0.5)}",
    ]
    assert exported["variables"] == [
        {"name": name, "type": "integer"} for name in ("s", "o", "a")
    ]
    assert exported["connections"][0] == {
        "source": "s",
        "target": "a",
        "connection_type": "directed",
        "annotation": "forecast",
        "description": "grouped relation",
    }
    assert [
        (edge["source"], edge["target"], edge["connection_type"])
        for edge in exported["connections"]
    ] == [("s", "a", "directed"), ("o", "a", "directed"), ("s", "o", "conditional")]


@pytest.mark.parametrize("standalone", [False, True], ids=["step3", "standalone"])
def test_structured_and_standalone_artifacts_preserve_topology(
    tmp_path: Path, standalone: bool
) -> None:
    source = SAMPLE.read_text(encoding="utf-8")
    parsed = MarkdownGNNParser().parse_string(source)
    assert parsed.success
    model = parsed.model
    data = parse_gnn_content(source) if standalone else model.to_dict()
    expected = {
        (source, target)
        for edge in model.connections
        for source in edge.source_variables
        for target in edge.target_variables
    }
    assert len(expected) == 11

    receipt = export_model(data, tmp_path, formats=["graphml", "gexf", "xml"])
    assert receipt["success"], receipt
    for extension, reader in (("graphml", nx.read_graphml), ("gexf", nx.read_gexf)):
        graph = reader(tmp_path / f"model.{extension}")
        labels = {node: values["label"] for node, values in graph.nodes(data=True)}
        assert set(labels.values()) == {variable.name for variable in model.variables}
        assert {
            (labels[source], labels[target]) for source, target in graph.edges()
        } == expected

    xml = parse(tmp_path / "model.xml")
    assert {
        (edge.get("source"), edge.get("target"))
        for edge in xml.findall("./connections/connection")
    } == expected
    parameters = xml.find("./sections/section[@name='InitialParameterization']")
    assert parameters is not None
    assert "D={(0.5, 0.5)}" in [line.text for line in parameters.findall("line")]
