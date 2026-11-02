"""Section readers key a spaced heading by its canonical section name.

``MarkdownGNNParser`` normalizes a heading that differs from a canonical section
name only by whitespace. The readers that split markdown themselves -- the
visualization fallback parser, the export metadata reader, the round-trip direct
parser and the renderers -- must agree, otherwise a document the parser accepts
produces empty ontology bindings or missing variables in those paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gnn.export.notation_metadata import _section_bodies
from gnn.parsers.common import (
    GNN_SECTION_HEADINGS,
    canonical_section_heading,
    match_section_heading,
)
from gnn.parsers.markdown_parser import MarkdownGNNParser
from gnn.render.discopy.discopy_renderer import DisCoPyRenderer
from gnn.render.rxinfer.rxinfer_renderer import RxInferRenderer
from gnn.testing.round_trip_markdown_parser import _DirectMarkdownParser
from gnn.visualization.parse.gnn_file_parser import GNNParser as VisualizationGNNParser
from gnn.visualization.parse.markdown import parse_gnn_content

pytestmark = [pytest.mark.unit, pytest.mark.fast]

HEADER = "# Heading probe comment\n\n"


def _document(
    ontology_heading: str, state_heading: str, parameters_heading: str
) -> str:
    return (
        HEADER
        + "## GNNSection\nHeadingProbe\n"
        + "## GNNVersionAndFlags\nGNN v1\n"
        + "## ModelName\nHeading probe\n"
        + f"## {state_heading}\ns[2,1,type=int]\n"
        + "## Connections\ns>s\n"
        + "## InitialParameterization\nA=[[1,0],[0,1]]\n"
        + "## Equations\ns = s\n"
        + "## Time\nStatic\n"
        + f"## {ontology_heading}\ns=HiddenState\n"
        + f"## {parameters_heading}\ndt: 0.1\n"
        + "## Footer\nProbe\n"
    )


CANONICAL_DOC = _document(
    "ActInfOntologyAnnotation", "StateSpaceBlock", "ModelParameters"
)
SPACED_DOC = _document(
    "ActInf Ontology Annotation", "State Space Block", "Model Parameters"
)


def test_shared_heading_list_matches_the_parser_section_parsers() -> None:
    assert set(GNN_SECTION_HEADINGS) == set(MarkdownGNNParser().section_parsers)


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("ActInfOntologyAnnotation", "ActInfOntologyAnnotation"),
        ("ActInf Ontology Annotation", "ActInfOntologyAnnotation"),
        ("  ActInf   Ontology  Annotation  ", "ActInfOntologyAnnotation"),
        ("State Space Block", "StateSpaceBlock"),
        ("ModelAnnotation", "ModelAnnotation"),
        ("ActInfOntologyAnnotations", "ActInfOntologyAnnotations"),
        ("NotASection At All", "NotASection At All"),
    ],
)
def test_canonical_section_heading_normalizes_only_whitespace(
    heading: str, expected: str
) -> None:
    assert canonical_section_heading(heading) == expected


def test_match_section_heading_reports_only_known_sections() -> None:
    assert match_section_heading("ActInfOntologyAnnotations") is None
    assert match_section_heading("Model Name") == "ModelName"


def test_visualization_markdown_reader_keys_the_spaced_sections() -> None:
    canonical = parse_gnn_content(CANONICAL_DOC)
    spaced = parse_gnn_content(SPACED_DOC)

    assert "ActInf Ontology Annotation" not in spaced["sections"]
    assert list(spaced["sections"]) == list(canonical["sections"])
    assert spaced["variables"] == canonical["variables"]
    assert spaced["variables"]
    assert (
        spaced["sections"]["ActInfOntologyAnnotation"]
        == canonical["sections"]["ActInfOntologyAnnotation"]
    )
    assert (
        spaced["raw_sections"]["StateSpaceBlock"]
        == canonical["raw_sections"]["StateSpaceBlock"]
    )


def test_visualization_file_parser_reads_the_spaced_document(tmp_path: Path) -> None:
    canonical_file = tmp_path / "canonical.md"
    spaced_file = tmp_path / "spaced.md"
    canonical_file.write_text(CANONICAL_DOC, encoding="utf-8")
    spaced_file.write_text(SPACED_DOC, encoding="utf-8")

    parser = VisualizationGNNParser()
    assert parser.parse_file(str(spaced_file)) == parser.parse_file(str(canonical_file))


def test_export_metadata_reads_the_spaced_sections() -> None:
    for name in ("Time", "ActInfOntologyAnnotation", "StateSpaceBlock"):
        assert _section_bodies(SPACED_DOC, name) == _section_bodies(CANONICAL_DOC, name)
    assert _section_bodies(SPACED_DOC, "Time") == ["Static\n"]
    assert _section_bodies(SPACED_DOC, "ActInfOntologyAnnotation") == [
        "s=HiddenState\n"
    ]
    assert _section_bodies(SPACED_DOC, "NoSuchSection") == []


def test_round_trip_direct_parser_keys_the_spaced_sections() -> None:
    sections = _DirectMarkdownParser()._extract_sections(SPACED_DOC)
    assert "ActInf Ontology Annotation" not in sections
    assert set(sections) == set(
        _DirectMarkdownParser()._extract_sections(CANONICAL_DOC)
    )


@pytest.mark.parametrize("renderer_class", [DisCoPyRenderer, RxInferRenderer])
def test_renderers_parse_the_spaced_document_like_the_canonical_one(
    renderer_class: type[DisCoPyRenderer] | type[RxInferRenderer],
) -> None:
    renderer = renderer_class()
    spaced = renderer._parse_gnn_content(SPACED_DOC, "probe")  # type: ignore[attr-defined]
    canonical = renderer._parse_gnn_content(CANONICAL_DOC, "probe")  # type: ignore[attr-defined]
    assert spaced == canonical
    assert spaced["model_parameters"]
