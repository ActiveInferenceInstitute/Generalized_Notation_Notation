"""Section-heading spelling: canonical, spaced and unrecognized headings.

``MarkdownGNNParser`` keys its section parsers by canonical heading name. A
heading that differs from one of those names only by whitespace must reach the
same parser and say so through ``ParseResult.warnings``; an ontology-looking
heading that names nothing must not leave silently empty ontology bindings.
"""

from __future__ import annotations

import logging

import pytest

from gnn.parsers import GNNParsingSystem
from gnn.parsers.common import GNNFormat
from gnn.parsers.markdown_parser import MarkdownGNNParser

pytestmark = [pytest.mark.unit, pytest.mark.fast]

BODY = (
    "## GNNSection\nHeadingProbe\n"
    "## GNNVersionAndFlags\nGNN v1\n"
    "## ModelName\nHeading probe\n"
    "## StateSpaceBlock\ns[2,1,type=int]\n"
    "## Connections\ns>s\n"
    "## Time\nStatic\n"
)


def _annotated(heading: str) -> str:
    return BODY + f"{heading}\ns=HiddenState\n"


def _warning_records(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    ]


def test_canonical_ontology_heading_parses_without_heading_warning() -> None:
    result = MarkdownGNNParser().parse_string(_annotated("## ActInfOntologyAnnotation"))

    assert [m.ontology_term for m in result.model.ontology_mappings] == ["HiddenState"]
    assert not [w for w in result.warnings if "heading" in w]


def test_spaced_ontology_heading_is_normalized_with_visible_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="gnn.parsers.markdown_parser"):
        result = MarkdownGNNParser().parse_string(
            _annotated("## ActInf Ontology Annotation")
        )

    assert [m.ontology_term for m in result.model.ontology_mappings] == ["HiddenState"]
    assert result.model.ontology_mappings[0].variable_name == "s"
    warning = next(w for w in result.warnings if "heading" in w)
    assert "'ActInf Ontology Annotation'" in warning
    assert "'ActInfOntologyAnnotation'" in warning
    assert "normalized" in warning
    assert any(
        "ActInf Ontology Annotation" in message for message in _warning_records(caplog)
    )


def test_spaced_heading_keeps_the_raw_heading_in_raw_sections() -> None:
    result = MarkdownGNNParser().parse_string(
        _annotated("## ActInf Ontology Annotation")
    )

    assert "ActInf Ontology Annotation" in result.model.raw_sections


def test_unknown_actinf_heading_warns_and_produces_no_bindings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="gnn.parsers.markdown_parser"):
        result = MarkdownGNNParser().parse_string(
            _annotated("## ActInf Ontology Annotations")
        )

    assert result.model.ontology_mappings == []
    warning = next(w for w in result.warnings if "heading" in w)
    assert "'ActInf Ontology Annotations'" in warning
    assert "produced no ontology bindings" in warning
    assert any(
        "ActInf Ontology Annotations" in message for message in _warning_records(caplog)
    )


def test_registry_path_normalizes_the_spaced_heading() -> None:
    result = GNNParsingSystem().parse_string(
        _annotated("## ActInf Ontology Annotation"), GNNFormat.MARKDOWN
    )

    assert result.success, result.errors
    assert [m.ontology_term for m in result.model.ontology_mappings] == ["HiddenState"]
    assert any("normalized" in w for w in result.warnings)


def test_spaced_section_heading_normalizes_generically() -> None:
    content = (
        "## GNNSection\nHeadingProbe\n"
        "## GNNVersionAndFlags\nGNN v1\n"
        "## Model Name\nHeading probe\n"
        "## StateSpaceBlock\ns[2,1,type=int]\n"
    )

    result = MarkdownGNNParser().parse_string(content)

    assert result.model.model_name == "Heading probe"
    assert any("'Model Name'" in w and "'ModelName'" in w for w in result.warnings)
