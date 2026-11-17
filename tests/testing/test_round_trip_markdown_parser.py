"""Unit tests for ``gnn.testing.round_trip_markdown_parser``.

Covers the ``_DirectMarkdownParser`` paths not exercised by
``test_round_trip_support.py``: file-based parsing, per-variable role and
data-type mapping branches, comment/blank-line skipping, malformed-line
tolerance, and time-specification edge cases.
"""

from __future__ import annotations

from pathlib import Path

from gnn.parsers.common import GNNInternalRepresentation
from gnn.testing.round_trip_markdown_parser import _DirectMarkdownParser

PARSER_GNN = """## ModelName
RoleModel

## StateSpaceBlock
s[2,2] # base hidden state
s_prime[2,2] # successor state
u[2] # control action
B[2,2,2,type=int] # transition
C[2,type=bool] # preference
D[2,type=str] # prior
π[3,foo] # policy with bad dim token

## Connections
s > s_prime
# a comment line
u > s_prime
no_arrow_line
split_arrow > middle > end

## InitialParameterization
# parameter comments are skipped
B = 0.5
C = true
weighted = x=y

## Time
Discrete
Step_size=1

## ActInfOntologyAnnotation
# mapping comments are skipped
s = hidden_state
malformed_line
"""


def _parse(content: str) -> GNNInternalRepresentation:
    return _DirectMarkdownParser().parse_content(content)


class TestParseFile:
    def test_parse_file_reads_disk_content(self, tmp_path: Path) -> None:
        source = tmp_path / "model.md"
        source.write_text("## ModelName\nFromDisk\n", encoding="utf-8")

        model = _DirectMarkdownParser().parse_file(source)

        assert model.model_name == "FromDisk"


class TestSectionExtraction:
    def test_preamble_before_first_heading_is_ignored(self) -> None:
        model = _parse("Intro prose\n\n## ModelName\nAfter\n")

        assert model.model_name == "After"
        assert model.raw_sections == {"ModelName": "After"}

    def test_version_defaults_when_flag_section_absent(self) -> None:
        model = _parse("## ModelName\nM\n")

        assert model.version == "1.0"

    def test_empty_time_section_yields_no_time_specification(self) -> None:
        model = _parse("## ModelName\nM\n\n## Time\n")

        assert model.time_specification is None


class TestVariableParsing:
    def test_role_mapping_for_agent_variable_names(self) -> None:
        model = _parse(PARSER_GNN)
        by_name = {v.name: v for v in model.variables}

        assert by_name["s"].var_type.value == "hidden_state"
        assert by_name["s_prime"].var_type.value == "hidden_state"
        assert by_name["u"].var_type.value == "action"
        assert by_name["B"].var_type.value == "transition_matrix"
        assert by_name["D"].var_type.value == "prior_vector"
        assert by_name["π"].var_type.value == "policy"

    def test_data_type_mapping(self) -> None:
        model = _parse(PARSER_GNN)
        by_name = {v.name: v for v in model.variables}

        assert by_name["s"].data_type.value == "float"
        assert by_name["B"].data_type.value == "integer"
        assert by_name["C"].data_type.value == "binary"
        assert by_name["D"].data_type.value == "categorical"

    def test_descriptions_and_non_dimension_tokens(self) -> None:
        model = _parse(PARSER_GNN)
        by_name = {v.name: v for v in model.variables}

        assert by_name["s"].description == "base hidden state"
        assert by_name["s_prime"].dimensions == [2, 2]
        # "foo" is not an int dimension nor a type= token; it is ignored.
        assert by_name["π"].dimensions == [3]

    def test_comment_and_blank_lines_skipped_in_variables(self) -> None:
        content = "## StateSpaceBlock\n\n# heading comment\nx[1] # only var\n"
        model = _parse(content)

        assert [(v.name, v.dimensions) for v in model.variables] == [("x", [1])]


class TestConnectionParsing:
    def test_single_directed_connections_parsed(self) -> None:
        model = _parse(PARSER_GNN)
        conns = [(c.source_variables, c.target_variables) for c in model.connections]

        assert conns == [(["s"], ["s_prime"]), (["u"], ["s_prime"])]

    def test_lines_without_arrow_or_multi_arrow_ignored(self) -> None:
        model = _parse(PARSER_GNN)

        # "no_arrow_line" and "split_arrow > middle > end" are both dropped.
        assert len(model.connections) == 2


class TestParameterParsing:
    def test_parameters_split_on_first_equals(self) -> None:
        model = _parse(PARSER_GNN)

        assert [(p.name, p.value) for p in model.parameters] == [
            ("B", "0.5"),
            ("C", "true"),
            ("weighted", "x=y"),
        ]


class TestTimeParsing:
    def test_time_spec_keys_lowercased_and_defaults_applied(self) -> None:
        model = _parse(PARSER_GNN)
        time_spec = model.time_specification
        assert time_spec is not None

        assert time_spec.time_type == "Discrete"
        assert time_spec.step_size == "1"
        assert time_spec.discretization is None
        assert time_spec.horizon is None


class TestOntologyParsing:
    def test_only_key_value_lines_become_mappings(self) -> None:
        model = _parse(PARSER_GNN)

        assert [
            (m.variable_name, m.ontology_term) for m in model.ontology_mappings
        ] == [("s", "hidden_state")]
