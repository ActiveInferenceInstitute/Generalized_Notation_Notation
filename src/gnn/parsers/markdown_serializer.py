"""Serialize GNN models using the native Markdown grammar."""

import re

from .base_serializer import BaseGNNSerializer
from .common import (
    ConnectionType,
    DataType,
    GNNInternalRepresentation,
    Parameter,
    match_section_heading,
)

_OPERATORS = {
    ConnectionType.DIRECTED: ">",
    ConnectionType.UNDIRECTED: "-",
    ConnectionType.BIDIRECTIONAL: "<->",
    ConnectionType.CONDITIONAL: "|",
}
_NATIVE_TYPES = {DataType.INTEGER: "int", DataType.BINARY: "bool"}
_BARE_MODEL_VALUE = re.compile(r"[A-Za-z_][A-Za-z0-9_. -]*")


def _endpoint(names: list[str]) -> str:
    return names[0] if len(names) == 1 else "(" + ",".join(names) + ")"


def _described(line: str, description: str | None, marker: str = "#") -> str:
    return f"{line} {marker} {description}" if description else line


def _matrix_row(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "(" + ",".join(_matrix_row(item) for item in value) + ")"
    return repr(value)


def _parameter_line(parameter: Parameter, *, model_option: bool) -> str:
    value = parameter.value
    if (
        not model_option
        and parameter.metadata.get("markdown_braced_matrix") is True
        and isinstance(value, (list, tuple))
    ):
        matrix = "{" + ",".join(_matrix_row(row) for row in value) + "}"
        return _described(f"{parameter.name}={matrix}", parameter.description)
    literal = repr(value)
    if model_option:
        if (
            isinstance(value, str)
            and _BARE_MODEL_VALUE.fullmatch(value)
            and value.lower() not in {"true", "false", "none"}
        ):
            literal = value
        elif (
            isinstance(value, (list, tuple))
            and len(value) > 1
            and all(type(item) in (int, float) for item in value)
        ):
            literal = ",".join(map(str, value))
    separator = ":" if model_option else " ="
    return _described(f"{parameter.name}{separator} {literal}", parameter.description)


class MarkdownSerializer(BaseGNNSerializer):
    """Serializer for GNN Markdown format."""

    def serialize(self, model: GNNInternalRepresentation) -> str:
        sections: list[str] = []

        def add(name: str, lines: str | list[str] | None) -> None:
            if lines is None:
                return
            content = lines if isinstance(lines, str) else "\n".join(lines)
            if content:
                sections.append(f"## {name}\n{content}")

        add("GNNSection", model.extensions.get("gnn_section", ""))
        add(
            "GNNVersionAndFlags",
            [f"GNN v{model.version}", *model.extensions.get("flags", [])],
        )
        add("ModelName", model.model_name)
        add("ModelAnnotation", model.annotation)

        declarations = []
        for variable in model.variables:
            references = variable.constraints.get("dimension_references", [])
            dimensions = [
                str(references[index] or dimension)
                if index < len(references)
                else str(dimension)
                for index, dimension in enumerate(variable.dimensions)
            ]
            data_type = _NATIVE_TYPES.get(variable.data_type, variable.data_type.value)
            attributes = [
                f"{key}={value}"
                for key, value in variable.constraints.items()
                if key != "dimension_references" and isinstance(value, str)
            ]
            fields = ",".join([*dimensions, f"type={data_type}", *attributes])
            declarations.append(
                _described(f"{variable.name}[{fields}]", variable.description)
            )
        add("StateSpaceBlock", declarations)

        connections = []
        for connection in model.connections:
            source = _endpoint(connection.source_variables)
            target = _endpoint(connection.target_variables)
            suffix = f":{connection.annotation}" if connection.annotation else ""
            connections.append(
                _described(
                    f"{source}{_OPERATORS[connection.connection_type]}{target}{suffix}",
                    connection.description,
                )
            )
        add("Connections", connections)

        add(
            "InitialParameterization",
            [
                _parameter_line(parameter, model_option=False)
                for parameter in model.parameters
                if parameter.section != "ModelParameters"
            ],
        )
        add(
            "Equations",
            "\n\n".join(
                f"{equation.label}: {equation.content}"
                if equation.label
                else equation.content
                for equation in model.equations
            ),
        )

        if time := model.time_specification:
            time_lines = [time.time_type]
            if time.discretization:
                time_variable = time.metadata.get("time_variable")
                suffix = f"={time_variable}" if time_variable is not None else ""
                time_lines.append(time.discretization + suffix)
            if time.horizon is not None:
                time_lines.append(f"ModelTimeHorizon = {time.horizon}")
            if time.step_size is not None:
                time_lines.append(f"StepSize = {time.step_size}")
            time_lines.extend(time.metadata.get("source_declarations", []))
            add("Time", time_lines)

        add(
            "ActInfOntologyAnnotation",
            [
                _described(
                    f"{mapping.variable_name} = {mapping.ontology_term}",
                    mapping.description,
                    "###",
                )
                for mapping in model.ontology_mappings
            ],
        )
        add(
            "ModelParameters",
            [
                _parameter_line(parameter, model_option=True)
                for parameter in model.parameters
                if parameter.section == "ModelParameters"
            ],
        )
        for name, content in model.extensions.items():
            if (
                isinstance(content, str)
                and name not in {"gnn_section", "flags", "footer", "signature"}
                and match_section_heading(name) is None
                and "\n" not in name
            ):
                add(name, content)
        add("Footer", model.extensions.get("footer", ""))
        signature = model.extensions.get("signature") or (
            f"Checksum: {model.checksum}" if model.checksum else ""
        )
        add("Signature", signature)
        return "\n\n".join(sections) + "\n"
