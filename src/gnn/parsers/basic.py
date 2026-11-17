#!/usr/bin/env python3
"""
GNN parser module for GNN pipeline.
"""

import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

# Single authoritative definitions live in gnn/types/definitions.py (import from
# the implementation module, not the package facade, to keep the parsers <-> types
# import graph acyclic at runtime).
from gnn.types.definitions import ParsedGNN, ValidationLevel


class _GNNParseAccumulator:
    """Internal mutable builder for GNN parse results.

    This is a local implementation detail of GNNParsingSystem._basic_parser.
    The public canonical type is gnn.types.ParsedGNN (a dataclass).
    """

    def __init__(self, file_path: Union[str, Path]) -> None:
        """Initialize the instance."""
        self.file_path = Path(file_path)
        self.file_name = self.file_path.name
        self.content = ""
        self.sections: list[dict[str, str]] = []
        self.variables: list[dict[str, str]] = []
        self.connections: list[dict[str, str]] = []
        self.parse_errors: list[str] = []
        self.parse_warnings: list[str] = []

    def add_section(self, section_name: str, section_content: str = "") -> None:
        """Add a section to the parsed GNN."""
        self.sections.append({"name": section_name, "content": section_content})

    def add_variable(
        self, variable_name: str, variable_type: str = "", variable_value: str = ""
    ) -> None:
        """Add a variable to the parsed GNN."""
        self.variables.append(
            {"name": variable_name, "type": variable_type, "value": variable_value}
        )

    def add_connection(
        self, source: str, target: str, connection_type: str = ""
    ) -> None:
        """Add a connection to the parsed GNN."""
        self.connections.append(
            {"source": source, "target": target, "type": connection_type}
        )

    def add_error(self, error_message: str) -> None:
        """Add a parse error."""
        self.parse_errors.append(error_message)

    def add_warning(self, warning_message: str) -> None:
        """Add a parse warning."""
        self.parse_warnings.append(warning_message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_path": str(self.file_path),
            "file_name": self.file_name,
            "sections": self.sections,
            "variables": self.variables,
            "connections": self.connections,
            "parse_errors": self.parse_errors,
            "parse_warnings": self.parse_warnings,
        }


class GNNParsingSystem:
    """System for parsing GNN files."""

    def __init__(self) -> None:
        """Initialize the GNN parsing system."""
        self.parsers: dict[str, Callable[..., Any]] = {}
        self.validators: dict[str, Callable[..., Any]] = {}

    def register_parser(self, format_name: str, parser_func: Callable) -> None:
        """Register a parser for a specific format."""
        self.parsers[format_name] = parser_func

    def register_validator(self, format_name: str, validator_func: Callable) -> None:
        """Register a validator for a specific format."""
        self.validators[format_name] = validator_func

    def parse_file(
        self, file_path: Union[str, Path], format_name: str = "auto"
    ) -> Optional[_GNNParseAccumulator]:
        """Parse a GNN file."""
        file_path = Path(file_path)

        if not file_path.exists():
            return None

        # Auto-detect format if not specified
        if format_name == "auto":
            format_name = self._detect_format(file_path)

        # Get appropriate parser
        parser = self.parsers.get(format_name)
        if parser:
            return cast("_GNNParseAccumulator | None", parser(file_path))
        else:
            # Recovery to basic parser
            return self._basic_parser(file_path)

    def _detect_format(self, file_path: Path) -> str:
        """Detect the format of a GNN file."""
        extension = file_path.suffix.lower()

        if extension == ".md":
            return "markdown"
        elif extension == ".gnn":
            return "gnn"
        elif extension == ".txt":
            return "text"
        else:
            return "markdown"  # Default to markdown

    def _basic_parser(self, file_path: Path) -> _GNNParseAccumulator:
        """Basic parser for GNN files."""
        parsed = _GNNParseAccumulator(file_path)

        try:
            with open(file_path) as f:
                parsed.content = f.read()

            # Extract sections
            import re

            section_pattern = r"^#+\s+(.+)$"
            matches = re.finditer(section_pattern, parsed.content, re.MULTILINE)

            for match in matches:
                section_name = match.group(1).strip()
                parsed.add_section(section_name)

            # Extract variables
            var_patterns: list[Any] = [
                r"(\w+)\s*:\s*(\w+)",  # name: type
                r"(\w+)\s*=\s*([^;\n]+)",  # name = value
            ]

            for pattern in var_patterns:
                matches = re.finditer(pattern, parsed.content)
                for match in matches:
                    var_name = match.group(1)
                    var_value = match.group(2)
                    # Normalize dimensions: strip type=... parts and filter out empty
                    if isinstance(var_value, str) and "type=" in var_value:
                        dims = [
                            d
                            for d in var_value.split(",")
                            if not d.strip().startswith("type=")
                        ]
                        var_value = ",".join(dims)
                    parsed.add_variable(var_name, "", var_value)

            # Extract connections
            conn_patterns: list[Any] = [
                r"(\w+)\s*->\s*(\w+)",  # source -> target
                r"(\w+)\s*→\s*(\w+)",  # source → target
            ]

            for pattern in conn_patterns:
                matches = re.finditer(pattern, parsed.content)
                for match in matches:
                    source = match.group(1)
                    target = match.group(2)
                    parsed.add_connection(source, target)

        except Exception as e:
            parsed.add_error(f"Failed to parse file: {e}")

        return parsed


class GNNFormatSpec:
    """Represents a GNN format specification (MIME types, extensions).

    Note: For the GNNFormat Enum used throughout the pipeline, see
    gnn.parsers.common.GNNFormat.
    """

    def __init__(self) -> None:
        """Initialize the GNN format spec."""
        self.name = "GNN"
        self.version = "1.0"
        self.extensions = [".gnn", ".md"]
        self.mime_types = ["text/gnn", "text/markdown"]


class GNNFormalParser:
    """Section-oriented formal parser backed by the built-in GNN parser."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self._system = GNNParsingSystem()

    def parse_file(self, file_path: Union[str, Path]) -> Optional[_GNNParseAccumulator]:
        """Parse a GNN file into the formal accumulator representation."""
        return self._system.parse_file(file_path)

    def parse_content(
        self, content: str, source_name: str = "<string>"
    ) -> _GNNParseAccumulator:
        """Parse GNN content into the formal accumulator representation."""
        parsed = ParsedGNNFormal(source_name)
        parsed.content = content

        import re

        for match in re.finditer(r"^#+\s+(.+)$", content, re.MULTILINE):
            parsed.add_section(match.group(1).strip())

        for pattern in (r"(\w+)\s*:\s*(\w+)", r"(\w+)\s*=\s*([^;\n]+)"):
            for match in re.finditer(pattern, content):
                parsed.add_variable(match.group(1), "", match.group(2))

        for pattern in (r"(\w+)\s*->\s*(\w+)", r"(\w+)\s*→\s*(\w+)"):
            for match in re.finditer(pattern, content):
                parsed.add_connection(match.group(1), match.group(2))

        return parsed

    def validate_syntax(self, content: str) -> Tuple[bool, List[str]]:
        """Validate GNN content via the formal schema_validator pipeline."""
        return validate_gnn_syntax(content, ValidationLevel.STANDARD)

    def visualize_parse_tree(self, content: str) -> str:
        """Return a readable outline of parsed sections, variables, and connections."""
        parsed = self.parse_content(content)
        lines: list[Any] = ["GNN parse outline"]
        lines.append(f"Sections: {len(parsed.sections)}")
        lines.extend(f"  - {section['name']}" for section in parsed.sections)
        lines.append(f"Variables: {len(parsed.variables)}")
        lines.extend(
            f"  - {variable['name']}: {variable['value']}"
            for variable in parsed.variables
        )
        lines.append(f"Connections: {len(parsed.connections)}")
        lines.extend(
            f"  - {connection['source']} -> {connection['target']}"
            for connection in parsed.connections
        )
        return "\n".join(lines)


class ParsedGNNFormal(_GNNParseAccumulator):
    """Formal parse result using the same accumulator shape as file parsing."""

    def __init__(self, file_path: Union[str, Path] = "<string>") -> None:
        """Initialize the instance."""
        super().__init__(file_path)


def parse_gnn_formal(file_path: Union[str, Any]) -> Optional[_GNNParseAccumulator]:
    """Parse a GNN file using the formal parser facade."""
    return GNNFormalParser().parse_file(file_path)


def get_parse_tree_visualization(content: str) -> str:
    """Return the formal parser's text outline for GNN content."""
    return GNNFormalParser().visualize_parse_tree(content)


def validate_gnn_syntax(
    file_path_or_content: Union[str, Path],
    validation_level: Union[ValidationLevel, str, None] = ValidationLevel.STANDARD,
    *,
    is_content: bool = False,
    **kwargs: Any,
) -> Tuple[bool, List[str]]:
    """
    Validate GNN markdown text via the formal schema_validator pipeline.

    Delegates to ``GNNValidator.validate_file`` (``gnn.schema_validator``);
    returned messages are the formal validator's errors (for example
    ``"Required section missing: ModelName"``). The former in-function regex
    heuristics are removed: there is one validation path. Validity means the
    formal validator's normative markdown gate — every required section
    (``gnn.schemas.section_contract.REQUIRED_SECTIONS``) present with
    substantive body content — a deliberate tightening over the earlier
    heuristic that accepted any document with one header, one variable,
    and one connection. The validator's level ladder applies unchanged at
    every level: even BASIC now runs the required-section structure checks,
    where the previous body was a no-op below STANDARD.

    Args:
        file_path_or_content: Path to an existing GNN file, or GNN content
            as a string. An input is treated as a file path only when
            ``Path(...).exists()`` is true; a path probe that raises (for
            example a content string longer than ``PATH_MAX``) selects
            content mode. Both modes validate the same bytes identically:
            the bytes are staged as a temporary markdown document and run
            through the same formal pipeline, so a given document yields
            the same verdict whether passed by path or as content. Content
            is validated as a GNN markdown document; JSON/XML/YAML model
            files should be validated with
            ``gnn.schema_validator.validate_gnn_file_comprehensive``,
            which honors the file extension.
        validation_level: ``ValidationLevel`` member, an accepted level
            string (enum value or name, e.g. ``"strict"`` or ``"STRICT"``),
            or ``None`` to use the validator's default level (STANDARD).
            Unknown level strings raise ``ValueError`` instead of silently
            skipping validation.
        is_content: Treat the input as literal text, including Path objects
            and strings that happen to name an existing file. No source path
            classification is performed in this mode.
        **kwargs: Other options are accepted for backward compatibility and
            ignored.

    Returns:
        Tuple of (is_valid, formal_validator_errors). The second element is
        the formal pipeline's error list — empty exactly when the document
        is valid. Validator warnings and suggestions are not part of this
        contract.

    Long inputs are never truncated; validation cost is linear in the
    input size (a ~2 MB document validates in well under a second). The
    previous implementation crashed on such content: its path probe raised
    ``ENAMETOOLONG`` and its broad handler turned that into a bogus
    ``"Validation error: ..."`` tuple; the probe now routes such input
    to content mode.
    """
    # Function-local import: gnn.schema_validator.syntax imports gnn.parsers
    # at module level, and gnn.parsers imports this module; a module-level
    # import here would be an import cycle.
    from gnn.schema_validator.validator import GNNValidator

    del kwargs  # accepted for backward compatibility; unused

    source_path: Optional[Path] = None
    if (
        not is_content
        and isinstance(file_path_or_content, (str, Path))
        and str(file_path_or_content)  # "" resolves to the CWD; always content
    ):
        try:
            path_exists = Path(file_path_or_content).exists()
        except (OSError, ValueError):
            # pathlib's exists() propagates OSError for errnos it does not
            # ignore (e.g. ENAMETOOLONG for very long strings); such input
            # can only be content, never a usable path.
            path_exists = False
        if path_exists:
            source_path = Path(file_path_or_content)

    validator = GNNValidator()

    if source_path is not None:
        try:
            content = source_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # Undecodable bytes (pickle/binary) and unreadable paths go
            # through the validator's file path, which applies its own
            # binary-format and error handling.
            result = validator.validate_file(
                source_path, validation_level=validation_level
            )
            return result.is_valid, list(result.errors)
    else:
        content = str(file_path_or_content)

    # Stage both modes identically so path and content inputs of the same
    # bytes always produce the same verdict. surrogatepass keeps staging
    # lossless for strings with lone surrogates; the validator's strict
    # read then reports them as an encoding error rather than crashing.
    with tempfile.TemporaryDirectory() as tmp_dir:
        staged = Path(tmp_dir) / "content.md"
        staged.write_text(content, encoding="utf-8", errors="surrogatepass")
        result = validator.validate_file(staged, validation_level=validation_level)

    return result.is_valid, list(result.errors)


def _convert_parse_result_to_parsed_gnn(
    parse_result: Any, source_format: str = "unknown"
) -> Optional[ParsedGNN]:
    """
    Convert a ParseResult to a ParsedGNN dataclass object.

    Args:
        parse_result: ParseResult object from parser
        source_format: Format hint for the source (e.g., "markdown", "json")

    Returns:
        ParsedGNN object or None if parse_result is None
    """
    if parse_result is None:
        return None

    try:
        # Import types needed for conversion
        from gnn.parsers.common import ParseResult as ParseResultType
        from gnn.types import GNNConnection, GNNVariable, ParsedGNN

        # Verify it's a ParseResult
        if not isinstance(parse_result, ParseResultType):
            # If it's already a ParsedGNN, return as-is
            if isinstance(parse_result, ParsedGNN):
                return parse_result
            # Otherwise, create minimal representation
            return ParsedGNN(
                gnn_section=f"{source_format.upper()}GNN",
                version="1.0",
                model_name="Unknown",
                model_annotation="",
                variables={},
                connections=[],
                parameters={},
                equations=[],
                time_config={},
                ontology_mappings={},
                model_parameters={},
                footer="",
                source_format=source_format,
            )

        model = parse_result.model

        # Convert variables
        variables: dict[Any, Any] = {}
        for var in getattr(model, "variables", []):
            var_name = getattr(var, "name", "unknown")
            variables[var_name] = GNNVariable(
                name=var_name,
                dimensions=getattr(var, "dimensions", []),
                data_type=str(getattr(var, "data_type", "categorical")),
                description=getattr(var, "description", ""),
                ontology_mapping=getattr(var, "ontology_mapping", None),
            )

        # Helper function to infer connection symbol
        def _infer_symbol_from_type(connection_type: str) -> str:
            """Infer connection symbol from type."""
            type_lower = connection_type.lower()
            if "directed" in type_lower or "->" in type_lower:
                return ">"
            elif "undirected" in type_lower or "-" in type_lower:
                return "-"
            elif "conditional" in type_lower or "|" in type_lower:
                return "|"
            else:
                return ">"

        # Convert connections
        connections: list[Any] = []
        for conn in getattr(model, "connections", []):
            connections.append(
                GNNConnection(
                    source=getattr(
                        conn, "source_variables", getattr(conn, "source", [])
                    ),
                    target=getattr(
                        conn, "target_variables", getattr(conn, "target", [])
                    ),
                    connection_type=str(getattr(conn, "connection_type", "directed")),
                    symbol=_infer_symbol_from_type(
                        str(getattr(conn, "connection_type", "directed"))
                    ),
                    description=getattr(conn, "description", ""),
                )
            )

        # Convert parameters
        parameters: dict[Any, Any] = {}
        for param in getattr(model, "parameters", []):
            param_name = getattr(param, "name", "unknown")
            parameters[param_name] = getattr(param, "value", None)

        return ParsedGNN(
            gnn_section=getattr(model, "gnn_section", f"{source_format.upper()}GNN"),
            version=getattr(model, "version", "1.0"),
            model_name=getattr(model, "model_name", "Unknown"),
            model_annotation=getattr(model, "annotation", ""),
            variables=variables,
            connections=connections,
            parameters=parameters,
            equations=getattr(model, "equations", []),
            time_config=getattr(model, "time_config", {}),
            ontology_mappings=getattr(model, "ontology_mappings", {}),
            model_parameters=getattr(model, "model_parameters", {}),
            footer=getattr(model, "footer", ""),
            source_format=source_format,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert parse result from format '{source_format}': {e}"
        ) from e
