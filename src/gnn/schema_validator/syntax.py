"""
GNN syntax parsing for the schema_validator package.

Owns GNNParser: the regex-based parser that turns GNN source text into a
ParsedGNN structure. The multi-level validator lives in
gnn/schema_validator/validator.py and imports GNNParser from here.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional, Union, cast

from gnn.schema_validator.format_detection import FormatDetectionMixin
from gnn.schema_validator.section_parsers import SectionParsersMixin
from gnn.types import (
    GNNConnection,
    GNNFormat,
    GNNVariable,
    ParsedGNN,
    ValidationLevel,
    ValidationResult,
)

logger = logging.getLogger(__name__)

# Try to import round-trip testing capabilities
try:
    from gnn.parsers import GNNParsingSystem

    ROUND_TRIP_AVAILABLE = True
except ImportError as exc:
    ROUND_TRIP_AVAILABLE = False
    logger.warning(
        "Enhanced parser unavailable (%s: %s); using basic GNN parsing",
        type(exc).__name__,
        exc,
    )


class GNNParser(FormatDetectionMixin, SectionParsersMixin):
    """Enhanced parser for GNN file format with multi-format support."""

    # Regular expressions for GNN syntax elements (enhanced)
    SECTION_PATTERN = re.compile(r"^## (.+)$")
    VARIABLE_PATTERN = re.compile(
        r"^([\w_π][\w\d_π+]*)(\[([^\]]+)\])?(?:,type=([a-zA-Z]+))?(?:\s*#\s*(.*))?$"
    )
    CONNECTION_PATTERN = re.compile(r"^(.+?)\s*(>|->|-|\|)\s*(.+?)(?:\s*#\s*(.*))?$")
    # Accept both '=' and ':' as assignment separators to support previous files
    PARAMETER_PATTERN = re.compile(
        r"^([\w_π][\w\d_π+]*)(\s*[:=]\s*)(.+?)(?:\s*#\s*(.*))?$"
    )
    ONTOLOGY_PATTERN = re.compile(
        r"^([\w_π][\w\d_π+]*)(\s*=\s*)([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*#\s*(.*))?$"
    )
    COMMENT_PATTERN = re.compile(r"^\s*#\s*(.*)$")

    # Enhanced format detection patterns
    FORMAT_SIGNATURES = MappingProxyType(
        {
            "json": (r"^\s*\{", r'"model_name":', r'"variables":'),
            "xml": (r"^\s*<\?xml", r"<gnn.*>", r"<model"),
            "yaml": (r"^---", r"model_name:", r"variables:"),
            "binary": (b"\x80\x03", b"pickle", b"\x00\x00\x00"),  # Pickle signatures
        }
    )

    def __init__(self, enhanced_validation: bool = True) -> None:
        """Initialize the instance."""
        self.enhanced_validation = enhanced_validation
        self.parsing_system: Optional[GNNParsingSystem]
        if enhanced_validation and ROUND_TRIP_AVAILABLE:
            self.parsing_system = GNNParsingSystem()
            logger.info("Enhanced multi-format parsing system initialized")
        else:
            self.parsing_system = None
            logger.info("Basic GNN parser initialized")
        # Expose basic schema metadata expected by tests and previous callers
        try:
            validator = (
                getattr(self.parsing_system, "validator", None)
                if self.parsing_system
                else None
            )
            self.schema = getattr(validator, "schema", {})
        except Exception as e:
            logger.debug(f"Could not load schema from parsing system: {e}")
            self.schema = {}

    def validate_file(
        self,
        file_path: Union[str, Path],
        validation_level: Optional[ValidationLevel] = None,
    ) -> "ValidationResult":
        """Validate a GNN file via the full ``GNNValidator`` implementation."""
        from gnn.schema_validator.validator import GNNValidator

        validator = GNNValidator()
        return validator.validate_file(file_path, validation_level=validation_level)

    def parse_file(
        self, file_path: Union[str, Path], format_hint: Optional[str] = None
    ) -> ParsedGNN:
        """Enhanced file parsing with format detection and validation."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Detect format
        detected_format = format_hint or self._detect_file_format(file_path)

        # Handle different formats
        if detected_format == "binary":
            return self._parse_binary_file(file_path)
        else:
            # Read as text
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try as binary if UTF-8 fails
                return self._parse_binary_file(file_path)

        return self.parse_content(content, str(file_path), detected_format)

    def parse_content(
        self, content: str, source_name: str = "<string>", format_hint: str = "markdown"
    ) -> ParsedGNN:
        """Parse content with format-specific handling and visible degradation.

        Behaviour by ``format_hint``:

        * ``"markdown"`` — parsed directly by the built-in markdown parser.
        * unknown hint (not a :class:`GNNFormat` value) — raises
          :class:`ValueError`; silently re-parsing a mistyped hint as
          markdown hides the caller's error.
        * known hint whose parser fails or reports ``success=False`` —
          recovers to markdown parsing, but the returned
          :class:`ParsedGNN` carries ``metadata["parse_degraded"]``
          (reason, requested_format, fallback) and a warning is logged,
          so the degraded result is distinguishable from an honest parse
          of the requested format.
        """
        if self.parsing_system and format_hint != "markdown":
            from gnn.parsers.common import ParseError

            try:
                format_enum = GNNFormat(format_hint)
            except ValueError as exc:
                supported = ", ".join(f.value for f in GNNFormat)
                raise ValueError(
                    f"Unknown format_hint {format_hint!r} for parse_content "
                    f"(supported: {supported})"
                ) from exc

            try:
                result = self.parsing_system.parse_string(content, format_enum)
            except (ParseError, ValueError) as exc:
                # GNNParsingSystem.parse_string wraps parser failures in
                # ParseError and raises ValueError for unregistered formats;
                # both mean the requested format cannot handle this content.
                reason = f"{type(exc).__name__}: {exc}"
                if exc.__cause__ is not None:
                    cause = exc.__cause__
                    reason += f" (caused by {type(cause).__name__}: {cause})"
                return self._degraded_markdown_fallback(
                    content,
                    source_name,
                    format_hint,
                    reason,
                )
            if result.success:
                return self._convert_parse_result_to_parsed_gnn(result, format_hint)
            return self._degraded_markdown_fallback(
                content,
                source_name,
                format_hint,
                "; ".join(result.errors) or "format parser reported success=False",
            )

        return self._parse_markdown_content(content, source_name)

    def _degraded_markdown_fallback(
        self, content: str, source_name: str, format_hint: str, reason: str
    ) -> ParsedGNN:
        """Recover to markdown parsing, marking the result visibly degraded."""
        logger.warning(
            "parse_content: %s parsing failed for %s (%s); falling back to "
            "markdown parsing — result is degraded",
            format_hint,
            source_name,
            reason,
        )
        parsed = self._parse_markdown_content(content, source_name)
        parsed.metadata["parse_degraded"] = {
            "reason": reason,
            "requested_format": format_hint,
            "fallback": "markdown",
        }
        return parsed

    def _convert_parse_result_to_parsed_gnn(
        self, result: Any, source_format: str
    ) -> ParsedGNN:
        """Convert ParseResult to ParsedGNN format."""
        model = result.model

        # Convert variables
        variables: dict[Any, Any] = {}
        for var in model.variables:
            data_type = getattr(var, "data_type", "categorical")
            variables[var.name] = GNNVariable(
                name=var.name,
                dimensions=getattr(var, "dimensions", []),
                data_type=str(getattr(data_type, "value", data_type)),
                description=getattr(var, "description", ""),
                ontology_mapping=getattr(var, "ontology_mapping", None),
            )

        # Convert connections
        connections: list[Any] = []
        for conn in model.connections:
            connections.append(
                GNNConnection(
                    source=getattr(conn, "source_variables", []),
                    target=getattr(conn, "target_variables", []),
                    connection_type=str(getattr(conn, "connection_type", "directed")),
                    symbol=self._infer_symbol_from_type(
                        str(getattr(conn, "connection_type", "directed"))
                    ),
                    description=getattr(conn, "description", ""),
                )
            )

        # Convert parameters
        parameters: dict[Any, Any] = {}
        for param in model.parameters:
            parameters[param.name] = param.value

        return ParsedGNN(
            gnn_section=getattr(model, "gnn_section", f"{source_format.upper()}GNN"),
            version=getattr(model, "version", "1.0"),
            model_name=model.model_name,
            model_annotation=model.annotation,
            variables=variables,
            connections=connections,
            parameters=parameters,
            equations=getattr(model, "equations", []),
            time_config=getattr(model, "time_config", {}),
            ontology_mappings=getattr(model, "ontology_mappings", {}),
            model_parameters=getattr(model, "model_parameters", {}),
            footer=getattr(model, "footer", ""),
            source_format=source_format,
            semantic_checksum=self._compute_semantic_checksum(model),
        )

    def _infer_symbol_from_type(self, connection_type: str) -> str:
        """Infer connection symbol from type."""
        symbol_map: dict[str, Any] = {
            "directed": ">",
            "undirected": "-",
            "conditional": "|",
            "bidirectional": "<->",
        }
        return cast("str", symbol_map.get(connection_type, ">"))

    def _compute_semantic_checksum(self, model: Any) -> str:
        """Compute semantic checksum for model."""
        # Create normalized representation
        checksum_data: dict[str, Any] = {
            "model_name": model.model_name,
            "variables": sorted([var.name for var in model.variables]),
            "connections_count": len(model.connections),
            "parameters_count": len(model.parameters),
        }

        checksum_str = json.dumps(checksum_data, sort_keys=True)
        return hashlib.md5(checksum_str.encode(), usedforsecurity=False).hexdigest()
