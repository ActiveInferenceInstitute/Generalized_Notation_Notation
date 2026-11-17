#!/usr/bin/env python3
"""
GNN Parser System module for GNN Processing Pipeline.

This module provides the main parsing system functionality.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, cast

from .alloy_serializer import AlloySerializer
from .asn1_serializer import ASN1Serializer
from .binary_parser import PickleGNNParser
from .binary_serializer import BinarySerializer
from .common import (
    GNNInternalRepresentation,
    GNNParser,
    ParseError,
    detect_gnn_format_from_path,
)

# Import converters and validators
from .converters import FormatConverter
from .coq_parser import CoqGNNParser
from .coq_serializer import CoqSerializer
from .functional_parser import HaskellGNNParser
from .functional_serializer import FunctionalSerializer
from .grammar_parser import BNFParser, EBNFParser
from .grammar_serializer import GrammarSerializer
from .isabelle_parser import IsabelleParser
from .isabelle_serializer import IsabelleSerializer
from .json_parser import JSONGNNParser
from .json_serializer import JSONSerializer
from .lean_parser import LeanGNNParser
from .lean_serializer import LeanSerializer
from .markdown_parser import MarkdownGNNParser

# Import serializers from individual files
from .markdown_serializer import MarkdownSerializer
from .maxima_parser import MaximaParser
from .maxima_serializer import MaximaSerializer
from .pkl_serializer import PKLSerializer
from .protobuf_parser import ProtobufGNNParser
from .protobuf_serializer import ProtobufSerializer
from .python_parser import PythonGNNParser
from .python_serializer import PythonSerializer
from .scala_parser import ScalaGNNParser
from .scala_serializer import ScalaSerializer
from .schema_parser import (
    AlloyParser,
    ASN1Parser,
    PKLParser,
    XSDParser,
    ZNotationParser,
)
from .temporal_parser import AgdaParser, TLAParser
from .temporal_serializer import AgdaSerializer, TLASerializer

# Import all parser modules
from .unified_parser import GNNFormat, ParseResult
from .validators import InternalModelValidator
from .xml_parser import PNMLParser, XMLGNNParser
from .xml_serializer import XMLSerializer
from .xsd_serializer import XSDSerializer
from .yaml_parser import YAMLGNNParser
from .yaml_serializer import YAMLSerializer
from .znotation_serializer import ZNotationSerializer

# Configure logging
logger = logging.getLogger(__name__)

# Registry of all available parsers and serializers
PARSER_REGISTRY: Dict[GNNFormat, Type["GNNParser"]] = {
    GNNFormat.MARKDOWN: MarkdownGNNParser,
    GNNFormat.SCALA: ScalaGNNParser,
    GNNFormat.LEAN: LeanGNNParser,
    GNNFormat.COQ: CoqGNNParser,
    GNNFormat.PYTHON: PythonGNNParser,
    GNNFormat.BNF: BNFParser,
    GNNFormat.EBNF: EBNFParser,
    GNNFormat.ISABELLE: IsabelleParser,
    GNNFormat.MAXIMA: MaximaParser,
    GNNFormat.XML: XMLGNNParser,
    GNNFormat.PNML: PNMLParser,
    GNNFormat.JSON: JSONGNNParser,
    GNNFormat.PROTOBUF: ProtobufGNNParser,
    GNNFormat.YAML: YAMLGNNParser,
    GNNFormat.XSD: XSDParser,
    GNNFormat.ASN1: ASN1Parser,
    GNNFormat.PKL: PKLParser,
    GNNFormat.ALLOY: AlloyParser,
    GNNFormat.Z_NOTATION: ZNotationParser,
    GNNFormat.TLA_PLUS: TLAParser,
    GNNFormat.AGDA: AgdaParser,
    GNNFormat.HASKELL: HaskellGNNParser,
    GNNFormat.PICKLE: PickleGNNParser,
}

SERIALIZER_REGISTRY: Dict[GNNFormat, Type] = {
    GNNFormat.MARKDOWN: MarkdownSerializer,
    GNNFormat.JSON: JSONSerializer,
    GNNFormat.XML: XMLSerializer,
    GNNFormat.YAML: YAMLSerializer,
    GNNFormat.SCALA: ScalaSerializer,
    GNNFormat.PROTOBUF: ProtobufSerializer,
    GNNFormat.PKL: PKLSerializer,
    GNNFormat.XSD: XSDSerializer,
    GNNFormat.ASN1: ASN1Serializer,
    GNNFormat.LEAN: LeanSerializer,
    GNNFormat.COQ: CoqSerializer,
    GNNFormat.PYTHON: PythonSerializer,
    GNNFormat.BNF: GrammarSerializer,
    GNNFormat.EBNF: GrammarSerializer,
    GNNFormat.ISABELLE: IsabelleSerializer,
    GNNFormat.MAXIMA: MaximaSerializer,
    GNNFormat.ALLOY: AlloySerializer,
    GNNFormat.Z_NOTATION: ZNotationSerializer,
    GNNFormat.TLA_PLUS: TLASerializer,
    GNNFormat.AGDA: AgdaSerializer,
    GNNFormat.HASKELL: FunctionalSerializer,
    GNNFormat.PICKLE: BinarySerializer,
}


class GNNParsingSystem:
    """
    Unified GNN Parsing System

    Provides comprehensive parsing, serialization, and conversion capabilities
    for all supported GNN specification formats.
    """

    def __init__(self, strict_validation: bool = True) -> None:
        """
        Initialize the parsing system.

        Args:
            strict_validation: Attach internal model validation diagnostics
                to successful file and string parses.
        """
        self.strict_validation = strict_validation
        self.parsers: Dict[GNNFormat, GNNParser] = {}
        self.serializers: Dict[GNNFormat, Any] = {}
        self.converter = FormatConverter()
        self.validator = InternalModelValidator()

        self._initialize_parsers()
        self._initialize_serializers()

    def _initialize_parsers(self) -> Any:
        """Initialize all available parsers."""
        for format_type, parser_class in PARSER_REGISTRY.items():
            try:
                self.parsers[format_type] = parser_class()
            except Exception as e:
                logger.warning(f"Failed to initialize parser for {format_type}: {e}")

    def _initialize_serializers(self) -> Any:
        """Initialize all available serializers."""
        for format_type, serializer_class in SERIALIZER_REGISTRY.items():
            try:
                self.serializers[format_type] = serializer_class()
            except Exception as e:
                logger.warning(
                    f"Failed to initialize serializer for {format_type}: {e}"
                )

    def parse_file(
        self, file_path: Union[str, Path], format_hint: Optional[GNNFormat] = None
    ) -> ParseResult:
        """
        Parse a GNN file.

        Args:
            file_path: Path to the file to parse
            format_hint: Optional format hint

        Returns:
            ParseResult with the parse outcome and any model validation diagnostics.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Detect format if not provided
        if format_hint is None:
            format_hint = self._detect_format(file_path)

        # Delegate to the concrete parser's file method to avoid double-reading
        parser = self.parsers.get(format_hint)
        if parser is None:
            raise ValueError(f"Unsupported format: {format_hint}")
        # Ensure parsers accept either str or Path
        if hasattr(parser, "parse_file"):
            result = parser.parse_file(str(file_path))
        else:
            # Recovery: call parse_string on file content
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            result = parser.parse_string(content)
        result.model.source_format = format_hint
        result.source_file = str(file_path)
        self._record_model_validation(result, format_hint)
        return result

    def parse_string(self, content: str, format: GNNFormat) -> ParseResult:
        """
        Parse GNN content from string.

        Args:
            content: String content to parse
            format: Format of the content

        Returns:
            ParseResult with the parse outcome and any model validation diagnostics.
        """
        if format not in self.parsers:
            raise ValueError(f"Unsupported format: {format}")

        parser = self.parsers[format]

        try:
            # Parse the content using the parser's string API
            result = parser.parse_string(content)

        except Exception as e:
            logger.error(f"Parsing failed for format {format}: {e}")
            raise ParseError(f"Failed to parse {format} content: {e}") from e

        self._record_model_validation(result, format)
        return result

    def _record_model_validation(self, result: ParseResult, format: GNNFormat) -> None:
        """Attach semantic checks without changing the parse outcome."""
        if not self.strict_validation or not result.success or result.model is None:
            return

        from gnn.types.definitions import ValidationLevel, ValidationResult

        report = ValidationResult(
            is_valid=False,
            validation_level=ValidationLevel.STRICT,
            format_tested=format.value,
            metadata={"available": True},
        )
        try:
            validation = self.validator.validate(result.model)
        except (
            AttributeError,
            LookupError,
            RecursionError,
            TypeError,
            ValueError,
        ) as exc:
            report.metadata["available"] = False
            report.errors.append(f"Internal model validation unavailable: {exc}")
        else:
            report.is_valid = validation.is_valid
            report.errors = [str(issue) for issue in validation.errors]
            report.warnings = [str(issue) for issue in validation.warnings]
        result.validation_result = report

        if report.errors:
            logger.warning("Internal model validation: %s", report.errors)

    def convert(
        self,
        model: GNNInternalRepresentation,
        from_format: GNNFormat,
        to_format: GNNFormat,
    ) -> GNNInternalRepresentation:
        """
        Convert a model from one format to another.

        Args:
            model: The model to convert
            from_format: Source format
            to_format: Target format

        Returns:
            Converted model
        """
        return cast(
            "GNNInternalRepresentation",
            self.converter.convert(model, from_format.value, to_format.value),
        )

    def serialize(self, model: GNNInternalRepresentation, format: GNNFormat) -> str:
        """
        Serialize a model to string.

        Args:
            model: The model to serialize
            format: Target format

        Returns:
            Serialized string
        """
        if format not in self.serializers:
            raise ValueError(f"Unsupported format: {format}")

        serializer = self.serializers[format]

        try:
            return cast("str", serializer.serialize(model))
        except Exception as e:
            logger.error(f"Serialization failed for format {format}: {e}")
            raise ParseError(f"Failed to serialize to {format}: {e}") from e

    def serialize_to_file(
        self,
        model: GNNInternalRepresentation,
        file_path: Union[str, Path],
        format: Optional[GNNFormat] = None,
    ) -> Any:
        """
        Serialize a model to file.

        Args:
            model: The model to serialize
            file_path: Output file path
            format: Optional format hint
        """
        file_path = Path(file_path)

        # Detect format from file extension if not provided
        if format is None:
            format = self._detect_format(file_path)

        # Serialize to string
        content = self.serialize(model, format)

        # Write to file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def convert_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        from_format: Optional[GNNFormat] = None,
        to_format: Optional[GNNFormat] = None,
    ) -> Path:
        """
        Convert a GNN file from one format to another.

        Parses ``input_path`` (format resolved from ``from_format`` or the file
        extension) and serializes it to ``output_path`` (format resolved from
        ``to_format`` or the output extension).

        Args:
            input_path: Path to the source GNN file
            output_path: Path for the converted output file
            from_format: Optional explicit source format
            to_format: Optional explicit target format

        Returns:
            The resolved output path

        Raises:
            FileNotFoundError: If the input file does not exist
            ValueError: If the source/target format is unsupported or the
                output extension is unrecognized
            ParseError: If parsing or serialization fails
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        result = self.parse_file(input_path, format_hint=from_format)
        if not result.success:
            raise ParseError(
                f"Failed to parse {input_path}: {'; '.join(result.errors)}"
            )

        if to_format is None:
            target_format = detect_gnn_format_from_path(output_path)
        else:
            target_format = to_format
        if target_format not in self.serializers:
            raise ValueError(f"Unsupported target format: {target_format}")

        self.serialize_to_file(result.model, output_path, target_format)
        return output_path

    def _detect_format(self, file_path: Path) -> GNNFormat:
        """
        Detect format from file extension.

        Args:
            file_path: Path to the file

        Returns:
            Detected format
        """
        return detect_gnn_format_from_path(file_path)

    def _detect_format_from_content(self, content: str) -> GNNFormat:
        """
        Detect format from content.

        Args:
            content: File content

        Returns:
            Detected format
        """
        # Simple heuristics for format detection
        if content.startswith("<?xml"):
            return GNNFormat.XML
        elif content.startswith("{") or content.startswith("["):
            return GNNFormat.JSON
        elif "---" in content[:100]:
            return GNNFormat.YAML
        elif "syntax" in content.lower() or "grammar" in content.lower():
            return GNNFormat.BNF
        elif "theorem" in content.lower() or "lemma" in content.lower():
            return GNNFormat.LEAN
        elif "import" in content.lower() and "scala" in content.lower():
            return GNNFormat.SCALA
        elif "import" in content.lower() and "python" in content.lower():
            return GNNFormat.PYTHON
        else:
            # Default to markdown for unknown formats
            return GNNFormat.MARKDOWN

    def get_supported_formats(self) -> List[GNNFormat]:
        """Get list of supported formats that can be parsed and serialized."""
        return [fmt for fmt in self.parsers.keys() if fmt in self.serializers]

    def get_available_parsers(self) -> Dict[GNNFormat, str]:
        """Get available parsers with their descriptions."""
        return {
            format: parser.__class__.__name__ for format, parser in self.parsers.items()
        }

    def get_available_serializers(self) -> Dict[GNNFormat, str]:
        """Get available serializers with their descriptions."""
        return {
            format: serializer.__class__.__name__
            for format, serializer in self.serializers.items()
        }
