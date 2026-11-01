"""
Common Infrastructure for GNN Parsers

This module provides the shared infrastructure used by all GNN parsers,
including the internal representation, AST nodes, and base classes.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import logging
import re
import uuid
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    cast,
)

if TYPE_CHECKING:
    from gnn.types import ValidationResult

from gnn.parsers.converters import ConversionError as _CanonicalConversionError

logger = logging.getLogger(__name__)

# Type variables for generic types
T = TypeVar("T")
E = TypeVar("E", bound=Enum)
K = TypeVar("K")
V = TypeVar("V")

# ================================
# MARKDOWN SECTION HEADINGS
# ================================

#: Canonical ``## `` section names of the GNN markdown format. A heading that
#: differs from one of these only by whitespace names the same section, so
#: ``## ActInf Ontology Annotation`` and ``## ActInfOntologyAnnotation`` are one
#: section in every reader -- not only in :class:`~gnn.parsers.markdown_parser.MarkdownGNNParser`.
GNN_SECTION_HEADINGS: tuple[str, ...] = (
    "GNNSection",
    "GNNVersionAndFlags",
    "ModelName",
    "ModelAnnotation",
    "StateSpaceBlock",
    "Connections",
    "InitialParameterization",
    "Equations",
    "Time",
    "ActInfOntologyAnnotation",
    "ModelParameters",
    "Footer",
    "Signature",
)


def collapse_heading_whitespace(heading: str) -> str:
    """Collapse whitespace so two heading spellings compare structurally."""
    return re.sub(r"\s+", "", heading)


def match_section_heading(heading: str) -> Optional[str]:
    """Return the canonical section name ``heading`` names, else ``None``.

    Only whitespace differences match. Any other spelling difference returns
    ``None`` so that an unrecognized heading stays visible as itself instead of
    being folded into a section it does not name.
    """
    collapsed = collapse_heading_whitespace(heading)
    for name in GNN_SECTION_HEADINGS:
        if collapse_heading_whitespace(name) == collapsed:
            return name
    return None


def canonical_section_heading(heading: str) -> str:
    """Return the canonical name for a known section heading, else ``heading``.

    Readers use this to key sections by the same names the GNN parser uses,
    whatever whitespace spelling the source document declares.
    """
    return match_section_heading(heading) or heading


# ================================
# EXCEPTIONS AND ERRORS
# ================================


class ParseError(Exception):
    """Base exception for parsing errors."""

    def __init__(
        self,
        message: str,
        line: Optional[int] = None,
        column: Optional[int] = None,
        source: Optional[str] = None,
    ) -> None:
        """Initialize the instance."""
        self.message = message
        self.line = line
        self.column = column
        self.source = source
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with location information."""
        msg = self.message
        if self.source:
            msg = f"{self.source}: {msg}"
        if self.line is not None:
            msg = f"Line {self.line}: {msg}"
            if self.column is not None:
                msg = f"Line {self.line}, Column {self.column}: {msg}"
        return msg


class ValidationError(Exception):
    """Exception for validation errors."""


class ValidationWarning(Warning):
    """Warning for validation issues."""


class ConversionError(_CanonicalConversionError):
    """Earlier ConversionError, kept for the gnn.parsers.converters migration."""

    def __init__(self, *args: object) -> None:
        warnings.warn(
            "gnn.parsers.common.ConversionError is the earlier name; import from "
            "gnn.parsers.converters instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args)


# ================================
# ENUMS AND CONSTANTS
# ================================


class GNNFormat(Enum):
    """Enumeration of all supported GNN formats."""

    MARKDOWN = "markdown"
    SCALA = "scala"
    LEAN = "lean"
    COQ = "coq"
    PYTHON = "python"
    BNF = "bnf"
    EBNF = "ebnf"
    ISABELLE = "isabelle"
    MAXIMA = "maxima"
    XML = "xml"
    PNML = "pnml"
    JSON = "json"
    PROTOBUF = "protobuf"
    YAML = "yaml"
    XSD = "xsd"
    ASN1 = "asn1"
    PKL = "pkl"
    ALLOY = "alloy"
    Z_NOTATION = "z_notation"
    TLA_PLUS = "tla_plus"
    AGDA = "agda"
    HASKELL = "haskell"
    PICKLE = "pickle"


FORMAT_EXTENSION_MAP: Dict[str, GNNFormat] = {
    ".md": GNNFormat.MARKDOWN,
    ".markdown": GNNFormat.MARKDOWN,
    ".gnn": GNNFormat.MARKDOWN,
    ".scala": GNNFormat.SCALA,
    ".lean": GNNFormat.LEAN,
    ".v": GNNFormat.COQ,
    ".py": GNNFormat.PYTHON,
    ".bnf": GNNFormat.BNF,
    ".ebnf": GNNFormat.EBNF,
    ".thy": GNNFormat.ISABELLE,
    ".mac": GNNFormat.MAXIMA,
    ".max": GNNFormat.MAXIMA,
    ".xml": GNNFormat.XML,
    ".pnml": GNNFormat.PNML,
    ".json": GNNFormat.JSON,
    ".proto": GNNFormat.PROTOBUF,
    ".yaml": GNNFormat.YAML,
    ".yml": GNNFormat.YAML,
    ".xsd": GNNFormat.XSD,
    ".asn1": GNNFormat.ASN1,
    ".pkl": GNNFormat.PKL,
    ".pickle": GNNFormat.PICKLE,
    ".als": GNNFormat.ALLOY,
    ".z": GNNFormat.Z_NOTATION,
    ".zed": GNNFormat.Z_NOTATION,
    ".tla": GNNFormat.TLA_PLUS,
    ".agda": GNNFormat.AGDA,
    ".hs": GNNFormat.HASKELL,
}

FORMAT_OUTPUT_EXTENSION_MAP: Dict[GNNFormat, str] = {
    GNNFormat.MARKDOWN: ".md",
    GNNFormat.SCALA: ".scala",
    GNNFormat.LEAN: ".lean",
    GNNFormat.COQ: ".v",
    GNNFormat.PYTHON: ".py",
    GNNFormat.BNF: ".bnf",
    GNNFormat.EBNF: ".ebnf",
    GNNFormat.ISABELLE: ".thy",
    GNNFormat.MAXIMA: ".max",
    GNNFormat.XML: ".xml",
    GNNFormat.PNML: ".pnml",
    GNNFormat.JSON: ".json",
    GNNFormat.PROTOBUF: ".proto",
    GNNFormat.YAML: ".yaml",
    GNNFormat.XSD: ".xsd",
    GNNFormat.ASN1: ".asn1",
    GNNFormat.PKL: ".pkl",
    GNNFormat.PICKLE: ".pickle",
    GNNFormat.ALLOY: ".als",
    GNNFormat.Z_NOTATION: ".z",
    GNNFormat.TLA_PLUS: ".tla",
    GNNFormat.AGDA: ".agda",
    GNNFormat.HASKELL: ".hs",
}


def is_binary_pickle_payload(file_path: Union[str, Path]) -> bool:
    """Return True when a path clearly contains binary pickle bytes."""
    path = Path(file_path)
    try:
        sample = path.read_bytes()[:16]
    except OSError:
        return False
    return sample.startswith(b"\x80") or b"\x00" in sample


def detect_gnn_format_from_path(file_path: Union[str, Path]) -> GNNFormat:
    """Detect a GNN format from path extension and deterministic pickle routing."""
    path = Path(file_path)
    extension = path.suffix.lower()
    if extension == ".pkl" and is_binary_pickle_payload(path):
        logger.warning(
            "Binary pickle payload detected in .pkl file %s; prefer .pickle for binary pickle inputs.",
            path,
        )
        return GNNFormat.PICKLE
    try:
        return FORMAT_EXTENSION_MAP[extension]
    except KeyError as exc:
        raise ValueError(f"Unknown file extension: {extension}") from exc


def get_supported_gnn_extensions(include_binary_pickle: bool = True) -> List[str]:
    """Return registered input extensions for GNN model discovery."""
    extensions = sorted(FORMAT_EXTENSION_MAP)
    if not include_binary_pickle:
        extensions = [ext for ext in extensions if ext != ".pickle"]
    return extensions


def get_extension_for_format(format_type: GNNFormat) -> str:
    """Return the canonical output extension for a GNN format."""
    return FORMAT_OUTPUT_EXTENSION_MAP.get(format_type, f".{format_type.value}")


class VariableType(Enum):
    """Types of variables in Active Inference."""

    HIDDEN_STATE = "hidden_state"
    OBSERVATION = "observation"
    ACTION = "action"
    POLICY = "policy"
    LIKELIHOOD_MATRIX = "likelihood_matrix"
    TRANSITION_MATRIX = "transition_matrix"
    PREFERENCE_VECTOR = "preference_vector"
    PRIOR_VECTOR = "prior_vector"


class DataType(Enum):
    """Data types for GNN variables."""

    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    BINARY = "binary"
    INTEGER = "integer"
    FLOAT = "float"
    COMPLEX = "complex"


class ConnectionType(Enum):
    """Types of connections between variables."""

    DIRECTED = "directed"
    UNDIRECTED = "undirected"
    CONDITIONAL = "conditional"
    BIDIRECTIONAL = "bidirectional"


# ================================
# AST NODE HIERARCHY
# ================================


@dataclass
class ASTNode:
    """Base class for all AST nodes."""

    node_type: str = ""
    source_location: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> Any:
        """Normalize fields after dataclass initialization."""
        if not self.node_type:
            self.node_type = self.__class__.__name__

    def get_children(self) -> List["ASTNode"]:
        """Get all child nodes."""
        children: list[Any] = []
        for value in self.__dict__.values():
            if isinstance(value, ASTNode):
                children.append(value)
            elif isinstance(value, list):
                children.extend([item for item in value if isinstance(item, ASTNode)])
        return children

    def accept(self, visitor: "ASTVisitor") -> Any:
        """Accept a visitor for traversal."""
        return visitor.visit(self)


@dataclass
class Variable(ASTNode):
    """AST node for variable definitions."""

    name: str = ""
    var_type: VariableType = VariableType.HIDDEN_STATE
    dimensions: List[int] = field(default_factory=list)
    data_type: DataType = DataType.CATEGORICAL
    description: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> Any:
        """Normalize fields after dataclass initialization."""
        super().__post_init__()
        self.node_type = "Variable"


@dataclass
class Connection(ASTNode):
    """AST node for connections between variables."""

    source_variables: List[str] = field(default_factory=list)
    target_variables: List[str] = field(default_factory=list)
    connection_type: ConnectionType = ConnectionType.DIRECTED
    weight: Optional[float] = None
    annotation: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self) -> Any:
        """Normalize fields after dataclass initialization."""
        super().__post_init__()
        self.node_type = "Connection"


@dataclass
class Parameter(ASTNode):
    """AST node for parameter assignments."""

    name: str = ""
    value: Any = None
    type_hint: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self) -> Any:
        """Normalize fields after dataclass initialization."""
        super().__post_init__()
        self.node_type = "Parameter"


@dataclass
class Equation(ASTNode):
    """AST node for mathematical equations."""

    label: Optional[str] = None
    content: str = ""
    format: str = "latex"  # latex, mathml, ascii
    description: Optional[str] = None

    def __post_init__(self) -> Any:
        """Normalize fields after dataclass initialization."""
        super().__post_init__()
        self.node_type = "Equation"


@dataclass
class TimeSpecification(ASTNode):
    """AST node for time configuration."""

    time_type: str = "Static"  # "Static", "Dynamic"
    discretization: Optional[str] = None  # "DiscreteTime", "ContinuousTime"
    horizon: Optional[Union[int, str]] = None
    step_size: Optional[float] = None

    def __post_init__(self) -> Any:
        """Normalize fields after dataclass initialization."""
        super().__post_init__()
        self.node_type = "TimeSpecification"


@dataclass
class OntologyMapping(ASTNode):
    """AST node for ontology mappings."""

    variable_name: str = ""
    ontology_term: str = ""
    description: Optional[str] = None

    def __post_init__(self) -> Any:
        """Normalize fields after dataclass initialization."""
        super().__post_init__()
        self.node_type = "OntologyMapping"


@dataclass
class Section(ASTNode):
    """AST node for GNN sections."""

    section_name: str = ""
    content: List[ASTNode] = field(default_factory=list)
    raw_content: Optional[str] = None

    def __post_init__(self) -> Any:
        """Normalize fields after dataclass initialization."""
        super().__post_init__()
        self.node_type = "Section"


# ================================
# INTERNAL REPRESENTATION
# ================================


@dataclass
class GNNInternalRepresentation:
    """
    Unified internal representation for GNN models across all formats.

    This is the canonical representation that all parsers convert to
    and all serializers convert from.
    """

    # Core identification
    model_name: str
    version: str = "1.0"
    annotation: str = ""

    # Model structure
    variables: List[Variable] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    parameters: List[Parameter] = field(default_factory=list)
    equations: List[Equation] = field(default_factory=list)
    time_specification: Optional[TimeSpecification] = None
    ontology_mappings: List[OntologyMapping] = field(default_factory=list)

    # Metadata
    source_format: Optional[GNNFormat] = None
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    checksum: Optional[str] = None

    # Extension points
    extensions: Dict[str, Any] = field(default_factory=dict)
    raw_sections: Dict[str, str] = field(default_factory=dict)

    def get_variables_by_type(self, var_type: VariableType) -> List[Variable]:
        """Get all variables of a specific type."""
        return [var for var in self.variables if var.var_type == var_type]

    def get_variable_by_name(self, name: str) -> Optional[Variable]:
        """Get a variable by name."""
        for var in self.variables:
            if var.name == name:
                return var
        return None

    def get_connections_for_variable(self, variable_name: str) -> List[Connection]:
        """Get all connections involving a specific variable."""
        connections: list[Any] = []
        for conn in self.connections:
            if (
                variable_name in conn.source_variables
                or variable_name in conn.target_variables
            ):
                connections.append(conn)
        return connections

    def get_parameter_by_name(self, name: str) -> Optional[Parameter]:
        """Get a parameter by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None

    def validate_structure(self) -> List[str]:
        """Validate the internal structure and return any issues."""
        issues: list[Any] = []

        # Check required fields
        if not self.model_name:
            issues.append("Model name is required")

        # Check variable name uniqueness
        variable_names = [var.name for var in self.variables]
        if len(variable_names) != len(set(variable_names)):
            issues.append("Variable names must be unique")

        # Check connection references
        for conn in self.connections:
            for var_name in conn.source_variables + conn.target_variables:
                if not self.get_variable_by_name(var_name):
                    issues.append(f"Connection references unknown variable: {var_name}")

        # Check ontology mapping references
        for mapping in self.ontology_mappings:
            if not self.get_variable_by_name(mapping.variable_name):
                issues.append(
                    f"Ontology mapping references unknown variable: {mapping.variable_name}"
                )

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""

        def serialize_obj(obj: Any) -> Any:
            """Helper to serialize objects with enums to dict."""
            if isinstance(obj, Enum):
                return obj.value
            elif hasattr(obj, "__dict__"):
                result: dict[Any, Any] = {}
                for key, value in obj.__dict__.items():
                    if isinstance(value, Enum):
                        result[key] = value.value
                    elif isinstance(value, list):
                        result[key] = [
                            serialize_obj(item)
                            if hasattr(item, "__dict__")
                            else item.value
                            if isinstance(item, Enum)
                            else item
                            for item in value
                        ]
                    elif isinstance(value, dict):
                        result[key] = {
                            k: serialize_obj(v)
                            if hasattr(v, "__dict__")
                            else v.value
                            if isinstance(v, Enum)
                            else v
                            for k, v in value.items()
                        }
                    elif hasattr(value, "__dict__"):
                        result[key] = serialize_obj(value)
                    else:
                        result[key] = value
                return result
            else:
                return obj

        result: dict[str, Any] = {
            "model_name": self.model_name,
            "version": self.version,
            "annotation": self.annotation,
            "variables": [serialize_obj(var) for var in self.variables],
            "connections": [serialize_obj(conn) for conn in self.connections],
            "parameters": [serialize_obj(param) for param in self.parameters],
            "equations": [serialize_obj(eq) for eq in self.equations],
            "time_specification": serialize_obj(self.time_specification)
            if self.time_specification
            else None,
            "ontology_mappings": [
                serialize_obj(mapping) for mapping in self.ontology_mappings
            ],
            "source_format": self.source_format.value if self.source_format else None,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "checksum": self.checksum,
            "extensions": self.extensions,
            "raw_sections": self.raw_sections,
        }

        # Renderer-facing parameter views.
        if self.parameters:
            merged = {str(p.name).strip(): p.value for p in self.parameters if p.name}
            result["initialparameterization"] = merged
            result["model_parameters"] = merged

        return result


# ================================
# VISITOR PATTERN
# ================================


class ASTVisitor(ABC):
    """Abstract base class for AST visitors."""

    @abstractmethod
    def visit(self, node: ASTNode) -> Any:
        """Visit an AST node."""

    def visit_children(self, node: ASTNode) -> List[Any]:
        """Visit all children of a node."""
        return [child.accept(self) for child in node.get_children()]


class PrintVisitor(ASTVisitor):
    """Visitor that prints the AST structure."""

    def __init__(self, indent: int = 0) -> None:
        """Initialize the instance."""
        self.indent = indent

    def visit(self, node: ASTNode) -> str:
        """Visit and print a node."""
        indent_str = "  " * self.indent
        result = f"{indent_str}{node.node_type}"

        if hasattr(node, "name"):
            result += f": {node.name}"
        elif hasattr(node, "label") and node.label:
            result += f": {node.label}"

        # Visit children with increased indentation
        child_visitor = PrintVisitor(self.indent + 1)
        for child in node.get_children():
            result += "\n" + child.accept(child_visitor)

        return result


# ================================
# PARSER PROTOCOLS AND INTERFACES
# ================================


class GNNParser(Protocol):
    """Protocol defining the interface for GNN parsers."""

    def parse_file(self, file_path: str) -> "ParseResult":
        """Parse a GNN file and return the result."""
        ...

    def parse_string(self, content: str) -> "ParseResult":
        """Parse GNN content from a string."""
        ...

    def get_supported_extensions(self) -> List[str]:
        """Get file extensions supported by this parser."""
        ...


class GNNSerializer(Protocol):
    """Protocol defining the interface for GNN serializers."""

    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Serialize a GNN model to string format."""
        ...

    def serialize_to_file(
        self, model: GNNInternalRepresentation, file_path: str
    ) -> None:
        """Serialize a GNN model to a file."""
        ...


# ================================
# PARSE RESULT
# ================================


@dataclass
class ParseResult:
    """Result of parsing a GNN specification."""

    model: GNNInternalRepresentation
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    parse_time: Optional[float] = None
    source_file: Optional[str] = None
    validation_result: Optional["ValidationResult"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return bool(self.errors)

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return bool(self.warnings)

    def add_error(self, error: str) -> Any:
        """Add an error to the result."""
        self.errors.append(error)
        self.success = False

    def add_warning(self, warning: str) -> Any:
        """Add a warning to the result."""
        self.warnings.append(warning)


# ================================
# BASE PARSER IMPLEMENTATION
# ================================


class BaseGNNParser(ABC):
    """Abstract base class for GNN parsers."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.current_file: Optional[str] = None
        self.current_line: int = 0
        self.current_column: int = 0

    EMBEDDED_JSON_PATTERNS: ClassVar[list[str]] = []
    """Regex patterns locating embedded ``MODEL_DATA`` JSON in comments.

    Each concrete parser declares its own comment prefixes (e.g. ``//`` for
    Scala, ``--`` for Lean). Empty by default: parsers without embedded data
    inherit a no-op extractor.
    """

    EMBEDDED_LENIENT_MODEL_NAME: ClassVar[str] = "Unknown Model"
    """Default model name applied by the lenient embedded-data applier."""

    @abstractmethod
    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a GNN file."""

    @abstractmethod
    def parse_string(self, content: str) -> ParseResult:
        """Parse GNN content from string."""

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""

    def create_parse_error(
        self, message: str, line: Optional[int] = None, column: Optional[int] = None
    ) -> ParseError:
        """Create a parse error with location information."""
        return ParseError(
            message=message,
            line=line or self.current_line,
            column=column or self.current_column,
            source=self.current_file,
        )

    def create_empty_model(
        self, name: str = "Unnamed Model"
    ) -> GNNInternalRepresentation:
        """Create an empty GNN model."""
        return GNNInternalRepresentation(
            model_name=name, created_at=datetime.now(), modified_at=datetime.now()
        )

    def _extract_embedded_json_data(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract embedded JSON model data using this parser's comment patterns.

        Concrete parsers declare their ``MODEL_DATA`` comment patterns in the
        :attr:`EMBEDDED_JSON_PATTERNS` class attribute; the first pattern whose
        captured JSON parses wins.
        """
        return extract_embedded_json_data(content, self.EMBEDDED_JSON_PATTERNS)

    def _parse_from_embedded_data(
        self, embedded_data: Dict[str, Any], result: ParseResult
    ) -> ParseResult:
        """Parse a model from embedded JSON data (shared strict variant).

        Replaces the per-parser copies of this routine (scala, lean, coq,
        isabelle, python, haskell, TLA, agda), which were verbatim identical.
        """
        try:
            # Create model from embedded data
            model = GNNInternalRepresentation(
                model_name=embedded_data.get("model_name", "Unknown Model"),
                annotation=embedded_data.get("annotation", ""),
            )

            # Parse variables
            for var_data in embedded_data.get("variables", []):
                var = Variable(
                    name=var_data["name"],
                    var_type=VariableType(var_data["var_type"]),
                    data_type=DataType(var_data["data_type"]),
                    dimensions=var_data.get("dimensions", []),
                )
                model.variables.append(var)

            # Parse connections
            for conn_data in embedded_data.get("connections", []):
                conn = Connection(
                    annotation=conn_data.get("annotation"),
                    source_variables=conn_data["source_variables"],
                    target_variables=conn_data["target_variables"],
                    connection_type=ConnectionType(conn_data["connection_type"]),
                )
                model.connections.append(conn)

            # Parse parameters
            for param_data in embedded_data.get("parameters", []):
                param = Parameter(
                    name=param_data["name"],
                    value=param_data["value"],
                    type_hint=param_data.get("param_type", "constant"),
                )
                model.parameters.append(param)

            # Set time specification if present
            if embedded_data.get("time_specification"):
                time_data = embedded_data["time_specification"]
                model.time_specification = TimeSpecification(
                    time_type=time_data.get("time_type", "dynamic"),
                    discretization=time_data.get("discretization"),
                    horizon=time_data.get("horizon"),
                    step_size=time_data.get("step_size"),
                )

            # Set ontology mappings if present
            if embedded_data.get("ontology_mappings"):
                for mapping_data in embedded_data["ontology_mappings"]:
                    mapping = OntologyMapping(
                        variable_name=mapping_data["variable_name"],
                        ontology_term=mapping_data["ontology_term"],
                        description=mapping_data.get("description"),
                    )
                    model.ontology_mappings.append(mapping)

            result.model = model
            result.success = True
            return result

        except Exception as e:
            result.add_error(f"Failed to parse embedded data: {e}")
            return result

    def _parse_embedded_data_lenient(
        self, embedded_data: Dict[str, Any], result: ParseResult
    ) -> ParseResult:
        """Apply embedded JSON data onto ``result.model`` in place.

        Lenient counterpart of :meth:`_parse_from_embedded_data`: enum fields
        use ``.get`` defaults instead of strict casts and the existing model
        object is mutated rather than replaced. Used by parsers whose
        round-trip fidelity relies on re-reading embedded ``MODEL_DATA``.
        """
        try:
            result.model.model_name = embedded_data.get(
                "model_name", self.EMBEDDED_LENIENT_MODEL_NAME
            )
            result.model.annotation = embedded_data.get("annotation", "")

            # Restore variables
            for var_data in embedded_data.get("variables", []):
                var = Variable(
                    name=var_data["name"],
                    var_type=VariableType(var_data.get("var_type", "hidden_state")),
                    data_type=DataType(var_data.get("data_type", "categorical")),
                    dimensions=var_data.get("dimensions", []),
                )
                result.model.variables.append(var)

            # Restore connections
            for conn_data in embedded_data.get("connections", []):
                conn = Connection(
                    annotation=conn_data.get("annotation"),
                    source_variables=conn_data.get("source_variables", []),
                    target_variables=conn_data.get("target_variables", []),
                    connection_type=ConnectionType(
                        conn_data.get("connection_type", "directed")
                    ),
                )
                result.model.connections.append(conn)

            # Restore parameters
            for param_data in embedded_data.get("parameters", []):
                param = Parameter(name=param_data["name"], value=param_data["value"])
                result.model.parameters.append(param)

            # Restore time specification
            if embedded_data.get("time_specification"):
                time_data = embedded_data["time_specification"]
                result.model.time_specification = TimeSpecification(
                    time_type=time_data.get("time_type", "dynamic"),
                    discretization=time_data.get("discretization"),
                    horizon=time_data.get("horizon"),
                    step_size=time_data.get("step_size"),
                )

            # Restore ontology mappings
            for mapping_data in embedded_data.get("ontology_mappings", []):
                mapping = OntologyMapping(
                    variable_name=mapping_data.get("variable_name", ""),
                    ontology_term=mapping_data.get("ontology_term", ""),
                    description=mapping_data.get("description"),
                )
                result.model.ontology_mappings.append(mapping)

            return result

        except Exception as e:
            result.add_error(f"Failed to parse embedded data: {e}")
            return result


def normalize_variable_name(name: str) -> str:
    """
    Normalize variable name for consistent reference.

    This function ensures that variable names are treated consistently
    throughout the codebase, regardless of case or minor variations.

    Args:
        name: Variable name to normalize

    Returns:
        Normalized variable name
    """
    # Remove any whitespace first so 'pi' check works on trimmed input
    name = name.strip()

    # Handle special Unicode characters like π
    if name == "π" or name.lower() == "pi":
        return "π"  # Standardize on Unicode π

    # Keep case sensitivity for specific Active Inference standard variables
    if name in ["A", "B", "C", "D", "E", "F", "G"]:
        return name

    return name


def parse_dimensions(dim_str: str) -> List[int]:
    """Parse dimension string like '[2,3,4]' or '[2,3,type=float]' into list of integers."""
    try:
        # Remove brackets and split by comma
        dim_str = dim_str.strip("[]")
        if not dim_str:
            return []

        dimensions: list[Any] = []
        for dim in dim_str.split(","):
            dim = dim.strip()
            # Stop parsing if we encounter a type specification
            if dim.startswith("type="):
                break
            if dim.isdigit():
                dimensions.append(int(dim))
            else:
                # Handle symbolic dimensions by defaulting to 1
                dimensions.append(1)

        return dimensions
    except (ValueError, TypeError, AttributeError):
        return [1]  # Default dimension


def safe_enum_convert(
    enum_class: Type[E], value: Any, default: Optional[E] = None
) -> Optional[E]:
    """Safely convert string to enum, handling case insensitivity.

    Returns the converted enum value, the provided default, or None if no
    default was given and conversion failed.
    """
    if isinstance(value, enum_class):
        return value

    if isinstance(value, str):
        # Try exact match first
        try:
            return enum_class(value)
        except ValueError as e:
            logger.debug("Exact enum match failed, trying lowercase: %s", e)

        # Try lowercase
        try:
            return enum_class(value.lower())
        except ValueError as e:
            logger.debug("Lowercase enum match failed, trying uppercase: %s", e)

        # Try uppercase
        try:
            return enum_class(value.upper())
        except ValueError as e:
            logger.debug(
                "All enum conversions failed, falling through to default: %s", e
            )

    return default


# Tier 1: exact single-uppercase-letter Active Inference matrix/vector conventions
_ACTINF_SINGLE_CHAR: Dict[str, VariableType] = {
    "A": VariableType.LIKELIHOOD_MATRIX,
    "B": VariableType.TRANSITION_MATRIX,
    "C": VariableType.PREFERENCE_VECTOR,
    "D": VariableType.PRIOR_VECTOR,
    "E": VariableType.POLICY,  # Habit / initial policy prior
    "F": VariableType.HIDDEN_STATE,  # Variational free energy → state space
    "G": VariableType.POLICY,  # Expected free energy drives policy
}


def infer_variable_type(name: str) -> VariableType:
    """
    Infer variable type from its name according to Active Inference conventions.

    Three matching tiers in priority order:
      1. Exact single-char (uppercase A-G, Unicode π)
      2. Lowercase first-char prefix (s→state, o→obs, u/a→action)
      3. Substring in full name (descriptive longer names)

    Args:
        name: Variable name

    Returns:
        Inferred variable type
    """
    name_lower = name.lower()

    # Tier 1: exact single-char matches
    if name == "π" or name_lower == "pi":
        return VariableType.POLICY
    if name in _ACTINF_SINGLE_CHAR:
        return _ACTINF_SINGLE_CHAR[name]

    # Tier 2: lowercase first-char prefix conventions
    if name_lower.startswith("s"):
        return VariableType.HIDDEN_STATE
    if name_lower.startswith("o"):
        return VariableType.OBSERVATION
    if name_lower.startswith("u") or name_lower.startswith("a"):
        return VariableType.ACTION

    # Tier 3: substring match for longer descriptive names
    if "state" in name_lower or "hidden" in name_lower:
        return VariableType.HIDDEN_STATE
    if "obs" in name_lower or "perception" in name_lower:
        return VariableType.OBSERVATION
    if "action" in name_lower or "control" in name_lower:
        return VariableType.ACTION
    if "policy" in name_lower or "pi" in name_lower or "habit" in name_lower:
        return VariableType.POLICY
    if "prior" in name_lower:
        return VariableType.PRIOR_VECTOR
    if "likelihood" in name_lower:
        return VariableType.LIKELIHOOD_MATRIX
    if "transition" in name_lower:
        return VariableType.TRANSITION_MATRIX
    if "preference" in name_lower or "utility" in name_lower:
        return VariableType.PREFERENCE_VECTOR

    return VariableType.HIDDEN_STATE


def parse_connection_operator(op: str) -> ConnectionType:
    """Parse connection operator string to ConnectionType."""
    op = op.strip()

    if op in [">", "->"]:
        return ConnectionType.DIRECTED
    elif op == "-":
        return ConnectionType.UNDIRECTED
    elif op == "|":
        return ConnectionType.CONDITIONAL
    elif op == "<->":
        return ConnectionType.BIDIRECTIONAL
    else:
        return ConnectionType.DIRECTED  # Default


def extract_embedded_json_data(
    content: str, patterns: List[str]
) -> Optional[Dict[str, Any]]:
    """Extract embedded JSON model data from source file content.

    Tries each pattern in order, returning the first successfully parsed JSON object.

    Args:
        content: Source file content to search.
        patterns: List of regex patterns with one capture group for the JSON body.

    Returns:
        Parsed dict if found, None otherwise.
    """
    import json
    import re

    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
        if match:
            try:
                return cast("dict[str, Any] | None", json.loads(match.group(1)))
            except json.JSONDecodeError as e:
                logger.debug("Pattern did not yield valid JSON, trying next: %s", e)
                continue
    return None


# Export all public classes and functions
__all__: list[Any] = [
    "ParseError",
    "ValidationError",
    "ValidationWarning",
    "ConversionError",
    "GNNFormat",
    "VariableType",
    "DataType",
    "ConnectionType",
    "ASTNode",
    "Variable",
    "Connection",
    "Parameter",
    "Equation",
    "TimeSpecification",
    "OntologyMapping",
    "Section",
    "GNNInternalRepresentation",
    "ParseResult",
    "ASTVisitor",
    "PrintVisitor",
    "BaseGNNParser",  # GNNParser and GNNSerializer Protocols are internal; use BaseGNNParser ABC
    "normalize_variable_name",
    "parse_dimensions",
    "infer_variable_type",
    "parse_connection_operator",
    "safe_enum_convert",
    "extract_embedded_json_data",
]
