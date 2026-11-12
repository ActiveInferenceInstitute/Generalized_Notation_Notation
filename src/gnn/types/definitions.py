"""
Shared types for GNN module to avoid circular imports.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:  # runtime re-export lives in gnn/types/__init__.py
    from gnn.parsers.common import GNNFormat, GNNInternalRepresentation


class ValidationLevel(Enum):
    """Provide ValidationLevel behavior."""

    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    RESEARCH = "research"
    ROUND_TRIP = "round_trip"


class GNNSyntaxError(Exception):
    """Provide GNNSyntaxError behavior."""

    def __init__(
        self,
        message: str,
        line: Optional[int] = None,
        column: Optional[int] = None,
        format_context: Optional[str] = None,
    ) -> None:
        """Initialize the instance."""
        super().__init__(message)
        self.line = line
        self.column = column
        self.format_context = format_context

    def format_message(self) -> str:
        """Provide format message behavior."""
        msg = str(self)
        if self.line is not None:
            msg += f" (line {self.line}"
            if self.column is not None:
                msg += f", column {self.column}"
            msg += ")"
        if self.format_context:
            msg += f" in {self.format_context} format"
        return msg


@dataclass
class ValidationResult:
    """Provide ValidationResult behavior."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    format_tested: Optional[str] = None
    round_trip_results: List["RoundTripResult"] = field(default_factory=list)
    cross_format_consistent: Optional[bool] = None
    semantic_checksum: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    def add_round_trip_result(self, result: "RoundTripResult") -> None:
        """Provide add round trip result behavior."""
        self.round_trip_results.append(result)
        if not result.success:
            self.errors.extend(result.errors)
            self.warnings.extend(result.warnings)

    def get_round_trip_success_rate(self) -> float:
        """Return round trip success rate."""
        if not self.round_trip_results:
            return 0.0
        successful = sum(1 for r in self.round_trip_results if r.success)
        return (successful / len(self.round_trip_results)) * 100


@dataclass
class GNNVariable:
    """Provide GNNVariable behavior."""

    name: str
    dimensions: List[Union[int, str]]
    data_type: str
    description: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    line_number: Optional[int] = None
    ontology_mapping: Optional[str] = None
    format_specific_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GNNConnection:
    """Provide GNNConnection behavior."""

    source: Union[str, List[str]]
    target: Union[str, List[str]]
    connection_type: str
    symbol: str
    description: Optional[str] = None
    line_number: Optional[int] = None
    weight: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedGNN:
    """Provide ParsedGNN behavior."""

    gnn_section: str
    version: str
    model_name: str
    model_annotation: str
    variables: Dict[str, GNNVariable]
    connections: List[GNNConnection]
    parameters: Dict[str, Any]
    equations: List[Dict[str, str]]
    time_config: Dict[str, Any]
    ontology_mappings: Dict[str, str]
    model_parameters: Dict[str, Any]
    footer: str
    signature: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_format: Optional[str] = None
    semantic_checksum: Optional[str] = None
    round_trip_verified: bool = False


# ``ParseResult`` has one declaration, in ``gnn.parsers.common``: the parser
# fleet constructs it with a model, and this module re-exports the same class
# object so typed consumers and parser consumers cannot diverge. Importing the
# implementation module (never the ``gnn.types`` facade) keeps the runtime
# import graph acyclic.
from gnn.parsers.common import (  # noqa: E402  single-declaration re-export
    ParseResult as ParseResult,
)


@dataclass
class RoundTripResult:
    """Provide RoundTripResult behavior."""

    source_format: "GNNFormat"
    target_format: "GNNFormat"
    success: bool
    original_model: "Optional[GNNInternalRepresentation]" = None
    converted_content: Optional[str] = None
    parsed_back_model: "Optional[GNNInternalRepresentation]" = None
    differences: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    test_time: float = 0.0
    checksum_original: Optional[str] = None
    checksum_converted: Optional[str] = None

    def add_difference(self, diff: str) -> None:
        """Provide add difference behavior."""
        self.differences.append(diff)
        self.success = False

    def add_warning(self, warning: str) -> Any:
        """Provide add warning behavior."""
        self.warnings.append(warning)

    def add_error(self, error: str) -> Any:
        """Provide add error behavior."""
        self.errors.append(error)
        self.success = False


@dataclass
class ComprehensiveTestReport:
    """Complete test report for all round-trip tests."""

    reference_file: str
    test_timestamp: datetime = field(default_factory=datetime.now)
    total_tests: int = 0
    successful_tests: int = 0
    failed_tests: int = 0
    round_trip_results: List[RoundTripResult] = field(default_factory=list)
    format_matrix: "Dict[Tuple[GNNFormat, GNNFormat], bool]" = field(
        default_factory=dict
    )
    semantic_differences: List[str] = field(default_factory=list)
    critical_errors: List[str] = field(default_factory=list)

    def add_result(self, result: RoundTripResult) -> None:
        """Provide add result behavior."""
        self.round_trip_results.append(result)
        self.total_tests += 1
        if result.success:
            self.successful_tests += 1
        else:
            self.failed_tests += 1
            self.semantic_differences.extend(result.differences)
            self.critical_errors.extend(result.errors)
        self.format_matrix[(result.source_format, result.target_format)] = (
            result.success
        )

    def get_success_rate(self) -> float:
        """Return success rate."""
        return (
            (self.successful_tests / self.total_tests) * 100
            if self.total_tests > 0
            else 0.0
        )

    def get_format_summary(self) -> "Dict[GNNFormat, Dict[str, int]]":
        """Return format summary."""
        format_summary: dict[Any, Any] = {}
        for result in self.round_trip_results:
            fmt = result.target_format
            if fmt not in format_summary:
                format_summary[fmt] = {"success": 0, "failure": 0, "total": 0}
            format_summary[fmt]["total"] += 1
            if result.success:
                format_summary[fmt]["success"] += 1
            else:
                format_summary[fmt]["failure"] += 1
        return format_summary
