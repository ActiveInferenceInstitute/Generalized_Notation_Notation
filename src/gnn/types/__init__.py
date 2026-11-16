"""Shared dataclass types and lazy aliases for parser-owned types.

The shared dataclasses load independently of the parsers. ``GNNFormat``,
``GNNInternalRepresentation`` and ``ParseResult`` resolve to their original
classes in :mod:`gnn.parsers.common` when requested.
"""

from typing import TYPE_CHECKING, Any

from gnn.types.definitions import (
    ComprehensiveTestReport,
    GNNConnection,
    GNNSyntaxError,
    GNNVariable,
    ParsedGNN,
    RoundTripResult,
    ValidationLevel,
    ValidationResult,
)

if TYPE_CHECKING:
    from gnn.parsers.common import (
        GNNFormat as GNNFormat,
    )
    from gnn.parsers.common import (
        GNNInternalRepresentation as GNNInternalRepresentation,
    )
    from gnn.parsers.common import (
        ParseResult as ParseResult,
    )

_PARSER_EXPORTS = frozenset({"GNNFormat", "GNNInternalRepresentation", "ParseResult"})


def __getattr__(name: str) -> Any:
    if name not in _PARSER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from gnn.parsers import common

    value = getattr(common, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _PARSER_EXPORTS)


__all__ = [
    "ComprehensiveTestReport",
    "GNNConnection",
    "GNNFormat",
    "GNNInternalRepresentation",
    "GNNSyntaxError",
    "GNNVariable",
    "ParseResult",
    "ParsedGNN",
    "RoundTripResult",
    "ValidationLevel",
    "ValidationResult",
]
