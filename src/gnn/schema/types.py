"""Structured declarations and diagnostics for the lightweight schema parser."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GNNParseError:
    """Structured parse error with code, line number, and human-readable message."""

    code: str  # e.g. GNN-E001
    message: str
    line: Optional[int] = None
    file: Optional[str] = None
    severity: str = "error"  # "error" or "warning"

    def __str__(self) -> str:
        """Return the string representation."""
        loc = f":{self.line}" if self.line else ""
        src = f" [{self.file}{loc}]" if self.file else ""
        return f"[{self.code}] {self.message}{src}"


@dataclass
class GNNConnectionEdge:
    """A parsed connection/edge between two state-space variables."""

    source: str
    target: str
    directed: bool  # True for '>', False for '-'
    label: Optional[str] = None
    line: Optional[int] = None


@dataclass
class GNNVariable:
    """A parsed state-space variable declaration."""

    name: str
    dimensions: List[str]  # can be ints or symbolic names
    dtype: str = "float"
    default: Optional[str] = None
    line: Optional[int] = None
