"""
GNN Markdown Parser — Parameter and Matrix Parsing Mixin

Parameter assignment, matrix value, and value parsing helpers extracted from markdown_parser.py.
Provides parsing for the standard GNN Markdown format; the reference parser implementation
that other formats can use as a model.
"""

import ast
import json
import logging
from typing import Any, Optional

from .common import (
    Parameter,
)

logger = logging.getLogger(__name__)


class ParameterParsingMixin:
    """Mixin providing parameter and matrix parsing helpers.

    Public surface: _parse_parameter_assignment, _parse_parameter_value,
    _parse_matrix_rows, _parse_matrix_row, _parse_single_value.

    All methods are self-contained within this mixin — no host-class methods are
    called. Host classes (e.g. MarkdownGNNParser) inherit this mixin solely for
    namespace consolidation; no duck-typing contract is required from the host.
    """

    @staticmethod
    def _split_inline_comment(line: str) -> tuple[str, Optional[str]]:
        """Split a line at its first ``#`` comment delimiter outside any
        bracket or quote region.

        The rest of the parser treats ``#`` (not ``###``) as the inline-comment
        delimiter, so parameter values must agree. Scanning brackets and quotes
        keeps ``#`` inside a ``{...}`` matrix row, a ``[...]`` list, a
        ``(...)`` tuple, or a quoted string intact while still stripping a
        trailing ``# comment`` after balanced constructs.

        Returns:
            (code_without_comment, comment_or_None)
        """
        depth = 0
        quote: Optional[str] = None
        i = 0
        n = len(line)
        while i < n:
            ch = line[i]
            if quote is not None:
                if ch == quote:
                    quote = None
                elif ch == "\\":
                    i += 2
                    continue
                i += 1
                continue
            if ch in ("'", '"'):
                quote = ch
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            elif ch == "#" and depth == 0:
                return line[:i].strip(), line[i + 1 :].strip()
            i += 1
        return line.strip(), None

    def _parse_parameter_assignment(self, line: str) -> Optional[Parameter]:
        """Parse a single parameter assignment line."""
        try:
            # Extract a leading inline comment (''#'' is the parser-wide
            # comment delimiter; ''###'' was a nonstandard former marker).
            line, comment = self._split_inline_comment(line)

            # Split by first '='
            if "=" not in line:
                return None

            name, value_str = line.split("=", 1)
            name = name.strip()
            value_str = value_str.strip()

            # Parse value
            value = self._parse_parameter_value(value_str)

            return Parameter(name=name, value=value, description=comment)

        except Exception as e:
            logger.warning(f"Failed to parse parameter assignment '{line}': {e}")
            return None

    def _parse_parameter_value(self, value_str: str) -> Any:
        """Parse a parameter value string."""
        value_str = value_str.strip()

        try:
            # Handle GNN matrix format: A={ (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0) }
            if value_str.startswith("{") and value_str.endswith("}"):
                # Remove comments from the string
                lines = value_str.split("\n")
                cleaned_lines: list[Any] = []
                for line in lines:
                    # Remove comments (everything after #)
                    if "#" in line:
                        line = line.split("#")[0]
                    line = line.strip()
                    if line:
                        cleaned_lines.append(line)

                # Reconstruct the value string without comments
                cleaned_value = " ".join(cleaned_lines)

                # Handle matrix format: { (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0) }
                if cleaned_value.startswith("{") and cleaned_value.endswith("}"):
                    inner = cleaned_value[1:-1].strip()

                    # Enhanced parsing logic for nested tuples
                    rows = self._parse_matrix_rows(inner)

                    # Parse each row
                    matrix: list[Any] = []
                    for row in rows:
                        row = row.strip()
                        parsed_row = self._parse_matrix_row(row)
                        if parsed_row is not None:
                            matrix.append(parsed_row)

                    return matrix

            # Try to evaluate as Python literal (bounded against DoS).
            from gnn.utils.runtime_safety.safe_eval import safe_literal_eval

            return safe_literal_eval(value_str)

        except (ValueError, SyntaxError):
            # Handle other formats

            if value_str.startswith("[") and value_str.endswith("]"):
                try:
                    # The guard-only spelling changes no lengths or brackets.
                    # Reuse the existing length/depth limits before JSON parsing;
                    # decode the untouched source to retain quoted text exactly.
                    guarded = safe_literal_eval(
                        value_str.replace("true", "True").replace("false", "False")
                    )
                    pending = [guarded]
                    while pending:
                        item = pending.pop()
                        if isinstance(item, dict):
                            raise ValueError(
                                "JSON array mappings retain existing handling"
                            )
                        if isinstance(item, list):
                            pending.extend(item)
                    return json.loads(value_str)
                except (ValueError, SyntaxError) as exc:
                    logger.debug("JSON array retains existing value handling: %s", exc)

            # Matrix format: {(1,2,3);(4,5,6)}
            if (
                value_str.startswith("{")
                and value_str.endswith("}")
                and ";" in value_str
            ):
                inner = value_str[1:-1]
                rows = inner.split(";")
                matrix = []
                for row in rows:
                    row = row.strip("()")
                    if row:
                        row_values = [
                            float(x.strip()) for x in row.split(",") if x.strip()
                        ]
                        matrix.append(row_values)
                return matrix

            # Tuple format: (1,2,3)
            elif value_str.startswith("(") and value_str.endswith(")"):
                inner = value_str[1:-1]
                values = [
                    self._parse_single_value(x.strip())
                    for x in inner.split(",")
                    if x.strip()
                ]
                return tuple(values)

            # List format: [1,2,3]
            elif value_str.startswith("[") and value_str.endswith("]"):
                inner = value_str[1:-1]
                values = [
                    self._parse_single_value(x.strip())
                    for x in inner.split(",")
                    if x.strip()
                ]
                return values

            # Single value
            else:
                return self._parse_single_value(value_str)

    def _parse_matrix_rows(self, inner: str) -> list:
        """Parse matrix rows, handling nested structures properly."""
        rows: list[Any] = []
        current_row = ""
        paren_count = 0
        i = 0

        while i < len(inner):
            char = inner[i]

            if char == "(":
                paren_count += 1
            elif char == ")":
                paren_count -= 1

            if char == "," and paren_count == 0:
                # End of a row
                if current_row.strip():
                    rows.append(current_row.strip())
                current_row = ""
            else:
                current_row += char

            i += 1

        # Add the last row
        if current_row.strip():
            rows.append(current_row.strip())

        return rows

    def _parse_matrix_row(self, row: str) -> Any:
        """Parse a single matrix row, handling both simple and nested tuple formats."""
        row = row.strip()

        # Simple tuple: (1.0, 0.0, 0.0)
        if (
            row.startswith("(")
            and row.endswith(")")
            and not self._has_nested_tuples(row)
        ):
            inner_row = row[1:-1]
            row_values = [
                self._parse_single_value(x.strip())
                for x in inner_row.split(",")
                if x.strip()
            ]
            return row_values

        # Nested tuples: ( (1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0) )
        elif self._has_nested_tuples(row):
            # Remove outer parentheses if present
            if row.startswith("(") and row.endswith(")"):
                row = row[1:-1].strip()

            # Parse nested tuples
            tuples = self._extract_tuples(row)
            row_values = []
            for tuple_str in tuples:
                tuple_str = tuple_str.strip("()")
                tuple_values = [
                    self._parse_single_value(x.strip())
                    for x in tuple_str.split(",")
                    if x.strip()
                ]
                row_values.append(tuple_values)

            return row_values

        return None

    def _has_nested_tuples(self, row: str) -> bool:
        """Check if a row contains nested tuples."""
        # Look for pattern like ( (...), (...) ) or ((...), (...))
        paren_count = 0
        found_inner_tuple = False

        for char in row:
            if char == "(":
                paren_count += 1
                if paren_count >= 2:
                    found_inner_tuple = True
            elif char == ")":
                paren_count -= 1

        return found_inner_tuple

    def _extract_tuples(self, row: str) -> list:
        """Extract individual tuples from a nested tuple string."""
        tuples: list[Any] = []
        current_tuple = ""
        paren_count = 0

        for char in row:
            if char == "(":
                paren_count += 1
                current_tuple += char
            elif char == ")":
                paren_count -= 1
                current_tuple += char

                if paren_count == 0 and current_tuple.strip():
                    # Complete tuple found
                    tuples.append(current_tuple.strip())
                    current_tuple = ""
            elif paren_count > 0:
                current_tuple += char
            # Skip characters outside parentheses (commas, spaces)

        return tuples

    def _parse_single_value(self, value_str: str) -> Any:
        """Parse a single value."""
        value_str = value_str.strip()

        # Boolean
        if value_str.lower() in ["true", "false"]:
            return value_str.lower() == "true"

        # Number — fall through to string handling if not numeric
        try:
            if "." in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            logger.debug("Value is not numeric, continuing to string/other type checks")
            # Not a number; continue to string/other type checks below

        # String (remove quotes if present)
        if value_str.startswith('"') and value_str.endswith('"'):
            return value_str[1:-1]
        elif value_str.startswith("'") and value_str.endswith("'"):
            return value_str[1:-1]

        # Return as string
        return value_str
