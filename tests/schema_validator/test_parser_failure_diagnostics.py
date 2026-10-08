"""Actual parser boundaries retain source causes and optional import recovery."""

import subprocess
import sys

from gnn.schema_validator.syntax import GNNParser
from gnn.types import GNNFormat


def test_registered_parser_failure_preserves_original_cause(caplog) -> None:
    class BrokenSourceParser:
        def parse_string(self, content):
            raise ValueError("invalid authored value")

    parser = GNNParser()
    assert parser.parsing_system is not None
    parser.parsing_system.parsers[GNNFormat.JSON] = BrokenSourceParser()
    parsed = parser.parse_content("## ModelName\nRecovered\n", "sensor.json", "json")
    diagnostic = parsed.metadata["parse_degraded"]
    assert diagnostic["requested_format"] == "json"
    assert diagnostic["fallback"] == "markdown"
    assert "ParseError" in diagnostic["reason"]
    assert "ValueError: invalid authored value" in diagnostic["reason"]
    assert "sensor.json" in caplog.text
    assert "ValueError" in caplog.text


def test_missing_optional_parser_import_keeps_basic_mode_and_reports_cause() -> None:
    # Core types remain required. Refuse only the enhanced system import,
    # leaving the actual basic parser and validator graph intact.
    program = """
import builtins
import gnn.types
original_import = builtins.__import__
def missing_enhanced_dependency(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "gnn.parsers" and "GNNParsingSystem" in fromlist:
        raise ModuleNotFoundError("missing parser dependency witness")
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = missing_enhanced_dependency
from gnn.schema_validator import GNNParser, GNNValidator, ROUND_TRIP_AVAILABLE
from gnn.schema_validator import syntax, validator
assert ROUND_TRIP_AVAILABLE is syntax.ROUND_TRIP_AVAILABLE is validator.ROUND_TRIP_AVAILABLE is False
parser = GNNParser()
assert parser.parsing_system is None
assert parser.parse_content("## ModelName\\nBasic\\n").model_name == "Basic"
assert GNNValidator(enable_cross_validation=False).parser.parsing_system is None
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError: missing parser dependency witness" in result.stderr
