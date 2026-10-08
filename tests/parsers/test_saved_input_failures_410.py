"""Saved-file parser failures remain explicit through the public registry.

These checks exercise real filesystem and UTF-8 failures. They do not establish
formal-language validity, native compiler execution, or model equivalence.
"""

from pathlib import Path

import pytest

from gnn.parsers import GNNFormat, GNNParsingSystem
from gnn.parsers.common import ParseError

TEXT_FORMATS = [format for format in GNNFormat if format is not GNNFormat.PICKLE]


@pytest.mark.parametrize("format", TEXT_FORMATS, ids=lambda format: format.value)
@pytest.mark.parametrize("failure", ["invalid_utf8", "directory"])
def test_saved_text_input_failure_does_not_return_a_successful_model(
    tmp_path: Path, format: GNNFormat, failure: str
) -> None:
    source = tmp_path / "authored.input"
    if failure == "directory":
        source.mkdir()
        sentinel = source / "caller-owned.txt"
        sentinel.write_bytes(b"preserve this directory and its contents\n")
    else:
        source.write_bytes(b"\xff\xfe\x80\n")
    before = (
        {entry.name: entry.read_bytes() for entry in source.iterdir()}
        if source.is_dir()
        else source.read_bytes()
    )

    system = GNNParsingSystem()
    try:
        result = system.parse_file(source, format_hint=format)
    except (ParseError, UnicodeError, OSError, ValueError) as error:
        assert str(error), "A refused saved input must retain a useful diagnosis"
    else:
        assert not result.success, (
            f"{format.value} reported success after the actual {failure} failure"
        )
        assert result.errors, "A failed model must retain its input diagnosis"

    after = (
        {entry.name: entry.read_bytes() for entry in source.iterdir()}
        if source.is_dir()
        else source.read_bytes()
    )
    assert after == before
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [source.name]
