"""Public editor parsing remains faithful and bounded on hostile Markdown."""

from __future__ import annotations

import random
import re

import pytest

from gnn.gui.gui_1.markdown import (
    MAX_MARKDOWN_CHARS,
    parse_state_space_from_markdown,
    remove_state_space_entry,
    update_state_space_entry,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_state_line_grammar_matches_bounded_legacy_specification() -> None:
    legacy = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\[(.*?)\]\s*(#\s*(.*))?$")
    rng = random.Random(741)
    cases = [
        "A[2, 3, type=float] # matrix",
        "A[2] junk ]  # final delimiter",
        "A[2] # ] remains a comment",
        "A[2] trailing",
        "A[2]   ",
        "_A[2]",
        "é[2]",
        "A[]",
    ]
    for _ in range(1200):
        cases.append("A[" + "".join(rng.choices("234 ,]#xt=_\t", k=36)))
    for line in cases:
        match = legacy.match(line)
        expected = []
        if match:
            dims = []
            typ = ""
            for part in match.group(2).split(","):
                part = part.strip()
                if part.startswith("type="):
                    typ = part.split("=", 1)[1]
                elif part:
                    try:
                        dims.append(int(part))
                    except ValueError:
                        pass
            entry = {"name": match.group(1), "dims": dims, "type": typ}
            comment = (match.group(4) or "").strip()
            if comment:
                entry["comment"] = comment
            expected = [entry]
        assert parse_state_space_from_markdown("## State Space\n" + line) == expected


@pytest.mark.parametrize("fragment", [" ", "] ", "]x", "[", "\t\t"])
def test_hostile_delimiters_preserve_editor_operations(fragment: str) -> None:
    hostile = "A[" + fragment * 65_536 + "!"
    doc = "## State Space\n" + hostile + "\nB[2,type=int] # retained\n"
    assert parse_state_space_from_markdown(doc) == [
        {"name": "B", "dims": [2], "type": "int", "comment": "retained"}
    ]
    updated = update_state_space_entry(doc, "B", "C", [3], "float")
    assert hostile in updated
    assert parse_state_space_from_markdown(updated) == [
        {"name": "C", "dims": [3], "type": "float", "comment": "retained"}
    ]
    removed = remove_state_space_entry(doc, "B")
    assert hostile in removed and parse_state_space_from_markdown(removed) == []


@pytest.mark.parametrize("operation", ["parse", "update", "remove"])
def test_editor_refuses_oversized_document_before_splitting(operation: str) -> None:
    class UnsplitDocument(str):
        def splitlines(self, *args: object, **kwargs: object) -> list[str]:
            raise AssertionError("oversized input must be refused before splitting")

    document = UnsplitDocument("x" * (MAX_MARKDOWN_CHARS + 1))
    with pytest.raises(ValueError, match="editor limit"):
        if operation == "parse":
            parse_state_space_from_markdown(document)
        elif operation == "update":
            update_state_space_entry(document, "A", "B", [2])
        else:
            remove_state_space_entry(document, "A")
