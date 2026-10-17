"""Byte coverage controls independent of model output quality."""

from __future__ import annotations

import pytest

from gnn.llm.source_parts import partition_source


@pytest.mark.parametrize("limit", [4, 5, 7, 31, 1024, None])
def test_parts_preserve_every_original_byte(limit: int | None) -> None:
    source = "## ModelName\r\nπ 🧠 é 中文\r\n" * 100
    parts = partition_source(source, limit)
    assert b"".join(part.text.encode() for part in parts) == source.encode()
    offset = 0
    for part in parts:
        assert part.start_byte == offset
        assert part.end_byte - offset == len(part.text.encode())
        if limit is not None:
            assert part.end_byte - offset <= limit
        offset = part.end_byte
    assert offset == len(source.encode())


@pytest.mark.parametrize("limit", [True, 0, 3, -1, 4.5, "100"])
def test_invalid_byte_budgets_refuse(limit: object) -> None:
    with pytest.raises(ValueError, match="source_chunk_bytes"):
        partition_source("model", limit)  # type: ignore[arg-type]


def test_empty_source_keeps_one_explicit_empty_part() -> None:
    parts = partition_source("", 4)
    assert len(parts) == 1
    assert parts[0].start_byte == parts[0].end_byte == 0
