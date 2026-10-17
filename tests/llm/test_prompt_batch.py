"""Named answer bundles preserve explicit task coverage and reject ambiguity."""

from __future__ import annotations

import pytest

from gnn.llm.prompt_batch import parse_answers, response_schema


@pytest.mark.parametrize(
    "content",
    [
        "{}",
        '{"one":"answer"}',
        '{"one":"a","two":"b","extra":"c"}',
        '{"one":"a","two":null}',
        '{"one":"a","two":" "}',
        '{"one":"a","one":"b","two":"c"}',
        "not JSON",
    ],
)
def test_missing_extra_empty_or_ambiguous_answers_refuse(content: str) -> None:
    with pytest.raises(ValueError):
        parse_answers(content, ["one", "two"])


def test_exact_named_answers_and_schema() -> None:
    labels = ["one", "two"]
    assert parse_answers('{"two":"second","one":"first"}', labels) == {
        "one": "first",
        "two": "second",
    }
    schema = response_schema(labels)
    assert schema["required"] == labels
    assert set(schema["properties"]) == set(labels)
    assert schema["additionalProperties"] is False
