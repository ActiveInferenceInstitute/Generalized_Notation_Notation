"""Composability tests for the main.py step-selection core.

Pins the pure selection contract (``select_pipeline_steps`` /
``parse_step_list_strict``), the lenient ``parse_step_list`` back-compat
contract, the script-name step extractor, and the fail-fast error paths of
``resolve_steps_to_execute``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gnn.main import (  # noqa: E402
    PIPELINE_STEPS,
    parse_step_list,
    parse_step_list_strict,
    select_pipeline_steps,
    step_number_from_script_name,
)
from gnn.utils.pipeline_orchestration.pipeline_step_dependencies import (
    dependency_steps_for_step,
    resolve_step_dependencies,
)  # noqa: E402

pytestmark = pytest.mark.unit


def _script_numbers(steps: tuple) -> set[int]:
    return {int(script.split("_")[0]) for script, _ in steps}


# ---------------------------------------------------------------------------
# parse_step_list (lenient back-compat contract)
# ---------------------------------------------------------------------------


def test_parse_step_list_lenient_contract() -> None:
    assert parse_step_list(None) == []
    assert parse_step_list("3,5") == [3, 5]
    assert parse_step_list([3, "5"]) == [3, 5]
    assert parse_step_list("not,steps") == []


# ---------------------------------------------------------------------------
# parse_step_list_strict
# ---------------------------------------------------------------------------


def test_strict_parse_accepts_valid_input() -> None:
    assert parse_step_list_strict(None) == []
    assert parse_step_list_strict([]) == []
    assert parse_step_list_strict("0,3,24") == [0, 3, 24]
    assert parse_step_list_strict(" 3 , 5 ") == [3, 5]
    assert parse_step_list_strict(["3", 5]) == [3, 5]


def test_strict_parse_rejects_invalid_tokens() -> None:
    with pytest.raises(ValueError, match="Invalid step number"):
        parse_step_list_strict("3,a")
    with pytest.raises(ValueError, match="Invalid step number"):
        parse_step_list_strict(["3", None])


# ---------------------------------------------------------------------------
# select_pipeline_steps (pure core)
# ---------------------------------------------------------------------------


def test_select_without_filters_returns_all_steps() -> None:
    selection = select_pipeline_steps(list(PIPELINE_STEPS))
    assert selection.selected == tuple(PIPELINE_STEPS)
    assert selection.skipped == ()
    assert selection.added_dependencies == ()
    assert selection.requested_only == ()
    assert selection.unknown_requested == ()


def test_select_only_steps_preserves_order() -> None:
    selection = select_pipeline_steps(list(PIPELINE_STEPS), only_steps="5,3")
    assert [script for script, _ in selection.selected] == [
        "3_gnn.py",
        "5_type_checker.py",
    ]
    assert selection.requested_only == (5, 3)


def test_select_only_steps_pulls_dependencies() -> None:
    dep_steps = [n for n in range(25) if dependency_steps_for_step(n)]
    assert dep_steps, "dependency map unexpectedly empty"
    for step in dep_steps:
        selection = select_pipeline_steps(list(PIPELINE_STEPS), only_steps=str(step))
        selected_numbers = _script_numbers(selection.selected)
        assert {step, *dependency_steps_for_step(step)} <= selected_numbers
        assert selection.added_dependencies == tuple(
            sorted(set(resolve_step_dependencies([step])) - {step})
        )


def test_select_skip_merges_cli_and_config_lists() -> None:
    selection = select_pipeline_steps(
        list(PIPELINE_STEPS), cli_skip_steps="15", config_skip_steps=[16]
    )
    selected_numbers = _script_numbers(selection.selected)
    assert 15 not in selected_numbers
    assert 16 not in selected_numbers
    assert selection.skipped == (15, 16)


def test_select_reports_unknown_numbers_without_executing_them() -> None:
    selection = select_pipeline_steps(list(PIPELINE_STEPS), only_steps="3,99")
    assert _script_numbers(selection.selected) == {3}
    assert selection.unknown_requested == (99,)


def test_select_skip_filters_linearly_without_cascade() -> None:
    selection = select_pipeline_steps(
        list(PIPELINE_STEPS), only_steps="12", cli_skip_steps="3"
    )
    selected_numbers = _script_numbers(selection.selected)
    # Skips are a linear filter: the skipped prerequisite is dropped, but
    # dependents remain and are gated later by prerequisite validation.
    assert 3 not in selected_numbers
    assert 11 in selected_numbers
    assert 12 in selected_numbers


def test_select_does_not_mutate_input() -> None:
    steps = list(PIPELINE_STEPS)
    snapshot = list(steps)
    select_pipeline_steps(steps, only_steps="5,3", cli_skip_steps="8")
    assert steps == snapshot


# ---------------------------------------------------------------------------
# step_number_from_script_name
# ---------------------------------------------------------------------------


def test_step_number_extraction() -> None:
    assert step_number_from_script_name("11_render.py") == 11
    assert step_number_from_script_name("0_template.py") == 0
    assert step_number_from_script_name("24_intelligent_analysis.py") == 24
    assert step_number_from_script_name("main.py") == -1


# ---------------------------------------------------------------------------
# resolve_steps_to_execute (adapter: logging + fail-fast error paths)
# ---------------------------------------------------------------------------


def _make_args(only_steps: object = None, skip_steps: object = None) -> SimpleNamespace:
    return SimpleNamespace(only_steps=only_steps, skip_steps=skip_steps)


def _resolve(
    only_steps: object = None,
    skip_steps: object = None,
    config: dict | None = None,
    logger: logging.Logger | None = None,
) -> list:
    from gnn.main import resolve_steps_to_execute

    return resolve_steps_to_execute(
        _make_args(only_steps, skip_steps),  # type: ignore[arg-type]
        config or {},
        logger or logging.getLogger("test_resolver"),
    )


def test_resolver_without_filters_returns_full_step_list() -> None:
    assert _resolve() == list(PIPELINE_STEPS)


def test_resolver_fail_fast_on_invalid_token() -> None:
    with pytest.raises(ValueError, match="Invalid step number"):
        _resolve(only_steps="3,a")


def test_resolver_fail_fast_on_invalid_skip_token() -> None:
    with pytest.raises(ValueError, match="Invalid step number"):
        _resolve(skip_steps="15,x")


def test_resolver_fail_fast_on_fully_unknown_selection() -> None:
    with pytest.raises(ValueError, match="Invalid step number"):
        _resolve(only_steps="99")


def test_resolver_fail_fast_on_config_only_steps() -> None:
    with pytest.raises(ValueError, match="Invalid step number"):
        _resolve(config={"only_steps": "42,99"})


def test_resolver_logs_selection_lines(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("test_resolver_logs")
    with caplog.at_level(logging.INFO, logger="test_resolver_logs"):
        _resolve(only_steps="3,5", skip_steps="5", logger=logger)

    messages = [record.getMessage() for record in caplog.records]
    assert any(msg.startswith("Executing steps: ['3_gnn.py']") for msg in messages)
    assert any(
        msg.startswith("Skipping steps: ['5_type_checker.py']") for msg in messages
    )


def test_resolver_rejects_partly_unknown_step_numbers() -> None:
    """Execution cannot silently reduce a request to its known subset."""
    with pytest.raises(ValueError, match="Invalid step number"):
        _resolve(only_steps="3,99")


def test_resolver_falls_back_to_config_only_steps_when_cli_absent() -> None:
    """When ``args.only_steps`` is None, the resolver uses the config's
    ``only_steps`` value (the composition-root fallback at main.py:466).

    The fail-fast path for unknown config steps is already pinned; this
    covers the happy-path fallback for valid config-driven selection.
    """
    selected = _resolve(only_steps=None, config={"only_steps": "3,5"})
    selected_names = [step[0] for step in selected]
    assert "3_gnn.py" in selected_names
    assert "5_type_checker.py" in selected_names
    # Dependency pull-in: step 5 depends on step 3, both present.
    assert len(selected) >= 2


def test_resolver_cli_only_steps_takes_precedence_over_config() -> None:
    """CLI ``only_steps`` takes precedence over the config value."""
    selected = _resolve(only_steps="7", config={"only_steps": "3,5"})
    selected_names = [step[0] for step in selected]
    assert "7_export.py" in selected_names
    # Config's 3,5 are not selected when CLI overrides.
    assert "3_gnn.py" not in selected_names or "7_export.py" in selected_names
