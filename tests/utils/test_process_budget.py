"""Absolute invocation/request budgets reserve cleanup instead of adding a tail."""

import pytest

from gnn.utils.runtime_safety.process_budget import resolve_process_deadlines


@pytest.mark.parametrize(
    ("timeout", "absolute", "work", "allowance", "ceiling"),
    [
        (45, None, 145, 1, None),
        (None, None, None, 1, None),
        (30, 120, 119, 1, 120),
        (1, 120, 101, 1, 120),
        (30, 100.8, 100.6, 0.2, 100.8),
        (30, 99, 99, 0, 99),
    ],
)
def test_work_and_cleanup_share_absolute_budget(
    monkeypatch: pytest.MonkeyPatch, timeout, absolute, work, allowance, ceiling
) -> None:
    monkeypatch.setattr(
        "gnn.utils.runtime_safety.process_budget.time.monotonic", lambda: 100
    )
    actual_work, actual_allowance, actual_ceiling = resolve_process_deadlines(
        timeout, deadline_monotonic=absolute
    )
    assert (
        actual_work == pytest.approx(work) if work is not None else actual_work is None
    )
    assert actual_allowance == pytest.approx(allowance)
    assert actual_ceiling == ceiling


@pytest.mark.parametrize("value", [True, -1, "1", float("inf"), float("nan")])
def test_invalid_local_timeout_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="timeout"):
        resolve_process_deadlines(value)


@pytest.mark.parametrize("value", [True, "1", float("inf"), float("nan")])
def test_invalid_absolute_deadline_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="deadline"):
        resolve_process_deadlines(1, deadline_monotonic=value)


@pytest.mark.parametrize("value", [True, -1, "1", float("inf"), float("nan")])
def test_invalid_cleanup_allowance_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="allowance"):
        resolve_process_deadlines(1, cleanup_seconds=value)
