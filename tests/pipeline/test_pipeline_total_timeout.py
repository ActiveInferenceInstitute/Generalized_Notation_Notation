#!/usr/bin/env python3
"""``pipeline.timeout.total`` in input/config.yaml is enforced, not decorative.

A full run took 7,742 s against a configured 7,200 s total with nothing
stopping it: no code read the key. It now arms a deadline that caps every
step timeout and records steps starting after it as SKIPPED.
"""

from __future__ import annotations

import logging
import time

import pytest

from gnn.main import _arm_pipeline_deadline, _deadline_skip_result
from gnn.pipeline.step_timeouts import (
    PIPELINE_DEADLINE_ENV,
    get_step_timeout,
    pipeline_budget_remaining,
)

LOGGER = logging.getLogger("t")


@pytest.fixture(autouse=True)
def _no_inherited_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PIPELINE_DEADLINE_ENV, raising=False)
    monkeypatch.delenv("GNN_STEP_TIMEOUT_12", raising=False)
    monkeypatch.delenv("GNN_STEP_TIMEOUT_SCALE", raising=False)


def test_without_deadline_step_timeouts_are_unchanged() -> None:
    assert pipeline_budget_remaining() is None
    assert get_step_timeout("12_execute.py") == 7200


def test_step_timeout_is_capped_at_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PIPELINE_DEADLINE_ENV, str(time.time() + 120))
    assert 100 <= get_step_timeout("12_execute.py") <= 120
    monkeypatch.setenv(PIPELINE_DEADLINE_ENV, str(time.time() - 5))
    assert get_step_timeout("12_execute.py") == 1


def test_config_total_arms_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    _arm_pipeline_deadline({"timeout": {"total": 600}}, LOGGER)
    remaining = pipeline_budget_remaining()
    assert remaining is not None and 590 < remaining <= 600


@pytest.mark.parametrize(
    "settings", [{}, {"timeout": {"total": 0}}, {"timeout": {"total": True}}]
)
def test_missing_or_invalid_total_leaves_no_deadline(settings: dict) -> None:
    _arm_pipeline_deadline(settings, LOGGER)
    assert pipeline_budget_remaining() is None


def test_enclosing_deadline_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    outer = str(time.time() + 30)
    monkeypatch.setenv(PIPELINE_DEADLINE_ENV, outer)
    _arm_pipeline_deadline({"timeout": {"total": 9999}}, LOGGER)
    assert pipeline_budget_remaining() <= 30


def test_steps_after_the_deadline_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _deadline_skip_result() is None
    monkeypatch.setenv(PIPELINE_DEADLINE_ENV, str(time.time() + 60))
    assert _deadline_skip_result() is None
    monkeypatch.setenv(PIPELINE_DEADLINE_ENV, str(time.time() - 1))
    skipped = _deadline_skip_result()
    assert skipped is not None
    assert skipped["status"] == "SKIPPED"
    assert "pipeline.timeout.total" in skipped["skip_reason"]
