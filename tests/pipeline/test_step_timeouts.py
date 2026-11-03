"""Real-behavior tests for ``src/pipeline/step_timeouts.py``.

Timeout configuration decides whether a legitimate long-running step (e.g.
Step 12 executing every model across all frameworks) is killed mid-run, so
the resolution order — env override, explicit per-step config, and *no*
silent default — is product behavior, not administrative metadata. A step
whose budget is not declared fails closed with ``StepTimeoutNotDeclared``:
the old silent 180-second fallback killed a legitimate 202.6-second
``20_website.py`` build during the 2026-11-02 full-corpus run. The
in-process tier is held to the same knobs (BC-13): its budget resolves
through ``get_step_timeout`` with the ``GNN_STEP_TIMEOUT_{N}`` /
``GNN_STEP_TIMEOUT_SCALE`` env overrides.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from gnn.pipeline import step_executor, step_timeouts
from gnn.pipeline.step_executor import execute_step_in_process
from gnn.pipeline.step_registry import STEPS
from gnn.pipeline.step_timeouts import (
    STEP_TIMEOUTS,
    StepTimeoutNotDeclared,
    get_step_timeout,
)
from gnn.utils.arguments.pipeline_arguments import PipelineArguments

LOGGER = logging.getLogger("test_step_timeouts")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASICS_DIR = PROJECT_ROOT / "input" / "gnn_files" / "basics"


def _pipeline_args(output_dir: Path) -> PipelineArguments:
    """Build pipeline args pointing at the small basics fixture dir."""
    return PipelineArguments(target_dir=BASICS_DIR, output_dir=output_dir)


def test_known_step_returns_configured_timeout() -> None:
    assert get_step_timeout("3_gnn.py") == 300
    assert get_step_timeout("12_execute.py") == 7200
    # 20_website.py declares a budget well above its measured full-corpus cost
    # (271.0 s); the previous 180 s implicit default killed it mid-build.
    assert get_step_timeout("20_website.py") == 900


def test_comprehensive_flag_selects_dict_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GNN_STEP_TIMEOUT_2", raising=False)
    assert get_step_timeout("2_tests.py") == 900
    assert get_step_timeout("2_tests.py", comprehensive=True) == 1200


def test_undeclared_step_fails_closed() -> None:
    with pytest.raises(StepTimeoutNotDeclared):
        get_step_timeout("99_unknown.py")


def test_missing_dict_variant_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(STEP_TIMEOUTS, "1_setup.py", {"default": 5})
    assert get_step_timeout("1_setup.py") == 5
    with pytest.raises(StepTimeoutNotDeclared):
        get_step_timeout("1_setup.py", comprehensive=True)


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GNN_STEP_TIMEOUT_3", "42")
    assert get_step_timeout("3_gnn.py") == 42


def test_env_override_cannot_invent_a_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An operator override is an absolute value for a declared step; it is
    # not a way to run a step the pipeline never sized.
    monkeypatch.setenv("GNN_STEP_TIMEOUT_99", "77")
    with pytest.raises(StepTimeoutNotDeclared):
        get_step_timeout("99_unknown.py")


def test_invalid_env_value_falls_back_to_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GNN_STEP_TIMEOUT_3", "not-a-number")
    assert get_step_timeout("3_gnn.py") == 300


def test_every_registered_step_declares_a_timeout() -> None:
    # The guard that makes "no entry" a failure rather than a default: a step
    # added to the registry without a measured budget cannot ship silently.
    missing = [
        step.script_name for step in STEPS if step.script_name not in STEP_TIMEOUTS
    ]
    assert missing == []


def test_every_declared_budget_is_positive() -> None:
    # Guard against a typo introducing a zero/negative timeout that would
    # kill a step instantly.
    for name, cfg in STEP_TIMEOUTS.items():
        values = cfg.values() if isinstance(cfg, dict) else [cfg]
        for v in values:
            assert isinstance(v, int) and v > 0, f"{name}: bad timeout {v}"


def test_scale_multiplier_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GNN_STEP_TIMEOUT_3", raising=False)
    monkeypatch.setenv("GNN_STEP_TIMEOUT_SCALE", "3")
    assert get_step_timeout("3_gnn.py") == 900  # 300 * 3
    assert get_step_timeout("4_model_registry.py") == 540  # 180 * 3
    # Scaling multiplies declared budgets; it does not manufacture one.
    with pytest.raises(StepTimeoutNotDeclared):
        get_step_timeout("99_unknown.py")


def test_scale_multiplier_invalid_or_nonpositive_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GNN_STEP_TIMEOUT_3", raising=False)
    for bad in ("not-a-number", "0", "-2"):
        monkeypatch.setenv("GNN_STEP_TIMEOUT_SCALE", bad)
        assert get_step_timeout("3_gnn.py") == 300  # unchanged


def test_per_step_env_override_beats_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GNN_STEP_TIMEOUT_3", "42")
    monkeypatch.setenv("GNN_STEP_TIMEOUT_SCALE", "3")
    assert get_step_timeout("3_gnn.py") == 42


class TestInProcessExecutorEnvOverrides:
    """BC-13: the in-process tier resolves its budget through the same knobs."""

    def test_per_step_env_override_bounds_in_process_step(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """GNN_STEP_TIMEOUT_3 replaces the 300s config budget for step 3."""
        release = threading.Event()

        def slow_step(**kwargs: Any) -> bool:
            return release.wait(timeout=60)

        monkeypatch.setattr(
            step_executor, "resolve_step_function", lambda name: slow_step
        )
        monkeypatch.setattr(step_executor, "_CANCEL_GRACE_SECONDS", 0.2)
        monkeypatch.setenv("GNN_STEP_TIMEOUT_3", "1")
        try:
            started = time.monotonic()
            receipt = execute_step_in_process(
                "3_gnn.py", _pipeline_args(tmp_path), LOGGER
            )
            elapsed = time.monotonic() - started
        finally:
            release.set()

        # 1s env budget fired (not the 300s configured default), the receipt
        # records the force kill, and the executor returned at the budget.
        assert receipt["exit_code"] == -1
        assert receipt["force_killed"] is False
        assert "TIMEOUT" in receipt["stderr"]
        assert 0.8 <= elapsed < 1 + 0.2 + 3

    def test_timeout_scale_reaches_in_process_budget(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """GNN_STEP_TIMEOUT_SCALE multiplies the in-process budget both ways."""

        def slowish_step(**kwargs: Any) -> bool:
            time.sleep(1.2)
            return True

        monkeypatch.setattr(
            step_executor, "resolve_step_function", lambda name: slowish_step
        )
        monkeypatch.setattr(step_timeouts, "STEP_TIMEOUTS", {"3_gnn.py": 2})
        monkeypatch.delenv("GNN_STEP_TIMEOUT_3", raising=False)

        # Unscaled: the 2s budget clears the 1.2s step.
        receipt = execute_step_in_process("3_gnn.py", _pipeline_args(tmp_path), LOGGER)
        assert receipt["exit_code"] == 0
        assert receipt["force_killed"] is False

        # GNN_STEP_TIMEOUT_SCALE=0.5 halves the budget to 1s < 1.2s of work.
        monkeypatch.setenv("GNN_STEP_TIMEOUT_SCALE", "0.5")
        monkeypatch.setattr(step_executor, "_CANCEL_GRACE_SECONDS", 0.2)
        timed = execute_step_in_process(
            "3_gnn.py", _pipeline_args(tmp_path / "scaled"), LOGGER
        )
        assert timed["exit_code"] == -1
        assert timed["force_killed"] is False
        assert "TIMEOUT" in timed["stderr"]
