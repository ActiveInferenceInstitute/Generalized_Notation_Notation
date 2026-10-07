#!/usr/bin/env python3
"""W8-B empirical resource measurement: child peak-RSS + in-process trio.

Covers the three additive measurement surfaces:

- the subprocess envelope's ``child_peak_rss_mb`` across a memory-allocating
  child (peak observed on the poll cadence, loose lower bound only),
- the in-process pipeline naming trio (``memory_usage_mb`` /
  ``peak_memory_mb`` / ``memory_delta_mb``) on the kronecker execution
  envelope and the pymdp rollout results,
- the ``execution_summary.json`` slim keep-list carrying the new envelope
  keys through ``_write_execution_summaries``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.execute.jax.kronecker_executor import (  # noqa: E402
    run_kronecker_factorized_execution,
)
from gnn.execute.processor import _write_execution_summaries  # noqa: E402
from gnn.execute.subprocess_envelope import run_subprocess_envelope  # noqa: E402

PYTHON = sys.executable
LOGGER = logging.getLogger(__name__)

RSS_KEYS = ("child_peak_rss_mb", "rss_sample_interval_seconds", "rss_samples_count")
TRIO_KEYS = ("memory_usage_mb", "peak_memory_mb", "memory_delta_mb")

_TINY_A = [[0.9, 0.1], [0.1, 0.9]]
_TINY_B = [[[0.9, 0.1], [0.1, 0.9]]]  # (states, states, actions): one action
_TINY_C = [0.0, 1.0]
_TINY_D = [0.5, 0.5]


def _tiny_spec(horizon: int = 4) -> dict[str, Any]:
    """Minimal discrete POMDP spec for ``run_pymdp_simulation``."""
    return {
        "model_name": "resource-trio-test",
        "gnn_section": "ActInfPOMDP",
        "initialparameterization": {
            "A": _TINY_A,
            "B": _TINY_B,
            "C": _TINY_C,
            "D": _TINY_D,
        },
        "model_parameters": {
            "num_timesteps": horizon,
            "num_hidden_states": 2,
            "num_obs": 2,
            "num_actions": 1,
            "batch_size": 1,
            "policy_len": 1,
            "random_seed": 7,
        },
    }


def test_envelope_child_peak_rss_tracks_allocating_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Peak sampled across the poll slices bounds the child's real footprint."""
    from gnn.execute import subprocess_envelope

    release = tmp_path / "sample-observed"
    original_sample = subprocess_envelope._ChildRssSampler.sample

    def sample_and_release(sampler: Any) -> None:
        original_sample(sampler)
        if sampler.peak_rss_mb is not None and sampler.peak_rss_mb >= 30.0:
            release.touch()

    monkeypatch.setattr(
        subprocess_envelope._ChildRssSampler, "sample", sample_and_release
    )
    # Touch every page and hold it until the real sampler observes it. A fixed
    # 0.6-second sleep can finish between samples when the parent is descheduled
    # by the full suite. This handshake has a finite failure bound and exercises
    # the production sampler rather than substituting an RSS value.
    child = """
import pathlib, sys, time
release = pathlib.Path(sys.argv[1])
blob = bytearray(50 * 1024 * 1024)
deadline = time.monotonic() + 15
while not release.exists():
    for offset in range(0, len(blob), 4096):
        blob[offset] = 1
    if time.monotonic() >= deadline:
        raise RuntimeError('parent did not observe resident allocation')
    time.sleep(0.01)
"""
    result = run_subprocess_envelope([PYTHON, "-c", child, str(release)], timeout=20)
    assert result["success"] is True, result.get("error")
    peak = result["child_peak_rss_mb"]
    assert peak is not None, result
    # Loose lower bound only: psutil RSS counts shared pages and samples on a
    # 0.25s cadence, so receipts never claim an exact peak (macOS included).
    assert peak >= 30.0, peak
    assert result["rss_samples_count"] > 0
    assert release.exists()


def test_kronecker_envelope_carries_memory_trio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """In-process kronecker run reports the pipeline naming trio."""
    import gnn.execute.jax.kronecker_factorized as kronecker_factorized

    monkeypatch.setattr(
        kronecker_factorized,
        "run_factorized_active_inference",
        lambda model: {
            "success": True,
            "model_kind": "synthetic",
            "validation": {"all_valid": True},
            "model_parameters": {},
        },
    )
    envelope = run_kronecker_factorized_execution(
        object(), tmp_path, model_name="synthetic"
    )
    assert envelope["success"] is True
    for key in TRIO_KEYS:
        assert key in envelope
        assert isinstance(envelope[key], float)
    assert envelope["memory_usage_mb"] > 0
    assert envelope["peak_memory_mb"] >= envelope["memory_usage_mb"]
    # The persisted runtime summary carries the same trio.
    for key in TRIO_KEYS:
        assert key in envelope["summary"]


@pytest.mark.needs_pymdp
def test_pymdp_rollout_carries_memory_trio(tmp_path: Path) -> None:
    """Real pymdp rollout results carry the pipeline naming trio."""
    from gnn.execute.pymdp.simulation import run_pymdp_simulation

    success, results = run_pymdp_simulation(_tiny_spec(), tmp_path / "run")
    assert success, results.get("error", results)
    for key in TRIO_KEYS:
        assert key in results
        assert isinstance(results[key], float)
    assert results["memory_usage_mb"] > 0
    assert results["peak_memory_mb"] >= results["memory_usage_mb"]


def test_slim_keep_list_carries_envelope_rss_keys(tmp_path: Path) -> None:
    """The three envelope keys survive execution_summary.json slimming."""
    source = tmp_path / "input"
    source.mkdir()
    script = source / "demo.py"
    script.write_text("print(1)")
    result: dict[str, Any] = {
        "run_id": "run-a",
        "configuration": {"frameworks": ["pymdp"]},
        "target_directory": str(source),
        "status": "success",
        "success": True,
        "exit_code": 0,
        "execution_details": [
            {
                "script_path": str(script),
                "framework": "pymdp",
                "success": True,
                "child_peak_rss_mb": 12.34,
                "rss_sample_interval_seconds": 0.25,
                "rss_samples_count": 5,
            }
        ],
        "successful_executions": 1,
        "failed_executions": 0,
    }
    result.update(timestamp="2026-09-04", output_directory=str(tmp_path))
    result["execution_details"][0].update(
        script_name="demo.py", executor="python", stdout="full text"
    )
    _write_execution_summaries(tmp_path, result, True, LOGGER)
    slim = json.loads((tmp_path / "summaries/execution_summary.json").read_text())
    detail = json.loads(
        (tmp_path / "summaries/execution_summary_detail.json").read_text()
    )
    for key in RSS_KEYS:
        expected = result["execution_details"][0][key]
        assert slim["execution_details"][0][key] == expected
        assert detail["execution_details"][0][key] == expected
