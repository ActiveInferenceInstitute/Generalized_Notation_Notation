#!/usr/bin/env python3
"""Pin the PyMDP model-kind gate (`pymdp_kind_refusal` + its choke point).

PyMDP renders only discrete POMDPs with categorical A/B/C/D[/E] matrices.
These tests assert every branch of ``pymdp_kind_refusal`` and that
``run_pymdp_simulation`` refuses continuous/composed specs BEFORE importing
pymdp (the gate precedes ``_require_pymdp_1``), while discrete specs keep the
existing ImportError-shaped failure contract.

Ground truth probed 2026-09-23: a *garbage text* file parses vacuously as a
STRUCTURAL ``POMDPStateSpace`` (extractor succeeds with defaults), so the
structured ``failed`` receipt only fires for files the extractor genuinely
cannot read (missing / undecodable) — both are pinned below.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnn.execute.pymdp.simulation import (  # noqa: E402
    pymdp_kind_refusal,
    run_pymdp_simulation,
)
from gnn.extract.pomdp_extractor import extract_pomdp_from_file  # noqa: E402
from gnn.render.pomdp_contract import (  # noqa: E402
    ModelKind,
    detect_pomdp_space_model_kinds,
)

PROJECT_ROOT = SRC

#: The full linear-Gaussian contract every continuous exemplar declares
#: (mirrors tests/render/test_composed_model_kinds.py::_LGSSM_BLOCK).
_LGSSM_BLOCK = {
    "F": [[1.0, 0.0], [0.0, 1.0]],
    "H": [[1.0, 0.0], [0.0, 1.0]],
    "Q": [[0.05, 0.0], [0.0, 0.05]],
    "R": [[0.1, 0.0], [0.0, 0.1]],
    "prior_mean": [0.0, 0.0],
    "prior_cov": [[0.5, 0.0], [0.0, 0.5]],
}

_DISCRETE_SPEC: dict[str, Any] = {
    "initialparameterization": {
        "A": [[1.0], [1.0]],
        "B": [[[1.0, 0.0], [0.0, 1.0]]],
        "C": [0.0, 0.0],
        "D": [1.0, 0.0],
    },
}


def test_discrete_spec_not_refused() -> None:
    """(a) A categorical discrete spec passes the gate untouched."""
    assert pymdp_kind_refusal(dict(_DISCRETE_SPEC)) is None


def test_singleton_continuous_spec_refused() -> None:
    """(b) A pure LGSSM spec is refused with the continuous-spec reason."""
    spec = {"initialparameterization": dict(_LGSSM_BLOCK)}
    gate = pymdp_kind_refusal(spec)
    assert gate is not None
    assert gate["success"] is False
    assert gate["unsupported"] is True
    assert gate["status"] == "unsupported"
    assert gate["reason"].startswith("continuous-spec:")
    assert gate["model_kinds"] == ["continuous"]


def test_composed_continuous_spec_refused_as_composition() -> None:
    """(c) LGSSM + multi-agent inside initialparameterization → composition."""
    initial = dict(_LGSSM_BLOCK)
    initial["nr_agents"] = 2
    spec = {"initialparameterization": initial}
    gate = pymdp_kind_refusal(spec)
    assert gate is not None
    assert gate["success"] is False
    assert gate["unsupported"] is True
    assert gate["status"] == "unsupported"
    assert gate["reason"].startswith("unsupported-composition:")
    assert gate["model_kinds"] == ["continuous", "multi_agent"]


def test_lightweight_parse_receipt_composed_exemplar() -> None:
    """(d) A parse receipt (file_path only) refuses via late extraction."""
    exemplar = PROJECT_ROOT / "input/gnn_files/continuous/multi_agent_lgssm.md"
    assert exemplar.exists(), f"exemplar missing: {exemplar}"
    receipt: dict[str, Any] = {"success": True, "file_path": str(exemplar)}
    gate = pymdp_kind_refusal(receipt)
    assert gate is not None
    assert gate["unsupported"] is True
    assert gate["status"] == "unsupported"
    assert gate["reason"].startswith("unsupported-composition:")
    assert gate["model_kinds"] == ["continuous", "multi_agent"]


def test_lightweight_parse_receipt_plain_continuous_exemplar() -> None:
    """(e) A plain (singleton-continuous) exemplar file refuses likewise."""
    folder = PROJECT_ROOT / "input/gnn_files/continuous"
    singleton_path = None
    for candidate in sorted(folder.glob("*.md")):
        if candidate.name == "multi_agent_lgssm.md":
            continue
        space, _errors = extract_pomdp_from_file(
            candidate, strict_validation=False, on_error="collect"
        )
        if space is None:
            continue
        if detect_pomdp_space_model_kinds(space) == frozenset({ModelKind.CONTINUOUS}):
            singleton_path = candidate
            break
    if singleton_path is None:
        # Zero-skip contract (tests/test_zero_skip_contracts.py) bans
        # pytest.skip: the exemplar corpus is committed, so a miss means
        # the corpus contract broke and must fail loudly.
        raise AssertionError(
            "no singleton-continuous exemplar in input/gnn_files/continuous"
        )
    receipt: dict[str, Any] = {"success": True, "file_path": str(singleton_path)}
    gate = pymdp_kind_refusal(receipt)
    assert gate is not None
    assert gate["unsupported"] is True
    assert gate["status"] == "unsupported"
    assert gate["reason"].startswith("continuous-spec:")
    assert gate["model_kinds"] == ["continuous"]


def test_unreadable_file_receipt_failed(tmp_path: Path) -> None:
    """(f) An extractor-unreadable receipt yields a structured failed receipt.

    Only files the extractor cannot read at all (missing, undecodable)
    return ``space is None``; see module docstring for the garbage-text
    ground truth pinned by the follow-up assertion.
    """
    receipt: dict[str, Any] = {
        "success": True,
        "file_path": str(tmp_path / "does_not_exist.md"),
    }
    gate = pymdp_kind_refusal(receipt)
    assert gate is not None
    assert gate["success"] is False
    assert gate["unsupported"] is False
    assert gate["status"] == "failed"
    assert "pymdp-kind-gate: extraction failed" in gate["reason"]
    assert "GNN-E999" in gate["reason"]

    # Garbage TEXT parses vacuously as STRUCTURAL → gate returns None (the
    # engine's existing matrix checks fail loud downstream), NOT a refusal.
    garbage = tmp_path / "garbage.md"
    garbage.write_text("not a gnn file", encoding="utf-8")
    assert pymdp_kind_refusal({"file_path": str(garbage)}) is None


def test_run_pymdp_refuses_composed_without_pymdp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(g) The gate precedes _require_pymdp_1: no pymdp import needed."""

    def _boom() -> Any:
        raise AssertionError("_require_pymdp_1 must not run for refusals")

    monkeypatch.setattr("gnn.execute.pymdp.simulation._require_pymdp_1", _boom)
    spec = {"initialparameterization": {**_LGSSM_BLOCK, "nr_agents": 2}}
    success, receipt = run_pymdp_simulation(spec, tmp_path)
    assert success is False
    assert receipt["unsupported"] is True
    assert receipt["status"] == "unsupported"
    assert receipt["reason"].startswith("unsupported-composition:")


def test_run_pymdp_discrete_still_reaches_pymdp_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(h) Discrete specs pass the gate and keep the ImportError contract."""

    def _boom() -> Any:
        raise ImportError("pymdp unavailable (pinned test)")

    monkeypatch.setattr("gnn.execute.pymdp.simulation._require_pymdp_1", _boom)
    success, receipt = run_pymdp_simulation(dict(_DISCRETE_SPEC), tmp_path)
    assert success is False
    assert "pymdp unavailable (pinned test)" in receipt["error"]
    assert "suggestion" in receipt
