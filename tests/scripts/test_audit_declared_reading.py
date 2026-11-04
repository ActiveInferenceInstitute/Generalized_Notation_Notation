#!/usr/bin/env python3
"""Contract tests for the declaration-reading audit.

Fixtures are hand-built cells. Belief histories come from an independent
naive-loop reference in this file (anchored by hand-computed literals), not
from the tool's own recursion, so the assertions test the verdict logic
instead of re-encoding the tool's output table.

The violation path is the one the superseded fitting readers missed: an
artifact that follows the transposed layout and not the declared reading must
raise a layout finding, never a pass. The live-tree test pins the tool's
declared readings to the repository files that state them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.helpers import load_module_from_path

REPO_ROOT = Path(__file__).resolve().parents[2]
audit_declared_reading = load_module_from_path(
    "audit_declared_reading", REPO_ROOT / "scripts" / "audit_declared_reading.py"
)

# Declared layout: A[observation, state]; B[next_state, previous_state, action].
A = [[0.9, 0.2], [0.1, 0.8]]
B = [[0.7, 0.3], [0.3, 0.7]]
B_ASYMMETRIC = [[0.8, 0.1], [0.2, 0.9]]
A_SYMMETRIC = [[0.5, 0.5], [0.5, 0.5]]
B_SYMMETRIC = [[0.7, 0.3], [0.3, 0.7]]
D = [0.6, 0.4]
ACTIONS = [0, 1]
OBSERVATIONS = [0, 1]

# Hand-computed declared-reading posteriors for the first transition:
# normalize(A[o0=0] * D) = normalize([0.9, 0.2] * [0.6, 0.4]) = [27/31, 4/31],
# then B[a0=0] predicted ([201/310, 109/310]) and observed as o1=1
# (A[1, :] = [0.1, 0.8]) normalised to [201/1073, 872/1073].
HAND_COMPUTED = [
    [27 / 31, 4 / 31],
    [201 / 1073, 872 / 1073],
]


def _reference_beliefs(
    a, b, d, actions, observations, *, a_row: bool, b_as_declared: bool
) -> list[list[float]]:
    """Independent prior-first recursion, written as explicit loops."""
    states = len(d)
    first = [
        a[observations[0]][state] if a_row else a[state][observations[0]]
        for state in range(states)
    ]
    posterior = [first[state] * d[state] for state in range(states)]
    total = sum(posterior)
    history = [[value / total for value in posterior]]
    for index in range(len(actions) - 1):
        predicted = [0.0] * states
        for next_state in range(states):
            for previous_state in range(states):
                if b_as_declared:
                    weight = b[next_state][previous_state]
                else:
                    weight = b[previous_state][next_state]
                predicted[next_state] += weight * history[-1][previous_state]
        total = sum(predicted)
        predicted = [value / total for value in predicted]
        observation = observations[index + 1]
        likelihood = [
            a[observation][state] if a_row else a[state][observation]
            for state in range(states)
        ]
        posterior = [likelihood[state] * predicted[state] for state in range(states)]
        total = sum(posterior)
        history.append([value / total for value in posterior])
    return history


def _judge(
    name: str,
    beliefs: list[list[float]],
    *,
    a=A,
    b=B,
    d=D,
    actions=ACTIONS,
    observations=OBSERVATIONS,
) -> dict:
    return audit_declared_reading.judge_cell(
        name,
        a_matrix=a,
        b_matrix=b,
        d_vector=d,
        actions=actions,
        observations=observations,
        beliefs=beliefs,
        frame="jax",
    )


def test_hand_computed_literals_anchor_the_reference() -> None:
    honest = _reference_beliefs(
        A, B, D, ACTIONS, OBSERVATIONS, a_row=True, b_as_declared=True
    )
    assert len(honest) == len(HAND_COMPUTED)
    for row, expected in zip(honest, HAND_COMPUTED):
        assert row == pytest.approx(expected, abs=1e-9)


def test_declared_reading_reproduces_and_exits_zero(capsys) -> None:
    record = _judge(
        "jax/honest",
        _reference_beliefs(
            A, B, D, ACTIONS, OBSERVATIONS, a_row=True, b_as_declared=True
        ),
    )
    assert record["outcome"] == "declared-reading-reproduced"
    assert record["declared_error"] <= audit_declared_reading.DEFAULT_ATOL
    assert "finding" not in record
    assert audit_declared_reading.main([], cells=[record]) == 0
    assert "declared-reading-reproduced" in capsys.readouterr().out


def test_a_column_layout_is_a_finding_not_a_pass() -> None:
    """The superseded readers passed this cell; the declared reading fails."""
    record = _judge(
        "jax/transposed-a",
        _reference_beliefs(
            A, B, D, ACTIONS, OBSERVATIONS, a_row=False, b_as_declared=True
        ),
    )
    assert record["outcome"] == "layout-finding"
    assert "a-column" in record["alternative_readings_reproduced"]
    assert record["declared_error"] > audit_declared_reading.DEFAULT_ATOL
    assert record["finding"]["declared_reading"] == (
        audit_declared_reading.DECLARED_READING
    )
    assert audit_declared_reading.main([], cells=[record]) == 1


def test_b_transposed_layout_is_a_finding() -> None:
    record = _judge(
        "jax/transposed-b",
        _reference_beliefs(
            A, B_ASYMMETRIC, D, ACTIONS, OBSERVATIONS, a_row=True, b_as_declared=False
        ),
        b=B_ASYMMETRIC,
    )
    assert record["outcome"] == "layout-finding"
    assert "b-transposed" in record["alternative_readings_reproduced"]
    assert record["declared_error"] > audit_declared_reading.DEFAULT_ATOL


def test_symmetric_matrices_report_alternatives_without_a_finding() -> None:
    record = _judge(
        "jax/symmetric",
        _reference_beliefs(
            A_SYMMETRIC,
            B_SYMMETRIC,
            D,
            ACTIONS,
            OBSERVATIONS,
            a_row=True,
            b_as_declared=True,
        ),
        a=A_SYMMETRIC,
        b=B_SYMMETRIC,
    )
    assert record["outcome"] == "declared-reading-reproduced"
    assert record["alternative_also_reproduces"] is True
    assert "finding" not in record
    assert audit_declared_reading.main([], cells=[record]) == 0


def test_unexplained_history_is_reported_not_a_finding() -> None:
    other_model = _reference_beliefs(
        A,
        B_ASYMMETRIC,
        [0.9, 0.1],
        ACTIONS,
        OBSERVATIONS,
        a_row=True,
        b_as_declared=True,
    )
    record = _judge("jax/unexplained", other_model)
    assert record["outcome"] == "not-reproduced"
    assert "finding" not in record
    assert audit_declared_reading.main([], cells=[record]) == 2


def test_not_comparable_cell_reports_a_reason() -> None:
    record = _judge("jax/wide", [[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
    assert record["outcome"] == "not-comparable"
    assert record["reason"]
    assert audit_declared_reading.main([], cells=[record]) == 2


def test_soft_mode_reports_findings_and_exits_zero() -> None:
    record = _judge(
        "jax/transposed-a",
        _reference_beliefs(
            A, B, D, ACTIONS, OBSERVATIONS, a_row=False, b_as_declared=True
        ),
    )
    assert audit_declared_reading.main(["--soft"], cells=[record]) == 0


def test_json_payload_counts_findings(capsys) -> None:
    honest = _judge(
        "jax/honest",
        _reference_beliefs(
            A, B, D, ACTIONS, OBSERVATIONS, a_row=True, b_as_declared=True
        ),
    )
    violating = _judge(
        "jax/transposed-a",
        _reference_beliefs(
            A, B, D, ACTIONS, OBSERVATIONS, a_row=False, b_as_declared=True
        ),
    )
    assert audit_declared_reading.main(["--json"], cells=[honest, violating]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["layout_findings"] == 1
    assert payload["summary"]["reproduced"] == 1
    assert payload["summary"]["finding_cells"] == ["jax/transposed-a"]
    assert payload["declared_reading"] == audit_declared_reading.DECLARED_READING


def test_declared_readings_match_the_repository_contract() -> None:
    """Live-tree pin: the judged declaration is the one the repository states."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from gnn.render.pomdp_contract import CANONICAL_B_ORDER

    assert audit_declared_reading.DECLARED_B_ORDER == CANONICAL_B_ORDER
    extractor = (REPO_ROOT / "src/gnn/render/jax/jax_spec_extract.py").read_text(
        encoding="utf-8"
    )
    assert "[observation, state]" in extractor
    contracts = (REPO_ROOT / "src/gnn/render/execution_contracts.py").read_text(
        encoding="utf-8"
    )
    assert audit_declared_reading.DECLARED_B_ORDER in contracts
