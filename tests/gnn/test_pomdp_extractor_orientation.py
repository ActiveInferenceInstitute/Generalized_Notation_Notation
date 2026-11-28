"""B-matrix orientation contract: provenance detection, contradiction, canonicalization.

Contract under test (pinned decisions #1, #11, #12):
- matrix_provenance["B"] gains declared_order / detected_order / canonical_order /
  contradiction (contradiction only on positive evidence, never on
  doubly-stochastic ambiguity).
- strict_validation=True surfaces a contradiction as a structured error;
  stored B_matrix is NEVER transposed at extraction time.
- canonicalize_pomdp(spec) returns a NEW spec with B in
  (next_state, previous_state, action) order and leaves the input untouched.

Fixture strategy: the committed actinf exemplar is comment-consistent (impl-
bcanonical fixed it), so tests inject their own contradictory B comments into
a tmp_path copy rather than depending on the exemplar's phrasing.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from gnn.extract.pomdp_extractor import (
    canonicalize_pomdp,
    extract_pomdp_from_file,
)

REPO = Path(__file__).resolve().parents[2]

# Row-stochastic-only slices: each row sums to 1.0, no column does —
# orientation detection is decisive (positive evidence).
ROW_STOCHASTIC_SLICES: list[list[list[float]]] = [
    [[0.8, 0.2, 0.0], [0.7, 0.3, 0.0], [0.0, 0.6, 0.4]],
    [[0.1, 0.9, 0.0], [0.0, 0.4, 0.6], [0.3, 0.3, 0.4]],
    [[0.6, 0.4, 0.0], [0.2, 0.2, 0.6], [0.5, 0.0, 0.5]],
]
# Doubly-stochastic slices (permutations): ambiguity — never a contradiction.
DOUBLY_STOCHASTIC_SLICES: list[list[list[float]]] = [
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
]


def _b_literal(slices: list[list[list[float]]]) -> str:
    """Render 3 slices as a GNN InitialParameterization nested literal."""
    body = ",\n".join(
        "  ( "
        + ", ".join("(" + ", ".join(f"{v:.1f}" for v in row) + ")" for row in slc)
        + " )"
        for slc in slices
    )
    return "B={\n" + body + "\n}"


def _gnn_file(
    tmp_path: Path,
    name: str,
    slices: list[list[list[float]]],
    *,
    declared_comment: str = (
        "# Transition matrix: B[next_state, previous_state, actions]"
    ),
    parameter_comment: str = (
        "# B: stored as B[action, previous_state, next_state]; each slice is a "
        "transition matrix with rows as previous states and columns as next "
        "states."
    ),
) -> Path:
    """Copy of the actinf exemplar with test-controlled B comments and data.

    Both injected comments describe the semantic axis order; the contradiction
    tests pass comments that disagree with each other and with the data's
    detected orientation. Comment text is fully test-owned, so exemplar
    phrasing changes never break this fixture.
    """
    template = (
        REPO / "input" / "gnn_files" / "discrete" / "actinf_pomdp_agent.md"
    ).read_text()
    content = template
    # Replace the declaration comment line (match the current '# Transition
    # matrix: B[...]' line by regex-free prefix split) and the whole
    # InitialParameterization B comment + literal block.
    lines = content.splitlines(keepends=True)
    out: list[str] = []
    in_b_param_comment = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# Transition matrix: B["):
            out.append(declared_comment + "\n")
            continue
        if stripped.startswith("# B: 3 states") and stripped.endswith(
            "action selection."
        ):
            out.append(parameter_comment + "\n")
            in_b_param_comment = True
            continue
        if in_b_param_comment:
            if stripped.startswith("B={"):
                in_b_param_comment = False  # literal follows; keep it for now
            out.append(line)
            continue
        out.append(line)
    content = "".join(out)
    # Swap the B literal for the test-controlled slices.
    content = content.replace(
        "B={\n"
        "  ( (1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0) ),\n"
        "  ( (0.0,1.0,0.0), (1.0,0.0,0.0), (0.0,0.0,1.0) ),\n"
        "  ( (0.0,0.0,1.0), (0.0,1.0,0.0), (1.0,0.0,0.0) )\n"
        "}",
        _b_literal(slices),
    )
    assert _b_literal(slices) in content, "B literal swap failed"
    assert declared_comment in content and parameter_comment in content, (
        "test-owned B comments not injected"
    )
    path = tmp_path / name
    path.write_text(content)
    return path


def _b_provenance(spec: Any) -> dict[str, Any]:
    provenance = spec.matrix_provenance or {}
    assert "B" in provenance, f"matrix_provenance missing B: {provenance.keys()}"
    return cast(dict[str, Any], provenance["B"])


def _extract_collect(path: Path, strict: bool = True) -> tuple[Any, list[Any]]:
    """extract with on_error='collect', asserting the (spec, errors) tuple shape."""
    result: Any = extract_pomdp_from_file(
        path, strict_validation=strict, on_error="collect"
    )
    assert isinstance(result, tuple) and len(result) == 2
    spec, errors = result
    assert isinstance(errors, list)
    return spec, errors


def test_row_stochastic_only_with_contradictory_comments_flags_contradiction(
    tmp_path: Path,
) -> None:
    """Positive evidence + disagreeing comments -> contradiction error record."""
    path = _gnn_file(
        tmp_path, "b_row_stochastic_contradictory.md", ROW_STOCHASTIC_SLICES
    )
    spec, errors = _extract_collect(path, strict=True)
    assert spec is not None
    prov_b = _b_provenance(spec)
    for key in (
        "declared_order",
        "declared_order_explicit",
        "detected_order",
        "canonical_order",
    ):
        assert key in prov_b, f"matrix_provenance['B'] missing {key}: {prov_b}"
    assert prov_b["contradiction"] is True
    assert len(errors) >= 1, "contradiction must surface as a structured error"


def test_doubly_stochastic_is_ambiguous_never_contradiction(tmp_path: Path) -> None:
    """Doubly-stochastic data is ambiguous: no contradiction, no error."""
    path = _gnn_file(tmp_path, "b_doubly_stochastic.md", DOUBLY_STOCHASTIC_SLICES)
    spec, errors = _extract_collect(path, strict=True)
    assert spec is not None
    prov_b = _b_provenance(spec)
    assert prov_b["contradiction"] is False
    assert errors == [], f"doubly-stochastic ambiguity must not error: {errors}"


def test_declared_order_explicit_tracks_comment_presence(tmp_path: Path) -> None:
    """declared_order_explicit separates parsed declarations from defaults."""
    declared = _gnn_file(tmp_path, "b_declared.md", DOUBLY_STOCHASTIC_SLICES)
    spec, _errors = _extract_collect(declared, strict=True)
    prov_b = _b_provenance(spec)
    assert prov_b["declared_order_explicit"] is True

    unstated = _gnn_file(
        tmp_path,
        "b_unstated.md",
        DOUBLY_STOCHASTIC_SLICES,
        declared_comment="# Transition matrix for the context process",
    )
    spec_unstated, _errors_unstated = _extract_collect(unstated, strict=True)
    prov_unstated = _b_provenance(spec_unstated)
    assert prov_unstated["declared_order_explicit"] is False
    assert prov_unstated["declared_order"] == prov_unstated["canonical_order"]


def test_extraction_never_transposes_stored_b(tmp_path: Path) -> None:
    """Stored B_matrix is as-written; orientation work happens in provenance."""
    path = _gnn_file(
        tmp_path, "b_row_stochastic_contradictory.md", ROW_STOCHASTIC_SLICES
    )
    spec, _errors = _extract_collect(path, strict=False)
    assert spec is not None
    assert spec.B_matrix is not None
    assert [list(map(list, slc)) for slc in spec.B_matrix] == ROW_STOCHASTIC_SLICES


def test_canonicalize_pomdp_transposes_detected_action_first_fixture(
    tmp_path: Path,
) -> None:
    """canonicalize_pomdp: (action, prev, next) -> (next, prev, action); input untouched."""
    path = _gnn_file(tmp_path, "b_action_first.md", ROW_STOCHASTIC_SLICES)
    spec, _errors = _extract_collect(path, strict=False)
    assert spec is not None and spec.B_matrix is not None
    before = copy.deepcopy(spec.B_matrix)

    canonical = canonicalize_pomdp(spec)

    assert canonical is not spec, "canonicalize_pomdp must return a new spec"
    orig = spec.B_matrix
    assert orig == before, "input spec.B_matrix must be untouched"
    # (next, prev, action) = transpose of the as-stored (action, prev, next)
    expected = [
        [[orig[a][s][s_next] for a in range(3)] for s in range(3)]
        for s_next in range(3)
    ]
    assert canonical.B_matrix == expected


def test_canonicalize_leaves_all_other_fields_intact(tmp_path: Path) -> None:
    """Canonicalization is a pure B-order copy: counts and A stay identical."""
    path = _gnn_file(tmp_path, "b_action_first.md", ROW_STOCHASTIC_SLICES)
    spec, _errors = _extract_collect(path, strict=False)
    assert spec is not None
    canonical = canonicalize_pomdp(spec)
    assert canonical.num_states == spec.num_states
    assert canonical.num_observations == spec.num_observations
    assert canonical.num_actions == spec.num_actions
    assert canonical.A_matrix == spec.A_matrix


def test_bistable_exemplar_keeps_authored_column_stochastic_action_blocks() -> None:
    source = REPO / "input" / "gnn_files" / "discrete" / "two_state_bistable.md"
    spec = extract_pomdp_from_file(source, strict_probabilities=True, on_error="raise")
    assert spec is not None
    stored = np.array(spec.B_matrix)
    expected = np.stack([[[0.8, 0.3], [0.2, 0.7]], [[0.3, 0.8], [0.7, 0.2]]], axis=-1)
    assert _b_provenance(spec)["detected_order"] == [
        "action",
        "next_state",
        "previous_state",
    ]
    canonical = canonicalize_pomdp(spec)
    np.testing.assert_array_equal(canonical.B_matrix, expected)
    np.testing.assert_array_equal(canonicalize_pomdp(canonical).B_matrix, expected)
    np.testing.assert_array_equal(spec.B_matrix, stored)


@pytest.mark.parametrize(
    ("order", "stored_axes"),
    [
        ("next_state_previous_state_action", (0, 1, 2)),
        ("action_next_state_previous_state", (2, 0, 1)),
        ("action_previous_state_next_state", (2, 1, 0)),
    ],
)
def test_transition_consumers_use_the_same_declared_layout(order, stored_axes):
    from gnn.execute.pymdp.simulation import _canonicalise_B
    from gnn.render.pomdp_contract import canonicalise_b_matrix
    from gnn.visualization.matrix.extract import collect_visualization_matrices

    action_zero = [[0.7, 0.1, 0.2], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]]
    action_one = [[0.4, 0.2, 0.7], [0.3, 0.3, 0.2], [0.3, 0.5, 0.1]]
    expected = np.stack([action_zero, action_one], axis=-1)
    stored = expected.transpose(stored_axes).tolist()
    params = {"b_tensor_order": order}
    rendered, _ = canonicalise_b_matrix(
        stored, num_states=3, num_actions=2, model_parameters=params
    )
    executed = _canonicalise_B(stored, 3, 2, order)
    displayed = collect_visualization_matrices(
        {"parameters": {"B": stored}, "model_parameters": params}
    )["B"]
    np.testing.assert_allclose(rendered, expected, atol=1e-15, rtol=0)
    np.testing.assert_allclose(executed, expected, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(displayed, expected)
