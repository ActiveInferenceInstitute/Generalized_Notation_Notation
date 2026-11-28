"""Opt-in strict probability and dimension contract (#400 item 3).

Contract under test:
- ``strict_probabilities=True`` (requires ``strict_validation=True``) checks the
  as-written discrete tensors: ``A``/``A_*`` observation-axis conditionals and
  ``B``/``B_*`` next-state conditionals sum to 1 (atol 1e-6), ``D``/``D_*``
  priors are probability vectors, entries are finite and non-negative outside
  ``C``, and ``D``/``C`` lengths match ``num_states``/``num_observations``.
- ``on_error="raise"`` raises ``GNNExtractionError`` (``GNN-E007`` probability,
  ``GNN-E002`` dimension) located at the tensor's assignment line;
  ``on_error="collect"`` returns the spec plus structured errors;
  ``on_error="lenient"`` only logs warnings.
- Without the opt-in, ``strict_validation=True`` keeps its prior behavior:
  shipped fixtures with rounded priors or unsupported layouts still extract.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import pytest

from gnn.extract import extract_to_json, main
from gnn.extract.pomdp_extractor import (
    GNNExtractionError,
    POMDPExtractor,
    POMDPStateSpace,
    extract_pomdp_from_content,
    extract_pomdp_from_file,
)
from gnn.extract.pomdp_stochastic import (
    DIMENSION_CODE,
    PROBABILITY_CODE,
    stochastic_diagnostics,
)

REPO = Path(__file__).resolve().parents[2]

VALID_A = "A={ (0.8, 0.1), (0.1, 0.1), (0.1, 0.8) }"
# Canonical B[next, previous, action]: sum over the outer (next) axis is 1 for
# every (previous, action); slices are not row-stochastic, so the stored order
# is decisively detected as canonical.
VALID_B = "B={ ( (0.9, 0.2), (0.3, 0.6) ), ( (0.1, 0.8), (0.7, 0.4) ) }"
VALID_C = "C={ (0.0, 1.0, 2.0) }"
VALID_D = "D={ (0.5, 0.5) }"

TEMPLATE = """## GNNSection
StrictProbe

## ModelName
Strict Probe

## StateSpaceBlock
A[3,2,type=float]       # A[observation, state]
B[2,2,2,type=float]     # B[next_state, previous_state, action]
C[3,type=float]
D[2,type=float]
s[2,1,type=float]
o[3,1,type=int]
u[1,type=int]

## Connections
D>s
s-A
A-o
s-B
C>G

## InitialParameterization
{A}
{B}
{C}
{D}

## Time
Dynamic
"""


def _content(**blocks: str) -> str:
    values = {"A": VALID_A, "B": VALID_B, "C": VALID_C, "D": VALID_D}
    values.update(blocks)
    content = TEMPLATE
    for key, block in values.items():
        content = content.replace("{" + key + "}", block)
    return content


def _line_of(content: str, prefix: str) -> int:
    return next(
        index
        for index, line in enumerate(content.split("\n"), start=1)
        if line.startswith(prefix)
    )


def _strict(content: str, on_error: str, **kwargs: Any) -> Any:
    return extract_pomdp_from_content(
        content,
        strict_validation=True,
        on_error=on_error,  # type: ignore[call-overload]
        strict_probabilities=True,
        **kwargs,
    )


NEGATIVE_CONTROLS = {
    "A_columns_sum_2_7": (
        {"A": "A={ (0.9, 0.9), (0.9, 0.9), (0.9, 0.9) }"},
        PROBABILITY_CODE,
        "A=",
        "sums to 2.7",
    ),
    "B_next_state_sums_6": (
        {"B": "B={ ( (3.0, 3.0), (3.0, 3.0) ), ( (3.0, 3.0), (3.0, 3.0) ) }"},
        PROBABILITY_CODE,
        "B=",
        "sums to 6",
    ),
    "D_length_mismatch": (
        {"D": "D={ (0.25, 0.25, 0.25, 0.25) }"},
        DIMENSION_CODE,
        "D=",
        "length 4 does not match num_states 2",
    ),
    "C_length_mismatch": (
        {"C": "C={ (0.0, 1.0) }"},
        DIMENSION_CODE,
        "C=",
        "length 2 does not match num_observations 3",
    ),
    "A_negative_entry": (
        {"A": "A={ (1.1, 0.1), (-0.2, 0.1), (0.1, 0.8) }"},
        PROBABILITY_CODE,
        "A=",
        "negative probability entry -0.2",
    ),
    "B_negative_entry": (
        {"B": "B={ ( (1.2, 0.2), (0.3, 0.6) ), ( (-0.2, 0.8), (0.7, 0.4) ) }"},
        PROBABILITY_CODE,
        "B=",
        "negative probability entry -0.2",
    ),
    "D_not_normalized": (
        {"D": "D={ (0.7, 0.7) }"},
        PROBABILITY_CODE,
        "D=",
        "prior sums to 1.4",
    ),
}


# --- positive controls -------------------------------------------------------


@pytest.mark.parametrize("on_error", ["raise", "collect", "lenient"])
def test_valid_stochastic_model_passes_and_payload_is_unchanged(
    on_error: str,
) -> None:
    content = _content()
    baseline = extract_pomdp_from_content(content, strict_validation=True)
    result = _strict(content, on_error)
    if on_error == "collect":
        spec, errors = result
        assert errors == []
    else:
        spec = result
    assert spec is not None and baseline is not None
    assert spec.to_dict() == baseline.to_dict()


@pytest.mark.parametrize(
    ("label", "a_block", "b_block"),
    [
        # Doubly stochastic action-outer permutation slices: axis-0 sums are 2,
        # but the orientation is ambiguous, so the action-outer axis admits it.
        (
            "permutation_slices",
            VALID_A,
            "B={ ( (1.0, 0.0), (0.0, 1.0) ), ( (0.0, 1.0), (1.0, 0.0) ) }",
        ),
        # Identical sticky doubly stochastic slices (sum 1.8 over axis 0).
        (
            "sticky_slices",
            VALID_A,
            "B={ ( (0.9, 0.1), (0.1, 0.9) ), ( (0.9, 0.1), (0.1, 0.9) ) }",
        ),
    ],
)
def test_ambiguous_square_b_orientation_is_admitted(
    label: str, a_block: str, b_block: str
) -> None:
    spec, errors = _strict(_content(A=a_block, B=b_block), "collect")
    assert spec is not None
    assert spec.matrix_provenance["B"]["detected_order"] is None, label
    assert errors == [], label


def test_square_doubly_stochastic_a_is_admitted() -> None:
    content = _content(
        A="A={ (0.7, 0.3), (0.3, 0.7) }",
        C="C={ (0.0, 1.0) }",
    ).replace("A[3,2,type=float]", "A[2,2,type=float]")
    content = content.replace("C[3,type=float]", "C[2,type=float]").replace(
        "o[3,1,type=int]", "o[2,1,type=int]"
    )
    spec = _strict(content, "raise")
    assert spec is not None and spec.num_observations == 2


def test_square_a_is_checked_on_the_observation_axis_only() -> None:
    """Row-stochastic (state-major) square A is not silently transposed."""
    content = (
        _content(A="A={ (0.9, 0.1), (0.6, 0.4) }", C="C={ (0.0, 1.0) }")
        .replace("A[3,2,type=float]", "A[2,2,type=float]")
        .replace("C[3,type=float]", "C[2,type=float]")
        .replace("o[3,1,type=int]", "o[2,1,type=int]")
    )
    with pytest.raises(GNNExtractionError) as info:
        _strict(content, "raise")
    assert info.value.code == PROBABILITY_CODE
    assert "observation axis" in info.value.message


def test_detected_action_outer_b_is_checked_on_its_next_state_axis() -> None:
    """Row-stochastic slices with a declared action-outer order pass."""
    content = _content(
        B="B={ ( (0.9, 0.1), (0.4, 0.6) ), ( (0.2, 0.8), (0.5, 0.5) ) }"
    ).replace(
        "B[next_state, previous_state, action]",
        "B[action, previous_state, next_state]",
    )
    spec, errors = _strict(content, "collect")
    assert spec.matrix_provenance["B"]["detected_order"] == [
        "action",
        "previous_state",
        "next_state",
    ]
    assert errors == []


def test_unsupported_b_layout_fails_closed_with_axis_hint() -> None:
    """A previous-state-first array with three actions has no supported order."""
    block = (
        "B={ ( (0.8, 0.3, 0.6), (0.2, 0.7, 0.4) ), "
        "( (0.3, 0.8, 0.1), (0.7, 0.2, 0.9) ) }"
    )
    content = _content(B=block).replace("B[2,2,2,type=float]", "B[2,2,3,type=float]")
    with pytest.raises(GNNExtractionError) as info:
        _strict(content, "raise")
    assert info.value.code == PROBABILITY_CODE
    assert "axis 1 is stochastic" in info.value.message


def test_probability_atol_is_configurable() -> None:
    content = _content(D="D={ (0.49999, 0.5) }")
    with pytest.raises(GNNExtractionError, match="prior sums to 0.99999"):
        _strict(content, "raise")
    assert _strict(content, "raise", probability_atol=1e-4) is not None


# --- negative controls -------------------------------------------------------


@pytest.mark.parametrize("control", sorted(NEGATIVE_CONTROLS))
def test_negative_control_raises_located_structured_error(control: str) -> None:
    blocks, code, anchor, fragment = NEGATIVE_CONTROLS[control]
    content = _content(**blocks)
    with pytest.raises(GNNExtractionError) as info:
        _strict(content, "raise")
    error = info.value
    assert error.code == code
    assert fragment in error.message
    assert error.section == "InitialParameterization"
    assert error.line == _line_of(content, anchor)
    assert error.severity == "error"


@pytest.mark.parametrize("control", sorted(NEGATIVE_CONTROLS))
def test_negative_control_collects_spec_and_errors(control: str) -> None:
    blocks, code, _anchor, fragment = NEGATIVE_CONTROLS[control]
    spec, errors = _strict(_content(**blocks), "collect")
    assert spec is not None, "collect keeps the faithful spec"
    assert [error.code for error in errors] == [code]
    assert fragment in errors[0].message


def test_collect_reports_every_violation_in_payload_order() -> None:
    content = _content(
        A=NEGATIVE_CONTROLS["A_columns_sum_2_7"][0]["A"],
        C=NEGATIVE_CONTROLS["C_length_mismatch"][0]["C"],
        D=NEGATIVE_CONTROLS["D_length_mismatch"][0]["D"],
    )
    _spec, errors = _strict(content, "collect")
    assert [(error.code, error.message.split(":")[0]) for error in errors] == [
        (PROBABILITY_CODE, "A"),
        (DIMENSION_CODE, "C"),
        (DIMENSION_CODE, "D"),
    ]


@pytest.mark.parametrize("control", sorted(NEGATIVE_CONTROLS))
def test_lenient_mode_logs_and_keeps_the_spec(
    control: str, caplog: pytest.LogCaptureFixture
) -> None:
    blocks, code, _anchor, fragment = NEGATIVE_CONTROLS[control]
    with caplog.at_level(logging.WARNING, logger="gnn.extract.pomdp_extractor"):
        spec = _strict(_content(**blocks), "lenient")
    assert spec is not None
    assert any(
        code in record.getMessage() and fragment in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize("control", sorted(NEGATIVE_CONTROLS))
def test_default_strict_validation_keeps_prior_behavior(control: str) -> None:
    """Without the opt-in, strict raise/collect admit the same tensors as before."""
    content = _content(**NEGATIVE_CONTROLS[control][0])
    assert (
        extract_pomdp_from_content(content, strict_validation=True, on_error="raise")
        is not None
    )
    spec, errors = extract_pomdp_from_content(
        content, strict_validation=True, on_error="collect"
    )
    assert spec is not None and errors == []
    spec, errors = extract_pomdp_from_content(
        content, strict_validation=False, on_error="collect"
    )
    assert spec is not None and errors == []


def test_file_entry_point_forwards_the_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "bad_a.md"
    path.write_text(_content(**NEGATIVE_CONTROLS["A_columns_sum_2_7"][0]))
    with pytest.raises(GNNExtractionError) as info:
        extract_pomdp_from_file(path, on_error="raise", strict_probabilities=True)
    assert info.value.code == PROBABILITY_CODE
    assert extract_pomdp_from_file(path, on_error="raise") is not None


# --- configuration -----------------------------------------------------------


def test_strict_probabilities_requires_strict_validation() -> None:
    with pytest.raises(ValueError, match="requires strict_validation=True"):
        POMDPExtractor(strict_validation=False, strict_probabilities=True)
    with pytest.raises(ValueError, match="requires strict_validation=True"):
        extract_pomdp_from_content(
            _content(), strict_validation=False, strict_probabilities=True
        )


@pytest.mark.parametrize("atol", [-1e-6, math.inf, math.nan, True, "1e-6"])
def test_invalid_probability_atol_is_refused(atol: Any) -> None:
    with pytest.raises(ValueError, match="probability_atol"):
        POMDPExtractor(strict_probabilities=True, probability_atol=atol)


# --- helper-level contract ---------------------------------------------------


def _space(**fields: Any) -> POMDPStateSpace:
    base: dict[str, Any] = {"num_states": 2, "num_observations": 2, "num_actions": 1}
    base.update(fields)
    return POMDPStateSpace(**base)


@pytest.mark.parametrize(
    ("matrices", "code", "fragment"),
    [
        ({"A": [[math.nan, 0.5], [0.5, 0.5]]}, PROBABILITY_CODE, "non-finite"),
        ({"B": [[math.inf, 0.0], [0.0, 1.0]]}, PROBABILITY_CODE, "non-finite"),
        ({"C": [0.0, -math.inf]}, PROBABILITY_CODE, "non-finite"),
        ({"D": [0.5, "half"]}, PROBABILITY_CODE, "non-numeric"),
        ({"A": [[1.0, 0.0], [0.0]]}, DIMENSION_CODE, "ragged"),
        ({"A": [0.5, 0.5]}, DIMENSION_CODE, "observation and a state axis"),
        ({"D_f0": [[0.5, 0.5], [0.2, 0.2]]}, PROBABILITY_CODE, "prior row 1"),
        # Factorized A slices: sum over the leading observation axis.
        (
            {"A_m0": [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.6], [0.5, 0.5]]]},
            PROBABILITY_CODE,
            "column [0, 1]",
        ),
        # Time-outer B_t (t, next, previous, action) with one bad conditional.
        (
            {
                "B_t": [
                    [[[0.6, 0.1], [0.3, 0.1]], [[0.4, 0.9], [0.7, 0.9]]],
                    [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.2]]],
                ]
            },
            PROBABILITY_CODE,
            "B_t: conditional",
        ),
    ],
)
def test_helper_rejects(matrices: dict[str, Any], code: str, fragment: str) -> None:
    diagnostics = stochastic_diagnostics(_space(matrices=matrices))
    assert [diagnostic.code for diagnostic in diagnostics] == [code]
    assert fragment in diagnostics[0].message


@pytest.mark.parametrize(
    "matrices",
    [
        # Action-outer factorized B (action, previous, next) as in B_agent*.
        {"B_agent1": [[[0.9, 0.1], [0.2, 0.8]], [[0.5, 0.5], [0.0, 1.0]]]},
        # Single-action passive factor slice and a prior per factor.
        {"B_f1": [[0.7, 0.4], [0.3, 0.6]], "D_f1": [0.25, 0.75]},
        # Preferences are log-scale: negative finite entries are admissible.
        {"C_m0": [-3.0, 0.0, 2.5]},
        # Scalars (parse-level values) are not probability tensors.
        {"A_scale": 2.0},
        # Passive 2-D main B[next, previous] and a 1-D factor transition.
        {"B": [[0.9, 0.2], [0.1, 0.8]], "B_vec": [0.25, 0.75]},
        # No square state pair: any stochastic axis admits the factor slice.
        {"B_x": [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]},
        # Non-probability keys are ignored.
        {"E": [5.0, 5.0], "Alpha": [[9.0]]},
    ],
)
def test_helper_admits(matrices: dict[str, Any]) -> None:
    assert stochastic_diagnostics(_space(matrices=matrices)) == []


def test_helper_reads_attribute_tensors_and_skips_continuous_models() -> None:
    bad = {"A_matrix": [[0.9, 0.9], [0.9, 0.9]]}
    assert [d.tensor for d in stochastic_diagnostics(_space(**bad))] == ["A"]
    assert stochastic_diagnostics(_space(model_kind="continuous", **bad)) == []


# --- headless JSON / CLI -----------------------------------------------------


def test_extract_to_json_and_cli_surface_the_opt_in(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.md"
    bad.write_text(_content(**NEGATIVE_CONTROLS["B_next_state_sums_6"][0]))
    good = tmp_path / "good.md"
    good.write_text(_content())

    envelope = json.loads(
        extract_to_json(bad, on_error="raise", strict_probabilities=True)
    )
    assert envelope["status"] == "error"
    assert envelope["error"]["code"] == PROBABILITY_CODE
    assert envelope["error"]["section"] == "InitialParameterization"
    assert "status" not in json.loads(extract_to_json(bad, on_error="raise"))

    assert main([str(bad), "--strict-probabilities", "--compact"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == PROBABILITY_CODE
    assert main([str(good), "--strict-probabilities"]) == 0
    assert json.loads(capsys.readouterr().out)["model_name"] == "Strict Probe"
    with pytest.raises(SystemExit) as exit_info:
        main([str(bad), "--no-strict", "--strict-probabilities"])
    assert exit_info.value.code == 2


def test_cli_default_branch_keeps_prior_behavior_without_the_opt_in(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without ``--strict-probabilities`` the CLI keeps lenient extraction."""
    bad = tmp_path / "bad.md"
    bad.write_text(_content(**NEGATIVE_CONTROLS["B_next_state_sums_6"][0]))

    assert main([str(bad)]) == 0
    assert json.loads(capsys.readouterr().out)["model_name"] == "Strict Probe"
    assert main([str(bad), "--no-strict", "--compact"]) == 0
    out = capsys.readouterr().out
    assert "\n" not in out.strip()
    assert "status" not in json.loads(out)


# --- shipped fixture census --------------------------------------------------

# Shipped discrete fixtures that the opt-in contract rejects, pinned so a fix
# or a new violation is a visible, reviewed change. Default strict extraction
# of every one of them is unchanged (they remain admitted without the opt-in).
KNOWN_NONSTOCHASTIC_INPUT_FIXTURES = {
    # Uniform priors authored as 0.33333 x 3 (sum 0.99999 > atol 1e-6).
    "discrete/actinf_pomdp_agent.md": {"D"},
    "discrete/multi_armed_bandit.md": {"D"},
    # Phase t=0 conditional (previous=1, action=1) sums to 0.8.
    "discrete/time_varying_dynamics.md": {"B_t"},
    # Generated likelihood column rounds to 0.999969.
    "pymdp_scaling_study/pymdp_scaling_N64_T100.md": {"A"},
}


def _discrete_input_fixtures() -> list[Path]:
    paths: list[Path] = []
    for path in sorted((REPO / "input" / "gnn_files").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        if "## StateSpaceBlock" in text and "## InitialParameterization" in text:
            paths.append(path)
    return paths


def test_shipped_input_fixture_census_matches_the_pinned_set() -> None:
    root = REPO / "input" / "gnn_files"
    observed: dict[str, set[str]] = {}
    checked = 0
    for path in _discrete_input_fixtures():
        spec = extract_pomdp_from_file(path, strict_validation=False)
        if spec is None or spec.model_kind != "discrete":
            continue
        checked += 1
        tensors = {diagnostic.tensor for diagnostic in stochastic_diagnostics(spec)}
        if tensors:
            observed[path.relative_to(root).as_posix()] = tensors
    assert checked >= 20, f"census found only {checked} discrete fixtures"
    assert observed == KNOWN_NONSTOCHASTIC_INPUT_FIXTURES


@pytest.mark.parametrize(
    "relative",
    sorted(KNOWN_NONSTOCHASTIC_INPUT_FIXTURES),
)
def test_known_fixtures_still_extract_under_default_strict_raise(
    relative: str,
) -> None:
    path = REPO / "input" / "gnn_files" / relative
    spec = extract_pomdp_from_file(path, strict_validation=True, on_error="raise")
    assert spec is not None
    _spec, errors = extract_pomdp_from_file(
        path, on_error="collect", strict_probabilities=True
    )
    assert {error.code for error in errors} <= {PROBABILITY_CODE, DIMENSION_CODE}
    assert errors, relative


def test_authored_corpus_adds_only_pinned_strict_errors() -> None:
    """End-to-end opt-in check over an authored GNN corpus (read-only).

    Defaults to the shipped ``input/gnn_files`` corpus, where the opt-in may
    add errors only for the pinned census entries. Set
    ``GNN_STRICT_PROBABILITY_CORPUS`` to an external directory of authored
    specs (e.g. a ``fep_evolution`` checkout) to require that the opt-in adds
    no errors to any discrete spec under it.
    """
    external = os.environ.get("GNN_STRICT_PROBABILITY_CORPUS")
    root = Path(external) if external else REPO / "input" / "gnn_files"
    allowed = set() if external else set(KNOWN_NONSTOCHASTIC_INPUT_FIXTURES)
    checked = 0
    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        if "## StateSpaceBlock" not in text or "## InitialParameterization" not in text:
            continue
        spec = extract_pomdp_from_content(text, strict_validation=False)
        if spec is None or spec.model_kind != "discrete":
            continue
        checked += 1
        _base, baseline = extract_pomdp_from_content(text, on_error="collect")
        _spec, opted = extract_pomdp_from_content(
            text, on_error="collect", strict_probabilities=True
        )
        added = [str(error) for error in opted[len(baseline) :]]
        assert [str(error) for error in opted[: len(baseline)]] == [
            str(error) for error in baseline
        ], f"{path}: the opt-in must not change pre-existing errors"
        if path.relative_to(root).as_posix() not in allowed:
            assert added == [], f"{path}: {added}"
    assert checked, f"no discrete GNN specs under {root}"
