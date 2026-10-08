"""Authored temporal-language files survive public parsing and saved interchange.

This is Python declaration extraction, not TLA+/Agda compilation or proof checking.
"""

import json
from pathlib import Path

import pytest

from gnn.parsers import GNNFormat, GNNParsingSystem


def save_and_reopen(source: Path, text: str):
    """Use real source bytes, the registered serializer, and the saved JSON reader."""
    source.write_text(text, encoding="utf-8")
    before = source.read_bytes()
    system = GNNParsingSystem()
    parsed = system.parse_file(source)
    assert parsed.success, parsed.errors
    assert parsed.source_file == str(source)
    target = source.parent / "interchange" / f"{source.stem}.json"
    system.serialize_to_file(parsed.model, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    reopened = system.parse_file(target)
    assert reopened.success, reopened.errors
    assert reopened.source_file == str(target)
    assert source.read_bytes() == before
    return parsed.model, reopened.model, payload


def test_authored_tla_scalar_dependencies_and_operator_values_reopen_from_json(
    tmp_path: Path,
) -> None:
    text = r"""---- MODULE SensorClock ----
EXTENDS Integers
VARIABLES hidden_state, observation, action_control, policy_choice, belief
\* Scalar declarations follow above.
CONSTANTS gain, horizon
\* The environment supplies these symbolic constants.
Init == hidden_state = 0 /\ observation = 1

Advance == hidden_state' = hidden_state + action_control

Spec == Init /\ [][Advance]_<<hidden_state, observation, action_control, policy_choice, belief>>

====
"""
    parsed, reopened, payload = save_and_reopen(tmp_path / "clock.tla", text)

    expected_variables = {
        "hidden_state": "hidden_state",
        "observation": "observation",
        "action_control": "action",
        "policy_choice": "policy",
        "belief": "hidden_state",
    }
    expected_equations = {
        "Init": r"hidden_state = 0 /\ observation = 1",
        "Advance": "hidden_state' = hidden_state + action_control",
        "Spec": r"Init /\ [][Advance]_<<hidden_state, observation, action_control, policy_choice, belief>>",
    }
    expected_dependencies = {
        ("hidden_state", "Init"),
        ("observation", "Init"),
        ("hidden_state", "Advance"),
        ("action_control", "Advance"),
        *((name, "Spec") for name in expected_variables),
    }
    for model in (parsed, reopened):
        assert model.model_name == "SensorClock"
        assert {
            variable.name: variable.var_type.value for variable in model.variables
        } == expected_variables
        assert all(variable.dimensions == [1] for variable in model.variables)
        assert all(
            variable.data_type.value == "categorical" for variable in model.variables
        )
        assert {parameter.name: parameter.value for parameter in model.parameters} == {
            "gain": None,
            "horizon": None,
        }
        assert {equation.label: equation.content for equation in model.equations} == (
            expected_equations
        )
        assert {equation.format for equation in model.equations} == {"tla"}
        assert {
            (connection.source_variables[0], connection.target_variables[0])
            for connection in model.connections
        } == expected_dependencies
        assert all(
            connection.connection_type.value == "directed"
            for connection in model.connections
        )
        assert model.time_specification.time_type == "Dynamic"
        assert model.time_specification.discretization == "DiscreteTime"
        assert model.time_specification.horizon == "Unbounded"
    assert parsed.source_format is GNNFormat.TLA_PLUS
    assert reopened.source_format is GNNFormat.JSON
    assert payload["model_name"] == "SensorClock"
    assert {row["label"]: row["content"] for row in payload["equations"]} == (
        expected_equations
    )


@pytest.mark.parametrize("spaced", [False, True])
@pytest.mark.parametrize("body", ["value = value", "value =\n  value"])
def test_authored_agda_types_and_identity_equation_reopen_from_json(
    tmp_path: Path, spaced: bool, body: str
) -> None:
    text = """module Sensor.Evidence where
open import Data.Nat using (ℕ)
open import Data.Bool using (Bool)
open import Data.Real using (ℝ)

data HiddenState : Set where
  counted : ℕ → HiddenState

data Observation : Set where
  detected : Bool → Observation

data Action : Set where
  controlled : ℝ → Action

data Category : Set where
  labelled : Category

identity : HiddenState → HiddenState
identity value = value
"""
    if not spaced:
        text = text.replace("\n\ndata ", "\ndata ")
    text = text.replace("identity value = value", f"identity {body}")
    parsed, reopened, payload = save_and_reopen(tmp_path / "evidence.agda", text)

    expected_variables = {
        "HiddenState": ("hidden_state", "integer"),
        "Observation": ("observation", "binary"),
        "Action": ("action", "float"),
        "Category": ("hidden_state", "categorical"),
    }
    expected_equations = {
        "counted": "counted : ℕ → HiddenState",
        "detected": "detected : Bool → Observation",
        "controlled": "controlled : ℝ → Action",
        "labelled": "labelled : Category",
        "identity": f"identity : HiddenState → HiddenState\nidentity {body}",
    }
    for model in (parsed, reopened):
        assert model.model_name == "Sensor_Evidence"
        assert {
            variable.name: (variable.var_type.value, variable.data_type.value)
            for variable in model.variables
        } == expected_variables
        # Every declaration above has one constructor and no vector extent.
        assert all(variable.dimensions == [1] for variable in model.variables)
        assert {eq.label: eq.content for eq in model.equations} == expected_equations
        assert {eq.format for eq in model.equations} == {"agda"}
        assert {
            (connection.source_variables[0], connection.target_variables[0])
            for connection in model.connections
        } == {
            ("HiddenState", "counted"),
            ("Observation", "detected"),
            ("Bool", "detected"),
            ("Action", "controlled"),
            ("Category", "labelled"),
            ("HiddenState", "identity"),
        }
    assert parsed.source_format is GNNFormat.AGDA
    assert reopened.source_format is GNNFormat.JSON
    assert payload["model_name"] == "Sensor_Evidence"
    assert {row["label"]: row["content"] for row in payload["equations"]} == (
        expected_equations
    )


def test_agda_complete_short_header_does_not_consume_following_data_keyword(
    tmp_path: Path,
) -> None:
    text = """module ShortName where
data HiddenState : Set where
  counted : HiddenState
postulate
  d : HiddenState → HiddenState
data Observation : Set where
  seen : Observation
"""
    parsed, reopened, payload = save_and_reopen(tmp_path / "short.agda", text)

    for model in (parsed, reopened):
        assert model.model_name == "ShortName"
        assert {eq.label: eq.content for eq in model.equations} == {
            "counted": "counted : HiddenState",
            "d": "d : HiddenState → HiddenState",
            "seen": "seen : Observation",
        }
        assert {variable.name for variable in model.variables} == {
            "HiddenState",
            "Observation",
        }
        assert ("HiddenState", "d") in {
            (connection.source_variables[0], connection.target_variables[0])
            for connection in model.connections
        }
    assert (
        next(row for row in payload["equations"] if row["label"] == "d")["content"]
        == "d : HiddenState → HiddenState"
    )


@pytest.mark.parametrize("extension", [".tla", ".agda"])
@pytest.mark.parametrize("failure", ["directory", "invalid_utf8"])
def test_temporal_file_read_failure_cannot_return_a_successful_model(
    tmp_path: Path, extension: str, failure: str
) -> None:
    source = tmp_path / f"unreadable{extension}"
    if failure == "directory":
        source.mkdir()
    else:
        source.write_bytes(b"\xff invalid source")

    result = GNNParsingSystem().parse_file(source)

    assert not result.success
    assert result.source_file == str(source)
    assert (
        result.model.variables
        == result.model.equations
        == result.model.parameters
        == []
    )
    error = " ".join(result.errors)
    assert "Failed to read" in error
    if failure == "directory":
        assert str(source) in error
    else:
        assert "utf-8" in error and "decode" in error
        assert source.read_bytes() == b"\xff invalid source"


def test_missing_temporal_file_retains_its_actual_path_in_the_public_error(
    tmp_path: Path,
) -> None:
    source = tmp_path / "missing.tla"
    with pytest.raises(FileNotFoundError, match="missing.tla"):
        GNNParsingSystem().parse_file(source)
    assert not source.exists()


def test_saved_temporal_interchange_refuses_an_occupied_output_parent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tiny.tla"
    content = "---- MODULE Tiny ----\nVARIABLE state\n\\* scalar\n"
    source.write_text(content, encoding="utf-8")
    parsed = GNNParsingSystem().parse_file(source)
    assert parsed.success, parsed.errors
    occupied = tmp_path / "occupied"
    occupied.write_bytes(b"keep prior artifact")

    with pytest.raises(FileExistsError):
        GNNParsingSystem().serialize_to_file(parsed.model, occupied / "model.json")

    assert occupied.read_bytes() == b"keep prior artifact"
    assert source.read_text(encoding="utf-8") == content
    assert parsed.model.model_name == "Tiny"
