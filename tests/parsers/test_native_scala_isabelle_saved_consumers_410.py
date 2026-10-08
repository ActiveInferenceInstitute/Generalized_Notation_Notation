"""Authored Scala/Isabelle declarations reach saved public interchange unchanged.

No language compiler, theorem prover, or inference backend executes these files.
"""

from pathlib import Path

import pytest

from gnn.parsers import GNNFormat, GNNParsingSystem
from gnn.parsers.common import ParseError
from tests.parsers.test_native_temporal_saved_consumers_410 import save_and_reopen


def test_authored_scala_typed_extents_and_mapping_text_reopen_from_json(
    tmp_path: Path,
) -> None:
    text = """package sensor.evidence
object SensorEvidence {
case class State(dim: Int = 3)
case class Observation(size: Int = 4)
case class Action(dim: Int = 2)
case class Policy(size: Int = 5)
val state_index: Int = 1
val observation_flag: Boolean = false
val likelihood_matrix: Matrix[Fin[3], Fin[2], Double] = authored_likelihood
val transition_matrix: Matrix[Fin[2], Fin[2], Complex] = authored_transition
val policy_scores: Vector[Fin[5], Float] = authored_policy
val preference_weights: Vector[Fin[3], Double] = signed_preferences
val prior_counts: Vector[Fin[2], Long] = authored_prior
type Probability = Double
LikelihoodMapping(matrix = authored_likelihood)
TransitionMapping(matrix = authored_transition)
PreferenceMapping(values = signed_preferences)
PriorMapping(vector = authored_prior)
// q(state) = posterior(state)
/* F = KL(q || prior) */
}
"""
    parsed, reopened, payload = save_and_reopen(tmp_path / "sensor.scala", text)

    expected_variables = {
        "State": ("hidden_state", "categorical", [3]),
        "Observation": ("observation", "categorical", [4]),
        "Action": ("action", "categorical", [2]),
        "Policy": ("policy", "categorical", [5]),
        "state_index": ("hidden_state", "integer", [1]),
        "observation_flag": ("observation", "binary", [1]),
        "likelihood_matrix": ("likelihood_matrix", "float", [3, 2]),
        "transition_matrix": ("transition_matrix", "complex", [2, 2]),
        "policy_scores": ("policy", "float", [5]),
        "preference_weights": ("preference_vector", "float", [3]),
        "prior_counts": ("prior_vector", "integer", [2]),
    }
    expected_parameters = {
        "likelihood_mapping": "matrix = authored_likelihood",
        "transition_mapping": "matrix = authored_transition",
        "preferences_mapping": "values = signed_preferences",
        "priors_mapping": "vector = authored_prior",
    }
    for model in (parsed, reopened):
        assert model.model_name == "SensorEvidence"
        assert {
            variable.name: (
                variable.var_type.value,
                variable.data_type.value,
                variable.dimensions,
            )
            for variable in model.variables
        } == expected_variables
        assert {parameter.name: parameter.value for parameter in model.parameters} == (
            expected_parameters
        )
        assert model.extensions["type_aliases"] == {"Probability": "Double"}
        assert {equation.content for equation in model.equations} == {
            "q(state) = posterior(state)",
            "F = KL(q || prior)",
        }
        assert {equation.format for equation in model.equations} == {"ascii"}
    assert parsed.source_format is GNNFormat.SCALA
    assert reopened.source_format is GNNFormat.JSON
    assert payload["extensions"]["type_aliases"] == {"Probability": "Double"}
    assert {row["name"]: row["value"] for row in payload["parameters"]} == (
        expected_parameters
    )


def test_authored_isabelle_types_definitions_and_unverified_formulas_reopen_from_json(
    tmp_path: Path,
) -> None:
    text = """theory SensorEvidence imports Main begin
datatype s_state = State nat
datatype o_sensor = Detected bool
datatype u_control = Control real
datatype pi_choice = Choice "nat list"
datatype metadata_pair = Pair "nat * bool"
datatype signal = Red | Blue
definition gain :: "real" where "gain = -0.125"
definition horizon :: "nat" where "horizon = 3"
definition enabled :: "bool" where "enabled = False"
definition transition :: "nat ⇒ nat" where "transition x = x + 1"
lemma observer_identity: "o_sensor s_state = o_sensor s_state" by simp
theorem reflexive_state: "s_state = s_state" by simp
end
"""
    parsed, reopened, payload = save_and_reopen(tmp_path / "sensor.thy", text)

    expected_variables = {
        "s_state": ("hidden_state", "integer", [1]),
        "o_sensor": ("observation", "binary", [1]),
        "u_control": ("action", "float", [1]),
        "pi_choice": ("policy", "integer", [1]),
        "metadata_pair": ("hidden_state", "integer", [2]),
        "signal": ("hidden_state", "categorical", [1]),
    }
    # Definitions remain authored expression text, rather than evaluated values.
    expected_parameters = {
        "gain": ("gain = -0.125", "real"),
        "horizon": ("horizon = 3", "nat"),
        "enabled": ("enabled = False", "bool"),
        "transition": ("transition x = x + 1", "nat ⇒ nat"),
    }
    expected_equations = {
        "observer_identity": "o_sensor s_state = o_sensor s_state",
        "reflexive_state": "s_state = s_state",
    }
    for model in (parsed, reopened):
        assert model.model_name == "SensorEvidence"
        assert {
            variable.name: (
                variable.var_type.value,
                variable.data_type.value,
                variable.dimensions,
            )
            for variable in model.variables
        } == expected_variables
        assert {
            parameter.name: (parameter.value, parameter.type_hint)
            for parameter in model.parameters
        } == expected_parameters
        assert {equation.label: equation.content for equation in model.equations} == (
            expected_equations
        )
        assert {equation.format for equation in model.equations} == {"isabelle"}
        assert {
            (connection.source_variables[0], connection.target_variables[0])
            for connection in model.connections
        } == {("s_state", "o_sensor")}
    assert parsed.source_format is GNNFormat.ISABELLE
    assert reopened.source_format is GNNFormat.JSON
    assert {
        row["name"]: (row["value"], row["type_hint"]) for row in payload["parameters"]
    } == (expected_parameters)


@pytest.mark.parametrize("failure", ["directory", "invalid_utf8"])
def test_isabelle_read_failure_retains_cause_instead_of_success(
    tmp_path: Path, failure: str
) -> None:
    source = tmp_path / "unreadable.thy"
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
    assert "read" in error
    if failure == "directory":
        assert str(source) in error
    else:
        assert "utf-8" in error and "decode" in error
        assert source.read_bytes() == b"\xff invalid source"


@pytest.mark.parametrize("failure", ["directory", "invalid_utf8"])
def test_scala_read_failure_raises_the_public_causal_parse_error(
    tmp_path: Path, failure: str
) -> None:
    source = tmp_path / "unreadable.scala"
    if failure == "directory":
        source.mkdir()
    else:
        source.write_bytes(b"\xff invalid source")

    with pytest.raises(ParseError) as raised:
        GNNParsingSystem().parse_file(source)

    assert str(source) in str(raised.value)
    assert "Failed to read file" in str(raised.value)
    if failure == "directory":
        assert isinstance(raised.value.__cause__, IsADirectoryError)
    else:
        assert isinstance(raised.value.__cause__, UnicodeDecodeError)
        assert source.read_bytes() == b"\xff invalid source"
