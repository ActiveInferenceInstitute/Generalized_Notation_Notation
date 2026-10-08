"""Public PyMDP source generation preserves data; these tests never run a rollout."""

import ast
import copy
import pickle
import py_compile

import numpy as np
import pytest

from gnn.render.pymdp import render_gnn_to_pymdp


def authored_model():
    return {
        "model_name": "Asymmetric observations Ω",
        "annotation": "Signed preferences and four actions",
        "variables": [
            {"name": "A", "dimensions": [3, 2]},
            {"name": "B", "dimensions": [2, 2, 4]},
        ],
        "model_parameters": {"num_timesteps": 7},
        "initialparameterization": {
            "A": [[0.7, 0.1], [0.2, 0.3], [0.1, 0.6]],
            "B": [
                [[0.9, 0.4, 0.2, 0.75], [0.2, 0.7, 0.1, 0.55]],
                [[0.1, 0.6, 0.8, 0.25], [0.8, 0.3, 0.9, 0.45]],
            ],
            "C": [-2.0, 0.75, 1.25],
            "D": [0.8, 0.2],
            "E": [0.1, 0.2, 0.3, 0.4],
        },
    }


def emitted_main_literals(path):
    """Read the consumer's saved program, without invoking its inference main."""
    module = ast.parse(path.read_text(encoding="utf-8"))
    main = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in main.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id
        in {
            "A_data",
            "B_data",
            "C_data",
            "D_data",
            "E_data",
            "gnn_spec",
            "num_actions_default",
            "num_timesteps",
        }
    }


@pytest.mark.parametrize("mode", ["pipeline", "standalone"])
@pytest.mark.parametrize("representation", ["list", "tuple", "ndarray"])
@pytest.mark.parametrize(
    "parameter_key", ["initialparameterization", "initial_parameterization"]
)
def test_saved_pymdp_program_preserves_nondefault_values_and_compiles(
    tmp_path, mode, representation, parameter_key
):
    spec = authored_model()
    expected = copy.deepcopy(spec["initialparameterization"])
    if representation == "tuple":

        def tuples(value):
            return (
                tuple(tuples(item) for item in value)
                if isinstance(value, list)
                else value
            )

        spec["initialparameterization"] = {
            name: tuples(value) for name, value in expected.items()
        }
    elif representation == "ndarray":
        spec["initialparameterization"] = {
            name: np.asarray(value) for name, value in expected.items()
        }
    spec[parameter_key] = spec.pop("initialparameterization")
    before = pickle.dumps(spec, protocol=5)
    path = tmp_path / mode / "authored.py"
    ok, message, warnings = render_gnn_to_pymdp(spec, path, {"mode": mode})
    assert ok and str(path) in message and warnings == []
    py_compile.compile(str(path), cfile=str(tmp_path / "compiled.pyc"), doraise=True)
    literals = emitted_main_literals(path)
    for name, values in expected.items():
        np.testing.assert_array_equal(literals[f"{name}_data"], values)
    if mode == "pipeline":
        assert (
            "Hidden States: 2" in path.read_text()
            and "Observations:  3" in path.read_text()
        )
        assert literals["gnn_spec"][parameter_key] == expected
        assert literals["gnn_spec"]["model_parameters"]["num_timesteps"] == 7
    else:
        assert literals["num_actions_default"] == 4 and literals["num_timesteps"] == 7
    assert pickle.dumps(spec, protocol=5) == before


def test_pipeline_saved_spec_preserves_literal_words_and_typed_metadata(tmp_path):
    spec = authored_model()
    spec["consumer_metadata"] = {
        "note": 'true false null are authored words; "quoted" \\ path\nΩ',
        "enabled": True,
        "disabled": False,
        "missing": None,
        "path": tmp_path / "source Ω.md",
        "count": np.int64(17),
    }
    before = pickle.dumps(spec, protocol=5)
    path = tmp_path / "metadata.py"
    assert render_gnn_to_pymdp(spec, path)[0]
    py_compile.compile(str(path), cfile=str(tmp_path / "metadata.pyc"), doraise=True)
    assert emitted_main_literals(path)["gnn_spec"]["consumer_metadata"] == {
        "note": 'true false null are authored words; "quoted" \\ path\nΩ',
        "enabled": True,
        "disabled": False,
        "missing": None,
        "path": str(tmp_path / "source Ω.md"),
        "count": 17,
    }
    assert pickle.dumps(spec, protocol=5) == before


@pytest.mark.parametrize("mode", ["pipeline", "standalone"])
@pytest.mark.parametrize("missing", ["A", "B", "C", "D"])
def test_missing_required_pymdp_data_refuses_before_any_artifact_write(
    tmp_path, mode, missing
):
    spec = authored_model()
    del spec["initialparameterization"][missing]
    before = pickle.dumps(spec, protocol=5)
    sentinel = tmp_path / "existing.py"
    sentinel.write_bytes(b"existing consumer artifact\n")
    for path in [sentinel, tmp_path / "absent" / "new.py"]:
        ok, message, warnings = render_gnn_to_pymdp(spec, path, {"mode": mode})
        assert not ok and "Missing required PyMDP" in message and missing in message
        assert warnings == []
    assert sentinel.read_bytes() == b"existing consumer artifact\n"
    assert not (tmp_path / "absent").exists()
    assert pickle.dumps(spec, protocol=5) == before


@pytest.mark.parametrize("transition", ["B_t0", "B_regime_day"])
def test_nonstationary_pipeline_retains_data_and_standalone_refuses(
    tmp_path, transition
):
    spec = authored_model()
    spec["initialparameterization"][transition] = copy.deepcopy(
        spec["initialparameterization"]["B"]
    )
    before = pickle.dumps(spec, protocol=5)
    path = tmp_path / "pipeline.py"
    assert render_gnn_to_pymdp(spec, path, {"mode": "pipeline"})[0]
    assert (
        emitted_main_literals(path)["gnn_spec"]["initialparameterization"][transition]
        == spec["initialparameterization"][transition]
    )
    sentinel = tmp_path / "standalone.py"
    sentinel.write_bytes(b"retain previous program\n")
    for destination in [sentinel, tmp_path / "absent" / "standalone.py"]:
        ok, message, _ = render_gnn_to_pymdp(spec, destination, {"mode": "standalone"})
        assert not ok and "unsupported-nonstationary" in message
    assert (
        sentinel.read_bytes() == b"retain previous program\n"
        and not (tmp_path / "absent").exists()
    )
    assert pickle.dumps(spec, protocol=5) == before


@pytest.mark.parametrize("mode", ["pipeline", "standalone"])
def test_explicit_unknown_execution_contract_refuses_without_replacing_program(
    tmp_path, mode
):
    spec = authored_model()
    spec["model_parameters"]["execution_contract"] = "unknown_scientific_v99"
    before = pickle.dumps(spec, protocol=5)
    path = tmp_path / "existing.py"
    path.write_bytes(b"existing scientific contract\n")
    ok, message, warnings = render_gnn_to_pymdp(spec, path, {"mode": mode})
    assert not ok and "unsupported-execution-contract" in message
    assert "unknown_scientific_v99" in message and warnings == []
    assert path.read_bytes() == b"existing scientific contract\n"
    assert pickle.dumps(spec, protocol=5) == before


@pytest.mark.parametrize("mode", ["pipeline", "standalone"])
def test_saved_pymdp_program_preserves_optional_policy_prior_absence(tmp_path, mode):
    spec = authored_model()
    del spec["initialparameterization"]["E"]
    before = pickle.dumps(spec, protocol=5)
    path = tmp_path / "without_policy_prior.py"
    assert render_gnn_to_pymdp(spec, path, {"mode": mode})[0]
    py_compile.compile(str(path), cfile=str(tmp_path / "compiled.pyc"), doraise=True)
    literals = emitted_main_literals(path)
    assert literals["E_data"] is None
    assert literals["D_data"] == [0.8, 0.2]
    if mode == "pipeline":
        assert "E" not in literals["gnn_spec"]["initialparameterization"]
    assert pickle.dumps(spec, protocol=5) == before
