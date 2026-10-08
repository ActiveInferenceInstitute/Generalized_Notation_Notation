"""Public PyMDP source generation preserves data; these tests never run a rollout."""

import ast
import copy
import json
import os
import pickle
import py_compile
import subprocess
import sys
from pathlib import Path

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
        "nonfinite_words": "nan, inf, -inf and Infinity are authored text",
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
        "nonfinite_words": "nan, inf, -inf and Infinity are authored text",
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


def saved_main(tree):
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )


def default_output_expression(tree, mode):
    name = "output_dir" if mode == "pipeline" else "out_dir"
    return next(
        node.value
        for node in saved_main(tree).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    )


@pytest.mark.parametrize("mode", ["pipeline", "standalone"])
@pytest.mark.parametrize("field", ["model_name", "annotation"])
@pytest.mark.parametrize(
    "text",
    [
        "O'Brien observations",
        'Double "quoted" Ω',
        r"Backslash \ sensor",
        "Newline\nsensor",
        'Triple """ delimiter',
        "Brace {not_python} sensor",
    ],
)
def test_saved_pymdp_authored_display_text_is_literal_data_in_every_public_slot(
    tmp_path, monkeypatch, mode, field, text
):
    spec = authored_model()
    spec[field] = text
    before = pickle.dumps(spec, protocol=5)
    path = tmp_path / "consumer.py"
    assert render_gnn_to_pymdp(spec, path, {"mode": mode})[0]
    py_compile.compile(str(path), cfile=str(tmp_path / "compiled.pyc"), doraise=True)
    tree = ast.parse(path.read_text())
    doc = ast.get_docstring(tree, clean=False)
    assert doc is not None
    assert f"Model:        {spec['model_name']}\n" in doc
    assert f"Description:  {spec['annotation']}\n" in doc
    if mode == "pipeline":
        embedded = emitted_main_literals(path)["gnn_spec"]
        assert (
            embedded["model_name"] == spec["model_name"]
            and embedded["annotation"] == spec["annotation"]
        )
        messages = [
            ast.literal_eval(node.value.args[0])
            for node in saved_main(tree).body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "info"
        ]
        assert messages[0] == f"Running pymdp 1.0.0 rollout for {spec['model_name']}"
    else:
        results = next(
            node.value
            for node in saved_main(tree).body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "results"
        )
        names = {
            ast.literal_eval(key): value
            for key, value in zip(results.keys, results.values, strict=True)
        }
        assert ast.literal_eval(names["model_name"]) == spec["model_name"]
    # Execute only the saved Path expression against actual Path/os. No model main.
    monkeypatch.delenv("PYMDP_OUTPUT_DIR", raising=False)
    expression = ast.Expression(default_output_expression(tree, mode))
    generated_path = eval(
        compile(expression, str(path), "eval"), {"Path": Path, "os": os}
    )
    assert generated_path == Path(f"output/pymdp_simulations/{spec['model_name']}")
    assert pickle.dumps(spec, protocol=5) == before
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("mode", ["pipeline", "standalone"])
def test_saved_pymdp_default_name_and_existing_output_override_contract(
    tmp_path, monkeypatch, mode
):
    spec = authored_model()
    del spec["model_name"]
    before = pickle.dumps(spec, protocol=5)
    path = tmp_path / "default.py"
    assert render_gnn_to_pymdp(spec, path, {"mode": mode})[0]
    tree = ast.parse(path.read_text())
    assert "Model:        GNN_Model\n" in ast.get_docstring(tree, clean=False)
    expression = ast.Expression(default_output_expression(tree, mode))
    monkeypatch.delenv("PYMDP_OUTPUT_DIR", raising=False)
    assert eval(
        compile(expression, str(path), "eval"), {"Path": Path, "os": os}
    ) == Path("output/pymdp_simulations/GNN_Model")
    if mode == "pipeline":
        override = tmp_path / "existing override Ω"
        monkeypatch.setenv("PYMDP_OUTPUT_DIR", str(override))
        assert (
            eval(compile(expression, str(path), "eval"), {"Path": Path, "os": os})
            == override
        )
        assert not override.exists()
    assert pickle.dumps(spec, protocol=5) == before


@pytest.mark.parametrize("mode", ["pipeline", "standalone"])
def test_saved_pymdp_native_import_retains_safe_metadata_without_invoking_main(
    tmp_path, mode
):
    import gnn

    spec = authored_model()
    spec["model_name"] = 'Quoted """ Ω \\ name\n{not_a_call}'
    spec["annotation"] = 'Authored "quoted" true false null\nΩ'
    before = pickle.dumps(spec, protocol=5)
    path = tmp_path / "native.py"
    assert render_gnn_to_pymdp(spec, path, {"mode": mode})[0]
    checkout = Path(gnn.__file__).resolve().parents[2]
    env = os.environ.copy()
    env["GNN_PROJECT_ROOT"] = str(checkout)
    command = (
        "import json,runpy,sys; from pathlib import Path; import gnn; "
        "namespace=runpy.run_path(sys.argv[1],run_name='native_metadata_only'); "
        "assert callable(namespace['main']); "
        "assert hasattr(namespace['Agent'],'update_empirical_prior'); "
        "print('NATIVE_METADATA_JSON='+json.dumps({'doc':namespace['__doc__'],'gnn_origin':str(Path(gnn.__file__).resolve())}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", command, str(path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    row = next(
        line.removeprefix("NATIVE_METADATA_JSON=")
        for line in result.stdout.splitlines()
        if line.startswith("NATIVE_METADATA_JSON=")
    )
    metadata = json.loads(row)
    assert metadata["gnn_origin"] == str(Path(gnn.__file__).resolve())
    assert f"Model:        {spec['model_name']}\n" in metadata["doc"]
    assert f"Description:  {spec['annotation']}\n" in metadata["doc"]
    assert not (tmp_path / "output").exists()
    assert pickle.dumps(spec, protocol=5) == before


@pytest.mark.parametrize("mode", ["pipeline", "standalone"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_embedded_metadata_refuses_before_replacing_saved_program(
    tmp_path, mode, nonfinite
):
    spec = authored_model()
    spec["consumer_metadata"] = {"nested": [{"measurement": nonfinite}]}
    before = pickle.dumps(spec, protocol=5)
    previous = tmp_path / "previous.py"
    previous.write_bytes(b"caller-owned previous source\n")
    absent = tmp_path / "absent" / "new.py"
    for destination in (previous, absent):
        ok, message, warnings = render_gnn_to_pymdp(spec, destination, {"mode": mode})
        assert not ok, message
        assert "Out of range float values are not JSON compliant" in message
        assert warnings == []
    assert previous.read_bytes() == b"caller-owned previous source\n"
    assert not absent.parent.exists()
    assert pickle.dumps(spec, protocol=5) == before
