"""Generated public DisCoPy programs build native diagrams and export real JSON."""

import ast
import importlib.util
import json
import pickle
import py_compile
import subprocess
import sys

import pytest
from discopy.monoidal import Box, Ty

from gnn.render.discopy import render_gnn_to_discopy


def authored_model(name="Categorical observations Ω"):
    return {
        "model_name": name,
        "variables": [
            {"name": "A", "dimensions": [3, 2]},
            {"name": "B", "dimensions": [2, 2, 4]},
        ],
        "model_parameters": {"num_hidden_states": 9, "num_obs": 8, "num_actions": 7},
        "initial_parameterization": {"A": [[0.7, 0.1], [0.2, 0.3], [0.1, 0.6]]},
        "connections": [{"source": "A", "target": "B"}],
    }


def load_saved_module(path):
    descriptor = importlib.util.spec_from_file_location(
        "generated_native_discopy_consumer", path
    )
    assert descriptor is not None and descriptor.loader is not None
    module = importlib.util.module_from_spec(descriptor)
    descriptor.loader.exec_module(module)
    return module


@pytest.mark.parametrize("permutation", [None, [2, 0, 1]])
@pytest.mark.parametrize(
    "parameter_key", ["initial_parameterization", "initialparameterization"]
)
def test_generated_discopy_native_types_composition_and_json_preserve_identity(
    tmp_path, permutation, parameter_key
):
    spec = authored_model()
    spec[parameter_key] = spec.pop("initial_parameterization")
    before = pickle.dumps(spec, protocol=5)
    options = {"matrix_permutations": {"A": permutation}} if permutation else {}
    options_before = pickle.dumps(options, protocol=5)
    path = tmp_path / "generated.py"
    ok, message, _ = render_gnn_to_discopy(spec, path, options)
    assert ok and str(path) in message
    py_compile.compile(str(path), cfile=str(tmp_path / "compiled.pyc"), doraise=True)
    module = load_saved_module(path)
    state, observation, action, probability = module.define_types()
    assert (state, observation, action, probability) == (
        Ty("S"),
        Ty("O"),
        Ty("A"),
        Ty("P"),
    )
    components = module.define_model_components(state, observation, action, probability)
    expected = {
        "A_matrix": ("A", Ty("S"), Ty("O", "P")),
        "B_matrix": ("B", Ty("S", "A"), Ty("S", "P")),
        "C_vector": ("C", Ty(), Ty("O", "P")),
        "D_vector": ("D", Ty(), Ty("S", "P")),
        "E_vector": ("E", Ty(), Ty("A", "P")),
        "state_inference": ("StateInf", Ty("O"), Ty("S", "P")),
        "policy_inference": ("PolicyInf", Ty("S", "P"), Ty("A", "P")),
        "action_selection": ("ActionSel", Ty("A", "P"), Ty("A")),
    }
    assert set(components) == set(expected)
    for name, (label, domain, codomain) in expected.items():
        component = components[name]
        assert isinstance(component, Box)
        assert (component.name, component.dom, component.cod) == (
            label,
            domain,
            codomain,
        )
    circuit = module.create_active_inference_circuit(
        state, observation, action, probability, components
    )
    loop = circuit["perception_action_loop"]
    assert (loop.dom, loop.cod) == (Ty("O"), Ty("A"))
    assert [box.name for box in loop.boxes] == ["StateInf", "PolicyInf", "ActionSel"]
    analysis = module.analyze_circuit_structure(circuit)
    assert analysis == {
        "num_components": 8,
        "loop_domain": "O",
        "loop_codomain": "A",
        "model_domain": "O",
        "model_codomain": "A",
    }
    output = tmp_path / "actual_json"
    module.export_circuit_data(circuit, analysis, str(output))
    assert json.loads((output / "circuit_analysis.json").read_text()) == analysis
    info = json.loads((output / "circuit_info.json").read_text())
    assert info["model_name"] == "Categorical observations Ω"
    assert info["parameters"] == {
        "num_states": 2,
        "num_observations": 3,
        "num_actions": 4,
    }
    assert info["components"] == list(expected) and info["analysis"] == analysis
    expected_metadata = (
        {"A": {"axis": "rows", "shape": [3, 2], "permutation": [2, 0, 1]}}
        if permutation
        else {}
    )
    assert info["matrix_permutation_metadata"] == expected_metadata
    assert info["matrix_permutation_applied_to_diagram"] is False
    assert pickle.dumps(spec, protocol=5) == before
    assert pickle.dumps(options, protocol=5) == options_before


@pytest.mark.parametrize(
    "name",
    [
        "Unicode observations Ω",
        "O'Brien observations",
        'Double "quoted" observations \\ sensor\nnext line Ω',
        'Triple """ quoted observations',
        "Brace {not_a_variable} observations",
    ],
)
def test_generated_discopy_actual_entrypoint_exports_authored_model_name(
    tmp_path, name
):
    spec = authored_model(name)
    before = pickle.dumps(spec, protocol=5)
    path = tmp_path / "consumer.py"
    assert render_gnn_to_discopy(spec, path)[0]
    py_compile.compile(str(path), cfile=str(tmp_path / "compiled.pyc"), doraise=True)
    assert ast.get_docstring(ast.parse(path.read_text()), clean=False) == (
        "\nDisCoPy Categorical Diagram Generation\n"
        f"Generated from GNN Model: {name}\n\n"
        "This script creates categorical diagrams representing the Active Inference model\n"
        "structure using DisCoPy's compositional framework.\n"
    )
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    info = json.loads((tmp_path / "discopy_diagrams" / "circuit_info.json").read_text())
    assert info["model_name"] == name
    assert info["parameters"] == {
        "num_states": 2,
        "num_observations": 3,
        "num_actions": 4,
    }
    assert info["analysis"]["num_components"] == 8
    assert pickle.dumps(spec, protocol=5) == before


@pytest.mark.parametrize(
    "matrix,permutation,reason",
    [
        ("missing", [0], "missing matrix"),
        ("A", [0, 0, 2], "each row index"),
        ("A", [0, 1], "does not match"),
        ("A", [0, 1, 3], "each row index"),
        ("scalar", [0], "scalar matrix"),
    ],
)
def test_discopy_public_permutation_refusal_preserves_existing_artifact(
    tmp_path, matrix, permutation, reason
):
    spec = authored_model()
    spec["initial_parameterization"]["scalar"] = 0.5
    options = {"matrix_permutations": {matrix: permutation}}
    before = pickle.dumps((spec, options), protocol=5)
    sentinel = tmp_path / "existing.py"
    sentinel.write_bytes(b"retained previous categorical diagram\n")
    for path in [sentinel, tmp_path / "absent" / "new.py"]:
        ok, message, warnings = render_gnn_to_discopy(spec, path, options)
        assert not ok and reason in message and warnings == []
    assert sentinel.read_bytes() == b"retained previous categorical diagram\n"
    assert not (tmp_path / "absent").exists()
    assert pickle.dumps((spec, options), protocol=5) == before


def test_discopy_unknown_execution_contract_refuses_before_writes(tmp_path):
    spec = authored_model()
    spec["model_parameters"]["execution_contract"] = "unknown_scientific_v99"
    before = pickle.dumps(spec, protocol=5)
    path = tmp_path / "absent" / "consumer.py"
    ok, message, warnings = render_gnn_to_discopy(spec, path)
    assert not ok and "unsupported-execution-contract" in message
    assert "unknown_scientific_v99" in message and warnings == []
    assert not path.parent.exists()
    assert pickle.dumps(spec, protocol=5) == before
