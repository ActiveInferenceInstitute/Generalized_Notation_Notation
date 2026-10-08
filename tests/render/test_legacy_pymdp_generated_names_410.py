"""Legacy exported generator source/metadata fidelity, without simulation execution."""

import ast
import pickle
import py_compile

import pytest

from gnn.render import generate_pymdp_code


TEXT_CASES = [
    ("O'Brien observations", "o_brien_observations"),
    ('Double "quoted" Ω', "double_quoted_ω"),
    (r"Backslash \ sensor", "backslash_sensor"),
    ("Newline\nsensor", "newline_sensor"),
    ('Triple """ delimiter', "triple_delimiter"),
    ("Brace {not_python} sensor", "brace_not_python_sensor"),
]


@pytest.mark.parametrize("field", ["model_name", "source_file"])
@pytest.mark.parametrize("text,slug", TEXT_CASES)
def test_legacy_pymdp_saved_metadata_and_report_text_are_literal_data(
    tmp_path, field, text, slug
):
    model = {"model_name": "Original model", "source_file": "source Ω.md"}
    model[field] = text
    before = pickle.dumps(model, protocol=5)
    path = tmp_path / "legacy.py"
    source = generate_pymdp_code(model, path)
    assert path.read_text() == source
    py_compile.compile(str(path), cfile=str(tmp_path / "compiled.pyc"), doraise=True)
    tree = ast.parse(source)
    doc = ast.get_docstring(tree, clean=False)
    assert f"Model: {model['model_name']}\n" in doc
    assert f"Generated from GNN specification: {model['source_file']}\n" in doc
    agent = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    simulate = next(
        node
        for node in agent.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_simulation"
    )
    results = next(
        node.value
        for node in simulate.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "results"
    )
    metadata = next(
        value
        for key, value in zip(results.keys, results.values, strict=True)
        if ast.literal_eval(key) == "metadata"
    )
    literal_fields = {
        ast.literal_eval(key): value
        for key, value in zip(metadata.keys, metadata.values, strict=True)
    }
    assert ast.literal_eval(literal_fields["model_name"]) == model["model_name"]
    assert ast.literal_eval(literal_fields["gnn_source"]) == model["source_file"]
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    literal_prints = [
        node.args[0].value
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    assert f"📁 Model: {model['model_name']}" in literal_prints
    assert f"📄 Source: {model['source_file']}" in literal_prints
    report_lines = [
        node.args[0].value
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    assert f"**Model:** {model['model_name']}\n" in report_lines
    # Sanitized artifact prefixes remain distinct from authored display strings.
    expected_slug = slug if field == "model_name" else "original_model"
    constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert f"pymdp_{expected_slug}" in constants
    assert f"pymdp_{expected_slug}_basic_export.json" in constants
    assert pickle.dumps(model, protocol=5) == before


def test_legacy_pymdp_missing_name_and_source_keep_original_defaults(tmp_path):
    path = tmp_path / "default.py"
    source = generate_pymdp_code({}, path)
    py_compile.compile(str(path), cfile=str(tmp_path / "compiled.pyc"), doraise=True)
    doc = ast.get_docstring(ast.parse(source), clean=False)
    assert "Model: GNN Model\n" in doc
    assert "Generated from GNN specification: unknown.md\n" in doc
    assert "pymdp_gnn_model_basic_export.json" in source


def test_legacy_pymdp_invalid_matrix_refuses_before_replacing_existing_program(
    tmp_path,
):
    path = tmp_path / "existing.py"
    path.write_bytes(b"retained legacy program\n")
    model = {"model_name": "Original model", "state_space": {"A": "not_a_matrix"}}
    before = pickle.dumps(model, protocol=5)
    with pytest.raises(ValueError, match="refusing to interpolate raw text"):
        generate_pymdp_code(model, path)
    assert path.read_bytes() == b"retained legacy program\n"
    assert pickle.dumps(model, protocol=5) == before
