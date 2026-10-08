"""Exported legacy DisCoPy generator metadata; no analysis or model main runs."""

import ast
import json
import os
import pickle
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

from gnn.render import generate_discopy_code


@pytest.mark.parametrize("field", ["model_name", "source_file"])
@pytest.mark.parametrize(
    "text,class_fragment",
    [
        ("O'Brien observations", "OBrienObservations"),
        ('Double "quoted" Ω', "DoubleQuotedΩ"),
        (r"Backslash \ sensor", "BackslashSensor"),
        ("Newline\nsensor", "NewlineSensor"),
        ('Triple """ delimiter', "TripleDelimiter"),
        ("Brace {not_python} sensor", "BraceNot_pythonSensor"),
    ],
)
def test_legacy_discopy_saved_source_keeps_literal_metadata_and_class_identity(
    tmp_path, field, text, class_fragment
):
    model = {"model_name": "Original model", "source_file": "source Ω.md"}
    model[field] = text
    before = pickle.dumps(model, protocol=5)
    path = tmp_path / "legacy.py"
    source = generate_discopy_code(model, path)
    assert path.read_text() == source
    py_compile.compile(str(path), cfile=str(tmp_path / "compiled.pyc"), doraise=True)
    tree = ast.parse(source)
    doc = ast.get_docstring(tree, clean=False)
    assert f"Enhanced DisCoPy categorical analysis for {model['model_name']}\n" in doc
    assert f"Generated from GNN specification: {model['source_file']}\n" in doc
    analyzer = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    expected_fragment = class_fragment if field == "model_name" else "OriginalModel"
    assert analyzer.name == f"Enhanced{expected_fragment}CategoricalAnalyzer"
    constructor = next(
        node
        for node in analyzer.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assignments = {
        node.targets[0].attr: ast.literal_eval(node.value)
        for node in constructor.body
        if isinstance(node, ast.Assign)
    }
    assert assignments == {
        "model_name": model["model_name"],
        "gnn_source": model["source_file"],
        "analysis_history": [],
        "performance_metrics": {},
    }
    assert pickle.dumps(model, protocol=5) == before


def test_legacy_discopy_default_name_source_and_identifier_are_unchanged(tmp_path):
    path = tmp_path / "default.py"
    source = generate_discopy_code({}, path)
    py_compile.compile(str(path), cfile=str(tmp_path / "compiled.pyc"), doraise=True)
    tree = ast.parse(source)
    assert "Enhanced DisCoPy categorical analysis for GNN Model\n" in ast.get_docstring(
        tree, clean=False
    )
    assert "Generated from GNN specification: unknown.md\n" in ast.get_docstring(
        tree, clean=False
    )
    assert (
        next(node.name for node in tree.body if isinstance(node, ast.ClassDef))
        == "EnhancedGnnModelCategoricalAnalyzer"
    )


def test_legacy_discopy_native_metadata_constructor_does_not_run_analysis(tmp_path):
    import gnn

    model = {
        "model_name": 'Quoted """ Ω \\ name\n{literal}',
        "source_file": 'Quoted "source" Ω \\ path\nsource.md',
    }
    before = pickle.dumps(model, protocol=5)
    path = tmp_path / "native.py"
    generate_discopy_code(model, path)
    command = (
        "import json,runpy,sys; from pathlib import Path; import gnn; "
        "namespace=runpy.run_path(sys.argv[1],run_name='legacy_metadata_only'); "
        "analyzer=namespace['EnhancedQuotedΩNameLiteralCategoricalAnalyzer'](); "
        "assert analyzer.analysis_history==[] and analyzer.performance_metrics=={}; "
        "print('NATIVE_METADATA_JSON='+json.dumps({'model_name':analyzer.model_name,'source_file':analyzer.gnn_source,'gnn_origin':str(Path(gnn.__file__).resolve())}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", command, str(path)],
        cwd=tmp_path,
        env=os.environ.copy(),
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
    assert json.loads(row) == {**model, "gnn_origin": str(Path(gnn.__file__).resolve())}
    assert not (tmp_path / "output").exists()
    assert pickle.dumps(model, protocol=5) == before
