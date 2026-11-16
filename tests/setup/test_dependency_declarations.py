"""Dependency hygiene contract witnesses, independent of installed packages."""

import ast
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_dep_hygiene.py"
spec = importlib.util.spec_from_file_location("dep_hygiene", SCRIPT)
assert spec and spec.loader
hygiene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hygiene)


def test_all_union_retains_versions_and_markers():
    extras = {
        "a": ["a>=1; python_version >= '3.12'"],
        "dev": ["dev"],
        "all": ['a >= 1; python_version >= "3.12"'],
    }
    assert hygiene.all_extra_errors(extras) == []
    extras["all"] = ["a>=1"]
    assert len(hygiene.all_extra_errors(extras)) == 2


def test_runtime_import_guards_and_type_only():
    tree = ast.parse("""import eager
try:
    import optional
except ImportError:
    pass
if TYPE_CHECKING:
    import typed
if typing.TYPE_CHECKING:
    import typed2
def work():
    import deferred
try:
    import unguarded
except ValueError:
    pass
""")
    visitor = hygiene.RuntimeImports()
    visitor.visit(tree)
    assert [(n, guarded) for n, _, guarded in visitor.imports] == [
        ("eager", False),
        ("optional", True),
        ("deferred", True),
        ("unguarded", False),
    ]


def test_runtime_declarations_exclude_dev(tmp_path):
    (tmp_path / "sample.py").write_text(
        "import yaml\ntry:\n import fastapi\nexcept ImportError:\n pass\n"
    )
    project = {
        "dependencies": ["pyyaml>=6"],
        "optional-dependencies": {"api": ["fastapi"], "dev": ["devonly"]},
    }
    assert hygiene.runtime_import_errors(project, tmp_path) == []
    (tmp_path / "sample.py").write_text("import fastapi\nimport devonly\n")
    assert len(hygiene.runtime_import_errors(project, tmp_path)) == 2
