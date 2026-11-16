"""Source admission is narrower than installed-package Python support."""

import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_installed_python_range_remains_qualified():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["requires-python"] == ">=3.11,<3.15"


@pytest.mark.needs_python315
def test_python315_source_syntax_and_stdlib_imports():
    python = shutil.which("python3.15")
    assert python is not None
    probe = r"""
import ast
import pathlib
import runpy
import sys
assert sys.version_info[:2] == (3, 15)
root = pathlib.Path(sys.argv[1])
paths = sorted((root / "src/gnn").rglob("*.py"))
assert paths
for path in paths:
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
    compile(source, str(path), "exec")
stdlib = set(sys.stdlib_module_names) | {"__main__"}
before = set(sys.modules)
registry = runpy.run_path(str(root / "src/gnn/pipeline/step_registry.py"))
assert len(registry["STEPS"]) == 25
assert set(registry["discover_steps"]()) == set(range(25))
types = runpy.run_path(str(root / "src/gnn/types/definitions.py"))
assert types["ValidationLevel"].STRICT.value == "strict"
assert not {name.split(".")[0] for name in set(sys.modules) - before} - stdlib
print(f"Python 3.15: {len(paths)} source files compiled; two stdlib modules admitted")
"""
    result = subprocess.run(
        [python, "-I", "-B", "-S", "-c", probe, str(ROOT)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
