"""Tests for scripts/check_stdlib_shadow_guard.py — the COMP-010 guard gate.

Fixtures are tmp_path trees exercising the guard contract: a compliant
numbered script passes; a script missing the canonical guard, or importing
before it, fails; allowed pre-guard imports (``sys``, ``__future__``) pass;
and the live tree stays compliant. The gate enforces the accepted resolution
of composability item COMP-010 (docs/standards/composability.md).
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from tests.helpers import load_module_from_path

REPO_ROOT = Path(__file__).resolve().parents[2]

_GUARD = (
    "import sys\n"
    "\n"
    'if globals().get("__package__") is None and not sys.flags.safe_path:\n'
    "    sys.path.pop(0)  # Keep adjacent gnn/types from shadowing stdlib types.\n"
)


def _load_gate() -> ModuleType:
    return load_module_from_path(
        "check_stdlib_shadow_guard",
        REPO_ROOT / "scripts" / "check_stdlib_shadow_guard.py",
    )


def _write_script(root: Path, name: str, body: str) -> Path:
    directory = root / "src" / "gnn"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


def test_compliant_script_passes(tmp_path: Path) -> None:
    gate = _load_gate()
    script = _write_script(tmp_path, "3_gnn.py", _GUARD + "from gnn import x\n")
    assert gate.check_file(script, tmp_path) == []


def test_missing_guard_fails(tmp_path: Path) -> None:
    gate = _load_gate()
    script = _write_script(tmp_path, "3_gnn.py", "import sys\n\nfrom gnn import x\n")
    failures = gate.check_file(script, tmp_path)
    assert len(failures) == 1
    assert "missing the canonical stdlib-shadowing guard" in failures[0]


def test_import_before_guard_fails(tmp_path: Path) -> None:
    gate = _load_gate()
    body = "import sys\n\nfrom typing import cast\n\n" + _GUARD
    script = _write_script(tmp_path, "3_gnn.py", body)
    failures = gate.check_file(script, tmp_path)
    assert len(failures) == 1
    assert "precedes the guard" in failures[0]


def test_allowed_pre_guard_imports_pass(tmp_path: Path) -> None:
    gate = _load_gate()
    body = "from __future__ import annotations\n\n" + _GUARD + "import json\n"
    script = _write_script(tmp_path, "3_gnn.py", body)
    assert gate.check_file(script, tmp_path) == []


def test_nested_module_level_import_before_guard_fails(tmp_path: Path) -> None:
    gate = _load_gate()
    body = (
        "import sys\n"
        "try:\n"
        "    import json\n"
        "except ImportError:\n"
        "    json = None\n\n" + _GUARD
    )
    script = _write_script(tmp_path, "3_gnn.py", body)
    failures = gate.check_file(script, tmp_path)
    assert len(failures) == 1
    assert "precedes the guard" in failures[0]


def test_function_body_import_is_ignored(tmp_path: Path) -> None:
    gate = _load_gate()
    body = "import sys\n\ndef wrapper():\n    import os\n\n" + _GUARD
    script = _write_script(tmp_path, "3_gnn.py", body)
    assert gate.check_file(script, tmp_path) == []


def test_non_canonical_pop_fails(tmp_path: Path) -> None:
    gate = _load_gate()
    body = (
        "import sys\n"
        "\n"
        'if globals().get("__package__") is None and not sys.flags.safe_path:\n'
        "    sys.path.pop()\n"
    )
    script = _write_script(tmp_path, "3_gnn.py", body)
    failures = gate.check_file(script, tmp_path)
    assert len(failures) == 1
    assert "missing the canonical stdlib-shadowing guard" in failures[0]


def test_non_numbered_files_are_ignored(tmp_path: Path) -> None:
    gate = _load_gate()
    _write_script(tmp_path, "main.py", "import os\n")
    _write_script(tmp_path, "3_gnn.py", _GUARD + "import json\n")
    assert gate.collect_failures(tmp_path / "src" / "gnn", tmp_path) == []


def test_empty_scan_is_a_gate_error(tmp_path: Path) -> None:
    gate = _load_gate()
    failures = gate.collect_failures(tmp_path / "src" / "gnn", tmp_path)
    assert len(failures) == 1
    assert "no files match" in failures[0]


def test_main_reports_failures_and_exit_code(tmp_path: Path, capsys) -> None:
    gate = _load_gate()
    _write_script(tmp_path, "3_gnn.py", "import sys\n")
    assert gate.main([], scan_root=tmp_path / "src" / "gnn", base=tmp_path) == 1
    out = capsys.readouterr().out
    assert "3_gnn.py" in out
    assert "failure(s)" in out


def test_live_tree_is_compliant() -> None:
    gate = _load_gate()
    assert gate.main([], scan_root=REPO_ROOT / "src" / "gnn", base=REPO_ROOT) == 0
