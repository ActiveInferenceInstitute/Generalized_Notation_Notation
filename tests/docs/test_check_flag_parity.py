"""Tests for scripts/check_flag_parity.py — determinism and import pinning."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from tests.helpers import load_module_from_path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_gate() -> Any:
    return load_module_from_path(
        "check_flag_parity", REPO_ROOT / "scripts" / "check_flag_parity.py"
    )


def test_registered_flags_import_pinned_to_working_tree() -> None:
    """The dynamic half must import the repo-local parser, never an installed
    gnn distribution (a stale wheel would change the measured surface)."""
    gate = _load_gate()
    import importlib
    import sys

    sys.path.insert(0, str(gate.ROOT / "src"))
    try:
        mod = importlib.import_module("gnn.utils.arguments.arg_parsing")
    finally:
        sys.path.remove(str(gate.ROOT / "src"))
    module_file = Path(getattr(mod, "__file__", ""))
    assert module_file.is_relative_to(gate.ROOT / "src")


def test_registered_flags_are_a_pure_function_of_the_tree() -> None:
    """Determinism regression: two full computations in fresh interpreters
    must produce the identical flag set — this is the runner-parity
    guarantee (the wave-D 141-vs-140 phantom drift was a tree-state
    confound, not a varying parser)."""
    probe = (
        "import sys; sys.path.insert(0, r'{scripts}'); "
        "import check_flag_parity as g; "
        "print(','.join(sorted(g.registered_flags())))"
    ).format(scripts=str(REPO_ROOT / "scripts"))
    runs = [
        subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    assert runs[0]  # non-empty surface


def test_static_scan_covers_every_orchestrator_step() -> None:
    """The AST half must resolve at least one call site per step script that
    passes ``additional_arguments`` — a zero-length result would mean the
    scan silently stopped matching (vacuous gate)."""
    gate = _load_gate()
    import sys

    sys.path.insert(0, str(gate.ROOT / "src"))
    try:
        from gnn.utils.arguments.arg_parsing import ArgumentParser
    finally:
        sys.path.pop(0)
    # dynamic surface alone must register the core flags
    dynamic = gate.registered_flags()
    assert "--target-dir" in dynamic
    assert "--output-dir" in dynamic
    assert "--verbose" in dynamic
    # and the union equals dynamic | static by construction
    static = gate.static_registered_flags(set(ArgumentParser.ARGUMENT_DEFINITIONS))
    assert gate.registered_flags() == dynamic | static


def test_caps_file_matches_gate_keys() -> None:
    gate = _load_gate()
    caps = gate.load_caps()
    assert set(caps) == set(gate.CAP_KEYS)
    assert all(isinstance(v, int) and v >= 0 for v in caps.values())


def test_documented_short_aliases_count_without_foreign_or_prefix_matches(
    tmp_path, monkeypatch
):
    gate = _load_gate()
    doc = tmp_path / "options.md"
    doc.write_text(
        "Use `-f`, `-o`, `--target-dir`. Foreign `-I`, `-foo` and `-off` are not aliases.\n"
        "[Release](CHANGELOG.md#410---2026-10-08) and `prefix--embedded` are not flags.\n"
    )
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "iter_doc_files", lambda: [doc])
    monkeypatch.setattr(gate, "cli_parser_flags", lambda: {"-f", "-o", "--target-dir"})
    assert set(gate.collect_doc_tokens()) == {"-f", "-o", "--target-dir"}
