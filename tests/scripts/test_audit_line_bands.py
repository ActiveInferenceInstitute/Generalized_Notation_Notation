#!/usr/bin/env python3
"""Tests for scripts/audit_line_bands.py (SCOPE M-13 band audit).

Fixtures are tmp_path trees exercising the band contract: the reporting
band (>1200 lines or >40960 bytes) selects table rows, the hard band
(>1200 lines or >49152 bytes) drives the exit code, --soft always exits 0,
and --json output is machine-parseable with the exact row set. Fixture
trees use the repo's ``<base>/src/gnn`` layout with ``base=tmp_path`` so
reported paths read ``src/gnn/...`` like the real scan.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts" / "audit_line_bands.py"
_spec = importlib.util.spec_from_file_location("audit_line_bands", _SCRIPT)
assert _spec is not None and _spec.loader is not None
audit_line_bands = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("audit_line_bands", audit_line_bands)
_spec.loader.exec_module(audit_line_bands)


def _make_py(path: Path, lines: int) -> Path:
    """Write a Python file with exactly `lines` physical lines."""
    body = b"\n".join([b"# pad"] * lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body + b"\n")
    return path


def _make_bytes(path: Path, size: int) -> Path:
    """Write a Python file padded to exactly `size` bytes in ~64-byte
    lines, keeping line counts under the 1200-line band so byte-band
    fixtures isolate the byte condition."""
    unit = b"# " + b"p" * 61 + b"\n"  # exactly 64 bytes per line
    repeats, remainder = divmod(size, len(unit))
    body = unit * repeats + b"#" * remainder
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    assert path.stat().st_size == size, f"padded to {path.stat().st_size}, want {size}"
    return path


def _run(argv: list[str], tmp_path: Path, **kwargs: Any) -> int:
    return audit_line_bands.main(
        argv, scan_root=tmp_path / "src" / "gnn", base=tmp_path, **kwargs
    )


def test_small_tree_soft_mode_passes(tmp_path: Path) -> None:
    _make_py(tmp_path / "src" / "gnn" / "small.py", 10)
    assert _run(["--soft"], tmp_path) == 0


def test_small_tree_default_mode_exits_zero(tmp_path: Path) -> None:
    _make_py(tmp_path / "src" / "gnn" / "small.py", 10)
    assert _run([], tmp_path) == 0


def test_oversized_lines_fail_hard_but_pass_soft(tmp_path: Path, capsys) -> None:
    _make_py(tmp_path / "src" / "gnn" / "big.py", 1201)
    _make_py(tmp_path / "src" / "gnn" / "ok.py", 500)
    assert _run([], tmp_path) == 1
    text = capsys.readouterr().out
    assert "src/gnn/big.py" in text
    assert "ok.py" not in text
    assert _run(["--soft"], tmp_path) == 0


def test_oversized_bytes_fail_hard(tmp_path: Path) -> None:
    _make_bytes(tmp_path / "src" / "gnn" / "wide.py", audit_line_bands.HARD_BYTES + 1)
    assert _run([], tmp_path) == 1


def test_band_boundaries_are_exact(tmp_path: Path) -> None:
    # Exactly at each boundary: not reported, no failure.
    _make_py(tmp_path / "src" / "gnn" / "at_lines.py", audit_line_bands.REPORT_LINES)
    _make_bytes(
        tmp_path / "src" / "gnn" / "at_report_bytes.py", audit_line_bands.REPORT_BYTES
    )
    # 49152 B sits above the 40 KiB *reporting* band but exactly on the
    # hard-byte boundary: table row, not a hard failure.
    _make_bytes(
        tmp_path / "src" / "gnn" / "at_hard_bytes.py", audit_line_bands.HARD_BYTES
    )
    rows = audit_line_bands.audit(tmp_path / "src" / "gnn", base=tmp_path)
    assert [r["path"] for r in rows] == ["src/gnn/at_hard_bytes.py"]
    assert not audit_line_bands.over_hard_band(rows[0])
    assert _run(["--soft"], tmp_path) == 0
    assert _run([], tmp_path) == 0
    # One past the line boundary: reported and, being hard-band, failing.
    _make_py(
        tmp_path / "src" / "gnn" / "over_lines.py", audit_line_bands.REPORT_LINES + 1
    )
    rows = audit_line_bands.audit(tmp_path / "src" / "gnn", base=tmp_path)
    assert [r["path"] for r in rows] == [
        "src/gnn/over_lines.py",
        "src/gnn/at_hard_bytes.py",  # still a table row (above report band)
    ]
    assert _run([], tmp_path) == 1


def test_bytes_band_file_is_report_only(tmp_path: Path) -> None:
    """40-48 KiB with <=1200 lines: in the table, not a hard failure."""
    size = audit_line_bands.REPORT_BYTES + 1  # >40KiB, <48KiB
    _make_bytes(tmp_path / "src" / "gnn" / "midband.py", size)
    rows = audit_line_bands.audit(tmp_path / "src" / "gnn", base=tmp_path)
    assert len(rows) == 1
    assert rows[0]["lines"] <= audit_line_bands.REPORT_LINES
    assert _run([], tmp_path) == 0


def test_json_mode_parses_and_counts_hard(tmp_path: Path, capsys) -> None:
    _make_py(tmp_path / "src" / "gnn" / "big.py", 1500)
    _make_bytes(tmp_path / "src" / "gnn" / "wide.py", audit_line_bands.HARD_BYTES + 1)
    assert _run(["--json"], tmp_path) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["over_hard_count"] == 2
    assert payload["files"][0]["path"] == "src/gnn/big.py"  # lines desc
    assert payload["files"][0]["lines"] == 1500
    assert all(f["over_hard"] for f in payload["files"])
    assert payload["band"]["hard_lines"] == audit_line_bands.HARD_LINES


def test_json_soft_mode_exits_zero(tmp_path: Path, capsys) -> None:
    _make_py(tmp_path / "src" / "gnn" / "big.py", 1500)
    assert _run(["--json", "--soft"], tmp_path) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["over_hard_count"] == 1


def test_output_deterministic_across_calls(tmp_path: Path) -> None:
    _make_py(tmp_path / "src" / "gnn" / "b_two.py", 1300)
    _make_py(tmp_path / "src" / "gnn" / "a_three.py", 1300)
    _make_py(tmp_path / "src" / "gnn" / "c_one.py", 1400)
    root = tmp_path / "src" / "gnn"
    runs = [
        json.loads(
            audit_line_bands.render_json(audit_line_bands.audit(root, base=tmp_path))
        )
        for _ in range(3)
    ]
    assert all(run == runs[0] for run in runs[1:])
    assert [f["path"] for f in runs[0]["files"]] == [
        "src/gnn/c_one.py",
        "src/gnn/a_three.py",
        "src/gnn/b_two.py",
    ]


def test_real_repo_scan_is_deterministic_and_matches_band() -> None:
    """Live-tree audit: rows all over-band, repeat call identical, paths
    repo-root relative (src/gnn/...)."""
    rows_a = audit_line_bands.audit(audit_line_bands.SCAN_ROOT)
    rows_b = audit_line_bands.audit(audit_line_bands.SCAN_ROOT)
    assert rows_a == rows_b
    assert all(row["path"].startswith("src/gnn/") for row in rows_a)
    for row in rows_a:
        assert row["lines"] > audit_line_bands.REPORT_LINES or (
            row["bytes"] > audit_line_bands.REPORT_BYTES
        )
