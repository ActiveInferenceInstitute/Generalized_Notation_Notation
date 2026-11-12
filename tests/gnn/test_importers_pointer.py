"""Sender-side coverage for the reader's pointer, party and variable surfaces.

``fep_formal`` pins GNN by content hash inside a two-party custody record, and
``active_inference_battleship`` pins a differently named file; both need the
reader to be told which file and which sub-record to read. A consumer whose
manuscript states the pin needs the seven GNN-derived variables without
re-reading the JSON. These tests pin those three surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from gnn.importers import (
    IMPORTER_VARIABLES_SCHEMA,
    POINTER_PATH,
    ImporterPin,
    PinError,
    check_importer,
    check_pointer,
    importer_variables,
    read_pin,
)
from gnn.importers import main as importers_main

COMMIT = "8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1"
VALID_SPEC = """\
# GNN Example: minimal two-state model
# GNN Version: 1.0

## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
Minimal two-state model

## ModelAnnotation
A minimal model used by the importer-conformance tests.

## StateSpaceBlock
A[2,2,type=float]
B[2,2,1,type=float]
C[2,type=float]
D[2,type=float]
s[2,1,type=float]
o[2,1,type=int]
t[1,type=int]

## Connections
D>s
s-A
A-o
s-B
B>s

## InitialParameterization
A={(0.9,0.1),(0.1,0.9)}
B={(1.0,0.0),(0.0,1.0)}
C=(0.0,0.0)
D=(0.5,0.5)

## Time
t=1

## Footer
Minimal fixture.
"""


def _write_spec(root: Path, relative: str = "models/min.gnn.md") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(VALID_SPEC, encoding="utf-8")
    return path


def _write_pin(root: Path, payload: Dict[str, Any], relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _pointer(root: Path, text: str) -> Path:
    path = root / POINTER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_read_pin_reads_a_party_sub_record(tmp_path: Path) -> None:
    path = _write_pin(
        tmp_path,
        {
            "schema_version": 1,
            "fep_lean": {"commit": "a" * 40},
            "gnn": {"commit": COMMIT, "owners": {"pyproject.toml": "b" * 64}},
        },
        "specs/source-pin.json",
    )

    pin = read_pin(path, party="gnn")

    assert pin.commit == COMMIT
    assert pin.party == "gnn"
    assert pin.raw["schema_version"] == 1


def test_read_pin_reports_an_absent_party(tmp_path: Path) -> None:
    path = _write_pin(
        tmp_path,
        {"schema_version": 1, "fep_lean": {"commit": "a" * 40}},
        "specs/source-pin.json",
    )

    with pytest.raises(PinError, match="no 'gnn' sub-record"):
        read_pin(path, party="gnn")


def test_check_pointer_accepts_a_complete_pointer(tmp_path: Path) -> None:
    pin = read_pin(_write_pin(tmp_path, {"commit": COMMIT}, ".github/gnn-pin.json"))
    _pointer(
        tmp_path,
        f"Pin: `.github/gnn-pin.json` at {COMMIT[:12]}.\n"
        "Job `gnn-conformance` in `.github/workflows/ci.yml`.\n",
    )

    assert (
        check_pointer(
            tmp_path, pin, workflow=".github/workflows/ci.yml", job="gnn-conformance"
        )
        == []
    )


def test_check_pointer_reports_a_missing_pointer(tmp_path: Path) -> None:
    pin = read_pin(_write_pin(tmp_path, {"commit": COMMIT}, ".github/gnn-pin.json"))

    codes = [item["code"] for item in check_pointer(tmp_path, pin)]

    assert codes == ["pointer-missing"]


def test_check_pointer_reports_each_unnamed_fact(tmp_path: Path) -> None:
    pin = read_pin(_write_pin(tmp_path, {"commit": COMMIT}, ".github/gnn-pin.json"))
    _pointer(tmp_path, "Nothing here names the pin.\n")

    codes = [
        item["code"]
        for item in check_pointer(
            tmp_path, pin, workflow=".github/workflows/ci.yml", job="gnn-conformance"
        )
    ]

    assert codes == [
        "pointer-pin-unnamed",
        "pointer-revision-unnamed",
        "pointer-workflow-unnamed",
        "pointer-job-unnamed",
    ]


def test_check_importer_reads_a_differently_named_pin(tmp_path: Path) -> None:
    _write_spec(tmp_path)
    _write_pin(
        tmp_path,
        {"repository": "ActiveInferenceInstitute/GEO_INFER", "revision": COMMIT},
        ".github/gnn-pair.json",
    )

    report = check_importer(
        tmp_path,
        pin_path=Path(".github/gnn-pair.json"),
        expected_repository="ActiveInferenceInstitute/GEO_INFER",
    )

    assert report["pin"]["path"] == ".github/gnn-pair.json"
    assert report["pin"]["commit"] == COMMIT
    assert report["ok"] is True


def test_check_importer_reports_only_the_absent_declared_pin(tmp_path: Path) -> None:
    """An explicit pin that is missing is one finding, not two."""
    _write_spec(tmp_path)

    report = check_importer(tmp_path, pin_path=Path("specs/gone.json"))

    assert [item["code"] for item in report["findings"]] == ["pin-missing"]


def test_check_importer_rejects_a_short_commit(tmp_path: Path) -> None:
    _write_spec(tmp_path)
    _write_pin(tmp_path, {"commit": "8f293d9b"}, ".github/gnn-pin.json")

    report = check_importer(tmp_path)

    assert "pin-commit-shape" in [item["code"] for item in report["findings"]]


def test_check_importer_records_the_pointer_verdict(tmp_path: Path) -> None:
    _write_spec(tmp_path)
    _write_pin(tmp_path, {"commit": COMMIT}, ".github/gnn-pin.json")

    report = check_importer(tmp_path, require_pointer=True)

    assert report["pointer"] == {"path": POINTER_PATH, "ok": False}
    assert "pointer-missing" in [item["code"] for item in report["findings"]]


def test_check_importer_without_the_flag_ignores_the_pointer(tmp_path: Path) -> None:
    _write_spec(tmp_path)
    _write_pin(tmp_path, {"commit": COMMIT}, ".github/gnn-pin.json")

    report = check_importer(tmp_path)

    assert "pointer" not in report


def test_importer_variables_carry_the_pin_facts(tmp_path: Path) -> None:
    pin = ImporterPin(
        path=tmp_path / "specs" / "gnn-pin.json",
        commit=COMMIT,
        tag="v5.0.3",
        repository="ActiveInferenceInstitute/Generalized_Notation_Notation",
        schema="gnn-importer-pin/v1",
    )

    variables = importer_variables(
        pin,
        repo_root=tmp_path,
        job="gnn-conformance",
        job_name="gnn conformance (pinned)",
    )

    assert variables["schema"] == IMPORTER_VARIABLES_SCHEMA
    assert variables["gnn_pin_path"] == "specs/gnn-pin.json"
    assert variables["gnn_commit"] == COMMIT
    assert variables["gnn_tag"] == "v5.0.3"
    assert variables["gnn_job_key"] == "gnn-conformance"


def test_importer_variables_use_empty_strings_for_absent_facts(tmp_path: Path) -> None:
    pin = ImporterPin(path=tmp_path / "gnn" / "upstream.json", commit=COMMIT)

    variables = importer_variables(pin, repo_root=tmp_path)

    assert variables["gnn_tag"] == ""
    assert variables["gnn_job_key"] == ""


def test_cli_threads_the_pin_and_pointer_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_spec(tmp_path)
    _write_pin(
        tmp_path,
        {"schema_version": 1, "gnn": {"commit": COMMIT}},
        "specs/source-pin.json",
    )
    _pointer(tmp_path, f"`specs/source-pin.json` at {COMMIT[:12]}\n")

    code = importers_main(
        [
            "check",
            "--repo",
            str(tmp_path),
            "--pin",
            "specs/source-pin.json",
            "--pin-party",
            "gnn",
            "--require-pointer",
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "pointer: .github/GNN.md -> ok" in out
    assert "[gnn]" in out
