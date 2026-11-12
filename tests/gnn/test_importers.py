"""Tests for the downstream importer conformance reader (``gnn.importers``).

The reader is the single place that understands the pin shapes the estate
actually carries (``commit``/``revision``, ``tag``/``release``, four file
locations), so these tests pin each accepted shape and each finding code. The
specification fixtures are real documents run through the real validator: a
"valid" fixture that stopped validating would make every importer check pass
for the wrong reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnn.importers import (
    PIN_SCHEMA,
    PinError,
    check_importer,
    discover_specs,
    find_pin,
    main,
    normalize_repository,
    read_pin,
)

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


def _write_pin(
    root: Path, payload: dict, relative: str = ".github/gnn-pin.json"
) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_spec(
    root: Path, text: str = VALID_SPEC, relative: str = "models/min.gnn.md"
) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _canonical_pin(**overrides: object) -> dict:
    payload = {
        "schema": PIN_SCHEMA,
        "repository": "ActiveInferenceInstitute/Generalized_Notation_Notation",
        "tag": "v5.0.3",
        "commit": COMMIT,
        "recorded": "2026-11-12",
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "ActiveInferenceInstitute/Generalized_Notation_Notation",
            "ActiveInferenceInstitute/Generalized_Notation_Notation",
        ),
        (
            "https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation",
            "ActiveInferenceInstitute/Generalized_Notation_Notation",
        ),
        (
            "https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation.git",
            "ActiveInferenceInstitute/Generalized_Notation_Notation",
        ),
        (
            "git@github.com:ActiveInferenceInstitute/Generalized_Notation_Notation.git",
            "ActiveInferenceInstitute/Generalized_Notation_Notation",
        ),
        (
            "ActiveInferenceInstitute/Generalized_Notation_Notation/tree/main",
            "ActiveInferenceInstitute/Generalized_Notation_Notation",
        ),
        ("", None),
        ("not-a-repo", None),
        (None, None),
    ],
)
def test_normalize_repository_accepts_every_shape_in_use(
    value: object, expected: str | None
) -> None:
    assert normalize_repository(value) == expected


def test_read_pin_accepts_the_canonical_schema(tmp_path: Path) -> None:
    pin = read_pin(_write_pin(tmp_path, _canonical_pin()))

    assert pin.commit == COMMIT
    assert pin.tag == "v5.0.3"
    assert pin.schema == PIN_SCHEMA
    assert pin.recorded == "2026-11-12"
    assert pin.repository == "ActiveInferenceInstitute/Generalized_Notation_Notation"


def test_read_pin_normalizes_revision_and_release(tmp_path: Path) -> None:
    """The legacy keys are the same fact; the reader must not care which is used."""
    pin = read_pin(
        _write_pin(
            tmp_path,
            {
                "repository": "ActiveInferenceInstitute/Generalized_Notation_Notation",
                "revision": COMMIT,
                "release": "v5.0.3",
            },
            relative="gnn/upstream.json",
        )
    )

    assert pin.commit == COMMIT
    assert pin.tag == "v5.0.3"
    assert pin.schema is None


def test_read_pin_accepts_a_single_declared_spec_file(tmp_path: Path) -> None:
    pin = read_pin(_write_pin(tmp_path, _canonical_pin(spec_files="models/one.gnn.md")))

    assert pin.spec_files == ("models/one.gnn.md",)


def test_read_pin_rejects_a_pin_without_a_revision(tmp_path: Path) -> None:
    with pytest.raises(PinError, match="no commit/revision"):
        read_pin(_write_pin(tmp_path, {"repository": "x/y", "tag": "v5.0.3"}))


def test_read_pin_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / ".github" / "gnn-pin.json"
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(PinError, match="must be a JSON object"):
        read_pin(path)


def test_find_pin_prefers_the_canonical_location(tmp_path: Path) -> None:
    _write_pin(tmp_path, {"commit": COMMIT}, relative="gnn/upstream.json")
    canonical = _write_pin(tmp_path, _canonical_pin())

    assert find_pin(tmp_path) == canonical


def test_find_pin_returns_none_when_the_repository_carries_no_pin(
    tmp_path: Path,
) -> None:
    assert find_pin(tmp_path) is None


def test_discover_specs_honours_the_declared_list(tmp_path: Path) -> None:
    declared = _write_spec(tmp_path, relative="models/declared.gnn.md")
    _write_spec(tmp_path, relative="models/other.gnn.md")
    pin = read_pin(
        _write_pin(tmp_path, _canonical_pin(spec_files=["models/declared.gnn.md"]))
    )

    assert discover_specs(tmp_path, pin) == [declared]


def test_discover_specs_skips_dependency_directories(tmp_path: Path) -> None:
    kept = _write_spec(tmp_path, relative="models/kept.gnn.md")
    _write_spec(tmp_path, relative=".venv/lib/site-packages/vendored.gnn.md")
    _write_spec(tmp_path, relative="node_modules/pkg/vendored.gnn.md")

    assert discover_specs(tmp_path) == [kept]


def test_check_importer_passes_a_conformant_repository(tmp_path: Path) -> None:
    _write_pin(tmp_path, _canonical_pin())
    _write_spec(tmp_path)

    report = check_importer(tmp_path, expected_commit=COMMIT, expected_tag="v5.0.3")

    assert report["ok"] is True
    assert report["pin"]["commit"] == COMMIT
    assert report["specs"]["count"] == 1
    assert report["specs"]["invalid"] == []
    assert report["findings"] == []


def test_check_importer_reports_a_missing_pin_as_an_error(tmp_path: Path) -> None:
    _write_spec(tmp_path)

    report = check_importer(tmp_path)

    assert report["ok"] is False
    assert [item["code"] for item in report["findings"]] == ["pin-missing"]


def test_check_importer_can_allow_a_missing_pin(tmp_path: Path) -> None:
    _write_spec(tmp_path)

    report = check_importer(tmp_path, require_pin=False)

    assert report["ok"] is True
    assert report["findings"][0]["level"] == "warning"


def test_check_importer_reports_a_commit_mismatch(tmp_path: Path) -> None:
    _write_pin(tmp_path, _canonical_pin())
    _write_spec(tmp_path)

    report = check_importer(tmp_path, expected_commit="0" * 40)

    assert report["ok"] is False
    assert "pin-commit-mismatch" in [item["code"] for item in report["findings"]]


def test_check_importer_reports_a_tag_mismatch(tmp_path: Path) -> None:
    _write_pin(tmp_path, _canonical_pin())
    _write_spec(tmp_path)

    report = check_importer(tmp_path, expected_tag="v9.9.9")

    assert "pin-tag-mismatch" in [item["code"] for item in report["findings"]]


def test_check_importer_flags_a_legacy_schema_as_a_warning_only(tmp_path: Path) -> None:
    _write_pin(tmp_path, {"repository": "x/y", "revision": COMMIT, "tag": "v5.0.3"})
    _write_spec(tmp_path)

    report = check_importer(tmp_path)

    assert report["ok"] is True
    assert [item["code"] for item in report["findings"]] == ["pin-schema-legacy"]


def test_check_importer_rejects_a_specification_missing_a_required_section(
    tmp_path: Path,
) -> None:
    """The compliance gate is the normative section contract, not a parse."""
    _write_pin(tmp_path, _canonical_pin())
    without_time = VALID_SPEC.split("## Time")[0] + "## Footer\nMinimal fixture.\n"
    _write_spec(tmp_path, text=without_time)

    report = check_importer(tmp_path)

    assert report["ok"] is False
    assert report["specs"]["invalid"][0]["file"] == "models/min.gnn.md"
    assert "Time" in report["specs"]["invalid"][0]["errors"][0]


def test_check_importer_reports_a_declared_spec_that_is_absent(tmp_path: Path) -> None:
    _write_pin(tmp_path, _canonical_pin(spec_files=["models/gone.gnn.md"]))

    report = check_importer(tmp_path)

    assert report["ok"] is False
    assert "spec-missing" in [item["code"] for item in report["findings"]]


def test_check_importer_warns_when_a_pin_carries_no_specification(
    tmp_path: Path,
) -> None:
    _write_pin(tmp_path, _canonical_pin())

    report = check_importer(tmp_path)

    assert report["ok"] is True
    assert [item["code"] for item in report["findings"]] == ["spec-absent"]


def test_cli_exit_codes_and_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_pin(tmp_path, _canonical_pin())
    _write_spec(tmp_path)

    assert main(["check", "--repo", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["pin"]["tag"] == "v5.0.3"

    (tmp_path / "models" / "min.gnn.md").write_text("# not a model\n", encoding="utf-8")
    assert main(["check", "--repo", str(tmp_path)]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_cli_no_validate_skips_the_parser(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_pin(tmp_path, _canonical_pin())
    _write_spec(tmp_path, text="# not a model\n")

    assert main(["check", "--repo", str(tmp_path), "--no-validate"]) == 0
    assert "specs: 1" in capsys.readouterr().out
