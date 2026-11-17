"""Tests for the downstream importer conformance reader (``gnn.importers``).

The reader is the single place that understands the pin shapes the estate
actually carries (``commit``/``revision``, ``tag``/``release``, four file
locations), so these tests pin each accepted shape and each finding code. The
specification fixtures are real documents run through the real validator: a
"valid" fixture that stopped validating would make every importer check pass
for the wrong reason.
"""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

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
    "repository",
    [
        "https://evil.invalid/ActiveInferenceInstitute/Generalized_Notation_Notation",
        "https://ActiveInferenceInstitute/Generalized_Notation_Notation",
        "https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/tree/main",
        "https://user@github.com/ActiveInferenceInstitute/Generalized_Notation_Notation",
        "https://github.com:443/ActiveInferenceInstitute/Generalized_Notation_Notation",
        "https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation?ref=main",
        "",
        17,
    ],
)
def test_invalid_declared_repository_cannot_become_an_omitted_identity(
    tmp_path: Path, repository: object
) -> None:
    path = _write_pin(tmp_path, _canonical_pin(repository=repository))
    _write_spec(tmp_path)
    with pytest.raises(PinError, match="repository"):
        read_pin(path)
    report = check_importer(tmp_path)
    assert report["ok"] is False
    assert report["specs"]["count"] == 1
    assert report["specs"]["invalid"] == []
    assert [item["code"] for item in report["findings"]] == ["pin-unreadable"]


def test_genuinely_omitted_legacy_repository_remains_supported(tmp_path: Path) -> None:
    payload = _canonical_pin()
    payload.pop("repository")
    assert read_pin(_write_pin(tmp_path, payload)).repository is None
    _write_spec(tmp_path)
    report = check_importer(tmp_path, expected_commit=COMMIT, expected_tag="v5.0.3")
    assert report["ok"] is True
    assert report["pin"]["repository"] is None
    assert report["specs"]["count"] == 1
    assert report["specs"]["invalid"] == []
    assert report["findings"] == []


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
            None,
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


def test_check_importer_rejects_a_pin_that_names_a_fork(tmp_path: Path) -> None:
    """A pin naming a fork tracks code this distribution does not publish."""
    _write_pin(
        tmp_path, _canonical_pin(repository="someone/Generalized_Notation_Notation")
    )
    _write_spec(tmp_path)

    report = check_importer(tmp_path)

    assert report["ok"] is False
    assert "pin-repository-mismatch" in [item["code"] for item in report["findings"]]


def test_check_importer_accepts_a_url_form_repository(tmp_path: Path) -> None:
    _write_pin(
        tmp_path,
        _canonical_pin(
            repository="https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation.git"
        ),
    )
    _write_spec(tmp_path)

    report = check_importer(tmp_path)

    assert report["ok"] is True


def test_check_importer_flags_an_unversioned_schema_as_a_warning_only(
    tmp_path: Path,
) -> None:
    _write_pin(
        tmp_path,
        {
            "repository": "ActiveInferenceInstitute/Generalized_Notation_Notation",
            "revision": COMMIT,
            "tag": "v5.0.3",
        },
    )
    _write_spec(tmp_path)

    report = check_importer(tmp_path)

    assert report["ok"] is True
    assert [item["code"] for item in report["findings"]] == ["pin-schema-unversioned"]


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


@pytest.mark.parametrize(
    "declaration",
    [
        True,
        3,
        None,
        [None],
        [7],
        ["../outside.gnn.md"],
        ["/outside.gnn.md"],
        ["C:\\outside.gnn.md"],
        ["models//min.gnn.md"],
    ],
)
def test_pin_rejects_invalid_declared_source_paths(
    tmp_path: Path, declaration: object
) -> None:
    with pytest.raises(PinError):
        read_pin(_write_pin(tmp_path, _canonical_pin(spec_files=declaration)))


def test_explicit_empty_source_list_does_not_discover_other_documents(
    tmp_path: Path,
) -> None:
    _write_spec(tmp_path)
    pin = read_pin(_write_pin(tmp_path, _canonical_pin(spec_files=[])))
    assert discover_specs(tmp_path, pin) == []


def test_declared_symlink_is_rejected_without_parsing_external_source(
    tmp_path: Path,
) -> None:
    consumer = tmp_path / "consumer"
    external = tmp_path / "outside.gnn.md"
    external.write_bytes(b"\xff")
    link = consumer / "models" / "linked.gnn.md"
    link.parent.mkdir(parents=True)
    link.symlink_to(external)
    _write_pin(consumer, _canonical_pin(spec_files=["models/linked.gnn.md"]))
    report = check_importer(consumer)
    assert report["ok"] is False
    assert report["specs"]["invalid"] == []
    assert "spec-path-unsafe" in [item["code"] for item in report["findings"]]


@pytest.mark.parametrize(
    "aliases",
    [
        {"commit": COMMIT, "revision": "f" * 40},
        {"tag": "v5.0.3", "release": "v5.1.0"},
    ],
)
def test_conflicting_legacy_aliases_do_not_select_one_pin(
    tmp_path: Path, aliases: dict
) -> None:
    with pytest.raises(PinError, match="disagree"):
        read_pin(_write_pin(tmp_path, _canonical_pin(**aliases)))


def test_duplicate_identity_keys_are_not_last_value_wins(tmp_path: Path) -> None:
    pin = tmp_path / "pin.json"
    pin.write_text(
        '{"commit":"' + COMMIT + '","commit":"' + "f" * 40 + '"}',
        encoding="utf-8",
    )
    with pytest.raises(PinError, match="duplicate JSON member"):
        read_pin(pin)


def test_invalid_utf8_pin_returns_a_json_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pin = tmp_path / ".github" / "gnn-pin.json"
    pin.parent.mkdir()
    pin.write_bytes(b"\xff")
    assert main(["check", "--repo", str(tmp_path), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert "pin-unreadable" in [item["code"] for item in report["findings"]]


@pytest.fixture
def registry_script() -> Iterator[ModuleType]:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "check_importer_registry.py"
    spec = importlib.util.spec_from_file_location("_tested_importer_registry", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.fixture(
    params=[
        (
            "v5.0.3",
            COMMIT,
            "17a35bb2114e76a38119128d290d12377cb641652d70788000482366dca0a14a",
        ),
        (
            "v5.1.0",
            "75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf",
            "43b33c0b2b40386afa3f306fec4cc13bc1fd7c3b570fbf11d5008bad5c8e6b51",
        ),
    ]
)
def registry_source_case(tmp_path: Path, request: pytest.FixtureRequest) -> dict:
    """Use real pinned Git source and the independently frozen manifest."""
    tag, commit, digest = request.param
    root = Path(__file__).resolve().parents[2]
    archived = subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(root),
            "archive",
            "--format=tar",
            commit,
            "--",
            "src/gnn",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    source = tmp_path / "source"
    source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archived.stdout)) as bundle:
        bundle.extractall(source, filter="data")
    consumer = tmp_path / "consumer"
    _write_pin(consumer, _canonical_pin(tag=tag, commit=commit))
    _write_spec(consumer)
    return {
        "consumer": consumer,
        "package": source / "src" / "gnn",
        "manifest": root / "interop" / f"{tag}-source-manifest.json",
        "digest": digest,
        "registry": {
            "upstream": {"tag": "deliberately-not-the-member", "commit": "f" * 40},
            "importers": [
                {
                    "repo": "fixture-member",
                    "pin": {
                        "kind": "file",
                        "path": ".github/gnn-pin.json",
                        "tag": tag,
                        "commit": commit,
                    },
                    "specs": {"count": 1},
                }
            ],
        },
    }


def test_registry_uses_member_pin_and_actual_guarded_parser(
    registry_script: ModuleType, registry_source_case: dict
) -> None:
    case = registry_source_case
    findings = registry_script.Findings()
    registry_script.check_local_repo(
        case["registry"],
        "fixture-member",
        case["consumer"],
        findings,
        importer_python=Path(sys.executable),
        source_manifest=case["manifest"],
        source_manifest_sha256=case["digest"],
        package_root=case["package"],
        origin_mode="source",
    )
    assert findings.failed is False, findings.items
    assert findings.items == []


def test_registry_refuses_changed_real_package_content(
    registry_script: ModuleType, registry_source_case: dict
) -> None:
    case = registry_source_case
    with (case["package"] / "__init__.py").open("ab") as stream:
        stream.write(b"\n# Changed after the independently frozen release.\n")
    findings = registry_script.Findings()
    registry_script.check_local_repo(
        case["registry"],
        "fixture-member",
        case["consumer"],
        findings,
        source_manifest=case["manifest"],
        source_manifest_sha256=case["digest"],
        package_root=case["package"],
        origin_mode="source",
    )
    assert findings.failed is True
    assert "local-source-content-mismatch" in [item["code"] for item in findings.items]


def test_registry_requires_independently_frozen_manifest(
    registry_script: ModuleType, tmp_path: Path
) -> None:
    findings = registry_script.Findings()
    registry_script.check_local_repo(
        {"importers": [{"repo": "fixture-member"}]},
        "fixture-member",
        tmp_path,
        findings,
    )
    assert [item["code"] for item in findings.items] == [
        "local-source-manifest-required"
    ]


@pytest.mark.parametrize("shadow_module", ["argparse", "_colorize"])
def test_registry_keeps_unadmitted_parent_modules_off_import_path(
    registry_script: ModuleType, registry_source_case: dict, shadow_module: str
) -> None:
    case = registry_source_case
    shadow = case["package"].parent / f"{shadow_module}.py"
    shadow.write_text(
        'raise RuntimeError("unadmitted source-parent module executed")\n',
        encoding="utf-8",
    )
    findings = registry_script.Findings()
    registry_script.check_local_repo(
        case["registry"],
        "fixture-member",
        case["consumer"],
        findings,
        source_manifest=case["manifest"],
        source_manifest_sha256=case["digest"],
        package_root=case["package"],
        origin_mode="source",
    )
    assert findings.failed is False, findings.items
    assert findings.items == []


@pytest.mark.parametrize(
    ("returncode", "ok", "levels", "valid"),
    [
        (1, False, [], False),
        (1, False, ["warning"], False),
        (0, True, ["error"], False),
        (0, True, [], True),
        (1, False, ["error"], True),
    ],
)
def test_registry_child_verdict_must_match_its_error_findings(
    registry_script: ModuleType,
    returncode: int,
    ok: bool,
    levels: list[str],
    valid: bool,
) -> None:
    """Schema vectors alone are not source or native qualification evidence."""
    report = {
        "ok": ok,
        "qualified": ok,
        "specs": {"count": 1},
        "findings": [
            {"level": level, "code": "controlled-unit-report", "message": "Unit vector"}
            for level in levels
        ],
    }
    assert registry_script._valid_local_reader_report(report, returncode) is valid
