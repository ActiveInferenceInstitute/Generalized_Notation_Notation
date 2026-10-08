"""Native public CLI consumers retain UTF-8 output and caller stream custody.

These run local commands and authored files without models or providers. Forced
cp1252/ASCII subprocess streams reproduce encoding boundaries on any host;
they do not establish native Windows installed-wheel acceptance.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "input/gnn_files/basics/static_perception.md"


def _model(tmp_path: Path, name: str = "model.md") -> Path:
    path = tmp_path / name
    path.write_bytes(SOURCE.read_bytes())
    return path


def _invoke(
    tmp_path: Path, command: list[str], encoding: str
) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO / "src")
    environment["PYTHONIOENCODING"] = encoding + ":strict"
    child = subprocess.run(  # nosec B603 — selected entrypoint and owned files
        command, cwd=tmp_path, env=environment, capture_output=True, timeout=45
    )
    number = len(list(tmp_path.glob("child-*.json")))
    (tmp_path / f"child-{number}-stdout.bin").write_bytes(child.stdout)
    (tmp_path / f"child-{number}-stderr.bin").write_bytes(child.stderr)
    (tmp_path / f"child-{number}.json").write_text(
        json.dumps(
            {
                "command": command,
                "stdio_encoding": encoding,
                "returncode": child.returncode,
                "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return child


def _entrypoint(kind: str) -> list[str]:
    if kind == "module":
        return [sys.executable, "-P", "-m", "gnn.cli"]
    executable = Path(sys.executable).parent / ("gnn.exe" if os.name == "nt" else "gnn")
    assert executable.is_file(), (
        "selected environment must contain public gnn console entrypoint"
    )
    return [str(executable)]


@pytest.mark.parametrize("encoding", ["cp1252", "ascii"])
@pytest.mark.parametrize("entrypoint", ["module", "console"])
@pytest.mark.parametrize("filename", ["static_perception.md", "sensor_μ_状态.md"])
def test_native_successful_validate_retains_unicode_and_exit_zero(
    tmp_path: Path, encoding: str, entrypoint: str, filename: str
) -> None:
    path = _model(tmp_path, filename)
    original = path.read_bytes()
    child = _invoke(
        tmp_path, [*_entrypoint(entrypoint), "validate", str(path)], encoding
    )
    assert child.returncode == 0, child.stderr.decode("utf-8", errors="replace")
    output = child.stdout.decode("utf-8", errors="strict")
    assert output == f"✅ {path}: valid (7 variables, 6 connections)\n"
    assert "�" not in child.stderr.decode("utf-8", errors="strict")
    assert path.read_bytes() == original == SOURCE.read_bytes()


@pytest.mark.parametrize("encoding", ["cp1252", "ascii"])
def test_native_parse_summary_preserves_unicode_filename_and_source_counts(
    tmp_path: Path, encoding: str
) -> None:
    path = _model(tmp_path, "sensor_μ_状态.md")
    original = path.read_bytes()
    child = _invoke(
        tmp_path,
        [*_entrypoint("module"), "parse", str(path), "--format", "summary"],
        encoding,
    )
    assert child.returncode == 0, child.stderr.decode("utf-8", errors="replace")
    output = child.stdout.decode("utf-8", errors="strict")
    assert output.splitlines()[:3] == [
        "File: " + path.name,
        "Variables: 7",
        "Connections: 6",
    ]
    assert path.read_bytes() == original


@pytest.mark.parametrize("encoding", ["cp1252", "ascii"])
def test_native_missing_source_retains_error_envelope_and_unicode_stderr(
    tmp_path: Path, encoding: str
) -> None:
    path = tmp_path / "missing_μ_状态.md"
    child = _invoke(
        tmp_path, [*_entrypoint("module"), "validate", str(path), "--json"], encoding
    )
    assert child.returncode == 1
    payload = json.loads(child.stdout.decode("utf-8", errors="strict"))
    assert payload["status"] == "error" and payload["meta"]["command"] == "validate"
    assert str(path) in payload["error"]
    assert str(path) in child.stderr.decode("utf-8", errors="strict")
    assert not path.exists()


@pytest.mark.parametrize("encoding", ["cp1252", "ascii"])
def test_native_json_validate_preserves_existing_structured_values(
    tmp_path: Path, encoding: str
) -> None:
    path = _model(tmp_path, "sensor_μ_状态.md")
    child = _invoke(
        tmp_path, [*_entrypoint("module"), "validate", str(path), "--json"], encoding
    )
    assert child.returncode == 0, child.stderr.decode("utf-8", errors="replace")
    payload = json.loads(child.stdout.decode("utf-8", errors="strict"))
    assert payload["status"] == "success" and payload["error"] is None
    assert payload["data"]["file"] == str(path)
    assert payload["data"]["valid"] is True
    assert payload["data"]["variables_count"] == 7
    assert payload["data"]["connections_count"] == 6
    assert payload["data"]["semantic"]["valid"] is True


@pytest.mark.parametrize("encoding", ["cp1252", "ascii"])
@pytest.mark.parametrize("malformed", [False, True])
def test_native_python_main_restores_stream_settings_after_success_or_argument_error(
    tmp_path: Path, encoding: str, malformed: bool
) -> None:
    path = _model(tmp_path)
    arguments = ["unknown_μ_状态"] if malformed else ["validate", str(path)]
    code = (
        "import json,sys; before=[(id(s),s.encoding,s.errors) for s in (sys.stdout,sys.stderr)]; "
        "from gnn.cli import main; imported=[(id(s),s.encoding,s.errors) for s in (sys.stdout,sys.stderr)]; "
        "\ntry:\n    rc=main(json.loads(sys.argv[1]))\n"
        "except SystemExit as exc:\n    rc=exc.code\n"
        "after=[(id(s),s.encoding,s.errors) for s in (sys.stdout,sys.stderr)]; "
        "print('CALLER_RECEIPT='+json.dumps({'before':before,'imported':imported,'after':after,'returncode':rc}))"
    )
    child = _invoke(
        tmp_path, [sys.executable, "-P", "-c", code, json.dumps(arguments)], encoding
    )
    assert child.returncode == 0, child.stderr.decode("utf-8", errors="replace")
    receipt_line = next(
        line
        for line in child.stdout.splitlines()
        if line.startswith(b"CALLER_RECEIPT=")
    )
    receipt = json.loads(receipt_line.split(b"=", 1)[1])
    assert receipt["before"] == receipt["imported"] == receipt["after"]
    assert receipt["before"][0][1:] == [encoding, "strict"]
    assert receipt["returncode"] == (2 if malformed else 0)
    if malformed:
        assert "unknown_μ_状态" in child.stderr.decode("utf-8", errors="strict")
    else:
        assert "✅".encode("utf-8") in child.stdout


def test_native_python_main_preserves_stringio_and_existing_closed_capture_error(
    tmp_path: Path,
) -> None:
    path = _model(tmp_path, "sensor_μ_状态.md")
    code = (
        "import contextlib,io,json,sys; from gnn.cli import main; "
        "original=[(id(s),s.encoding,s.errors) for s in (sys.stdout,sys.stderr)]; "
        "out=io.StringIO(); err=io.StringIO(); "
        "\nwith contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):\n"
        "    rc=main(['validate',sys.argv[1]])\n"
        "closed=io.StringIO(); closed.close()\n"
        "with contextlib.redirect_stdout(closed),contextlib.redirect_stderr(err):\n"
        "    failed=main(['validate',sys.argv[1]])\n"
        "after=[(id(s),s.encoding,s.errors) for s in (sys.stdout,sys.stderr)]; "
        "print(json.dumps({'returncode':rc,'closed_returncode':failed,'stdout':out.getvalue(),"
        "'stderr':err.getvalue(),'same_streams':original==after,'closed':closed.closed}))"
    )
    child = _invoke(tmp_path, [sys.executable, "-P", "-c", code, str(path)], "cp1252")
    assert child.returncode == 0, child.stderr.decode("utf-8", errors="replace")
    payload = json.loads(child.stdout)
    assert payload["returncode"] == 0 and payload["closed_returncode"] == 1
    assert payload["same_streams"] is True and payload["closed"] is True
    assert payload["stdout"] == f"✅ {path}: valid (7 variables, 6 connections)\n"
    assert "closed file" in payload["stderr"]


def test_native_python_main_preserves_closed_native_text_stream_error(
    tmp_path: Path,
) -> None:
    path = _model(tmp_path)
    output = tmp_path / "caller-closed.txt"
    code = (
        "import contextlib,io,json,sys; from gnn.cli import main; "
        "closed=open(sys.argv[2],'w',encoding='cp1252',errors='strict'); closed.close(); "
        "settings=[closed.encoding,closed.errors,closed.closed]; err=io.StringIO(); "
        "\nwith contextlib.redirect_stdout(closed),contextlib.redirect_stderr(err):\n"
        "    rc=main(['validate',sys.argv[1]])\n"
        "print(json.dumps({'returncode':rc,'same_settings':settings==[closed.encoding,closed.errors,closed.closed],"
        "'stderr':err.getvalue()}))"
    )
    child = _invoke(
        tmp_path, [sys.executable, "-P", "-c", code, str(path), str(output)], "cp1252"
    )
    assert child.returncode == 0, child.stderr.decode("utf-8", errors="replace")
    payload = json.loads(child.stdout)
    assert payload["returncode"] == 1 and payload["same_settings"] is True
    assert "closed file" in payload["stderr"]
    assert output.read_bytes() == b""
