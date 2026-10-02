"""The paired runtime must preserve virtual-environment interpreter identity."""

from __future__ import annotations

import json
import subprocess
import venv
from pathlib import Path

import pytest

from scripts.run_geo_interchange_checks import _python_path


@pytest.mark.needs_posix
def test_relative_venv_interpreter_keeps_installed_runtime(tmp_path, monkeypatch):
    environment = tmp_path / "runtime"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    monkeypatch.chdir(tmp_path)
    interpreter = _python_path(Path("runtime/bin/python"))
    assert interpreter == environment / "bin/python"
    assert interpreter.is_symlink()
    receipt = subprocess.run(
        [
            str(interpreter),
            "-I",
            "-c",
            "import json,sys; print(json.dumps(sys.prefix))",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert Path(json.loads(receipt.stdout)) == environment


def test_validator_timeout_preserves_partial_failure_receipt(tmp_path):
    import json
    import sys

    import pytest

    from scripts.run_geo_interchange_checks import (
        VALIDATOR_RELATIVE,
        _validator_receipt,
    )

    validator = tmp_path / "geo" / VALIDATOR_RELATIVE
    validator.parent.mkdir(parents=True)
    validator.write_text(
        "import time\nprint('partial-validator-evidence', flush=True)\ntime.sleep(10)\n"
    )
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    with pytest.raises(SystemExit, match="validator failed"):
        _validator_receipt(
            tmp_path / "geo",
            Path(sys.executable),
            tmp_path,
            Path(sys.executable),
            receipts,
            timeout=0.3,
        )
    failure = json.loads((receipts / "validator-failed.json").read_text())
    assert failure["error_type"] == "TimeoutExpired"
    assert failure["cleanup_verified"] is True
    assert failure["streams_drained"] is True
    assert "partial-validator-evidence" in failure["stdout_tail"]
    assert not (receipts / "interchange.json").exists()
