"""External artifacts distinguish launch refusal from successful cleanup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from gnn.pipeline.output_lease import OutputLease, OutputLeaseError
from gnn.utils.pipeline_orchestration.execution_utils import execute_command_streaming


@pytest.mark.parametrize(
    "refusal", ["missing-executable", "missing-cwd", "expired-budget"]
)
def test_launch_refusal_does_not_create_artifact_or_certify_cleanup(
    tmp_path: Path, refusal: str, capsys: pytest.CaptureFixture[str]
) -> None:
    marker = tmp_path / "worker-created.txt"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('started')",
        str(marker),
    ]
    cwd = tmp_path
    timeout = 3
    if refusal == "missing-executable":
        command[0] = str(tmp_path / "missing-python")
    elif refusal == "missing-cwd":
        cwd = tmp_path / "missing-workspace"
    else:
        timeout = 0
    result = execute_command_streaming(
        command,
        cwd=cwd,
        timeout=timeout,
        capture_output=False,
    )
    assert result["status"] == "FAILED" and result["exit_code"] == -1, result
    assert result["execution_error_type"] == (
        "TimeoutError" if refusal == "expired-budget" else "FileNotFoundError"
    ), result
    assert not marker.exists()
    assert result["stdout"] == ""
    assert result["stderr"]  # Disabling capture must not erase admission diagnostics.
    assert "Execution error:" in capsys.readouterr().err
    assert result["cleanup_verified"] is False
    assert result["streams_drained"] is False
    assert "containment" not in result  # No owned worker was created.


def test_output_lease_refuses_regular_file_without_replacing_it(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.write_bytes(b"operator data must survive\n")
    before = output.stat()
    with pytest.raises(OutputLeaseError):
        with OutputLease(output, "refused-file"):
            pytest.fail("A file cannot be admitted as a pipeline output directory")
    after = output.stat()
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
    assert output.read_bytes() == b"operator data must survive\n"
    assert list(tmp_path.iterdir()) == [output]
