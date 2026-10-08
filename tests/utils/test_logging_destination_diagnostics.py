"""An unavailable real log destination retains useful console diagnostics."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("mode", ["basic", "pipeline", "json"])
def test_occupied_log_directory_keeps_console_and_typed_cause(
    tmp_path: Path, mode: str
) -> None:
    blocked = tmp_path / "occupied"
    blocked.write_text("existing evidence", encoding="utf-8")
    program = """
import logging
import sys
from pathlib import Path
from gnn.utils.logging.logging_utils import BasicPipelineLogger
from gnn.utils.logging_utils import PipelineLogger
path = Path(sys.argv[1])
mode = sys.argv[2]
if mode == "basic":
    BasicPipelineLogger.initialize(log_dir=path)
elif mode == "pipeline":
    PipelineLogger.initialize(log_dir=path)
else:
    PipelineLogger.initialize()
    PipelineLogger.enable_json_logging(path)
logging.getLogger("consumer").warning("console still usable")
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(blocked), mode],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "FileExistsError" in result.stdout
    assert str(blocked) in result.stdout
    assert "console still usable" in result.stdout
    assert blocked.read_text() == "existing evidence"
