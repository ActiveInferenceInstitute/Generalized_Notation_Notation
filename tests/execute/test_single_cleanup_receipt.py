"""Step12 keeps the supervised subprocess cleanup verdict in persisted receipts."""

import json
import logging
import sys
from pathlib import Path

from gnn.execute.processor import single


def test_cleanup_failure_survives_execution_and_persisted_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    script = tmp_path / "fixture.py"
    script.write_text('print("fixture")\n')
    cleanup = {
        "success": False,
        "return_code": -9,
        "stdout": "",
        "stderr": "",
        "error_type": "ProcessCleanupFailure",
        "error": "child cleanup unverified",
        "execution_error_type": "TimeoutExpired",
        "containment": "process_group_and_observed_descendants",
        "cleanup_verified": False,
        "streams_drained": False,
        "cleanup_timeout_seconds": 0.2,
        "cleanup_error": "pipe remained open",
        "cancelled": False,
    }
    monkeypatch.setattr(
        single, "run_subprocess_envelope", lambda *args, **kwargs: cleanup
    )
    result = single.execute_single_script(
        {
            "path": script,
            "name": script.name,
            "framework": "python",
            "executor": sys.executable,
        },
        tmp_path / "output",
        False,
        logging.getLogger(__name__),
    )
    assert not result["success"] and result["error_type"] == "ProcessCleanupFailure"
    persisted = json.loads(Path(result["structured_result_file"]).read_text())
    for key in (
        "containment",
        "cleanup_verified",
        "streams_drained",
        "cleanup_timeout_seconds",
        "cleanup_error",
        "execution_error_type",
        "error_type",
    ):
        assert persisted[key] == result[key] == cleanup[key]
