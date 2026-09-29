#!/usr/bin/env python3
"""Pin the per-script execution result envelope factories.

The four factories in ``execute.processor`` were unified onto
``_base_execution_envelope``; these tests assert each factory still emits
exactly the historical key set and the historical values for the
discriminating fields, so the dedup cannot silently drift a key.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnn.execute.processor import (  # noqa: E402
    _base_execution_envelope,
    _make_distributed_dispatch_failure_result,
    _make_local_worker_pool_failure_result,
    _make_skipped_result,
    _new_execution_result,
)
from gnn.execute.types import ScriptExecutionContext  # noqa: E402

# The 14-key shared prefix every envelope must start with.
_BASE_KEYS = {
    "script_path",
    "script_name",
    "framework",
    "model_name",
    "executor",
    "success",
    "skipped",
    "status",
    "attempts_started",
    "return_code",
    "stdout",
    "stderr",
    "execution_time",
    "timestamp",
}

_SKIP_KEYS = _BASE_KEYS | {
    "error",
    "error_type",
    "execution_metadata",
    "skip_category",
}
_LOCAL_FAIL_KEYS = _BASE_KEYS | {"error", "error_type", "worker_pool_error_type"}
_DIST_FAIL_KEYS = _BASE_KEYS | {
    "error",
    "error_type",
    "dispatch_error_type",
    "dispatch_max_retries",
}
_NEW_RESULT_KEYS = set(_BASE_KEYS)


def _script_info() -> dict:
    return {
        "path": "/tmp/sample/model_a/pymdp/model_a_pymdp.py",
        "name": "model_a_pymdp.py",
        "framework": "pymdp",
        "executor": "python3",
    }


def test_base_envelope_has_exactly_the_shared_keys() -> None:
    env = _base_execution_envelope(
        script_path="p",
        script_name="n",
        framework="pymdp",
        model_name="m",
        executor="python3",
        status="failed",
        skipped=False,
    )
    assert set(env) == _BASE_KEYS
    assert env["success"] is False
    assert env["skipped"] is False
    assert env["status"] == "failed"
    assert env["attempts_started"] == 0
    assert env["return_code"] is None
    assert env["stdout"] == ""
    assert env["stderr"] == ""
    assert env["execution_time"] == 0
    assert isinstance(env["timestamp"], str)


def test_make_skipped_result_key_set_and_values() -> None:
    import logging

    result = _make_skipped_result(
        _script_info(), "pymdp", "model_a", "python3", logging.getLogger("t")
    )
    assert set(result) == _SKIP_KEYS
    assert result["success"] is False
    assert result["skipped"] is True
    assert result["status"] == "skipped"
    assert result["error_type"] == "DependencyNotInstalled"
    assert result["execution_metadata"] == {}


def test_make_skipped_result_rxinfer_loads_metadata_block() -> None:
    import logging

    info = _script_info()
    info["framework"] = "rxinfer"
    # No sidecar exists → metadata is {}, but the key must be present.
    result = _make_skipped_result(
        info, "rxinfer", "model_a", "julia", logging.getLogger("t")
    )
    assert set(result) == _SKIP_KEYS
    assert result["execution_metadata"] == {}


def test_make_local_worker_pool_failure_result_key_set() -> None:
    result = _make_local_worker_pool_failure_result(
        _script_info(), RuntimeError("boom")
    )
    assert set(result) == _LOCAL_FAIL_KEYS
    assert result["success"] is False
    assert result["skipped"] is False
    assert result["status"] == "failed"
    assert result["error_type"] == "LocalWorkerPoolError"
    assert result["worker_pool_error_type"] == "RuntimeError"


def test_failure_factories_derive_model_framework_from_path() -> None:
    # Path has ≥3 parts, so model_name/framework come from the path, not the
    # discovery metadata fallback.
    info = {
        "path": "/tmp/model_a/pymdp/model_a_pymdp.py",
        "name": "model_a_pymdp.py",
        "framework": "pymdp",
        "executor": "python3",
    }
    result = _make_local_worker_pool_failure_result(info, RuntimeError("x"))
    assert result["model_name"] == "model_a"
    assert result["framework"] == "pymdp"


def test_failure_factories_short_path_falls_back_to_metadata() -> None:
    # <3 path parts → fall back to discovery metadata + "unknown_model".
    info = {
        "path": "short_pymdp.py",
        "name": "short_pymdp.py",
        "framework": "pymdp",
        "executor": "python3",
    }
    result = _make_local_worker_pool_failure_result(info, RuntimeError("x"))
    assert result["model_name"] == "unknown_model"
    assert result["framework"] == "pymdp"


def test_make_distributed_dispatch_failure_result_key_set() -> None:
    result = _make_distributed_dispatch_failure_result(
        _script_info(), ValueError("nope"), "ray", 3
    )
    assert set(result) == _DIST_FAIL_KEYS
    assert result["success"] is False
    assert result["error_type"] == "DistributedDispatchError"
    assert result["dispatch_error_type"] == "ValueError"
    assert result["dispatch_max_retries"] == 3
    assert "ray" in result["error"]
    assert "nope" in result["error"]


def test_new_execution_result_key_set() -> None:
    ctx = ScriptExecutionContext(
        script_path=Path("/tmp/sample/model_a/pymdp/model_a_pymdp.py"),
        script_name="model_a_pymdp.py",
        framework="pymdp",
        model_name="model_a",
        executor="python3",
    )
    result = _new_execution_result(ctx)
    assert set(result) == _NEW_RESULT_KEYS
    assert result["success"] is False
    assert result["skipped"] is False
    assert result["status"] == "failed"
