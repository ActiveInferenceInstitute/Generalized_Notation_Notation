#!/usr/bin/env python3
"""Pin list_frameworks registry introspection and MCP dependency serialization.

``execute.list_frameworks`` exposes the ExecutorFrameworkSpec registry; the
MCP ``check_execute_dependencies_mcp`` tool previously returned
non-JSON-serializable ValidationResult dataclass objects — the fix maps them
to plain dicts, which these tests assert.
"""

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnn.execute import list_frameworks  # noqa: E402
from gnn.execute.mcp import check_execute_dependencies_mcp  # noqa: E402

EXPECTED_FRAMEWORKS = {
    "pymdp",
    "rxinfer",
    "discopy",
    "activeinference_jl",
    "jax",
    "numpyro",
    "pytorch",
    "ngclearn",
    "thrml",
    "lean",
    "stan",
    "bnlearn",
}


@pytest.fixture(autouse=True)
def deterministic_registry_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep metadata assertions separate from native package-load acceptance."""
    from gnn.execute.executor import _RunnerState

    monkeypatch.setattr(
        "gnn.execute.executor._runner_state",
        lambda framework: _RunnerState(True, None),
    )


def test_list_frameworks_returns_one_record_per_registered_backend() -> None:
    records = list_frameworks()
    assert isinstance(records, list)
    assert {r["framework"] for r in records} == EXPECTED_FRAMEWORKS
    for record in records:
        assert set(record) == {
            "framework",
            "result_key",
            "available",
            "operation",
            "readiness",
        }
        assert record["readiness"]["framework"] == record["framework"]
        assert record["readiness"]["available"] is record["available"]
        assert record["result_key"].endswith("_executions")
        assert isinstance(record["available"], bool)
        assert record["operation"].startswith("execute_")


def test_list_frameworks_result_keys_match_registry_order() -> None:
    records = list_frameworks()
    # result_key is derived from the spec; ensure each maps back to its framework.
    by_fw = {r["framework"]: r["result_key"] for r in records}
    assert by_fw["pymdp"] == "pymdp_executions"
    assert by_fw["rxinfer"] == "rxinfer_executions"
    assert by_fw["discopy"] == "discopy_executions"
    assert by_fw["activeinference_jl"] == "activeinference_executions"
    assert by_fw["jax"] == "jax_executions"
    assert by_fw["numpyro"] == "numpyro_executions"
    assert by_fw["pytorch"] == "pytorch_executions"


def test_check_execute_dependencies_mcp_returns_json_serializable_payload() -> None:
    payload = check_execute_dependencies_mcp()
    # The MCP contract is a JSON response; the payload must round-trip.
    serialized = json.dumps(payload, default=str)
    deserialized = json.loads(serialized)
    assert deserialized["success"] is True
    # 'dependencies' must be a list of plain dicts (not dataclass objects).
    deps = deserialized.get("dependencies", [])
    assert isinstance(deps, list)
    for entry in deps:
        assert isinstance(entry, dict)
        # Each ValidationResult exposes component + status at minimum.
        assert "component" in entry
        assert "status" in entry
