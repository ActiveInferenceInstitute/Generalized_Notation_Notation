#!/usr/bin/env python3
"""Pin the explicit-refusal probe receipts for the continuous executor path.

Slice C of the t-0070 "explicit refusals" sweep: the RxInfer metadata
sidecar/TOML loaders, the rxinfer dispatch envelope, the device-discovery
fallback, and the prior-summary merge exclusion must surface receipts or logs
instead of silently degrading.
"""

import hashlib
import json
import logging
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gnn.execute.metadata as metadata_mod  # noqa: E402
from gnn.execute.executor import (  # noqa: E402
    GNNExecutor,
    _synthesize_rxinfer_envelope,
)
from gnn.execute.metadata import (  # noqa: E402
    _load_rxinfer_execution_metadata_from_script,
    _load_rxinfer_execution_metadata_sidecar,
    _merge_prior_execution_summary,
)
from gnn.execute.processor import _make_skipped_result  # noqa: E402


def _write_script(tmp_path: Path) -> Path:
    script = tmp_path / "model_a_rxinfer.jl"
    script.write_text('println("ok")\n', encoding="utf-8")
    return script


def _prior_results_file(
    tmp_path: Path, scope: Path, run_id: str, receipts: dict
) -> Path:
    prior = {
        "receipt_identity": {
            "run_id": run_id,
            "config_sha256": hashlib.sha256(
                json.dumps({}, sort_keys=True, default=str).encode()
            ).hexdigest(),
        },
        "invocation_receipts": receipts,
    }
    results_file = tmp_path / "execution.json"
    results_file.write_text(json.dumps(prior), encoding="utf-8")
    return results_file


def _execution_results(scope: Path, run_id: str) -> dict:
    return {
        "run_id": run_id,
        "configuration": {},
        "target_directory": str(scope),
        "execution_details": [],
    }


def test_corrupt_sidecar_returns_unreadable_receipt(tmp_path: Path) -> None:
    script = _write_script(tmp_path)
    sidecar = script.with_suffix(".metadata.json")
    sidecar.write_text("{not json", encoding="utf-8")
    receipt = _load_rxinfer_execution_metadata_sidecar(script)
    assert receipt["metadata_probe"] == "unreadable"
    assert receipt["probe_source"] == str(sidecar)
    assert receipt["probe_reason"]


def test_stale_sha_sidecar_returns_sha_mismatch(tmp_path: Path) -> None:
    script = _write_script(tmp_path)
    sidecar = script.with_suffix(".metadata.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema": "gnn_rxinfer_execution_metadata_v1",
                "script_sha256": "0" * 64,
                "agent_count": 3,
            }
        ),
        encoding="utf-8",
    )
    receipt = _load_rxinfer_execution_metadata_sidecar(script)
    assert receipt["metadata_probe"] == "sha_mismatch"
    assert receipt["probe_source"] == str(sidecar)


def test_wrong_schema_sidecar_returns_schema_mismatch(tmp_path: Path) -> None:
    script = _write_script(tmp_path)
    sidecar = script.with_suffix(".metadata.json")
    sidecar.write_text(
        json.dumps({"schema": "some_other_v9", "agent_count": 3}), encoding="utf-8"
    )
    receipt = _load_rxinfer_execution_metadata_sidecar(script)
    assert receipt["metadata_probe"] == "schema_mismatch"
    assert receipt["probe_source"] == str(sidecar)


def test_absent_sidecar_returns_empty_dict(tmp_path: Path) -> None:
    # Pins test_execute_envelope_factories semantics: nothing on disk → {}.
    script = _write_script(tmp_path)
    assert _load_rxinfer_execution_metadata_sidecar(script) == {}
    assert _load_rxinfer_execution_metadata_from_script(script) == {}


def test_valid_toml_wins_over_corrupt_sidecar(tmp_path: Path) -> None:
    script = _write_script(tmp_path)
    script.with_suffix(".metadata.json").write_text("{broken", encoding="utf-8")
    toml = script.with_suffix(".toml")
    toml.write_text(
        '[[agents]]\nid = "a1"\n[[agents]]\nid = "a2"\n[model]\nnr_agents = 2\n',
        encoding="utf-8",
    )
    data = _load_rxinfer_execution_metadata_from_script(script)
    assert "metadata_probe" not in data
    assert data["agent_count"] == 2
    assert data["metadata_provenance"] == "rxinfer_toml_sidecar"
    assert data["topology"]["source"] == str(toml)


def test_corrupt_sidecar_and_missing_toml_surface_probe(tmp_path: Path) -> None:
    script = _write_script(tmp_path)
    script.with_suffix(".metadata.json").write_text("{broken", encoding="utf-8")
    receipt = _load_rxinfer_execution_metadata_from_script(script)
    assert receipt["metadata_probe"] == "unreadable"
    assert receipt["probe_source"].endswith(".metadata.json")


def test_make_skipped_result_rxinfer_carries_probe_receipt(tmp_path: Path) -> None:
    script_dir = tmp_path / "sample" / "model_a" / "rxinfer"
    script_dir.mkdir(parents=True)
    script = script_dir / "model_a_rxinfer.jl"
    script.write_text('println("ok")\n', encoding="utf-8")
    script.with_suffix(".metadata.json").write_text("{broken", encoding="utf-8")
    info = {
        "path": str(script),
        "name": script.name,
        "framework": "rxinfer",
        "executor": "julia",
    }
    result = _make_skipped_result(
        info, "rxinfer", "model_a", "julia", logging.getLogger("t")
    )
    metadata = result["execution_metadata"]
    assert metadata["metadata_probe"] == "unreadable"
    assert metadata["probe_source"] == str(script.with_suffix(".metadata.json"))


def test_envelope_elapsed_probe_without_log(tmp_path: Path) -> None:
    script = _write_script(tmp_path)
    envelope = _synthesize_rxinfer_envelope(script, success=True)
    assert envelope["elapsed_seconds"] is None
    assert (
        envelope["elapsed_seconds_note"]
        == "rxinfer execution log sidecar missing or unparseable"
    )


def test_envelope_elapsed_from_valid_log(tmp_path: Path) -> None:
    script = _write_script(tmp_path)
    log = script.parent / f"{script.stem}_execution_log.json"
    log.write_text(json.dumps({"elapsed_seconds": 1.25}), encoding="utf-8")
    envelope = _synthesize_rxinfer_envelope(script, success=True)
    assert envelope["elapsed_seconds"] == pytest.approx(1.25)
    assert "elapsed_seconds_note" not in envelope


def test_device_discovery_fallback_receipt(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "model_b_pymdp.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    import gnn.execute.executor as executor_mod

    def _boom() -> list[str]:
        raise RuntimeError("no hardware probe available")

    monkeypatch.setattr(executor_mod, "get_available_hardware", _boom)
    executor = GNNExecutor(output_dir=str(tmp_path / "out"))
    result = executor.execute_gnn_model(str(script), execution_type="pymdp")
    assert result["execution_device"] == "cpu"
    fallback = result["execution_device_fallback"]
    assert fallback.startswith("cpu: device discovery failed")
    assert "no hardware probe available" in fallback


def test_merge_keeps_verified_prior_count_unchanged(tmp_path: Path) -> None:
    scope = tmp_path / "scope_current"
    other = tmp_path / "scope_other"
    scope.mkdir()
    other.mkdir()
    script = other / "m.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    script_identity = {
        "path": str(script.resolve()),
        "sha256": metadata_mod._sha256_file(script),
    }
    receipts = {
        str(other): {
            "input_identity": metadata_mod._execution_input_identity(other),
            "execution_details": [
                {"script_path": str(script), "script_identity": script_identity}
            ],
        }
    }
    results_file = _prior_results_file(tmp_path, scope, "fixed-run", receipts)
    execution_results = _execution_results(scope, "fixed-run")
    _merge_prior_execution_summary(
        execution_results, results_file, logging.getLogger("t")
    )
    merged = execution_results["invocation_receipts"]
    assert str(other) in merged  # verified prior survives the merge
    assert str(scope) in merged


def test_merge_oserror_branch_logs_and_excludes(
    tmp_path: Path, caplog, monkeypatch
) -> None:
    scope = tmp_path / "scope_current"
    other = tmp_path / "scope_other"
    scope.mkdir()
    other.mkdir()
    receipts = {
        str(other): {
            "input_identity": metadata_mod._execution_input_identity(other),
            "execution_details": [],
        }
    }
    results_file = _prior_results_file(tmp_path, scope, "fixed-run", receipts)
    execution_results = _execution_results(scope, "fixed-run")

    original = metadata_mod._execution_input_identity

    def _boom(target):
        if str(target) == str(other):
            raise OSError("identity probe exploded")
        return original(target)

    monkeypatch.setattr(metadata_mod, "_execution_input_identity", _boom)
    with caplog.at_level(logging.DEBUG, logger="gnn.execute.metadata"):
        _merge_prior_execution_summary(
            execution_results, results_file, logging.getLogger("t")
        )
    assert str(other) not in execution_results["invocation_receipts"]
    assert any(
        "excluded from merge" in record.message and "scope_other" in record.message
        for record in caplog.records
    )
    assert str(scope) in execution_results["invocation_receipts"]
