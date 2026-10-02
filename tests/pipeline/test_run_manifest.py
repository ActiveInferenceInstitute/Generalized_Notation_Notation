#!/usr/bin/env python3
"""
Tests for pipeline/run_manifest.py — durable run manifest emission.

Real objects only: every test builds a REAL synthetic run output directory under ``tmp_path``
with real ``N_*_output/`` step subdirectories, real JSON artifact files, and a
real ``pipeline_execution_summary.json``. Assertions use real serialization,
real sha256 checksums (via durable_streams), and a NEGATIVE control that proves
artifact tampering is detected.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.pipeline.durable_streams import (
    read_stream_manifest,
    read_trace,
    replay_trace,
)
from gnn.pipeline.run_manifest import emit_run_manifests, verify_run_manifests


def _build_fake_run(root: Path) -> Path:
    """Create a realistic completed-run output directory under ``root``.

    Returns the run output directory path. It contains two ``N_*_output/`` step
    dirs with JSON artifacts and a ``00_pipeline_summary`` with a real summary.
    """
    run_dir = root / "output"
    run_dir.mkdir(parents=True)

    gnn = run_dir / "3_gnn_output"
    gnn.mkdir()
    (gnn / "gnn_processing_summary.json").write_text(
        json.dumps({"models": 2, "status": "ok"}, indent=2),
        encoding="utf-8",
    )
    (gnn / "format_statistics.json").write_text(
        json.dumps({"markdown": 5}, indent=2),
        encoding="utf-8",
    )

    setup = run_dir / "1_setup_output"
    setup.mkdir()
    (setup / "environment_setup_summary.json").write_text(
        json.dumps({"python": "3.11"}, indent=2),
        encoding="utf-8",
    )

    summary: Dict[str, Any] = {
        "run_hash": "deadbeef",
        "overall_status": "SUCCESS",
        "steps": [
            {"step_number": 1, "script_name": "1_setup.py", "status": "SUCCESS"},
            {"step_number": 3, "script_name": "3_gnn.py", "status": "SUCCESS"},
        ],
    }
    summary_dir = run_dir / "00_pipeline_summary"
    summary_dir.mkdir()
    (summary_dir / "pipeline_execution_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return run_dir


def test_emit_produces_manifests_and_sound_trace(tmp_path: Path) -> None:
    """emit_run_manifests yields >=1 manifest and a trace_integrity_ok trace."""
    run_dir = _build_fake_run(tmp_path)

    result = emit_run_manifests(run_dir)

    # Three JSON artifacts across the two step dirs.
    assert result["stream_count"] == 3
    # Two step records in the summary -> two trace events.
    assert result["trace_event_count"] == 2
    assert result["trace_integrity_ok"] is True

    manifest_dir = Path(result["manifest_dir"])
    assert manifest_dir.is_dir()
    assert (manifest_dir / "index.json").is_file()
    assert (manifest_dir / "execution_trace.json").is_file()

    # verify returns no problems on a freshly emitted, untampered run.
    assert verify_run_manifests(manifest_dir, run_dir) == []


def test_round_trip_validates_clean(tmp_path: Path) -> None:
    """Emitted manifests re-validate clean against the run dir (round-trip)."""
    run_dir = _build_fake_run(tmp_path)
    emit_run_manifests(run_dir, manifest_out=tmp_path / "emitted")

    problems = verify_run_manifests(tmp_path / "emitted", run_dir)
    assert problems == [], f"expected clean round-trip, got: {problems}"


def test_negative_tampered_artifact_is_detected(tmp_path: Path) -> None:
    """NEGATIVE: modifying an artifact after emission fires a checksum mismatch."""
    run_dir = _build_fake_run(tmp_path)
    result = emit_run_manifests(run_dir)
    manifest_dir = Path(result["manifest_dir"])

    # Sanity: clean before tampering.
    assert verify_run_manifests(manifest_dir, run_dir) == []

    # Tamper with one artifact's bytes AFTER emission.
    target = run_dir / "3_gnn_output" / "gnn_processing_summary.json"
    target.write_text(
        json.dumps({"models": 999, "status": "TAMPERED"}, indent=2),
        encoding="utf-8",
    )

    problems = verify_run_manifests(manifest_dir, run_dir)
    assert problems, "expected verification to report the tampered artifact"
    assert any("checksum mismatch" in p for p in problems), problems


def test_negative_missing_artifact_is_detected(tmp_path: Path) -> None:
    """NEGATIVE: deleting an artifact after emission fires a missing-source error."""
    run_dir = _build_fake_run(tmp_path)
    result = emit_run_manifests(run_dir)
    manifest_dir = Path(result["manifest_dir"])

    (run_dir / "1_setup_output" / "environment_setup_summary.json").unlink()

    problems = verify_run_manifests(manifest_dir, run_dir)
    assert problems, "deleting an artifact should produce manifest problems"
    assert any("does not exist" in p for p in problems), problems


def test_determinism_identical_stream_count_and_trace_digest(tmp_path: Path) -> None:
    """Emitting twice over the same run dir is deterministic (count + trace digest)."""
    run_dir = _build_fake_run(tmp_path)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    result_a = emit_run_manifests(run_dir, manifest_out=out_a)
    result_b = emit_run_manifests(run_dir, manifest_out=out_b)

    assert result_a["stream_count"] == result_b["stream_count"]
    assert result_a["trace_event_count"] == result_b["trace_event_count"]

    trace_a = read_trace(out_a / "execution_trace.json")
    trace_b = read_trace(out_b / "execution_trace.json")
    assert replay_trace(trace_a) == replay_trace(trace_b)


def test_trace_falls_back_to_step_dirs_without_summary(tmp_path: Path) -> None:
    """Without a summary, the trace is built from the sorted step directories."""
    run_dir = _build_fake_run(tmp_path)
    # Remove the summary so the directory-fallback path is exercised.
    (run_dir / "00_pipeline_summary" / "pipeline_execution_summary.json").unlink()

    result = emit_run_manifests(run_dir, manifest_out=tmp_path / "nosummary")

    # Two step dirs -> two trace events, still sound, manifests still emitted.
    assert result["trace_event_count"] == 2
    assert result["trace_integrity_ok"] is True
    assert result["stream_count"] == 3
    assert verify_run_manifests(tmp_path / "nosummary", run_dir) == []


def test_missing_run_dir_raises(tmp_path: Path) -> None:
    """A non-existent run output dir raises FileNotFoundError."""
    import pytest

    with pytest.raises(FileNotFoundError):
        emit_run_manifests(tmp_path / "nope")


def test_stream_ids_are_stable_and_unique(tmp_path: Path) -> None:
    """Each emitted manifest carries a unique, path-derived stream id."""
    run_dir = _build_fake_run(tmp_path)
    emit_run_manifests(run_dir, manifest_out=tmp_path / "ids")
    index = json.loads((tmp_path / "ids" / "index.json").read_text(encoding="utf-8"))
    ids = [m["stream_id"] for m in index["manifests"]]
    assert len(ids) == len(set(ids)) == 3
    # Ids are path-derived (readable slug prefix) and injective (hash suffix).
    assert any(i.startswith("3_gnn_output_gnn_processing_summary_json_") for i in ids)


def test_collision_prone_filenames_each_get_a_manifest(tmp_path: Path) -> None:
    """NEGATIVE: distinct files that slugify identically (a.b.json / a-b.json) must
    each get their OWN manifest — no silent overwrite that leaves bytes unbound."""
    run_dir = tmp_path / "output"
    step = run_dir / "3_gnn_output"
    step.mkdir(parents=True)
    (step / "a.b.json").write_text(json.dumps({"x": 1}), encoding="utf-8")
    (step / "a-b.json").write_text(json.dumps({"x": 2}), encoding="utf-8")
    out = tmp_path / "m"
    result = emit_run_manifests(run_dir, manifest_out=out)

    manifest_files = list(out.glob("*.manifest.json"))
    # stream_count must equal the number of manifest files actually on disk.
    assert result["stream_count"] == len(manifest_files) == 2
    assert verify_run_manifests(out, run_dir) == []
    # Tampering EITHER artifact is now detected (both are bound).
    (step / "a-b.json").write_text(json.dumps({"x": 999}), encoding="utf-8")
    assert any("checksum mismatch" in p for p in verify_run_manifests(out, run_dir))


def test_negative_tampered_summary_breaks_trace_binding(tmp_path: Path) -> None:
    """NEGATIVE: flipping a step status in the summary AFTER emission is caught by
    the trace's re-binding to ground truth (not just structural integrity)."""
    run_dir = _build_fake_run(tmp_path)
    out = tmp_path / "bind"
    emit_run_manifests(run_dir, manifest_out=out)
    assert verify_run_manifests(out, run_dir) == []

    summary_path = run_dir / "00_pipeline_summary" / "pipeline_execution_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["steps"][1]["status"] = "FAILED"  # flip 3_gnn.py SUCCESS -> FAILED
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    problems = verify_run_manifests(out, run_dir)
    assert any("re-derived from the live run summary" in p for p in problems)


def _build_fake_run_with_binaries(root: Path) -> Path:
    """Create the fake run plus binary artifacts covering all four formats.

    Calls ``_build_fake_run`` then adds ``graph.png`` / ``anim.gif`` under
    ``3_gnn_output`` and a real ``beliefs.npy`` (numpy) / ``trace.csv`` under
    ``12_execute_output``, so the binary-artifact inventory (Perf#5) has one
    png, gif, npy, and csv to record.
    """
    run_dir = _build_fake_run(root)
    gnn = run_dir / "3_gnn_output"
    (gnn / "graph.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake-png-data")
    (gnn / "anim.gif").write_bytes(b"GIF89a" + b"fake")
    execute = run_dir / "12_execute_output"
    execute.mkdir()
    np.save(execute / "beliefs.npy", np.arange(6, dtype=np.float64))
    (execute / "trace.csv").write_text("t,belief\n0,0.5\n", encoding="utf-8")
    return run_dir


def test_binary_artifacts_recorded_additively(tmp_path: Path) -> None:
    """emit_run_manifests records binaries additively: the JSON inventory and
    stream_count keep their exact meaning, one FILE manifest per binary is
    written, and verification re-validates clean."""
    run_dir = _build_fake_run_with_binaries(tmp_path)
    result = emit_run_manifests(run_dir)
    manifest_dir = Path(result["manifest_dir"])

    index = json.loads((manifest_dir / "index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == "3.2"
    # The JSON inventory is untouched: still exactly the 3 JSON artifacts.
    assert index["stream_count"] == 3
    assert index["binary_count"] == 4
    assert result["binary_count"] == 4

    binary_artifacts = index["binary_artifacts"]
    assert {entry["source"] for entry in binary_artifacts} == {
        "3_gnn_output/anim.gif",
        "3_gnn_output/graph.png",
        "12_execute_output/beliefs.npy",
        "12_execute_output/trace.csv",
    }
    assert {entry["format"] for entry in binary_artifacts} == {
        "png",
        "gif",
        "npy",
        "csv",
    }
    for entry in binary_artifacts:
        assert (manifest_dir / entry["manifest_file"]).is_file()

    # Dtype labels: raw-byte containers (png/gif/npy) are uint8; csv is text.
    dtype_by_format = {
        entry["format"]: read_stream_manifest(
            manifest_dir / entry["manifest_file"]
        ).dtype
        for entry in binary_artifacts
    }
    assert dtype_by_format == {
        "png": "uint8",
        "gif": "uint8",
        "npy": "uint8",
        "csv": "text",
    }

    assert verify_run_manifests(manifest_dir, run_dir) == []


def test_negative_tampered_binary_is_detected(tmp_path: Path) -> None:
    """NEGATIVE: rewriting a binary artifact after emission fires a checksum
    mismatch naming the binary's relative source path."""
    run_dir = _build_fake_run_with_binaries(tmp_path)
    result = emit_run_manifests(run_dir)
    manifest_dir = Path(result["manifest_dir"])

    # Sanity: clean before tampering.
    assert verify_run_manifests(manifest_dir, run_dir) == []

    (run_dir / "3_gnn_output" / "graph.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"tampered-png-data"
    )

    problems = verify_run_manifests(manifest_dir, run_dir)
    assert problems, "expected verification to report the tampered binary"
    assert any("checksum mismatch for 3_gnn_output/graph.png" in p for p in problems), (
        problems
    )


def test_negative_deleted_binary_is_detected(tmp_path: Path) -> None:
    """NEGATIVE: deleting a binary artifact after emission fires both the
    missing-source error and the binary inventory difference."""
    run_dir = _build_fake_run_with_binaries(tmp_path)
    result = emit_run_manifests(run_dir)
    manifest_dir = Path(result["manifest_dir"])

    (run_dir / "12_execute_output" / "trace.csv").unlink()

    problems = verify_run_manifests(manifest_dir, run_dir)
    assert problems, "deleting a binary should produce manifest problems"
    assert any("source file does not exist" in p for p in problems), problems
    assert any("Binary artifact inventory differs" in p for p in problems), problems


def test_legacy_3_1_index_still_verifies_without_binaries(tmp_path: Path) -> None:
    """A legacy ``"3.1"`` index (binary keys stripped) still re-validates clean
    when the run contains no binary artifacts."""
    run_dir = _build_fake_run(tmp_path)
    result = emit_run_manifests(run_dir)
    manifest_dir = Path(result["manifest_dir"])

    index_path = manifest_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["schema_version"] = "3.1"
    index.pop("binary_count")
    index.pop("binary_artifacts")
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    assert verify_run_manifests(manifest_dir, run_dir) == []


def test_legacy_3_1_index_reports_unrecorded_binaries(tmp_path: Path) -> None:
    """A legacy ``"3.1"`` index with binary artifacts on disk reports them as
    unrecorded rather than silently ignoring them."""
    run_dir = _build_fake_run(tmp_path)
    result = emit_run_manifests(run_dir)
    manifest_dir = Path(result["manifest_dir"])

    index_path = manifest_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["schema_version"] = "3.1"
    index.pop("binary_count")
    index.pop("binary_artifacts")
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    # A binary lands on disk after the legacy downgrade.
    (run_dir / "3_gnn_output" / "late.png").write_bytes(b"\x89PNG\r\n\x1a\nlate")

    problems = verify_run_manifests(manifest_dir, run_dir)
    assert any("Binary artifacts not recorded in the index" in p for p in problems), (
        problems
    )


def test_binary_count_mismatch_detected(tmp_path: Path) -> None:
    """NEGATIVE: hand-editing binary_count in the index breaks verification."""
    run_dir = _build_fake_run_with_binaries(tmp_path)
    result = emit_run_manifests(run_dir)
    manifest_dir = Path(result["manifest_dir"])

    index_path = manifest_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["binary_count"] = 99
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    problems = verify_run_manifests(manifest_dir, run_dir)
    assert any(
        "binary_count does not match binary manifest inventory" in p for p in problems
    ), problems


def test_emit_determinism_with_binaries(tmp_path: Path) -> None:
    """Emitting over two separate copies of the same fake run is deterministic:
    equal counts, identical binary stream ids/sources, identical trace digests."""
    run_a = _build_fake_run_with_binaries(tmp_path / "a")
    run_b = _build_fake_run_with_binaries(tmp_path / "b")
    out_a = tmp_path / "manifests_a"
    out_b = tmp_path / "manifests_b"
    result_a = emit_run_manifests(run_a, manifest_out=out_a)
    result_b = emit_run_manifests(run_b, manifest_out=out_b)

    assert result_a["stream_count"] == result_b["stream_count"]
    assert result_a["binary_count"] == result_b["binary_count"]

    index_a = json.loads((out_a / "index.json").read_text(encoding="utf-8"))
    index_b = json.loads((out_b / "index.json").read_text(encoding="utf-8"))
    streams_a = {
        (entry["stream_id"], entry["source"]) for entry in index_a["binary_artifacts"]
    }
    streams_b = {
        (entry["stream_id"], entry["source"]) for entry in index_b["binary_artifacts"]
    }
    assert streams_a == streams_b

    trace_a = read_trace(out_a / "execution_trace.json")
    trace_b = read_trace(out_b / "execution_trace.json")
    assert replay_trace(trace_a) == replay_trace(trace_b)
