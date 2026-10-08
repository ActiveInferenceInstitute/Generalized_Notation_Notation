"""Invocation selection, output ownership, and hard worker deadline regressions."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import pytest

from gnn.pipeline.artifact_ownership import (
    changed_artifacts,
    snapshot_files,
    verify_owned_artifacts,
)
from gnn.pipeline.output_lease import OutputLease, OutputLeaseError
from gnn.pipeline.run_context import build_run_context, current_run_context
from gnn.utils.pipeline_orchestration.execution_utils import execute_command_streaming

pytestmark = pytest.mark.pipeline


def _context(tmp_path: Path, config: dict | None = None):
    source = tmp_path / "input"
    for folder in ("first", "second", "docs", "archived_gnn_files"):
        (source / folder).mkdir(parents=True, exist_ok=True)
    (source / "first" / "model.md").write_bytes(b"first source\n")
    (source / "second" / "model.md").write_bytes(b"second source\n")
    (source / "docs" / "README.md").write_text("documentation")
    (source / "archived_gnn_files" / "old.md").write_text("old")
    return build_run_context(
        source, tmp_path / "output", "test-run", [3, 9, 13, 16, 20, 24], config or {}
    )


def _activate(context, monkeypatch: pytest.MonkeyPatch) -> None:
    path = Path(context.output_root) / "context.json"
    context.write(path)
    monkeypatch.setenv("GNN_RUN_CONTEXT_FILE", str(path))
    monkeypatch.setenv("GNN_RUN_ID", context.run_id)


def test_duplicate_stems_keep_distinct_original_identities_and_exact_bytes(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    assert len(context.models) == 2
    assert len({model.model_id for model in context.models}) == 2
    assert len({model.artifact_stem for model in context.models}) == 2
    view = context.input_view(3)
    for model in context.models:
        data = (view / f"{model.artifact_stem}.md").read_bytes()
        assert data == Path(model.source_path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == model.sha256
    assert set(context.exclusions) == {
        ("docs/README.md", "documentation"),
        ("archived_gnn_files/old.md", "archived"),
    }


def test_matrix_selection_is_exact_and_run_consumer_keeps_corpus(
    tmp_path: Path,
) -> None:
    context = _context(
        tmp_path,
        {
            "testing_matrix": {
                "enabled": True,
                "default_steps": [],
                "folders": {"first": [3, 9], "second": [16]},
            }
        },
    )
    assert [model.relative_path for model in context.selected_models(3)] == [
        "first/model.md"
    ]
    assert [model.relative_path for model in context.selected_models(16)] == [
        "second/model.md"
    ]
    assert not context.selected_models(20)
    assert len(context.selected_models(13)) == 2
    assert len(context.selected_models(24)) == 2
    assert not list(context.input_view(20).iterdir())


def test_original_source_change_fails_before_new_view(tmp_path: Path) -> None:
    context = _context(tmp_path)
    Path(context.models[0].source_path).write_text("mutated")
    with pytest.raises(ValueError, match="changed after run planning"):
        context.input_view(3)
    with pytest.raises(ValueError, match="changed after run planning"):
        context.verify_sources()


def test_nonrecursive_selection_excludes_nested_models_before_staging(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input"
    (source / "nested").mkdir(parents=True)
    (source / "model.md").write_bytes(b"root model")
    (source / "nested/model.md").write_bytes(b"nested model")
    context = build_run_context(
        source, tmp_path / "output", "nonrecursive", [3], {}, recursive=False
    )
    assert [model.relative_path for model in context.models] == ["model.md"]
    assert (context.input_view(3) / "model.md").read_bytes() == b"root model"
    assert ("nested", "non_recursive_directory") in context.exclusions


def test_configuration_is_an_immutable_snapshot(tmp_path: Path) -> None:
    supplied = {"llm": {"max_files": None}}
    context = _context(tmp_path, supplied)
    supplied["llm"]["max_files"] = 1
    copy = context.input_config
    copy["llm"]["max_files"] = 2
    assert context.input_config["llm"]["max_files"] is None


def test_repeated_preliminary_snapshots_never_overwrite_prior_bytes(
    tmp_path: Path,
) -> None:
    from datetime import datetime

    import gnn.main as orchestrator

    summary = {
        "run_id": "immutable-snapshots",
        "start_time": datetime.now().isoformat(),
        "overall_status": "RUNNING",
        "steps": [],
    }
    output = tmp_path / "output"
    logger = logging.getLogger(__name__)
    orchestrator._write_preliminary_pipeline_summary(summary, output, logger)
    current = output / "00_pipeline_summary/current_summary.json"
    first_snapshot = Path(json.loads(current.read_text())["snapshot_path"])
    first_bytes = first_snapshot.read_bytes()
    orchestrator._write_preliminary_pipeline_summary(summary, output, logger)
    second_snapshot = Path(json.loads(current.read_text())["snapshot_path"])
    assert second_snapshot != first_snapshot
    assert first_snapshot.read_bytes() == first_bytes
    assert json.loads(second_snapshot.read_text())["overall_status"] == "RUNNING"


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "600"])
def test_invalid_total_deadline_is_rejected(tmp_path: Path, value) -> None:
    with pytest.raises(ValueError, match="positive number or null"):
        _context(tmp_path, {"pipeline": {"timeout": {"total": value}}})


@pytest.mark.parametrize("value", [True, "600", float("nan"), float("inf")])
def test_deserialized_context_rejects_invalid_absolute_deadline(
    tmp_path: Path, value
) -> None:
    from gnn.pipeline.run_context import RunContext

    context = _context(tmp_path)
    path = tmp_path / "run_context.json"
    context.write(path)
    payload = json.loads(path.read_text())
    payload["deadline_monotonic"] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="deadline"):
        RunContext.read(path)


def test_explicit_total_deadline_bounds_local_budgets(tmp_path: Path) -> None:
    context = _context(tmp_path, {"pipeline": {"timeout": {"total": 0.2}}})
    assert 0 < context.bounded_timeout(300) <= 0.2
    time.sleep(0.21)
    assert context.remaining_seconds() == 0


def test_context_identity_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    _activate(context, monkeypatch)
    monkeypatch.setenv("GNN_RUN_ID", "another-run")
    with pytest.raises(ValueError, match="identity"):
        current_run_context()


def test_shared_source_discovery_keeps_collision_safe_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.pipeline.run_context import model_provenance, selected_model_sources

    context = _context(tmp_path)
    _activate(context, monkeypatch)
    paths = selected_model_sources(Path(context.input_root), 13)
    assert len({path.stem for path in paths}) == 2
    assert {model_provenance(path, 13)["model_id"] for path in paths} == {
        model.model_id for model in context.models
    }
    assert all(
        path.read_bytes()
        == Path(model_provenance(path, 13)["source_path"]).read_bytes()
        for path in paths
    )


def test_stale_final_summary_never_satisfies_current_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.intelligent_analysis.processor import resolve_pipeline_summary_path
    from gnn.website.collection import _load_pipeline_summary

    context = _context(tmp_path)
    _activate(context, monkeypatch)
    summaries = Path(context.output_root) / "00_pipeline_summary"
    summaries.mkdir()
    (summaries / "pipeline_execution_summary.json").write_text(
        json.dumps({"run_id": "old-run"})
    )
    with pytest.raises(ValueError, match="unavailable"):
        resolve_pipeline_summary_path(Path(context.output_root))
    assert _load_pipeline_summary(Path(context.output_root)) == {}
    (summaries / "current_summary.json").write_text(
        json.dumps({"run_id": context.run_id, "overall_status": "RUNNING"})
    )
    assert (
        resolve_pipeline_summary_path(Path(context.output_root)).name
        == "current_summary.json"
    )
    assert (
        _load_pipeline_summary(Path(context.output_root))["overall_status"] == "RUNNING"
    )


def test_current_inventory_excludes_stale_artifacts_and_detects_changes(
    tmp_path: Path,
) -> None:
    from gnn.pipeline.run_manifest import emit_run_manifests, verify_run_manifests

    root = tmp_path / "output"
    step = root / "3_gnn_output"
    step.mkdir(parents=True)
    (step / "old.json").write_text("{}")
    before = snapshot_files(step)
    artifact = step / "new.json"
    artifact.write_text('{"new": true}')
    summary = {
        "run_id": "test-run",
        "artifact_inventory_contract": "current-run-v1",
        "steps": [
            {"status": "SUCCESS", "artifacts": changed_artifacts(step, root, before)}
        ],
    }
    summaries = root / "00_pipeline_summary"
    summaries.mkdir()
    (summaries / "pipeline_execution_summary.json").write_text(json.dumps(summary))
    emission = emit_run_manifests(root)
    index = json.loads((Path(emission["manifest_dir"]) / "index.json").read_text())
    assert [entry["source"] for entry in index["manifests"]] == [
        "3_gnn_output/new.json"
    ]
    assert verify_run_manifests(emission["manifest_dir"], root) == []
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_owned_artifacts(root, summary)


def test_inventory_detects_stat_preserving_byte_changes(tmp_path: Path) -> None:
    artifact = tmp_path / "same.json"
    artifact.write_bytes(b"old")
    before = snapshot_files(tmp_path)
    original = artifact.stat()
    artifact.write_bytes(b"new")
    os.utime(artifact, ns=(original.st_atime_ns, original.st_mtime_ns))
    records = changed_artifacts(tmp_path, tmp_path, before)
    assert records == [
        {"path": "same.json", "sha256": hashlib.sha256(b"new").hexdigest()}
    ]


def test_parser_receipt_projects_only_selected_consumer_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.pipeline.parse_receipt import selected_parse_receipt

    source = tmp_path / "input"
    for folder in ("first", "second"):
        (source / folder).mkdir(parents=True)
        (source / folder / "model.md").write_text(folder)
    context = build_run_context(
        source,
        tmp_path / "output",
        "parser-run",
        [3, 6],
        {
            "testing_matrix": {
                "enabled": True,
                "folders": {"first": [3, 6], "second": [3]},
                "default_steps": [],
            }
        },
    )
    _activate(context, monkeypatch)
    producer_view = context.input_view(3)
    receipt = {
        "run_id": context.run_id,
        "processed_files": [
            {
                "model_id": model.model_id,
                "file_path": str(producer_view / f"{model.artifact_stem}.md"),
                "parse_success": False,
            }
            for model in context.models
        ],
    }
    projected = selected_parse_receipt(receipt, 6)
    assert len(projected["processed_files"]) == 1
    record = projected["processed_files"][0]
    assert Path(record["file_path"]).stem == context.selected_models(6)[0].artifact_stem
    assert record["producer_file_path"] != record["file_path"]
    receipt["processed_files"] = []
    with pytest.raises(ValueError, match="no current-run receipt"):
        selected_parse_receipt(receipt, 6)


def test_current_parser_failure_does_not_claim_complete_selected_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.processing.multi_format_processor import process_gnn_multi_format

    source = tmp_path / "input"
    source.mkdir()
    exemplar = Path(__file__).parents[2] / "input/gnn_files/discrete/simple_mdp.md"
    (source / "valid.md").write_bytes(exemplar.read_bytes())
    (source / "broken.json").write_text("{ invalid JSON")
    context = build_run_context(source, tmp_path / "output", "partial-parse", [3], {})
    _activate(context, monkeypatch)
    assert not process_gnn_multi_format(
        context.input_view(3),
        Path(context.output_root),
        logging.getLogger(__name__),
        serialize_preset="minimal",
    )
    receipt = json.loads(
        (
            Path(context.output_root) / "3_gnn_output/gnn_processing_results.json"
        ).read_text()
    )
    assert receipt["summary"]["successful_parses"] == 1
    assert receipt["summary"]["failed_parses"] == 1
    assert {record["model_id"] for record in receipt["processed_files"]} == {
        model.model_id for model in context.models
    }


def test_deserialized_context_rejects_a_source_symlink_escape(tmp_path: Path) -> None:
    from gnn.pipeline.run_context import RunContext

    context = _context(tmp_path)
    path = tmp_path / "run_context.json"
    context.write(path)
    source = Path(context.models[0].source_path)
    outside = tmp_path / "outside.md"
    outside.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes input root"):
        RunContext.read(path)


def test_output_lease_prevents_concurrent_ownership(tmp_path: Path) -> None:
    with OutputLease(tmp_path, "first"):
        with pytest.raises(OutputLeaseError, match="owned by another run"):
            with OutputLease(tmp_path, "second"):
                pass
    with OutputLease(tmp_path, "next"):
        pass


@pytest.mark.needs_posix
@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_output_lock_refuses_external_links_without_clobbering(
    tmp_path: Path, link_kind: str
) -> None:
    outside = tmp_path / "external-sentinel.txt"
    outside.write_bytes(b"must stay byte-identical")
    output = tmp_path / "output"
    output.mkdir()
    lock = output / ".gnn_run.lock"
    if link_kind == "symlink":
        lock.symlink_to(outside)
    else:
        os.link(outside, lock)
    with pytest.raises(OutputLeaseError, match="symlink|single-link"):
        with OutputLease(output, "unsafe-output"):
            pytest.fail("External-linked output lock was accepted")
    assert outside.read_bytes() == b"must stay byte-identical"


@pytest.mark.needs_posix
def test_output_lock_nonregular_file_is_rejected_without_blocking(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    os.mkfifo(output / ".gnn_run.lock")
    with pytest.raises(OutputLeaseError, match="regular single-link"):
        with OutputLease(output, "unsafe-output"):
            pytest.fail("Nonregular output lock was accepted")


@pytest.mark.needs_posix
def test_hard_deadline_kills_worker_child_and_grandchild(tmp_path: Path) -> None:
    import psutil

    pid_file = tmp_path / "grandchild.pid"
    child_file = tmp_path / "child.pid"
    grandchild = "import time; time.sleep(60)"
    child = f"import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c',{grandchild!r}]); pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(60)"
    worker = f"import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c',{child!r}]); pathlib.Path({str(child_file)!r}).write_text(str(p.pid)); time.sleep(60)"
    started = time.monotonic()
    result = execute_command_streaming(
        [sys.executable, "-c", worker],
        timeout=0.75,
        print_stdout=False,
        print_stderr=False,
    )
    assert result["status"] == "TIMEOUT"
    assert result["force_killed"] is True
    assert time.monotonic() - started < 5
    for file in (child_file, pid_file):
        pid = int(file.read_text())
        assert (
            not psutil.pid_exists(pid)
            or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        )


@pytest.mark.needs_posix
def test_invocation_deadline_includes_real_child_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psutil

    context = _context(tmp_path, {"pipeline": {"timeout": {"total": 0.9}}})
    _activate(context, monkeypatch)
    pid_file = tmp_path / "budget-child.pid"
    child = "import time; time.sleep(60)"
    worker = f"import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c',{child!r}]); pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(60)"
    result = execute_command_streaming(
        [sys.executable, "-c", worker],
        timeout=60,
        print_stdout=False,
        print_stderr=False,
    )
    assert result["status"] == "TIMEOUT"
    assert result["stop_reason"] == "total_timeout"
    assert result["cleanup_verified"] and result["streams_drained"]
    assert 0 < result["cleanup_timeout_seconds"] < 0.225
    assert result["cleanup_completed_monotonic"] <= context.deadline_monotonic
    assert result["work_deadline_monotonic"] < context.deadline_monotonic
    pid = int(pid_file.read_text())
    assert (
        not psutil.pid_exists(pid)
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    )


@pytest.mark.needs_posix
def test_normal_worker_exit_cleans_detached_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    import psutil

    from gnn.utils.runtime_safety.process_tree import DescendantTracker

    pid_file = tmp_path / "daemon.pid"
    observed = tmp_path / "observed.flag"
    original_start = DescendantTracker.start

    def start_with_observation_handshake(self):
        original_start(self)

        def signal_observed():
            until = time.monotonic() + 2
            while time.monotonic() < until:
                if self.observed_processes:
                    observed.write_text("observed")
                    return
                time.sleep(0.01)

        threading.Thread(target=signal_observed, daemon=True).start()

    monkeypatch.setattr(DescendantTracker, "start", start_with_observation_handshake)
    daemon = "import time; time.sleep(60)"
    worker = f"import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c',{daemon!r}],start_new_session=True); pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); flag=pathlib.Path({str(observed)!r})\nwhile not flag.exists(): time.sleep(0.01)"
    result = execute_command_streaming(
        [sys.executable, "-c", worker],
        timeout=3,
        print_stdout=False,
        print_stderr=False,
    )
    witnesses = {}
    if result["status"] != "SUCCESS":
        for label, path in (("daemon_pid", pid_file), ("observation_flag", observed)):
            try:
                witnesses[label] = {
                    "path": str(path),
                    "present": True,
                    "text": path.read_text(encoding="utf-8", errors="replace"),
                }
            except OSError as error:
                witnesses[label] = {
                    "path": str(path),
                    "present": False if isinstance(error, FileNotFoundError) else None,
                    "read_error_type": type(error).__name__,
                    "read_error": str(error),
                }
    assert result["status"] == "SUCCESS", json.dumps(
        {"supervisor": result, "owned_witnesses": witnesses}, indent=2, default=str
    )
    assert result["observed_descendants"] >= 1
    pid = int(pid_file.read_text())
    assert (
        not psutil.pid_exists(pid)
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    )


@pytest.mark.needs_posix
def test_observation_failure_still_kills_child_before_releasing_pipeline_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psutil

    import gnn.main as orchestrator
    from gnn.utils.arguments.pipeline_arguments import PipelineArguments
    from gnn.utils.runtime_safety.process_tree import DescendantTracker

    source = tmp_path / "input"
    source.mkdir()
    (source / "model.md").write_text("## ModelName\nModel\n")
    child_pid = tmp_path / "child.pid"
    worker_pid = tmp_path / "worker.pid"
    child = "import time; time.sleep(60)"
    worker = f"import subprocess,sys,time,pathlib,os; pathlib.Path({str(worker_pid)!r}).write_text(str(os.getpid())); p=subprocess.Popen([sys.executable,'-c',{child!r}]); pathlib.Path({str(child_pid)!r}).write_text(str(p.pid)); time.sleep(60)"
    original_stop = DescendantTracker.stop

    def failed_stop(self, timeout=0.1):
        original_stop(self, timeout=timeout)
        raise RuntimeError("Injected observation failure")

    monkeypatch.setattr(DescendantTracker, "stop", failed_stop)
    monkeypatch.setattr(
        "gnn.utils.arguments.arg_parsing.build_step_command_args",
        lambda *args, **kwargs: [sys.executable, "-c", worker],
    )
    args = PipelineArguments(
        target_dir=source, output_dir=tmp_path / "output", only_steps="3"
    )
    assert orchestrator.main(args, {"pipeline": {"timeout": {"total": 0.75}}}) == 1
    for path in (worker_pid, child_pid):
        pid = int(path.read_text())
        assert (
            not psutil.pid_exists(pid)
            or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        )
    with OutputLease(args.output_dir, "after-failed-cleanup"):
        pass
    receipt = json.loads(
        (
            args.output_dir / "00_pipeline_summary/pipeline_execution_summary.json"
        ).read_text()
    )
    assert receipt["overall_status"] == "FAILED"
    assert receipt["steps"][0]["cleanup_verified"] is False


def test_explicit_model_scope_does_not_union_other_receipt_models() -> None:
    from gnn.analysis.processor import (
        _filter_execution_summary,
        _scope_from_execution_summary,
    )

    receipt = {
        "execution_details": [
            {"framework": "pymdp", "model_name": "wanted", "success": True},
            {"framework": "pymdp", "model_name": "other", "success": True},
        ]
    }
    scope = _scope_from_execution_summary(receipt, target_model_names={"wanted"})
    assert scope["models"] == {"wanted"}
    assert (
        len(_filter_execution_summary(receipt, None, {"wanted"})["execution_details"])
        == 1
    )


def test_standalone_matrix_empty_selection_does_not_dispatch_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.pipeline.summary_wiring import execute_pipeline_step
    from gnn.utils.arguments.pipeline_arguments import PipelineArguments

    source = tmp_path / "input"
    (source / "excluded").mkdir(parents=True)
    (source / "excluded/model.md").write_text("excluded")
    monkeypatch.delenv("GNN_RUN_CONTEXT_FILE", raising=False)
    args = PipelineArguments(target_dir=source, output_dir=tmp_path / "output")
    receipt = execute_pipeline_step(
        "3_gnn.py",
        args,
        logging.getLogger(__name__),
        pipeline_config={"testing_matrix": {"enabled": True, "default_steps": []}},
    )
    assert receipt["status"] == "SKIPPED"
    assert receipt["skip_reason"] == "no_selected_models"
    assert not args.output_dir.exists()


def test_standalone_matrix_folders_share_one_logical_step_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.pipeline.summary_wiring import execute_pipeline_step
    from gnn.utils.arguments.pipeline_arguments import PipelineArguments

    source = tmp_path / "input"
    for folder in ("first", "second", "third"):
        (source / folder).mkdir(parents=True)
    monkeypatch.delenv("GNN_RUN_CONTEXT_FILE", raising=False)
    monkeypatch.setattr(
        "gnn.pipeline.step_timeouts.get_step_timeout", lambda *args, **kwargs: 0.3
    )
    monkeypatch.setattr(
        "gnn.utils.arguments.arg_parsing.build_step_command_args",
        lambda *args, **kwargs: [sys.executable, "-c", "import time; time.sleep(0.2)"],
    )
    args = PipelineArguments(target_dir=source, output_dir=tmp_path / "output")
    started = time.monotonic()
    receipt = execute_pipeline_step(
        "3_gnn.py",
        args,
        logging.getLogger(__name__),
        pipeline_config={"testing_matrix": {"enabled": True, "default_steps": [3]}},
    )
    assert receipt["status"] == "FAILED"
    assert receipt["stop_reason"] == "step_timeout"
    assert 1 <= len(receipt["folder_results"]) < 3
    assert str(source / "third") in receipt["unfinished_folders"]
    assert time.monotonic() - started < 2


@pytest.mark.parametrize("mode", ["serial", "parallel", "consolidated"])
def test_top_level_deadline_fails_honestly_and_restores_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    import gnn.main as orchestrator
    from gnn.utils.arguments.pipeline_arguments import PipelineArguments

    source = tmp_path / "input"
    source.mkdir()
    (source / "model.md").write_text(
        "## ModelName\nDeadline model\n## StateSpaceBlock\ns[2,1,type=hidden]\n"
    )
    monkeypatch.setenv("GNN_RUN_ID", "caller-run")
    monkeypatch.delenv("GNN_RUN_CONTEXT_FILE", raising=False)
    args = PipelineArguments(
        target_dir=source,
        output_dir=tmp_path / "output",
        only_steps="3,5",
        parallel=mode == "parallel",
        consolidated_steps=mode == "consolidated",
    )
    started = time.monotonic()
    assert orchestrator.main(args, {"pipeline": {"timeout": {"total": 0.03}}}) == 1
    assert time.monotonic() - started < 5
    receipt = json.loads(
        (
            args.output_dir / "00_pipeline_summary" / "pipeline_execution_summary.json"
        ).read_text()
    )
    assert receipt["overall_status"] == "FAILED"
    assert receipt["stop_reason"] == "total_timeout"
    assert os.environ["GNN_RUN_ID"] == "caller-run"
    assert "GNN_RUN_CONTEXT_FILE" not in os.environ
