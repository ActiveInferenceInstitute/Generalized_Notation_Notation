"""Terminal claims bind verified evidence and fail together after publication errors."""

import hashlib
import json
import logging
import os
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

import pytest

import gnn.main as orchestrator
from gnn.pipeline._io import atomic_write_text
from gnn.pipeline.run_manifest import verify_run_manifests
from gnn.pipeline.run_session import load_session, status_report
from gnn.utils.arguments.pipeline_arguments import PipelineArguments

pytestmark = pytest.mark.pipeline


def test_dashboard_preserves_data_without_creating_injected_html_nodes(
    tmp_path: Path,
) -> None:
    import gnn.pipeline.summary_wiring as wiring

    class ScriptInventory(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.scripts: list[dict[str, str | None]] = []

        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            if tag == "script":
                self.scripts.append(dict(attrs))

    payload = {
        "run_id": "review-run",
        "overall_status": "FAILED",
        "steps": [
            {
                "stdout": '</script><script id="injected">alert(1)</script><script><!--&>\u2028\u2029'
            }
        ],
    }
    path = tmp_path / "pipeline_execution_summary.json"
    wiring._write_performance_dashboard(path, payload, logging.getLogger(__name__))
    rendered = (tmp_path / "performance_dashboard.html").read_text()
    template = (
        Path(wiring.__file__).parent / "performance_dashboard.template.html"
    ).read_text()
    expected = ScriptInventory()
    expected.feed(template)
    actual = ScriptInventory()
    actual.feed(rendered)
    assert actual.scripts == expected.scripts
    encoded = rendered.split("const rawData = ", 1)[1]
    decoded, end = json.JSONDecoder().raw_decode(encoded)
    assert decoded == payload and encoded[end:].startswith(";")
    assert "<" not in encoded[:end] and "&" not in encoded[:end]
    assert "\u2028" not in encoded[:end] and "\u2029" not in encoded[:end]


@pytest.fixture
def parsed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PipelineArguments:
    source = (
        Path(__file__).resolve().parents[2] / "input/gnn_files/discrete/simple_mdp.md"
    )
    target = tmp_path / "input"
    target.mkdir()
    (target / "simple_mdp.md").write_bytes(source.read_bytes())
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GNN_RUN_ID", raising=False)
    return PipelineArguments(
        target_dir=target,
        output_dir=tmp_path / "output",
        only_steps="3",
    )


def _summary(
    args: PipelineArguments, filename: str = "pipeline_execution_summary.json"
) -> dict:
    return json.loads((args.output_dir / "00_pipeline_summary" / filename).read_text())


def test_success_is_published_after_candidate_verification(
    parsed_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gnn.pipeline.run_manifest as manifests

    real_verify = manifests.verify_run_manifests
    inspected: list[Path] = []

    def inspect_candidate(
        manifest_dir: Path, output_dir: Path, *, summary_path: Path
    ) -> list[str]:
        # The public current snapshot remains preliminary while candidate evidence
        # is verified. No canonical final SUCCESS receipt has been published.
        assert (
            _summary(parsed_run, "current_summary.json")["overall_status"] == "RUNNING"
        )
        assert not (
            output_dir / "00_pipeline_summary/pipeline_execution_summary.json"
        ).exists()
        candidate = json.loads(summary_path.read_text())
        assert candidate["overall_status"] == "SUCCESS"
        session = load_session(summary_path.parent / "run_session.json")
        assert session.final_status == "SUCCESS"
        inspected.append(summary_path)
        return real_verify(manifest_dir, output_dir, summary_path=summary_path)

    monkeypatch.setattr(manifests, "verify_run_manifests", inspect_candidate)
    assert orchestrator.main(parsed_run) == 0
    assert len(inspected) == 1
    summary = _summary(parsed_run)
    assert _summary(parsed_run, "current_summary.json") == summary
    session_path = parsed_run.output_dir / "00_pipeline_summary/run_session.json"
    assert (
        summary["run_session_sha256"]
        == hashlib.sha256(session_path.read_bytes()).hexdigest()
    )
    session = load_session(session_path)
    assert status_report(session)["done"]
    assert session.evidence_integrity == summary["evidence_integrity"]
    assert (
        real_verify(parsed_run.output_dir / "v3_run_manifest", parsed_run.output_dir)
        == []
    )
    # Replacing a terminal checkpoint invalidates the durable evidence join.
    corrupted = json.loads(session_path.read_text())
    corrupted["final_status"] = "FAILED"
    atomic_write_text(session_path, json.dumps(corrupted))
    assert any(
        "session checksum" in problem
        for problem in real_verify(
            parsed_run.output_dir / "v3_run_manifest", parsed_run.output_dir
        )
    )


@pytest.mark.parametrize("failure_phase", ["manifest", "session", "report"])
def test_finalization_failure_invalidates_all_terminal_claims(
    parsed_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch, failure_phase: str
) -> None:
    # Leave a real older successful report/session/manifests in the same output
    # root; the second invocation must not present those as its successful result.
    assert orchestrator.main(parsed_run) == 0
    prior_id = _summary(parsed_run)["run_id"]
    if failure_phase == "manifest":
        monkeypatch.setattr(
            "gnn.pipeline.run_manifest.verify_run_manifests",
            lambda *a, **k: ["injected manifest failure"],
        )
    elif failure_phase == "session":
        real_finalize = orchestrator._finalize_pipeline_summary

        def corrupt_checkpoint(summary: dict) -> None:
            real_finalize(summary)
            session_path = (
                parsed_run.output_dir / "00_pipeline_summary/run_session.json"
            )
            value = json.loads(session_path.read_text())
            value["units"][0]["artifact_hashes"] = {"forged.json": "0" * 64}
            atomic_write_text(session_path, json.dumps(value))

        monkeypatch.setattr(
            orchestrator, "_finalize_pipeline_summary", corrupt_checkpoint
        )
    else:

        def broken_report(*args: object, **kwargs: object) -> None:
            raise ValueError("injected report renderer failure")

        monkeypatch.setattr(orchestrator, "_write_final_pipeline_report", broken_report)

    assert orchestrator.main(parsed_run) == 1
    summary = _summary(parsed_run)
    assert summary["run_id"] != prior_id
    assert summary["overall_status"] == "FAILED"
    assert summary["evidence_integrity"]["status"] == "failed"
    assert _summary(parsed_run, "current_summary.json") == summary
    report = (parsed_run.output_dir / "PIPELINE_REPORT.md").read_text()
    assert "FAILED" in report
    assert "🟢 SUCCESS" not in report
    assert summary["error"] in report
    assert "No errors recorded" not in report
    dashboard = (
        parsed_run.output_dir / "00_pipeline_summary/performance_dashboard.html"
    ).read_text()
    assert '"overall_status": "FAILED"' in dashboard
    session_path = parsed_run.output_dir / "00_pipeline_summary/run_session.json"
    session = load_session(session_path)
    assert session.session_id == summary["run_id"]
    assert session.final_status == "FAILED"
    assert not status_report(session)["done"]
    assert (
        summary["run_session_sha256"]
        == hashlib.sha256(session_path.read_bytes()).hexdigest()
    )
    index = json.loads(
        (parsed_run.output_dir / "v3_run_manifest/index.json").read_text()
    )
    assert index["verification_status"] == "FAILED"
    assert index["provenance"]["run_id"] == summary["run_id"]
    assert not index["trace_integrity_ok"]
    assert verify_run_manifests(
        parsed_run.output_dir / "v3_run_manifest", parsed_run.output_dir
    )
    history = json.loads(
        (parsed_run.output_dir / "00_pipeline_summary/.history/index.json").read_text()
    )
    assert (
        Path(history[summary["run_hash"]]["summary_path"])
        == parsed_run.output_dir / "00_pipeline_summary/pipeline_execution_summary.json"
    )


def test_session_checkpoint_failure_is_fatal_for_current_invocation(
    parsed_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_open(*args: object, **kwargs: object) -> None:
        raise OSError("checkpoint unavailable")

    monkeypatch.setattr("gnn.pipeline.run_session_wiring.open_run_session", fail_open)
    assert orchestrator.main(parsed_run) == 1
    summary = _summary(parsed_run)
    assert summary["overall_status"] == "FAILED"
    assert summary["steps"] == []
    assert "checkpoint unavailable" in summary["error"]
    assert (
        load_session(
            parsed_run.output_dir / "00_pipeline_summary/run_session.json"
        ).final_status
        == "FAILED"
    )


def test_elapsed_receipt_ignores_host_wall_clock_adjustment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gnn.pipeline.summary_wiring as wiring

    monkeypatch.setattr(wiring.time, "monotonic", lambda: 100.0)
    assert (
        wiring._elapsed_run_duration(
            {"start_time": "2040-01-01T00:00:00", "start_monotonic": 95.0},
            datetime(2030, 1, 1),
        )
        == 5.0
    )


@pytest.mark.needs_posix
def test_manifest_archive_rejects_external_history_symlink(tmp_path: Path) -> None:
    from gnn.pipeline.finalization import _replace_manifest_directory

    output = tmp_path / "output"
    old = output / "v3_run_manifest"
    old.mkdir(parents=True)
    (old / "index.json").write_bytes(b"prior manifest")
    staged = output / "candidate"
    staged.mkdir()
    (staged / "index.json").write_bytes(b"candidate manifest")
    outside = tmp_path / "external-history"
    outside.mkdir()
    summary_dir = output / "00_pipeline_summary"
    summary_dir.mkdir()
    (summary_dir / ".history").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="history directory escapes"):
        _replace_manifest_directory(output, staged)
    assert (old / "index.json").read_bytes() == b"prior manifest"
    assert (staged / "index.json").read_bytes() == b"candidate manifest"
    assert not list(outside.iterdir())


@pytest.mark.parametrize("mismatch", ["run_id", "output_root"])
def test_candidate_must_join_actual_invocation_context(
    parsed_run: PipelineArguments, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    from gnn.pipeline.finalization import finalize_run_evidence
    from gnn.pipeline.run_context import build_run_context

    context = build_run_context(
        parsed_run.target_dir, parsed_run.output_dir, "active-run", [3], {}
    )
    monkeypatch.setattr("gnn.pipeline.run_context.current_run_context", lambda: context)
    summary = {"run_id": "other-run" if mismatch == "run_id" else "active-run"}
    destination = (
        parsed_run.output_dir
        if mismatch == "run_id"
        else parsed_run.output_dir.parent / "other-output"
    )
    with pytest.raises(ValueError, match="different invocation|output differs"):
        finalize_run_evidence(destination, summary, None)
    assert not destination.exists()


@pytest.mark.needs_posix
def test_history_escape_recovery_never_writes_outside_owned_output(
    parsed_run: PipelineArguments,
) -> None:
    outside = parsed_run.output_dir.parent / "external-history"
    outside.mkdir()
    summary_dir = parsed_run.output_dir / "00_pipeline_summary"
    summary_dir.mkdir(parents=True)
    (summary_dir / ".history").symlink_to(outside, target_is_directory=True)
    assert orchestrator.main(parsed_run) == 1
    assert not list(outside.iterdir())
    # Unsafe ownership is rejected before creating an invocation or logging.
    # A failure finalizer must not try to write through the rejected tree.
    assert not (summary_dir / "run_context.json").exists()
    assert not (summary_dir / "pipeline_execution_summary.json").exists()


@pytest.mark.needs_posix
@pytest.mark.parametrize(
    "relative", ["00_pipeline_summary", "00_pipeline_logs", "3_gnn_output/nested"]
)
def test_main_rejects_external_output_directory_before_any_producer_writes(
    parsed_run: PipelineArguments, relative: str
) -> None:
    outside = parsed_run.output_dir.parent / "external-output"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_bytes(b"external bytes must stay unchanged")
    link = parsed_run.output_dir / relative
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)

    assert orchestrator.main(parsed_run) == 1
    assert sentinel.read_bytes() == b"external bytes must stay unchanged"
    assert sorted(path.name for path in outside.iterdir()) == ["sentinel.txt"]
    assert not (parsed_run.output_dir / "00_pipeline_summary/run_context.json").exists()
    assert not (parsed_run.output_dir / "00_pipeline_logs/pipeline.log").exists()
    assert not (parsed_run.output_dir / "PIPELINE_REPORT.md").exists()


@pytest.mark.needs_posix
@pytest.mark.parametrize(
    "relative", ["PIPELINE_REPORT.md", "00_pipeline_logs/pipeline.log"]
)
def test_main_rejects_output_hardlinks_without_modifying_external_bytes(
    parsed_run: PipelineArguments, relative: str
) -> None:
    outside = parsed_run.output_dir.parent / "external-sentinel.txt"
    outside.write_bytes(b"external bytes must stay unchanged")
    link = parsed_run.output_dir / relative
    link.parent.mkdir(parents=True)
    os.link(outside, link)

    assert orchestrator.main(parsed_run) == 1
    assert outside.read_bytes() == b"external bytes must stay unchanged"
    assert link.read_bytes() == outside.read_bytes()
    assert not (parsed_run.output_dir / "00_pipeline_summary/run_context.json").exists()


@pytest.mark.needs_posix
def test_ownership_refusal_does_not_use_existing_hardlinked_file_handler(
    parsed_run: PipelineArguments,
) -> None:
    outside = parsed_run.output_dir.parent / "external-log.txt"
    outside.write_bytes(b"external bytes must stay unchanged")
    log_path = parsed_run.output_dir / "00_pipeline_logs/pipeline.log"
    log_path.parent.mkdir(parents=True)
    os.link(outside, log_path)
    handler = logging.FileHandler(log_path)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        assert orchestrator.main(parsed_run) == 1
        handler.flush()
        assert outside.read_bytes() == b"external bytes must stay unchanged"
    finally:
        root_logger.removeHandler(handler)
        handler.close()


@pytest.mark.needs_posix
def test_output_symlink_cycle_is_a_safe_ownership_refusal(
    parsed_run: PipelineArguments,
) -> None:
    parsed_run.output_dir.mkdir()
    (parsed_run.output_dir / "first").symlink_to("second")
    (parsed_run.output_dir / "second").symlink_to("first")
    assert orchestrator.main(parsed_run) == 1
    assert not (parsed_run.output_dir / "00_pipeline_summary").exists()


def test_main_reuses_ordinary_history_without_clobbering_unrelated_files(
    parsed_run: PipelineArguments,
) -> None:
    history = parsed_run.output_dir / "00_pipeline_summary/.history"
    history.mkdir(parents=True)
    sentinel = history / "unrelated.txt"
    sentinel.write_bytes(b"retained history")
    assert orchestrator.main(parsed_run) == 0
    first_run = _summary(parsed_run)["run_id"]
    assert orchestrator.main(parsed_run) == 0
    assert _summary(parsed_run)["run_id"] != first_run
    assert sentinel.read_bytes() == b"retained history"
    assert (history / "index.json").is_file()
    assert (
        verify_run_manifests(
            parsed_run.output_dir / "v3_run_manifest", parsed_run.output_dir
        )
        == []
    )
