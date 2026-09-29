#!/usr/bin/env python3
"""Tests for the consolidated in-process step executor (S2-11 / V4-STAGE).

Proves the equivalence contract from ADR 0001: consolidated in-process
execution produces the same downstream artifacts as the canonical subprocess
path for a small target dir, receipts record the execution mode, and steps
outside the whitelist are refused.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pytest

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from gnn.execute.subprocess_envelope import CancelToken  # noqa: E402
from gnn.pipeline import step_executor  # noqa: E402
from gnn.pipeline.step_executor import (  # noqa: E402
    UnsupportedStepError,
    can_execute_in_process,
    clear_parsed_model_carrier,
    execute_step_in_process,
)
from gnn.utils.arguments.pipeline_arguments import PipelineArguments  # noqa: E402
from gnn.utils.errors.error_handling import status_from_exit_code  # noqa: E402
from gnn.utils.pipeline_orchestration.execution_utils import (
    execute_command_streaming,  # noqa: E402
)

PROJECT_ROOT = SRC

LOGGER = logging.getLogger("test_step_executor")
BASICS_DIR = PROJECT_ROOT / "input" / "gnn_files" / "basics"


def _pipeline_args(output_dir: Path) -> PipelineArguments:
    """Build pipeline args pointing at the small basics fixture dir."""
    return PipelineArguments(target_dir=BASICS_DIR, output_dir=output_dir)


@pytest.fixture(autouse=True)
def _isolated_parsed_model_carrier() -> Iterator[None]:
    """Keep the executor's parsed-model carrier cache out of test state."""
    clear_parsed_model_carrier()
    yield
    clear_parsed_model_carrier()


def _artifact_files(step_output: Path) -> list[str]:
    """Sorted artifact-relative file list under a step's output dir."""
    return sorted(
        p.relative_to(step_output).as_posix()
        for p in step_output.rglob("*")
        if p.is_file()
    )


def _run_step3_subprocess(output_dir: Path) -> None:
    """Run the canonical numbered-script subprocess for step 3."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "gnn" / "3_gnn.py"),
        "--target-dir",
        str(BASICS_DIR),
        "--output-dir",
        str(output_dir),
    ]
    result = subprocess.run(  # nosec B603 - fixed argv, no shell
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def _run_step_subprocess(script_name: str, output_dir: Path) -> None:
    """Run the canonical numbered-script subprocess for *script_name*."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "gnn" / script_name),
        "--target-dir",
        str(BASICS_DIR),
        "--output-dir",
        str(output_dir),
    ]
    result = subprocess.run(  # nosec B603 - fixed argv, no shell
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def _seed_step3_in_process(output_dir: Path) -> None:
    """Populate *output_dir*/3_gnn_output via the (already pinned) executor."""
    seed = execute_step_in_process("3_gnn.py", _pipeline_args(output_dir), LOGGER)
    assert seed["exit_code"] == 0, seed["stderr"]


def _assert_matching_summaries(
    sub_dir: Path, in_dir: Path, summary_name: str, fields: list[str]
) -> dict[str, Any]:
    """Compare path-independent summary fields across the two runs."""
    sub_summary = json.loads((sub_dir / summary_name).read_text())
    in_summary = json.loads((in_dir / summary_name).read_text())
    for field in fields:
        assert in_summary[field] == sub_summary[field], field
    return in_summary


def _assert_matching_export_fields(
    sub_summary: dict[str, Any], in_summary: dict[str, Any]
) -> None:
    """Compare the step-7 per-file export record across the two runs."""
    sub_files = {e["file_name"]: e["success"] for e in sub_summary["files_exported"]}
    in_files = {e["file_name"]: e["success"] for e in in_summary["files_exported"]}
    assert in_files == sub_files
    assert in_files and all(in_files.values())


class TestConsolidatedEquivalence:
    """In-process mode produces the same downstream artifacts as subprocess."""

    def test_step3_in_process_matches_subprocess_artifacts(
        self, tmp_path: Path
    ) -> None:
        """Step 3 via the executor matches the subprocess artifact contract."""
        subprocess_out = tmp_path / "subprocess"
        in_process_out = tmp_path / "inprocess"
        _run_step3_subprocess(subprocess_out)

        args = _pipeline_args(in_process_out)
        step_result = execute_step_in_process("3_gnn.py", args, LOGGER)

        assert step_result["exit_code"] == 0
        assert step_result["status"] == "SUCCESS"

        sub_dir = subprocess_out / "3_gnn_output"
        in_dir = in_process_out / "3_gnn_output"
        assert sub_dir.is_dir() and in_dir.is_dir()
        sub_files = _artifact_files(sub_dir)
        in_files = _artifact_files(in_dir)
        assert in_files == sub_files
        assert in_files, "no step-3 artifacts produced"

        # Model count: every source GNN file parses successfully, and the
        # per-model directories match the source file names.
        sub_summary = json.loads((sub_dir / "gnn_processing_summary.json").read_text())
        in_summary = json.loads((in_dir / "gnn_processing_summary.json").read_text())
        model_count = len(list(BASICS_DIR.glob("*.md")))
        assert in_summary["total_files"] == sub_summary["total_files"]
        assert in_summary["successful_parses"] == sub_summary["successful_parses"]
        assert in_summary["total_files"] == model_count
        assert in_summary["formats_per_file"] == sub_summary["formats_per_file"]

    def _run_step7_parity(self, tmp_path: Path) -> tuple[Path, Path]:
        """Run step 7 both ways; return the two 7_export_output dirs.

        Step 7 consumes step 3's parse artifacts, so each root is seeded
        with an in-process step-3 run (its subprocess-vs-executor parity is
        pinned by test_step3_in_process_matches_subprocess_artifacts).
        """
        subprocess_out = tmp_path / "subprocess"
        in_process_out = tmp_path / "inprocess"
        _seed_step3_in_process(subprocess_out)
        _run_step_subprocess("7_export.py", subprocess_out)

        _seed_step3_in_process(in_process_out)
        step_result = execute_step_in_process(
            "7_export.py", _pipeline_args(in_process_out), LOGGER
        )

        assert step_result["exit_code"] == 0
        assert step_result["status"] == "SUCCESS"
        return subprocess_out / "7_export_output", in_process_out / "7_export_output"

    def test_step7_export_in_process_matches_subprocess_artifacts(
        self, tmp_path: Path
    ) -> None:
        """Step 7 via the executor matches the subprocess artifact contract."""
        sub_dir, in_dir = self._run_step7_parity(tmp_path)
        assert sub_dir.is_dir() and in_dir.is_dir()
        sub_files = _artifact_files(sub_dir)
        in_files = _artifact_files(in_dir)
        assert in_files == sub_files
        assert in_files, "no step-7 export artifacts produced"

        sub_summary = json.loads((sub_dir / "export_results.json").read_text())
        in_summary = json.loads((in_dir / "export_results.json").read_text())
        for field in (
            "total_files",
            "successful_exports",
            "failed_exports",
            "formats_generated",
        ):
            assert in_summary["summary"][field] == sub_summary["summary"][field], field
        _assert_matching_export_fields(sub_summary, in_summary)

    def test_step8_visualization_in_process_matches_subprocess_artifacts(
        self, tmp_path: Path
    ) -> None:
        """Step 8 via the executor matches the subprocess artifact contract."""
        subprocess_out = tmp_path / "subprocess"
        in_process_out = tmp_path / "inprocess"
        # Step 8's artifact set depends on the step-3 parse inputs (the
        # JSON-primary path adds ontology legends), so both roots get the
        # same seed; the executor additionally runs its prerequisite check,
        # which the bare CLI reference does not perform.
        _seed_step3_in_process(subprocess_out)
        _run_step_subprocess("8_visualization.py", subprocess_out)

        _seed_step3_in_process(in_process_out)
        step_result = execute_step_in_process(
            "8_visualization.py", _pipeline_args(in_process_out), LOGGER
        )
        assert step_result["exit_code"] == 0
        assert step_result["status"] == "SUCCESS"

        sub_dir = subprocess_out / "8_visualization_output"
        in_dir = in_process_out / "8_visualization_output"
        assert sub_dir.is_dir() and in_dir.is_dir()
        sub_files = _artifact_files(sub_dir)
        in_files = _artifact_files(in_dir)
        assert in_files == sub_files
        assert in_files, "no step-8 visualization artifacts produced"

        summary = _assert_matching_summaries(
            sub_dir,
            in_dir,
            "visualization_summary.json",
            [
                "processed_files",
                "total_visualizations",
                "success",
                "warnings",
                "errors",
            ],
        )
        assert summary["success"] is True
        model_count = len(list(BASICS_DIR.glob("*.md")))
        assert summary["processed_files"] == model_count

    def test_step11_render_in_process_matches_subprocess_artifacts(
        self, tmp_path: Path
    ) -> None:
        """Step 11 via the executor matches the subprocess artifact contract."""
        subprocess_out = tmp_path / "subprocess"
        in_process_out = tmp_path / "inprocess"
        # Identical inputs on both roots; the seed also satisfies the
        # executor's prerequisite check (the bare CLI does not perform one).
        _seed_step3_in_process(subprocess_out)
        _run_step_subprocess("11_render.py", subprocess_out)

        _seed_step3_in_process(in_process_out)
        step_result = execute_step_in_process(
            "11_render.py", _pipeline_args(in_process_out), LOGGER
        )
        assert step_result["exit_code"] == 0
        assert step_result["status"] == "SUCCESS"

        sub_dir = subprocess_out / "11_render_output"
        in_dir = in_process_out / "11_render_output"
        assert sub_dir.is_dir() and in_dir.is_dir()
        sub_files = _artifact_files(sub_dir)
        in_files = _artifact_files(in_dir)
        assert in_files == sub_files
        assert in_files, "no step-11 render artifacts produced"

        # The receipt carries a fresh run id and absolute paths; compare the
        # path-independent outcome scalars instead.
        _assert_matching_summaries(
            sub_dir,
            in_dir,
            "render_processing_summary.json",
            [
                "total_files",
                "successful_files",
                "failed_files",
                "total_framework_attempts",
                "successful_framework_renderings",
                "framework_success_rate",
            ],
        )


class TestConsolidatedReceipt:
    """Receipt fields record which execution mode ran."""

    def test_receipt_records_consolidated_mode(self, tmp_path: Path) -> None:
        """The executor stamps execution_mode="consolidated" on its receipt."""
        step_result = execute_step_in_process(
            "3_gnn.py", _pipeline_args(tmp_path / "receipt"), LOGGER
        )

        assert step_result["execution_mode"] == "consolidated"
        assert step_result["status"] in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}
        assert step_result["exit_code"] in (0, 2)
        assert isinstance(step_result["prerequisite_check"], bool)

    def test_record_step_result_defaults_execution_mode(self) -> None:
        """Subprocess receipts default to execution_mode="subprocess" and the
        consolidated stamp is never overwritten by the shared recording tail."""
        from gnn.main import _record_step_result

        def recorded_summary() -> dict[str, Any]:
            return {
                "steps": [],
                "performance_summary": {
                    "peak_memory_mb": 0.0,
                    "total_steps": 1,
                    "failed_steps": 0,
                    "critical_failures": 0,
                    "successful_steps": 0,
                    "warnings": 0,
                },
            }

        summary = recorded_summary()
        _record_step_result(
            {"status": "SUCCESS", "stdout": "", "stderr": "", "exit_code": 0},
            1,
            "3_gnn.py",
            "GNN file processing",
            datetime.now(),
            datetime.now(),
            0.1,
            summary,
            1,
            None,
            LOGGER,
        )
        assert summary["steps"][0]["execution_mode"] == "subprocess"

        summary = recorded_summary()
        _record_step_result(
            {
                "status": "SUCCESS",
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
                "execution_mode": "consolidated",
            },
            1,
            "3_gnn.py",
            "GNN file processing",
            datetime.now(),
            datetime.now(),
            0.1,
            summary,
            1,
            None,
            LOGGER,
        )
        assert summary["steps"][0]["execution_mode"] == "consolidated"


class TestConsolidatedRefusal:
    """Unknown or unsupported steps are refused by the executor."""

    def test_unknown_step_refused(self, tmp_path: Path) -> None:
        """Steps outside the whitelist raise UnsupportedStepError."""
        args = _pipeline_args(tmp_path)
        with pytest.raises(UnsupportedStepError):
            execute_step_in_process("9_advanced_viz.py", args, LOGGER)
        with pytest.raises(UnsupportedStepError):
            execute_step_in_process("99_nonsense.py", args, LOGGER)

    def test_alias_resolved_step_outside_whitelist_refused(
        self, tmp_path: Path
    ) -> None:
        """Consolidated aliases resolve first, then the whitelist decides:
        13_audio canonicalizes to 15_audio, which is not in the slice."""
        args = _pipeline_args(tmp_path)
        with pytest.raises(UnsupportedStepError):
            execute_step_in_process("13_audio.py", args, LOGGER)

    def test_gate_matches_whitelist_and_matrix(self, tmp_path: Path) -> None:
        """can_execute_in_process encodes exactly the slice gate."""
        args = _pipeline_args(tmp_path)
        assert can_execute_in_process("0_template.py", args)
        assert can_execute_in_process("3_gnn.py", args)
        assert can_execute_in_process("5_type_checker.py", args)
        assert can_execute_in_process("7_export.py", args)
        assert can_execute_in_process("8_visualization.py", args)
        assert can_execute_in_process("11_render.py", args)
        assert not can_execute_in_process("9_advanced_viz.py", args)
        assert not can_execute_in_process("99_nonsense.py", args)
        # Matrix enabled alone does not block: execute_pipeline_step only
        # forks per-folder when a subfolder actually allows the step.
        matrix_config = {"testing_matrix": {"enabled": True, "default_steps": [3]}}
        assert can_execute_in_process("3_gnn.py", args, pipeline_config=matrix_config)
        # Uniform routing (every subfolder gets the step) collapses to one
        # root run, so the step stays consolidated-eligible.
        dispatch_target = tmp_path / "dispatch"
        (dispatch_target / "model_a").mkdir(parents=True)
        (dispatch_target / "model_b").mkdir(parents=True)
        dispatch_args = PipelineArguments(
            target_dir=dispatch_target, output_dir=tmp_path
        )
        assert can_execute_in_process(
            "3_gnn.py", dispatch_args, pipeline_config=matrix_config
        )
        # A matrix that really routes (model_b opts out) keeps the step on the
        # per-folder subprocess path.
        routed_config = {
            "testing_matrix": {
                "enabled": True,
                "default_steps": [3],
                "folders": {"model_b": [5]},
            }
        }
        assert not can_execute_in_process(
            "3_gnn.py", dispatch_args, pipeline_config=routed_config
        )

    def test_global_steps_disabled_produces_skipped_receipt(
        self, tmp_path: Path
    ) -> None:
        """A disabled global step mirrors the subprocess skip without running."""
        args = _pipeline_args(tmp_path)
        matrix_config = {
            "testing_matrix": {"enabled": True, "global_steps": {"0_template": False}}
        }
        step_result = execute_step_in_process(
            "0_template.py", args, LOGGER, pipeline_config=matrix_config
        )
        assert step_result["status"] == "SKIPPED"
        assert step_result["exit_code"] == 0
        assert step_result["execution_mode"] == "consolidated"


class TestWhitelistedSteps:
    """Every whitelisted step runs in-process and writes its standard dir."""

    @pytest.mark.parametrize(
        ("script", "dir_name"),
        [
            ("0_template.py", "0_template_output"),
            ("5_type_checker.py", "5_type_checker_output"),
            ("7_export.py", "7_export_output"),
            ("8_visualization.py", "8_visualization_output"),
            ("11_render.py", "11_render_output"),
        ],
    )
    def test_step_produces_standard_output_dir(
        self, tmp_path: Path, script: str, dir_name: str
    ) -> None:
        if script == "7_export.py":
            # Step 7 hard-fails without step 3's parse artifacts; seed them
            # so the bare in-process run exercises the export path itself.
            _seed_step3_in_process(tmp_path)
        step_result = execute_step_in_process(script, _pipeline_args(tmp_path), LOGGER)
        assert step_result["execution_mode"] == "consolidated"
        assert step_result["exit_code"] in (0, 2), step_result["stderr"]
        assert (tmp_path / dir_name).is_dir()


class TestInProcessTimeoutAndCapture:
    """Slice A: wall-clock timeout and stdout/stderr capture in-process."""

    def test_timeout_receipt_parity_with_subprocess_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A timed-out in-process step records the subprocess timeout receipt.

        Parity pin: the receipt keys are unchanged (identical to a
        completing in-process run), the exit-code semantics match the
        subprocess tier's timeout (-1 with the FAILED status derived the
        same way), and the stderr text carries the TIMEOUT vocabulary plus
        the honest cancellation notice (this seeded step never reaches a
        safe point, so it records the abandoned-daemon variant).
        """

        def fast_step(**kwargs: Any) -> bool:
            return True

        release = threading.Event()

        def slow_step(**kwargs: Any) -> bool:
            return release.wait(timeout=60)

        monkeypatch.setattr(
            step_executor, "resolve_step_function", lambda name: slow_step
        )
        monkeypatch.setattr(step_executor, "_CANCEL_GRACE_SECONDS", 0.2)
        try:
            step_result = execute_step_in_process(
                "3_gnn.py", _pipeline_args(tmp_path), LOGGER, timeout_seconds=1
            )
        finally:
            release.set()

        # The same subprocess machinery main.py uses, timed out the same way.
        sub = execute_command_streaming(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=1,
            print_stdout=False,
            print_stderr=False,
            capture_output=True,
        )
        sub_exit = sub.get("exit_code", -1)
        sub_status = status_from_exit_code(sub_exit, [])

        assert step_result["exit_code"] == sub_exit == -1
        assert step_result["status"] == sub_status == "FAILED"
        assert "TIMEOUT" in step_result["stderr"]
        # BC-13: the honest notice states the abandonment (the seeded step
        # never reaches a cancellation safe point), and the receipt carries
        # the additive force_killed flag.
        assert "abandoned" in step_result["stderr"]
        assert "daemon" in step_result["stderr"]
        assert step_result["force_killed"] is True
        assert step_result["execution_mode"] == "consolidated"

        # Receipt schema unchanged: same keys as a completing in-process run.
        monkeypatch.setattr(
            step_executor, "resolve_step_function", lambda name: fast_step
        )
        normal = execute_step_in_process(
            "3_gnn.py", _pipeline_args(tmp_path / "normal"), LOGGER
        )
        assert set(step_result) == set(normal)
        assert normal["force_killed"] is False

    def test_captured_output_lands_in_receipt_and_console(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        """Tee semantics: step output is captured AND still reaches console."""

        def chatty_step(**kwargs: Any) -> bool:
            print("step stdout line")
            sys.stderr.write("step stderr line\n")
            return True

        monkeypatch.setattr(
            step_executor, "resolve_step_function", lambda name: chatty_step
        )
        step_result = execute_step_in_process(
            "3_gnn.py", _pipeline_args(tmp_path), LOGGER
        )

        assert step_result["exit_code"] == 0
        assert step_result["status"] == "SUCCESS"
        assert "step stdout line" in step_result["stdout"]
        assert "consolidated in-process execution completed" in step_result["stdout"]
        assert "step stderr line" in step_result["stderr"]
        captured = capsys.readouterr()
        assert "step stdout line" in captured.out
        assert "step stderr line" in captured.err


class TestInProcessForceKill:
    """BC-13: cooperative token kill path, force_killed receipt, no wedge."""

    def test_token_injected_only_when_declared(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cancel token rides as a kwarg only for steps that declare it."""
        seen: dict[str, Any] = {}
        received: list[Any] = []

        def tokenless_step(**kwargs: Any) -> bool:
            seen.update(kwargs)
            return True

        def token_aware_step(cancel_token: Any = None, **kwargs: Any) -> bool:
            received.append(cancel_token)
            return True

        monkeypatch.setattr(
            step_executor, "resolve_step_function", lambda name: tokenless_step
        )
        receipt = execute_step_in_process("3_gnn.py", _pipeline_args(tmp_path), LOGGER)
        assert receipt["exit_code"] == 0
        assert receipt["force_killed"] is False
        assert "cancel_token" not in seen

        monkeypatch.setattr(
            step_executor, "resolve_step_function", lambda name: token_aware_step
        )
        execute_step_in_process("3_gnn.py", _pipeline_args(tmp_path / "aware"), LOGGER)
        # The executor binds the token to the declared parameter itself.
        assert len(received) == 1
        assert isinstance(received[0], CancelToken)

    def test_cooperative_hang_stops_promptly_with_force_killed_receipt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A token-aware hanging step stops at its safe point; receipt says so."""
        done = threading.Event()

        def hanging_step(cancel_token: Any = None, **kwargs: Any) -> bool:
            assert cancel_token is not None
            while not cancel_token.cancelled:
                time.sleep(0.01)
            done.set()
            return False

        monkeypatch.setattr(
            step_executor, "resolve_step_function", lambda name: hanging_step
        )
        started = time.monotonic()
        step_result = execute_step_in_process(
            "3_gnn.py", _pipeline_args(tmp_path), LOGGER, timeout_seconds=1
        )
        elapsed = time.monotonic() - started

        assert step_result["exit_code"] == -1
        assert step_result["status"] == "FAILED"
        assert step_result["force_killed"] is True
        assert "TIMEOUT" in step_result["stderr"]
        assert "cancel token" in step_result["stderr"]
        assert "abandoned" not in step_result["stderr"]
        assert done.wait(timeout=5), "step thread did not terminate promptly"
        assert elapsed < 5  # returned near the deadline, not after the hang

    def test_noncooperating_hang_is_abandoned_without_wedging(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A step reaching no safe point is abandoned; the executor moves on."""
        release = threading.Event()

        def stubborn_step(**kwargs: Any) -> bool:
            release.wait(timeout=120)
            return True

        monkeypatch.setattr(
            step_executor, "resolve_step_function", lambda name: stubborn_step
        )
        monkeypatch.setattr(step_executor, "_CANCEL_GRACE_SECONDS", 0.2)
        started = time.monotonic()
        try:
            step_result = execute_step_in_process(
                "3_gnn.py", _pipeline_args(tmp_path), LOGGER, timeout_seconds=1
            )
        finally:
            release.set()
        elapsed = time.monotonic() - started

        assert step_result["exit_code"] == -1
        assert step_result["force_killed"] is True
        assert "abandoned" in step_result["stderr"]
        assert "daemon" in step_result["stderr"]
        assert elapsed < 1 + 0.2 + 5

        # A subsequent step still runs: the executor is not wedged.
        def fast_step(**kwargs: Any) -> bool:
            return True

        monkeypatch.setattr(
            step_executor, "resolve_step_function", lambda name: fast_step
        )
        follow = execute_step_in_process(
            "3_gnn.py", _pipeline_args(tmp_path / "after"), LOGGER
        )
        assert follow["exit_code"] == 0
        assert follow["force_killed"] is False

    def test_timeout_marks_partial_artifacts_without_deleting(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Timed-out artifacts are marked by a sentinel, never deleted."""

        artifact_written = threading.Event()

        def writing_hang(
            output_dir: Any = None, cancel_token: Any = None, **kwargs: Any
        ) -> bool:
            out = Path(str(output_dir))
            out.mkdir(parents=True, exist_ok=True)
            (out / "partial_artifact.txt").write_text("mid-write", encoding="utf-8")
            artifact_written.set()
            assert cancel_token is not None
            while not cancel_token.cancelled:
                time.sleep(0.01)
            return False

        monkeypatch.setattr(
            step_executor, "resolve_step_function", lambda name: writing_hang
        )
        step_result = execute_step_in_process(
            "3_gnn.py", _pipeline_args(tmp_path), LOGGER, timeout_seconds=1
        )
        # The marker is written when the step returns, so the artifact must
        # already exist then; a worker thread scheduled too late to write it
        # inside the timeout window fails here, not as a misleading
        # possibly_partial mismatch below.
        assert artifact_written.is_set(), (
            "writing_hang did not write its partial artifact before the "
            "timeout expired (worker thread scheduled late)"
        )
        assert step_result["force_killed"] is True

        step_output = tmp_path / "3_gnn_output"
        artifact = step_output / "partial_artifact.txt"
        marker = step_output / ".gnn_step_timed_out"
        assert artifact.is_file()
        assert artifact.read_text(encoding="utf-8") == "mid-write"
        payload = json.loads(marker.read_text(encoding="utf-8"))
        assert payload["step"] == "3_gnn"
        assert payload["force_killed"] is True
        assert "partial_artifact.txt" in payload["possibly_partial"]

        # A later successful run of the same step re-authors the directory.
        def fast_step(**kwargs: Any) -> bool:
            return True

        monkeypatch.setattr(
            step_executor, "resolve_step_function", lambda name: fast_step
        )
        follow = execute_step_in_process("3_gnn.py", _pipeline_args(tmp_path), LOGGER)
        assert follow["exit_code"] == 0
        assert not marker.exists()

    def test_timeout_marker_tolerates_coarse_filesystem_mtimes(
        self, tmp_path: Path
    ) -> None:
        """A file stamped slightly before the step start is still a candidate.

        Filesystem mtimes come from a coarser clock than ``time.time()`` (Linux
        uses the kernel's coarse clock), so an artifact written right after the
        step started can carry an mtime a few ms *earlier* than the recorded
        start. The scan must not drop it; a genuinely stale file stays out.
        """
        started = time.time()
        fresh = tmp_path / "fresh.txt"
        stale = tmp_path / "stale.txt"
        fresh.write_text("mid-write", encoding="utf-8")
        stale.write_text("old run", encoding="utf-8")
        os.utime(fresh, (started - 0.005, started - 0.005))
        os.utime(stale, (started - 3600, started - 3600))

        step_executor._mark_timed_out_artifacts(
            tmp_path, "3_gnn", timeout_seconds=1, started=started, logger=LOGGER
        )

        payload = json.loads(
            (tmp_path / step_executor._TIMEOUT_MARKER_NAME).read_text(encoding="utf-8")
        )
        assert payload["possibly_partial"] == ["fresh.txt"]

    def test_parallel_worker_slot_released_after_hang(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The worker slot frees at deadline+grace even for a stubborn step."""
        release = threading.Event()

        def stubborn_step(**kwargs: Any) -> bool:
            release.wait(timeout=120)
            return True

        monkeypatch.setattr(
            step_executor, "resolve_step_function", lambda name: stubborn_step
        )
        monkeypatch.setattr(step_executor, "_CANCEL_GRACE_SECONDS", 0.2)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                execute_step_in_process,
                "3_gnn.py",
                _pipeline_args(tmp_path),
                LOGGER,
                timeout_seconds=1,
            )
            try:
                # A wedged slot would surface here as a future TimeoutError.
                step_result = future.result(timeout=10)
            finally:
                release.set()
        assert step_result["force_killed"] is True

    def test_receipt_schema_additive_over_subprocess_tier(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """force_killed is additive: consolidated keys = subprocess keys + 2."""

        def fast_step(**kwargs: Any) -> bool:
            return True

        monkeypatch.setattr(
            step_executor, "resolve_step_function", lambda name: fast_step
        )
        receipt = execute_step_in_process("3_gnn.py", _pipeline_args(tmp_path), LOGGER)
        # execute_pipeline_step's step_result shape (main.py), the tier this
        # receipt must stay compatible with.
        subprocess_keys = frozenset(
            {
                "status",
                "stdout",
                "stderr",
                "memory_usage_mb",
                "peak_memory_mb",
                "memory_delta_mb",
                "exit_code",
                "retry_count",
                "prerequisite_check",
                "dependency_warnings",
            }
        )
        assert set(receipt) - subprocess_keys == {"execution_mode", "force_killed"}
        assert receipt["force_killed"] is False


class TestParsedModelCarrier:
    """Slice C: in-memory parsed-model handoff from step 3 to steps 7/8."""

    @staticmethod
    def _whitelist_with(monkeypatch: pytest.MonkeyPatch, *stems: str) -> None:
        """Pin the carrier consumer stems into the executor's whitelist.

        The whitelist is registry-owned and may expand per slice; the union
        keeps these tests independent of that churn in either direction.
        """
        monkeypatch.setattr(
            step_executor,
            "CONSOLIDATED_IN_PROCESS_STEMS",
            step_executor.CONSOLIDATED_IN_PROCESS_STEMS | frozenset(stems),
        )

    def test_step3_collects_carrier_and_step7_receives_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Carrier collected once after step 3, forwarded as a step kwarg."""
        step3 = execute_step_in_process(
            "3_gnn.py", _pipeline_args(tmp_path), LOGGER, collect_parsed_model=True
        )
        assert step3["exit_code"] == 0, step3["stderr"]

        received: dict[str, Any] = {}

        def fake_export(**kwargs: Any) -> bool:
            received.update(kwargs)
            return True

        self._whitelist_with(monkeypatch, "7_export")
        monkeypatch.setattr(
            step_executor, "resolve_step_function", lambda name: fake_export
        )
        result = execute_step_in_process(
            "7_export.py", _pipeline_args(tmp_path), LOGGER, collect_parsed_model=True
        )

        assert result["exit_code"] == 0, result["stderr"]
        assert "parsed_model" in received
        carrier = received["parsed_model"]
        assert isinstance(carrier, dict)
        assert isinstance(carrier["results"]["processed_files"], list)
        assert carrier["results"]["processed_files"]
        models = carrier["models"]
        assert models
        # Carrier payloads are the exact content of step 3's on-disk JSON.
        step3_dir = tmp_path / "3_gnn_output"
        on_disk = json.loads(
            (step3_dir / "gnn_processing_results.json").read_text(encoding="utf-8")
        )
        assert carrier["results"]["summary"] == on_disk["summary"]
        for entry in on_disk["processed_files"]:
            if not entry.get("parse_success"):
                continue
            parsed_file = json.loads(
                Path(entry["parsed_model_file"]).read_text(encoding="utf-8")
            )
            assert models[Path(entry["file_path"]).stem] == parsed_file

        # Backward compat: without the flag, steps behave exactly as before.
        received.clear()
        execute_step_in_process("7_export.py", _pipeline_args(tmp_path), LOGGER)
        assert "parsed_model" not in received

    def test_step8_receives_carrier_and_carrier_load_matches_disk_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Step 8 gets the kwarg; a carrier model load equals a disk load."""
        args = _pipeline_args(tmp_path)
        step3 = execute_step_in_process(
            "3_gnn.py", args, LOGGER, collect_parsed_model=True
        )
        assert step3["exit_code"] == 0, step3["stderr"]

        received: dict[str, Any] = {}

        def fake_visualization(**kwargs: Any) -> bool:
            received.update(kwargs)
            return True

        self._whitelist_with(monkeypatch, "8_visualization")
        monkeypatch.setattr(
            step_executor, "resolve_step_function", lambda name: fake_visualization
        )
        result = execute_step_in_process(
            "8_visualization.py", args, LOGGER, collect_parsed_model=True
        )
        assert result["exit_code"] == 0, result["stderr"]
        models = received["parsed_model"]["models"]
        assert models

        from gnn.visualization.core.parsed_model import load_visualization_model

        viz_dir = tmp_path / "8_visualization_output"
        for gnn_file in sorted(BASICS_DIR.glob("*.md")):
            if gnn_file.stem not in models:
                continue
            content = gnn_file.read_text(encoding="utf-8")
            from_disk = load_visualization_model(gnn_file, content, viz_dir)
            from_carrier = load_visualization_model(
                gnn_file, content, viz_dir, parsed_model=models[gnn_file.stem]
            )
            assert from_carrier == from_disk

    def test_step7_artifacts_byte_identical_with_carrier_on_vs_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Carrier-served export is byte-identical to the disk re-read.

        Extends the :70-104 parity approach against the same step-3 run:
        step 3's parsed JSON embeds per-node UUIDs and timestamps, so two
        independent step-3 runs can never byte-match. One step-3 run feeds
        both step-7 runs instead — carrier on (in-memory) vs carrier off
        (disk re-read) — proving the carrier neither changes nor loses
        content. Only the step-7 run timestamp in ``export_results.json``
        may differ.
        """
        self._whitelist_with(monkeypatch, "7_export")
        run_dir = tmp_path / "run"
        args = _pipeline_args(run_dir)

        step3 = execute_step_in_process(
            "3_gnn.py", args, LOGGER, collect_parsed_model=True
        )
        assert step3["exit_code"] == 0, step3["stderr"]

        on_step7 = execute_step_in_process(
            "7_export.py", args, LOGGER, collect_parsed_model=True
        )
        assert on_step7["exit_code"] == 0, on_step7["stderr"]
        export_dir = run_dir / "7_export_output"
        carrier_snapshot = tmp_path / "carrier_snapshot"
        shutil.copytree(export_dir, carrier_snapshot)
        # Clean slate: any file the carrier-on run left behind must not be
        # mistaken for carrier-off output in the file-list comparison.
        shutil.rmtree(export_dir)
        off_step7 = execute_step_in_process("7_export.py", args, LOGGER)
        assert off_step7["exit_code"] == 0, off_step7["stderr"]

        on_files = _artifact_files(carrier_snapshot)
        off_files = _artifact_files(export_dir)
        assert on_files == off_files
        assert on_files, "no step-7 artifacts produced"
        for rel in on_files:
            on_bytes = (carrier_snapshot / rel).read_bytes()
            off_bytes = (export_dir / rel).read_bytes()
            if on_bytes == off_bytes:
                continue
            # Only the run timestamp inside export_results.json may differ.
            assert rel == "export_results.json"
            on_json = json.loads(on_bytes)
            off_json = json.loads(off_bytes)
            on_json.pop("timestamp", None)
            off_json.pop("timestamp", None)
            assert on_json == off_json


class TestMatrixTargetFolders:
    """Per-folder forking only when the testing matrix actually routes."""

    @staticmethod
    def _tree(root: Path, *names: str) -> Path:
        for name in names:
            (root / name).mkdir(parents=True)
        return root

    def test_uniform_routing_collapses_to_one_root_run(self, tmp_path: Path) -> None:
        from gnn.pipeline.step_executor import matrix_target_folders

        root = self._tree(tmp_path / "in", "a", "b", "c")
        matrix = {"enabled": True, "default_steps": [9, 16, 20], "folders": {}}
        for step in (9, 16, 20):
            assert matrix_target_folders(step, root, matrix) == []

    def test_partial_routing_forks_only_selected_folders(self, tmp_path: Path) -> None:
        from gnn.pipeline.step_executor import matrix_target_folders

        root = self._tree(tmp_path / "in", "a", "b", "c")
        matrix = {"enabled": True, "default_steps": [16], "folders": {"b": [3]}}
        assert [p.name for p in matrix_target_folders(16, root, matrix)] == ["a", "c"]
        assert [p.name for p in matrix_target_folders(3, root, matrix)] == ["b"]

    def test_excluded_folder_keeps_per_folder_dispatch(self, tmp_path: Path) -> None:
        from gnn.pipeline.step_executor import matrix_target_folders

        root = self._tree(tmp_path / "in", "a", "b", "archived_gnn_files")
        matrix = {"enabled": True, "default_steps": [16]}
        # A root run would also read the archive, so keep forking per folder.
        assert [p.name for p in matrix_target_folders(16, root, matrix)] == ["a", "b"]

    def test_non_recursive_run_keeps_per_folder_dispatch(self, tmp_path: Path) -> None:
        from gnn.pipeline.step_executor import matrix_target_folders

        root = self._tree(tmp_path / "in", "a", "b")
        matrix = {"enabled": True, "default_steps": [16]}
        folders = matrix_target_folders(16, root, matrix, recursive=False)
        assert [p.name for p in folders] == ["a", "b"]

    def test_disabled_matrix_and_setup_steps_never_fork(self, tmp_path: Path) -> None:
        from gnn.pipeline.step_executor import matrix_target_folders

        root = self._tree(tmp_path / "in", "a")
        assert matrix_target_folders(16, root, {"enabled": False}) == []
        assert (
            matrix_target_folders(2, root, {"enabled": True, "default_steps": [2]})
            == []
        )
