"""Fail-closed release scheduling controls (stdlib only)."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from release_gate import (
    REQUIRED_MATRIX_JOBS,
    WORKFLOWS,
    evaluate,
    filter_runs,
    main,
    release_ready,
)


class ReleaseGateControls(unittest.TestCase):
    def setUp(self):
        self.sha = "a" * 40
        self.run = dict(
            run_number=8,
            run_attempt=1,
            head_sha=self.sha,
            status="completed",
            conclusion="success",
        )
        self.jobs = [dict(status="completed", conclusion="success")]

    def state(self, runs=None, jobs=None):
        return evaluate(
            [self.run] if runs is None else runs,
            self.jobs if jobs is None else jobs,
            self.sha,
        )["state"]

    def test_complete(self):
        self.assertEqual(self.state(), "success")

    def test_missing(self):
        self.assertEqual(self.state(runs=[]), "missing")
        self.assertEqual(self.state(jobs=[]), "incomplete_jobs")

    def test_latest_failure_cannot_borrow_prior_success(self):
        failed = dict(self.run, run_number=9, conclusion="failure")
        self.assertEqual(self.state(runs=[failed, self.run]), "failed")

    def test_rerun_attempt_cannot_borrow_prior_success(self):
        pending = dict(self.run, run_attempt=2, status="in_progress", conclusion=None)
        self.assertEqual(self.state(runs=[self.run, pending]), "pending")

    def test_wrong_source(self):
        self.run["head_sha"] = "b" * 40
        self.assertEqual(self.state(), "wrong_source")

    def test_same_commit_main_and_tag_cannot_qualify_each_other(self):
        main = dict(self.run, head_branch="main", event="push")
        tag = dict(
            self.run,
            head_branch="v4.2.0",
            event="push",
            run_number=9,
            conclusion="failure",
        )
        self.assertEqual(
            self.state(runs=filter_runs([main, tag], "main", "push")), "success"
        )
        self.assertEqual(
            self.state(runs=filter_runs([main, tag], "v4.2.0", "push")), "failed"
        )
        self.assertEqual(filter_runs([main], "v4.2.0", "push"), [])

    def test_manual_run_cannot_fill_push_only_qualification(self):
        manual = dict(self.run, head_branch="v4.2.0", event="workflow_dispatch")
        self.assertEqual(filter_runs([manual], "v4.2.0", "push"), [])
        self.assertEqual(filter_runs([manual], "v4.2.0", None), [manual])

    def test_push_cannot_fill_main_manual_qualification(self):
        push = dict(self.run, head_branch="main", event="push")
        manual = dict(push, event="workflow_dispatch", run_number=7)
        self.assertEqual(
            filter_runs([push, manual], "main", "workflow_dispatch"), [manual]
        )
        self.assertEqual(filter_runs([push], "main", "workflow_dispatch"), [])

    def test_every_job_must_finish_successfully(self):
        for conclusion in ["skipped", "cancelled", "failure", None]:
            jobs = copy.deepcopy(self.jobs)
            jobs.append(dict(status="completed", conclusion=conclusion))
            self.assertEqual(self.state(jobs=jobs), "incomplete_jobs")

    def test_an_all_green_partial_matrix_cannot_qualify(self):
        required = REQUIRED_MATRIX_JOBS["installed-platforms.yml"]
        jobs = [dict(self.jobs[0], name=name) for name in sorted(required)]
        complete = evaluate([self.run], jobs, self.sha, required_jobs=required)
        self.assertEqual(complete["state"], "success")
        partial = evaluate([self.run], jobs[1:], self.sha, required_jobs=required)
        self.assertEqual(partial["state"], "incomplete_jobs")
        self.assertEqual(partial["missing_jobs"], [jobs[0]["name"]])


class ReleaseRosterControls(unittest.TestCase):
    def test_release_workflow_concurrency_separates_event_ref_and_commit(self):
        workflows = Path(__file__).resolve().parent / "workflows"
        for name in WORKFLOWS:
            with self.subTest(workflow=name):
                source = (workflows / name).read_text(encoding="utf-8")
                group = next(
                    line for line in source.splitlines() if line.startswith("  group: ")
                )
                for field in ("github.event_name", "github.ref", "github.sha"):
                    self.assertIn(field, group)
                if "\n  pull_request:\n" in source:
                    branch, release_phase = group.split(" || ", 1)
                    self.assertIn("'pull_request' && github.ref", branch)
                    self.assertNotIn("github.sha", branch)
                    self.assertIn(
                        "format('{0}-{1}-{2}', github.event_name, github.ref, github.sha)",
                        release_phase,
                    )
                    self.assertIn(
                        "cancel-in-progress: ${{ github.event_name == 'pull_request' }}",
                        source,
                    )

    def test_incomplete_roster_cannot_report_ready(self):
        observations = {name: {"state": "success"} for name in WORKFLOWS}
        self.assertTrue(release_ready(observations))
        self.assertFalse(release_ready({}))
        for name in WORKFLOWS:
            with self.subTest(workflow=name):
                filtered = observations.copy()
                del filtered[name]
                self.assertFalse(release_ready(filtered))
                filtered[name] = {"state": "missing"}
                self.assertFalse(release_ready(filtered))


class ReleaseDispatchControls(unittest.TestCase):
    def test_dispatch_requires_explicit_main_manual_phase(self):
        for ref, event in (
            ("main", None),
            ("main", "push"),
            ("v5.1.0", "workflow_dispatch"),
        ):
            with self.subTest(ref=ref, event=event):
                argv = [
                    "release_gate.py",
                    "--revision",
                    "a" * 40,
                    "--ref",
                    ref,
                    "--output",
                    "unused.json",
                    "--dispatch-missing",
                ]
                if event is not None:
                    argv.extend(["--event", event])
                with patch("sys.argv", argv), patch("release_gate.gh") as gh:
                    with (
                        redirect_stderr(io.StringIO()) as stderr,
                        self.assertRaises(SystemExit) as exc,
                    ):
                        main()
                    self.assertEqual(exc.exception.code, 2)
                    self.assertIn(
                        "requires --ref main --event workflow_dispatch",
                        stderr.getvalue(),
                    )
                    gh.assert_not_called()

    def test_dispatch_only_missing_manual_runs_and_recheck_arrivals(self):
        revision = "a" * 40
        states = {
            "missing.yml": "missing",
            "newly-queued.yml": "missing",
            "failed.yml": "failed",
            "pending.yml": "pending",
            "stale.yml": "wrong_source",
            "partial.yml": "incomplete_jobs",
            "passed.yml": "success",
        }
        calls = []

        def inspect(workflow, observed_revision, ref, event):
            self.assertEqual(
                (observed_revision, ref, event), (revision, "main", "workflow_dispatch")
            )
            state = states[workflow]
            if workflow == "newly-queued.yml" and workflow in calls:
                state = "pending"
            calls.append(workflow)
            return workflow, {"state": state}

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "receipt.json"
            args = [
                "release_gate.py",
                "--revision",
                revision,
                "--ref",
                "main",
                "--event",
                "workflow_dispatch",
                "--output",
                str(output),
                "--dispatch-missing",
            ]
            with (
                patch("sys.argv", args),
                patch("release_gate.WORKFLOWS", tuple(states)),
                patch("release_gate.gh", return_value={"sha": revision}),
                patch("release_gate.inspect", side_effect=inspect),
                patch("release_gate.subprocess.run") as dispatch,
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main(), 1)
            commands = [call.args[0] for call in dispatch.call_args_list]
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][:4], ["gh", "workflow", "run", "missing.yml"])
            self.assertEqual(commands[0][-2:], ["--ref", "main"])
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["event"], "workflow_dispatch")
            self.assertEqual(receipt["dispatched"], ["missing.yml"])
            self.assertEqual(
                receipt["workflows"]["newly-queued.yml"]["state"], "pending"
            )
            self.assertFalse(receipt["ready"])


if __name__ == "__main__":
    unittest.main()
