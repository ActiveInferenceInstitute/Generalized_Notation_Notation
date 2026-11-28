"""Fail-closed release scheduling controls (stdlib only)."""

import copy
import unittest

from release_gate import (
    REQUIRED_MATRIX_JOBS,
    WORKFLOWS,
    evaluate,
    filter_runs,
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


if __name__ == "__main__":
    unittest.main()
