"""Collect release workflow readiness without rerunning accepted work.

Uses the authenticated GitHub CLI. This is a workflow-status gate, not a
substitute for independent artifact, manuscript or scientific acceptance.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote

REPOSITORY = "ActiveInferenceInstitute/Generalized_Notation_Notation"
WORKFLOWS = (
    "ci.yml",
    "comprehensive-native-coverage.yml",
    "installed-platforms.yml",
    "local-gates.yml",
    "mcp-audit.yml",
    "docs-audit.yml",
    "actionlint.yml",
    "codeql.yml",
    "fep-lean-paired-revision.yml",
    "geo-infer-interchange.yml",
    "rxinfer.yml",
    "full-extras.yml",
    "gridworld.yml",
    "supply-chain-audit.yml",
    "pair-pin-freshness.yml",
    "custody-re-render.yml",
)


def gh(*arguments: str) -> Any:
    return json.loads(
        subprocess.check_output(["gh", *arguments], text=True, timeout=120)
    )


def pages(endpoint: str, key: str) -> list[dict]:
    result = []
    page = 1
    while page <= 1000:
        separator = "&" if "?" in endpoint else "?"
        data = gh("api", f"{endpoint}{separator}per_page=100&page={page}")[key]
        result.extend(data)
        if len(data) < 100:
            return result
        page += 1
    raise RuntimeError("Actions pagination exceeded the bounded collection limit")


def evaluate(runs: list[dict], jobs: list[dict], revision: str) -> dict:
    """Require the newest run/attempt, exact head and every declared job."""
    if not runs:
        return {"state": "missing"}
    run = max(runs, key=lambda item: (item["run_number"], item["run_attempt"]))
    timings = []
    for job in jobs:
        if job.get("started_at") and job.get("completed_at"):
            start = datetime.datetime.fromisoformat(
                job["started_at"].replace("Z", "+00:00")
            )
            end = datetime.datetime.fromisoformat(
                job["completed_at"].replace("Z", "+00:00")
            )
            timings.append(
                {"name": job["name"], "seconds": (end - start).total_seconds()}
            )
    result = {"run": run, "jobs": jobs, "job_timings": timings}
    if run["head_sha"] != revision:
        result["state"] = "wrong_source"
    elif run["status"] != "completed":
        result["state"] = "pending"
    elif run["conclusion"] != "success":
        result["state"] = "failed"
    elif not jobs or any(
        job["status"] != "completed" or job["conclusion"] != "success" for job in jobs
    ):
        result["state"] = "incomplete_jobs"
    else:
        result["state"] = "success"
    return result


def filter_runs(runs: list[dict], ref: str, event: str | None) -> list[dict]:
    """Main and tag workflows at one commit remain independent qualifications."""
    return [
        run
        for run in runs
        if run["head_branch"] == ref and (event is None or run["event"] == event)
    ]


def release_ready(observations: dict[str, dict]) -> bool:
    """Require a successful observation for each workflow in the declared roster."""
    return (
        bool(WORKFLOWS)
        and observations.keys() == set(WORKFLOWS)
        and all(result["state"] == "success" for result in observations.values())
    )


def inspect(
    workflow: str, revision: str, ref: str, event: str | None
) -> tuple[str, dict]:
    endpoint = (
        f"repos/{REPOSITORY}/actions/workflows/{workflow}/runs?head_sha={revision}"
    )
    if event:
        endpoint += f"&event={event}"
    runs = filter_runs(pages(endpoint, "workflow_runs"), ref, event)
    if not runs:
        return workflow, evaluate([], [], revision)
    latest = max(runs, key=lambda item: (item["run_number"], item["run_attempt"]))
    # Attempt-specific jobs: a successful earlier attempt cannot fill missing jobs.
    jobs = pages(
        f"repos/{REPOSITORY}/actions/runs/{latest['id']}/attempts/{latest['run_attempt']}/jobs",
        "jobs",
    )
    return workflow, evaluate(runs, jobs, revision)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Full immutable Git commit")
    parser.add_argument(
        "--ref", required=True, help="Remote branch or tag resolving to that commit"
    )
    parser.add_argument(
        "--event", choices=("push", "pull_request", "workflow_dispatch")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dispatch-missing",
        action="store_true",
        help="Dispatch only workflows with no run; never retry failures",
    )
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full commit SHA")
    if args.dispatch_missing and args.event:
        parser.error(
            "dispatch requires unfiltered events so existing automatic runs are retained"
        )
    endpoint = f"repos/{REPOSITORY}/commits/{quote(args.ref, safe='')}"
    if gh("api", endpoint)["sha"] != args.revision:
        parser.error("remote ref no longer resolves to requested revision")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        observations = dict(
            executor.map(
                lambda name: inspect(name, args.revision, args.ref, args.event),
                WORKFLOWS,
            )
        )
    dispatches: list[str] = []
    receipt = {
        "repository": REPOSITORY,
        "revision": args.revision,
        "ref": args.ref,
        "event": args.event,
        "ready": False,
        "workflows": observations,
        "dispatched": dispatches,
        "state": "dispatching" if args.dispatch_missing else "observing",
    }
    # Reserve before any dispatch; reuse refuses without launching more work.
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    if args.dispatch_missing:
        for name, result in observations.items():
            if result["state"] == "missing":
                if gh("api", endpoint)["sha"] != args.revision:
                    raise RuntimeError("remote ref moved before dispatch")
                subprocess.run(
                    [
                        "gh",
                        "workflow",
                        "run",
                        name,
                        "--repo",
                        REPOSITORY,
                        "--ref",
                        args.ref,
                    ],
                    check=True,
                    timeout=90,
                )
                dispatches.append(name)
                args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    if gh("api", endpoint)["sha"] != args.revision:
        raise RuntimeError(
            "remote ref moved during collection; observation cannot qualify the ref"
        )
    ready = release_ready(observations)
    receipt = {
        "repository": REPOSITORY,
        "revision": args.revision,
        "ref": args.ref,
        "observed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "event": args.event,
        "ready": ready,
        "workflows": observations,
        "dispatched": dispatches,
        "scope": "Workflow status only; artifact/source/native/manuscript verification remains required.",
    }
    # Unique output files preserve failed and pending observations for review.
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    for name, result in observations.items():
        print(f"{name}: {result['state']}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
