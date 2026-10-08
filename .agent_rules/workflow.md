# Task workflow and integration

Use this guide when starting, taking over or closing out work.
[Root AGENTS](../AGENTS.md) governs scope; [release.md](release.md) governs
publication and companion custody.

## Establish state

Identify repository, checkout, branch, full SHA, dirty paths, relevant history,
issue/PR discussion and current CI. Identify active workers, environments, output
roots and companion operations before launching duplicates.

Use an isolated branch/checkout and task-owned evidence directory. Preserve
existing edits and failed logs; inspect changes before integrating them. Never
reset, force-push or discard another worker's work.

Record the outcome and acceptance criteria. Read the closest code guide,
relevant rules/tests and the current [TO-DO.md](../TO-DO.md) scope.

## Implement and integrate

Fix the concrete trigger through the canonical public path. Keep code in its
owning package, update affected schemas/examples and add meaningful regression
coverage. Avoid unrelated restructuring.

Independent read-only checks may run concurrently. Writes to shared paths,
environments, output leases and evidence manifests need one owner. Ownership is
limited to assigned resources, not a global repository lease.

Commit recoverable increments normally, inspect the diff and run applicable
checks before pushing. Reconcile upstream without overwriting unrelated work.
Preserve source/artifact ancestry when receipts cite producer commits.

## Closeout

Report behavior, exact source, commands/terminal outcomes, artifacts and limits.
Distinguish local from hosted verification, branch push from main merge, draft
readiness from publication and cancellation request from observed cleanup.

When access fails, retain repository/branch, producer commits, last terminal
evidence and the next gate. Continue independent authorized work; pending checks
remain pending. Completed backlog items belong in history/receipts; future tasks
remain bounded acceptance items in the canonical backlog.
