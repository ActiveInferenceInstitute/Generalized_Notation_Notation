# CI gates and faster verified pushes

Use this guide for scheduling/readiness. [Workflow AGENTS](../.github/workflows/AGENTS.md)
and actual YAML own triggers, selections, matrices, artifacts and statuses.

## Gate selection

Run focused checks, then mandatory hosted gates for the exact pushed revision.
Documentation work still runs workflows without path filters. Derive the roster
from workflow/check metadata. Identify source head versus synthetic merge,
run/attempt and artifact binding. Missing, pending, cancelled or skipped required
work cannot establish success.

## Preserve integrity while scheduling

Standard Python 3.12 tests and quality run independently. `test (3.12)` uses
`always()` and accepts only successful `pytest (3.12)` and `quality (3.12)`.
Preserve coupling and 3.11/3.13 statuses. Aggregates reject missing/failed/skipped
or cancelled dependencies.

Install each runner's frozen environment once; subsequent commands use
`uv run --frozen --no-sync`. Preserve markers, extras, membership, JUnit/coverage,
source-bound receipts, security gates and deadlines. Cache versioned immutable
inputs rather than unverified old-source results. Parallelize independent checks;
serialize shared evidence promotion and publication.

## Evidence and retries

Retain raw step/job times and terminal outcomes plus artifact names/sizes/hashes.
Inspect later dependent jobs and paginated collections. A successful first page
is insufficient. Retry classified infrastructure failures with bounded attempts
and retained failed evidence. Scientific, collection, coverage and custody failures
need repairs; do not weaken selections, denominators, thresholds or deadlines.

Core and comprehensive/native lanes have different selections. Preserve their
configured floors, strict integer assertions and reconciled JUnit membership.
[Testing](testing.md) owns evidence, [performance](performance.md) owns timings,
and [release](release.md) owns main/tag/asset checks. Claim acceleration only after
equivalent work has terminal results.
