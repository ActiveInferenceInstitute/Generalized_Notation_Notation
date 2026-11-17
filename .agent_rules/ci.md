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

Standard Python 3.12 tests and quality run independently as `pytest (3.12)` and
`quality (3.12)`; the retired `test (3.12)` alias is not required by any ruleset
(only `gitvault/gnn-repo-gates` is). Pull requests run `test (3.13)`/`test (3.14)`;
the Python 3.11 legs run on version tags, the weekly schedule and manual dispatch
while `requires-python` still admits 3.11. Aggregates reject
missing/failed/skipped or cancelled dependencies.

Install each runner's frozen environment once; subsequent commands use
`uv run --frozen --no-sync`. Preserve markers, extras, membership, JUnit/coverage,
source-bound receipts, security gates and deadlines. Cache versioned immutable
inputs rather than unverified old-source results. Parallelize independent checks;
serialize shared evidence promotion and publication.

Documentation/anchor and terminology checks run once in `docs-audit.yml` on
all changes. Structural repository checks run in `local-gates.yml`; quality
owns Ruff, mypy and public contracts. Keep these owners synchronized with YAML.
Julia caches retain package inputs, not prior test success. GridWorld provisions
both Julia projects and the explicit Gemma 3 1B Ollama profile before running all 25 steps.
Weekly/manual external-link audits retain all observations and fail on 404/410;
403/429, timeouts and transport failures remain inconclusive.

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
