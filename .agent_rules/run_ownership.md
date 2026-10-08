# Run identity, deadlines and artifact ownership

Use this guide for selection, execution, resume and result collection.
[v4 migration](../docs/development/run_ownership_migration.md) owns the contract;
implementation owners include [run_context.py](../src/gnn/pipeline/run_context.py),
[output_lease.py](../src/gnn/pipeline/output_lease.py) and
[run_manifest.py](../src/gnn/pipeline/run_manifest.py).

## Freeze work once

Resolve configuration, selected steps/frameworks and sources at the invocation
boundary. Retain run ID, relative path, path-derived `model_id` and source SHA-256.
Display names are labels; duplicate filenames need distinct artifact stems.
Renames change path identity, while edits change source hashes.

Pass exact selected bytes through input views. Empty selection is zero work.
Record documentation/archive exclusions. Do not broaden selection during retries
or rescan a mutable directory in child workers.

## Admit current evidence

Render indexes select scripts; result indexes select analysis. Receipts bind run,
model, source, script and execution identity. Steps 23/24 read the matching current
summary. Historical files cannot enter a new manifest merely because they exist.

A `RUNNING` snapshot is progress evidence. Final acceptance requires completed
required work and current source/output/durable-manifest verification. Preserve
successful siblings while retaining aggregate failure and unfinished work.

## Own resources and time

Use distinct output roots for concurrent invocations. Acquire the supported OS
lease before producer writes and retain it until workers/evidence finish.
Validate entries before logging/writing; preserve rejected historical evidence.

Intersect readiness, execution, transfer, observation, termination, reaping and
stream draining with the monotonic deadline. Reuse canonical supervisor/budget
utilities. Exhausted required work fails despite earlier successes.

Report platform containment. Cooperative callbacks may outlive return; remote
cancellation/close requests do not prove termination. Caller-owned clients,
clusters and preexisting runtimes remain caller resources. POSIX descendant
checks, process-group fallback and Windows direct-worker observation establish
only their recorded scope. Direct-worker success is not descendant containment.

## Resume and verify

Validate every checkpoint source/configuration and artifact/response binding.
Earlier successful directories are insufficient. Exercise stale indexes, source
edits, duplicate/nested paths, empty selection, concurrent leases, exhausted
budgets, live children and cleanup failure through the public entrypoint.
