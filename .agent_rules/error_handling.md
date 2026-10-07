# Outcomes and error handling

Required work reports whether it actually completed. Preserve public errors
and actionable context, narrowing caught exceptions to their cause. Optional
absence, unsupported models, invalid options, backend failures, timeout and
failed cleanup remain distinct outcomes.

Numbered scripts use shared
[exit-code coercion](../src/gnn/utils/errors/error_handling.py) through the
[script factory](../src/gnn/utils/pipeline_orchestration/pipeline_template.py).
Read those owners for Boolean, integer and truthiness handling. Zero
denotes success, one failure and two warning; scheduling policy determines
subsequent work. Do not force Steps 8, 9 or 12 to return zero after failure.

Continuing a run can preserve successful siblings and diagnostics. It cannot
relabel failed work or admit inherited files as new results. Record selected
source identity, failed operation and supported remediation without backend
substitution or silently reduced scope.

Use existing supervision/readiness owners. The shared deadline covers
preparation, dispatch, result retrieval and cleanup. Exhaustion, cancellation,
surviving owned workers and incomplete required streams/artifacts prevent
success. State platform containment limits precisely; process observation is
not an operating-system sandbox for hostile code.

Tests reproduce the real failure through its consumer and check the outcome,
sibling evidence and bounded cleanup. See
[run ownership](../docs/development/run_ownership_migration.md) and
[runtime safety](../src/gnn/utils/runtime_safety/AGENTS.md).
