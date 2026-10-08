# Pipeline architecture

The live [step registry](../src/gnn/pipeline/step_registry.py) defines 25 steps,
script paths and metadata. [Root AGENTS](../AGENTS.md) describes module owners
and processor-level artifact flow; scheduling prerequisites are separate.

Numbered `src/gnn/N_module.py` scripts parse arguments, establish logging,
resolve the step output directory and delegate to their module. The
[script factory](../src/gnn/utils/pipeline_orchestration/pipeline_template.py)
owns that common behavior. Domain algorithms belong to named leaf owners.
The executable [thin-script gate](../scripts/check_thin_orchestrators.py)
enforces a 150-line hard ceiling and a committed measured ratchet. Preserve
the stricter current ratchet; do not raise it to accommodate new logic.

Use canonical `gnn.*` imports and preserve declared public exports/signatures.
Share existing frozen selection, run context, resolved configuration,
readiness, outcomes and artifact indexes; do not create parallel owners.
The [run-ownership migration](../docs/development/run_ownership_migration.md)
governs current versus inherited evidence across serial/parallel/matrix runs.

A model's display name is separate from its stable source-relative identity.
Re-read source bytes only under an explicit verified contract. Build reports
and websites from current-run indexes; output directory presence alone cannot
admit evidence. Optional enrichments must retain their invocation identity.

Subprocess, in-process and distributed paths preserve requested work and
failure semantics. Shared deadlines cover admission, execution, retrieval and
cleanup. A scheduling policy may continue after failure while retaining the
failure; it must not force a successful exit. See [error handling](error_handling.md).

Root [manuscript and companion custody](../AGENTS.md) applies to changed owners.

## Related contracts

[Run selection, current artifacts and deadlines](run_ownership.md) owns the cross-cutting guidance.
