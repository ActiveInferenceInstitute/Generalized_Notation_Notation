# TO-DO - GNN Pipeline Roadmap

**Last Updated**: 2026-10-01
**Current Version**: 4.0.0
**Next Target**: v4.0.0 acceptance and publication — see remaining receipts below

## v4.0.0 — Active implementation

The current program is [SCOPE-2026-10-01.md](SCOPE-2026-10-01.md).
It supersedes the active rows from SCOPE-2026-09-23.md; prior scope documents,
CHANGELOG.md and Git history preserve their audit trail. Already-landed CLI,
extractor, manuscript and render/execute processor splits and bnlearn
execution are removed from this open ledger. Documentation mirror coverage is
verified by the current strict audit rather than the stale wave census.

| Work | Acceptance still required |
| --- | --- |
| Manuscript #232–#234, plotting #244 / PR #249 | Actual stamped render, hydration/digests and complete paired custody |
| Probability/result contracts #235/#242 | Strict source values, tiny/rank/axis regressions and native replay receipts |
| Pipeline #236/#237/#239/#240/#243 / PR #248 | Current-run IDs/coverage, byte ownership, lease and real process deadlines in all modes |
| Readiness #238/#247 and distributed #246 / PR #231 | Bounded diagnoses, genuine Dask/Ray cancellation and transfer acceptance |
| LLM #241 | Full corpus scheduling, exact provider/model, fair budgets and checkpointed partial coverage |
| Pipeline CI #245 and v4 migration | Default/pipeline tests, complete declared gates, installed wheel and independent review |
| THRML backend | Released `thrml==0.1.4`, bounded faithful discrete mappings, configurable CLI/API/MCP parity, native sampling and installed-wheel acceptance on both JAX splits |
| Experimental PR #226 and continuous agents | cpomdp release-wheel acceptance on both JAX splits and native asymmetric agent exemplars |
| Publication | Content → manuscript render → companion fep_lean review/merge → final GNN pair pin; GEO-INFER verification, hosted checks and remote SHA parity |

Proposal-only autonomous mode remains unchanged. Multi-agent continuous
capability requires explicit per-agent models; a shared vector and an agent
count alone cannot establish independent state/control semantics.

## Verification Commands

Use `uv run` for roadmap verification checks:

```bash
uv run python scripts/run_v3_orchestration_acceptance.py --strict
uv run python scripts/emit_run_manifest.py output --out /tmp/gnn-v3-run-manifest
uv run python scripts/generate_pipeline_container_plan.py --config input/config.yaml --out /tmp/gnn-v3-container-plan.json
uv run python scripts/run_session_acceptance.py --manifest input/model_family_manifest.json --output-dir /tmp/gnn-v3-session-acceptance --session /tmp/gnn-v3-session.json --strict
uv run python src/gnn/main.py --autonomous --target-dir input/gnn_files --output-dir /tmp/gnn-autonomous-smoke

uv run python docs/development/docs_audit.py --strict --check-anchors --no-write
uv run python scripts/check_gnn_doc_patterns.py --strict
uv run python scripts/check_maintained_doc_terms.py --strict
uv run python scripts/check_repo_terminology.py --strict
uv run python scripts/check_doc_path_references.py
uv run python scripts/check_capability_contracts.py
uv run python scripts/run_semantic_fidelity_gate.py --output-dir /tmp/semantic_fidelity --strict
uv run python scripts/run_cross_framework_reliability.py --output-dir /tmp/cross_framework --strict
git diff --check
```

## Conventions

- Keep this file limited to unchecked, forward-looking work.
- Move shipped-version details to release notes, changelog entries, or durable
  verification artifacts.
- Keep closed work out of this file: completed items are removed when they
  land; the audit trail lives in `CHANGELOG.md` and git history.
- Scope open items with concrete tasks, file paths, verification commands, and
  acceptance criteria so the next session can execute without re-deriving them.
