# Dual-repo improvement cycle — GNN + fep_lean — 2026-09-05

Second cycle of the day (context: `comprehensive-review-20260905.md`). Deep
scoping across both repos plus their bridge co-operation surface, followed by
a custody-aware execution wave. Nothing was committed or pushed; the wave2
dirty state was preserved; no source pins, dependency pins, or locks were
touched; no Lean/Lake or native builds were run; uv toolchain only.

## Scoping (9-lens dual-repo scout pool, read-only)

Key discoveries, all file:line grounded:

- **Custody rosters measured, not assumed.** `gnn.owners` = 271 paths:
  src/gnn (82), src/gnn/render (53), src/gnn/utils (51), src/gnn/execute (45),
  src/gnn/pipeline (31), src/gnn/ontology (4), plus uv.lock, pyproject.toml,
  src/gnn/main.py, docs/gnn/gnn_syntax.md, docs/other/fep_lean/bridge-contract.md.
  `fep_lean.owners` = 146 paths. Everything else — src/gnn/export/**,
  tests/**, src/{analysis,llm,mcp,gui,integration,report,audio,
  model_registry,setup,rxinfer}/**, scripts/**, README/CHANGELOG/TO-DO,
  all AGENTS.md files — is unrostered. Per-file `grep -F` against
  `source-pin.json` is the check of record (a jq `index(.)` membership
  construction produced a false all-bound result once; distribution +
  `bridge status` disproved it).
- **Cross-framework reliability gate fails (exit 1)**: 7/9 families pass;
  `continuous` and `hierarchical` fail on comparator metric key-set asymmetry
  (jax sparse schemas vs richer pymdp/rxinfer/numpyro outputs), not runtime —
  every required framework execution passed, Julia included.
- **GEO-INFER (the only open TO-DO area) scoped per-item**: GNN-05 [MEDIUM]
  safest, GNN-04 [MEDIUM] needs GEO-repo CI access, GNN-02/03 [MAJOR] need
  cross-repo contract co-design.
- **Hygiene sweep**: 25 findings; dominant pattern is broad `except Exception`
  with silent fallbacks; TODO/FIXME hygiene essentially clean.
- **Docs drift**: 4 stale API claims (src/gnn/llm/AGENTS.md ×2,
  src/gnn/utils/AGENTS.md ×2). Two scoping claims corrected during execution:
  the arg_parsing "duplicate parse_step_arguments" is false (two different
  classes, both live), and gui/runner.py's second `return None` is live
  (test-covered).
- **fep_lean**: Python gates green (mypy 0/82, ruff clean, 1996 collected);
  backlog = 4 receipt-promised TODO.md rows (credential/Lean/sequencing
  blocked); code hygiene debt ≈ zero; one doc inconsistency flagged
  (TODO.md FEP-H2-SMOOTH row vs "H2.7 accepted" status — owner arbitration
  needed, closure requires their own probe + changelog rule).
- **Bridge**: contract v0.4 identical on both sides (programmatic mirror
  check exists at bridge/operations.py:196-200); Direction-1 stages S1–S4
  demonstrably wired, S5 certificate loop partial; emitters still live as
  duplicated per-slice spec scripts.

## Landed (all unrostered; every number verbatim from runs)

| Item | Files | Verification |
| --- | --- | --- |
| GNN-05 Step 7 opt-in GEO-INFER wiring | src/gnn/export/processor.py, src/gnn/7_export.py, + new tests/export/test_export_geo_pipeline.py (7 tests) | export dir 103 passed; orchestrator e2e 1 passed; scoped mypy+ruff clean |
| Export hygiene | formatters.py (dead insecure minidom fallback removed — defusedxml is a hard dep), processor.py (JSON recovery writer logs swallowed exception) | included above |
| analysis / model_registry / report / audio hygiene | 4 files: metric excepts narrowed + logged; registry hash except narrowed; health-score and audio fail-safes log | 460 passed (78s) + 17 passed |
| Misc hygiene + doc drift | src/gnn/analysis/rxinfer/gif_animator.py, src/gnn/integration/graph.py, src/gnn/gui/runner.py, src/gnn/mcp/processor.py, src/gnn/llm/mcp.py, src/gnn/llm/AGENTS.md (2 signatures fixed), scripts/run_v3_orchestration_acceptance.py (stale docstring) | 398 passed (51s); integration 64 passed |
| utils docs | src/gnn/utils/AGENTS.md (2 signatures corrected) | utils 211 + 15 contracts passed |
| Gate fix during integration | src/gnn/integration/graph.py log message reworded after my sweep caught `check_repo_terminology` flagging "legacy" | terminology gate clean |

`process_export` gained an opt-in `geo_infer` options mapping with the
distinct visible-failure contract (missing `step_seconds` → named `ValueError`
before any output; five default formats byte-identical otherwise). Step 7
reaches it via environment options `GNN_GEO_STEP_SECONDS` /
`GNN_GEO_STATE_IDS` / `GNN_GEO_SPACE_KIND` — CLI flag registration in the
shared step-argument registry is custody-bound and deferred (below).

## Custody-refresh backlog (designed + once-tested, then hand-reverted)

Execution workers initially also applied fixes inside rostered files; when
the roster truth was established mid-wave, those workers halted and
hand-reverted their exact hunks (verified via scoped `git diff` — only
pre-existing wave2 hunks remain; E1's post-revert snapshot tags matched
pre-edit reads). The GNN-side pin binding stayed clean through the entire
cycle. Ready-to-apply designs, all requiring one coordinated
`bridge pin --gnn-root` + `emit --refresh-digests` + certificate re-check
pass afterwards (owner action, "pin only after reviewing the settled owner
changes"):

1. **Comparator fix** (src/gnn/pipeline/cross_framework_reliability.py):
   intersection-based metric comparison with additive `skipped` notes; ran
   the strict gate at 9/9 families, exit 0, ~307s against the now-reverted
   code (/tmp/gnn-cfr-fix-0905 — historical artifact of reverted code).
2. **gnn hygiene** ×4 files: schema_parser ×3, schema_validator,
   cross_format_validator, maxima_parser — narrow handlers + 2 warnings
   (120 gnn tests passed against the reverted-later code).
3. **execute hygiene** ×4 files: detection, pymdp_simulation, executor,
   validator — logging on silent cpu fallbacks + version-check cause.
4. **src/gnn/render/visualization_suite.py:87**: HDF5 failure logging.
5. **Step 7 CLI flags**: register `geo_step_seconds`/`geo_state_ids`/
   `geo_space_kind` in `ArgumentParser.ARGUMENT_DEFINITIONS`
   (src/gnn/utils/arg_parsing.py, rostered), then switch 7_export.py from env
   options to CLI flags and update docs.

## Deliberately deferred (blocked or owner-gated)

- GNN-02/03: cross-repo contract co-design with the GEO consumer (spec-first).
- GNN-04: CI workflow requires the GEO repo's validator in runners.
- Bridge emitter promotion: contract v0.5 co-edit + coordinated receipt
  refresh across both repos — a dedicated cycle.
- fep_lean TODO.md rows: all receipt-promised (FEP-FULL-155 needs paid API;
  FEP-EVIDENCE-CURRENT blocked on owner settlement; FEP-H2-SMOOTH is a Lean
  slice; FEP-H3-SCIENCE is post-H2 gated). The FEP-H2-SMOOTH row vs "H2.7
  accepted" status inconsistency is flagged for owner arbitration; closing it
  requires fep_lean's own closure probe + changelog rule.
- Dead-test rewrites (tests/advanced_visualization/test_..._overall.py
  ×3 sites, tests/gnn/test_gnn_parsing.py ×3 sites): intent review needed.
- fep_lean code edits: none this cycle — every actionable surface is
  custody-bound (82 Python sources bind H2.7 acceptance) or receipt-promised.

## Verification receipts (integrator-run unless noted)

- Full suite (command of record, once, final tree): **4700 passed,
  14 skipped, 0 failed in 399.97s** — skips are the allowlisted sklearn (11),
  torch (1), D2 CLI (2). An earlier full run mid-cycle caught the two GNN-05
  integration seams (doc line-count contract; orchestrator unregistered-arg
  warning); both root-caused and fixed above, then re-receipted.
- mypy full: `Success: no issues found in 987 source files`.
- ruff check src scripts: clean; format: 1017 files already formatted.
- Doc audits ×4 (strict/check-only): broken links 0, anchors 0, gaps 0;
  patterns/terminology/maintained-terms clean.
- Capability contracts verified; semantic fidelity gate passed.
- fep_lean: mypy 0/82, ruff clean, 1996 collected (no full run — no fep_lean
  changes from this cycle).
- `fep-lean bridge status` (final): source_binding errors only
  `src/fep_lean/output/{manuscript,rendering}.py` — the concurrent fep lane's
  edits, present before this cycle; **zero gnn-side owner drift**; freshness
  STALE (pre-existing, resolves at the next deliberate re-pin); syntax
  surface, contract mirror, and formal projection all pass.

## Dirty-state ledger (close of cycle)

308 porcelain entries (217 M / 2 D / 89 untracked). Modified-today files =
this cycle's ~20 landed files + the concurrent docs/manuscript/audit lanes'
surfaces (manuscript/, output/{manuscript,pdf,slides,web,reports,data,
figures}, src/gnn/manuscript_variables.py, src/gnn/mcp/audit_report.json — avoided
entirely by this cycle). The 2 deletions (src/gnn/llm demo removals) and the
`out/` artifacts predate today. This report adds one untracked file.

## Cycle 2 — "push all" custody cycle and three-repo integration (same day)

The user ordered all remaining improvements landed and all three sibling
repos (GNN, fep_lean, GEO-INFER) pushed informatively to main.

**Custody-free + rostered landing.** Re-applied `uv_management.py:785`
accelerator-fallback logging (unnecessarily reverted in cycle 1 — unrostered);
rewrote 21 weak test sites (gnn parsing, advanced visualization) to assert
probed real behavior, zero deletions; landed the full rostered set with
gate-verified designs: comparator fix (strict gate 9/9, exit 0, 320s), gnn ×4,
execute ×4, visualization_suite logging, and Step 7 GEO opt-in restored to
CLI flags registered in the step-argument registry (`arg_parsing.py`
ARGUMENT_DEFINITIONS/STEP_ARGUMENTS + `step_config.py` optional_args; env-var
shim removed, clean cutover). Full suite on the landed tree: **4701 passed,
14 skipped, 0 failed**.

**Coordinated custody cycle.** Pre-pin drift union reconciled exactly
(2 pre-existing fep-lane files + precisely the 12 edited rostered files,
nothing unlisted); `bridge pin` → `emit --refresh-digests` (finite +
continuous) → `emit --check` → status ok. P3 certificates correctly remain
bound to their original pin ("existing receipts are not promoted");
re-issuing C1/C2 requires native Lean witnesses (program's own loop).

**Three-repo integration and push.** Upstream had moved under both repos
(GNN +7: geo-infer-interchange PR #25, CI enforcement; GEO +12: GNN
space-time interchange). Local states were committed informatively (GNN:
manuscript lane / quality wave / custody cycle; fep_lean: manuscript lane /
custody re-pin; GEO: staged RISK→INSURANCE rename / module pass / lock+CI),
then merged with origin: GNN 34 conflicts resolved by a 3-worker pool
(upstream canonical for reviewed interchange internals incl. its improved
OpenAIProvider running-loop fix; local canonical for Step 7 CLI flags,
hygiene, strengthened tests; one W3 resolution bug — TO-DO.md gutted to
4 lines — caught by the capability-contracts gate and restored to the true
union; process_export visible-failure contract adopts upstream's
return-False, CLI-level ValueError kept, test + docs adapted). GEO-INFER:
14 conflicts resolved inline (upstream policy-only inference params, verified
ISA checklist + local receipt, README/TODO counts recounted from the merged
tree: 45 modules, 917 src / 685 test files). Post-merge gates: mypy 0/994,
ruff clean, doc audits clean, **full suite 4755 passed / 14 skipped /
0 failed**. Custody re-pinned at each pushed state; final bridge status ok
(binding clean, finite + continuous FRESH).

**Pushed to main**: fep_lean `d16f252` + `bad6833`; GNN `f3b208a0`,
`0da46dcd4`, `dfab53b89`, merge `143af3762`; GEO-INFER `2855d786`,
`d66b9ae7`, `c39a3c59`, merge `2935d016`. All three working trees clean.

**Remaining open (unchanged)**: GNN-02/03 contract co-design, GNN-04
CI pairing (needs GEO runner access), bridge emitter promotion (v0.5),
P3 certificate re-issuance (native Lean), fep_lean receipt-promised rows,
H2.7 TODO arbitration, GPU/browser deferred-verification rows in GEO TODO.

### Post-close integrity sweep (same session)

Three false-clean corrections, all fixed and pushed: (1) the GEO-INFER
secrets scan first ran against a transposed SHA and silently no-op'd —
re-run against `f098b292..2935d016` is clean (one hit is a truncated
illustrative JWT in EXAMPLES API docs, not a credential); largest pushed
blobs are text (locks, references, geojson). (2) GEO's own
`validate_documentation.py --strict` passes over the recounted counts
(30 authoritative pages). (3) The GNN merge had written literal
`@theirs` placeholder stubs into seven AGENTS/README docs (md files
bypass ruff and conflict-marker greps; the shrink-check couldn't flag
them post-push because origin/main was already the corrupted state).
Proper resolutions restored: utils keeps local's verified signatures +
upstream's Step 7 options-file note; analysis and render/README keep
local's honest-probe and render-receipts sections; the test-double
naming set adopts upstream for consistency with merged mcp_stubs.py.
Pushed as `2bb61d429`; doc audits (unfiltered), pattern/terminology/
maintained-term gates, and doc contracts all green; bridge status ok.
