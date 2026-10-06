# TO-DO — GNN implementation, verification and release backlog

Updated 2026-10-05. **Implementation and release work have resumed.** The
scoped backlog and historical evidence remain preserved. Nothing below is a completed
release claim. Prior evidence remains in the scope documents, verification
ledger, commit history and reviewed release candidates.

The implementation branch is `codex/gnn-reliability-v4`. Its historical artifact commit is
`883e44c04e95ef4c69d7f998b9669a0a0a49ad29` (A3), with qualified production source
`8d202439eaa386d13a9b93888a7d560657579a4b`. Main was last observed at
`536d949829f6aed11dc540e5c5dec77578b25016`. Package metadata declares 4.0.0;
main integration, the version tag and GitHub release remain pending.

This is the active forward backlog. [SCOPE-2026-10-01.md](SCOPE-2026-10-01.md)
and [the verification ledger](docs/development/verification_2026_10_01.json)
contain historical stages; reconcile their final prepared successors before
publication. Do not restart accepted work merely because an older ledger entry
still says pending. Do not overwrite this handoff backlog with an older candidate.

## v4.0.0 — P0: finish the existing release safely

These tasks depend on one another in the order below. Source custody, local
checks, hosted checks, installed-wheel tests, native numerical witnesses, visual
review and publication are separate evidence planes. Keep their identities and
failures separate. Do not sum overlapping test selections.

The bounded `--autonomous` mode remains proposal-only in v4.0.0; it does not
modify its own source or grant autonomous publication authority.

- [ ] **R1 — qualify the current companion run.** Fep companion PR
  [#45](https://github.com/ActiveInferenceInstitute/fep_formal/pull/45) is pushed
  at `cd97a8f8785855625da4ebadad9318a35fd63863`. Its current
  [run 37384045009](https://github.com/ActiveInferenceInstitute/fep_formal/actions/runs/37384045009),
  attempt 1, completed with all 20 required jobs successful and documentation
  skipped. Current source/native/artifact checks passed; Python reported 2,495
  passed / 16 skipped with 91.40% statement coverage. Independent PDF review
  blocked receipt promotion: 151 unresolved references and 676 literal equation
  labels survive the existing render gate. Repair cross-reference rendering and
  fail-closed checks, then renew exact-source hosted/native/render acceptance.
  The passing cd97 results remain historical after any owner changes. The older
  P2 run 37380889327 was cancelled after
  supersession. Preserve its lint failure and the local 419-node attempt
  (413 passed / six failed). The one-import repair, canonical two ignored
  prerequisites and six-only passing replay are qualified; they do not prove a
  fresh full suite. Use the reviewed bounded collectors, current heads, fresh
  destinations, storage admission and observed cleanup. **Accept:** actual
  current-source formal/native/render evidence, with failures explicitly retained.

- [ ] **R2 — close companion render custody and main integration.** Independently
  review the actual template/render/native/audit artifacts, font evidence, four
  figures and finite PDF views. Apply only verified `docs/render-acceptance.json`
  and `docs/render-fonts.json` output bytes, using a normal successor commit.
  Renew current checks rather than promoting old E/C2/P2 receipts. **Accept:**
  all 20 required jobs successful, documentation lane explicitly skipped, all
  15 OS/Python distribution jobs successful, valid current coverage XML at or
  above the declared 89% floor, current source closure, independent review,
  normal PR merge and remote main SHA parity. Preserve unrelated primary Fep work.

- [ ] **R3 — seal public documentation and evidence.** Reconcile the prepared
  eight-document release candidate with current `cd97a8f` facts and this backlog.
  Destinations: `CITATION.cff`, `docs/VERSION_MAP.md`, `tests/AGENTS.md`,
  `TO-DO.md`, `SCOPE-2026-10-01.md`,
  `docs/development/verification_2026_10_01.json`, `CHANGELOG.md`, `SECURITY.md`.
  Preserve historical evidence and all failed attempts. Set the actual software
  publication date only at closeout; keep manuscript token date 2026-10-02 as
  its authored epoch. Record snapshot-time pending steps accurately and attach
  post-snapshot publication facts separately. **Accept:** exact file review,
  source/namespace parity, privacy checks, truthful state labels, link/anchor
  audits and fresh independent content approval. Do not broadly copy staging files.

- [ ] **R4 — adopt the reviewed API policy and resolve alert #12 narrowly.**
  The independently reviewed trusted-filesystem SECURITY wording is published
  on the implementation branch, and alert #12 received a narrow request-string
  false-positive disposition. CodeQL passed at published head
  `8badfad724f5a0dd2159146496f68ef71f388fb0` in
  [run 37392490778](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/actions/runs/37392490778).
  Renew this check after subsequent source changes. Attach
  the qualified 95 API negative/parity cases and static guard evidence to
  [CodeQL alert #12](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/security/code-scanning/12).
  Retain the narrowly justified request-string traversal disposition and renew
  CodeQL. **Accept:** policy adoption and a fresh current check;
  retain the non-atomic concurrent filesystem limitation. GUI alerts #13–#16
  remain separate. This is not a general security or deployed-service certificate.

- [ ] **R5 — final GNN pair pin and repository checks.** Follow
  [the canonical pairing order](docs/development/fep_lean_paired_revision.md).
  After accepted companion main integration, make the reviewed companion pin
  bump part of the final GNN content commit. Preserve the 25 steps, `gnn.*`,
  proposal-only autonomous mode and the qualified package namespace.
  **Accept:** strict manuscript/token/hydration/fresh-render checks; current
  finite/continuous pair checks; declared repository gates; all required hosted
  checks at the exact final head. A3's captured 18-check snapshot had 15 successes
  and three failures (CodeQL #12 and two pair legs); it is not final green evidence.
  Owner or count-changing edits require renewed custody/render/package acceptance.

- [ ] **R6 — normal GNN main integration.** Update PR
  [#251](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/pull/251)
  around the final implementation and evidence. Obtain final independent
  infrastructure/security review, verify current base/head/check identities,
  and merge normally. **Accept:** main remote SHA and tree parity, clean primary
  checkout fast-forward, preserved rollback baseline and no force/amend/squash.
  Preserve any intervening unrelated work.

- [ ] **R7 — reciprocal GEO-INFER pairing.** After GNN main M exists, update
  GEO-INFER's `.github/gnn-pair.json` to M through a focused normal PR.
  Inspect live GEO main first; its last reviewed snapshot is
  `510f1008e698b45aedc4306d65caf898da6cda04`. **Accept:** a pin-only diff,
  current Python 3.11/3.12 interchange checks, the three retained schema contracts,
  normal merge and remote parity. Complete this before a version tag or release.

- [ ] **R8 — publish and directly verify v4.0.0.** Recheck tag/release absence;
  verify that the accepted ordinary wheel's namespace matches final main M.
  Create a normal annotated `v4.0.0` tag and GitHub release with public-safe
  notes, wheel, manuscript PDF, source-binding/verification receipts and
  checksums. **Accept:** tag object and peeled M parity, release URL/ID,
  downloaded asset hash parity and a public post-publication receipt carrying
  the GNN/Fep/GEO identities and rollback evidence. No PyPI upload or
  version-specific archival DOI is currently claimed.

- [ ] **R9 — attach evidence and close only qualified issues/PRs.** After the
  resulting main change and current required checks exist, attach issue-specific
  evidence before closing work. Keep
  [#241](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues/241)
  and [#250](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues/250)
  open until their scientific acceptance is met. Resolve superseded PRs
  [#226](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/pull/226),
  [#231](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/pull/231),
  [#248](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/pull/248)
  and [#249](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/pull/249)
  with explicit links to integrated behavior and evidence, not unchanged merges.

## Existing issue acceptance map

Implementation exists for the following work; final attachment/closeout follows
R1–R9. Use existing issues rather than duplicate reports.

| Issues | Evidence required for closure |
| --- | --- |
| [#232](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues/232), [#233](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues/233), [#234](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues/234) | Current rendered Markdown/TeX commit stamps, shared grammar and exclusions, strict main checks, unchanged-inherited-drift PR controls, preamble controls and accepted manuscript custody. |
| [#235](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues/235) | Round-trip/tiny-probability/all-rank malformed-input regressions plus the qualified four full-duration NumPyro cases and source/script/result bindings. |
| [#236](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues/236), [#237](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues/237), [#239](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues/239), [#240](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues/240) | Frozen selection, nested/duplicate/empty/excluded cases, serial/parallel/consolidated identity agreement, current indexes and summary snapshots, once-only analysis/site assembly, stale-run refusal. |
| [#238](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues/238), [#247](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues/247) | Shared structured readiness diagnoses, bounded interpreter-specific probes, registry-derived setup/help, explicit CmdStan installation and genuine retained toolchain evidence. |
| [#242](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues/242) | Qualified coordination/swarm/GridWorld/bistable RxInfer render→execute→analysis results, real family/agent/factor traces, readable PNG/GIF/HTML and current artifact custody. |
| [#243](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues/243) | One deadline, retained partial/non-success outcomes, real child/grandchild cancellation/reaping, retries/result-transfer controls and the narrow manifest ignore rule. |
| [#244](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues/244) | Fixed-point annotations and generated-code regressions; current finite plot/PDF review with clearly stated visual scope. |
| [#245](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues/245) | Current deterministic pipeline PR lane and count/source receipts for the existing weekly all-extras lane; no duplicate schedule. |
| [#246](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues/246) | Typed Dask/Ray cancellation/retrieval with successful siblings/order/script identity, genuine scaling acceptance, and Matplotlib API shapes on the supported splits. |

## P1 — scientific and security follow-ups

- [ ] **S1 — complete-source LLM admission and coverage (#241).** Preserve the
  historical partial result: 38 structural analyses, 33 summaries, 82/342
  prompts and 260 unfinished after an explicit 900-second budget. Read-only
  metadata declares an 8,192-token training context; complete N64/N32 sources
  contain 2,673,574/342,214 bytes. Capacity mismatch is an inference, not an
  exact tokenizer count or native refusal. Design verified context admission
  or an explicitly source-preserving multi-request method; configuration changes
  must be explicit. **Accept:** every selected model and prompt completed with
  exact provider/model/source/prompt/checkpoint identities, truthful partial
  failure controls, default 600 seconds/model and 45-second request ceilings.
  Do not start a long run merely to rediscover capacity limits; never silently
  sample, truncate, swap models or manufacture coverage.

- [ ] **S2 — scientifically repair six authored examples (#250).** Establish
  intended probabilities for the unchanged-baseline invalid examples before
  editing values. All six explicit probability and behavioral proposals were
  approved on 2026-10-05, including block resets, the approximate temporal
  controller and contingent episodic T-maze policies. Source corrections are
  implemented locally; final independent source review and faithful native
  execution remain pending. Equation preservation and canonical action-axis
  metadata have been repaired; these do not establish backend acceptance.
  **Accept:** explicit source corrections, finite/nonnegative
  mass, dimensions/orientation, float32/float64 preservation, parser/serializer
  round trips and fresh native family acceptance. Retain rejection witnesses;
  do not normalize, omit examples or alter validator tolerances to hide failures.

- [ ] **S3 — THRML structural zeros (upstream #72).** Track
  [the validated upstream report](https://github.com/extropic-ai/thrml/issues/72).
  Upstream marked the report fixed on 2026-10-05; a released fix was not
  established by the current release check. Keep strict rejection until a released
  fix is verified. **Accept:**
  both supported JAX splits, exact-zero/inactive-padding NaN regressions,
  independent numerical witnesses and unchanged source/seed/sample semantics.
  No epsilon repair or widening based only on an upstream commit or shape match.

- [ ] **S4 — GUI complexity alerts #13–#16.** Establish bounded input admission
  or linear parsing with preserved editor behavior. **Accept:** adversarial
  multiline/delimiter growth measurements, refusal before excessive allocation,
  meaningful parser equivalence and live editor responsiveness. Existing bounded
  split-line tests do not prove linear complexity for arbitrary input.

- [ ] **S5 — stronger filesystem/platform boundaries.** Scope descriptor-based
  operations against concurrently replaced path components and explicit Windows
  lease/process support. Keep request-string containment separate from hostile
  filesystem mutation. **Accept:** real symlink/rename/race and platform controls,
  preserved API error contracts, bounded cleanup and fresh independent security
  review. No claim of atomic confinement or unsupported-platform containment.

- [ ] **S6 — raise measured coverage with meaningful tests.** The qualified
  default CI statement coverage is 70.56–70.59%, above its declared 60% floor;
  the documentation's >80% goal is not yet demonstrated. Prioritize untested
  failure paths and shared contracts, including optional/live surfaces.
  **Accept:** reproducible per-environment coverage reports, behavior-based tests
  and an explicit reviewed floor change when justified. Do not inflate totals,
  combine incompatible runs or add tests that merely mirror implementation.

## P2 — modularity, composability and maintainability

- [ ] **M1 — refresh the large-module backlog from actual source.** The current
  handoff scan finds large owners including `rxinfer_bridge.py` (1,801 lines),
  `website/generator.py` (1,665), `gui/gui_2/ui.py` (1,545),
  `parsers/schema_parser.py` (1,534),
  `render/activeinference_jl/activeinference_renderer.py` (1,449),
  `visualization/matrix/visualizer.py` (1,388),
  `utils/logging/logging_utils.py` (1,335), `pipeline/summary_wiring.py` (1,308),
  `execute/executor.py` (1,305), and `security/processor.py` (1,235).
  Inventory related RxInfer strategies/analyzer, framework comparison, main,
  parser common, PyMDP simulation, API parity, setup and LLM modules as well.
  **Accept:** one justified boundary per focused PR, preserved public exports
  and signatures, meaningful behavioral parity/failure cases, explicit ownership
  and measured complexity/size changes. Reconcile completed extractions first.

- [ ] **M2 — unify remaining execution/result contracts.** Audit shared run
  context, scope metadata, prerequisites, backend options, readiness diagnoses,
  model-kind adapters, outcomes and artifact indexes for duplicated logic.
  **Accept:** thin numbered orchestrators and common serial/parallel/matrix plans;
  no fallback discovery for explicit empty selection, stale-summary import,
  fabricated uncertainty or duplicate corpus aggregation.

- [ ] **M3 — align Python/CLI/REST/MCP configuration and discovery.** Derive
  backend inventory/help/setup groups and documented schemas from live metadata.
  Test option precedence, unknown keys, invalid types, explicit limits and public
  facade imports. **Accept:** equivalent admission/outcomes across entrypoints,
  installed-package isolation, clear migration and coherent supported/unsupported
  capability labels. Registry declarations alone do not prove readiness.

- [ ] **M4 — reduce error/documentation/dependency ratchets.** Reinventory broad
  exceptions, suppressed diagnostics, flag-documentation gaps and optional
  dependency duplication. Replace one reviewed cause at a time with typed
  outcomes and clear contextual reasons. **Accept:** ratchet improvement without
  widening catches, hiding failures, eager optional imports or changing public
  behavior; run the existing flag/import/thin/dependency gates.

- [ ] **M5 — audit current documentation and signposts.** Review actual README,
  SPEC, ARCHITECTURE, AGENTS, module guides, examples and API docs against source.
  The implementation's AGENTS already includes v4/experimental/executor
  corrections; do not copy the older September instructions over it.
  Clarify standalone discovery versus frozen selected views and current-run
  aggregation; derive inventories from registries. **Accept:** runnable commands,
  verified paths/anchors, explicit release/experimental/legacy distinctions and
  current terminology/doc-contract audits.

- [ ] **M6 — preserve manuscript/visualization scientific meaning.** Audit labels,
  units, axes, categorical versus Gaussian uncertainty, VFE iterations versus
  EFE timesteps, accessible legends, animation traces and source provenance.
  **Accept:** data validation, artifact hashes and finite manual/browser review
  reported separately; real template rendering/custody after count changes.
  Do not claim all-page review or browser behavior from static source checks.

- [ ] **M7 — measure scalability and performance on matched corpora.** Benchmark
  selected identities/coverage, phase timings, memory, artifact counts and resource
  admission with identical source/configuration/environment. Include nested and
  large inputs, repeated/concurrent runs and bounded distributed transfer.
  **Accept:** reproducible receipts and explicit variance; no speed claim from
  unmatched corpora, two noisy trials or declaration counts. Keep dense tensor
  file/disk guardrails and resource estimates separate from measured RSS/JIT.

- [ ] **M8 — broaden package/platform acceptance deliberately.** Resolve Python
  3.14 scientific-wheel readiness and verify any added supported split outside
  the checkout. Retain both current JAX/Matplotlib splits, API scratch isolation,
  registry concurrency, RSS under load and zero tracked-file drift.
  **Accept:** ordinary installation, actual imports/execution and meaningful
  negative cases; live GUI/network/audio/toolchains remain explicit opt-ins.

## P2 — additions and experimental promotion

- [ ] **E1 — extend multi-agent continuous semantics.** Reconcile existing
  continuous/factored/hybrid/nonstationary and independent Gaussian support
  before adding coupling or new control semantics. **Accept:** composed
  model-kind dispatch, native per-agent state/control identities, asymmetric
  dimensions and verified JAX/RxInfer exemplars with covariance-aware metrics.

- [ ] **E2 — extend THRML only through explicit scientific contracts.** Current
  `thrml==0.1.4` is explicitly selected finite strictly positive categorical
  fixed-action smoothing on CPU/JAX. Scope continuous/hybrid/coupled agents,
  policy search or hardware support separately. **Accept:** defensible mapping,
  released dependency compatibility on both splits, admission/failure/resource
  controls and independent native witnesses. No convergence, energy or speed
  claim without direct evidence. See [the THRML guide](docs/gnn/implementations/thrml.md).

- [ ] **E3 — extend cpomdp while preserving search admission.** Current adapter
  targets `cpomdp==0.4.4`; passive models perform no search. Keep EFE defaults,
  nine actions for two-dimensional control or `2p+1`, and caps of 4,096 policies,
  32,768 step evaluations and 128 MiB estimated allocation. **Accept:** exact
  constructor compatibility, finite/nonboolean options, all-invalid rejection,
  independent Kalman/EFE results, complete score/policy/source receipts and
  process watchdogs. Never silently reduce requested search; allocation estimates
  are not total RSS limits.

- [ ] **E4 — strengthen formal/numerical evidence binding.** Scope independent
  extraction and semantic witnesses beyond hash/value/shape custody.
  **Accept:** exact translation/model/inference/dtype assumptions and compatible
  numerical comparisons. Retained Lean statements, shape matches and hashes
  must not be promoted into universal runtime equivalence.

- [ ] **E5 — archive the actual version after publication.** Create and directly
  verify a v4-specific archive/DOI after R8. Keep the verified concept DOI and
  previous v3 archive separate. Any PyPI publication is a distinct release task
  with explicit destination, artifact and credential ownership.

## Verification and execution rules

Read the repository's current declared commands and CI before running work.
Use focused checks for each change; broaden only for new failures or changed
contracts. Existing accepted evidence is retained, not silently rerun/relabelled.
Representative repository commands are:

```bash
uv lock --check
uv run --frozen --no-sync ruff check src/gnn scripts
uv run --frozen --no-sync ruff format --check src scripts
uv run --frozen --no-sync mypy src --config-file pyproject.toml
uv run --frozen --no-sync python docs/development/docs_audit.py --strict --check-anchors --no-write
uv run --frozen --no-sync python scripts/check_repo_terminology.py --strict
uv run --frozen --no-sync python scripts/check_maintained_doc_terms.py --strict
uv run --frozen --no-sync python scripts/check_gnn_doc_patterns.py --strict
uv run --frozen --no-sync python scripts/check_capability_contracts.py --strict
uv run --frozen --no-sync python scripts/run_v3_orchestration_acceptance.py --strict
uv run --frozen --no-sync python -m pytest tests/ -q --tb=short -rsx -m 'not ollama and not pipeline and not mcp and not env_heavy and not toolchain'
uv run --frozen --no-sync python -m pytest tests/ -q --tb=short -rsx -m 'pipeline and not ollama and not env_heavy and (not toolchain or needs_posix)'
uv run --frozen --no-sync python scripts/check_manuscript_tokens.py --strict
uv run --frozen --no-sync python scripts/check_hydrated_prose.py
```

Also run the declared import, flag, thin-orchestrator, dependency, security and
manuscript gates appropriate to the change. Inspect diffs and tracked-file
bookends. Use fresh independent review for shared infrastructure/security.
Preserve unrelated work, credentials and private evidence. No force push, broad
cleanup, phantom provenance/amended custody commits, silent normalization or
blanket issue/security closure. Resume one scoped task at a time; remove landed
items from this backlog and retain exact evidence in history.
