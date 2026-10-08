# TO-DO — GNN next steps

Updated 2026-10-08.
Current software version is **4.1.0**. Its [GitHub release](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/tag/v4.1.0) records exact tag, source, checks and artifacts; version metadata alone does not establish acceptance.
Every checkbox below is remaining work. Remove accepted bounded work while
retaining its evidence in [CHANGELOG.md](CHANGELOG.md), release receipts and Git
history. Historical failures retain their original source epochs.

## Scope and delivery order

There are **seven remaining workstreams: two externally blocked minor items and five major extensions**.
The bounded minor/medium campaign is complete at its independently accepted scope.
Coverage and installed-platform evidence are recorded in their maintained guides.
Accepted bounded M1/M2/M3/M4/M5/M6/M7/M8/S5/S6 implementation is recorded in the
[ownership review](docs/development/gnn_4_1_0_ownership_review.json),
[migration guide](docs/development/run_ownership_migration.md),
[visual review](docs/development/gnn_4_1_0_visual_review.json) and
[measurement report](src/gnn/analysis/PERFORMANCE.md).
Retain any precise unimplemented residual under its stable identifier; focused
passing counts alone do not complete a broader contract. These accepted scopes
do not certify every backend, lower algorithmic complexity, hostile-tree
confinement, universal accessibility or whole-pipeline acceleration.

Size describes implementation scope, not a promised version or deadline.
P1 denotes foundational correctness/security/capability gaps; P2 denotes
subsequent maintainability and expansion. Existing S/M/E identifiers stay stable.
Accept the native coverage and installed matrices independently. Released THRML
verification and exact-version archival depend on external availability; design
S1/E4 contracts before implementing their related major extensions.

## Minor work

- [ ] **S3 — verify a released THRML structural-zero fix (P1).**
  Scope: identify a released upstream version addressing
  [inactive categorical padding](https://github.com/extropic-ai/thrml/issues/72),
  evaluate its compatibility and test exact-zero/inactive-padding behavior on
  both supported JAX splits. Accept: independent numerical witnesses, no NaNs,
  unchanged source/seed/sample semantics and ordinary locked installation.
  Keep strict rejection until those checks pass. Depends on a suitable released
  fix; substantial adapter or inference changes require separate scope under E2.

- [ ] **E5 — archive the exact released version and verify its DOI (P2).**
  Scope: archive the released tag, distribution artifacts, manuscript, checksums
  and citation metadata; identify the version-specific archive separately from
  the project concept DOI. Accept: public archive/DOI resolution, matching
  version metadata and directly downloaded artifact hash parity. A PyPI delivery
  is a separate publication task with an identified package owner/destination,
  accepted wheel/sdist, installation verification and credential ownership.
  Depends on archive-provider availability and the selected release artifacts.

## Major work

- [ ] **S1 — complete source-preserving long-context LLM processing (P1).**
  Scope: design context admission and a complete multi-request/checkpoint method
  for large selected model sources and every required prompt. Preserve exact
  provider/model/source identities, declared budgets and original source content.
  Accept: genuine native completion for every selected source/prompt, truthful
  structural/native coverage and unfinished-work reporting, resumable checkpoints
  and timeout/cancellation/context-refusal controls. No silent truncation,
  sampling, model substitution or manufactured completion. Depends on verified
  model/provider context admission and the M2/M3 result/configuration contracts.

- [ ] **E1 — extend coupled multi-agent continuous semantics (P2).**
  Scope: reconcile existing continuous/factored/hybrid/nonstationary and
  independent Gaussian contracts, then specify additional coupling, control
  and observation semantics. Accept: explicit per-agent state/control identity,
  asymmetric dimensions, composed model-kind dispatch and independent native
  JAX/RxInfer exemplars with covariance-aware results. Unsupported compositions
  must refuse explicitly. Depends on M2/M3 and E4's semantic witness method;
  compatible shapes alone do not establish an inference contract.

- [ ] **E2 — expand THRML through separate scientific contracts (P2).**
  Scope: evaluate coupled agents, policy/action search, continuous/hybrid
  models and hardware execution as separately scoped extensions of the
  [THRML adapter](docs/gnn/implementations/thrml.md). Accept: defensible model and
  inference mappings, released compatibility on both supported splits,
  admission/failure/resource controls and independent native witnesses.
  Hardware, convergence, energy and speed claims each require their own direct
  evidence. Depends on S3 where structural zeros are required, plus M2/M3/E4.

- [ ] **E3 — extend cpomdp control/search capabilities safely (P2).**
  Scope: specify new control/search behavior through the
  [cpomdp adapter](docs/gnn/implementations/cpomdp.md), preserving passive
  no-search behavior, explicit EFE defaults and complete policy/source identity.
  Accept: released constructor compatibility, finite/nonboolean options,
  all-invalid rejection, independent Kalman/EFE witnesses, complete score and
  policy receipts, and bounded evaluation/allocation/process behavior.
  Never silently reduce requested search; allocation estimates do not certify
  total RSS. Depends on M2/M3, E4's numerical contracts and M7 resource evidence.

- [ ] **E4 — bind formal statements to numerical semantics (P1).**
  Scope: specify independent extraction and semantic witnesses for declared
  finite/continuous models, translations, inference algorithms and precision
  assumptions beyond source/hash/value/shape custody. Accept: exact assumptions,
  compatible numerical comparisons, counterexamples/refusals for mismatched
  semantics and source-bound formal/native/runtime evidence. Keep Lean theorem
  statements, generated-runner execution and bridge custody distinct; universal
  equivalence requires an actual proof. Coordinate changed FEP/GEO owners through
  the paired-revision procedure rather than relabelling older receipts.

## Verification and execution rules

Freeze the scope, owner, source baseline and acceptance criteria before each
implementation PR. Use focused existing checks for the changed behavior;
broaden only for new failures or changed contracts. Preserve public model/source
semantics, the 25-step workflow and explicit unsupported outcomes.
`--autonomous` remains proposal-only; expanded authority requires a separately
scoped runtime/security contract.

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

Read current CI and runtime configuration for the exact commands/environments.
Run the appropriate import, flag, thin-orchestrator, dependency and security
gates. Changes to FEP/GEO-bound owners require the
[paired-revision procedure](docs/development/fep_lean_paired_revision.md).
Manuscript input/count changes require the full figure, template-render and
custody ritual in [AGENTS.md](AGENTS.md). Keep local checks, hosted tests,
installation, native numerical witnesses and finite visual review separate;
report source identities, failures, optional skips and actual review coverage.

Merge through the existing required gates, preserve unrelated work and inspect
tracked-file bookends. Publication requires accepted exact-source checks, tag
and remote parity, verified distribution contents and public download hashes.
Retain credentials and private evidence locally. No force push, amended custody,
silent normalization, fallback success or unsupported readiness claims.
