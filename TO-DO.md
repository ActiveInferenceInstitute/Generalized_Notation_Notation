# TO-DO — GNN next steps

Updated 2026-10-07. Baseline: [GNN 4.0.1](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/tag/v4.0.1).
Every checkbox below is remaining work. Each workstream must become focused PRs
with an explicit owner, source baseline, acceptance evidence and dependencies.
Remove landed work from this file; retain its evidence in the release receipts,
[CHANGELOG.md](CHANGELOG.md) and Git history.

## Scope and delivery order

There are **17 workstreams: five minor, seven medium and five major**. The sizes
describe implementation scope, not promised release versions or deadlines.
Existing S/M/E identifiers remain stable for links and related scope documents.
P1 denotes foundational correctness, security or capability gaps; P2 denotes
subsequent maintainability and expansion work. Size and priority are separate.

| Size | Scope | Release decision |
| --- | --- | --- |
| Minor | Bounded documentation, diagnostics, dependency verification or publication work | Fixes and documentation can land in a patch; an artifact-only archive need not change the software version. |
| Medium | Changes spanning shared modules, entrypoints, resource behavior or supported environments | Preserve supported public contracts; choose a patch or additive minor release from the actual behavior change. |
| Major | New scientific semantics, source-preserving LLM capability or deeper formal guarantees | Review the contract first. Additive capabilities may fit a minor release; breaking public contracts require a major version and migration guidance. |

Start with bounded M5/M4 work and S3 released-fix verification. Prioritize S5/S6
security and coverage improvements alongside the M1/M2/M3 ownership and contract
work. Establish M7 measurements before performance claims and M8 installation
evidence before adding supported environments. Define S1/E4 contracts early;
implement E1/E2/E3 only after their shared contracts and numerical acceptance
methods are ready. E5 archival work can proceed independently.

## Minor work

- [ ] **M5 — make maintained documentation consistent and runnable (P2).**
  Scope: README, SPEC, ARCHITECTURE, AGENTS, module guides, examples and API docs;
  clarify discovery versus selected views, current-run aggregation and supported
  versus experimental capabilities. Derive inventories from the live registries.
  Accept: commands exercised in their declared environments, correct paths and
  anchors, consistent terminology and passing documentation-contract audits.
  Deliver bounded documentation PRs; source or public-contract repairs belong in
  the relevant medium or major workstream.

- [ ] **M4 — reduce diagnostics and dependency ratchets (P2).**
  Scope: inventory broad exceptions, hidden failure reasons, flag-documentation
  gaps and duplicated optional-dependency declarations; repair one cause per PR
  with typed outcomes and useful context. Accept: observable error and import
  behavior preserved, meaningful failure cases and passing existing flag,
  import-boundary, thin-orchestrator and dependency gates. Shared configuration
  redesign belongs to M3.

- [ ] **S3 — verify a released THRML structural-zero fix (P1).**
  Scope: identify a released upstream version addressing
  [inactive categorical padding](https://github.com/extropic-ai/thrml/issues/72),
  evaluate its compatibility and test exact-zero/inactive-padding behavior on
  both supported JAX splits. Accept: independent numerical witnesses, no NaNs,
  unchanged source/seed/sample semantics and ordinary locked installation.
  Keep strict rejection until those checks pass. Depends on a suitable released
  fix; substantial adapter or inference changes require separate scope under E2.

- [ ] **M6 — improve scientific visualization and manuscript clarity (P2).**
  Scope: labels, units, axes, accessible legends, animation trace identity,
  categorical versus Gaussian uncertainty and VFE iterations versus EFE
  timesteps. Accept: validated plotted data, source and artifact hashes, finite
  manual/browser review with explicit coverage, and fresh template/custody
  acceptance when manuscript inputs or counts change. Small presentation fixes
  must preserve the scientific contract; semantic changes depend on E4.

- [ ] **E5 — archive the exact released version and verify its DOI (P2).**
  Scope: archive the released tag, distribution artifacts, manuscript, checksums
  and citation metadata; identify the version-specific archive separately from
  the project concept DOI. Accept: public archive/DOI resolution, matching
  version metadata and directly downloaded artifact hash parity. A PyPI delivery
  is a separate publication task with an identified package owner/destination,
  accepted wheel/sdist, installation verification and credential ownership.
  Depends on archive-provider availability and the selected release artifacts.

## Medium work

- [ ] **M1 — extract coherent module ownership boundaries (P2).**
  Scope: refresh the size/complexity inventory from actual source, reconcile
  existing extractions, then prioritize RxInfer bridge/strategy/analysis,
  website generation, GUI, schema parsing, visualization, logging, execution
  and security owners. Accept: one justified boundary per focused PR, preserved
  public exports/signatures, behavioral parity and failure cases, clear
  ownership and measured complexity changes. Split responsibilities rather
  than files solely to satisfy a line-count target.

- [ ] **M2 — unify execution and result contracts (P1).**
  Scope: shared run context, frozen selection, scope metadata, prerequisites,
  backend options, readiness diagnoses, model-kind adapters, outcomes and
  artifact indexes across serial, parallel and matrix execution. Accept: thin
  numbered orchestrators, equivalent plans/results and negative controls for
  explicit empty selection, inherited artifacts, duplicate aggregation and
  incompatible uncertainty. Establish the shared boundaries alongside M1;
  public schema changes require a compatibility decision.

- [ ] **M3 — align Python, CLI, REST and MCP admission/discovery (P1).**
  Scope: derive backend inventory, help, setup groups and schemas from shared
  live metadata; unify option precedence, types, unknown-key handling, limits
  and supported/unsupported diagnoses. Accept: equivalent admission and
  outcomes through the actual public entrypoints, installed-package facade
  isolation and documented migration for any changed public contract.
  Depends on M2 for execution/result behavior; metadata registration alone
  does not establish native backend readiness.

- [ ] **S5 — strengthen filesystem and platform boundaries (P1).**
  Scope: define the filesystem adversary model, use descriptor-based operations
  where concurrent path replacement matters, and scope native Windows lease,
  process and cleanup behavior. Accept: real symlink/rename/race controls,
  native platform tests, preserved API errors and bounded cancellation/cleanup,
  plus independent security review. State the precise confinement guarantees
  and remaining limits. Depends on appropriate native platform runners and
  M2 where lease/process behavior crosses shared execution contracts.

- [ ] **S6 — raise coverage through meaningful behavior checks (P1).**
  Scope: produce a reproducible coverage-gap inventory, then target untested
  failure paths and shared contracts, including optional and live surfaces in
  explicitly provisioned lanes. Target the documented **>80% statement coverage**
  goal on the declared supported matrix. Accept: per-environment reports,
  externally observable behavior checks and qualified exclusions; raise the
  enforced floor only after the evidence supports it. Keep overlapping
  selections separate and avoid tests that merely mirror implementation.

- [ ] **M7 — measure and improve scalability and performance (P2).**
  Scope: matched source/configuration/backend/LLM-mode corpora, phase timings,
  CPU/wall time, RSS, artifact counts and resource admission. Include nested
  and large inputs, repeated/concurrent runs and bounded distributed transfer.
  Optimize demonstrably dominant bottlenecks after accepting the baseline.
  Accept: comparable before/after receipts, explicit variance, complete selected
  coverage and measured tradeoffs. Preserve output
  semantics; estimated allocation and disk guards are separate from measured
  RSS/JIT behavior. Depends on M2's identity/result contracts.

- [ ] **M8 — broaden installed-package and platform acceptance (P2).**
  Scope: assess Python 3.14 scientific-wheel readiness and any additional
  supported OS/runtime split using ordinary installation outside the checkout.
  Retain both current JAX/Matplotlib splits, API scratch isolation, registry
  concurrency and tracked-file cleanliness. Accept: genuine imports/native
  execution, supported-platform cancellation and memory controls under load,
  and meaningful negative cases. GUI, network, audio and external toolchains
  remain explicitly provisioned lanes. Depends on released dependency
  compatibility and native runners; do not advertise support before acceptance.

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
semantics, the 25-step workflow and explicit unsupported outcomes. Autonomous
mode remains proposal-only unless a separately scoped contract changes it.

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
