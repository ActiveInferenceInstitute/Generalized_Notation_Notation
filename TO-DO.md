# TO-DO — GNN next steps

Updated 2026-11-27.
Current software version is owned by [pyproject.toml](pyproject.toml).
[GitHub release records](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases)
own exact tags, accepted sources, checks and public artifacts. Version metadata
or a target below does not establish publication.
Every checkbox below is remaining work. Remove accepted bounded work while
retaining its evidence in [CHANGELOG.md](CHANGELOG.md), release receipts and Git
history. Historical failures retain their original source epochs.

## Scope and delivery order

The scientific and release roadmap has **seven remaining workstreams: two
externally blocked minor items and five major extensions**.
The maintainability residuals are complete; their evidence is recorded in
[CHANGELOG.md](CHANGELOG.md). The [package improvements](#package-improvements)
below have their own acceptance criteria and follow the scientific prerequisites
where they touch those workstreams.
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

The [remaining roadmap contracts](docs/development/remaining_roadmap_contracts.md)
specify the next bounded units for all seven items. The 2026-10-16 bounded review added
terminal-stop LLM completion admission and consistent direct cpomdp options;
these corrections do not complete S1 or E3. The THRML upstream issue is closed
(2026-10-05), but the inspected latest PyPI release remains 0.1.4 (rechecked
2026-11-12; the project's release list still holds only 0.1.3 and 0.1.4), so S3 stays blocked on
released compatibility. Full source-preserving multi-request processing,
independent semantic extraction and coupled/native extension acceptance remain
required.

The 2026-11-15 integration added the `gnn_llm_request_plan_v1` request plan,
the finite `gnn-semantic-witness-v4` smoothing witness and ontology alignment
diagnostics ([CHANGELOG.md](CHANGELOG.md)). None of these completes a
workstream. E4 formal evidence stays unavailable.

The 2026-11-16 Wave 3B change wires that request plan into the corpus
scheduler as the opt-in `request_plan` option of `process_llm`
(`src/gnn/llm/plan_scheduler.py`). With it, the scheduler builds the plan, keeps
per-window checkpoints, preserves byte offsets, and admits replies only on
terminal stop. Offline tests use injected replies and confirm three things:
- windows reconstruct the original bytes;
- a resumed run issues only the unfinished requests;
- a truncated or uncounted reply is refused and not checkpointed.

This offline wiring is complete, but S1 stays open:
- live multi-window acceptance against an external provider has not been run;
- native terminal-completion evidence is still missing;
- corpus `joint_source` admission refuses until a native tokenizer count exists.

The ontology pin now targets Active_Inference_Ontology
`ad808b12d51ec990227d52c528f62db36dfd54bc` (PR #19). Its published terms resolve
`LikelihoodMapping`, `TransitionMatrix` and `Preferences`. Of the 64 declared
labels, 15 resolve and 49 remain `candidate_only` alignments. `import gnn` and
`import gnn.ontology` no longer require `psutil` (#400, item 2).

## Published coordinated checkpoint

[GNN 5.1.0](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/tag/v5.1.0)
is the current published release: tag `v5.1.0` at commit
`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf` (2026-11-12), the first tagged tree
that carries the one importer reader (`src/gnn/importers.py`), so a member can
advance its pin past a release and call that reader. The record names the exact
commit and uploads no artifacts; the
[5.0.1 release](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/tag/v5.0.1)
remains the latest release with uploaded source-bound assets.
[docs/downstream/importers.json](docs/downstream/importers.json) records
`v5.1.0` for every member with a pin file. A member's pin move belongs to its
own change; refresh its row once that change merges.

[GNN 5.0.1](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/tag/v5.0.1),
[FEP_Formal 2.1.0](https://github.com/ActiveInferenceInstitute/fep_formal/releases/tag/v2.1.0)
and [GEO-INFER 0.4.0](https://github.com/ActiveInferenceInstitute/GEO_INFER/releases/tag/v0.4.0)
are published. Their exact accepted sources, artifacts, companion pins and
native evidence retain their original identities. The earlier coordinated-major
release preparation is complete; it is not a remaining scientific workstream.
New source changes follow the active
[paired-revision procedure](docs/development/fep_lean_paired_revision.md).

GNN's software source archive is version DOI
[10.5281/zenodo.23631909](https://doi.org/10.5281/zenodo.23631909).
The FEP_Formal software source archive is version DOI
[10.5281/zenodo.23631895](https://doi.org/10.5281/zenodo.23631895).
These software records are separate from the FEP manuscript deposit and from
E5's historical GNN 4.1.0 distribution-inventory requirement.

<a id="versioned-delivery-plan-430-through-490"></a>

## Current delivery priorities

The former 4.3.0–4.9.0 sequence was a planning horizon. It does not describe
unpublished releases to produce after 5.0.1. Use the stable workstream IDs below
and assign a new version only after the actual change and compatibility review.
Accepted foundations remain described in the
[roadmap contracts](docs/development/remaining_roadmap_contracts.md).

| Order | Workstream and next bounded unit | Acceptance before integration |
|---|---|---|
| 1 | **E4:** independent source-bound numerical semantics for explicitly admitted compositions | Independent extraction and numerical references, asymmetric dimensions, full covariance, identity/axis/conditioning counterexamples, separate formal and native evidence, current companion custody |
| 2 | **S1:** explicit joint synthesis with source and response identity, declared context admission and resumable checkpoints | Every selected byte and required prompt accounted for; supported provider contract and genuine native terminal completion; no silent truncation, fallback or manufactured completion; cancellation, budget, refusal and resume controls |
| 3 | **E1:** coupled continuous-agent semantics beyond independent Gaussian products | Explicit coupling and control identity, cross-covariance, independent reference, zero-coupling and permutation controls; unsupported compositions refuse |
| 4 | **E3:** separately specified cpomdp search beyond accepted finite controls | Complete policy/component receipts against independent Kalman/EFE references; action order, ties/all-invalid rejection and total resource bounds |
| External prerequisite | **S3:** qualify a suitable ordinary released THRML structural-zero fix | Locked compatibility on supported JAX splits, independent exact-zero/inactive-padding witnesses and no changed source/seed/sample semantics |
| After its prerequisites | **E2:** separately specified coupled finite THRML inference | Independent native categorical witnesses and admission/resource controls; continuous models, search, hardware, energy and speed retain separate scopes |
| Archive-owner access | **E5:** complete the historical exact-version distribution inventory | Preserve existing source ZIP/DOI; add only accepted version-matching distributions/checksums where provider controls permit, then verify every public download |

E5 can proceed independently when archive access permits. S3 and dependent E2
scope remain blocked without suitable released compatibility. Do not substitute
an epsilon, a private upstream patch or relabeled historical evidence. Review
S1 context/synthesis and E3 search designs before implementation; independent
fragment answers and exhaustive enumeration retain narrower accepted contracts.

### Release iteration and CI work across every minor

Keep one deterministic preflight before dispatch, one dispatch owner, and the
exact-source workflow collector/artifact acceptance from the
[release procedure](docs/releases/README.md#repeatable-release-qualification).
For each target:

- **No hosted qualification is possible while the organization's billing lock
  stands.** The test-bearing workflows dispatched on `main` on 2026-11-12 all
  failed within seconds because their jobs never started:
  [CI 43274809991](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/actions/runs/43274809991),
  [Local gates 43274809976](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/actions/runs/43274809976),
  [Installed Platforms 43274810045](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/actions/runs/43274810045)
  and [native coverage 43274809920](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/actions/runs/43274809920)
  each report "The job was not started because your account is locked due to a
  billing issue"; the only successful runs among the hundred most recent are
  Dependabot and Dependency Graph updates, which execute no repository test.
  Qualify changes with local `./run.sh gates` receipts and record them as local;
  the organization's billing owner must clear the lock before any hosted result
  can be cited here or in a release record.
- Re-derive the installed matrix for the current PyMDP and Python requirements.
  The lock resolves `inferactively-pymdp==1.0.5` and `jax==0.12.0.1` across
  the supported CPython 3.12–3.14 range. The Installed Platforms lane produced
  no receipt (run
  [43274810045](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/actions/runs/43274810045)
  failed to start), so the installed-platform acceptance this file requires
  before publication is still outstanding and locally owned.
- Record cold/warm setup, queue, job and test durations from provider/JUnit
  receipts; identify the measured bottleneck before changing scheduling.
- Reuse frozen installations within a job and cache toolchain inputs keyed to
  their locks. Always rerun actual tests and source-bound artifact checks.
- Consolidate demonstrably duplicated checks under one workflow owner; preserve
  required status aggregation, failure diagnostics, marker selections, coverage
  denominators and native/platform/companion profiles.
- Compare wall-clock and runner-time receipts before/after on equivalent sources
  and environments. Accept speed claims only at that measured scope.
- Keep main/tag qualification concurrent where the existing exact-tree process
  permits it. Every ref/event still needs its own successful terminal results
  and independent artifact acceptance before publication.

Detailed scientific contracts remain in
[remaining_roadmap_contracts.md](docs/development/remaining_roadmap_contracts.md).
The lists below own outstanding work; this table owns target-version sequencing.

## Minor work

- [ ] **S3 — verify a released THRML structural-zero fix (P1).**
  Scope: identify a released upstream version addressing
  [inactive categorical padding](https://github.com/extropic-ai/thrml/issues/72),
  evaluate its compatibility and test exact-zero/inactive-padding behavior on
  the supported CPython and JAX matrix. Accept: independent numerical witnesses,
  no NaNs, unchanged source/seed/sample semantics and ordinary locked installation.
  Keep strict rejection until those checks pass. Depends on a suitable released
  fix; substantial adapter or inference changes require separate scope under E2.

- [ ] **E5 — archive the exact released version and verify its DOI (P2).**
  Scope: archive the released tag, distribution artifacts, manuscript, checksums
  and citation metadata; identify the version-specific archive separately from
  the project concept DOI. Accept: public archive/DOI resolution, matching
  version metadata and directly downloaded artifact hash parity. A PyPI delivery
  is a separate publication task with an identified package owner/destination,
  accepted wheel/sdist, installation verification and credential ownership.
  The [v4.1.0 source archive](https://zenodo.org/records/23245091) resolves
  at version DOI `10.5281/zenodo.23245091`. On 2026-10-16, its 3,289 source blobs
  matched tagged Git bytes; the archived manuscript matched the public release.
  Public wheel/sdist/manuscript downloads also matched published SHA-256 sums.
  Remaining: archive the separate wheel, sdist and distribution checksums with
  the version record, then verify their public downloads and complete inventory.
  The current Zenodo source archive contains none of those distribution files;
  completing that operation requires archive-owner access.

## Major work

- [ ] **S1 — complete source-preserving long-context LLM processing (P1).**
  Progress: byte-preserving multipart prompts retain native fragment responses,
  exact part receipts and resumable checkpoints. The unreleased opt-in
  [synthesis modes](docs/llm/README.md#explicit-cross-part-synthesis-and-resume)
  add a final exact-source request (`joint_source`) or explicitly derived
  fragment evidence (`part_evidence`), with source/request/response dependency
  hashes, reviewed Ollama context admission and strict native-stop resume.
  A bounded explicit Qwen profile completed 21 requests over two source parts,
  returned both controlled cross-part facts and resumed with no new generations.
  Default Smollm partial runs and the Gemma controlled-fact failure remain
  retained; the default model is unchanged. This establishes the admitted local
  profile, not general scientific interpretation or independent attention.
  The opt-in corpus `request_plan` scheduler (`gnn_llm_request_plan_v1`) now
  writes per-window checkpoints, enforces the tighter request timeout, and
  refuses terminal replies received after cancellation or the plan deadline.
  The corpus scheduler has offline qualification.
  Remaining: live external-provider multi-window acceptance; native tokenizer
  counts for corpus `joint_source` plans; broader large-source/provider and
  answer-quality acceptance; native `part_evidence` qualification. Every joint source and prompt must fit
  the selected context. Preserve exact provider/model/source identities,
  budgets, original bytes and explicit refusal/unfinished states. No silent
  truncation, sampling, model substitution or manufactured completion. Depends
  on verified model/provider admission and M2/M3 result/configuration contracts.

- [ ] **E1 — extend coupled multi-agent continuous semantics (P2).**
  Native JAX now filters two explicitly coupled Gaussian agents from fixed
  observations and controls, with full cross-covariance and independent
  prefix-conditioning, permutation and zero-coupling controls. Native RxInfer
  qualification is still required for this joint filter.
  Scope: reconcile existing continuous/factored/hybrid/nonstationary and
  independent Gaussian contracts, then specify additional coupling, control
  and observation semantics. Accept: explicit per-agent state/control identity,
  asymmetric dimensions, composed model-kind dispatch and independent native
  JAX/RxInfer exemplars with covariance-aware results. Unsupported compositions
  must refuse explicitly. Depends on M2/M3 and E4's semantic witness method;
  compatible shapes alone do not establish an inference contract.

- [ ] **E2 — expand THRML through separate scientific contracts (P2).**
  Native THRML 0.1.4 now has a bounded, strictly positive two-agent CPU
  smoothing contract with fixed observations and actions. Python 3.12 and
  3.14 witnesses compare the retained draws with exact joint enumeration.
  Scope: evaluate coupled agents, policy/action search, continuous/hybrid
  models and hardware execution as separately scoped extensions of the
  [THRML adapter](docs/gnn/implementations/thrml.md). Accept: defensible model and
  inference mappings, released compatibility on the supported runtime matrix,
  admission/failure/resource controls and independent native witnesses.
  Hardware, convergence, energy and speed claims each require their own direct
  evidence. Depends on S3 where structural zeros are required, plus M2/M3/E4.

- [ ] **E3 — extend cpomdp control/search capabilities safely (P2).**
  Progress: E3-FA1 adds source-declared ordered finite actions, explicit control
  matrices and units, complete policy/component receipts and a total evaluation
  cap; its independent numerical witnesses retain a separate bounded scope.
  The generated runner, lock, adapter guide and native tests agree on
  `cpomdp==0.5.1`. The bounded review exercised the installed passive, parity
  and EFE fixtures and the readiness refusal for an incompatible runtime.
  Source-declared explicit candidate sequences also passed independent Kalman
  and EFE comparisons on the same released runtime, including policy identity
  and stable authored ties.
  The earlier 0.5.0 adoption was reviewed against the FEP_Formal merge
  `9d77cd47fd170e1e4071c0944d3c5ee3cb7a4886` (#296). The later qualified
  companion revision is recorded in [`.github/fep-lean-pair.json`](.github/fep-lean-pair.json).
  Remaining scope: specify further control/search behavior through the
  [cpomdp adapter](docs/gnn/implementations/cpomdp.md), preserving passive
  no-search behavior, explicit EFE defaults and complete policy/source identity.
  Accept: released constructor compatibility, finite/nonboolean options,
  all-invalid rejection, independent Kalman/EFE witnesses, complete score and
  policy receipts, and bounded evaluation/allocation/process behavior.
  Never silently reduce requested search; allocation estimates do not certify
  total RSS. Depends on M2/M3, E4's numerical contracts and M7 resource evidence.

- [ ] **E4 — bind formal statements to numerical semantics (P1).**
  Progress: the 4.3.0 [witness foundation](docs/development/remaining_roadmap_contracts.md#e4-foundation-source-bound-numerical-witness-v1)
  independently reads a restricted source subset, enumerates finite paths and
  conditions joint Gaussians. Native bnlearn/JAX comparisons and counterexamples
  cover their declared filtering fixtures. Opt-in native bnlearn smoothing
  adds one-step and four-step [v4 replay](docs/development/remaining_roadmap_contracts.md#e4-replayable-finite-smoothing-v4)
  on the restricted fixed-action source contract. The unreleased
  [E4-IG1 composition extension](docs/development/e4_independent_gaussian_composition.md)
  adds independent extraction and joint-prefix conditioning for two passive
  Gaussian agents, full retained child metadata and separate FEP Kalman/Joseph
  replay. Its finite product-weight algebra has separate native proof controls;
  it is not a Gaussian conditioning or program-equivalence theorem. Broader
  source contracts, coupled models, algorithms and formal equivalence remain open.
  Scope: specify independent extraction and semantic witnesses for declared
  finite/continuous models, translations, inference algorithms and precision
  assumptions beyond source/hash/value/shape custody. Accept: exact assumptions,
  compatible numerical comparisons, counterexamples/refusals for mismatched
  semantics and source-bound formal/native/runtime evidence. Keep Lean theorem
  statements, generated-runner execution and bridge custody distinct; universal
  equivalence requires an actual proof. **E4-BN1** now has stationary finite A/B/D CPT compilation and independent
  native fixed-action filtering witnesses. Composition, causal discovery and
  policy optimization remain outside that narrow contract; runtime success
  alone cannot establish those translations. Coordinate changed FEP/GEO owners through
  the paired-revision procedure rather than relabelling older receipts.

## Maintainability residuals

The bounded maintainability residuals are complete; their evidence is recorded
in [CHANGELOG.md](CHANGELOG.md).

## Package improvements

The November 15 review used GNN main at `d18d8095d2f1`. Its static Python
inventory covered 781 files under `src/gnn` and 219 distinct import edges
between 50 groups. Groups combine numbered entry points and distinguish
top-level packages and support modules. The count includes imports inside
functions and optional branches and excludes type-checking-only imports.
Read the imports together with the
[step registry](src/gnn/pipeline/step_registry.py),
[framework registry](src/gnn/render/framework_registry.py),
[MCP discovery](src/gnn/mcp/discovery.py) and
[dependency declarations](pyproject.toml) for string-based imports, subprocess
toolchains, package extras and artifact dependencies.

| Boundary | Packages and contracts to exercise |
| --- | --- |
| Source, notation and compliance | `parsers`, `grammars`, `types`, `schema`, `schemas`, `schema_validator`, `type_systems`, `type_checker`, `validation`, `ontology`, `formal_specs`, `extract`, `processing`, `model_registry`, `gnn_examples` and the [importer reader](src/gnn/importers.py). |
| Scheduling and run ownership | `pipeline`, `multimodel`, `template`, `setup`, `testing`, `security` and `utils` own selection, environment, policy, execution support or custody across the numbered steps. |
| Rendering, execution and interchange | `render`, `execute` and `rxinfer_interchange` connect to PyMDP/JAX, the Python scientific extras, Julia and CmdStan runtimes, and the [FEP_Formal and GEO_INFER paired contracts](docs/development/fep_lean_paired_revision.md). The [framework list](src/gnn/frameworks.py) owns backend names, including the execution-only Lean bridge. |
| Results and presentation | `analysis`, `integration`, `report`, `intelligent_analysis`, `visualization`, `advanced_visualization`, `audio`, `llm`, `ml_integration`, `research`, `website`, `doc`, `documentation` and `manuscript` have distinct result, enrichment, documentation or publication responsibilities. |
| Entry points and editor surfaces | `cli`, `api`, `mcp`, `lsp` and `gui` expose different transports over shared model and pipeline behavior. Use the [interoperability contracts](docs/development/interop_api.md) to identify installed import paths and response owners. |
| Public downstreams | The [importer registry](docs/downstream/importers.json) and [workflow roster](docs/workflows/gnn-fep.md) track pins, specs and conformance. The [PyMDP parser bridge](https://github.com/ActiveInferenceInstitute/pymdp/blob/1cccd9e159b59377c9a8dba6c0c954fb2efb2b4e/pymdp/gnn.py), [COGANT pipeline bridge](https://github.com/ActiveInferenceInstitute/COGANT/blob/5a3833e08d916c9a5040f754fdf4de4f4c444331/cogant/py/cogant/gnn/upstream_bridge/pipeline.py) and [Active_Fedference backend](https://github.com/ActiveInferenceInstitute/Active_Fedference/blob/2089dd3cf3a4d9c0881256da5c9df4f8c280c240/src/fedference/gnn/backend.py) also exercise GNN behavior. Record the actual source, wheel, subprocess or interchange boundary for each consumer. |

### Minor package improvements

No minor package improvement remains open. Importer evidence categories and
unavailable-observation reporting are recorded in [CHANGELOG.md](CHANGELOG.md).

### Medium package improvements

- [ ] **Qualify the remaining wheel runtime profiles (P1).** Local Linux checks
  qualified ordinary base wheel and sdist installs, the API and LSP profiles,
  native Gradio and Pillow, and strict downstream typing with `gnn/py.typed`.
  The [installed-platform lane](.github/workflows/installed-platforms.yml)
  retains Linux, macOS and Windows coverage and includes CPython 3.13 on Linux.
  Qualify the exact release through the hosted matrix and record actual
  runtime, backend and missing-extra evidence for scientific profiles.
- [ ] **Qualify affected importer upgrades one consumer at a time (P1).**
  Use the [canonical released reader](src/gnn/importers.py) on
  the member's own pin and specs before retiring local conformance code.
  Use the measured dtype, device and agent identity already recorded in the
  [PyMDP simulation receipt](src/gnn/execute/pymdp/SPEC.md).
  Active_Fedference now calls the public `gnn.execute.pymdp` entry point and
  passed its locked runtime profile. Use the PyMDP parser bridge and COGANT's
  subprocess contract as independent compatibility checks for future changes.

### Major package improvements

Reports, read-only replay and explicit resume now use the verified run evidence
described in the [run ownership guide](docs/development/run_ownership_migration.md).

- [ ] **Reduce the remaining import surface (P2).** `lint-imports`
  enforces twelve contracts ([contracts](pyproject.toml),
  [direction tests](tests/architecture/test_dependency_direction.py)):
  `gnn.api` and `gnn.operations` below `gnn.cli`; `gnn.pipeline` below
  `gnn.report`; parsers, types and importers below both transports,
  `gnn.report` and `gnn.execute`; `gnn.pipeline`, `gnn.setup` and
  `gnn.security` below `gnn.execute`; `gnn.execute`, `gnn.setup` and
  `gnn.llm` below `gnn.api`; and `gnn.extract` below `gnn.render`. Eleven
  reciprocal package pairs remain, counting
  function-scope imports; `KNOWN_RECIPROCAL_PAIRS` in the direction tests
  freezes the set:
  - `execute`/`utils`, `pipeline`/`utils` and `processing`/`utils`: the
    orchestration helpers under `gnn.utils.pipeline_orchestration` and the
    call-time runtime bridges reach upward.
  - `analysis`/`execute`, `analysis`/`pipeline`, `pipeline`/`processing`,
    `main`/`pipeline`: result adapters, the model-family ledger and run
    identity cross in both directions.
  - `parsers`/`types`, `parsers`/`schema_validator`, `parsers`/`utils`,
    `schema_validator`/`testing`: notation-side pairs, each closed today by a
    function-scope import. `GNNValidator` builds the round-trip tester from
    `gnn.testing` when a caller enables it, and the MCP validation tools pass
    that flag through, so cutting `schema_validator`/`testing` needs an
    injected tester without changing what those tools return.

  Add a contract for each pair cut, and preserve public re-export identity and
  installed entry points with fresh-process checks for parser-only, CLI, API
  and scientific consumers. Change extras or distribution boundaries only
  after the installed profiles and downstream probes pass.
- [ ] **Share versioned benchmark evidence across projects (P2).** Qualify
  an independent public consumer of the GNN scaling manifest in the
  [research contracts](docs/development/interop_research.md). Record exact
  producer and consumer revisions, model/fixture digests, axes, seeds where
  applicable, units, toolchain and unsupported outcomes. Verify fixture
  interpretation and refusals independently, preserve the original results,
  and qualify timing or result adapters with their own evidence. Bind changes
  to paired FEP or GEO owners through the existing revision procedure. E4's
  native numerical acceptance retains its own criteria.

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
uv run --frozen --no-sync ruff check src/gnn scripts tests docs
uv run --frozen --no-sync ruff format --check src scripts tests docs
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
custody ritual in [AGENTS.md](AGENTS.md); `just manuscript-custody` runs the dry
half of that ritual (token map, hydrated prose, figure and render freshness) and
`just manuscript-figures` rebuilds the figures — both run inside `just quality`. Keep local checks, hosted tests,
installation, native numerical witnesses and finite visual review separate;
report source identities, failures, optional skips and actual review coverage.

Merge through the existing required gates, preserve unrelated work and inspect
tracked-file bookends. Publication requires accepted exact-source checks, tag
and remote parity, verified distribution contents and public download hashes.
Retain credentials and private evidence locally. No force push, amended custody,
silent normalization, fallback success or unsupported readiness claims.
