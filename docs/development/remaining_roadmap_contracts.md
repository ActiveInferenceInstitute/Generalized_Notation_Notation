# Remaining roadmap contracts

This document specifies implementation boundaries and acceptance cases. The
original 2026-10-16 design review used GNN
`2b105fde2c0f8fcd50eec5a45ab22d9925c01383`; the E4 foundation section records
the later 4.3.0 bounded implementation. It does not complete the seven items in
[TO-DO](../../TO-DO.md), certify a release, or replace [scientific admission](backend_scientific_contracts.md) and
[run ownership](run_ownership_migration.md).

Target-version sequencing for **4.3.0–4.9.0** is owned by
[TO-DO.md](../../TO-DO.md#versioned-delivery-plan-430-through-490).
These contracts define acceptance boundaries and do not certify those targets
as implemented. Historical numerical/archive observations keep their dated scope.

## Ownership and delivery

| Item | Implementation owners | Next independently reviewable unit |
| --- | --- | --- |
| S1 | Step 13 corpus scheduler, request worker, provider adapters | Source-preserving request-plan admission and segment checkpoints |
| E4 | Extraction, semantic fidelity, analysis, FEP/GEO companions | Independent source-to-numerics witness schema and negative controls |
| E1 | Composed-kind dispatch and JAX/RxInfer Gaussian renderers | Explicit joint Gaussian coupling contract |
| E2 | THRML admission, payload compiler, runtime and analysis | One separately named inference extension at a time |
| E3 | cpomdp option admission, generator, executor and analysis | Explicit finite action-set contract with bounded exhaustive search |
| S3 | THRML dependency and native scientific tests | Verify a suitable ordinary released wheel on both lock splits |
| E5 | Release artifacts, citation metadata and archive receipt | Verify the exact version archive and public artifact hashes |

Content changes to bridge-bound owners require the
[paired revision procedure](fep_lean_paired_revision.md). Count-changing manuscript
inputs require the complete render ritual. Historical receipts retain their
original source identities; a local patch cannot refresh hosted or companion
acceptance by changing metadata.

## S1: complete multi-request processing

The current whole-context path remains authoritative. Context refusal is
unfinished work. A token-limited reply is retained with `incomplete` status and
cannot enter completion checkpoints or the cache. The review patch binds that
rule through `terminal_stop_v1` and request-cache version 3.

The next implementation should admit an immutable request plan before issuing
provider calls. Each plan must contain:

- Exact source-byte digest, encoding, byte length and ordered segment offsets.
  Reading, reassembly and hashing must preserve newlines, Unicode and the final
  byte. Segments must cover the source exactly once, without gaps or overlaps.
- Exact provider, endpoint identity, model identity and prompt-definition digest.
  Context capacity and tokenizer/chat-template identity must have verifiable
  origins. Character counts and guessed tokens cannot certify context admission.
- The context reservation for system messages, task instructions, prior segment
  receipts and output generation. Every request must fit its admitted budget.
- Every selected source/prompt unit, its ordered segment requests and synthesis
  dependencies. A segment summary is evidence about that segment; it cannot
  satisfy a task requiring the entire original source by itself.
- A versioned task contract identifying what synthesis may conclude from segment
  outputs. Tasks requiring simultaneous full-source context must refuse when it
  cannot fit. Segmentation alone does not preserve arbitrary reasoning semantics.
- The invocation's monotonic deadline, per-request limit and cancellation token.
  No automatic model substitution, truncation or budget increase is permitted.

Checkpoint each segment response with its request digest, content digest,
provider/model receipt and terminal finish reason. Revalidate those records on
resume before scheduling dependent requests. Only a complete, accepted synthesis
may complete the original prompt unit. Interrupted segments, failed synthesis,
missing dependencies and late publication remain unfinished. Cache identity must
include the complete request plan and completion policy.

Acceptance must include exact reconstruction of CRLF and Unicode sources,
changed source/model/endpoint/tokenizer/prompt rejection, interrupted segment and
synthesis resume, duplicate stems, cancellation, exhausted context/output budgets,
corrupt checkpoints, and genuine provider runs beyond one context window. Test
injections validate scheduling only. S1 remains open until native coverage is
complete for every required source/prompt in the selected acceptance corpus.

### S1-RP1: immutable request plan v1

[`llm/request_plan.py`](../../src/gnn/llm/request_plan.py) implements the plan
fields above as `gnn_llm_request_plan_v1`. It records the exact source digest,
byte length and contiguous UTF-8 segment offsets, together with the provider,
endpoint, model and prompt digests. It also records the reviewed context
capacity and template identity, the four context reservations, the
`part_evidence` or `joint_source` task contract, deadlines, and the no
truncation, substitution or budget-increase policy. `joint_source` refuses
without a matching native-tokenizer count or when it cannot fit.

Segment and synthesis checkpoints bind the plan digest, request digest and a
`terminal_stop_v1` outcome. Resume revalidation rejects changed plan fields,
corrupt or conflicting checkpoints, incomplete replies and stale synthesis
dependencies. Tests cover CRLF, BOM, Unicode and trailing-byte reconstruction,
changed source/model/endpoint/template/prompt rejection, interrupted segment and
synthesis resume, cancellation and exhausted context or deadline budgets.

This unit is not yet wired into the corpus scheduler, request worker, cache or
existing checkpoint versions. Its tests inject outcomes. Duplicate-stem corpus
handling and genuine provider runs beyond one context window are still
unverified, so S1 remains open.

## E4: independent semantic witnesses

Source hashes, matching values and shapes are necessary identity checks. They do
not prove extraction correctness or equivalence between inference algorithms.
The proposed witness should record:

- Source digest and exact declared equations, variable/axis identities, initial
  time convention, parameters, observations, actions and conditioning horizon.
- Independently extracted reference parameters. A reference that calls the same
  extractor or copies generated parameters cannot detect that extractor's error.
- The named inference question: filtering, smoothing, prediction, policy scoring
  or control. Preserve log-preference sign, EFE convention and policy/action axes.
- Reference algorithm and provenance, native library/runtime versions, generated
  script digest, dtype and justified absolute/relative or Monte Carlo tolerances.
- Formal statement and assumptions, theorem/proof identity and bridge revision as
  separate optional fields. Missing proof evidence stays unavailable.
- Per-quantity comparisons and explicit refusal reasons. A global Boolean must
  not erase unavailable or incompatible comparisons.

For finite models, begin with independent path enumeration on a small asymmetric
HMM. Compare filtering and full-sequence smoothing separately on the same fixed
observations/actions. For Gaussian models, begin with independent NumPy Kalman
updates using asymmetric state/observation dimensions, nonidentity observation
matrices, nonzero means and correlated positive-definite covariances. Check the
first observation against the declared prior without an extra predict step.
Existing cpomdp NumPy Kalman/EFE and THRML log-space smoothing tests are useful
numerical controls; they do not establish independent source extraction or Lean
proofs.

Negative controls must swap observation/state axes, perturb one transition,
shift an action in time, add an extra predict step, change conditioning horizon,
change EFE sign, omit cross covariance, substitute a source digest and change
precision. Each must fail or refuse with a quantity-specific diagnosis. Add a
new witness version before extending its scientific meaning.

### E4 foundation: source-bound numerical witness v1

The [test-only reference reader](../../tests/helpers/numerical_witness.py) and
[actual source-to-native suite](../../tests/render/test_source_numerical_witnesses.py)
implement one deliberately restricted subset. The reader imports no GNN parser,
canonicalizer, renderer or backend. It accepts one-line JSON-valued assignments in
`InitialParameterization`, colon-delimited scalar `ModelParameters`, exact declared
equations/state identities and a versioned `NumericalWitness` JSON section. Other
source grammars and inference contracts refuse explicitly; this is not a second
complete GNN parser. Syntax accepts CRLF/LF lines while source-byte hashes retain
the untouched bytes, including line endings and the final newline.

- **Finite:** one stationary categorical state, observation and exogenous action
  factor; canonical `A[observation,state]` and
  `B[next_state,previous_state,action]`; normalized A/B/D and zero C. Exact latent
  path enumeration computes each filtering prefix independently and full-sequence
  smoothing separately. The asymmetric native fixture has two states, three
  observations, two transition actions and four observation times. Native bnlearn
  posterior comparisons cover filtering only; smoothing remains reference evidence.
- **Gaussian:** one passive stationary LGSSM, with no authored control or policy
  contract. The asymmetric fixture has three states, two observations, a
  nonidentity rectangular H, nonzero prior means and correlated positive-definite
  prior/Q/R. The reference constructs the full joint Gaussian from independent
  initial/process innovations and conditions each observation prefix. It does not
  call a Kalman update. Every mean and full covariance is compared with the actual
  generated JAX program. Its first observation conditions the declared prior;
  generated one-based time notation maps to result index zero. The test inspects
  actual JAX float64 arrays and x64 configuration in the execution process.

Reader limits are eight observation times, four state/observation dimensions,
four finite actions and 4,096 enumerated paths. Deterministic comparisons use
float64, absolute tolerance 1e-12 and relative tolerance 1e-10 for these small,
well-conditioned fixtures. These bounds are deliberately wider than observed
roundoff; they are not Monte Carlo tolerances or permission to substitute float32.
Records name source, axes, ordered identities, equations, initial time, inference
question, every conditioning horizon, actual observations/actions and each compared
quantity. Native receipts additionally retain checkout/dirty state, generated
source/program/result bytes and hashes, plus actual Python/NumPy/framework
versions. All-extras CI
requires both receipts from its actual checkout and retains them with its JUnit
artifact; missing native execution cannot pass that receipt step.

Counterexamples cover transposed rectangular source matrices, changed transition
probabilities, shifted action timing, extra initial prediction, erased cross
covariance, covariance-only corruption, future conditioning, source substitution
and precision/axis mismatch. Unsupported EFE/sign questions refuse rather than
borrowing this filtering contract. The suite also rejects duplicate, nonfinite,
boolean, ambiguous and over-budget source declarations.

These finite tests establish their named numerical comparisons. Formal evidence
is explicitly unavailable in the witness schema. They do not refresh Q7/native
Lean receipts or synchronize FEP_Formal, and they do not prove general extraction,
backend-wide equivalence, planning, causal discovery, coupled models or arbitrary
GNN compositions. Broader E4 work remains open.

### E4 replayable semantic evidence v2

The additive `gnn witness` command and
[`validation/semantic_witness.py`](../../src/gnn/validation/semantic_witness.py)
make the restricted source contract above available outside the test suite.
The source reader imports no production parser, extractor, renderer or backend.
It independently sums finite latent paths or conditions joint Gaussian prefixes.
The independent test oracle remains separate; FEP_Formal's audit uses forward
Bayes and recursive Kalman/Joseph updates instead of those reference algorithms.
This extends evidence delivery and replay, not the admitted model semantics.

`gnn-semantic-witness-v2` retains exact UTF-8 source/result/program bytes and
digests, the independently read source contract, reference quantities, native
metadata, assumptions and per-quantity comparisons. Native outputs report their
actual dtype, runtime parameters, seed, axis convention, prior-first timing and
conditioning horizon. Missing declarations stay unavailable; contradictory
declarations refuse admission. Ordered source labels remain a declared mapping
of numeric axes; they are not manufactured native label provenance. Filtering
and smoothing remain distinct; the native adapters here admit filtering only.

Reference parameters must exactly match reported float64 runtime parameters.
Admitted posterior comparisons retain absolute tolerance `1e-12` and relative
tolerance `1e-10`, full covariance and explicit error diagnostics. A failed,
refused or unavailable admission cannot acquire posterior agreement from matching
arrays. Replay recomputes the complete receipt from retained raw artifacts,
including its reference; a coherently edited saved reference/result cannot borrow
an earlier passing verdict. `gnn witness --verify` additionally requires the
replayed numerical outcome to pass before returning success. Output paths cannot
alias source/result/program inputs.

The byte digest of a supplied program does not authenticate its execution.
`execution_source_verified` remains false and formal evidence remains unavailable
in GNN receipts. FEP_Formal's separate `fep-semantic-witness-audit-v1` replays the
source independently and records qualified theorem references, source digests,
unmatched assumptions and live checkout identities. General multidimensional
Gaussian filtering remains a structural analogue of the referenced scalar
formalism. Source custody, a targeted Lean compilation and numerical agreement
are separate observations; none establishes universal translation equivalence.
Historical pins/native receipts are preserved until coordinated qualification.

The companion's bounded `specs/gnn-semantic-witness-v1/semantic_laws.lean`
adds finite-weight normalization/reconstruction/nonnegativity and scalar
arbitrary-observation-gain measurement identities, Joseph/Schur covariance
agreement and posterior-variance positivity. Its native verifier retains the
positive compilation, axiom output and expected rejection of zero-normalizer
and omitted-observation-noise claims. These algebraic laws do not prove the
GNN parser, generated program, multidimensional filter or translation correct.
The website presents their compiler receipt separately from numerical replay.

Step 20 displays these records in the existing Analysis page with offline
filters, explicit per-quantity outcomes and separate proof/execution/custody
fields. The viewer labels saved receipts as unverified reported evidence; it
does not rerun a numerical algorithm or compiler in the browser. E4 remains open
for broader source grammars, model compositions, inference algorithms and proofs.

### E4 replayable finite smoothing v4

`gnn-semantic-witness-v4` adds one question:
`gnn_finite_fixed_action_smoother_v1`. This is full-sequence smoothing on the
restricted finite source contract with exogenous transition actions. Schema v3
already names witness compositions, and v2 filtering receipts and their
smoothing refusals are unchanged and still replay.

The reference comes from the raw source bytes by enumerating every complete
latent path, conditioned on the whole horizon (`conditioning_horizon` equals
`T` at every time), in float64. Comparisons use `atol=1e-12` and `rtol=1e-10`
and report per-time error. The receipt also retains the forward-filtering
posteriors as a named negative control. On the asymmetric fixture these differ
before the final time and agree only there. `gnn witness --verify` replays the
complete receipt, reference included.

Negative controls fail, refuse or raise with explicit diagnoses:

- filtering posteriors substituted for smoothing (`filtering_posteriors_substituted_for_smoothing`);
- filtering contracts, modes or questions declared for a smoothing source;
- a conditioning-horizon mismatch;
- a transposed rectangular `A` in either the result or the source;
- a perturbed `B`, whether declared or hidden;
- action timing shifted by one step (`action_timing_shifted_earlier`/`_later`);
- float32 values with and without a float32 declaration (`float32_precision_rounding`);
- source, reference and schema tampering on replay.

Only `bnlearn` and `pgmpy` framework labels are admitted. No native smoothing
adapter produces these results yet; the tests use the separate test oracle as
the reported runtime. `execution_source_verified` stays false and formal
evidence stays unavailable.

### Ontology bridge alignment diagnostics

`ontology/bridge.py` adds `suggest_declared_label_alignments()` and
`summarize_ontology_coverage()` (`gnn-ontology-alignment-diagnostics-v1`). For
each of the 49 unresolved declared labels in the pinned v5 bridge (64 declared,
15 resolved, 429 terms, ontology commit
`ad808b12d51ec990227d52c528f62db36dfd54bc`), they assign a deterministic semantic family and rank
candidate `term-*` anchors copied verbatim from the published release terms.
Every candidate is marked `candidate_only`.

These diagnostics do not change the bridge. `resolve_declared_label()` still
returns unresolved with no identifier. The pin and the bridge copy are
byte-unchanged, and no `obo:` or provisional identifier is emitted. Promoting a
candidate needs a reviewed release of the upstream ontology.

### E4-P1: authored planning admission

The shared `render/planning_admission.py` guard refuses unsupported source
horizons and policy identities on the six ordinary one-step planning adapters,
including direct and Step 11 entry points. It checks all supported parameter
containers before canonicalization can hide conflicts. This closes silent
one-step fallback for these explicit declarations only. The tracked five-step,
64-policy example retains its original source and remains unsupported for
planning until it supplies an executable policy-set/pruning contract. Broader
policy search and independent numerical qualification remain open.

### E4-BN1: finite bnlearn source filtering

The new `render/bnlearn/source_program.py` installs declared finite A/B/D CPTs
and performs native conditional filtering under exogenous transition actions.
Asymmetric native witnesses compare every posterior against independent forward
filtering. Source values, canonical axes, normalization provenance, timesteps,
seed and source/script identities remain observable. Metadata-only compatibility
calls retain a separately labelled synthetic smoke contract.

This closes the finite stationary filtering slice only. Source-specific causal
DAG discovery, independent/dependent component composition, C/E control semantics
and universal translation proofs remain open. Missing/invalid CPTs and unsupported
compositions refuse instead of reusing the synthetic demonstration. Extend the
witness contract before admitting those semantics.

## E1: coupled continuous agents

The existing independent-agent contract is not a coupling declaration. A first
coupled contract should define a joint linear-Gaussian system explicitly, with
ordered agent IDs and separate state, observation and control dimensions. Record
all block offsets and every cross-agent dynamics, observation and covariance
block; symmetry and positive definiteness apply to the assembled covariances.
Missing cross blocks must have a declared meaning rather than an inferred zero.

Start with centralized joint filtering and explicitly declared exogenous controls.
Do not call it decentralized inference or introduce policy optimization under the
same contract. Define observations and action timing before generator dispatch.
JAX and RxInfer must consume the same declared joint system, retain full posterior
cross covariances, and expose per-agent views by the recorded offsets.

Acceptance requires two agents with unequal dimensions, directional coupling,
nonzero cross covariance and independently specified observations. Compare against
E4's joint Gaussian reference. Setting coupling to zero must recover the existing
independent model. Permuting agent identities must permute all corresponding
blocks and results consistently. Missing controls, ambiguous joint priors and
unsupported factored/hybrid/nonstationary compositions must refuse explicitly.
Native RxInfer acceptance requires a provisioned Julia project environment.

## E2 and S3: THRML extensions

The maintained released contract is fixed-action categorical Gibbs smoothing with
strictly positive probabilities. The upstream
[padding issue](https://github.com/extropic-ai/thrml/issues/72) was closed on
2026-10-05, while the [PyPI metadata](https://pypi.org/pypi/thrml/json) inspected
on 2026-10-16 still identifies 0.1.4 as latest. Closure alone does not verify a
new released wheel. Preserve structural-zero refusal until S3's native witnesses
pass on both supported Python/JAX splits.

A future structural-zero witness must preserve active negative-infinite log
potentials while ensuring inactive padding cannot contaminate unrelated nodes.
Check conditional logits, independent-node empirical frequencies, source/seed/
sample identity, finite outputs and zero-frequency impossible states. Do not
substitute epsilon clipping or a local monkeypatch for released compatibility.

E2 admits a named [positive two-agent CPU contract](../gnn/implementations/thrml.md#source-declared-two-agent-coupled-smoothing)
for cross-agent previous-state dependence with conditionally independent next states,
fixed observations and fixed actions. Its native witness compares empirical THRML
marginals to an independent exact enumeration from the authored source tables.

Other coupled categorical factors, policy search, continuous/hybrid models and hardware
execution require different named contracts. For each, define factor semantics,
conditioning, normalization, resource bounds and independent witnesses before
adding capability registration. Hardware operation needs actual device receipts;
software sampling cannot establish hardware, convergence, energy or speed claims.

## E3: cpomdp action/search extension

The maintained compass-rose action set and exhaustive search remain unchanged.
Direct options now share typed public admission; invalid options refuse without
replacing a caller's existing artifact.

E3-FA1 now admits a source-declared finite action set with an explicit
version, ordered action IDs, units and a control matrix mapping each action into
state space. Validate finite, nonboolean components and exact asymmetric control
dimensions. Preserve the authored action order for ties and policy identity.
Do not silently deduplicate actions or reduce the horizon.

Compute policy count by bounded multiplication before enumeration. Bound total
policy-step evaluations and retained score/allocation estimates before building
arrays, and retain the process deadline/RSS supervision. Passive mode must not
allocate a planner. Receipts must record the complete declared action set,
horizon, source contract, every score and selected policy/action. All-invalid
scores fail before selection. Native acceptance must compare Kalman posteriors
and every EFE component against an independent reference, including tie and
invalid-score cases. Estimated allocation remains separate from measured RSS.

The [candidate sequence contract](../gnn/implementations/cpomdp.md#source-declared-candidate-sequences)
adds bounded evaluation of an ordered source-authored subset. It records each
candidate's native score and validity, and uses the source order for ties.
Candidate costs and receipts describe the supplied sequences. Continuous action
spaces and optimization beyond a finite set remain separate work.

## E5: exact-version archive

The [v4.1.0 version record](https://zenodo.org/records/23245091) resolves via
`10.5281/zenodo.23245091`, separately from concept DOI
`10.5281/zenodo.7803313`. A fresh 2026-10-16 download matched all 3,289 tagged
Git blobs and the release manuscript. Public wheel, sdist and manuscript bytes
matched GitHub digests and published SHA-256 sums. The automatic Zenodo record
contains the tagged source ZIP only; separate wheel, sdist and distribution
checksums are absent. E5 remains open for those archived assets and requires
archive-owner access.

Archive only the selected immutable release, with its source SHA, tag, wheel,
sdist, manuscript and checksums. The current working tree is a later review patch
and cannot substitute for v4.1.0 release bytes. Verify a version-specific archive
record independently of the concept DOI. Download every advertised artifact from
its public record and compare its digest with the selected release inventory;
check citation metadata and DOI resolution after publication.

A local package build or an unverified archive URL does not satisfy E5. Preserve
failed provider checks and leave the item open when public resolution or artifact
hash parity cannot be established. Archive publication and package-index delivery
remain distinct operations with their own destination and ownership contracts.

### E3-FA1 boundary

The source-declared finite-control implementation is specified in the
[cpomdp guide](../gnn/implementations/cpomdp.md#source-declared-finite-controls-e3-fa1).
It preserves authored action order, distinct IDs for equal vectors, units,
asymmetric control maps and complete per-policy EFE components. Admission also
bounds total evaluations across timesteps. This does not close E3's broader
control/search workstream, admit continuous action spaces, or establish E4's
independent source extraction or universal proof obligations.
