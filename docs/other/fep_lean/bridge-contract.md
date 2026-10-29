# Bridge contract: FEP_Formal and GNN

| Field | Value |
| --- | --- |
| Version | 0.6 |
| Date | 2026-09-06 |
| Interface clarification | 2026-10-28; canonical names, unchanged model semantics |
| Canonical copy | `../fep_formal/docs/design/gnn-bridge/bridge-contract.md` — edit there first |
| Mirror copy | this file, `docs/other/fep_lean/bridge-contract.md` |
| Change rule | substance changes bump the version and land in both checkouts in the same working session |

## 1. Purpose and scope

This contract is the formal articulation between two sibling repositories:

- FEP_Formal (`fep_formal`) — a standalone catalog of FEP / Active Inference /
  Bayesian Mechanics / Information Geometry / Thermodynamics topics with
  Lean 4 theorem bodies, a pinned workspace, and distinct evidence planes.
- `GeneralizedNotationNotation` (GNN) — a text-based notation for Active
  Inference generative models with a 25-step pipeline that parses, type
  checks, renders, and executes model documents.

Public Python imports use `fep_formal`, commands use `fep-formal`, and public
settings use `FEP_FORMAL_`. Legacy names remain deprecated aliases through 2.x.
Lean theorem namespaces and historical receipt keys retain their identities.

The contract scopes two directions of collaboration and nothing else. It
creates no capabilities, no catalogue rows, and no evidence claims. All
cross-references between the repositories are inline code paths, never
markdown links.

## 2. Parties and authority

| Concern | Authority |
| --- | --- |
| Semantics of FEP objects (laws, kernels, blankets, free energy, policies) | fep_formal |
| Syntax, parsing, rendering, and execution of model documents | GNN |
| The mapping between them | this contract, edited in both repositories |
| Evidence classification (what counts as what) | each repository's own contract; see section 7 |

## 3. Shared object and model kinds

The shared object is a **generative model instance** expressed on both
sides. The bridge recognizes exactly two model kinds, mirroring the two
kinds the GNN pipeline distinguishes:

1. **Discrete POMDP family** — categorical `A` (likelihood), `B`
   (transition, ordered `(next_state, previous_state, action)`), `C`
   (preferences), `D` (initial prior), optional `E` (habit), and an `F[1]`
   variational-free-energy readout. Counterpart carriers: `FiniteLaw`,
   `FiniteKernel`, the `active_inference.lean` `GenerativeModel`, and the
   `FiniteHMM` filtering stack.
2. **Continuous linear-Gaussian family** — `F/H/Q/R` with
   `prior_mean`/`prior_cov`, dynamics `x_t = F x_{t-1} + u_{t-1} + N(0,Q)`
   and observation `y_t = H x_t + N(0,R)`, optional closed loop via
   `goal_mean`/`control_gain`. Counterpart carriers: `LinearGaussianParameters`
   and the scalar/multivariate OU semigroup family.

Model-kind detection stays with GNN's own rules (the `GNNSection` value
drives detection, e.g. a `continuous` keyword or the presence of `F/H/Q/R`).
Emitted documents must be detectable under those rules without special
casing on the GNN side.

## 4. The interchange artifact

The only interchange artifact is a **GNN document** (`.md` in GNN syntax,
version `GNN v1.x`), whose section inventory is normatively defined by
`GeneralizedNotationNotation/docs/gnn/gnn_syntax.md`:

| Section | Status | Bridge use |
| --- | --- | --- |
| `GNNSection` | required | short identifier without spaces; bridge convention prefixes `FepLean` (e.g. `FepLeanPOMDPAgent`, `FepLeanContinuousOU`) and includes the `continuous` keyword for the continuous kind so kind detection stays mechanical |
| `GNNVersionAndFlags` | required | pins the syntax version the projection targets |
| `ModelName` | required | human-readable title |
| `StateSpaceBlock` | required | variable and matrix declarations derived from Lean index types |
| `Connections` | required | dependency edges derived from the Lean carrier structure |
| `InitialParameterization` | optional | concrete values; see the rounding policy (section 9) |
| `Equations` | optional | LaTeX restatement of the Lean-defined dynamics |
| `Time` | optional | static/dynamic and time-horizon facts from the Lean definition |
| `ActInfOntologyAnnotation` | optional | variable bindings validated by GNN step 10 against its canonical vocabulary |
| `ModelParameters` | optional | numeric parameters (`num_hidden_states`, `num_obs`, `num_actions`, `num_timesteps`, ...) |
| `Footer` | optional | closing marker |
| `Signature` | optional | **bridge convention: mandatory provenance** |

**Provenance rule.** Every bridge-emitted document carries, in its
provenance section: source repository and commit digest, the Lean module and
definition projected, the projection tool identity, and the syntax version
targeted. A document without provenance is not a bridge artifact.

## 5. Direction 1: Lean to GNN (render and execute)

| Stage | Owner | Input to output | Acceptance |
| --- | --- | --- | --- |
| S1 Selection | fep_formal | a named Lean definition family (e.g. the finite posterior–decision–action certificate carrier or an `active_inference.lean` `GenerativeModel` instance) | the named definition compiles warning-free under `fep-formal verify` before projection |
| S2 Projection | fep_formal (prospective module beside `src/fep_formal/formal/`) | Lean definition → typed projection: index types, law/kernel components, dependencies, parameters, timescale | deterministic and regenerable; no judgment calls |
| S3 Emission | bridge tooling | projection → GNN document with provenance | passes `uv run gnn validate <file> --strict` in the GNN checkout |
| S4 Toolchain | GNN | steps 3 (parse), 5 (type check), 10 (ontology), 11 (render), 12 (execute) | render and execution summaries produced; statuses reported honestly (`unsupported` is distinct from `failed`) |
| S5 Certificate | joint | execution-derived quantities vs Lean-witnessed properties | every compared quantity names its evidence plane; disagreement is a finding, never silently averaged away |

## 6. Direction 2: GNN to Lean (formalize steps and methods)

| Stage | Owner | Target | Acceptance |
| --- | --- | --- | --- |
| S1 Syntax surface freeze | GNN | the normative section inventory in `docs/gnn/gnn_syntax.md` | a pinned version reference both sides cite |
| S2 Document AST | fep_formal (prospective) | GNN sections, state-space blocks, and connections as Lean inductive types | compiles in the pinned workspace |
| S3 Well-formedness | fep_formal | decidable predicates mirroring parse, connection grammar, state-space typing, and ontology binding checks | every `input/gnn_files` exemplar decides correctly |
| S4 Dynamic semantics | fep_formal | discrete-family denotation over `FiniteLaw`/`FiniteKernel`/`FiniteHMM`; continuous-family denotation over `LinearGaussianParameters` | theorem statements for both families accepted in a slice |
| S5 FEP instantiation | both | GNN conventions ↔ fep_formal definitions (`A/B/C/D` ↔ `GenerativeModel`; `F[1]` ↔ `variationalFreeEnergy`; the `G=ExpectedFreeEnergy` ontology binding ↔ `expectedFreeEnergy_eq_risk_add_ambiguity`) | alignment statements named as prospective until proven |
| S6 Renderer statements | fep_formal | semantics-preservation statements for render targets | statements first; proofs are later slices |

Direction 2 detail lives in
`fep_formal/docs/design/gnn-bridge/direction-2-gnn-to-lean.md` (cross-repository reference).

## 7. Evidence firewall

Each claim names exactly one evidence plane, and planes never substitute for
each other:

| Plane | Producer | Establishes | Never establishes |
| --- | --- | --- | --- |
| Native Lean compilation | `fep-formal verify` | the named body compiles without warnings or `sorry` | semantic reach, executed behavior |
| Semantic review | `config/theorem_maturity.yaml` | how far a theorem proxy reaches toward its topic label | compilation sufficiency |
| Numerical witness | `fep-formal dashboard` | typed, explanatory finite witnesses | proof or empirical validation |
| GNN pipeline run | steps 3–12 | the document parses, renders, and executes | mathematical correctness of the model |

A bridge certificate may only state agreement between quantities that each
carry their own plane; it may not reclassify a simulation statistic as a
proved property.

## 8. Synchronization and ownership

- Canonical contract: `fep_formal/docs/design/gnn-bridge/bridge-contract.md`.
  Mirror: `GeneralizedNotationNotation/docs/other/fep_lean/bridge-contract.md`.
- Contract edits land in both checkouts in the same working session; the
  mirror stays identical to the canonical body except for (a) the
  canonical-pointer header rows and (b) cross-repository markdown links,
  which the mirror renders as inline code paths.
- `fep_formal/docs/design/gnn-bridge/` is maintained by fep_formal-side agents;
  `docs/other/fep_lean/` in the GNN checkout by GNN-side agents.
- Version bumps follow the change rule in the header table.

## 9. Policies and no-go registry

| Trigger | Action |
| --- | --- |
| A field of the GNN document cannot be derived deterministically from the named Lean definition | stop; narrow the extraction contract; never hand-fit the document |
| Syntax surface drift in `docs/gnn/gnn_syntax.md` | freeze the pinned version per slice; re-freeze explicitly before extending |
| Numeric rounding of exact Lean values — terminating decimals emit exactly; non-terminating exact Lean reals emit as float64 (shortest round-trip repr), with the exact formula recorded verbatim in provenance, and consumers treat the float as an approximation, never as the Lean value | rounding policy fixed once in this contract's first slice and extended by v0.2; the digest of the exact source is recorded in provenance |
| A projected model exceeds a backend (e.g. continuous model on a categorical-only renderer) | report `unsupported`; never distort the model |
| A desired ontology binding is absent from the GNN canonical vocabulary | either use existing terms or open an explicit vocabulary-extension request on the GNN side; never emit bindings that fail step 10 |
| Contract edit without the mirror edit | revert until both land |
| Pressure to claim execution "verifies" a theorem or a theorem "validates" execution | refuse; section 7 is non-negotiable |

## 10. Source custody and evidence operations (v0.3)

The W2 source pin records explicit commit references and relevant owner-content
hashes for both repositories. Hashes describe the actual working tree; a commit
reference does not claim that dirty owner bytes are committed. Owner additions,
deletions, or changes invalidate the pin. Unrelated HEAD movement does not.
The pin and emitted artifacts are excluded from their own owner digests.

Pinning and emission are explicit operations. Default status, freshness checks,
and numerical comparison do not write reports or documents. A digest refresh
may change only designated fields in the unique Signature section; reordered,
deleted, duplicated, or otherwise changed model content requires deliberate
re-emission. Finite and continuous emission follow the same custody rules.

Numerical receipts bind the comparison policy, actual result artifact, document,
and owner snapshot. A result from an older run may be compared, but is not thereby
execution evidence for the new sources. Such receipts explicitly keep
`execution_source_verified: false` and `native_claim_ready: false`. A concrete
renderer-payload proof has a separate artifact/native verification receipt and
does not establish runtime behavior or general renderer correctness.

The historical P3 comparison default is read-only; report writing requires
`--output`. Supported source-bound operations use `fep-formal bridge` with an
explicit `--gnn-root`.

v0.6 extraction-package migration: the pinned render route is
`gnn.extract.pomdp_extractor` (the extraction concern is packaged at
`src/gnn/extract/`; `python -m gnn.extract` is preserved via the package
`__main__`). v0.5 rename migration: the canonical GNN Python package is `gnn`
(src-layout `src/gnn/`); the GNN owner roster is re-pinned to the single glob
`src/gnn/**/*.py` plus fixed owners `pyproject.toml`, `uv.lock`,
`src/gnn/main.py`, the mirror, and the pinned syntax files
(`docs/gnn/gnn_syntax.md`, `src/gnn/pipeline/step_registry.py`).

## 11. Concrete artifact proofs and shared receipts (v0.4)

The artifact slices have separate mathematical contracts. Acceptance requires a
current native receipt for the exact slice; a generated probe or passing Python
test alone does not establish native acceptance.

| Slice | Checked artifact and mathematical scope | Excluded claims |
| --- | --- | --- |
| Q5 | A canonical PyMDP render and a separately authored asymmetric control; restricted literal tables instantiate fixed finite payload statements. | Python execution, general renderer correctness, or arbitrary model equivalence. |
| Q6 | Two canonical Boolean Julia runners; their embedded JSON tables agree with independent symmetric and asymmetric Lean payloads. | Consumed runtime preferences, EFE equivalence, or ActiveInference.jl behavior: the runner transforms C and does not consume E in its action selector. |
| Q7 | A canonical scalar-OU JAX render; decoded binary64 coefficients have bounded error against the selected exact-real OU formulas and one-step prediction. | Floating-point execution, accumulated trajectory error, a general SDE solution, or physical applicability. |

The source locations are `fep_formal/specs/gnn-bridge-q5-artifact-proof/`,
`fep_formal/specs/gnn-bridge-q6-activeinference-artifact/`, and
`fep_formal/specs/gnn-bridge-q7-continuous-ou-proof/`. Each slice declares an
immutable contract consumed by the shared
`fep_formal/src/fep_formal/verification/gnn_artifact_receipt.py` engine.

Schema-2 native receipts bind source snapshots before and after compilation,
the full slice contract, exact generator/extractor/probe bytes, canonical
render provenance, resolved toolchain binaries, recursive formal dependencies,
compiled imports, native commands, and complete theorem axiom reports. Checks
reject stale bytes, changed commands, warning diagnostics, and unapproved
axioms. Default and `--check` operation are read-only; `--compile` explicitly
creates fresh evidence. Schema-1 historical receipts require regeneration.

The sealing order is final source pin, explicit finite/continuous emission,
canonical render refresh, deterministic probe regeneration, native compilation,
and read-only receipt validation. Renderer-input custody and generated-proof
custody remain distinct. Checked local Python and toolchain binaries are
trusted inputs; these checks provide content custody, not a Python sandbox or
independent authentication of the host compiler.

Runtime evidence remains separate. In particular, the retained one-timestep
Q7 fixture observes its prior without applying a transition. A nonstationary
multi-step JAX regression exercises transition behavior but does not become a
native proof or establish a whole-trajectory error bound.

## 12. Horizon acceptance boundary (v0.4)

Horizon 2 terminal acceptance combines the exact mandatory native-test roster,
unchanged native-source snapshots, the explicitly enabled Fin4 supplement,
independent typed scalar/Fin4 numerical witnesses, retained predecessor
contracts, and three distinct source-bound Lean, domain, and skeptical reviews.
The validator under `fep_formal/specs/done/horizon-2-smooth-stochastic/readiness/`
rejects absent, stale, malformed, or incomplete evidence.

An accepted terminal record opens only read-only H3.G0 eligibility review.
`fep_formal/specs/h3-reference-study/` validates the accepted carrier and required
prospective study declarations. It does not select a dataset, invent license
or sampling metadata, or authorize H3.0--H3.7 empirical execution. Numerical
witnesses and bridge proofs do not replace these study requirements.

## 13. GNN package layout and verify-document (v0.5)

The canonical GNN Python package is `gnn` (src-layout `src/gnn/`). Every
former top-level pipeline module (including the numbered step drivers and
`main.py`) lives under `src/gnn/`; tests live outside the wheel under
`tests/`. The root package is the one aggregate facade.

The `verify-document` operation (Direction 2 S7) checks one emitted GNN
document: `fep-formal bridge verify-document --gnn-root PATH --document PATH`
`[--model finite|continuous] [--receipt PATH] [--fail-on-warnings]`. It
extracts the typed payload through the pinned render route
(`gnn.extract.pomdp_extractor.extract_pomdp_from_file`, strict validation; full
StateSpaceBlock inventory via `gnn.schema.parse_state_space`), constructs a
`FEP.GnnDocument.GnnDocument` value in a compile-once Lean probe, and decides
`documentWellFormed doc = true` in the committed `lean/FepSketches/`
workspace. The receipt schema is `{"schema_version": 2, "document" (sha256),
"model_family", "extracted_family", "toolchain" (lean version), "status"
("ok" | "failed" | "skipped"), "warnings"}`. Missing Lean toolchain yields
`status: "skipped"` and a non-zero exit — fail-closed, never a silent pass.

The canonical `MODEL_DATA` payload schema embedded by the GNN Lean surface is
`{"schema_version": 1, "model_family": "finite" | "continuous",
"state_spaces": [{"decl", "dims", "value_type"}], "parameterizations":
[{"var_name", "payload"}], "ontology_bindings": [{"var_name", "term"}]}`
plus the required keys the shared strict reconstruction path consumes
(`model_name`, `variables`, `connections`, `parameters`, `equations`,
`time_specification`, `ontology_mappings`).

No-go: `verify-document` does not auto-prove `DiscreteConforms` or
`ContinuousConforms` for arbitrary documents. That is research-grade and
stays out of scope; conformance proofs remain explicit, document-specific
Lean artifacts (§7 evidence firewall).
