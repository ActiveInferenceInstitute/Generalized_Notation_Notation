# GNN and fep_lean: collaboration program (GNN side)

> **Document Metadata**
> **Type**: Research program | **Audience**: Researchers, Developers, Agents | **Complexity**: Advanced
> **Cross-References**: [README.md](README.md) | [fep_lean overview](fep_lean.md) | [Bridge contract mirror](bridge-contract.md)
> **Last Updated**: 2026-09-06

## Overview

The sibling repository `fep_lean` proves invariants about Active Inference
generative models in Lean 4; this repository specifies, renders, and executes
those models. The collaboration program connects the two layers in two
directions:

- **Direction 1 (Lean to GNN).** Generative models that have Lean expressions
  in `fep_lean` are projected into GNN documents, which this pipeline then
  validates, renders, and executes. Execution-derived quantities travel back
  as certificates checked against Lean-witnessed properties.
- **Direction 2 (GNN to Lean).** GNN's steps and methods — the sectioned
  document grammar, connection grammar, state-space typing, ontology binding
  checks, and the dynamic meaning of its two model families — are formalized
  in Lean, reusing `fep_lean`'s carriers.

The shared rules live in the bridge contract: canonical copy at
`../fep_lean/docs/design/gnn-bridge/bridge-contract.md`, mirror at
[bridge-contract.md](bridge-contract.md). Cross-repo references are inline
code paths, never markdown links.

## What fep_lean provides

A catalogue of 155 FEP / Active Inference / Bayesian Mechanics / Information
Geometry / Thermodynamics topics with Lean 4 theorem bodies, distinct evidence
planes, and a pinned workspace. The bridge-relevant core:
`FiniteLaw`/`FiniteKernel` (finite probability), the `GenerativeModel` POMDP
structure with both expected-free-energy decompositions, `FiniteHMM`
filtering, blanket factorization with native `CondIndepFun`, policy trees,
and the linear-Gaussian/OU semigroup family. Full inventory:
[fep_lean.md](fep_lean.md).

## GNN side surface (v0.5)

Since contract v0.5 the GNN-side Python package is canonically `gnn`
(src-layout `src/gnn/`); the root package is the one aggregate facade
(`parse_gnn_file`, `run_pipeline`, `render_gnn_spec`, `GNNExecutor`,
`generate_exports`, `ModelRegistry`, `load_config`, …). The executor
registers `lean` as an execution family: `GNNExecutor.execute_gnn_model`
with `execution_type="lean"`, or the Step 12 runner, drives the fep_lean
bridge `verify-document` operation over rendered `*.lean` / emitted GNN
`*.md` documents and records per-document receipts. The serializer behind
the Lean surface (`gnn.parsers.lean_serializer`) emits the frozen
13-section `FEP.GnnDocument` inventory with a canonical `-- MODEL_DATA:`
typed payload.

## Lean-derived models in the pipeline

A fep_lean-emitted document is ordinary GNN syntax and flows through the
standard pipeline. Conventions fixed by the bridge contract:

- **Identifier.** The `GNNSection` value is prefixed `FepLean` (e.g.
  `FepLeanPOMDPAgent`, `FepLeanContinuousOU`), keeping identifiers
  space-free per [normative syntax](../../gnn/gnn_syntax.md).
- **Kind detection.** Continuous-family documents include the `continuous`
  keyword (or declare `F/H/Q/R`) so model-kind detection stays mechanical;
  discrete-family documents use the categorical declarations.
- **Provenance.** The provenance section carries source repository, commit
  digest, the projected Lean module and definition, and the projection tool
  identity.
- **Required sections.** `GNNSection`, `GNNVersionAndFlags`, `ModelName`,
  `StateSpaceBlock`, `Connections`; `InitialParameterization`, `Equations`,
  `Time`, `ActInfOntologyAnnotation`, and `ModelParameters` as applicable.

Step touchpoints:

| Step | Module | Role for bridge documents |
| --- | --- | --- |
| 3 parse | `src/gnn/` | parse the emitted document; first acceptance gate |
| 5 type check | `src/gnn/type_checker/` | state-space typing and dimension consistency |
| 10 ontology | `src/gnn/ontology/` | validate bindings against `src/gnn/ontology/act_inf_ontology_terms.json`; unknown bindings need an explicit vocabulary decision, not silent misspellings |
| 11 render | `src/gnn/render/` | ten render targets; continuous models on categorical-only backends report `unsupported`, never execute |
| 12 execute | `src/gnn/execute/` | eight execute targets; outputs land under `output/12_execute_output/summaries/execution_summary.json` |

Commands of record (from the repository root):

```bash
uv run gnn validate input/gnn_files/discrete/actinf_pomdp_agent.md --strict
uv run python src/gnn/main.py --target-dir input/gnn_files --output-dir output \
  --only-steps "3,5,10,11,12" --verbose
uv run python src/gnn/11_render.py --target-dir input/gnn_files --output-dir output \
  --frameworks "pymdp,jax" --strict-framework-success
uv run python src/gnn/12_execute.py --target-dir input/gnn_files --output-dir output \
  --render-output-dir output/11_render_output --frameworks "pymdp,jax" --timeout 600
```

## Formalizing GNN steps and methods in Lean

Direction 2 is owned by the fep_lean side and specified in
`../fep_lean/docs/design/gnn-bridge/direction-2-gnn-to-lean.md`. In brief,
the Lean targets are:

| GNN method | Lean target (prospective) |
| --- | --- |
| Sectioned document grammar (step 3) | document AST as inductive types; connection grammar (`A>B`, `A-B`, `:label`) as edge predicates |
| State-space typing (step 5) | typed block formation; dimension consistency with `ModelParameters` |
| Ontology binding validity (step 10) | binding predicate over the frozen canonical vocabulary |
| Validation rules (step 6) | propositions mirroring each maintained rule |
| Dynamic semantics of the two model families | discrete family denoted over `FiniteLaw`/`FiniteKernel`/`FiniteHMM`; continuous family over `LinearGaussianParameters` |
| Renderer and executor behavior (steps 11–12) | semantics-preservation statements per target (statements first, proofs later) |

GNN-side responsibilities in each slice: pin the syntax version being
formalized, curate the exemplars the formalization must decide correctly,
and adjudicate vocabulary questions (step 10) raised by the formalization.

## Exemplar correspondence

| GNN exemplar | fep_lean counterpart |
| --- | --- |
| `input/gnn_files/discrete/actinf_pomdp_agent.md` | `active_inference.lean` `GenerativeModel`; the exemplar's ontology bindings (`A=LikelihoodMatrix`, `B=TransitionMatrix`, `C=LogPreferenceVector`, `D=PriorOverHiddenStates`, `E=Habit`, `F=VariationalFreeEnergy`, `G=ExpectedFreeEnergy`, `s=HiddenState`, `o=Observation`, `π=PolicyVector`, `u=Action`) name the same quantities the Lean body defines |
| `input/gnn_files/discrete/simple_mdp.md` | finite Markov kernels (`finite_markov_dynamics.lean`) |
| `input/gnn_files/continuous/continuous_navigation.md` | `linear_gaussian_semigroup.lean` (linear-Gaussian transition/readout with noise) |
| `input/gnn_files/continuous/predictive_coding_agent.md` | predictive-coding and Gaussian-filtering strands |

## Phases

| Phase | Side | Outcome |
| --- | --- | --- |
| P1 spike | both | one finite model projected from Lean, validated, rendered, executed |
| P2 emitter | fep_lean | deterministic projection module; regenerable documents with provenance |
| P3 certificate | both | execution summaries checked against Lean-witnessed properties, evidence-plane labeled |
| P4 continuous spike | both | linear-Gaussian/OU family onto the GNN continuous family |
| Q1–Q4 formalization | fep_lean | document AST, well-formedness, denotations for both families, renderer statements |

Acceptance criteria and no-go actions for every phase live in the fep_lean
direction documents (`../fep_lean/docs/design/gnn-bridge/direction-1-lean-to-gnn.md`
and `.../direction-2-gnn-to-lean.md`).

## Open problems (GNN-side responsibilities)

- **Vocabulary extensions.** Lean-specific ontology bindings beyond the
  exemplar terms need a decision process against
  `src/gnn/ontology/act_inf_ontology_terms.json` before emission.
- **Syntax version pinning.** Each formalization slice pins one
  `GNNVersionAndFlags` surface; drift requires an explicit re-freeze.
- **Float boundary.** `type=float` state-space values are approximate
  numerics; the formal object will carry the rounding boundary explicitly.
- **Artifact custody.** Render and execution summaries for bridge documents
  must remain bound to the provenance digest of the Lean source that
  generated them.

## Findings filed by fep_lean Direction 2 (Q2–Q4, 2026-09-04)

Conventions the Lean formalization needs frozen on this side before
corresponding fragments can be stated or parsed. Inline code paths only
per the bridge contract's cross-reference rule.

1. **Brace-block payload shape (Q2 + Q3).** The matrix payload shape
   `((…), (…))` (one parenthesized row per matrix row, comma-separated)
   and vector shape `(…)` are the frozen transcription target the bridge
   relies on when reading `InitialParameterization` entries — see the
   exemplar transcriptions in `src/fep_lean/formal/gnn_denotation.lean`
   and `src/fep_lean/formal/gnn_denotation_continuous.lean`
   (fep_lean). A future payload-string parser on the GNN side must
   accept this shape for both families before transcription can be
   mechanized.
2. **Exemplar `F` bakes in Euler discretization (Q3).** In
   `input/gnn_files/continuous/stochastic_dynamics.md` the
   `ModelParameters` record `dt = 0.1` and the declared `F` is the
   Euler-discretized one-step map, not the drift matrix. Identifying
   one-step `F` with a continuous-time law (`exp(-t·precision)`) is a
   recorded no-go (fep_lean `specs/gnn-bridge-p4-continuous-spike/README.md`,
   family-wide argument). A future dynamics-gauge slice needs a
   declared `StepTime`-style semantic field or a drift-matrix section
   on the GNN side before any such identification can be stated.
3. **pgmpy `TabularCPD` parent-instantiation enumeration order (Q4).**
   The bnlearn render target emits `pgmpy`/`bnlearn` network scripts
   whose CPT tables depend on pgmpy's parent-instantiation (column)
   enumeration order. That order is a GNN-side convention that has not
   been frozen; until it is, the bnlearn CPT-layout fragment keeps its
   no-go row in fep_lean's Q4 statement inventory
   (`src/fep_lean/formal/gnn_render_statements.lean` proof-schedule
   table) and no preservation statement is minted for it. Freezing the
   enumeration convention here reopens the fragment.
