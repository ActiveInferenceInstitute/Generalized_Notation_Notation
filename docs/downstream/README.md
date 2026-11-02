# Downstream registry — repositories that apply GNN

Who already uses Generalized Notation Notation (GNN), what they model, which GNN
revision they pin, and where their conformance evidence lives. One row per
repository; each row links to a page with the full evidence.

At a glance: twenty-one repositories carry a committed GNN specification and
record an upstream revision (nineteen of them `v5.0.3` at commit `8f293d9b`;
GEO_INFER records the same commit without a tag; `fep_formal` pins by content
hash), and five more record the revision in their own contract rather than
in a pin file.

| Repository | What it models | GNN specification | Pinned GNN revision | Conformance / validation evidence |
| --- | --- | --- | --- | --- |
| [active_inference_language_model](active-inference-language-model.md) | Explicit discrete language stack in pymdp, letters through document position | `gnn/models/L1_letter.gnn.md` … `L6_narrative.gnn.md`; AILM-side contract `gnn/SPEC.md` | `v5.0.3` · `8f293d9b` in `gnn/upstream.json` | CI job `gnn-interop`; `scripts/verify_gnn_release.py` |
| [GEO_INFER](geo-infer.md) | Geospatial active inference monorepo (H3 topology, place and risk models) | Interchange contract `GEO-INFER-ACT/docs/gnn_interchange.md`; example `GEO-INFER-ACT/examples/cascadia/data/smith_river_homing.gnn.md` | `8f293d9b` in `.github/gnn-pair.json` (no tag field) | Workflow `gnn-interchange.yml` job "Paired GNN interchange"; `GEO-INFER-TEST/validate_gnn_interchange.py` |
| [COGANT](cogant.md) | Translates Python, JavaScript and TypeScript codebases into GNN state-space models; reverse synthesizer | Format docs `cogant/docs/theory/gnn_format.md`; emitted models `cogant/examples/zoo/*/model.gnn.md` | `v5.0.3` · `8f293d9b` in `cogant/docs/evaluation/gnn_v5_0_3_compatibility.json` | Manual CI job "Optional upstream GNN smoke"; `tests/unit/test_gnn_upstream_bridge.py` |
| [active_inference_battleship](active-inference-battleship.md) | GNN-based active inference for Battleship with a playable dashboard | `models/battleship.gnn.md`; contract table `docs/GNN_SPECIFICATION.md` | `v5.0.3` · `8f293d9b` in `models/gnn.pin.json` | `bun run model:check`; `scripts/contract.ts`, `scripts/parse_gnn.py` (no CI workflows) |
| [policy_entanglement](policy-entanglement.md) | Mean-field deformation framework for multi-stream policy ensembles | `gnn/bernoulli_toy.gnn.md`, `gnn/k_stream_ensemble.gnn.md`, `gnn/production_ensemble.gnn.md` | `v5.0.3` · `8f293d9b` in `gnn/upstream.json` | `tests/test_gnn_validation.py`, `tests/test_gnn_upstream_conformance.py` (no CI workflows) |
| [active_fractal_rabbit](active-fractal-rabbit.md) | Closed-loop active inference over synthetic waypoint data (claim boundary: synthetic-only) | `models/single_factor_reference.gnn.md`, `models/factorial_latent_mode.gnn.md` | `v5.0.3` · `8f293d9b` in `.github/gnn-pin.json` | CI step "Verify the recorded GNN pin against the fetched checkout"; `tests/test_gnn_spec.py` |
| [Active_Cardference](active-cardference.md) | Deck-generic card-game engine with GNN-specified active inference agents | `specs/kuhn_aif_p1.gnn.md` | `v5.0.3` · `8f293d9b` in `specs/gnn-pin.json` | CI step "Validate the GNN specification against the pinned parser"; `scripts/validate_gnn_spec.py` |
| [fep_formal](fep-formal.md) | Lean 4 catalog of FEP, active inference and Bayesian mechanics theorems | Bridge contract `docs/design/gnn-bridge/bridge-contract.md`; Lean `lean/FepSketches/gnn_document.lean` | Content hashes in `specs/gnn-bridge-w1-bridge-operations/syntax-pin.json` (no release tag) | Workflow `source-pair-qualification.yml` job `qualify-source-pair`; `docs/pin_audit.py` |
| [fundamentals](fundamentals.md) | Python companion to *Fundamentals of Active Inference* (Namjoshi, MIT Press 2026) | `specs/weather_pomdp.gnn.md` | `v5.0.3` · `8f293d9b` in `specs/gnn-pin.json` | CI job `gnn-conformance`; `tests/test_gnn_spec.py` |
| [cognitive](cognitive.md) | Discrete and continuous active inference package plus a knowledge vault | `specs/discrete_variational.gnn.md` | `v5.0.3` · `8f293d9b` in `specs/gnn-pin.json` | CI job `gnn-conformance`; `code/tests/test_gnn_conformance.py` |
| [ActiveBlockference](active-blockference.md) | CPU-oriented active inference for radCAD/cadCAD grid simulations | `specs/gridworld_3x3.gnn.md` | `v5.0.3` · `8f293d9b` in `specs/gnn-pin.json` | CI job `gnn-conformance`; `tests/test_gnn_spec.py` |
| [GEN24](gen24.md) | Synthetic domain material generation; GNN enters through the NHIM module | `NHIM/nhim_inquiry_agent.gnn.md` | `v5.0.3` · `8f293d9b` in `NHIM/gnn-pin.json` | CI step "Validate the GNN specification against the pinned parser"; `scripts/validate_gnn_spec.py` |
| [CEREBRUM](cerebrum.md) | Case-enabled reasoning engine with Bayesian representations | `models/case_conditioned_active_inference.gnn.md` | `v5.0.3` · `8f293d9b` in `.github/gnn-pin.json` | `tests/core/test_gnn_spec.py` (no GNN-specific CI job) |
| [Biofirm](biofirm.md) | Ecological control and bioregion decision analysis with a pymdp controller | `models/biofirm_ecosystem_variable.gnn.md` | `v5.0.3` · `8f293d9b` in `.github/gnn-pin.json` | `tests/test_gnn.py` (no GNN-specific CI job) |
| [Active_InferAnts](active-inferants.md) | Multi-language active inference framework (ant-colony multi-agent simulation) | `gnn/gaussian_agent.gnn.md` | `v5.0.3` · `8f293d9b` in `gnn/upstream.json` | `tests/test_gnn_conformance.py` (no GNN-specific CI job) |
| [ActInf_RxInfer](actinf-rxinfer.md) | RxInfer.jl working example: Beta-Bernoulli coin-bias inference | `specs/coin_flip.gnn.md` | `v5.0.3` · `8f293d9b` in `specs/gnn-pin.json` | CI job `gnn-conformance`; `scripts/validate_gnn_spec.py` |
| [Active_Skillference](active-skillference.md) | Micro-learning platform whose prerequisite graph rests on a discrete agent | `gnn/discrete_active_inference_agent.gnn.md` | `v5.0.3` · `8f293d9b` in `gnn/upstream.json` | `tests/test_gnn_conformance.py` (no GNN-specific CI job) |
| [ActiveInferenceImplementations](active-inference-implementations.md) | Educational RxInfer.jl notebooks: mean-field variational message passing | `specs/normal_vmp.gnn.md` | `v5.0.3` · `8f293d9b` in `specs/gnn-pin.json` | CI job `gnn-conformance`; `scripts/validate_gnn_spec.py` |
| [active_inference_power](active-inference-power.md) | Statistical-power simulations for adaptive studies with an embedded agent | `models/reference_action_loop.gnn.md`, `models/sequential_evidence_agent.gnn.md` | `v5.0.3` · `8f293d9b` in `models/gnn-pin.json` | CI step `scripts/validate_gnn_specs.py`; `tests/test_gnn_specs.py` |
| [ants](ants.md) | Ant-colony foraging in a T-maze (code for *Active Inferants*, 2021) | `gnn/ant_colony_foraging.gnn.md` | `v5.0.3` · `8f293d9b` in `gnn/upstream.json` | `tests/test_gnn_conformance.py` (no GNN-specific CI job) |
| [on_policy_distillation](on-policy-distillation.md) | On-policy distillation formalized as active inference: three small models | `gnn/bernoulli_toy.gnn.md`, `gnn/graph_world.gnn.md`, `gnn/si_tmaze.gnn.md` | `v5.0.3` · `8f293d9b` in `gnn/upstream.json` | `tests/test_gnn_upstream_conformance.py`, `tests/test_gnn.py` (no GNN-specific CI job) |

## How to read this registry

- **Pinned GNN revision** names the GNN release tag and commit recorded in the
  repository's own pin file, or in the contract named in the row. Nineteen of
  the twenty-one repositories pin GNN
  `v5.0.3` at commit `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`; `GEO_INFER`
  records the same commit without a tag field, and `fep_formal` pins by content
  hash instead of a release tag. A pin records what a repository declares, not
  what it currently passes.
- **Conformance / validation evidence** names the workflow job, test file or
  documented command that a repository uses to check its GNN surface. Where no
  GNN-specific CI job exists, the row says so and cites the test file instead.
- **What it models** is taken from the repository's own description or README,
  not from a third party.

## Method and scope

Every row was verified on 2026-11-02 from the repository's origin default branch
with read-only GitHub API calls: `git/trees/<branch>?recursive=1` to locate
files and `contents/<path>?ref=<branch>` to read pin files, specifications and
workflow definitions. File presence, pin contents and workflow job names are
verified; current CI status and local reproducibility are not. A row that could
not be verified says "not verified" rather than repeating an assumption.

Candidates came from scanning the organization for the markers of GNN use
(`.gnn.md` specification files and GNN pin files), and every repository below
was then read from origin directly. The twenty-one repositories in the table are
the ones with a committed GNN specification or pin on the default branch.
Repositories that only mention GNN in passing are not listed; repositories that
record the revision outside a pin file are in the section below.

## Revisions recorded in the repository's own contract

These repositories carry GNN files or GNN bridge code on the default branch and
record the upstream revision in their own contract rather than in a pin file.
Each row names the contract; every path and revision below was verified from
origin on 2026-11-02.

| Repository | GNN content | Where the revision is recorded |
| --- | --- | --- |
| `ActiveInferenceCategoryTheory` | `gnn/learner_perception_model.md`, `gnn/map_the_model.template.md` | Verification run in `gnn/README.md` at commit `8f293d9b` (v5.0.3) |
| `act_inf_metaanalysis` | `gnn/h1_fep_universality.md`, `gnn/h4_predictive_coding.md`, `gnn/h8_language_aif.md` | `gnn/README.md`: revision `v5.0.3`/`8f293d9b`, an archive extract recipe, and recorded model SHA-256 digests |
| `courses` | `course_development/gnn/01_gnn_syntax/` (syntax module, lab, model exemplars) | `course_development/gnn/01_gnn_syntax/models/README.md` provenance table (release `v5.0.3`, commit `8f293d9b`) with a `git show` and `diff` recipe against the pinned corpus |
| `pymdp` | Bridge code `pymdp/gnn.py`, `examples/gnn/publish_gnn_spec.py`, `test/test_gnn_bridge.py`, `docs-mkdocs/guides/gnn-interoperability.md` | `GNN_PINNED_RELEASE` and `GNN_PINNED_REVISION` constants in `pymdp/gnn.py` |
| `Active_Inference_Ontology` | Term export `GNN/act_inf_ontology_terms.json`, `GNN/README.md` | `GNN/declared-labels.json`: release, commit and SHA-256 of the audited file; this repository pins *it* at `4450e8dd` in `src/gnn/ontology/ontology-pin.json` |

## Exclusions

- `Active_Fedference` — listed in earlier internal notes as a GNN consumer, but
  no GNN content exists on its `main` branch as of 2026-11-02: a code search for
  `GNN` and for `Generalized Notation` returns zero results, and the tree
  contains no `.gnn.md` file. A GNN layer does exist on the open, unmerged pull
  request #8 (`src/fedference/gnn/_documents.py`, head `517012c428e4`), which
  declares upstream revision `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`; it is
  not on the default branch, so it is not counted here.

## Related

- [GNN documentation hub](../README.md)
- [Flat documentation index](../INDEX.md)
- [GEO_INFER interchange notes](../other/geo_infer/README.md)
- [FEP_Formal bridge notes](../other/fep_lean/README.md)
