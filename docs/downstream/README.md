# Downstream registry — repositories that apply GNN

Who already uses Generalized Notation Notation (GNN), what they model, which GNN
revision they pin, and where their conformance evidence lives. One row per
repository; each row links to a page with the full evidence, except the row
marked †, whose page is deferred to the same scheduled count-changing change as
the contract rows below (its link goes to the repository's own evidence).

At a glance: twenty-two repositories carry a committed GNN specification or pin
and record an upstream revision (twenty of them `v5.0.3` at commit `8f293d9b`;
`Active_Fedference` records the same commit in its backend lock without a tag
field; `fep_formal` pins by content hash), and five more record the revision in
their own contract rather than in a pin file.

| Repository | What it models | GNN specification | Pinned GNN revision | Conformance / validation evidence |
| --- | --- | --- | --- | --- |
| [active_inference_language_model](active-inference-language-model.md) | Explicit discrete language stack in pymdp, letters through document position | `gnn/models/L1_letter.gnn.md` … `L6_narrative.gnn.md`; AILM-side contract `gnn/SPEC.md` | `v5.0.3` · `8f293d9b` in `gnn/upstream.json` | CI job key `gnn-interop`, display `gnn conformance (pinned)`; `scripts/verify_gnn_release.py` |
| [GEO_INFER](geo-infer.md) | Geospatial active inference monorepo (H3 topology, place and risk models) | Interchange contract `GEO-INFER-ACT/docs/gnn_interchange.md`; example `GEO-INFER-ACT/examples/cascadia/data/smith_river_homing.gnn.md` | `v5.0.3` · `8f293d9b` in `.github/gnn-pair.json` | Workflow `Paired GNN interchange` job key `interchange`: `gnn conformance (pinned)` on Python 3.12 and `gnn interchange (3.11)` on Python 3.11; `GEO-INFER-TEST/validate_gnn_interchange.py` |
| [COGANT](cogant.md) | Translates Python, JavaScript and TypeScript codebases into GNN state-space models; reverse synthesizer | Format docs `cogant/docs/theory/gnn_format.md`; emitted models `cogant/examples/zoo/*/model.gnn.md` | `v5.0.3` · `8f293d9b` declared in the `upstream` extra of `cogant/pyproject.toml`, resolved in `cogant/uv.lock` | Receipt `cogant/docs/evaluation/gnn_v5_0_3_compatibility.json` (20/20 upstream steps); required CI job `gnn-conformance` (`gnn conformance (pinned)`); `tests/unit/test_gnn_upstream_bridge.py` |
| [active_inference_battleship](active-inference-battleship.md) | GNN-based active inference for Battleship with a playable dashboard | `models/battleship.gnn.md`; contract table `docs/GNN_SPECIFICATION.md` | `v5.0.3` · `8f293d9b` in `models/gnn.pin.json` | `bun run model:check`; `scripts/contract.ts`, `scripts/parse_gnn.py`; CI job `model-check` (`gnn conformance (pinned)`) |
| [policy_entanglement](policy-entanglement.md) | Mean-field deformation framework for multi-stream policy ensembles | `gnn/bernoulli_toy.gnn.md`, `gnn/k_stream_ensemble.gnn.md`, `gnn/production_ensemble.gnn.md` | `v5.0.3` · `8f293d9b` in `gnn/upstream.json` | `tests/test_gnn_validation.py`, `tests/test_gnn_upstream_conformance.py`; CI job `gnn-conformance` (`gnn conformance (pinned)`) |
| [active_fractal_rabbit](active-fractal-rabbit.md) | Closed-loop active inference over synthetic waypoint data (claim boundary: synthetic-only) | `models/single_factor_reference.gnn.md`, `models/factorial_latent_mode.gnn.md` | `v5.0.3` · `8f293d9b` in `.github/gnn-pin.json` | CI job `gnn-conformance` (`gnn conformance (pinned)`) reads the pin and validates the spec; `tests` job and `tests/test_gnn_spec.py` |
| [Active_Cardference](active-cardference.md) | Deck-generic card-game engine with GNN-specified active inference agents | `specs/kuhn_aif_p1.gnn.md` | `v5.0.3` · `8f293d9b` in `specs/gnn-pin.json` | CI step "Validate the GNN specification against the pinned parser"; `scripts/validate_gnn_spec.py` |
| [fep_formal](fep-formal.md) | Lean 4 catalog of FEP, active inference and Bayesian mechanics theorems | Bridge contract `docs/design/gnn-bridge/bridge-contract.md`; Lean `lean/FepSketches/gnn_document.lean` | Content hashes in `specs/gnn-bridge-w1-bridge-operations/syntax-pin.json` and, in `specs/gnn-bridge-w2-source-custody/source-pin.json`, GNN commit `0f563547` with six file hashes (no release tag) | `python` job step "Check the frozen-custody bridge gates"; CI job `gnn-conformance` publishes `gnn conformance (pinned)`; `tests/test_gnn_bridge_operations.py`; pair qualification `source-pair-qualification.yml` is dispatch-only |
| [fundamentals](fundamentals.md) | Python companion to *Fundamentals of Active Inference* (Namjoshi, MIT Press 2026) | `specs/weather_pomdp.gnn.md` | `v5.0.3` · `8f293d9b` in `specs/gnn-pin.json` | CI job `gnn-conformance`; `tests/test_gnn_spec.py` |
| [cognitive](cognitive.md) | Discrete and continuous active inference package plus a knowledge vault | `specs/discrete_variational.gnn.md` | `v5.0.3` · `8f293d9b` in `specs/gnn-pin.json` | CI job `gnn-conformance`; `code/tests/test_gnn_conformance.py` |
| [ActiveBlockference](active-blockference.md) | CPU-oriented active inference for radCAD/cadCAD grid simulations | `specs/gridworld_3x3.gnn.md` | `v5.0.3` · `8f293d9b` in `specs/gnn-pin.json` | CI job `gnn-conformance`; `tests/test_gnn_spec.py` |
| [GEN24](gen24.md) | Synthetic domain material generation; GNN enters through the NHIM module | `NHIM/nhim_inquiry_agent.gnn.md` | `v5.0.3` · `8f293d9b` in `NHIM/gnn-pin.json` | CI job `gnn-conformance` (`gnn conformance (pinned)`) fetches the pin and validates the spec; `scripts/validate_gnn_spec.py` |
| [CEREBRUM](cerebrum.md) | Case-enabled reasoning engine with Bayesian representations | `models/case_conditioned_active_inference.gnn.md` | `v5.0.3` · `8f293d9b` in `.github/gnn-pin.json` | CI job `gnn-conformance` (`gnn conformance (pinned)`); `tests/core/test_gnn_spec.py` |
| [Biofirm](biofirm.md) | Ecological control and bioregion decision analysis with a pymdp controller | `models/biofirm_ecosystem_variable.gnn.md` | `v5.0.3` · `8f293d9b` in `.github/gnn-pin.json` | CI job `gnn-conformance` (`gnn conformance (pinned)`); `tests/test_gnn.py` |
| [Active_InferAnts](active-inferants.md) | Multi-language active inference framework (ant-colony multi-agent simulation) | `gnn/gaussian_agent.gnn.md` | `v5.0.3` · `8f293d9b` in `gnn/upstream.json` | CI job `gnn-conformance` (`gnn conformance (pinned)`); `tests/test_gnn_conformance.py` |
| [ActInf_RxInfer](actinf-rxinfer.md) | RxInfer.jl working example: Beta-Bernoulli coin-bias inference | `specs/coin_flip.gnn.md` | `v5.0.3` · `8f293d9b` in `specs/gnn-pin.json` | CI job `gnn-conformance`; `scripts/validate_gnn_spec.py` |
| [Active_Skillference](active-skillference.md) | Micro-learning platform whose prerequisite graph rests on a discrete agent | `gnn/discrete_active_inference_agent.gnn.md` | `v5.0.3` · `8f293d9b` in `gnn/upstream.json` | CI job `gnn-conformance` in `gates.yml` (`gnn conformance (pinned)`); `tests/test_gnn_conformance.py` |
| [ActiveInferenceImplementations](active-inference-implementations.md) | Educational RxInfer.jl notebooks: mean-field variational message passing | `specs/normal_vmp.gnn.md` | `v5.0.3` · `8f293d9b` in `specs/gnn-pin.json` | CI job `gnn-conformance`; `scripts/validate_gnn_spec.py` |
| [active_inference_power](active-inference-power.md) | Statistical-power simulations for adaptive studies with an embedded agent | `models/reference_action_loop.gnn.md`, `models/sequential_evidence_agent.gnn.md` | `v5.0.3` · `8f293d9b` in `models/gnn-pin.json` | CI job `gnn-conformance` (`gnn conformance (pinned)`) runs `scripts/validate_gnn_specs.py` and `tests/test_gnn_specs.py` |
| [ants](ants.md) | Ant-colony foraging in a T-maze (code for *Active Inferants*, 2021) | `gnn/ant_colony_foraging.gnn.md` | `v5.0.3` · `8f293d9b` in `gnn/upstream.json` | CI job `gnn-conformance` (`gnn conformance (pinned)`); `tests/test_gnn_conformance.py` |
| [on_policy_distillation](on-policy-distillation.md) | On-policy distillation formalized as active inference: three small models | `gnn/bernoulli_toy.gnn.md`, `gnn/graph_world.gnn.md`, `gnn/si_tmaze.gnn.md` | `v5.0.3` · `8f293d9b` in `gnn/upstream.json` | CI job `release-gate` (`gnn conformance (pinned)`); `tests/test_gnn_upstream_conformance.py`, `tests/test_gnn.py` |
| [Active_Fedference](https://github.com/ActiveInferenceInstitute/Active_Fedference) † | Robust federated active inference research project (FedGVI-style generalized variational inference, discrete-categorical) with a reproducible manuscript and evidence pipeline; five declared models | `src/fedference/models/*.gnn.md`: `fedgvi-sites`, `hierarchy`, `hybrid-tracking`, `moving-observer`, `sentinel` | `8f293d9b` in `src/fedference/config/gnn-backend-lock.json` (no tag field) | CI job `gnn-shard` publishes `gnn conformance (pinned)` and checks out the locked revision; `.github/workflows/ci.yml` |

## How to read this registry

- **Pinned GNN revision** names the GNN release tag and commit recorded in the
  repository's own pin file or dependency declaration, or in the contract named
  in the row. Twenty of the twenty-two repositories pin GNN
  `v5.0.3` at commit `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`;
  `Active_Fedference` records the same commit in its backend lock without a tag
  field, and `fep_formal` pins by content hash instead of a release tag. A pin
  records what a repository declares, not what it currently passes.
- **Conformance / validation evidence** names the workflow job, test file or
  documented command that a repository uses to check its GNN surface. Where no
  GNN-specific CI job exists, the row says so and cites the test file instead.
- **What it models** is taken from the repository's own description or README,
  not from a third party.

## Method and scope

Every row was verified on 2026-11-02 from the repository's origin default branch
with read-only GitHub API calls: `git/trees/<branch>?recursive=1` to locate
files and `contents/<path>?ref=<branch>` to read pin files, specifications and
workflow definitions. Pin files and workflow job names were re-read on
2026-11-06: no revision value moved, and the job keys and display names below
are current. File presence, pin contents and workflow job names are verified;
current CI status and local reproducibility are not. A row that could not be
verified says "not verified" rather than repeating an assumption.

Candidates came from scanning the organization for the markers of GNN use
(`.gnn.md` specification files and GNN pin files), and every repository below
was then read from origin directly. The twenty-two repositories in the table are
the ones with a committed GNN specification or pin on the default branch.
Repositories that only mention GNN in passing are not listed; repositories that
record the revision outside a pin file are in the section below.

## Refresh contract

This registry is a dated snapshot of mechanisms, not a currency ledger: the
read date above names the last full pass, and each row names the pin file (or
contract) that records what the repository declares and the job, test or
command that carries its evidence. Currency is owned by those artifacts, not by
this page: when a row and the repository disagree, the repository wins.

Rows are re-verified from origin with the read-only method above as part of
GNN release-tag preparation and on request; a closeout that moves a row's
subject re-verifies it, as the
[paired-revision procedure](../development/fep_lean_paired_revision.md) does
for the `fep_formal` row when it bumps the companion pin. No GNN gate or
workflow reads this tree, by design: mechanism-named rows make no
per-repository currency claim a gate could check.

## Revisions recorded in the repository's own contract

These repositories carry GNN files or GNN bridge code on the default branch and
record the upstream revision in their own contract; four of these rows also
carry a `.github/gnn-pin.json` with a dedicated `gnn conformance (pinned)` job
(re-read 2026-11-06 and noted in their rows). Each row names the contract; every
path and revision below was verified from origin on 2026-11-02 and qualified on
2026-11-06. The contract file links to that file on the repository's
default branch, so a reader can reach it in one step. These five are rows rather
than their own pages on purpose: a page would add files under `docs/**`, which
moves the manuscript documentation file count, so pages for them are a scheduled
count-changing change rather than an oversight.

| Repository | GNN content | Where the revision is recorded |
| --- | --- | --- |
| [ActiveInferenceCategoryTheory](https://github.com/ActiveInferenceInstitute/ActiveInferenceCategoryTheory) | `gnn/learner_perception_model.md`, `gnn/map_the_model.template.md` | Verification run in [`gnn/README.md`](https://github.com/ActiveInferenceInstitute/ActiveInferenceCategoryTheory/blob/main/gnn/README.md) at commit `8f293d9b` (v5.0.3); it also carries `.github/gnn-pin.json` (`v5.0.3`) and a `gnn conformance (pinned)` job (2026-11-06) |
| [act_inf_metaanalysis](https://github.com/ActiveInferenceInstitute/act_inf_metaanalysis) | `gnn/h1_fep_universality.md`, `gnn/h4_predictive_coding.md`, `gnn/h8_language_aif.md` | [`gnn/README.md`](https://github.com/ActiveInferenceInstitute/act_inf_metaanalysis/blob/main/gnn/README.md): revision `v5.0.3`/`8f293d9b`, an archive extract recipe, and recorded model SHA-256 digests; it also carries `.github/gnn-pin.json` (`v5.0.3`) and a `gnn conformance (pinned)` job (2026-11-06) |
| [courses](https://github.com/ActiveInferenceInstitute/courses) | `course_development/gnn/01_gnn_syntax/` (syntax module, lab, model exemplars) | [`course_development/gnn/01_gnn_syntax/models/README.md`](https://github.com/ActiveInferenceInstitute/courses/blob/main/course_development/gnn/01_gnn_syntax/models/README.md) provenance table (release `v5.0.3`, commit `8f293d9b`) with a `git show` and `diff` recipe against the pinned corpus; it also carries `.github/gnn-pin.json` (`v5.0.3`) and a `gnn conformance (pinned)` job (2026-11-06) |
| [pymdp](https://github.com/ActiveInferenceInstitute/pymdp) | Bridge code `pymdp/gnn.py`, `examples/gnn/publish_gnn_spec.py`, `test/test_gnn_bridge.py`, `docs-mkdocs/guides/gnn-interoperability.md` | `GNN_PINNED_RELEASE` and `GNN_PINNED_REVISION` constants in [`pymdp/gnn.py`](https://github.com/ActiveInferenceInstitute/pymdp/blob/master/pymdp/gnn.py) (default branch `master`) |
| [Active_Inference_Ontology](https://github.com/ActiveInferenceInstitute/Active_Inference_Ontology) | Term export `GNN/act_inf_ontology_terms.json`, `GNN/README.md` | [`GNN/declared-labels.json`](https://github.com/ActiveInferenceInstitute/Active_Inference_Ontology/blob/main/GNN/declared-labels.json): release, commit and SHA-256 of the audited file; this repository pins *it* at `4450e8dd` in `src/gnn/ontology/ontology-pin.json`; it also carries `.github/gnn-pin.json` (`v5.0.3`) and a `gnn conformance (pinned)` job (2026-11-06) |

## Exclusions

- `Active_Fedference` was excluded on 2026-11-02 because no GNN content existed
  on its `main` branch then. That exclusion is stale: re-read on 2026-11-06, its
  `main` carries five declared `.gnn.md` models, a backend lock recording
  revision `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, and a `gnn-shard` job
  publishing `gnn conformance (pinned)`. It is the twenty-second row above.

## Related

- [GNN documentation hub](../README.md)
- [Flat documentation index](../INDEX.md)
- [GEO_INFER interchange notes](../other/geo_infer/README.md)
- [FEP_Formal bridge notes](../other/fep_lean/README.md)
