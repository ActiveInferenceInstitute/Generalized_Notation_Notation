# Downstream registry — repositories that apply GNN

Who already uses Generalized Notation Notation (GNN), what they model, which GNN
revision they pin, and where their conformance evidence lives. One row per
repository; each row links to a page with the full evidence, except rows marked
†, whose pages are deferred to the same scheduled count-changing change as the
contract rows below (their links go to the repository's own evidence).

The registry contains 42 repositories: 41 declare `v5.1.0` and one records
content custody.

At a glance: twenty-three repositories carry a committed GNN specification or a
pin in a dependency declaration and record an upstream revision. Twenty-two of
them pin `v5.1.0` at commit `75c834bd`,
and `fep_formal` records content hashes for the reviewed source pair.
Four more record the revision in their own contract rather than in a pin file,
and `pymdp` now records it in `.github/gnn-pin.json`. Fourteen carry a pin file
and a `gnn conformance (pinned)` job but no GNN specification, listed under
[pinned conformance baseline](#pinned-conformance-baseline-without-a-gnn-specification).

| Repository | What it models | GNN specification | Pinned GNN revision | Conformance / validation evidence | Evidence categories |
| --- | --- | --- | --- | --- | --- |
| [active_inference_language_model](active-inference-language-model.md) | Explicit discrete language stack in pymdp, letters through document position | `gnn/models/L1_letter.gnn.md` … `L6_narrative.gnn.md`; AILM-side contract `gnn/SPEC.md` | `v5.1.0` · `75c834bd` in `gnn/upstream.json` | CI job key `gnn-interop`, display `gnn conformance (pinned)`; `scripts/verify_gnn_release.py` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [GEO_INFER](geo-infer.md) | Geospatial active inference monorepo (H3 topology, place and risk models) | Interchange contract `GEO-INFER-ACT/docs/gnn_interchange.md`; example `GEO-INFER-ACT/examples/cascadia/data/smith_river_homing.gnn.md` | `v5.1.0` · `75c834bd` in `.github/gnn-pair.json` | Workflow `Paired GNN interchange` job key `interchange`: `gnn conformance (pinned)` on Python 3.12 and `gnn interchange (3.11)` on Python 3.11; `GEO-INFER-TEST/validate_gnn_interchange.py` | `reference`, `source_pin`, `conformance`, `paired_interchange` <!-- evidence_categories: reference, source_pin, conformance, paired_interchange --> |
| [COGANT](cogant.md) | Translates Python, JavaScript and TypeScript codebases into GNN state-space models; reverse synthesizer | Format docs `cogant/docs/theory/gnn_format.md`; emitted models `cogant/examples/zoo/*/model.gnn.md` | `v5.1.0` · `75c834bd` in `.github/gnn-pin.json`, the `upstream` extra of `cogant/pyproject.toml` and `cogant/uv.lock` | CI job `gnn-conformance` (`gnn conformance (pinned)`) runs the shared reader and the upstream bridge probe; default-branch admission merged in PR #23. The 5.0.3 compatibility receipt records an earlier revision. | `reference`, `source_pin`, `conformance`, `installed_import`, `cli_invocation` <!-- evidence_categories: reference, source_pin, conformance, installed_import, cli_invocation --> |
| [active_inference_battleship](active-inference-battleship.md) | GNN-based active inference for Battleship with a playable dashboard | `models/battleship.gnn.md`; contract table `docs/GNN_SPECIFICATION.md` | `v5.1.0` · `75c834bd` in `models/gnn.pin.json` | `bun run model:check`; `scripts/contract.ts`, `scripts/parse_gnn.py`; CI job `model-check` (`gnn conformance (pinned)`) | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [policy_entanglement](policy-entanglement.md) | Mean-field deformation framework for multi-stream policy ensembles | `gnn/bernoulli_toy.gnn.md`, `gnn/k_stream_ensemble.gnn.md`, `gnn/production_ensemble.gnn.md` | `v5.1.0` · `75c834bd` in `gnn/upstream.json` | `tests/test_gnn_validation.py`, `tests/test_gnn_upstream_conformance.py`; CI job `gnn-conformance` (`gnn conformance (pinned)`) | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [active_fractal_rabbit](active-fractal-rabbit.md) | Closed-loop active inference over synthetic waypoint data (claim boundary: synthetic-only) | `models/single_factor_reference.gnn.md`, `models/factorial_latent_mode.gnn.md` | `v5.1.0` · `75c834bd` in `.github/gnn-pin.json` | CI job `gnn-conformance` invokes `./run.sh gnn`, which runs the shared reader and `tests/test_gnn_spec.py` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [Active_Cardference](active-cardference.md) | Deck-generic card-game engine with GNN-specified active inference agents | `specs/kuhn_aif_p1.gnn.md` | `v5.1.0` · `75c834bd` in `specs/gnn-pin.json` | CI job `test` (`gnn conformance (pinned)`), step "GNN conformance against the pinned parser" (`./run.sh gnn`); `scripts/validate_gnn_spec.py` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [fep_formal](fep-formal.md) | Lean 4 catalog of FEP, active inference and Bayesian mechanics theorems | Bridge contract `docs/design/gnn-bridge/bridge-contract.md`; Lean `lean/FepSketches/gnn_document.lean` | Content hashes in `specs/gnn-bridge-w1-bridge-operations/syntax-pin.json` and `specs/gnn-bridge-w2-source-custody/source-pin.json`, with six GNN owner files at `c4489643` | `scripts/gate.sh gnn`, `tests/test_gnn_bridge_operations.py`, and the locally compiled Q7 receipt in FEP PR #276; hosted `source-pair-qualification.yml` is dispatch-only | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [fundamentals](fundamentals.md) | Python companion to *Fundamentals of Active Inference* (Namjoshi, MIT Press 2026) | `specs/weather_pomdp.gnn.md` | `v5.1.0` · `75c834bd` in `specs/gnn-pin.json` | CI job `gnn-conformance`; `tests/test_gnn_spec.py` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [cognitive](cognitive.md) | Discrete and continuous active inference package plus a knowledge vault | `specs/discrete_variational.gnn.md` | `v5.1.0` · `75c834bd` in `specs/gnn-pin.json` | CI job `gnn-conformance`; `code/tests/test_gnn_conformance.py` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [ActiveBlockference](active-blockference.md) | CPU-oriented active inference for radCAD/cadCAD grid simulations | `specs/gridworld_3x3.gnn.md` | `v5.1.0` · `75c834bd` in `specs/gnn-pin.json` | CI job `gnn-conformance`; `tests/test_gnn_spec.py` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [GEN24](gen24.md) | Synthetic domain material generation; GNN enters through the NHIM module | `NHIM/nhim_inquiry_agent.gnn.md` | `v5.1.0` · `75c834bd` in `NHIM/gnn-pin.json` | CI job `gnn-conformance` (`gnn conformance (pinned)`) fetches the pin and validates the spec; `scripts/validate_gnn_spec.py` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [CEREBRUM](cerebrum.md) | Case-enabled reasoning engine with Bayesian representations | `models/case_conditioned_active_inference.gnn.md` | `v5.1.0` · `75c834bd` in `.github/gnn-pin.json` | CI job `gnn-conformance` (`gnn conformance (pinned)`); `tests/core/test_gnn_spec.py` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [Biofirm](biofirm.md) | Ecological control and bioregion decision analysis with a pymdp controller | `models/biofirm_ecosystem_variable.gnn.md` | `v5.1.0` · `75c834bd` in `.github/gnn-pin.json` | CI job `gnn-conformance` (`gnn conformance (pinned)`); `tests/test_gnn.py` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [Active_InferAnts](active-inferants.md) | Multi-language active inference framework (ant-colony multi-agent simulation) | `gnn/gaussian_agent.gnn.md` | `v5.1.0` · `75c834bd` in `gnn/upstream.json` | CI job `gnn-conformance` (`gnn conformance (pinned)`); `tests/test_gnn_conformance.py` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [ActInf_RxInfer](actinf-rxinfer.md) | RxInfer.jl working example: Beta-Bernoulli coin-bias inference | `specs/coin_flip.gnn.md` | `v5.1.0` · `75c834bd` in `specs/gnn-pin.json` | CI job `gnn-conformance`; `scripts/validate_gnn_spec.py` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [Active_Skillference](active-skillference.md) | Micro-learning platform whose prerequisite graph rests on a discrete agent | `gnn/discrete_active_inference_agent.gnn.md` | `v5.1.0` · `75c834bd` in `gnn/upstream.json` | CI job `gnn-conformance` in `gates.yml` (`gnn conformance (pinned)`); `tests/test_gnn_conformance.py` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [ActiveInferenceImplementations](active-inference-implementations.md) | Educational RxInfer.jl notebooks: mean-field variational message passing | `specs/normal_vmp.gnn.md` | `v5.1.0` · `75c834bd` in `specs/gnn-pin.json` | `GNN conformance` step in the `gate` job invokes `./run.sh gnn` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [active_inference_power](active-inference-power.md) | Statistical-power simulations for adaptive studies with an embedded agent | `models/reference_action_loop.gnn.md`, `models/sequential_evidence_agent.gnn.md` | `v5.1.0` · `75c834bd` in `models/gnn-pin.json` | CI job `gnn-conformance` (`gnn conformance (pinned)`) runs `scripts/validate_gnn_specs.py` and `tests/test_gnn_specs.py` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [ants](ants.md) | Ant-colony foraging in a T-maze (code for *Active Inferants*, 2021) | `gnn/ant_colony_foraging.gnn.md` | `v5.1.0` · `75c834bd` in `gnn/upstream.json` | CI job `gnn-conformance` (`gnn conformance (pinned)`); `tests/test_gnn_conformance.py` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [on_policy_distillation](on-policy-distillation.md) | On-policy distillation formalized as active inference: three small models | `gnn/bernoulli_toy.gnn.md`, `gnn/graph_world.gnn.md`, `gnn/si_tmaze.gnn.md` | `v5.1.0` · `75c834bd` in `gnn/upstream.json` | CI job `release-gate` (`gnn conformance (pinned)`); `tests/test_gnn_upstream_conformance.py`, `tests/test_gnn.py` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [pymcp](https://github.com/ActiveInferenceInstitute/pymcp) † | Model Context Protocol and REST surface over PyMDP; GNN enters as a 3x3 gridworld reference test instance | `models/gridworld_3x3.gnn.md` | `v5.1.0` · `75c834bd` in `.github/gnn-pin.json` | CI job `gnn-conformance` publishes `gnn conformance (pinned)` and installs exactly the pinned revision before parsing the sketch; `.github/workflows/ci.yml` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [Active_Fedference](https://github.com/ActiveInferenceInstitute/Active_Fedference) † | Robust federated active inference research project (FedGVI-style generalized variational inference, discrete-categorical) with a reproducible manuscript and evidence pipeline; one standard GNN model and four project extension documents | `src/fedference/models/sentinel.gnn.md`; the moving-observer, hierarchy, hybrid-tracking and FedGVI-sites documents are Fedference extension payloads (`*.fed.md`, `*.fed.ir.json`), not GNN specifications | `v5.1.0` · `75c834bd` in `.github/gnn-pin.json` | CI job `gnn-shard` publishes `gnn conformance (pinned)` and qualifies the importer pin separately from the locked runtime revision (PR #35); `.github/workflows/ci.yml` | `reference`, `source_pin`, `conformance`, `cli_invocation` <!-- evidence_categories: reference, source_pin, conformance, cli_invocation --> |

## How to read this registry

- **Pinned GNN revision** names the GNN release tag and commit recorded in the
  repository's own pin file or dependency declaration, or in the contract named
  in the row. Twenty-two rows of the table
  pin `v5.1.0` at commit `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`, and `fep_formal`
  pins by content hash at its reviewed source revision. A pin records what
  a repository declares, not what it currently passes.
- **Conformance / validation evidence** names the workflow job, test file or
  documented command that a repository uses to check its GNN surface. Where no
  GNN-specific CI job exists, the row says so and cites the test file instead.
- **What it models** is taken from the repository's own description or README,
  not from a third party.

## Method and scope

Every row was verified from the repository's origin default branch with
read-only GitHub API calls: `git/trees/<branch>?recursive=1` to locate files and
`contents/<path>?ref=<branch>` to read pin files, specifications and workflow
definitions. The first full pass was on 2026-11-02; pin files and workflow job
names were re-read on 2026-11-06, and every row, plus the baseline and contract
rows, was re-read again on 2026-11-11, which is the observation date the values
below carry. File presence,
pin contents and workflow job names are verified for that observation date;
current CI status, branch rulesets and local reproducibility require separate
evidence. The 2026-11-15 targeted refresh re-read the changed pins and workflow
declarations from immutable revisions. The repository rows below or their
linked detail pages identify the corresponding source trees.

Candidates came from scanning every non-empty repository in the organization
for the markers of GNN use (`.gnn.md` specification files, GNN pin files and
`gnn conformance (pinned)` jobs), and every repository below was then read from
origin directly. The twenty-three repositories in the table are the ones with a
committed GNN specification or a pin in a dependency declaration; the fourteen
in the baseline section carry a pin file and a conformance job but no
specification; the five in the contract section record the revision in their own
contract. Repositories that only mention GNN in passing are not listed, and
`Working_Groups` and `aii-org` carry no GNN artifact; `Active_Inference_Un0` and
`Bots` are empty.

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
for the `fep_formal` row when it bumps the companion pin.

The tables below exist twice: as prose here and as
[importers.json](importers.json), one row per repository with its pin path,
revision, specification count and conformance job. `scripts/check_importer_registry.py`
keeps the two forms one artifact — same repositories, well-formed rows, each
row's pin resolving to the tag it names, the declared upstream release
resolving to its tag, and the spec-count contract per row kind — and runs
offline in `run.sh gates` and the `quality` job. That gate checks internal consistency only; it makes no claim about any
repository's current state. Currency stays report-only: the same script with
`--online` resolves each public repository's default branch once and reads its
pin, workflow and `.github/GNN.md` pointer from that commit. With `--json`,
`observations` records each checked repository's revision and whether the read
matched the registry, found drift or was unavailable. This report supports
release preparation and stays outside the blocking gate.

A consumer can check its own checkout against the same reader GNN ships:

```bash
python -m gnn.importers check --repo . --expected-commit 75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf
```

## Workflow groups

`importers.json` also carries a `workflow_groups` block: one entry per set of
repositories that share a concern, with the canonical implementation of that
concern and the step each member takes to use it. The `gnn-fep` group is the
[GNN–FEP contract](../workflows/gnn-fep.md): twenty-two members, one pin reader
(`src/gnn/importers.py`, called as `python -m gnn.importers check --repo .`).

Each member row records its role, how it reaches the reader (`direct`, or
`wrapper` with the file and the reason), its current arm, its pin path, its
`.github/GNN.md` pointer, its documentation hub and its manuscript producer.
`scripts/check_importer_registry.py` compares that block with the contract page's
member table in both directions and refuses a member that records a private pin
reader instead of the canonical command, so the contract and the data cannot
drift apart. It records the copies members still carry; it does not claim they
are already gone.

`Generalized_Notation_Notation` is the upstream member, not an importer to
migrate: it owns `src/gnn/importers.py` and has no GNN pin or pointer. Do not run
the importer command above on this checkout; its companion pins describe
dependencies, not a GNN revision it consumes. Follow the contract's
[upstream boundary](../workflows/gnn-fep.md#the-reader-is-not-for-this-repositorys-own-tree).
Verify the reader with `python -m pytest tests/gnn/test_importers.py` and the
group with `python scripts/check_importer_registry.py`. Retain `./run.sh gates`
as this repository's whole gate; its focused `lint`, `type` and `docs` arms
remain available.

## Pinned conformance baseline without a GNN specification

Fourteen repositories carry no `.gnn.md` specification but pin GNN and publish
the `gnn conformance (pinned)` context. Each carries `.github/gnn-pin.json` and
`.github/GNN.md`, plus a `gnn-conformance` job in the workflow named below
(re-read 2026-11-15). All fourteen declare `v5.1.0` · `75c834bd`. The job bodies share a baseline
shape: each reconstructs the pin, installs exactly that revision and runs the
repository's GNN sketch check, and `.github/GNN.md` records what it parses.
`Start` replaces the sketch check with an assertion that its registered revision
equals the pin, plus its orchestrator tests.

| Repository | Workflow carrying the job | Evidence categories |
| --- | --- | --- |
| [AEOS](https://github.com/ActiveInferenceInstitute/AEOS) | `.github/workflows/aeos.yml` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [ATLAS](https://github.com/ActiveInferenceInstitute/ATLAS) | `.github/workflows/gate.yml` at [a0abb07](https://github.com/ActiveInferenceInstitute/ATLAS/blob/a0abb07826d7d694a16d3ee87f4a93b65db1fa11/.github/workflows/gate.yml) | `reference`, `source_pin`, `conformance`, `installed_import`, `cli_invocation` <!-- evidence_categories: reference, source_pin, conformance, installed_import, cli_invocation --> |
| [ActiveInferenceJournal](https://github.com/ActiveInferenceInstitute/ActiveInferenceJournal) | `.github/workflows/gnn-conformance.yml` at [6bb7a85](https://github.com/ActiveInferenceInstitute/ActiveInferenceJournal/blob/6bb7a859de4c867884d27783bc1e0d41837e7bed/.github/workflows/gnn-conformance.yml) | `reference`, `source_pin`, `conformance`, `cli_invocation`, `installed_import` <!-- evidence_categories: reference, source_pin, conformance, cli_invocation, installed_import --> |
| [Garden-of-Iris](https://github.com/ActiveInferenceInstitute/Garden-of-Iris) | `.github/workflows/ci.yml` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [ISSS](https://github.com/ActiveInferenceInstitute/ISSS) | `.github/workflows/ci.yml` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [Journal_Utilities](https://github.com/ActiveInferenceInstitute/Journal_Utilities) | `.github/workflows/test.yml` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [Knowledge-Engineering](https://github.com/ActiveInferenceInstitute/Knowledge-Engineering) | `.github/workflows/ci.yml` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [NoOrg](https://github.com/ActiveInferenceInstitute/NoOrg) | `.github/workflows/ci.yml` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [Parr_et_al_2022_ActInf_Textbook](https://github.com/ActiveInferenceInstitute/Parr_et_al_2022_ActInf_Textbook) | `.github/workflows/ci.yml` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [Research-Discovery-Engine](https://github.com/ActiveInferenceInstitute/Research-Discovery-Engine) | `.github/workflows/ci.yml` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [Start](https://github.com/ActiveInferenceInstitute/Start) | `.github/workflows/ci.yml` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [Symposium](https://github.com/ActiveInferenceInstitute/Symposium) | `.github/workflows/ci.yml` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [gen25](https://github.com/ActiveInferenceInstitute/gen25) | `.github/workflows/docs.yml` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [institute_website](https://github.com/ActiveInferenceInstitute/institute_website) | `.github/workflows/ci.yml` | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |

## Revisions recorded in the repository's own contract

These repositories carry GNN files or GNN bridge code on the default branch and
record the upstream revision in their own contract; all five also carry a
`.github/gnn-pin.json`. Courses and Active Inference Ontology publish dedicated
conformance jobs. ActiveInferenceCategoryTheory and act_inf_metaanalysis run
conformance inside a combined job. The `pymdp` repository runs the reader from
its local `./gate.sh gnn` arm. Each row names the contract; every
path and revision below was verified from origin on 2026-11-02 and qualified on
2026-11-11. The contract file links to that file on the repository's default
branch, so a reader can reach it in one step; those links are navigation targets
that can change after the observation date. These five are rows rather
than their own pages on purpose: a page would add files under `docs/**`, which
moves the manuscript documentation file count, so pages for them are a scheduled
count-changing change rather than an oversight.

| Repository | GNN content | Where the revision is recorded | Evidence categories |
| --- | --- | --- | --- |
| [ActiveInferenceCategoryTheory](https://github.com/ActiveInferenceInstitute/ActiveInferenceCategoryTheory) | `gnn/learner_perception_model.md`, `gnn/map_the_model.template.md` | Verification run in [`gnn/README.md`](https://github.com/ActiveInferenceInstitute/ActiveInferenceCategoryTheory/blob/main/gnn/README.md) at commit `75c834bd` (v5.1.0); it also carries `.github/gnn-pin.json` (`v5.1.0`). The `gnn conformance (pinned)` step runs inside the `gate` job. [Workflow at 8d92412](https://github.com/ActiveInferenceInstitute/ActiveInferenceCategoryTheory/blob/8d9241208fa3b5f6c7220ed4c0cfd2774efcdeb0/.github/workflows/docs.yml) | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [act_inf_metaanalysis](https://github.com/ActiveInferenceInstitute/act_inf_metaanalysis) | `gnn/h1_fep_universality.md`, `gnn/h4_predictive_coding.md`, `gnn/h8_language_aif.md` | [`gnn/README.md`](https://github.com/ActiveInferenceInstitute/act_inf_metaanalysis/blob/main/gnn/README.md): revision `v5.1.0`/`75c834bd`, an archive extract recipe, and recorded model SHA-256 digests; it also carries `.github/gnn-pin.json` (`v5.1.0`). The `Pinned GNN conformance` step runs inside the `gate` job. [Workflow at 4f00863](https://github.com/ActiveInferenceInstitute/act_inf_metaanalysis/blob/4f00863556ed38d322b8eddb393cea20e8e09825/.github/workflows/ci.yml) | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [courses](https://github.com/ActiveInferenceInstitute/courses) | `course_development/gnn/01_gnn_syntax/` (syntax module, lab, model exemplars) | [`course_development/gnn/01_gnn_syntax/models/README.md`](https://github.com/ActiveInferenceInstitute/courses/blob/main/course_development/gnn/01_gnn_syntax/models/README.md) provenance table (release `v5.1.0`, commit `75c834bd`; the exemplars are byte-identical at `v5.0.3`) with a `git show` and `diff` recipe against the pinned corpus; it also carries `.github/gnn-pin.json` (`v5.1.0`) and a `gnn conformance (pinned)` job (2026-11-11) | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |
| [pymdp](https://github.com/ActiveInferenceInstitute/pymdp) | Bridge code `pymdp/gnn.py`, `examples/gnn/publish_gnn_spec.py`, `test/test_gnn_bridge.py`, `docs-mkdocs/guides/gnn-interoperability.md` | `.github/gnn-pin.json` (`v5.1.0` · `75c834bd`) and `.github/GNN.md`, which names the local `./gate.sh gnn` arm; `GNN_PINNED_RELEASE` and `GNN_PINNED_REVISION` in [`pymdp/gnn.py`](https://github.com/ActiveInferenceInstitute/pymdp/blob/master/pymdp/gnn.py) mirror the pin; the scaling-manifest verifier landed in PR #12 (default branch `master`) | `reference`, `source_pin`, `conformance`, `cli_invocation` <!-- evidence_categories: reference, source_pin, conformance, cli_invocation --> |
| [Active_Inference_Ontology](https://github.com/ActiveInferenceInstitute/Active_Inference_Ontology) | Term export `GNN/act_inf_ontology_terms.json`, `GNN/README.md` | [`GNN/declared-labels.json`](https://github.com/ActiveInferenceInstitute/Active_Inference_Ontology/blob/main/GNN/declared-labels.json) records the audited GNN source at `v5.1.0` / `75c834bd` with its SHA-256. GNN separately pins the ontology repository at `ad808b12d51ec990227d52c528f62db36dfd54bc` (15 of 64 declared labels resolved) in `src/gnn/ontology/ontology-pin.json`; the repository also carries `.github/gnn-pin.json` (`v5.1.0`) and a `gnn conformance (pinned)` job (re-read 2026-11-12). | `reference`, `source_pin`, `conformance` <!-- evidence_categories: reference, source_pin, conformance --> |

## Exclusions

- `Active_Fedference` was excluded on 2026-11-02 because no GNN content existed
  on its `main` branch then. That exclusion is stale: re-read on 2026-11-06, its
  `main` carries five declared `.gnn.md` models, a backend lock recording
  revision `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`, and a `gnn-shard` job
  publishing `gnn conformance (pinned)`. It is the last row of the table above.

## Related

- [GNN documentation hub](../README.md)
- [Flat documentation index](../INDEX.md)
- [GEO_INFER interchange notes](../other/geo_infer/README.md)
- [FEP_Formal bridge notes](../other/fep_lean/README.md)

## Evidence categories

Each row declares `evidence_categories` in the JSON and in the adjacent prose
marker. The strict gate checks both directions, including duplicates and unknown
categories. These categories identify evidence surfaces; they do not certify a
successful run or imply that a source pin is an installed package import.

- `installed_import`: a consumer checks an installed GNN Python distribution.
- `source_pin`: a file declares the upstream source revision or content custody.
- `cli_invocation`: a consumer invokes the GNN command or shared reader explicitly.
- `paired_interchange`: a consumer validates the paired GNN interchange contract.
- `conformance`: the row names a conformance workflow or local gate.
- `reference`: the row provides a documentation or contract reference.

Online observations record `matched`, `drift`, or `unavailable`, with a reason
for unavailable authentication, permission, HTTP, or transport observations.
Incomplete observations do not report partial findings as confirmed drift.

The online command exits 2 for incomplete observations, 1 for confirmed failures,
and 0 when checks pass. JSON `status` distinguishes these outcomes explicitly.
