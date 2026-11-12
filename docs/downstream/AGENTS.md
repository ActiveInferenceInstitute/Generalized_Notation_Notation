# docs/downstream — agent guide

## Overview

Registry of external repositories that apply GNN. [README.md](README.md) is the
index; one Markdown file per repository carries the full evidence, except rows
marked †, whose pages are deferred to the scheduled count-changing change (a
page would move `GNN_DOC_FILE_COUNT` and owe the SC-22 ritual). Every claim must
be verified against the repository it describes on that repository's origin
default branch.

## Contents

| File | Role |
|------|------|
| [README.md](README.md) | Index table, method, scope, pinned-conformance baseline, contract rows, exclusions |
| [importers.json](importers.json) | Machine-readable mirror of the README tables: one row per repository with its pin path, pin revision, specification count and conformance job. `scripts/check_importer_registry.py` keeps the two forms one artifact |
| `active-inference-language-model.md` | AILM language-stack models |
| `geo-infer.md` | GEO-INFER spatial interchange |
| `cogant.md` | Code-to-GNN translation |
| `active-inference-battleship.md` | Battleship model and dashboard |
| `policy-entanglement.md` | Policy-ensemble GNN examples |
| `active-fractal-rabbit.md` | Waypoint active-inference scaffold |
| `active-cardference.md` | Card-game agent specifications |
| `fep-formal.md` | Lean ↔ GNN bridge contract |
| `fundamentals.md` | Textbook companion |
| `cognitive.md` | Active inference package and vault |
| `active-blockference.md` | radCAD/cadCAD grid simulations |
| `gen24.md` | NHIM inquiry agent |
| `cerebrum.md` | Case-conditioned active inference |
| `biofirm.md` | Ecological control model |
| `active-inferants.md` | Multi-language framework |
| `actinf-rxinfer.md` | RxInfer.jl coin-flip example |
| `active-skillference.md` | Micro-learning prerequisite graph |
| `active-inference-implementations.md` | Educational RxInfer.jl notebooks |
| `active-inference-power.md` | Statistical-power simulations |
| `ants.md` | Ant-colony foraging simulation |
| `on-policy-distillation.md` | On-policy distillation models |

## Verification rules

- Verify from the repository itself on its origin default branch; never copy a
  claim from internal notes or from another row.
- Cite the pin file, or the contract that records the revision (each row names
  it), and the workflow job, test file or command that carries the evidence. If
  evidence is absent, write "not verified" or "no GNN-specific CI job" instead of
  implying more.
- Re-verify a row when its repository changes its pin file or GNN specification,
  as part of GNN release-tag preparation and on request. The README's
  "Refresh contract" section owns what is dated and what is not: rows name
  mechanisms, and each consumer's pin file and CI job own currency.
- The gate wired to this tree checks *internal consistency only*:
  `scripts/check_importer_registry.py` (errors only, offline, in `run.sh gates`
  and the `quality` job) proves that `importers.json` and the README tables list
  the same repositories, that every row is well formed, that each pin matches
  the upstream release this registry declares, and that the spec-count contract
  per row kind holds. It makes no claim about any repository's current state.
- Currency stays report-only. `--online` re-reads every public repository's pin
  and conformance job at origin and prints drift; it is for release preparation
  and never part of a blocking gate, because a mechanism-named row must not fail
  a build when a consumer moves.

## Related

- [../README.md](../README.md) — documentation hub
- [../SPEC.md](../SPEC.md) — doc-tree design requirements
- [../expected_dirs.txt](../expected_dirs.txt) — canonical directory list
