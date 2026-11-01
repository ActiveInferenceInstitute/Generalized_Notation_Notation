# Cascadia river homing: a GNN specification of a typed H3 model

> **Document Metadata**
> **Type**: Worked interchange example | **Audience**: Researchers, Developers | **Complexity**: Advanced
> **Cross-References**: [README.md](README.md) | [AGENTS.md](AGENTS.md)
> **Last Updated**: 2026-10-12 | **Version**: 0.1

## What the example is

GEO-INFER-ACT's Cascadia example models an adult salmon returning from the
Pacific up the lower Smith River (Del Norte County, California) to its natal
tributary, Rowdy Creek. The hidden state is one of 26 H3 resolution-8 cells
traced from measured USGS NHDPlus HR flow lines: the sea, 16 main-stem cells,
two estuary sloughs, three cells of lower Tryon Creek and four of lower
Rowdy Creek. Each six-hour step the fish senses a habitat cue and a natal-odor
cue (12 outcomes) and chooses hold, downstream, upstream or turn.

GEO-INFER renders the model as GNN v1 Markdown, this repository's strict
exporter turns it into a `gnn-geo-infer/1` artifact, and the artifact is
consumed twice: by GEO-INFER-ACT's `run_gnn_inference` (pymdp 1.0.4) and by a
Lean 4 instance in the FEP_Formal checkout. The canonical documentation is
`../GEO_INFER/GEO-INFER-ACT/docs/cascadia_river_homing.md` in the sibling
checkout (inline path, not a link: each repository validates its own links).

## What this repository checks

The fixtures in [tests/export/cascadia/](../../../tests/export/cascadia/) are
byte copies of the GEO-INFER source, its state-ID order and the artifact this
exporter produced. [test_geo_infer_cascadia.py](../../../tests/export/test_geo_infer_cascadia.py)
establishes:

| Check | Meaning |
| --- | --- |
| Golden export | `build_geo_infer_artifact` reproduces the committed artifact, and the CLI with `--space-kind h3` reproduces it byte for byte |
| B axis evidence | The strict extractor detects `next_state, previous_state, action` from the data with no contradiction |
| Transposed literal | Writing B with previous and next axes swapped is rejected |
| Exact literals | Every A, B, D and E literal is a dyadic decimal with denominator at most 64 |
| Locality | Positive transition mass only stays put or moves to an H3 neighbor at one resolution |
| Notation metadata | `derive_geo_metadata` reads `StepSeconds=21600` and `SpaceKind=h3` from the `## Time` section |
| Identifier validation | A noncanonical H3 identifier is rejected |

Run them with:

```bash
uv run --extra dev --extra geo-infer python -m pytest tests/export/test_geo_infer_cascadia.py -q
```

## Producing the artifact

The GEO-INFER example calls this exporter in its own environment:

```bash
uv run python -m gnn.export.geo_infer smith_river_homing.gnn.md \
  smith_river_homing.geo-infer.json --step-seconds 21600 \
  --space-kind h3 --state-ids smith_river_homing.state_ids.json
```

Its export record stores this repository's revision, the interpreter and
`h3` versions, and the source and artifact digests.

## Why the specification is shaped this way

- **Exact probabilities.** Every probability is a power-of-two fraction, so a
  decimal literal such as `0.140625` is exactly 9/64 in binary64. The GNN
  source, the artifact and the Lean tables read identical numbers.
- **Log-preferences.** `C` is bound to `LogPreferenceVector`, matching this
  repository's interchange contract and pymdp. FEP_Formal's normalized-`C`
  denotation therefore does not apply; its Cascadia instance carries
  `softmax C` explicitly and proves the exact relation between risk plus
  ambiguity and pymdp's score.
- **State identity outside the matrices.** H3 cells are passed with
  `--state-ids` in matrix order; the source annotation documents that order
  but the exporter never infers cells from prose.

## Assumption studies and proof custody

GEO-INFER's `examples/cascadia/cascadia_study.py` exports a separate exact
source and artifact for each of `reference`, `uniform_habit` and
`uninformative_odor`, then replays a declared process-seed grid. Run it in the
GEO-INFER environment with this checkout's interpreter supplied explicitly.
Install the two environments with their [separate required uv versions](../../development/geo_infer_2026_09.md#independent-uv-toolchains)
first, then invoke the installed GEO interpreter directly:

```bash
cd ../GEO_INFER
.venv/bin/python GEO-INFER-ACT/examples/cascadia/cascadia_study.py \
  --gnn-repo ../Generalized_Notation_Notation \
  --gnn-python ../Generalized_Notation_Notation/.venv/bin/python \
  --output GEO-INFER-ACT/examples/outputs/cascadia_seed_study_new
```

Every condition keeps the source, H3 order, domain, emitted artifact and
exporter's revision. The study checks matrix equality against the exact
compiler and never refreshes this repository's reference golden fixtures.
Seed outcomes retain null arrivals for runs that have not reached home by
the last recorded state. Study qualification concerns runtime replay and
numerical invariants; it does not confer Lean verification.

Changing E preserves A/B/D but changes policy priors. Removing odor
information changes A and may change the exact first update; the reference
57/128 evidence and 18/19 posterior must not be copied into that condition's
claims. Export success establishes the interchange artifact, not those
theorems. A new fep_formal instance and source-bound native receipt are
required for variant proof claims. The canonical reference artifact and its
existing proof remain a separate edition.
