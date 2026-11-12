# on_policy_distillation

**Repository**: [`ActiveInferenceInstitute/on_policy_distillation`](https://github.com/ActiveInferenceInstitute/on_policy_distillation)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls; job evidence re-read 2026-11-06 and 2026-11-11.

## What it models

On-policy distillation formalized as active inference in finite variational
models: reverse-KL free energy, student-induced sampling, a two-agent pymdp
classroom, and machine-checked reproducible manuscript artifacts. Three GNN
specifications carry the small models the argument depends on.

## GNN specification

- `gnn/bernoulli_toy.gnn.md` — the Bernoulli toy model.
- `gnn/graph_world.gnn.md` — the graph-world environment.
- `gnn/si_tmaze.gnn.md` — the student-induced T-maze.

## Pinned GNN revision

`gnn/upstream.json` records tag `v5.0.3`, commit
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, package version `5.0.3`,
`date_recorded` 2026-11-01, and the pinned parser entry point
`gnn.parsers.markdown_parser.MarkdownGNNParser`.

## Conformance and validation evidence

- `tests/test_gnn.py` validates the specifications with the pinned parser;
  `tests/test_gnn_upstream_conformance.py` checks the recorded upstream revision.
- A local `src/gnn` package (parser, upstream, concordance and model modules)
  carries the bridge code the tests exercise.
- `.github/workflows/ci.yml` job key `release-gate` publishes
  `gnn conformance (pinned)` (re-read 2026-11-06): it provisions the revision
  pinned in `gnn/upstream.json` and runs the conformance check as a named job
  instead of a hidden step; both GNN test modules remain part of the suite.

## Verification limits

Not verified: that the current `main` run is green. The pin file, specifications
and workflow definition were read at `main` on 2026-11-02.
