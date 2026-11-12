# policy_entanglement

**Repository**: [`ActiveInferenceInstitute/policy_entanglement`](https://github.com/ActiveInferenceInstitute/policy_entanglement)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls; job evidence re-read 2026-11-06 and 2026-11-11.

## What it models

A tunable mean-field deformation framework for multi-stream policy ensembles in
active inference. A single scalar coupling parameter interpolates between strict
mean-field factorization and arbitrary structured joint policy distributions,
with a closed-form entanglement decomposition of variational free energy, an
information-geometric and spectral treatment, and a machine-checked Lean 4
boundary alongside pymdp POMDP simulations.

## GNN specification

- `gnn/bernoulli_toy.gnn.md` — the base cross-stream coupling example.
- `gnn/k_stream_ensemble.gnn.md` — K=3 chain coupling, showing that the
  stock-GNN joint-variable encoding scales to K > 2.
- `gnn/production_ensemble.gnn.md`.
- Extension discussion: `manuscript/S08_gnn_generalized_notation_extension.md`;
  generated Lean: `gnn/generated/BernoulliToyGnn.lean`.

## Pinned GNN revision

`gnn/upstream.json` pins tag `v5.0.3`, commit
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, and names the parser module
`gnn.parsers.markdown_parser` and class `MarkdownGNNParser`.

## Conformance and validation evidence

- Tests: `tests/test_gnn_validation.py`, `test_gnn_parser.py`,
  `test_gnn_round_trip.py`, `test_gnn_upstream_conformance.py`,
  `test_gnn_harness_conformance.py`, `test_gnn_concordance.py`.
- Scripts: `scripts/gnn_to_pymdp.py`, `scripts/simulate_gnn.py`.
- `.github/workflows/ci.yml` job `gnn-conformance` publishes
  `gnn conformance (pinned)`: it reads the pin from `gnn/upstream.json` and
  validates against the pinned upstream parser (re-read 2026-11-06). The
  2026-11-02 statement that no workflows existed is superseded.

## Verification limits

Not verified: that the test suite passes on current `main`, and that every
`.gnn.md` example round-trips through the pinned parser. The pin file and test
names were read at `main` on 2026-11-02.
