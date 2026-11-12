# cognitive

**Repository**: [`ActiveInferenceInstitute/cognitive`](https://github.com/ActiveInferenceInstitute/cognitive)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls; job evidence re-read 2026-11-06 and 2026-11-11.

## What it models

A tested Python package for discrete and continuous active inference agents
(generative models, expected free energy, POMDP simulation), paired with an
Obsidian vault of roughly 2,100 linked pages explaining the mathematics,
biology, cognitive science and philosophy behind it. The package is the
executable reference; the vault is the explanation.

## GNN specification

- `specs/discrete_variational.gnn.md` — the declared discrete model.
- `docs/gnn.md` documents the GNN surface.

## Pinned GNN revision

`specs/gnn-pin.json` records tag `v5.0.3`, commit
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, `language_version` "GNN v1.1",
`recorded` 2026-11-01, and SHA-256 digests of five pinned parser files.

## Conformance and validation evidence

- `.github/workflows/quality.yml` job `gnn-conformance` provisions the pinned GNN
  parser runtime and validates the declaration with the pinned parser.
- `code/tests/test_gnn_conformance.py`, `code/tests/gnn_parser_probe.py`,
  `code/scripts/validate_gnn_spec.py`, and
  `code/tools/src/models/active_inference/gnn_spec.py`.

## Verification limits

Not verified: that the current `main` run is green. The pin file and workflow
definition were read at `main` on 2026-11-02.
