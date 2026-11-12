# Active_Cardference

**Repository**: [`ActiveInferenceInstitute/Active_Cardference`](https://github.com/ActiveInferenceInstitute/Active_Cardference)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls; job evidence re-read 2026-11-06 and 2026-11-11.

## What it models

A deck-generic card-game engine with GNN-specified active inference agents,
classical Monte Carlo validation and fair CPU-parallel benchmarking. The
repository is the project home; `gnn-cards` is the installable distribution and
`gnn_cards` the Python import path.

## GNN specification

- `specs/kuhn_aif_p1.gnn.md` — the active inference specification for the
  Kuhn-poker one-player surface.

## Pinned GNN revision

`specs/gnn-pin.json` records tag `v5.0.3`, commit
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, `language_version` "GNN v1.1",
`recorded` 2026-11-02, and SHA-256 digests of five pinned parser files
(`src/gnn/__init__.py`, `src/gnn/parsers/__init__.py`,
`src/gnn/parsers/common.py`, `src/gnn/parsers/markdown_parser.py`,
`src/gnn/parsers/markdown_parser_parameter.py`).

## Conformance and validation evidence

- `.github/workflows/ci.yml` job `test (py3.12)` provisions the pinned GNN parser
  runtime into `$RUNNER_TEMP/gnn-runtime` and runs
  `scripts/validate_gnn_spec.py` against that runtime with JSON output in the
  step "Validate the GNN specification against the pinned parser".
- `scripts/gnn_parser_probe.py` is the local probe.
- The README describes the validator as dependency-free with exit codes 0/1/2 and
  a conformance test in CI.

## Verification limits

Not verified: that the current `main` run is green. The README states M0 (the
L0–L2 core engine) is still in progress; the GNN integration spine is what is
present.
