# Active_Cardference

**Repository**: [`ActiveInferenceInstitute/Active_Cardference`](https://github.com/ActiveInferenceInstitute/Active_Cardference)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls; job evidence re-read 2026-11-06 and 2026-11-11; pin and workflow re-read 2026-11-16 at `5ead0056`.

## What it models

A deck-generic card-game engine with GNN-specified active inference agents,
classical Monte Carlo validation and fair CPU-parallel benchmarking. The
repository is the project home; `gnn-cards` is the installable distribution and
`gnn_cards` the Python import path.

## GNN specification

- `specs/kuhn_aif_p1.gnn.md` — the active inference specification for the
  Kuhn-poker one-player surface.

## Pinned GNN revision

`specs/gnn-pin.json` (schema `gnn-importer-pin/v1`) records tag `v5.1.0`,
commit `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`, `recorded` 2026-11-12
(adopted in Active_Cardference #73), and SHA-256 digests of five pinned parser files
(`src/gnn/__init__.py`, `src/gnn/parsers/__init__.py`,
`src/gnn/parsers/common.py`, `src/gnn/parsers/markdown_parser.py`,
`src/gnn/parsers/markdown_parser_parameter.py`).

## Conformance and validation evidence

- `.github/workflows/ci.yml` job `test` (`gnn conformance (pinned)`) ends with
  the step "GNN conformance against the pinned parser", which runs
  `./run.sh gnn`: it installs the commit recorded in `specs/gnn-pin.json` into a
  scratch runtime, runs the shared importer reader, runs the `gnn_runtime` tests
  with skips rejected, then validates every committed document with
  `scripts/validate_gnn_spec.py`. Receipts land in `docs/_build/gnn/`.
- `scripts/gnn_parser_probe.py` is the local probe.
- The README describes the validator as dependency-free with exit codes 0/1/2 and
  a conformance test in CI.

## Verification limits

Not verified: that the current `main` run is green. The README states M0 (the
L0–L2 core engine) is still in progress; the GNN integration spine is what is
present.
