# ActiveBlockference

**Repository**: [`ActiveInferenceInstitute/ActiveBlockference`](https://github.com/ActiveInferenceInstitute/ActiveBlockference)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls; job evidence re-read 2026-11-06 and 2026-11-11.

## What it models

A CPU-oriented active inference library for deterministic, discrete-state
radCAD and cadCAD grid simulations: strict YAML/TOML configuration, canonical
`(y, x)` dynamics, persisted diagnostics, rendered artefacts and fail-closed
validation.

## GNN specification

- `specs/gridworld_3x3.gnn.md` — the committed 3×3 gridworld model.
- `docs/gnn.md` documents the GNN surface.

## Pinned GNN revision

`specs/gnn-pin.json` records tag `v5.1.0`, commit
`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`, `language_version` "GNN v1.1",
`recorded` 2026-11-12, and SHA-256 digests of five pinned parser files.

## Conformance and validation evidence

- `.github/workflows/ci.yml` job `gnn-conformance` validates the committed GNN
  specification with the parser pinned in `specs/gnn-pin.json`, installed into a
  separate runtime.
- `scripts/validate_gnn_spec.py`, `scripts/gnn_parser_probe.py`,
  `scripts/export_gnn_spec.py`, `tests/test_gnn_spec.py`, and
  `blockference/gnn.py`.

## Verification limits

Not verified: that the current `main` run is green. The pin file and workflow
definition were read at `main` on 2026-11-02.
