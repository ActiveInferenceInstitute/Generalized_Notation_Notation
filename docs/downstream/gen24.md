# GEN24

**Repository**: [`ActiveInferenceInstitute/GEN24`](https://github.com/ActiveInferenceInstitute/GEN24)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls; job evidence re-read 2026-11-06 and 2026-11-11.

## What it models

An exploratory institute repository for generating and transforming synthetic
domain material, introduction letters, translations and audience-specific calls
to action, plus the experimental NHIM cognitive/ML module. GNN enters through
NHIM: the pin notes state the upstream parser is needed only for validation and
that GEN24 does not import GNN at runtime.

## GNN specification

- `NHIM/nhim_inquiry_agent.gnn.md` — the declared inquiry-agent model.
- `NHIM/nhim_model.py` is the paired implementation; `docs/gnn.md` documents the
  integration.

## Pinned GNN revision

`NHIM/gnn-pin.json` records release `v5.1.0`, revision
`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`, `recorded` 2026-11-24, the parser entrypoint,
the validator path and parser file hashes, plus an explicit re-pin procedure.

## Conformance and validation evidence

- `.github/workflows/ci.yml` job key `gnn-conformance`, display
  `gnn conformance (pinned)` (re-read 2026-11-06), reads the commit from
  `NHIM/gnn-pin.json`, checks the tag peels to it, and validates the
  specification against the pinned parser. The `tests` job remains for the
  general suite, where the parser-level GNN test skips.
- `scripts/validate_gnn_spec.py` and `tests/test_gnn_spec.py`.

## Verification limits

CI emits a warning and skips pinned-parser validation when the GNN checkout is
unavailable. Not verified: that the current `main` run is green.
