# CEREBRUM

**Repository**: [`ActiveInferenceInstitute/CEREBRUM`](https://github.com/ActiveInferenceInstitute/CEREBRUM)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls; job evidence re-read 2026-11-06 and 2026-11-11.

## What it models

CEREBRUM: Case-Enabled Reasoning Engine with Bayesian Representations for
Unified Modeling. The GNN model is a case-conditioned active inference
specification; the pin file declares the claim boundary
`framework-structure-only`.

## GNN specification

- `models/case_conditioned_active_inference.gnn.md` — listed in the pin file's
  `spec_files`.

## Pinned GNN revision

`.github/gnn-pin.json` (schema `gnn-importer-pin/v1`) records tag `v5.1.0`,
revision `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`, version 5.1.0,
`recorded` 2026-11-12, the spec directory and file, and the claim boundary.

## Conformance and validation evidence

- `tests/core/test_gnn_spec.py` exercises the spec surface;
  `src/core/gnn.py` implements it.
- `.github/workflows/ci.yml` job key `gnn-conformance` publishes
  `gnn conformance (pinned)` (re-read 2026-11-06): it validates
  `models/case_conditioned_active_inference.gnn.md` with the parser revision
  pinned in `.github/gnn-pin.json`, and `tests/core/test_gnn_spec.py` fails
  rather than skips when the explicitly selected pin is unavailable.

## Verification limits

Not verified that the current `main` run of `gnn-conformance` is green.
