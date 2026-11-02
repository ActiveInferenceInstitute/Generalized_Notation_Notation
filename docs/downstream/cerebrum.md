# CEREBRUM

**Repository**: [`ActiveInferenceInstitute/CEREBRUM`](https://github.com/ActiveInferenceInstitute/CEREBRUM)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls.

## What it models

CEREBRUM: Case-Enabled Reasoning Engine with Bayesian Representations for
Unified Modeling. The GNN model is a case-conditioned active inference
specification; the pin file declares the claim boundary
`framework-structure-only`.

## GNN specification

- `models/case_conditioned_active_inference.gnn.md` — listed in the pin file's
  `spec_files`.

## Pinned GNN revision

`.github/gnn-pin.json` (schema `cerebrum-gnn-pin-v1`) records tag `v5.0.3`,
revision `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, version 5.0.3,
`pinned_on` 2026-11-01, the spec directory and file, and the claim boundary.

## Conformance and validation evidence

- `tests/core/test_gnn_spec.py` exercises the spec surface;
  `src/core/gnn.py` implements it.
- `.github/workflows/ci.yml` runs `pytest` for the test suite. There is no
  GNN-specific job name on `main` (verified 2026-11-02), so the evidence is the
  test file run by the generic suite rather than a dedicated job.

## Verification limits

No GNN-specific workflow job; not verified that the current `main` run is green.
