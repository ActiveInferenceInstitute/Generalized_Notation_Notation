# Biofirm

**Repository**: [`ActiveInferenceInstitute/Biofirm`](https://github.com/ActiveInferenceInstitute/Biofirm)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls.

## What it models

An installable Python package for reproducible ecological control, bioregion
research and auditable decision analysis, with a pymdp active-inference
controller for simulating and comparing ecosystem trajectories. The pin file
declares the claim boundary `illustrative-ecological-control`.

## GNN specification

- `models/biofirm_ecosystem_variable.gnn.md` — listed in the pin file's
  `spec_files`.

## Pinned GNN revision

`.github/gnn-pin.json` (schema `biofirm-gnn-pin-v1`) records tag `v5.0.3`,
revision `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, version 5.0.3,
`pinned_on` 2026-11-01, the spec directory and file, and the claim boundary.

## Conformance and validation evidence

- `tests/test_gnn.py` exercises the spec surface; `src/biofirm/gnn.py`
  implements it.
- `.github/workflows/ci.yml` ("biofirm-ci") runs the test suite. There is no
  GNN-specific job name on `main` (verified 2026-11-02), so the evidence is the
  test file run by the generic suite rather than a dedicated job.

## Verification limits

No GNN-specific workflow job; not verified that the current `main` run is green.
