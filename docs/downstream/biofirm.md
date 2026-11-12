# Biofirm

**Repository**: [`ActiveInferenceInstitute/Biofirm`](https://github.com/ActiveInferenceInstitute/Biofirm)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls; job evidence re-read 2026-11-06 and 2026-11-11.

## What it models

An installable Python package for reproducible ecological control, bioregion
research and auditable decision analysis, with a pymdp active-inference
controller for simulating and comparing ecosystem trajectories. The pin file
declares the claim boundary `illustrative-ecological-control`.

## GNN specification

- `models/biofirm_ecosystem_variable.gnn.md` — listed in the pin file's
  `spec_files`.

## Pinned GNN revision

`.github/gnn-pin.json` (schema `gnn-importer-pin/v1`) records tag `v5.1.0`,
revision `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`, version 5.1.0,
`recorded` 2026-11-12, the spec directory and file, and the claim boundary.

## Conformance and validation evidence

- `tests/test_gnn.py` exercises the spec surface; `src/biofirm/gnn.py`
  implements it.
- `.github/workflows/ci.yml` job key `gnn-conformance` publishes
  `gnn conformance (pinned)` (re-read 2026-11-06): it validates
  `models/biofirm_ecosystem_variable.gnn.md` with the parser revision pinned in
  `.github/gnn-pin.json`, and `tests/test_gnn.py` fails rather than skips when
  the explicitly selected pin is unavailable.

## Verification limits

Not verified that the current `main` run of `gnn-conformance` is green.
