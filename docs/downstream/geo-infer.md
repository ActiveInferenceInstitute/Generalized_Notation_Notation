# GEO_INFER

**Repository**: [`ActiveInferenceInstitute/GEO_INFER`](https://github.com/ActiveInferenceInstitute/GEO_INFER)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls.

## What it models

A 45-module geospatial active inference monorepo (H3 topology, spatial Bayesian
models, place and risk analysis) in one uv/Python workspace. GNN is the
model-notation and matrix-provenance side of a JSON interchange with
GEO-INFER-SPACE, GEO-INFER-TIME and GEO-INFER-ACT; the repositories exchange JSON
artifacts and run in separate Python environments.

## GNN specification

- Interchange contract: `GEO-INFER-ACT/docs/gnn_interchange.md` (categorical
  `gnn-geo-infer/1`) and `GEO-INFER-ACT/docs/gnn_factored_interchange.md`.
- Consumer boundary implementation:
  `GEO-INFER-ACT/src/geo_infer_act/core/gnn_contract.py` (`GNNArtifact`).
- Example model:
  `GEO-INFER-ACT/examples/cascadia/data/smith_river_homing.gnn.md`; JSON artifacts
  `GEO-INFER-PLACE/locations/california/models/drought.gnn.json`,
  `fire.gnn.json`, `seismic.gnn.json`.

## Pinned GNN revision

`.github/gnn-pair.json` records `ActiveInferenceInstitute/Generalized_Notation_Notation`
at revision `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`. The file has no tag
field, so the revision itself is the pin. `GEO-INFER-TEST/src/geo_infer_test/gnn_pin_freshness.py`
checks pin freshness.

## Conformance and validation evidence

- `.github/workflows/gnn-interchange.yml` ("Paired GNN interchange"), a reusable
  workflow also called from `ci.yml` for main, tag, dispatch and schedule runs.
- `GEO-INFER-TEST/validate_gnn_interchange.py` and
  `GEO-INFER-TEST/validate_gnn_pin_freshness.py`.
- Unit tests `GEO-INFER-ACT/tests/unit/test_gnn_contract.py`,
  `test_gnn_factored_contract.py`, `test_gnn_gaussian_contract.py`,
  `test_gnn_factored_batch.py`, `test_gnn_metadata_derivation.py`,
  `test_gnn_spec.py`; integration test `test_gnn_model_mismatch.py`;
  receipts test `GEO-INFER-TEST/tests/unit/test_gnn_interchange_receipts.py`.

## Verification limits

Not verified: that the workflow passes on current `main`, and that the consumer
boundary still matches the producer contract version. File paths and the pin
file were read at `main` on 2026-11-02.
