# active_fractal_rabbit

**Repository**: [`ActiveInferenceInstitute/active_fractal_rabbit`](https://github.com/ActiveInferenceInstitute/active_fractal_rabbit)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls; job evidence re-read 2026-11-06 and 2026-11-11.

## What it models

A closed-loop research scaffold for active-inference analysis of synthetic
sporadic waypoint data. The external simulator boundary stays explicit:
FractalRabbit is pinned to its own commit, fetched and built into `.cache/`, and
not committed here; the inference layer runs a real pinned pymdp JAX agent
alongside a transparent reference filter. The pin file declares the claim
boundary `synthetic-only`.

## GNN specification

- `models/single_factor_reference.gnn.md`
- `models/factorial_latent_mode.gnn.md`

Both are listed in the pin file's `spec_files`.

## Pinned GNN revision

`.github/gnn-pin.json` (schema `active-fractal-rabbit-gnn-pin-v1`) records tag
`v5.0.3`, revision `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, version 5.0.3,
`pinned_on` 2026-11-01, and the claim boundary `synthetic-only`.

## Conformance and validation evidence

- `.github/workflows/ci.yml` job key `gnn-conformance`, display
  `gnn conformance (pinned)` (re-read 2026-11-06), reads the revision from
  `.github/gnn-pin.json`, fetches it, and validates the specifications with the
  pinned parser. The `tests` job remains the general suite and runs the
  documented gate, `pytest tests/` with the 90% coverage floor, where the
  checkout-dependent GNN checks skip with an explicit reason; the
  `gnn-conformance` job runs `pytest tests/test_gnn_spec.py -q` with
  `GNN_CHECKOUT` set.
- `src/active_fractal_rabbit/gnn_spec.py` implements the spec surface.

## Verification limits

The CI step emits a warning and skips pin re-verification when the pinned
checkout is unavailable. Not verified: that the current `main` run is green.
