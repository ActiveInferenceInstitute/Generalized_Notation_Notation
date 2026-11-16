# active_fractal_rabbit

**Repository**: [`ActiveInferenceInstitute/active_fractal_rabbit`](https://github.com/ActiveInferenceInstitute/active_fractal_rabbit)
· **Default branch**: `main` · **Verified**: pin, pointer and workflow read on 2026-11-15 at [5ae8174](https://github.com/ActiveInferenceInstitute/active_fractal_rabbit/tree/5ae81747573f3291bde8c02d7cccc323606c0128).

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

`.github/gnn-pin.json` (schema `gnn-importer-pin/v1`) records tag
`v5.1.0`, revision `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`, version 5.1.0,
`pinned_on` 2026-11-12, and the claim boundary `synthetic-only`.

## Conformance and validation evidence

- `.github/workflows/ci.yml` job key `gnn-conformance` invokes `./run.sh gnn`.
  The arm validates the pin, pointer and the two specifications with
  `python -m gnn.importers check`, then requires a nonempty, zero-skip run of
  `tests/test_gnn_spec.py` against the pinned parser. The project documents
  the separately published `gnn conformance (local)` status as required on `main`.
- `src/active_fractal_rabbit/gnn_spec.py` implements the spec surface.

## Verification limits

The source declares both checks. Current CI results and branch protection
require separate verification.
