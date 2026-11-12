# ActiveInferenceImplementations

**Repository**: [`ActiveInferenceInstitute/ActiveInferenceImplementations`](https://github.com/ActiveInferenceInstitute/ActiveInferenceImplementations)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls; job evidence re-read 2026-11-06 and 2026-11-11.

## What it models

Educational Jupyter notebooks that explore probabilistic and active inference
with RxInfer.jl: Beta-Bernoulli coin-bias inference and variational message
passing. The GNN specification restates the mean-field variational model the
notebook derives.

## GNN specification

- `specs/normal_vmp.gnn.md` — the declared mean-field variational
  message-passing model.

## Pinned GNN revision

`specs/gnn-pin.json` records tag `v5.0.3` and commit
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`.

## Conformance and validation evidence

- `.github/workflows/ci.yml` job `gnn-conformance` (display name
  "gnn conformance (pinned)") provisions the pinned parser runtime and
  runs the step "Validate the GNN specification against the pinned parser",
  which calls `scripts/validate_gnn_spec.py` against `specs/normal_vmp.gnn.md`.
  The check name carries no version: the pinned revision is recorded in
  `specs/gnn-pin.json`.
- `scripts/gnn_parser_probe.py` probes the provisioned parser runtime.

## Verification limits

Not verified: that the current `main` run is green. The pin file, specification
and workflow definition were read at `main` on 2026-11-02; the check name and
the pin file path were re-read at `main` on 2026-11-05.
