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

- `.github/workflows/ci.yml` runs `./run.sh gnn` in the `GNN conformance`
  step of its `gate` job. That arm calls `scripts/validate_gnn_spec.py`
  against `specs/normal_vmp.gnn.md` using the pinned parser and checks the
  parsed model against the Julia reader. [Workflow at 6cb731e](https://github.com/ActiveInferenceInstitute/ActiveInferenceImplementations/blob/6cb731e74aa2d6addf942e5348649e1771bc2f2d/.github/workflows/ci.yml)
- `scripts/gnn_parser_probe.py` probes the provisioned parser runtime.

## Verification limits

The pin file and specification were read at `main` on 2026-11-02. The workflow
was re-read at the linked revision on 2026-11-15. Current CI results require
separate verification.
