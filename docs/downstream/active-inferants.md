# Active_InferAnts

**Repository**: [`ActiveInferenceInstitute/Active_InferAnts`](https://github.com/ActiveInferenceInstitute/Active_InferAnts)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls; job evidence re-read 2026-11-06.

## What it models

A multi-language active inference framework that combines
ant-colony-optimization-inspired multi-agent simulation with variational message
passing, implemented across many programming languages for research and
benchmarking.

## GNN specification

- `gnn/gaussian_agent.gnn.md` — the declared Gaussian-agent model.

## Pinned GNN revision

`gnn/upstream.json` pins the upstream repository at tag `v5.0.3`, commit
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`.

## Conformance and validation evidence

- `tests/test_gnn_conformance.py`, `gnn_conformance.py`, and
  `scripts/validate_gnn_spec.py`.
- `.github/workflows/ci.yml` job key `gnn-conformance` publishes
  `gnn conformance (pinned)` (re-read 2026-11-06): it validates
  `gnn/gaussian_agent.gnn.md` with the parser revision pinned in
  `gnn/upstream.json`, and `tests/test_gnn_conformance.py` fails rather than
  skips when the pin is unavailable.

## Verification limits

Not verified that the current `main` run of `gnn-conformance` is green.
