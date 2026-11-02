# active_inference_power

**Repository**: [`ActiveInferenceInstitute/active_inference_power`](https://github.com/ActiveInferenceInstitute/active_inference_power)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls.

## What it models

A reproducible statistical-power simulation suite for adaptive studies with an
embedded active inference agent: multiple-testing corrections, sequential
e-processes, and action-loop operating characteristics, using real pymdp
inference. The GNN specifications carry the two agent models the simulations
run.

## GNN specification

- `models/reference_action_loop.gnn.md` — the reference action loop.
- `models/sequential_evidence_agent.gnn.md` — the sequential evidence agent.

## Pinned GNN revision

`models/gnn-pin.json` records tag `v5.0.3` and commit
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`.

## Conformance and validation evidence

- `tests/test_gnn_specs.py` validates the declarations against the pinned
  release.
- `.github/workflows/ci.yml` has no separate GNN job; its validation step runs
  `uv run --extra dev python scripts/validate_gnn_specs.py`.

## Verification limits

Not verified: that the current `main` run is green. The pin file, specifications
and workflow definition were read at `main` on 2026-11-02.
