# active_inference_power

**Repository**: [`ActiveInferenceInstitute/active_inference_power`](https://github.com/ActiveInferenceInstitute/active_inference_power)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls; job evidence re-read 2026-11-06 and 2026-11-11.

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

`models/gnn-pin.json` records tag `v5.1.0` and commit
`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`.

## Conformance and validation evidence

- `tests/test_gnn_specs.py` validates the declarations against the pinned
  release.
- `.github/workflows/ci.yml` job key `gnn-conformance` publishes
  `gnn conformance (pinned)` (re-read 2026-11-06): it reads
  `models/gnn-pin.json`, fetches the pin, and runs
  `python scripts/validate_gnn_specs.py` and `pytest tests/test_gnn_specs.py`
  through `uv` with the development extra.

## Verification limits

Not verified: that the current `main` run is green. The pin file, specifications
and workflow definition were read at `main` on 2026-11-02.
