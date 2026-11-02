# ants

**Repository**: [`ActiveInferenceInstitute/ants`](https://github.com/ActiveInferenceInstitute/ants)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls.

## What it models

An MDP-based active inference simulation of ant-colony foraging in a T-maze,
the code for *Active Inferants* (Friedman, Tschantz, Ramstead, Friston and
Constant, 2021, Frontiers in Behavioral Neuroscience). The GNN specification
restates the foraging generative model in the shared notation.

## GNN specification

- `gnn/ant_colony_foraging.gnn.md` — the declared foraging model.

## Pinned GNN revision

`gnn/upstream.json` records tag `v5.0.3` and commit
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`.

## Conformance and validation evidence

- `tests/test_gnn_conformance.py` checks the declaration against the pinned
  release; `gnn_conformance.py` and `scripts/validate_gnn_spec.py` implement it.
- `.github/workflows/ci.yml` has no GNN-specific job; the step
  "Test (pytest + coverage)" runs `uv run pytest`, which includes the
  conformance test.

## Verification limits

Not verified: that the current `main` run is green. The pin file, specification
and workflow definition were read at `main` on 2026-11-02.
