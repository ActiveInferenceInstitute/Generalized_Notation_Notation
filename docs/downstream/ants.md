# ants

**Repository**: [`ActiveInferenceInstitute/ants`](https://github.com/ActiveInferenceInstitute/ants)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls; job evidence re-read 2026-11-06 and 2026-11-11.

## What it models

An MDP-based active inference simulation of ant-colony foraging in a T-maze,
the code for *Active Inferants* (Friedman, Tschantz, Ramstead, Friston and
Constant, 2021, Frontiers in Behavioral Neuroscience). The GNN specification
restates the foraging generative model in the shared notation.

## GNN specification

- `gnn/ant_colony_foraging.gnn.md` — the declared foraging model.

## Pinned GNN revision

`gnn/upstream.json` records tag `v5.1.0` and commit
`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`.

## Conformance and validation evidence

- `tests/test_gnn_conformance.py` checks the declaration against the pinned
  release; `gnn_conformance.py` and `scripts/validate_gnn_spec.py` implement it.
- `.github/workflows/ci.yml` job key `gnn-conformance` publishes
  `gnn conformance (pinned)` (re-read 2026-11-06): it validates
  `gnn/ant_colony_foraging.gnn.md` with the parser revision pinned in
  `gnn/upstream.json`, and `tests/test_gnn_conformance.py` fails rather than
  skips when the pin is unavailable. The "Test (pytest + coverage)" step
  remains the general suite.

## Verification limits

Not verified: that the current `main` run is green. The pin file, specification
and workflow definition were read at `main` on 2026-11-02.
