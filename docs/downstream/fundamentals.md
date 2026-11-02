# fundamentals

**Repository**: [`ActiveInferenceInstitute/fundamentals`](https://github.com/ActiveInferenceInstitute/fundamentals)
· **Default branch**: `main` · **Verified**: 2026-11-02 from origin `main` with read-only GitHub API calls.

## What it models

Open-source Python code that follows along with Sanjeev V. Namjoshi's
*Fundamentals of Active Inference* (MIT Press, 2026): a tested library of the
book's algorithms plus thin orchestrator scripts that reproduce its figures and
numerical examples. The book itself is not open source.

## GNN specification

- `specs/weather_pomdp.gnn.md` — the worked POMDP example.
- `docs/gnn.md` and `docs/reference/gnn_spec.md` describe the GNN surface.

## Pinned GNN revision

`specs/gnn-pin.json` records tag `v5.0.3`, commit
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, `language_version` "GNN v1.1",
`recorded` 2026-11-01, and SHA-256 digests of five pinned parser files.

## Conformance and validation evidence

- `.github/workflows/ci.yml` job `gnn-conformance` ("gnn conformance (pinned
  v5.0.3)") provisions the pinned parser, digest-verifies the installed parser
  files before parsing, and re-runs `tests/test_gnn_spec.py` with that runtime.
- `scripts/validate_gnn_spec.py`, `scripts/gnn_parser_probe.py`, and
  `src/active_inference/gnn_spec.py` implement the spec surface.

## Verification limits

Not verified: that the current `main` run is green. The pin file and workflow
definition were read at `main` on 2026-11-02.
