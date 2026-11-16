# COGANT

**Repository**: [`ActiveInferenceInstitute/COGANT`](https://github.com/ActiveInferenceInstitute/COGANT)
· **Default branch**: `main` · **Verified**: pin, lock, pointer and workflow read on 2026-11-15 at [5a3833e](https://github.com/ActiveInferenceInstitute/COGANT/tree/5a3833e08d916c9a5040f754fdf4de4f4c444331).

## What it models

COGANT deterministically translates Python, JavaScript and TypeScript codebases
into active inference GNN state-space models, with a reverse synthesizer that
reconstructs runnable code from a GNN bundle. GNN is both the output format and
the validation target.

## GNN specification

- Format documentation: `cogant/docs/theory/gnn_format.md`,
  `cogant/docs/theory/gnn_format_reference.md`, and the concept page
  `cogant/docs/concepts/gnn.md`.
- Emitted models: `cogant/examples/zoo/04_pomdp_minimal/model.gnn.md`,
  `06_hierarchical/model.gnn.md`, `12_full_pomdp/model.gnn.md`.
- Bridge implementation: `cogant/py/cogant/gnn/upstream_bridge/`.

## Pinned GNN revision

`.github/gnn-pin.json` declares `v5.1.0` at
`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`. The optional `upstream`
dependency in `cogant/pyproject.toml` and the resolved GNN package in
`cogant/uv.lock` name that revision. The historical compatibility receipt
`cogant/docs/evaluation/gnn_v5_0_3_compatibility.json` records its 20 passing
upstream steps against `v5.0.3` at `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`.

## Conformance and validation evidence

- `.github/workflows/ci.yml` job key `gnn-conformance` runs the `gnn` arm of
  `run_gate.py`. The arm uses the shared `gnn.importers` reader, then runs the
  `cogant/scripts/gnn_conformance_probe.py` model check and selected upstream
  Steps 3, 5 and 6. The probe binds the installed parser to the pinned
  commit through its PEP 610 distribution metadata. The workflow also
  runs `cogant/tests/unit/test_gnn_upstream_bridge.py` and the bridge contract tests.
- `cogant/docs/cookbook/14_gnn_validation.md` is the validation walkthrough;
  `cogant/docs/evaluation/GNN_VALIDATION_REPORT.md` and
  `cogant/docs/evaluation/GNN_UPSTREAM_AUDIT_SURFACE.md` carry recorded results.

## Verification limits

The source read verifies the declarations and the check definition. Current
CI results and reproduction of the historical 5.0.3 receipt require separate
evidence.
