# fep_formal

**Repository**: [`ActiveInferenceInstitute/fep_formal`](https://github.com/ActiveInferenceInstitute/fep_formal)
· **Default branch**: `main` · **Source pair verified**: 2026-11-27.

## What it models

A standalone Lean 4 catalog of formally verified theorems spanning the Free
Energy Principle, active inference, Bayesian mechanics, information geometry and
thermodynamics. Its GNN surface is a bridge that gives Lean denotations to GNN
documents rather than a corpus of generative models.

## GNN specification

- Bridge contract: `docs/design/gnn-bridge/bridge-contract.md`; design notes
  `docs/design/gnn-bridge/direction-1-lean-to-gnn.md` and
  `direction-2-gnn-to-lean.md`.
- Lean modules: `lean/FepSketches/gnn_denotation.lean`,
  `gnn_denotation_continuous.lean`, `gnn_document.lean`,
  `gnn_render_statements.lean`.
- Bridge work packages: `specs/gnn-bridge-p1-finite-spike/`,
  `specs/gnn-bridge-p3-certificates/`, `specs/gnn-bridge-p4b-continuous-emission/`.

## Pinned GNN revision

The bridge source pin binds an exact GNN commit and its contract files by
content hash. Two pin files carry the bridge:

- `specs/gnn-bridge-w1-bridge-operations/syntax-pin.json` records SHA-256
  hashes for GNN's `docs/gnn/gnn_syntax.md` and
  `src/gnn/pipeline/step_registry.py`.
- `specs/gnn-bridge-w2-source-custody/source-pin.json` records GNN commit
  `c44896439ad8e2e2b50ef99f519182550098807e` with
  SHA-256 hashes for the six declared contract files (`pyproject.toml`,
  `uv.lock`, `src/gnn/main.py`, `docs/gnn/gnn_syntax.md`,
  `src/gnn/pipeline/step_registry.py`,
  `docs/other/fep_lean/bridge-contract.md`), and the `fep_lean` companion
  source commit `40e985910fe413f6355a75b706ec48ffeb29ea9f` with its own file
  hashes.

The [reciprocal GNN pin](../../.github/fep-lean-pair.json) names FEP's merged
qualification revision, including its native proof receipt.

## Conformance and validation evidence

- Automatic, on every push and pull request: the `.github/workflows/ci.yml`
  `python` job runs the step "Check the frozen-custody bridge gates", which
  validates committed bridge artifacts
  (`specs/gnn-bridge-q5-artifact-proof/generate_probe.py`,
  `specs/cascadia-geo-act-proof/generate_probe.py`,
  `specs/geo-infer-notation-bridge/check_geo_notation_bridge.py`, each in its
  read-only check mode), and its pytest selection includes
  `tests/test_gnn_bridge_operations.py`, which exercises the bridge's read-only
  checks and their refusal paths.
- A dedicated `.github/workflows/ci.yml` job key `gnn-conformance` publishes the
  `gnn conformance (pinned)` context on push and pull request when the
  workflow is enabled. The `python` job above checks bridge artifacts.
- `docs/pin_audit.py` (run in `ci.yml` as "Require the newest stable Lean-Mathlib
  release pair" and in the scheduled `pin-audit` job of `lean-latest.yml`)
  audits the Lean toolchain, Mathlib revision and model pins in project
  markdown. It does not audit the GNN syntax pin and is not bridge evidence.
- Run the live pair check `fep-formal bridge status`, pointed
  at a GNN checkout, compares the source pin against that checkout
  (`specs/gnn-bridge-w1-bridge-operations/bridge_status.py` is the compatibility
  entry point); the source pin is rewritten only by `fep-formal bridge pin`.
- Dispatch-only: `.github/workflows/source-pair-qualification.yml` job
  `qualify-source-pair` takes explicit `fep_revision` and `gnn_revision` inputs
  (40-character lowercase hex) and validates them when the workflow runs.
- [FEP PR #276](https://github.com/ActiveInferenceInstitute/fep_formal/pull/276)
  records local source-binding, finite and continuous emit checks, the Q7
  render, portability checks on four Python runtimes, and fresh Lean
  compilation of 12 Q7 theorem signatures. The independent receipt check
  passed on the frozen source pair.
- The two copies of the bridge contract are not byte-identical: the GNN mirror
  hashes `66ca4b50…` and the canonical `fep_formal` copy `8bfa57cb…`, differing
  only in the two table rows that name the canonical and the mirror copy. The
  bridge's own mirror check (`fep_formal.bridge.operations`) strips those two
  rows before comparing.
- Recorded bridge runs exist under `specs/gnn-bridge-p1-finite-spike/`
  (`gnn_output`, `gnn_output_run1_annotated`, `gnn_output_run2_accepted`) with
  Step 5 type-checker output.
- On the GNN side, the paired custody procedure is
  [../development/fep_lean_paired_revision.md](../development/fep_lean_paired_revision.md)
  and the mirrored contract is [../other/fep_lean/bridge-contract.md](../other/fep_lean/bridge-contract.md).

## Verification limits

The current receipt records local qualification for the frozen pair. Use
`fep-formal bridge status` with the intended GNN checkout to verify the bound
source files. Workflow status and native results for other revisions require
their own evidence.
