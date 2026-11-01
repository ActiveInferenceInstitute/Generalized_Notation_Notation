# geo_infer

> **Document Metadata**
> **Type**: External project documentation (research archive) | **Audience**: Researchers, Developers | **Complexity**: Advanced
> **Cross-References**: [AGENTS.md](AGENTS.md) | [Main documentation](../../README.md)
> **Last Updated**: 2026-10-12 | **Version**: 0.1

## Overview

GEO_INFER is the Active Inference Institute's spatial active-inference
monorepo covering geography, topology, environmental sensing,
and cognitive modeling over spatially indexed data. The sibling checkouts
live at `../GEO_INFER` (paths in this folder that start with `../GEO_INFER/`
are written as inline code on purpose: each repository validates its own
links, so cross-repo references are never markdown links) (local clones may carry the earlier GEO-INFER dirname; git remotes are canonical). The integration
is **artifact-level only**: the two projects run in separate environments and
exchange versioned JSON artifacts with SHA-256 provenance; neither checkout
imports the other, and their contract forbids cross-checkout imports.

The relevant modules on the GEO_INFER side are:

- **SPACE** — H3 spatial topology and the state-id space artifacts bind to.
- **TIME** — observation scheduling for spatially distributed observations.
- **ACT** — the inference consumer; its `core.gnn_contract` module is the
  typed consumer boundary for GNN artifacts.
- **TEST** — the cross-repo validation harness that checks GNN-emitted
  artifacts against the contracts.

## Contracts

[California monitoring](california_monitoring.md) connects three explicit
factored sensor models with PLACE's real historical indicators, ACT's exact
local evaluation and constrained allocation, and SPACE's statewide H3 methods.
`qualify_factored.py` independently reproduces each exported artifact; its
read-only check rejects source, tensor, clock and provenance drift.


Three versioned artifact contracts connect the pipeline to GEO_INFER. The
normative format documentation is `src/gnn/export/geo_infer_contract.md`;
the factored variant has its own normative doc,
`src/gnn/export/geo_infer_factored_contract.md`.

| Contract | Producer (this repo) | Content |
| --- | --- | --- |
| `gnn-geo-infer/1` | `src/gnn/export/geo_infer.py` | Categorical models (strict GNN; optional H3 `--space-kind h3` support behind `--extra geo-infer`) |
| `gnn-geo-infer/2` | `src/gnn/export/geo_infer_gaussian.py` | Linear-Gaussian models with `F`/`G`/`H`/`Q`/`R` matrices |
| `gnn-geo-infer/factored/1` | `src/gnn/export/geo_infer_factored.py` | Factored state-space models |

On the consumer side, the typed boundary is
`geo_infer_act.core.gnn_contract`: `GNNArtifact` (the parsed artifact type)
and `run_gnn_inference` (the entry point that drives ACT inference over an
artifact).

## Exporting an artifact

From this checkout:

```
uv run python -m gnn.export.geo_infer input/gnn_files/pomdp_gridworld/pomdp_gridworld_3x3.md /tmp/gridworld.geo-infer.json --step-seconds 60
```

This is the post-reorganization module path (`python -m gnn.export.geo_infer`).
The `--space-kind h3` option requires the optional H3 dependency: install
with `uv sync --extra geo-infer` first.

## Worked example: Cascadia river homing

[Agent/process pairs](model_pairs.md) explain how two separately reproduced
GNN artifacts support actual model-mismatch experiments in GEO-INFER-ACT.

[Configured model matrices](model_matrices.md) explain registry custody,
independent qualification of each selected export and raw/canonical identities.

[Cascadia river homing](cascadia_river_homing.md) is a real H3 model: 26
resolution-8 cells traced from measured NHDPlus HR lower Smith River flow
lines, exact dyadic A–E, and a natal-stream homing task. Its GNN source and
golden artifact are fixtures of `tests/export/test_geo_infer_cascadia.py`,
which re-exports them byte for byte and rejects a transposed transition
literal.

## Validating interchange

The validation harness lives in the GEO-INFER-TEST module of the sibling
checkout. First install the separate environments using the
[paired-runtime procedure](../../development/geo_infer_2026_09.md#independent-uv-toolchains):
GNN requires uv 0.13.6; the GEO installation alone requires uv 0.13.1.
Then run the installed GEO interpreter from its checkout, pointing it at this repo:

```
cd ../GEO_INFER
.venv/bin/python GEO-INFER-TEST/validate_gnn_interchange.py \
  --gnn-repo ../Generalized_Notation_Notation \
  --gnn-python ../Generalized_Notation_Notation/.venv/bin/python
```

The harness re-exports known exemplars with this repo's interpreter and
checks the produced artifacts against the contract schemas and provenance
requirements.

## Relationship to FEP_Formal

FEP_Formal verifies the well-formedness of GNN **source documents** in Lean 4,
upstream of export (see [docs/other/fep_lean/](../fep_lean/README.md));
explicitly, `fep-lean bridge verify-document` consumes GNN Markdown via
`gnn.extract.pomdp_extractor` and cannot validate GEO_INFER's interchange
artifact JSON (a different format, validated by `GNNArtifact`); the Lean
gate applies to the notation before export. GEO_INFER consumes the **exported
artifacts** downstream with a Python contract validator. Separately, the
Cascadia artifact-proof slice in fep_formal reads frozen exported tables and
compiles a named finite Lean instance; its receipt proves only the exact
statements bound to those bytes. The generic interchange validator does not
run Lean, and successful export does not inherit that slice's receipt.

## Related documentation

- [Experiment qualification](experiment_qualification.md) — real producer
  reproduction for each retained Cascadia condition, with failure controls.

- [FEP_Formal bridge program](../fep_lean/README.md) — upstream document
  well-formedness verification
- [Framework integration guide](../../gnn/integration/framework_integration_guide.md)
- [docs/other archive](../README.md) — this folder's parent

The canonical interchange documentation lives in the sibling checkout at
`../GEO_INFER/GEO-INFER-ACT/docs/gnn_interchange.md`.
