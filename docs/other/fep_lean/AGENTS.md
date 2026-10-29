# FEP_Formal (retained docs/other/fep_lean path)

## Current identity

The folder and linked filenames retain their published `fep_lean` names.
Current companion imports use `fep_formal`, commands use `fep-formal`, and
`FEP_FORMAL_ROOT` selects the checkout. Older public names are deprecated
aliases throughout FEP 2.x. The production resolver validates the exact project,
script and physical CLI contract; its provisioned environment/tool cache runs
offline. Historical evidence below retains its original Source pairs and dates;
current qualification requires the named current pair and fresh artifact/native
checks. The [paired-revision procedure](../../development/fep_lean_paired_revision.md)
owns current integration and closeout ordering.

## Overview

**Status**: active documentation program (draft) | **Version**: 0.2 | **Last
Updated**: 2026-09-04

## Purpose

Document, for GNN-side readers and agents, the formal collaboration program
with the sibling Lean 4 repository `fep_lean`: how generative models that
have Lean expressions can be rendered and executed by the GNN pipeline, and
how GNN's steps and methods are being formalized in Lean. This folder is the
GNN-side mirror of the fep_lean design program at
`../fep_formal/docs/design/gnn-bridge/` (cross-repo references are inline code
paths, never markdown links).

On the fep_lean side, W2 source custody and Q5–Q7 concrete artifact proofs
have current schema-2 native receipts. H2.7 acceptance opens only read-only
H3.G0 eligibility. Current evidence and exact boundaries live in
`../fep_formal/specs/gnn-bridge-w2-source-custody/WAVE2-REPORT.md`.
The mirrored bridge contract is version 0.6.

## Contents

| File | Role |
| --- | --- |
| [README.md](README.md) | Reader-facing entry point and navigation |
| [fep_lean.md](fep_lean.md) | Catalogue overview: formalized content, evidence planes, commands |
| [fep_lean_gnn.md](fep_lean_gnn.md) | The collaboration program: pipeline integration and the Lean formalization program |
| [bridge-contract.md](bridge-contract.md) | Mirror of the canonical cross-repo contract |
| [SPEC.md](SPEC.md) | Scope specification |

## Quick navigation

- [Main documentation](../../README.md)
- [docs/other archive](../README.md) and its [AGENTS.md](../AGENTS.md)
- [GNN syntax](../../gnn/gnn_syntax.md)
- [Active Inference theory folder](../../active_inference/README.md)

## Key concepts

- **Model-kind alignment.** GNN's discrete POMDP family (`A/B/C/D[/E]`, `F[1]`
  readout) corresponds to fep_lean's finite carrier family
  (`active_inference.lean` `GenerativeModel`, `FiniteLaw`/`FiniteKernel`,
  `FiniteHMM`); GNN's continuous linear-Gaussian family (`F/H/Q/R`) corresponds
  to fep_lean's Gaussian/OU semigroup family.
- **Evidence planes stay distinct.** A pipeline run establishes that a
  document parses, renders, and executes; a Lean compilation establishes that
  a named body compiles. Neither substitutes for the other.
- **Provenance rule.** Bridge-emitted documents carry source repo, commit
  digest, Lean definition, and generator identity in their provenance section.
- **`unsupported` vs `failed`.** Continuous models on categorical-only render
  backends report `unsupported` (excluded from execution) — a distinct status
  from `failed`; see `src/gnn/README.md`.

## Integration with pipeline

A fep_lean-emitted document flows through the standard pipeline: step 3
(parse), step 5 (type check), step 10 (ontology validation against
`src/gnn/ontology/act_inf_ontology_terms.json`), step 11 (render, nine targets),
step 12 (execute, eight targets). Naming convention for emitted documents:
`GNNSection` identifier prefixed `FepLean`, with the `continuous` keyword for
the continuous family so kind detection stays mechanical. Details:
[fep_lean_gnn.md](fep_lean_gnn.md).

## Editing rules

- The bridge contract in this folder is a **mirror**. Edit the canonical copy
  first (`../fep_formal/docs/design/gnn-bridge/bridge-contract.md`), then update
  this mirror in the same working session, keeping the bodies identical.
- Keep all links relative and verify targets exist; the repository-wide
  documentation audit (`uv run --extra dev python
  docs/development/docs_audit.py --strict --check-anchors --no-write`) checks
  them for every markdown file.
- Follow [style_guide.md](../../style_guide.md): `uv run python` command
  spellings, H1 title, metadata block, honest claims without embedded
  ungenerated counts.
- Do not add pipeline behavior claims for fep_lean integration that the
  pipeline does not yet have; this folder describes a scoped research
  program, and its phases are tracked in [fep_lean_gnn.md](fep_lean_gnn.md).

## Related resources

- [fep_lean.md](fep_lean.md) and [fep_lean_gnn.md](fep_lean_gnn.md)
- [Expected free energy](../../active_inference/expected_free_energy.md) and
  [generative models](../../active_inference/generative_models.md)
- [Framework integration guide](../../gnn/integration/framework_integration_guide.md)
- [Development guide](../../development/README.md)

---

**Status**: active | **Maintenance**: update together with the canonical
bridge program on the fep_lean side | **Last Updated**: 2026-09-04
