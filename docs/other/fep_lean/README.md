# fep_lean

> **Document Metadata**
> **Type**: External project documentation (research archive) | **Audience**: Researchers, Developers | **Complexity**: Advanced
> **Cross-References**: [AGENTS.md](AGENTS.md) | [Main documentation](../../README.md)
> **Last Updated**: 2026-09-06 | **Version**: 0.6

## Overview

`fep_lean` is the Active Inference Institute's standalone catalogue of 155
Free Energy Principle / Active Inference / Bayesian Mechanics / Information
Geometry / Thermodynamics topics, each carrying a reviewed invariant, explicit
assumptions, and a Lean 4 theorem body compiled in a pinned workspace. The
sibling checkout lives at `../fep_lean` (paths in this folder that start with
`../fep_lean/` are written as inline code on purpose: each repository
validates its own links, so cross-repo references are never markdown links).

The two projects are complementary layers:

- **GNN** specifies, renders, and executes generative-model instances: a
  sectioned document syntax (see [normative syntax](../../gnn/gnn_syntax.md))
  and a 25-step pipeline whose canonical registry is
  `src/gnn/pipeline/step_registry.py`.
- **fep_lean** states and proves invariants about the same objects: finite
  laws and kernels, Markov blankets, variational free energy, expected free
  energy, policy selection, and Gaussian/OU dynamics.

This folder documents the formal collaboration program between the two, from
the GNN side: how fep_lean-expressed models can be rendered and executed by
this pipeline, and how GNN's own steps and methods can be formalized in Lean.

Current status (2026-09-06): W2 source custody and the Q5 PyMDP, Q6 Julia
embedded-input, and Q7 scalar-OU coefficient proofs have validated schema-2
native receipts in `../fep_lean/specs/`. The
[bridge contract mirror](bridge-contract.md) is version 0.6: the GNN-side
Python package is canonically `gnn` (src-layout `src/gnn/`), the GNN owner
roster binds the declared notation contract (`pyproject.toml`, `uv.lock`,
`src/gnn/main.py`, `docs/gnn/gnn_syntax.md`,
`src/gnn/pipeline/step_registry.py` and this mirror) instead of the whole
`src/gnn/**/*.py` implementation tree, and the new `verify-document`
bridge operation checks any emitted document against the `FEP.GnnDocument`
typed surface (syntax + `WellFormed`; conformance proofs stay out of scope).
H2.7 acceptance opens only read-only H3.G0 eligibility; actual study metadata
is still required. Current records and exact test scopes live in
`../fep_lean/specs/gnn-bridge-w2-source-custody/WAVE2-REPORT.md`.

## Documents

| Document | Description |
| --- | --- |
| [fep_lean.md](fep_lean.md) | What the catalogue formalizes, its evidence planes, and commands of record |
| [fep_lean_gnn.md](fep_lean_gnn.md) | The GNN-side collaboration program: Lean-derived models in the pipeline, and the Lean formalization of GNN steps and methods |
| [bridge-contract.md](bridge-contract.md) | Mirror of the cross-repo articulation contract (canonical copy lives in the fep_lean checkout) |
| [AGENTS.md](AGENTS.md) | Agent scaffolding and editing rules for this folder |
| [SPEC.md](SPEC.md) | Scope specification for this documentation module |

## Key integration points

| Pipeline surface | Relevance to fep_lean |
| --- | --- |
| Step 3 (parse) and step 5 (type check) | First acceptance gate for any fep_lean-emitted GNN document (`gnn validate --strict`) |
| Step 10 (ontology) | Validates variable bindings against the canonical vocabulary; new Lean-specific bindings need an explicit vocabulary decision |
| Step 11 (render) and step 12 (execute) | Where a Lean-expressed generative model becomes a rendered program and a run; the discrete and continuous model kinds align with fep_lean's finite and Gaussian carrier families |
| Exemplars | `input/gnn_files/discrete/actinf_pomdp_agent.md` and the `input/gnn_files/continuous/` family are the correspondence anchors |

## Quick navigation

- [fep_lean catalogue overview](fep_lean.md)
- [Collaboration program](fep_lean_gnn.md)
- [Bridge contract mirror](bridge-contract.md)
- [GNN syntax](../../gnn/gnn_syntax.md)
- [Active Inference foundations](../../active_inference/fep_foundations.md)

## Related documentation

- [Active Inference: generative models](../../active_inference/generative_models.md),
  [expected free energy](../../active_inference/expected_free_energy.md),
  [GNN integration](../../active_inference/gnn_integration.md)
- [Pipeline guide](../../pipeline/README.md) and
  [PyMDP documentation](../../pymdp/README.md)
- [docs/other archive](../README.md) — this folder's parent

The canonical bridge documentation lives in the sibling checkout at
`../fep_lean/docs/design/gnn-bridge/` (design program, canonical contract,
and both direction programs).
