# GNN Documentation Hub - Agent Scaffolding

## Overview

**Purpose**: This directory serves as the centralized documentation hub and specification authority for the Generalized Notation Notation (GNN) framework.

**Pipeline Step**: N/A (Static Reference Material)

**Category**: Documentation / Specification

**Status**: ✅ Production Ready

**Documentation hub version**: v2.0.0 (aligns with [README.md](README.md) front matter; **GNN language** syntax is v1.1 per [gnn_syntax.md](gnn_syntax.md); **Python package** version is `4.1.0` in [pyproject.toml](../../pyproject.toml)).

---

## Core Functionality

### Primary Responsibilities

1. **Syntax Authority**: Maintains the definitive `.md` parser specifications required to author Active Inference matrices correctly. [`gnn_syntax.md`](gnn_syntax.md) is the normative specification; `reference/` and `language/` expand it and never override it.
2. **Architecture Mapping**: Defines the structural rules connecting `language/`, `reference/`, `implementations/`, and `integration/`.
3. **End-User Guidance**: Houses tutorials, external tool integrations, and operational guides.

### Load-bearing invariant for generation tasks

An agent authoring a GNN file must know that the keys placed in
`## InitialParameterization` determine the **model kind** the renderers
dispatch on — discrete `A`/`B`/`C`/`D`/`E` (FLAT / FACTORED),
linear-Gaussian `F`/`H`/`Q`/`R` plus `prior_mean`/`prior_cov` (CONTINUOUS),
`dirichlet_[A-E]` pseudo-counts (LEARNING), `[ABCDE]_level<N>`
(HIERARCHICAL), `[ABCDE]_agent<N>` (MULTI_AGENT). Detection is **structural**:
prose in a `ModelName` or annotation is never scanned and cannot change how a
model renders. Rules and precedence live in
[`gnn_syntax.md` § Parameterization families](gnn_syntax.md#parameterization-families),
enforced by `detect_model_kind` in
[`src/gnn/render/pomdp_contract.py`](../../src/gnn/render/pomdp_contract.py).

### Subsystem Indices Available

- `docs/gnn/advanced/`: Ontology, multi-agent, LLM/neurosymbolic topics, and advanced modeling patterns.
- `docs/gnn/implementations/`: Specific platform references (PyMDP, RxInfer, etc.).
- `docs/gnn/language/`: DSL specific syntax rules.
- `docs/gnn/modules/`: Component behaviors (including structural documentation for all 25 modules).
- `docs/gnn/operations/`: Internal pipeline processing guides, troubleshooting, and coherence checks.
- `docs/gnn/reference/`: Raw variable mappings and type systems.
- `docs/gnn/testing/`: Testing standards for the notation itself.
- `docs/gnn/tutorials/`: End-user construction guides.

## Implementation Details

These documents are consumed by human end-users and parsed globally by the pipeline's LLM context protocol (via Model Context Protocol) to ensure generative models created via text prompts are hallucination-free and syntactically correct against GNN v1.1.
