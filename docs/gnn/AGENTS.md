# GNN Documentation Hub - Agent Scaffolding

## Overview

**Purpose**: This directory serves as the centralized documentation hub and specification authority for the Generalized Notation Notation (GNN) framework.

**Pipeline Step**: N/A (Static Reference Material)

**Category**: Documentation / Specification


**Documentation hub version**: v2.0.0 (aligns with [README.md](README.md) front matter; **GNN language** syntax is v1.1 per [gnn_syntax.md](gnn_syntax.md); the **Python package** version is canonical in [pyproject.toml](../../pyproject.toml)).

---

## Core Functionality

### Primary Responsibilities

1. **Syntax Authority**: Maintains the definitive `.md` parser specifications required to author Active Inference matrices correctly. [`gnn_syntax.md`](gnn_syntax.md) is the normative specification; `reference/` and `language/` expand it and never override it.
2. **Architecture Mapping**: Defines the structural rules connecting `language/`, `reference/`, `implementations/`, and `integration/`.
3. **End-User Guidance**: Houses tutorials, external tool integrations, and operational guides.

### Load-bearing invariant for generation tasks

Classification reads typed fields, including the section identifier,
parameterization keys, and model settings. A model can declare multiple
families. Keep the full set from `detect_model_kinds` when documenting
backend admission. `detect_model_kind` provides the precedence winner.
See [normative model-kind rules](gnn_syntax.md#model-kind-detection) and
[backend admission](implementations/README.md#backend-admission).

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
