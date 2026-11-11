# pipeline (documentation)

## Overview

Canonical documentation for running and configuring the GNN pipeline (steps 0–24), hosted under `docs/pipeline/`.

**Status**: Documentation  
**Version**: 1.0  

---

## Purpose

- Execution setup: JAX, NumPyro and DisCoPy are core `uv sync` dependencies; PyTorch and bnlearn are manual installs; Stan needs the `stan` extra and a CmdStan toolchain; Julia backends remain environment-dependent
- Pointers to step scripts, orchestrator, and configuration

This tree is part of the GNN documentation system; implementation lives in `src/`.

## Contents

| File | Role |
|------|------|
| [README.md](README.md) | Human entrypoint: `uv`, frameworks, links to `STEP_INDEX` / `main.py` |
| [v3_orchestration.md](v3_orchestration.md) | v3.0.0 long-running orchestration contracts |
| [pipeline_stage_hardening_review.md](pipeline_stage_hardening_review.md) | Maintained stage-by-stage review goal and GridWorld proof contract |
| [validation_evidence_guide.md](validation_evidence_guide.md) | Commands and artifacts that certify examples, templates, docs, health, and proof paths |

## Quick navigation

- **Step index**: [src/gnn/STEP_INDEX.md](../../src/gnn/STEP_INDEX.md)
- **Stage hardening review**: [pipeline_stage_hardening_review.md](pipeline_stage_hardening_review.md)
- **Orchestrator**: [src/gnn/main.py](../../src/gnn/main.py)
- **Script catalog**: [PIPELINE_SCRIPTS.md](../PIPELINE_SCRIPTS.md)
- **Pipeline module**: [src/gnn/pipeline/AGENTS.md](../../src/gnn/pipeline/AGENTS.md)

## Integration

- **25 steps (0–24)**: thin orchestrators in `src/gnn/N_*.py`
- **Routing**: `input/config.yaml` (`testing_matrix`)

## Standards

Same conventions as other `docs/*` manifests: concrete paths, no duplicate claims about code that lives elsewhere.

---

**Maintenance**: Update when step count, dependency layout, or config keys change.
