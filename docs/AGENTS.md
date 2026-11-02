# docs/ — agent scaffolding

## Overview

Top-level documentation tree for GNN. **Human onboarding**: [README.md](README.md). **Flat link index**: [INDEX.md](INDEX.md). **Narrative tour**: [START_HERE.md](START_HERE.md). **Curated curricula**: [learning_paths.md](learning_paths.md). **Topic cross-links**: [CROSS_REFERENCE_INDEX.md](CROSS_REFERENCE_INDEX.md).

**Full directory inventory**: [INDEX.md](INDEX.md) (topic index) and [expected_dirs.txt](expected_dirs.txt) (canonical name list for scripts and audits).

## Contents

| Path | Role |
|------|------|
| [gnn/](gnn/) | GNN language, syntax, tutorials, pipeline module pages |
| [pipeline/](pipeline/) | Pipeline-oriented documentation |
| [development/](development/) | Dev workflows, `docs_audit.py`, link rewrites |
| [dev/](dev/) | `src/` doc inventory script and notes |
| [testing/](testing/) | Testing strategy guide (see `tests/` for layout) |
| [troubleshooting/](troubleshooting/) | Errors, FAQ, pipeline recovery |
| [api/](api/) | REST / API docs |
| [configuration/](configuration/) | Config reference |
| [templates/](templates/) | GNN templates |
| [downstream/](downstream/) | Repositories that apply GNN: specifications, pins and conformance evidence |
| [INDEX.md](INDEX.md), [START_HERE.md](START_HERE.md) | Doc navigation entry points |
| [SPEC.md](SPEC.md) | Versioning policy and doc-tree design requirements |
| Other top-level folders (`pymdp/`, `rxinfer/`, `bnlearn/`, …) | Framework and topic docs; see [INDEX.md](INDEX.md) |

**Last reviewed**: 2026-11-02

## Pipeline

Processing steps and modules live under `src/`; see [../src/gnn/AGENTS.md](../src/gnn/AGENTS.md).

## Related

- [../AGENTS.md](../AGENTS.md) — repository-wide module registry
- [../README.md](../README.md) — project overview
- [development/agents_readme_triple_review.md](development/agents_readme_triple_review.md) — three-pass AGENTS/README audit checklist
