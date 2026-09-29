# Developer Tooling Agents (scripts/)

## Purpose

This folder hosts the explicit, stateless developer workflow validation agents responsible for maintaining continuous documentation compliance, performance benchmarking, and pipeline orchestration across the GNN ecosystem.

## Components

### Audit & Compliance (13 scripts)

| Script | Purpose | Strict CI Gate |
|--------|---------|:--------------:|
| `check_repo_terminology.py` | Scans maintained source for banned terminology | ✅ `--strict` |
| `check_maintained_doc_terms.py` | Scans maintained Markdown for stale framework terms | ✅ `--strict` |
| `check_gnn_doc_patterns.py` | Scans `docs/` + `src/gnn/` for stale import paths, retired routing, and banned patterns | ✅ `--strict` |
| `check_doc_contracts.py` | Contract check for enforced quickstart sections, current CLI flag spellings, `input/config.yaml` naming, and the 9-render-target / 8-executor split | ✅ `--strict` |
| `check_external_links.py` | Scans maintained docs for dead external (http/https) URLs — informational, not CI-wired | ⚪ no (flaky external checks) |
| `check_mcp_skills_health.py` | Executes every registered MCP tool and verifies every SKILL.md documents a resolvable surface | ✅ `--strict` (local-gates.yml) |
| `check_capability_contracts.py` | Validates capability-contract claims against measured codebase state | ✅ exit 1 on mismatch |
| `check_manuscript_tokens.py` | Manuscript integrity gate: unknown `{{TOKEN}}`s, dangling `[@key]` citations, hard-coded counts, contradicted `input/...` path claims | ✅ hard gate (`--strict` also fails count warnings) |
| `check_hydrated_prose.py` | SC-22 PR-time custody drift guard: committed `output/manuscript/` must be what the committed token map hydrates to (a regenerated map without the full render ritual fails) | ✅ exit 1 on drift (local-gates.yml) |
| `check_pomdp_gridworld_outputs.py` | End-to-end GridWorld output validity check for the canonical POMDP test case | ✅ exit 1 on mismatch |
| `check_thin_orchestrators.py` | Enforces numbered thin-orchestrator line caps: 150 hard cap plus the registered ratchet in `thin_orchestrator_caps.json` | ✅ exit 1 on cap breach |
| `check_flag_parity.py` | Ratchets argparse-registered CLI flags against maintained-doc `--flag` mentions (phantom tokens + undocumented flags) | ✅ exit 1 over ratchet cap |
| `check_dep_hygiene.py` | Flags zero-consumer optional-dependency groups and ratchets stale `[tool.mypy.overrides]` entries | ✅ exit 1 on orphan extra or cap breach |

### Pipeline Orchestration (7 scripts)

| Script | Purpose | Delegates To |
|--------|---------|:------------:|
| `emit_run_manifest.py` | Thin CLI: emit durable v3 run manifests for a COMPLETED pipeline run | `pipeline.run_manifest` |
| `generate_pipeline_container_plan.py` | Generate auditable container plan from pipeline config | `pipeline.pipeline_container_plan` |
| `run_cross_framework_reliability.py` | Run profiled cross-framework reliability checks for maintained families | `pipeline.cross_framework_reliability` |
| `run_model_family_acceptance.py` | Run model-family acceptance tests (non-resumable) | `pipeline.model_family_acceptance` |
| `run_session_acceptance.py` | Run resumable, session-wrapped model-family acceptance | `pipeline.session_acceptance` |
| `run_semantic_fidelity_gate.py` | Run semantic fidelity gate for maintained model families | `pipeline.semantic_fidelity` |
| `run_v3_orchestration_acceptance.py` | End-to-end v3.0.0 orchestration acceptance gate (durable streams, run sessions, container plans) | `pipeline.*` |

### Manuscript Figure Generation (7 scripts)

| Script | Figure / Purpose |
|--------|-----------------|
| `manuscript_build_figures.py` | Build all manuscript figures (dispatches to backends) |
| `manuscript_fig_backend_matrix.py` | Backend matrix heatmap figure |
| `manuscript_fig_family_framework.py` | Model-family / framework compatibility figure |
| `manuscript_fig_orchestration.py` | v3.0.0 orchestration architecture figure |
| `manuscript_fig_pipeline_dag.py` | Pipeline DAG figure (25-step flow) |
| `manuscript_fig_repo_metrics.py` | Repository metrics figure |
| `manuscript_fig_triple_play.py` | Triple Play approach concept figure |

### Performance & Analysis (3 scripts)

| Script | Purpose |
|--------|---------|
| `scripts/experiments/run_pymdp_gnn_scaling_analysis.py` | Parameter grid scaling study (NxT) with visual meta-analysis and scaling-law fitting |
| `pymdp_spec_generator.py` | Generate pymdp specification from config |
| `z_generate_manuscript_variables.py` | Compute the deterministic `{{TOKEN}}` map from the live repo and hydrate manuscript sections before render |

### Maintenance (1 script)

| Script | Purpose |
|--------|---------|
| `scripts/experiments/add_module_docstrings.py` | Insert composed PEP 257 module docstrings into undocumented files (`--dry-run` supported) |

## Shared Utilities — `lib/`

The [`lib/`](lib/) subdirectory provides shared utility functions for multiple audit scripts:

- `lib/shared.py` — `repo_root()`, `should_skip_path()`, `is_generated_output()`, `add_strict_flag()`, `exit_with_findings()`
- `lib/manuscript_figure_tokens.py` — `load_tokens()`: the manuscript token map, recording which keys a figure generator reads so `manuscript_build_figures.py` can stamp figure provenance
- [`lib/AGENTS.md`](lib/AGENTS.md) — Documentation
- [`lib/README.md`](lib/README.md) — Quick reference
- [`lib/SPEC.md`](lib/SPEC.md) — Specification

## Operational Standards

- Strict adherence to Pythonic PEP validation and real-implementation testing principles.
- All scripts must contain structured `argparse` implementations mapped for headless CI.
- Orchestrators must implement explicit safety guardrails for resource-intensive operations.
- Automated manifest generation is required for all batch processing studies.
- All scripts follow the **thin orchestrator pattern**: parse args -> delegate to `src/` module -> return exit code (0/1/2). No domain logic lives in scripts.