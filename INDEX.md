# GeneralizedNotationNotation — repository index

Generalized Notation Notation (GNN) is a text-based language for writing Active
Inference generative models, plus the 25-step pipeline that parses, validates,
renders, executes, analyzes, and publishes them. Start with
[README.md](README.md) for purpose and quick start, [AGENTS.md](AGENTS.md) for
the agent contract, and [TO-DO.md](TO-DO.md) for open work.

## Gate

`./run.sh gates` is the canonical gate: `ruff format --check`, `ruff check`,
`mypy`, the default pytest selection under the `fail_under` coverage floor in
[pyproject.toml](pyproject.toml), the documentation/terminology/document-contract
audits, the capability-contract audit, the MCP functional surface with a
full-tree collection check, and the v3 orchestration acceptance gate. Individual
arms: `./run.sh lint|type|test|coverage|docs|contracts|acceptance`; `./run.sh sync`
installs the frozen dev environment. Each arm's command, the CI lane it mirrors,
and the measured wall time are in the [gate run guide](docs/RUN_GUIDE.md).

`just --list` lists the wider developer recipes. `just quality` adds the fast
non-test surfaces on top of the gate — import-linter, the thin-orchestrator and
stdlib-shadow guards, the ontology pin, flag parity, dependency hygiene,
validate-surface, `bandit`, skills health, manuscript custody, and the MCP tool
count. `just test` runs the default pytest selection; `just test-cov` reports
coverage against the same floor.

[.github/workflows/ci.yml](.github/workflows/ci.yml) runs the same surfaces as
the `quality (3.12)` and `pytest (3.12)` arms, plus the Python 3.11/3.13/3.14
test matrix.

## Root documents

| File | Holds |
|------|-------|
| [README.md](README.md) | Purpose, quick start, features, published-output evidence |
| [AGENTS.md](AGENTS.md) | Agent contract, module registry, pipeline flow |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Thin-orchestrator pattern and extension guides |
| [DOCS.md](DOCS.md) | Narrative system overview with diagrams |
| [SPEC.md](SPEC.md) | Repository-level specification |
| [Interoperability guide](docs/development/interop.md) | Typed contracts, API envelopes, benchmark results, provenance, and gaps |
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Installation and environment troubleshooting |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution workflow |
| [TO-DO.md](TO-DO.md) | Canonical backlog and remaining workstreams |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [SECURITY.md](SECURITY.md) | Security policy and reporting |
| [SUPPORT.md](SUPPORT.md) | Help channels |
| [CITATION.cff](CITATION.cff) | Citation metadata |
| [manuscript/](manuscript/) | Paper chapters (`00_abstract.md` … `99_references.md`) |

## Execution surface

| Path | Contents |
|------|----------|
| [src/gnn/main.py](src/gnn/main.py) | Pipeline orchestrator (argument parsing, step sequencing, summary) |
| [src/gnn/](src/gnn/) `0_*.py` … `24_*.py` | 25 thin step orchestrators, one per pipeline step |
| [src/gnn/pipeline/step_registry.py](src/gnn/pipeline/step_registry.py) | Canonical step list (`just steps` prints it) |
| [src/gnn/pipeline/](src/gnn/pipeline/) | Orchestration contracts: `durable_streams.py`, `run_session.py`, `container_plan.py` and their live wiring |
| [src/gnn/](src/gnn/), [src/gnn/render/](src/gnn/render/), [src/gnn/execute/](src/gnn/execute/) | Parsing/validation, code generation, execution |
| [src/gnn/cli/](src/gnn/cli/) | `gnn` console script (`run`, `validate`, `render`, `serve`, `templates list`, `pull`) |
| [src/gnn/mcp/](src/gnn/mcp/), [src/gnn/api/](src/gnn/api/) | MCP tools, the local HTTP server, and the job API |
| [src/gnn/SKILL.md](src/gnn/SKILL.md) | Agent-facing capability summary |
| [tests/](tests/) | Test suite; [README.md](tests/README.md) and [TEST_SUITE_SUMMARY.md](tests/TEST_SUITE_SUMMARY.md) describe its layout |

## Documentation tree

| Path | Contents |
|------|----------|
| [docs/README.md](docs/README.md) | Documentation hub |
| [docs/INDEX.md](docs/INDEX.md) | Flat link index |
| [docs/gnn/](docs/gnn/) | GNN language: syntax reference, modules, testing specs, operations |
| [docs/quickstart.md](docs/quickstart.md) | First-run tutorial |
| [docs/pipeline/](docs/pipeline/) | Pipeline internals, including [v3 orchestration](docs/pipeline/v3_orchestration.md) |
| [docs/pymdp/](docs/pymdp/), [docs/rxinfer/](docs/rxinfer/) | Framework integration guides |
| [docs/execution/](docs/execution/), [docs/export/](docs/export/) | Step-specific contracts |
| [docs/visualization/](docs/visualization/), [docs/advanced_visualization/](docs/advanced_visualization/) | Graph and advanced visualization |
| [docs/development/](docs/development/) | Maintainer tooling, including the documentation audit |
| [docs/deployment/](docs/deployment/) | Deployment notes |
| [docs/decisions/](docs/decisions/) | Architecture decision records |
| [docs/](docs/) | Static documentation; maintained folders carry `README.md` + `AGENTS.md` |

## Tooling, data, and configuration

| Path | Contents |
|------|----------|
| [scripts/](scripts/) | Gate and audit scripts: `check_capability_contracts.py`, `check_pomdp_gridworld_outputs.py`, `run_v3_orchestration_acceptance.py`, `run_model_family_acceptance.py`, `run_semantic_fidelity_gate.py`, `run_cross_framework_reliability.py`, `emit_run_manifest.py`, `generate_pipeline_container_plan.py`, `run_session_acceptance.py`, manuscript figure builders |
| [input/](input/) | `config.yaml` and sample models under `input/gnn_files/` |
| [output/](output/) | Curated publication evidence (manuscript, figures, PDF) is tracked; per-step run outputs are volatile and regenerated — see [output tracking](docs/development/output_tracking.md) |
| [justfile](justfile) | Developer recipes (`just --list`) |
| [pyproject.toml](pyproject.toml) | Dependencies, ruff/mypy/pytest/coverage configuration |
| [uv.toml](uv.toml), [uv.lock](uv.lock) | uv `required-version`, dependency constraints, and the frozen resolution |
| [Dockerfile](Dockerfile) | linux/amd64 test image |
| [.github/](.github/) | Workflows, issue templates, [repository hub](.github/README.md) |
| [.agent_rules/](.agent_rules/) | Repository rules referenced from `AGENTS.md` |

## Where each fact lives

| Fact | Canonical home |
|------|----------------|
| Gate definition | [run.sh](run.sh) (the arm), [justfile](justfile) (wider recipes), [CI](.github/workflows/ci.yml) |
| Step inventory | [src/gnn/pipeline/step_registry.py](src/gnn/pipeline/step_registry.py) |
| Module behavior | Each package's `AGENTS.md` |
| Open work | [TO-DO.md](TO-DO.md) |
| Shipped history | [CHANGELOG.md](CHANGELOG.md) |
| Measured results | Live command output; `output/` artifacts |
| Public type and API identities | [Typed/API contracts](docs/development/interop_api.md) |
| Benchmark producers, result shapes, data provenance | [Research contracts](docs/development/interop_research.md) |
