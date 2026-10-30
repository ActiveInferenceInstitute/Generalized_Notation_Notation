# Specification: Workflows

The local default and primary ordinary test matrix now include Python 3.14,
with exact uv 0.13.4. The named h5py/HDF5 host exception and removal checks are
in root README.md. General docs/local/MCP/security jobs select 3.14; coupled
3.12 required statuses and native/paired/manuscript/GUI rosters remain explicit
compatibility or evidence profiles. Existing acceptance receipts retain their
original source, runtime and tool identities; this configuration change does
not renew those receipts.

## Design Requirements
The `workflows` directory serves as the execution layer for the GNN repository's Continuous Integration and Continuous Deployment (CI/CD) pipeline. Its primary design requirement is to enforce real-implementation test coverage, maintain repository stability, and ensure code quality via automated execution graphs triggered by GitHub events.

The workflow system must:
1. Support deterministic isolation of testing environments utilizing `uv`.
2. Securely handle permission scopes and artifact generation.
3. Facilitate strict parallelized static analysis (CodeQL, Bandit, Actionlint) and security audits.
4. Offer resilient and debuggable job summaries.

## Components
The module defines strict automation routines via YAML configuration:

1. **`ci.yml`**: The primary test matrix (Python 3.11, 3.12, 3.13, 3.14) that executes the Pytest suite, Ruff linting, MCP tool count assertions, and Bandit security scans, halting on critical code health violations.
2. **`actionlint.yml`**: Safeguards workflow logic by running static analysis on the YAML files themselves before integration.
3. **`codeql.yml`**: Triggers GitHub's advanced CodeQL semantic scanning engine for deep security auditing.
4. **`dependency-review.yml`**: A supply chain guardrail enforcing license compliance (denying AGPL) and halting high-severity CVE introductions on Pull Requests.
5. **`docs-audit.yml`**: An orchestration layer executing the `docs_audit.py` strict enforcement of Markdown integrity.
6. **`supply-chain-audit.yml`**: A weekly scheduled audit explicitly exporting headless `uv` lockfiles and executing `pip-audit` via OSV vulnerability tracking.
7. **`mcp-audit.yml`**: A fast MCP tool-count regression gate (≥ 140 registered tools) on push/PR to `main`.
8. **`full-extras.yml`**: A weekly scheduled matrix validating optional dependency groups (audio, GUI, research, scaling) and running the full pytest suite under `--all-extras`.
9. **`pair-pin-freshness.yml`**: A nightly watchdog asserting each committed pair pin (fep_lean, GEO_INFER) stays ancestor-or-equal of its companion default-branch HEAD, exiting 2 with a "re-pin required" diagnosis naming the stale pair file, the pinned revision and the companion tip.
10. **`comprehensive-native-coverage.yml`**: Three independent exact-source Python environments measure native core, MCP and pipeline execution with full unchanged coverage scope, process provenance, exact raw/lane/combined line unions and fail-closed unreadable-data handling. Existing core reports and floor remain intact; a rounded percentage cannot establish the strict >80% roadmap target.
11. **`installed-platforms.yml`**: Five native installed-wheel environments build the exact candidate revision, install frozen hashed runtimes outside the checkout, and retain numerical/API/process/filesystem acceptance with zero selected skips and tracked-file bookends.
12. **`local-gates.yml`**: Source-bound manuscript/token/hydration/figure custody plus MCP/skills, import, orchestrator, flag, dependency, validation and documentation gates.
13. **`fep-lean-paired-revision.yml`**: Release-time (tags/manual) exact pinned companion bridge status and finite/continuous emission checks with both revisions retained. GNN owns its implementation source; the bridge binds only the declared notation contract. Custody agreement remains distinct from formal proof.
14. **`geo-infer-interchange.yml`**: Exact pinned companion interchange, deterministic replay and digests for supported categorical, Gaussian and factored contracts.
15. **`custody-re-render.yml`**: Daily/manual fresh template render and custody audit with retained artifacts and no repository writeback.
16. **`gridworld.yml`**: Weekly/manual report-only fresh GridWorld execution and publication-contract check with explicitly provisioned Julia environments.

## Technical Rules
- **Syntax**: 100% strict adherence to GitHub Actions YAML schemas.
- **Resilience**: Every workflow must declare explicit `timeout-minutes` and granular `permissions`.
- **Parallelism**: Implement appropriate `concurrency` groups to aggressively cancel redundant pipeline executions and conserve CI runner time.
- **Determinism**: Dependency installation must strictly use `uv sync --frozen` or equivalent zero-drift package resolution commands to guarantee reproducible CI environments.

Deterministic checks have one hosted owner: docs/anchors and terminology in
`docs-audit.yml`; import boundaries, thin orchestrators, flag/dependency/validation
and documentation path/pattern gates in `local-gates.yml`; Ruff, mypy, public
contracts and functional MCP tests in `ci.yml` quality. Julia caches
contain versioned package inputs and compiled code; each native test still runs.
External 403/429, network and timeout observations are inconclusive, never proof
that a URL is valid. `--strict` remains available to fail on every finding.

The ordinary test matrix owns the PyMDP contract, discrete-model and matrix
visualization suites. Quality does not rerun these three files: `test (3.12)`
already requires their complete ordinary-suite execution alongside quality.

Gridworld warms actual Julia imports before the bounded readiness probes. Its
explicit LLM profile uses `gemma3:1b`, 8,192 context tokens, seed 42,
4,096 output tokens, 180-second requests and a 600-second total budget. The
40-word bundled task style and 1,000-character JSON boundary still require
nonempty terminal answers for all nine tasks; truncation/boundary hits fail.
The generated profile, full run log and execution/analysis artifacts are retained.

The CI model is an explicit profile because the default 135M model returned
incomplete summaries in the full Gridworld run. Terminal coverage is runtime
acceptance, not a claim about scientific answer quality. Versioned Julia inputs
may be saved after a later stage fails; cached test outcomes are never accepted.

The all-extras lane uses pytest-xdist work stealing over the unchanged full
test selection, with the same coverage and JUnit outputs. Its Python 3.12
selection is explicit; installation happens once before `--frozen --no-sync`
commands. Selected test failures still fail the job.

Python setup checks the latest patch within each declared compatibility series.
Gridworld explicitly provisions that interpreter before uv sync, avoiding an
older preinstalled system Python satisfying only the minor-version selector.
