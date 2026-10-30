# Workflows Agent Guide

The local default and primary ordinary test matrix now include Python 3.14,
with exact uv 0.13.4. The named h5py/HDF5 host exception and removal checks are
in root README.md. General docs/local/MCP/security jobs select 3.14; coupled
3.12 required statuses and native/paired/manuscript/GUI rosters remain explicit
compatibility or evidence profiles. Existing acceptance receipts retain their
original source, runtime and tool identities; this configuration change does
not renew those receipts.

## Purpose

Defines behavior and guardrails for workflows in this directory. Human index of all `.github/` automation: [../README.md](../README.md).

## Workflow set

| File | Role |
|------|------|
| `comprehensive-native-coverage.yml` | Separate exact-source Python 3.11/3.12/3.13 full-source coverage with native child provenance and exact raw/lane/combined unions. The original core selectors, reports and 60% floor remain intact. All five declared native lanes, clean identities, successful JUnit, raw integrity and hook restoration are required. Comprehensive acceptance separately enforces an 80% floor and `5 * covered_lines > 4 * num_statements` in each environment. |
| `ci.yml` | Python 3.11/3.12/3.13/3.14 tests with JUnit, coverage, artifacts and summaries. Independent 3.12 quality checks run in parallel; the existing `test (3.12)` status requires both the native test lane and the quality lane to succeed. Quality includes Ruff/mypy/public contracts, functional MCP/capability evidence, v3 orchestration acceptance. MCP tool count ≥ `MCP_TOOL_FLOOR` (140, defined in `tests/mcp/test_mcp_audit.py`). Bandit retains SARIF upload and failure on findings. No path filter — runs on doc-only changes too. |
| `installed-platforms.yml` | Frozen ordinary installed-wheel acceptance outside the checkout in five native lanes: Linux/Python 3.11, 3.12, 3.14; macOS/3.14; Windows/3.14. CLI/API/resources, JAX/THRML numerical witnesses and native filesystem/worker boundaries retain source/lock/wheel identities. Selected skips fail acceptance; platform support is limited to accepted native guarantees. |
| `local-gates.yml` | Repository gates bind manuscript tokens, hydrated prose and figures, then check MCP/skills, imports, thin orchestrators, flags, dependencies, public validation and maintained documentation. PR inherited-drift qualification is limited to byte-identical base evidence; main/manual checks remain strict. |
| `mcp-audit.yml` | MCP tool count ≥ `MCP_TOOL_FLOOR` audit on push/PR to `main`. |
| `full-extras.yml` | Weekly all-extras suite: `uv sync --frozen --all-extras`, optional-import validation, full pytest (Python 3.12). |
| `docs-audit.yml` | Strict Markdown/anchor and repository/maintained-doc terminology audits on every push/PR; scheduled or explicitly requested manual external-link observations. |
| `actionlint.yml` | Lint workflow YAML when `.github/workflows/**` changes. |
| `dependency-review.yml` | PR gate: high-severity failures; AGPL deny list; PR comment summary on failure. |
| `codeql.yml` | Python CodeQL: `init` → `uv sync --frozen --extra dev` → `analyze`; skips doc-only paths on push/PR; weekly Monday 04:28 UTC cron + `workflow_dispatch`. |
| `supply-chain-audit.yml` | Scheduled `pip-audit` on frozen exports (core + all extras, no dev); bash `set -euo pipefail`; job summary. |
| `custody-re-render.yml` | Daily cron 07:14 UTC + `workflow_dispatch`; report-only (no commit back). Fresh manuscript render via the `docxology/template` checkout (a real directory at `projects/active/`): template `stage_03_render` → record render-custody manifest → strict token gate → `tests/test_manuscript_latex_log.py`; receipts + rendered evidence uploaded as artifact. |
| `fep-lean-paired-revision.yml` | Release-time (tags/manual) strict current-source bridge and Q7 checks on Python 3.11/3.12/3.14 against the exact pinned FEP companion, with source/runtime bookends and both revisions retained. GNN owns its implementation source; the bridge binds only the declared notation contract, so ordinary PR/main development is not gated on the pair. Historical fixed-baseline receipts cannot admit GNN 5.0; native proof and release acceptance remain separate. [Policy](../../docs/development/fep_lean_paired_revision.md#current-source-qualification-for-gnn-50x). |
| `geo-infer-interchange.yml` | Blocking pinned GEO/GNN interchange on Python 3.11/3.12, retaining both source revisions, schema contracts, replay and artifact digests. Coordinate companion changes through the same paired-revision procedure. |
| `rxinfer.yml` | Path-scoped native RxInfer 6.1.0/ReactiveMP 7 checks on Julia 1.13.2: exact categorical/Gaussian numerical witnesses, structured model inference, and migrated custom Halfspace rules. Frozen environment drift fails; logs are retained. |
| `gridworld.yml` | Weekly/manual report-only GridWorld execution with both Julia environments, Ollama and Gemma 3 1B; all 25 steps and the fresh publication contract remain required. Retain summaries, test/LLM observations and daemon logs. |
| `pair-pin-freshness.yml` | Nightly pair-pin freshness gate (BC-12, scope-comp-consumers §5.3): validates both committed pair pins and asserts each pinned companion revision is ancestor-or-equal of the companion default-branch HEAD (`scripts/check_pair_pin_freshness.py`); nightly cron + `workflow_dispatch`. Exit 2 "re-pin required" names the stale pair file; a red run means the pair pin needs a bump. |

## Standards

- Pin every third-party action to a full commit SHA with the version as a trailing comment (e.g. `uses: actions/checkout@<sha> # v7.0.1`) for supply-chain hardening; Dependabot reads the version comment. Resolve SHAs with `git ls-remote` — never guess.
- Pin GNN-owned `astral-sh/setup-uv` steps with `version: "0.13.4"` (the exact Dockerfile/uv.toml requirement).
- The paired GEO installation has a named companion-only exception: exact uv 0.13.1 satisfies GEO_INFER's own `uv.toml`. Use separate pinned setup steps for the two repositories; select GNN's uv 0.13.4 again before GNN installation or `uv run` checks.
- Use explicit `timeout-minutes`.
- Apply least-privilege `permissions` globally and per job.
- Use deterministic dependency operations (`uv sync --frozen`, `uv export --frozen`).

Install each CI environment once with `uv sync --frozen`; subsequent commands
in `ci.yml` use `uv run --frozen --no-sync` against that installed environment.
Keep test marker selections, per-Python coverage/JUnit artifacts and source-bound
pipeline receipts intact when changing scheduling. The `test (3.12)` aggregate
runs with `always()` and rejects failed, cancelled, skipped or missing lane
results; moving validation into parallel jobs must preserve this coupling.

CI sets `UV_PYTHON` to the test matrix version (and 3.14 for security and
documentation jobs), overriding the local `.python-version` pin. XML export
validation uses `defusedxml` and rejects entity declarations.

Installed-platform acceptance builds the exact PR head or event revision,
verifies it before building, and retains actual source/wheel/lock identities.

The installed-platform workflow also retains a Linux/amd64 Docker build and
non-root Python 3.14 CLI/runtime checks. Keep its narrower container scope
separate from the five installed-wheel lanes and full native coverage.

The FEP paired job uses uv 0.13.0 to honor its companion project
requirement; it never installs GNN. GNN-owned installs use uv 0.13.4.
Supply-chain resolution uses core Python 3.14 and the declared all-extras
Python 3.12 profile.

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

## Release qualification

Version tags (`v[0-9]*`) run all sixteen release workflows, including the
otherwise scheduled/manual extras, security, GridWorld, pair freshness and
manuscript reproduction profiles. Ordinary PR/main triggers and test selections
remain unchanged. Draft PRs already execute the full checks, so marking them
ready does not retrigger those same expensive workflows.

The [release procedure](../../docs/releases/README.md#repeatable-release-qualification)
uses the [status collector](../release_gate.py) to collect exact commit/attempt/job
identities and dispatch only missing work. It never retries failed work or
replaces independent native, archive, companion or manuscript acceptance.
