# Workflows Agent Guide

## Purpose

Defines behavior and guardrails for workflows in this directory. Human index of all `.github/` automation: [../README.md](../README.md).

## Workflow set

| File | Role |
|------|------|
| `comprehensive-native-coverage.yml` | Separate exact-source Python 3.11/3.12/3.13 full-source coverage with native child provenance and exact raw/lane/combined unions. The original core selectors, reports and 60% floor remain intact. All five declared native lanes, clean identities, successful JUnit, raw integrity and hook restoration are required. Comprehensive acceptance separately enforces an 80% floor and `5 * covered_lines > 4 * num_statements` in each environment. |
| `ci.yml` | Python 3.11/3.12/3.13 tests with JUnit, coverage, artifacts and summaries. Independent 3.12 quality checks run in parallel; the existing `test (3.12)` status requires both the native test lane and the quality lane to succeed. Quality includes Ruff/mypy/docs, functional MCP/capability evidence, focused PyMDP/POMDP tests and v3 orchestration acceptance. MCP tool count ≥ `MCP_TOOL_FLOOR` (140, defined in `tests/mcp/test_mcp_audit.py`). Bandit retains SARIF upload and failure on findings. No path filter — runs on doc-only changes too. |
| `installed-platforms.yml` | Frozen ordinary installed-wheel acceptance outside the checkout in five native lanes: Linux/Python 3.11, 3.12, 3.14; macOS/3.14; Windows/3.14. CLI/API/resources, JAX/THRML numerical witnesses and native filesystem/worker boundaries retain source/lock/wheel identities. Selected skips fail acceptance; platform support is limited to accepted native guarantees. |
| `local-gates.yml` | Repository gates bind manuscript tokens, hydrated prose and figures, then check MCP/skills, imports, thin orchestrators, flags, dependencies, public validation and maintained documentation. PR inherited-drift qualification is limited to byte-identical base evidence; main/manual checks remain strict. |
| `mcp-audit.yml` | MCP tool count ≥ `MCP_TOOL_FLOOR` audit on push/PR to `main`. |
| `full-extras.yml` | Weekly all-extras suite: `uv sync --frozen --all-extras`, optional-import validation, full pytest (Python 3.12). |
| `docs-audit.yml` | Strict Markdown audit when docs or `docs_audit.py` change. |
| `actionlint.yml` | Lint workflow YAML when `.github/workflows/**` changes. |
| `dependency-review.yml` | PR gate: high-severity failures; AGPL deny list; PR comment summary on failure. |
| `codeql.yml` | Python CodeQL: `init` → `uv sync --frozen --extra dev` → `analyze`; skips doc-only paths on push/PR; weekly Monday 04:28 UTC cron + `workflow_dispatch`. |
| `supply-chain-audit.yml` | Scheduled `pip-audit` on frozen exports (core + all extras, no dev); bash `set -euo pipefail`; job summary. |
| `custody-re-render.yml` | Daily cron 07:14 UTC + `workflow_dispatch`; report-only (no commit back). Fresh manuscript render via the `docxology/template` checkout (symlinked at `projects/active/`): template `stage_03_render` → record render-custody manifest → strict token gate → `tests/test_manuscript_latex_log.py`; receipts + rendered evidence uploaded as artifact. |
| `fep-lean-paired-revision.yml` | Paired-revision CI for the fep_lean bridge pair: validates `.github/fep-lean-pair.json`, checks out fep_lean at the pinned SHA, runs fep_lean's read-only bridge surface (status, emit `--check` finite/continuous) against this GNN checkout; blocking. Canonical custody ordering: [docs/development/fep_lean_paired_revision.md](../../docs/development/fep_lean_paired_revision.md). |
| `geo-infer-interchange.yml` | Blocking pinned GEO/GNN interchange on Python 3.11/3.12, retaining both source revisions, schema contracts, replay and artifact digests. Coordinate companion changes through the same paired-revision procedure. |
| `gridworld.yml` | Weekly/manual report-only GridWorld execution with provisioned Julia environments and a fresh publication-contract check. Volatile outputs are not committed; retain the pipeline summary. |
| `pair-pin-freshness.yml` | Nightly pair-pin freshness gate (BC-12, scope-comp-consumers §5.3): validates both committed pair pins and asserts each pinned companion revision is ancestor-or-equal of the companion default-branch HEAD (`scripts/check_pair_pin_freshness.py`); nightly cron + `workflow_dispatch`. Exit 2 "re-pin required" names the stale pair file; a red run means the pair pin needs a bump. |

## Standards

- Pin every third-party action to a full commit SHA with the version as a trailing comment (e.g. `uses: actions/checkout@<sha> # v7.0.1`) for supply-chain hardening; Dependabot reads the version comment. Resolve SHAs with `git ls-remote` — never guess.
- Pin `astral-sh/setup-uv` steps with `version: "0.12"` (the Dockerfile `UV_VERSION` bootstrap floor minor series).
- Use explicit `timeout-minutes`.
- Apply least-privilege `permissions` globally and per job.
- Use deterministic dependency operations (`uv sync --frozen`, `uv export --frozen`).

Install each CI environment once with `uv sync --frozen`; subsequent commands
in `ci.yml` use `uv run --frozen --no-sync` against that installed environment.
Keep test marker selections, per-Python coverage/JUnit artifacts and source-bound
pipeline receipts intact when changing scheduling. The `test (3.12)` aggregate
runs with `always()` and rejects failed, cancelled, skipped or missing lane
results; moving validation into parallel jobs must preserve this coupling.

CI sets `UV_PYTHON` to the test matrix version (and 3.12 for security and
documentation jobs), overriding the local `.python-version` pin. XML export
validation uses `defusedxml` and rejects entity declarations.

Installed-platform acceptance builds the exact PR head or event revision,
verifies it before building, and retains actual source/wheel/lock identities.
