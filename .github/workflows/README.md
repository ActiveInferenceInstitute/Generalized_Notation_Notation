# GitHub workflows

YAML workflows for CI, MCP tool-count audit, weekly all-extras suite, documentation audit, workflow lint, dependency review, CodeQL, and scheduled supply-chain checks. Parent hub (Dependabot + full index): [../README.md](../README.md). Agent guide: [AGENTS.md](AGENTS.md).

## Workflow files

| File | Triggers | Jobs / behavior |
|------|----------|-----------------|
| [ci.yml](ci.yml) | `push` / `pull_request` → `main` (no path filter); `workflow_dispatch` | Full Python 3.11/3.12/3.13 unit/integration selections with JUnit and coverage. Independent 3.12 quality checks run in parallel; `test (3.12)` accepts only successful `pytest (3.12)` and `quality (3.12)` lanes. Quality retains Ruff, types, documentation/contracts, collection, focused PyMDP/POMDP and MCP tests, and v3 orchestration acceptance. Pipeline contracts and extras retain separate mandatory lanes. Security retains Bandit SARIF, artifacts and failure on findings. |
| [mcp-audit.yml](mcp-audit.yml) | `push` / `pull_request` → `main`; `workflow_dispatch` | MCP tool count ≥ 140 via `tests.mcp.test_mcp_audit.count_mcp_tools`. |
| [full-extras.yml](full-extras.yml) | Weekly cron Sunday 06:00 UTC (`0 6 * * 0`); `workflow_dispatch` | `uv sync --frozen --all-extras`, optional-import checks (audio, GUI, research/scaling), full pytest suite under all extras (Python 3.12). |
| [docs-audit.yml](docs-audit.yml) | `push` / `pull_request` when `*.md`, `docs/**`, root `AGENTS.md`/`CLAUDE.md`/`README.md`/`SKILL.md`, or `docs/development/docs_audit.py` change; `workflow_dispatch` | Strict docs audit with anchors plus repository/doc terminology and GNN doc-pattern audits. |
| [actionlint.yml](actionlint.yml) | Changes under `.github/workflows/**`; `workflow_dispatch` | `rhysd/actionlint@v1.7.12` |
| [dependency-review.yml](dependency-review.yml) | `pull_request` → `main`; `workflow_dispatch` | High severity + AGPL deny; PR comment summary on failure. Fork PRs may get limited review. |
| [codeql.yml](codeql.yml) | `push` / `pull_request` (skips doc-only paths), weekly cron, `workflow_dispatch` | Init → `uv sync --frozen --extra dev` → analyze (Python). |
| [supply-chain-audit.yml](supply-chain-audit.yml) | Weekly cron Monday 06:00 UTC, `workflow_dispatch` | Two `pip-audit` jobs (OSV); strict shell; job summary. |
| [custody-re-render.yml](custody-re-render.yml) | Daily cron 07:14 UTC (`14 7 * * *`); `workflow_dispatch` | Report-only fresh manuscript render from the `docxology/template` checkout (symlinked at `projects/active/`): template `stage_03_render` → record render-custody manifest → strict token gate → `tests/test_manuscript_latex_log.py`; manifest + rendered evidence uploaded as artifact, nothing committed back. |
| [pair-pin-freshness.yml](pair-pin-freshness.yml) | Nightly cron 05:23 UTC (`23 5 * * *`); `workflow_dispatch` | Validates both committed pair pins (`.github/fep-lean-pair.json`, `.github/gnn-pair.json`) and asserts each pinned revision is ancestor-or-equal of the companion default-branch HEAD via `scripts/check_pair_pin_freshness.py` (git-only; full-history checkouts, `fetch-depth: 0`). Exit 2 "re-pin required" names the stale pair file, pinned revision and companion tip; receipts uploaded as artifact. |

## Local validation

### Scheduling and evidence

The three Python test environments keep the same marker selection, coverage
reports and `pytest-junit-<version>` artifacts. The quality lane emits the
existing `mcp-capabilities` artifact; the pipeline lane emits its JUnit and
source-bound count receipt. The required statuses `test (3.11)`, `test (3.12)`
and `test (3.13)` retain their names.

Python 3.12 quality checks and its main test suite run on separate runners.
The aggregate `test (3.12)` job uses `always()` and requires both lane results
to be `success`; failure, cancellation, skipping or a missing result prevents
acceptance. Each runner installs locked dependencies once; CI commands then
use `uv run --frozen --no-sync`. This changes scheduling and repeated setup,
while preserving test, security and custody coverage.

Assess speed from exact-run job/step timestamps and retained JUnit receipts.
Runner availability and test-duration variance affect elapsed time; report
before/after source identities and complete outcomes with timing comparisons.

### Workflow lint

```bash
actionlint .github/workflows/*.yml
```

Run from repo root (paths relative to root).

CI sets `UV_PYTHON` to the test matrix version (and 3.12 for security and
documentation jobs), overriding the local `.python-version` pin. XML export
validation uses `defusedxml` and rejects entity declarations.
