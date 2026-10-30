# GitHub workflows

The local default and primary ordinary test matrix now include Python 3.14,
with exact uv 0.13.4. The named h5py/HDF5 host exception and removal checks are
in root README.md. General docs/local/MCP/security jobs select 3.14; coupled
3.12 required statuses and native/paired/manuscript/GUI rosters remain explicit
compatibility or evidence profiles. Existing acceptance receipts retain their
original source, runtime and tool identities; this configuration change does
not renew those receipts.

The GEO pair installs each repository with its own exact tool: uv 0.13.1 is
a companion-only exception for GEO_INFER, while GNN retains uv 0.13.4.
Separate pinned setup steps install GEO first and GNN last, so the later GNN
qualification controls also use uv 0.13.4. These controls reuse the installed
Python 3.11 environment with `--frozen --no-sync`; the local default does not
replace its H3-enabled runtime. The [local paired-runtime procedure](../../docs/development/geo_infer_2026_09.md#independent-uv-toolchains)
uses the same split. Manuscript audit artifacts retain the full render output and the render
step outcome even when strict comparison rejects the older committed chain.
Those diagnostic artifacts require normal verification and an artifact commit
before they become accepted publication evidence.

The installed-platform workflow also builds the Dockerfile on Linux/amd64 with
its frozen Python 3.14 environment. It checks the non-root uv executable and
public CLI/stream boundaries. This container lane retains build/input/image
identities; it does not qualify every optional backend or the full default CMD.

YAML workflows for CI, MCP tool-count audit, weekly all-extras suite, documentation audit, workflow lint, dependency review, CodeQL, and scheduled supply-chain checks. Parent hub (Dependabot + full index): [../README.md](../README.md). Agent guide: [AGENTS.md](AGENTS.md).

## Workflow files

| File | Triggers | Jobs / behavior |
|------|----------|-----------------|
| [comprehensive-native-coverage.yml](comprehensive-native-coverage.yml) | `push` / `pull_request` → `main`; `workflow_dispatch` | Separate Python 3.11/3.12/3.13 reports from the exact source revision. The five frozen core, MCP, pipeline, THRML and Gradio native selections retain process/source identities, unchanged statement/exclusion scope, actual raw line-set unions, JUnit and hook-restoration receipts. The original core 60% gate remains; comprehensive acceptance enforces an 80% floor and the strict integer >80% comparison separately in each environment. |
| [ci.yml](ci.yml) | `push` / `pull_request` → `main` (no path filter); `workflow_dispatch` | Full Python 3.11/3.12/3.13/3.14 unit/integration selections with JUnit and coverage. Independent 3.12 quality checks run in parallel; `test (3.12)` accepts only successful `pytest (3.12)` and `quality (3.12)` lanes. Quality retains Ruff, types, documentation/contracts, collection and MCP tests, and v3 orchestration acceptance. Pipeline contracts and extras retain separate mandatory lanes. Security retains Bandit SARIF, artifacts and failure on findings. |
| [installed-platforms.yml](installed-platforms.yml) | `push` / `pull_request` → `main`; `workflow_dispatch` | Five native installed-wheel lanes: Linux/Python 3.11, 3.12, 3.14; macOS/3.14; Windows/3.14. Frozen hashed binary runtime installation outside the checkout, native JAX/THRML, CLI/templates, API scratch concurrency, owned worker cancellation/RSS and platform boundary tests. Selected skips prevent acceptance; receipts retain exact source, wheel, lock and platform identities. |
| [local-gates.yml](local-gates.yml) | `push` / `pull_request` → `main`; `workflow_dispatch` | Repository gates for manuscript tokens, hydrated prose, figure freshness, MCP/skills, import boundaries, numbered orchestrators, flags, dependency consumers, validation surfaces and documentation paths/patterns. |
| [fep-lean-paired-revision.yml](fep-lean-paired-revision.yml) | tags; `workflow_dispatch` (release-time only) | Strict current-source bridge and Q7 checks on Python 3.11/3.12/3.14 against the exact pinned FEP companion, retaining both revisions and runtime/source bookends. GNN owns its implementation source; the bridge binds only the declared notation contract, so ordinary PR/main development is not gated on the pair. Historical fixed-baseline receipts cannot admit GNN 5.0; native proof and release acceptance remain separate. [Policy](../../docs/development/fep_lean_paired_revision.md#current-source-qualification-for-gnn-50x). |
| [geo-infer-interchange.yml](geo-infer-interchange.yml) | `push` / `pull_request` → `main`; `workflow_dispatch` | Python 3.11/3.12 categorical, Gaussian and factored interchange with the pinned GEO companion, deterministic replay and artifact digests. H3 state labels alone do not establish geographic or inference claims. |
| [mcp-audit.yml](mcp-audit.yml) | `push` / `pull_request` → `main`; `workflow_dispatch` | MCP tool count ≥ 140 via `tests.mcp.test_mcp_audit.count_mcp_tools`. |
| [full-extras.yml](full-extras.yml) | Weekly cron Sunday 06:00 UTC (`0 6 * * 0`); `workflow_dispatch` | `uv sync --frozen --all-extras`, optional-import checks (audio, GUI, research/scaling), full pytest suite under all extras (Python 3.12). |
| [docs-audit.yml](docs-audit.yml) | `push` / `pull_request` → `main`; weekly Monday 06:00 UTC; `workflow_dispatch` | Strict docs/anchor and repository/maintained-doc terminology audits on every change. Weekly or opt-in manual external observations fail on HTTP 404/410; other failures remain inconclusive and are retained. |
| [actionlint.yml](actionlint.yml) | Changes under `.github/workflows/**`; `workflow_dispatch` | `rhysd/actionlint@v1.7.12` |
| [dependency-review.yml](dependency-review.yml) | `pull_request` → `main`; `workflow_dispatch` | High severity + AGPL deny; PR comment summary on failure. Fork PRs may get limited review. |
| [codeql.yml](codeql.yml) | `push` / `pull_request` (skips doc-only paths), weekly cron, `workflow_dispatch` | Init → `uv sync --frozen --extra dev` → analyze (Python). |
| [supply-chain-audit.yml](supply-chain-audit.yml) | Weekly cron Monday 06:00 UTC, `workflow_dispatch` | Two `pip-audit` jobs (OSV); strict shell; job summary. |
| [custody-re-render.yml](custody-re-render.yml) | Daily cron 07:14 UTC (`14 7 * * *`); `workflow_dispatch` | Report-only fresh manuscript render from the `docxology/template` checkout (a real directory at `projects/active/`): template `stage_03_render` → record render-custody manifest → strict token gate → `tests/test_manuscript_latex_log.py`; manifest + rendered evidence uploaded as artifact, nothing committed back. |
| [rxinfer.yml](rxinfer.yml) | Relevant source/example/workflow changes on push/PR; `workflow_dispatch` | Native RxInfer 6.1.0/ReactiveMP 7 on Julia 1.13.2; exact smoothing/conditioning witnesses, structured model families and custom Halfspace rules. Retains logs and rejects environment drift. |
| [gridworld.yml](gridworld.yml) | Weekly Monday 03:43 UTC (`43 3 * * 1`); `workflow_dispatch` | Report-only fresh GridWorld pipeline and publication-contract check with Julia 1.13.2, both frozen Julia environments and Ollama 0.40.6 plus the explicit Gemma 3 1B CI model. All 25 steps remain required. Fresh summaries, test output and LLM records stay on the runner and are retained. |
| [pair-pin-freshness.yml](pair-pin-freshness.yml) | Nightly cron 05:23 UTC (`23 5 * * *`); `workflow_dispatch` | Validates both committed pair pins (`.github/fep-lean-pair.json`, `.github/gnn-pair.json`) and asserts each pinned revision is ancestor-or-equal of the companion default-branch HEAD via `scripts/check_pair_pin_freshness.py` (git-only; full-history checkouts, `fetch-depth: 0`). Exit 2 "re-pin required" names the stale pair file, pinned revision and companion tip; receipts uploaded as artifact. |

## Local validation

### Scheduling and evidence

The four Python test environments keep the same marker selection, coverage
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

Comprehensive coverage has its own three concurrent environment jobs. Each job
runs all five declared lanes serially so the owned temporary observation
hook cannot mix lane data. This additional evidence retains the existing core
reports and gates. Read the [observer procedure and limits](../../docs/development/native_comprehensive_coverage.md)
before local measurement; do not share its owned environment during a run.

### Workflow lint

```bash
actionlint .github/workflows/*.yml
```

Run from repo root (paths relative to root).

CI sets `UV_PYTHON` to the test matrix version (and 3.14 for security and
documentation jobs), overriding the local `.python-version` pin. XML export
validation uses `defusedxml` and rejects entity declarations.

The installed-platform matrix explicitly checks out the candidate head for a
PR and the event revision for main/manual runs. It verifies that revision before
building the wheel and records the actual checkout, wheel and lock hashes.

The FEP bridge job selects its companion-required uv 0.13.0 independently of
GNN-owned uv 0.13.4 installs. Supply-chain requirement resolution uses core Python 3.14 and the declared
all-extras Python 3.12 profile, rather than prospective Python 3.15.

The docs audit runs the existing exit-code documentation contract immediately
after its frozen installation. `just release-preflight` runs the same test before
type checks and broader audits, catching wording drift before a release candidate
is dispatched. The full ordinary and native test selections remain unchanged.

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

The all-extras suite also retains `e4-finite-filtering.json` and
`e4-gaussian-filtering.json` in its existing JUnit artifact. Both independent
numerical witnesses must execute at the actual checkout before their receipt
step passes. The [E4 foundation contract](../../docs/development/remaining_roadmap_contracts.md#e4-foundation-source-bound-numerical-witness-v1)
separates those finite numerical comparisons from formal proof and broader backend
claims. The complete suite selection and coverage ownership remain unchanged.
