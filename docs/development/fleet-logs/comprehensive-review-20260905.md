# Comprehensive post-wave2 review — 2026-09-05

Verified locally on 2026-09-05. Read-only review of the wave2 tree
(`main` @ `64d49355a`, 287 uncommitted files) across nine parallel review
lenses — full strict mypy, ruff lint + format, focused wave2/identity tests,
LLM sync-wrapper tests, zero-skip + doc contract tests, the four doc-audit
gates, the strict v3 orchestration acceptance gate, one full-suite
ground-truth run, and a coherence/stale-doc sweep — followed by docs-only
improvements. Nothing was committed or pushed; the existing dirty state was
preserved; no source pins, dependency changes, or FEP edits.

## Gate results (measured 2026-09-05)

| Gate | Command | Observed result |
| --- | --- | --- |
| mypy (full, strict) | `uv run --extra dev mypy src --config-file pyproject.toml` | exit 0 — `Success: no issues found in 986 source files` |
| ruff lint | `uv run --extra dev ruff check src scripts` | exit 0 — `All checks passed!` |
| ruff format | `uv run --extra dev ruff format --check src scripts` | exit 0 — `1016 files already formatted` |
| Wave2 pipeline + run identity | pytest over `tests/pipeline/test_wave2_sessions.py`, `test_wave2_identity.py`, `test_wave2_manifests.py`, `test_wave2_container_plans.py`, `test_run_identity.py` | `68 passed in 1.88s` — 0 failed, 0 skipped |
| LLM sync wrappers | pytest over `tests/llm/test_openai_sync_analysis_contract.py`, `test_llm_sync_wrappers.py` | `13 passed in 1.20s` — 0 failed, 0 skipped, no network/Ollama needed |
| Zero-skip + doc contracts | pytest over `test_zero_skip_contracts.py`, `test_doc_accuracy_contracts.py`, `test_doc_contracts.py`, `test_docs_audit.py` | `11 passed in 0.24s` pre-edit; `11 passed in 0.27s` after the doc edits below |
| Doc audits | `docs/development/docs_audit.py --strict --check-anchors --no-write`; `scripts/check_gnn_doc_patterns.py --strict`; `scripts/check_repo_terminology.py --strict`; `scripts/check_maintained_doc_terms.py --strict` | all exit 0 — broken links 0, bad anchors 0, AGENTS/README gaps 0, no banned patterns, terminology clean |
| v3 orchestration acceptance | `uv run --extra dev python scripts/run_v3_orchestration_acceptance.py --strict` | `19/19 checks passed`, exit 0 (streams 9, session 6, container 4, negative controls firing) |
| Full suite (ground truth) | command of record: `uv run --extra dev python -m pytest tests/ -q --tb=no -rsx --ignore=tests/llm/test_llm_ollama.py --ignore=tests/llm/test_llm_ollama_integration.py` (run exactly once) | `4693 passed, 14 skipped in 536.44s (0:08:56)` — 0 failed, no collection errors |

Full-suite skips, all within the zero-skip allowlist: 11 sklearn
(`tests/ml_integration/test_ml_integration_inference.py`), 1 torch
(`tests/render/test_continuous_renderers.py:109`), 2 D2 CLI
(`tests/visualization/test_d2_visualizer.py:262,384`). The suite total
moved 4028 (pre-wave2 baseline, 2 failed) → 4102 (wave2 receipts) → 4693
today because later fleet lanes kept adding tests; both baseline failures
(doc-orchestrator line counts + zero-skip contract) remain fixed: 0 today.

## Coherence sweep findings

Verified coherent, no action:
- Zero-skip truth: the only `importorskip` occurrences under `tests/`
  are inside the contract module itself and the `DEFAULT_SKIP_ALLOWLIST`
  files (sklearn in `test_ml_integration_inference.py`; torch/cmdstanpy in
  `test_continuous_renderers.py`). No non-allowlisted occurrence.
- Security surface: `src/gnn/export/processor.py` reads XML/GraphML/GEXF via
  `defusedxml.ElementTree.parse` with DTDs, entities and external references
  forbidden; no raw `ET.parse`/`ET.fromstring` call sites in
  `src/gnn/export/format_exporters.py` or `formatters.py` (stdlib ElementTree is
  used for serialization only). `src/gnn/ml_integration/inference.py` implements
  the restricted global-allowlist unpickler; no `pickle.load(` call in the
  module. `defusedxml>=0.7.1` is declared in `pyproject.toml`.
- OpenAI sync wrapper: `src/gnn/llm/providers/openai_provider.py` `analyze` is
  synchronous, checks `asyncio.get_running_loop()` before creating the
  coroutine, runs the active-loop path through a single-worker
  ThreadPoolExecutor, and re-raises provider errors without re-running the
  billable request; `test_openai_sync_analysis_contract.py` and
  `test_llm_sync_wrappers.py` pin exactly this shipped behavior.
- Hygiene: zero TODO/FIXME/XXX in the wave2-owned files (`src/gnn/main.py`,
  `src/gnn/cli/__init__.py`, `src/gnn/pipeline/*.py`).
- No stale suite/mypy numbers in living docs; `CHANGELOG.md` historical
  entries are dated and correct as history. `docs/HANDOFF.md` is a
  deliberately superseded 2026-07-30 snapshot and was left untouched.

Fixed by this review (docs only; every edit re-verified — see below):
1. Discovery gap: all six durable-run topics from the wave2 receipt
   (`gnn-run-v2` hash schema, manifest index 3.1, legacy rejection/re-emission,
   source/config-bound DONE reuse, RUNNING recovery, single-writer boundary)
   were documented in `docs/development/durable-runs.md` but nothing linked to
   it from the repo's front doors. `README.md`'s v3.0.0 overview bullet and
   the Long-Running Orchestration section now point at it, naming `gnn-run-v2`
   and index schema 3.1 explicitly.
2. Missing changelog entry: `CHANGELOG.md` had no entry for the wave2 run
   identity hardening; an `[Unreleased]` → Changed entry ("Durable run
   identity and reproduction") now records it with the rules link.
3. Stale dependency lists: `src/gnn/export/AGENTS.md` and
   `docs/gnn/modules/07_export.md` Required Dependencies omitted the hardened
   reader; both now list `defusedxml` (mirroring `src/gnn/export/README.md`,
   which already documented it).

Awareness, deliberately not changed under the docs-only mandate:
- `src/gnn/export/formatters.py` keeps a stdlib `xml.dom.minidom` fallback on
  ImportError for output formatting. With `defusedxml` declared, that
  fallback is effectively dead in the uv env; it is serialization-side, not
  untrusted-input parsing. Candidate for simplification in a future code
  ownership window (API-adjacent).
- `src/gnn/ml_integration/AGENTS.md` lists `pickle` as a stdlib dependency, which
  is accurate; the restricted-loader behavior is documented in
  `src/gnn/ml_integration/README.md` (allowlist, extension opcodes, trailing
  bytes). No edit needed.

## Verification of the improvements

Post-edit re-run, all green: the four doc audits (broken links 0, bad
anchors 0, all AGENTS/README gap counts 0; no banned patterns; terminology
clean) and the 11 zero-skip + doc contract tests.

## Boundaries

- Review was a nine-item read-only scout pool; one lens (focused contract
  tests) could not launch commands in its scout environment and was re-run
  directly by the integrator. All numbers above are from the integrator's
  own re-verification or verbatim scout command output.
- The tree remains uncommitted; no git mutations of any kind;
  `pyproject.toml` untouched; nothing outside this repository touched.

## Follow-ups

- Consider dropping the dead `xml.dom.minidom` ImportError fallback in
  `src/gnn/export/formatters.py` when a code window opens.
- Keep the command of record as the measurement basis for future receipts;
  pass totals move as lanes add tests (4028 → 4102 → 4693 across receipts).
- The wave2 receipt's documentation handoff item (durable-run/reproduction
  docs for the six topics) is closed: coverage exists in
  `docs/development/durable-runs.md` and is now discoverable from README and
  CHANGELOG.

## Appendix — dirty-state ledger (advisory close-out, 2026-09-05)

Method: day-level mtime grouping over `git status --porcelain` (tracked-modified
and untracked), then minute-level `stat` on every file dated 2026-09-05.

- This session's footprint (verified): 4 tracked doc edits (`README.md`,
  `CHANGELOG.md`, `src/gnn/export/AGENTS.md`, `docs/gnn/modules/07_export.md`) plus
  this report file. Zero deletions, zero reverts, zero commits.
- No verification-run artifacts from this session's pytest/docs-audit runs:
  every `out/` artifact is dated 2026-09-04 13:47–13:59 (prior sessions), and
  `.pytest_cache` / `.mypy_cache` are gitignored.
- Concurrent-lane activity dated 2026-09-05, none of it this session's: a
  docs/manuscript/publication lane (`docs/manuscript/*.md` edited 02:38;
  `output/{manuscript,pdf,slides,web,data,reports,figures}` regenerated
  09:08–09:09; `src/gnn/manuscript_variables.py`) and an MCP audit regeneration
  (`src/gnn/mcp/audit_report.json`, 00:42). This is a shared fleet checkout.
- The 287 → 300 porcelain delta is therefore: +5 this session, the remainder
  concurrent-lane additions/edits on top of the pre-existing wave2 state.
- Directory-ownership rule adopted for any follow-on work in this checkout:
  `manuscript/`, `output/{manuscript,pdf,slides,web,reports,data}`,
  `src/gnn/manuscript_variables.py`, and `src/gnn/mcp/audit_report.json` belong to the
  active docs/manuscript/audit lanes and are avoided entirely.
