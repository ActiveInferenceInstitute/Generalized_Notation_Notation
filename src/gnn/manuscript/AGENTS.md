# gnn.manuscript — Manuscript Token Production

## Overview

**Purpose:** Deterministic production of the manuscript's `{{...}}` token
map: reads the repository snapshot at the current commit, computes every
count the manuscript cites, persists
`output/data/manuscript_variables.json` for audit, and supports config
metadata synchronization and cross-framework family selection.

**Pipeline step:** Manuscript tooling (feeds the render pipeline and the
token checker).

**Category:** Manuscript infrastructure.

**Status:** Maintained.

**Version:** 1.0.0

## Core Functionality

1. `generate_variables()` — compute the full token map from the repository
   snapshot at the current commit.
2. `RepositorySnapshot` — git-snapshot reader backing every count.
3. `load_variables()` / `save_variables()` — read/write the persisted map.
4. `sync_config_metadata()` / `config_metadata_drift()` — keep
   `manuscript/config.yaml` producer-owned fields in step.
5. `select_cross_framework_family()` — family selection for the
   cross-framework coverage note.
6. `token_checksum()` — integrity checksum over the emitted map.
7. `record_render_manifest()` / `custody_issues()` / `verify_fresh_render()`
   — render custody (`render_custody.py`): digest the committed render
   artifacts + hydrated inputs into `output/data/manuscript_render_manifest.json`
   after a render, audit the committed chain against it, and classify a
   fresh render against the committed manifest (`[FAIL]` = artifact missing
   or artifact+input joint drift → chain stale for HEAD; `[WARN]` =
   artifact-only drift with inputs matching → toolchain variance). The
   fresh comparison masks commit stamps and the TeX log banner's start
   timestamp (`mask_log_timestamp()`), so an identical re-render is clean.
8. `hydration_issues()` / `hydrate_text()` — the PR-time custody drift
   guard (`scripts/check_hydrated_prose.py`, `local-gates.yml` repo-gates):
   re-substitute `manuscript/*.md` from the committed token map, mirroring
   the template injector, and report every committed `output/manuscript/`
   file that differs. It needs no LaTeX. A count-changing PR that skips the
   full SC-22 ritual fails here instead of in the scheduled
   `custody-re-render.yml` after merge.

## Module Structure

- `__init__.py` — thin re-export surface (`__all__`).
- `variables.py` — public entry point: token-map assembly
  (`generate_variables`), the producer-owned config/preamble metadata
  writers, and the JSON round-trip helpers; re-exports the full public
  surface so every consumer import path is unchanged.
- `snapshot.py` — `RepositorySnapshot`: the git `ls-tree`/`cat-file` view
  of one commit that every count reads through.
- `sources.py` — per-surface census/derivation: steps, source counts, the
  framework registry, the MCP audit ledger, model families, the exemplar
  corpus, release metadata, and the generated coverage sentences.
- `tables.py` — multi-line token renderers (step/family/backend/capability
  tables) and cross-framework family selection.
- `render_custody.py` — render custody manifest: record after a render,
  audit the committed chain, verify a fresh render against the committed
  manifest, and check the hydrated prose against the token map at PR time
  (SC-22).

## Dependencies

`substitution.py` is the shared grammar/exclusion/injection contract used by the
token and hydration gates. Prefer the template injector when importable; only
template absence permits the headless fallback. Active preamble checks strip
comments and fences before checking declarations.

`gate_baseline.py` reads immutable evidence at a resolved PR merge base without
checking out source. Both sides run the HEAD audit implementation. Only an
unchanged diagnostic with identical supporting bytes may warn, naming the full
base SHA; new/worse drift and unavailable history fail. Main and scheduled gates
remain strict. If main is stale, fix main first.

The hydration guard also requires rendered Markdown and TeX to carry the token
map's commit stamp. This closes the prose-only bypass while retaining byte and
digest comparisons. The stamp does not independently prove renderer execution.

- Stdlib (`ast`, `hashlib`, `json`, `re`, `subprocess`, `fnmatch`,
  `pathlib`, `tomllib`) plus `yaml` (optional). Internal manuscript modules
  stay headless; importing them does not start pipeline or backend runtimes.

## Testing

```bash
uv run --extra dev python -m pytest tests/main/test_manuscript_variables.py tests/main/test_manuscript_variables_api.py -q
uv run --extra dev python -m pytest tests/test_manuscript_latex_log.py -q
uv run --extra dev python -m pytest tests/main/test_manuscript_gate_baseline.py -q
```
