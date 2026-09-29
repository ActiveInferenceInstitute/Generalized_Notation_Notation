# Manuscript Module Specification

## Overview
Deterministic `{{...}}` token production for the manuscript, plus the render custody manifest binding the committed PDF evidence to the token map that produced it. Custody-adjacent: this package records and audits manifests; it never runs renders.

## Components
- `variables.py` - The token producer: `RepositorySnapshot` (git `ls-tree`/`cat-file` view of one commit), `generate_variables(project_root) -> dict[str, str]` (flat `{{UPPERCASE_KEY}}` map consumed by the template's manuscript renderer), `save_variables`/`load_variables` (sorted, deterministic JSON), `token_checksum` (sha256 over canonical JSON), `config_metadata_drift`/`sync_config_metadata` and `preamble_metadata_drift`/`sync_preamble_metadata` (producer-owned `manuscript/config.yaml` and `preamble.md` fields), `select_cross_framework_family`
- `render_custody.py` - Custody manifest (`MANIFEST_VERSION = "gnn_render_manifest_v1"`): `record_render_manifest` digests committed artifacts + hydrated inputs into `output/data/manuscript_render_manifest.json`; `custody_issues` audits the committed chain; `verify_fresh_render` classifies a fresh render (`[FAIL]` = artifact missing, or artifact and render inputs drifted jointly — chain stale for HEAD; `[WARN]` = artifact drift with matching inputs — likely toolchain variance); `RECORD_COMMAND = "uv run python scripts/z_record_manuscript_render_manifest.py"`; `RENDERED_ARTIFACTS` = the committed `output/pdf/_combined_manuscript.{log,tex,md}` evidence; `mask_log_timestamp` masks the TeX log banner's start time in the fresh comparison; `hydration_issues` (PR-time guard, `scripts/check_hydrated_prose.py`) re-substitutes `manuscript/*.md` from the committed token map via `hydrate_text` (mirror of the template injector) and reports every committed `output/manuscript/` file that differs, with `HYDRATION_FIX` as the remedy
- `__init__.py` - Thin re-export surface (`__all__`, ten names)

## Contract
- Nothing is hard-coded: every quantitative token is computed from a repository source surface (`pyproject.toml`, `input/model_family_manifest.json`, the framework registry, `src/gnn/STEP_INDEX.md`, `CHANGELOG.md`, filesystem counts, the exemplar corpus).
- Counts describe one commit: sources are read from the commit named by `GNN_GIT_COMMIT` (HEAD) via `git ls-tree`/`git cat-file`; without git the snapshot falls back to the working tree and reports the `unknown` sentinel, on which the token gate (`scripts/check_manuscript_tokens.py`) and the figure build fail.
- Deterministic: no timestamps or wall-clock; two runs over an unchanged commit produce byte-identical JSON.
- Dependency-light: standard library plus optional `yaml`; no `gnn.*` imports, so the producer stays headless-importable.
- Custody chain: `HEAD → token map → hydrated prose (output/manuscript/) → committed PDF evidence`. The record step runs after a render (`RECORD_COMMAND`); the manual half of the SC-22 ordering ritual (regen → rebuild figures → render → record → commit) is stated in `scripts/z_generate_manuscript_variables.py`, whose thin orchestration wires `generate_variables` to the template's hydration.

## Key Exports
```python
from gnn.manuscript import (
    RepositorySnapshot,
    config_metadata_drift,
    generate_variables,
    load_variables,
    preamble_metadata_drift,
    save_variables,
    select_cross_framework_family,
    sync_config_metadata,
    sync_preamble_metadata,
    token_checksum,
)
```

## Receipts
```bash
uv run --extra dev python -m pytest tests/main/test_manuscript_variables.py \
  tests/main/test_manuscript_variables_api.py tests/main/test_manuscript_build_figures.py \
  tests/test_manuscript_token_gate.py tests/test_manuscript_latex_log.py \
  tests/test_manuscript_figure_freshness.py tests/test_manuscript_path_claims.py -q
```

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
