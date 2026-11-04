# Main Tests — Agent Scaffolding

## Overview

Tests for the `gnn` entry surfaces: the composable step-selection core in
`src/gnn/main.py` (`select_pipeline_steps`, `parse_step_list_strict`,
`step_number_from_script_name`, the `resolve_steps_to_execute` adapter) and
the manuscript-variable round-trip API in `src/gnn/manuscript/variables.py`
(`load_variables`, `token_checksum`) and the render-custody reader contracts in
`src/gnn/manuscript/render_custody.py` (`load_render_manifest`,
`strip_volatile_tokens`).

## Running

```bash
uv run --extra dev python -m pytest tests/main/ -q
```

## Files

- `test_main_step_selection.py` — pure selection contract, lenient
  `parse_step_list` acceptance, fail-fast error paths, log-line preservation.
- `test_manuscript_build_figures.py` — figure-build registry contract for
  `scripts/manuscript_build_figures.py` (registration, registry JSON,
  manuscript references, alt text).
- `test_manuscript_hydration_mirror.py` — the headless injector mirror, pinned
  to the recorded template output in
  `tests/fixtures/manuscript_hydration_9ce91f62a/` (destination link rebase),
  plus the expectation the PR-time hydration guard builds from it.
- `test_manuscript_render_custody.py` — fail-closed `render_custody` reader
  contracts: `load_render_manifest` missing/corrupt/non-object errors, the
  manifest's documented path, and `strip_volatile_tokens` (drops only
  `GNN_GIT_COMMIT`).
- `test_manuscript_variables.py` — producer-drift regression suite for
  `gnn.manuscript.generate_variables` against the live repository.
- `test_manuscript_variables_api.py` — `save_variables` → `load_variables`
  round-trip, validation errors, `token_checksum` stability.
