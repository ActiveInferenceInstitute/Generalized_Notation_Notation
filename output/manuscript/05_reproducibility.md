# Reproducibility {#sec:reproducibility}

Reproducibility in GNN is a source- and run-binding contract across the 0–24 processing steps. Fixed-source parsing and token production are distinct from stochastic inference, dependency provisioning, and environment-dependent execution. Reproducing a reported outcome requires the selected inputs, configuration, source revision, dependencies, dtype, seed, and retained command and artifact receipts. The commands below reproduce the pipeline, the validation gates, and the manuscript from clean checkouts of GNN and its render template.

## Pipeline Smoke Run

A bounded smoke run parses and validates the discrete examples. From the GNN checkout, install the locked development environment and select the parsing, type-checking, and validation stages:

```bash
uv sync --frozen --extra dev
uv run --frozen --no-sync python src/gnn/main.py \
  --target-dir input/gnn_files/discrete \
  --output-dir "$(mktemp -d)" --only-steps 3,5,6 --skip-llm
```

For pipeline-wide coverage, omit the step selection. The broader command includes setup, tests, and any admitted rendering and execution:

```bash
uv run --frozen --no-sync python src/gnn/main.py \
  --target-dir input/gnn_files/discrete \
  --output-dir /tmp/gnn-smoke \
  --skip-llm
```

The pipeline writes artifacts under the chosen output directory. The `--skip-llm` flag skips LLM processing and selects rule-based intelligent analysis. Setup and optional backend tools can still use the network. Default framework selection excludes experimental targets. Dependency readiness, model-kind admission, configured stages, and budgets determine the work performed. Inspect the current-run statuses and terminal exit code (0 success, 1 error, 2 warning). To exercise the registered families, use the manifest-driven acceptance gate below. Pointing `--target-dir` at `input/gnn_files` selects that tree's 12 corpus directories. All 9 registered family target directories lie inside that tree, so a single invocation reaches every registered family.

The discrete family exercises the categorical kind end to end. The continuous linear-Gaussian kind smoke-runs the same way, and the contrast between the two runs is itself a check of the per-kind contract:

```bash
uv run --frozen --no-sync python src/gnn/main.py \
  --target-dir input/gnn_files/continuous \
  --output-dir /tmp/gnn-smoke-continuous \
  --skip-llm
```

This selects the continuous specifications for the same pipeline. Admitted, ready continuous-capable lanes can render and execute filtering (and, for the closed-loop exemplar, belief-steering) programs; categorical-only targets record `unsupported`, while readiness and budget failures retain their own outcomes. A reader comparing the two receipts sees the kind taxonomy behaving as described in [@sec:system_context]: same pipeline, same steps, per-kind rendering and execution reach.

## Validation Gates

GNN provides strict gates with distinct verification scopes. The model-family acceptance gate runs the selected manifest families and evaluates their declared acceptance profiles; its receipt must record actual execution, unsupported and skipped work, and failures:

```bash
uv run python scripts/run_model_family_acceptance.py \
  --manifest input/model_family_manifest.json \
  --output-dir output/model_family_acceptance --strict
```

The semantic-fidelity gate verifies that a parse → serialize → parse round trip preserves variables, edges, dimensions, parameter shapes and values, parameter metadata, equations, time semantics, and ontology mappings across the 9 model families; the cross-framework gate profiles the 7 maintained backends (PyMDP, RxInfer.jl, JAX, NumPyro, PyTorch, ActiveInference.jl, DisCoPy) — refusing any framework outside that set — and records explicit compatible and unsupported statuses rather than silently degrading. Both write their ledgers to an output directory of your choosing:

```bash
uv run python scripts/run_semantic_fidelity_gate.py \
  --manifest input/model_family_manifest.json \
  --output-dir output/semantic_fidelity --strict
uv run python scripts/run_cross_framework_reliability.py \
  --manifest input/model_family_manifest.json \
  --output-dir output/cross_framework --strict
```

Under `--strict`, required mismatches make these gates exit non-zero, so these commands double as assertions in an automated reproduction run. Code-quality reproducibility is enforced separately through the developer command reference: `just lint` runs the Ruff linter over `src` and `scripts`, and the broader `just quality` recipe chains formatting, terminology, documentation, type, and security checks for a full pre-commit gate.

## Manuscript Reproducibility

The producer derives repository counts and capability claims from tracked files at the current commit. It writes the token map and table claim ledger under `output/data/`. The template invokes that producer and hydrates the manuscript into `output/manuscript/`. From the GNN checkout, check the existing publication custody before rebuilding and regenerate the figures:

```bash
uv run --frozen --no-sync python scripts/z_verify_fresh_render.py --committed-only
uv run --frozen --no-sync python -m scripts.manuscript_build_figures
```

Use the template revision declared in `output/data/manuscript_render_manifest.json` to reproduce the publication. From a separate checkout of the template containing this repository as a real directory at `projects/active/GeneralizedNotationNotation`, export the measured template revision and build epoch and render the PDF, HTML, and slide decks:

```bash
export GNN_TEMPLATE_REVISION="$(git rev-parse HEAD)"
export SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)"
uv sync --frozen
uv run --frozen --no-sync python scripts/pipeline/stage_03_render.py \
  --project GeneralizedNotationNotation
```

From the GNN checkout, prepare the rendered evidence by removing personal path prefixes:

```bash
uv run --frozen --no-sync python scripts/z_record_manuscript_render_manifest.py --prepare
```

From the same template root, validate the publication and write the Stage 04 reports:

```bash
uv run --frozen --no-sync python scripts/pipeline/stage_04_validate.py \
  --project GeneralizedNotationNotation
```

The validation stage checks the rendered outputs and their evidence, including the generated tables and their claims. From the GNN checkout, compare the fresh evidence with the existing custody manifest:

```bash
uv run --frozen --no-sync python scripts/z_verify_fresh_render.py
```

For an intentional source or template update, run the rendering and validation commands and explicitly record the renewed custody manifest from the project checkout:

```bash
uv run --frozen --no-sync python scripts/z_record_manuscript_render_manifest.py
uv run --frozen --no-sync python scripts/z_verify_fresh_render.py --committed-only
```

Commit the manifest with the new publication and Stage 04 evidence. The scheduled custody check leaves the baseline manifest intact.

The render needs a LaTeX installation providing the packages listed in `docs/manuscript/preamble.md` plus `seqsplit`; the template guards `seqsplit` with `\IfFileExists`, so a missing copy degrades rather than failing the build.

Because the variables file is regenerated before rendering, the counts in the rendered PDF track the repository state at the commit recorded in `output/data/manuscript_variables.json` (49fd25df3): a code change that alters, for example, the test inventory (709 test files, 7508 test functions) propagates into the prose on the next regeneration without any manual editing.

## Reproducibility Contract

- Do not cite results that cannot be regenerated or directly traced to a command in this repository.
- Keep generated outputs under `output/` and maintained manuscript source under `docs/manuscript/`. Regenerate owned build artifacts when their inputs change, and preserve the raw command receipts and numerical results needed to substantiate historical runs.
- Express every quantitative claim in the prose as a double-brace `{{...}}` token substituted by `scripts/z_generate_manuscript_variables.py`, never as a hard-coded number.
- Keep private data, credentials, and unpublished sensitive details out of the manuscript and out of version control.
- Record the exact verification commands — the smoke run, the acceptance and fidelity gates, and `just lint` — before marking this manuscript publication-ready.
