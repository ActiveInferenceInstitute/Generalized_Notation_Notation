# Pipeline Warning Troubleshooting Guide

A pipeline warning means the step produced a result with a caveat. Always inspect the
step summary and distinguish `SKIPPED`, `SUCCESS_WITH_WARNINGS`, and `FAILED` rather
than relying on a fixed framework count.

## First checks

```bash
uv run gnn preflight
uv run gnn health
cat output/00_pipeline_summary/pipeline_execution_summary.json | uv run python -m json.tool
```

The exact output root may be a custom `--output-dir`; use that run directory when it
is not `output/`.

## Why a step warned

Each entry in `steps` carries `warning_reason_source`, written even when no reason
exists, so an empty reason list is never ambiguous:

| `warning_reason_source` | Meaning |
|---|---|
| `step_output_scan` | An actionable warning line was found in the step's captured stdout/stderr; `warning_reasons` lists the distinct lines. |
| `step_declaration` | The step reported its own warnings through `warnings` or `dependency_warnings`. |
| `exit_code_only` | The step returned `SUCCESS_WITH_WARNINGS` and the summary captured no warning text, so no reason can be quoted. Read the step's own log. |
| `none` | No warning was recorded or attributed. |

A line counts as a warning line only when it carries a severity marker: `warning`,
`warn`, `warns` or `warnings` in any case, or `⚠️`, followed by `:`, a space, a tab, `]`,
`)` or the end of the line. A path or identifier that merely contains the word —
`output/warning_smoke/report.json`, for example — is not a warning line, while a
bracketed `[WARN]` is (`WARNING_PATTERN` and `_warning_lines` in
`src/gnn/pipeline/summary_wiring.py`). Safe-noise lines are excluded separately by
`_step_has_actionable_warning`.

`warning_reasons` holds the first 8 distinct reasons in first-seen order, each capped
at 200 characters, with ANSI escapes and leading log timestamps stripped and
whitespace collapsed, so the same warning compares equal across runs.
`warning_reason_count` is the number of distinct reasons found before truncation and
`warning_reasons_truncated` says whether the list was cut. That bound is the honest
limit of the summary: `exit_code_only` means the reason is unavailable there, not
that there was none.

```bash
uv run python - <<'PY'
import json
import pathlib

summary = json.loads(
    pathlib.Path(
        "output/00_pipeline_summary/pipeline_execution_summary.json"
    ).read_text()
)
for step in summary.get("steps", []):
    if step.get("warning_reason_source") not in (None, "none"):
        print(step.get("script_name"), step["warning_reason_source"])
        for reason in step.get("warning_reasons", []):
            print("   ", reason)
PY
```

## Missing framework warnings

Step 12 can skip an unavailable Python or Julia runtime. Check the executor status:

```bash
uv run gnn health
uv run python src/gnn/12_execute.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --render-output-dir output/11_render_output \
  --frameworks "pymdp,jax" \
  --verbose
```

For Julia, instantiate the matching committed environment:

```bash
julia --startup-file=no --project=src/gnn/execute/rxinfer -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=src/gnn/execute/activeinference_jl -e 'using Pkg; Pkg.instantiate()'
```

For optional PyTorch or bnlearn paths, read the availability reason in
`src/gnn/render/framework_registry.py` before installing anything. These are not part of
the default lock.

## Render warnings

Regenerate into an isolated output directory by running Step 11 again; there is no
`--force-regenerate` flag:

```bash
uv run python src/gnn/11_render.py \
  --target-dir input/gnn_files \
  --output-dir output/render-retry \
  --frameworks pymdp \
  --verbose
```

## Test-step warnings or failures

Run the test surface separately for a clearer error:

```bash
uv run --extra dev python -m pytest tests/ -q \
  --ignore=tests/llm/test_llm_ollama.py \
  --ignore=tests/llm/test_llm_ollama_integration.py
```

A focused pipeline can skip Step 2 with `--skip-steps "2"`; this does not replace
running the test suite when validating a change.

## LLM warnings

Step 13 uses the local Ollama configuration in `input/config.yaml` and environment
variables such as `OLLAMA_MODEL`. If no local or cloud provider is available, skip it
explicitly:

```bash
uv run python src/gnn/main.py --skip-llm --verbose
```

## Visualization warnings

Headless Matplotlib warnings are expected in environments without a display when the
artifact itself is produced. Inspect the generated files and step summary. Do not add
an undocumented `--include-d2` flag; D2 support is owned by the visualization module
and its own installed-tool checks.

## Reproduce and report

1. Re-run one model directory, one framework, and one or two steps.
2. Use `--verbose` and an isolated `--output-dir`.
3. Include the exact command and the relevant JSON/log summary.
4. State whether the result was skipped, warning, or failed.

```bash
uv run python src/gnn/main.py \
  --target-dir input/gnn_files/pomdp_gridworld \
  --output-dir output/repro \
  --only-steps "3,5,11,12" \
  --frameworks pymdp \
  --verbose
```
