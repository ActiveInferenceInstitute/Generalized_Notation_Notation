# Troubleshooting Guide

This guide is for the current GNN pipeline. Commands are run from the repository root
and use `uv run` so they target the project environment.

## First response

```bash
# Inspect the actual command surface and environment.
uv run python src/gnn/main.py --help
uv run gnn --help
uv run gnn preflight
uv run gnn health

# Re-run a small, observable path.
uv run python src/gnn/main.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --only-steps "3,5" \
  --verbose
```

Capture the relevant step summary under `output/<step>_output/` and the pipeline
summary under `output/00_pipeline_summary/` when reporting an issue.

## GNN validation errors

A strict GNN file must contain these sections:

```markdown
## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
Example

## StateSpaceBlock
s_f0[2,1,type=float]
o_m0[2,1,type=float]

## Connections
s_f0>o_m0
```

Validate a file directly:

```bash
uv run gnn validate path/to/model.md --strict --json
```

For directory discovery and resource checks:

```bash
uv run python src/gnn/5_type_checker.py \
  --target-dir path/to/model-directory \
  --strict --estimate-resources --verbose
```

The type-checker `--target-dir` is a directory input. A single `.md` path is not a
replacement for that directory argument. The parser discovers Markdown files; use
`.md` files rather than assuming `.gnn` discovery.

## Setup and dependency failures

```bash
uv sync --extra dev
uv run gnn preflight
uv run gnn health
uv run python src/gnn/1_setup.py --dev --verbose
```

To install selected optional groups:

```bash
uv run python src/gnn/1_setup.py \
  --install-optional \
  --optional-groups "audio,gui,graphs"
```

To rebuild the UV-managed environment, use the supported flag:

```bash
uv run python src/gnn/1_setup.py --recreate-uv-env --dev
```

The current flags are hyphenated. `--install_optional`, `--optional_groups`, and
`--recreate-venv` are obsolete spellings.

## Render and execute failures

Render a deliberately small target set first:

```bash
uv run python src/gnn/11_render.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --frameworks "pymdp" \
  --strict-framework-success \
  --verbose
```

Then execute only that isolated render output:

```bash
uv run python src/gnn/12_execute.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --render-output-dir output/11_render_output \
  --frameworks "pymdp" \
  --timeout 600 \
  --verbose
```

Step 12 has no dry-run mode. Use `uv run gnn health` to inspect dependency status.
There is also no `--force-regenerate` render flag; rerun Step 11 into an explicit
output directory when regeneration is required.

### PyMDP

The supported Python package is `inferactively-pymdp`:

```bash
uv run python -c "from pymdp import Agent; print('PyMDP OK')"
```

If this import fails, repair the environment with `uv sync` or follow the setup guide.

### JAX, NumPyro, DisCoPy

These are core Python dependencies in the normal project sync:

```bash
uv run python -c "import jax, numpyro, discopy; print('core backends OK')"
```

### RxInfer.jl and ActiveInference.jl

Use the committed Julia project for the framework being checked:

```bash
julia --startup-file=no --project=src/gnn/execute/rxinfer \
  -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=src/gnn/execute/activeinference_jl \
  -e 'using Pkg; Pkg.instantiate()'
```

### PyTorch and bnlearn

These targets are locked optional extras and are absent from a normal core
installation. Provision them with `uv sync --frozen --extra torch` or
`uv sync --frozen --extra bnlearn`. The locked PyTorch floor is declared in
`pyproject.toml`; do not replace it with a floating manual install. Consult
`src/gnn/render/framework_registry.py` for model admission and native readiness.

## Pipeline control and performance

Use only the supported step controls:

```bash
# Run a focused path.
uv run python src/gnn/main.py --only-steps "3,5,8" --verbose

# Skip expensive or environment-dependent steps.
uv run python src/gnn/main.py --skip-steps "2,13,15" --verbose

# Skip just LLM processing.
uv run python src/gnn/main.py --skip-llm --verbose

# Reduce generated animation artifacts in Step 16.
uv run python src/gnn/main.py --only-steps "16" --no-animations --verbose
```

There is no main-pipeline `--debug`, `--diagnostics`, `--conservative`,
`--conservative-memory`, or `--memory-efficient` option. Replace those imagined
modes with a focused `--only-steps` run, `--verbose`, `--estimate-resources`, and an
explicit `--target-dir`.

## Reading failures

1. Inspect `output/00_pipeline_summary/pipeline_execution_summary.json`.
2. Inspect the step-specific JSON or Markdown summary under `output/<step>_output/`.
3. Check whether a framework is `SKIPPED` because a dependency is unavailable; a skip
   is different from a failed execution.
4. Reproduce with one model directory, one framework, and an isolated output directory.
5. Include the exact command, Python/uv/Julia versions, framework selection, and the
   relevant summary/log files in an issue.

Useful commands:

```bash
cat output/00_pipeline_summary/pipeline_execution_summary.json | uv run python -m json.tool
find output -maxdepth 3 -type f -name '*.log' -o -name '*.json' | sort | head -80
```

## Documentation and code-quality checks

When a failure may be documentation drift, run the same checks used by CI:

```bash
uv run --extra dev python docs/development/docs_audit.py --strict --check-anchors --no-write
uv run --extra dev python scripts/check_doc_contracts.py --strict
uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict
```

See [Setup](../SETUP.md), [Configuration](../configuration/README.md), and the
[framework availability guide](../execution/FRAMEWORK_AVAILABILITY.md) for the
corresponding authoritative references.
