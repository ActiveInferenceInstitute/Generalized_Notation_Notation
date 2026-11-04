# GNN Quick Start Guide

This is the shortest supported path from a fresh checkout to validating and running a
maintained GNN model. Commands assume the repository root and use `uv` so they run in
the project environment.

## 1. Install the environment

```bash
uv sync --frozen --extra dev --python 3.14
```

For optional framework, audio, GUI, graph, research, or scaling dependencies, see the
[setup guide](SETUP.md). Do not create a second virtual environment under `src/`.

## 2. Check the local runtime

```bash
uv run --frozen --no-sync gnn preflight
uv run --frozen --no-sync gnn health
```

`preflight` checks the repository configuration and available tools. `health` reports
renderer and dependency availability; an unavailable optional backend is not the same
as a successful execution.

## 3. Validate a maintained model

Use a checked-in model first so the command exercises the repository's real parser:

```bash
uv run --frozen --no-sync gnn validate \
  input/gnn_files/discrete/actinf_pomdp_agent.md \
  --strict
```

A new GNN file must contain these enforced sections:

1. `## GNNSection`
2. `## GNNVersionAndFlags`
3. `## ModelName`
4. `## StateSpaceBlock`
5. `## Connections`

`InitialParameterization`, `Equations`, `Time`, ontology annotations, `Footer`, and
`Signature` are consumed by downstream steps even though the section validator does
not require all of them. Use the [syntax reference](gnn/reference/gnn_syntax.md) and
[quickstart tutorial](gnn/tutorials/quickstart_tutorial.md) when authoring a model.

To validate a directory with the numbered type-checker step:

```bash
uv run --frozen --no-sync python src/gnn/5_type_checker.py \
  --target-dir input/gnn_files \
  --strict \
  --verbose
```

## 4. Run a focused pipeline

Start with parsing, type checking, rendering, and execution. Keep outputs isolated so
one run cannot be confused with an earlier run:

```bash
uv run --frozen --no-sync python src/gnn/main.py \
  --target-dir input/gnn_files \
  --output-dir output/quickstart \
  --only-steps "3,5,11,12" \
  --frameworks pymdp \
  --verbose
```

The pipeline loads `input/config.yaml` automatically. Step selection uses
`--only-steps` and `--skip-steps`; there is no generic main-pipeline `--config`,
`--profile`, or `--dry-run` option. See the [configuration guide](configuration/README.md)
for the supported YAML surface.

If an optional runtime is unavailable, inspect the execution summary rather than
counting generated files:

```bash
uv run --frozen --no-sync python -m json.tool \
  output/quickstart/00_pipeline_summary/pipeline_execution_summary.json
```

Step 11's [renderer registry](../src/gnn/render/framework_registry.py) declares
backend and model-kind support. Step 12's direct runners and pipeline discovery
have separate inventories; see [pipeline boundaries](pipeline/README.md#step-boundaries).
Experimental cpomdp and THRML require explicit selection. PyTorch, bnlearn and
Stan need their declared extras; Stan also needs CmdStan. Julia backends use their
committed project environments. See [framework availability](execution/FRAMEWORK_AVAILABILITY.md)
and the [runtime exceptions](../README.md#runtime-defaults-and-required-profiles).

Model kind matters for execution: the PyMDP preset above runs **discrete
categorical** models. A **continuous linear-Gaussian** model (F/H/Q/R with
`prior_mean`/`prior_cov`), such as `input/gnn_files/continuous/continuous_navigation.md`,
renders to native LGSSM programs and executes on jax, numpyro, pytorch, stan,
rxinfer, and ngc-learn — categorical backends report it `unsupported`.

## 5. Inspect generated artifacts

Every run writes below the directory passed with `--output-dir`. Typical locations
include:

```text
output/quickstart/
├── 00_pipeline_summary/
├── 03_gnn_output/
├── 05_type_checker_output/
├── 11_render_output/
└── 12_execute_output/
```

The exact files depend on the selected model, steps, and available frameworks. Treat
rendered code as executable code and review it before running it outside the pipeline's
local safety boundary.

## Author a new model

Copy a maintained template or start from the tutorial's complete example:

```bash
uv run --frozen --no-sync gnn templates list
uv run --frozen --no-sync gnn templates show actinf-pomdp-2state
uv run --frozen --no-sync gnn pull actinf-pomdp-2state --output-dir input/gnn_files/my_models
```

Then validate the copied file with `gnn validate FILE --strict`. Keep model files under
a directory passed to `--target-dir`; directory discovery does not treat a single file
path as a model directory.

## Next references

- [Setup](SETUP.md) — environments, optional groups, and Julia runtimes
- [Configuration](configuration/README.md) — `input/config.yaml` and supported CLI flags
- [GNN tutorial](gnn/tutorials/quickstart_tutorial.md) — a complete model-writing walkthrough
- [Syntax reference](gnn/reference/gnn_syntax.md) — sections, declarations, connections, and parameters
- [Pipeline guide](pipeline/README.md) — step boundaries and framework selection
- [Troubleshooting](troubleshooting/README.md) — warnings, missing runtimes, and failures
- [Documentation hub](README.md) — navigation for theory, integration, and development
