# Pipeline Documentation

This is the canonical home for pipeline-specific guidance. The numbered scripts are
thin orchestrators; the authoritative order and metadata live in
`src/gnn/pipeline/step_registry.py`.

## Run a focused pipeline

After [frozen development setup](../../README.md#quick-start), run from the repository root:

```bash
uv run --frozen --no-sync python src/gnn/main.py \
  --target-dir input/gnn_files \
  --output-dir output/pipeline-example \
  --only-steps "3,5,11,12" \
  --verbose
```

Use `--skip-steps "13,15"` or `--skip-llm` when a local runtime is not available.
The main pipeline loads `input/config.yaml` automatically. See the
[configuration guide](../configuration/README.md) for the supported YAML sections.

## Step boundaries

- Steps 0–10 discover, parse, validate, export, visualize, and annotate models.
- Step 11's [renderer registry](../../src/gnn/render/framework_registry.py)
  declares targets, model-kind support and default/experimental selection.
- Step 12's [executor specifications](../../src/gnn/execute/executor_specs.py)
  own direct framework runners; [pipeline discovery](../../src/gnn/execute/detection.py)
  owns script detection and presets, including cpomdp and the Lean document route.
  These are distinct inventories; generator registration alone does not establish
  native execution or scientific acceptance.
- bnlearn execution (`src/gnn/execute/bnlearn/`) needs the `bnlearn` extra or
  Rscript and the R package for `.R` scripts. Stan executes its cmdstanpy driver
  with the `stan` extra and CmdStan. Julia backends use their committed project
  environments. Missing runtimes retain skipped/unavailable outcomes.
- Continuous linear-Gaussian admission is backend-specific. JAX, NumPyro,
  PyTorch, Stan and RxInfer have native paths; ngc-learn is continuous-only.
  Experimental cpomdp and THRML require explicit selection. See the
  [availability guide](../execution/FRAMEWORK_AVAILABILITY.md) for scope and setup.
- Lean checks emitted documents against the companion's `FEP.GnnDocument`
  typed surface; see [FEP bridge guidance](../other/fep_lean/README.md).
- Steps 13–24 provide LLM, ML, audio, analysis, integration, security, research,
  website, MCP, GUI, reporting, and intelligent-analysis surfaces.

A missing optional runtime is represented as skipped/unavailable in the execution
summary. It is not equivalent to a successful execution.

## Framework selection

```bash
# Render only selected targets.
uv run --frozen --no-sync python src/gnn/11_render.py \
  --target-dir input/gnn_files \
  --output-dir output/pipeline-example \
  --frameworks "pymdp,jax" \
  --strict-framework-success

# Execute only the selected framework scripts from a known render directory.
uv run --frozen --no-sync python src/gnn/12_execute.py \
  --target-dir input/gnn_files \
  --output-dir output/pipeline-example \
  --render-output-dir output/pipeline-example/11_render_output \
  --frameworks "pymdp,jax" \
  --timeout 600
```

`all` and `lite` are presets. The exact lists are implemented in
`src/gnn/execute/detection.py::parse_frameworks_parameter` (re-exported through the `src/gnn/execute/processor/` facade); do not infer executor
coverage from the renderer registry.

## Validation and acceptance checks

```bash
# Documentation contracts and link/anchor audit.
uv run --frozen --no-sync python docs/development/docs_audit.py --strict --check-anchors --no-write
uv run --frozen --no-sync python scripts/check_doc_contracts.py --strict

# Pipeline orchestration acceptance.
PYTHONPATH=src uv run --frozen --no-sync python scripts/run_v3_orchestration_acceptance.py --strict
```

## Related references

- [Pipeline scripts](../PIPELINE_SCRIPTS.md)
- [Source step index](../../src/gnn/STEP_INDEX.md)
- [Source pipeline README](../../src/gnn/README.md)
- [Framework integration guide](../gnn/integration/framework_integration_guide.md)
- [Setup](../SETUP.md)
- [Troubleshooting](../troubleshooting/README.md)

## Documents in this folder

- [v3.0.0 Long-Running Orchestration](v3_orchestration.md)
- [Pipeline Stage Hardening Review](pipeline_stage_hardening_review.md)
- [Validation Evidence Guide](validation_evidence_guide.md)
