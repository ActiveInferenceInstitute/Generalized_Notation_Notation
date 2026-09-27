# Pipeline Documentation

This is the canonical home for pipeline-specific guidance. The numbered scripts are
thin orchestrators; the authoritative order and metadata live in
`src/gnn/pipeline/step_registry.py`.

## Run a focused pipeline

```bash
uv run python src/gnn/main.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --only-steps "3,5,11,12" \
  --verbose
```

Use `--skip-steps "13,15"` or `--skip-llm` when a local runtime is not available.
The main pipeline loads `input/config.yaml` automatically. See the
[configuration guide](../configuration/README.md) for the supported YAML sections.

## Step boundaries

- Steps 0–10 discover, parse, validate, export, visualize, and annotate models.
- Step 11 has 9 render targets: PyMDP, RxInfer.jl, ActiveInference.jl, JAX,
  DisCoPy, PyTorch, NumPyro, Stan, and bnlearn.
- Step 12 has 10 executor families: PyMDP, JAX, DisCoPy, RxInfer.jl,
  ActiveInference.jl, PyTorch, NumPyro, Lean, Stan, and bnlearn
  (`src/gnn/execute/bnlearn/`). bnlearn programs skip with an install hint
  until their runtime is present (`uv sync --extra bnlearn`, or Rscript plus
  the R `bnlearn` package for `.R` scripts). Stan runs the
  rendered cmdstanpy driver (`src/gnn/execute/stan/`); it needs
  `uv sync --extra stan` plus a CmdStan toolchain, otherwise it is reported
  skipped. Continuous (linear-Gaussian) exemplars execute on jax, numpyro,
  pytorch, stan and rxinfer; the categorical backends report `unsupported` for
  them and are not executed. Lean verifies emitted documents against the
  fep_lean `FEP.GnnDocument` typed surface (requires the sibling fep_lean
  checkout; see `docs/other/fep_lean/README.md`).
- Steps 13–24 provide LLM, ML, audio, analysis, integration, security, research,
  website, MCP, GUI, reporting, and intelligent-analysis surfaces.

A missing optional runtime is represented as skipped/unavailable in the execution
summary. It is not equivalent to a successful execution.

## Framework selection

```bash
# Render only selected targets.
uv run python src/gnn/11_render.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --frameworks "pymdp,jax" \
  --strict-framework-success

# Execute only the selected framework scripts from a known render directory.
uv run python src/gnn/12_execute.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --render-output-dir output/11_render_output \
  --frameworks "pymdp,jax" \
  --timeout 600
```

`all` and `lite` are presets. The exact lists are implemented in
`src/gnn/execute/detection.py::parse_frameworks_parameter` (re-exported through the `src/gnn/execute/processor/` facade); do not infer executor
coverage from the renderer registry.

## Validation and acceptance checks

```bash
# Documentation contracts and link/anchor audit.
uv run --extra dev python docs/development/docs_audit.py --strict --check-anchors --no-write
uv run --extra dev python scripts/check_doc_contracts.py --strict

# Pipeline orchestration acceptance.
PYTHONPATH=src uv run --extra dev python scripts/run_v3_orchestration_acceptance.py --strict
```

## Related references

- [Pipeline scripts](../PIPELINE_SCRIPTS.md)
- [Source step index](../../src/gnn/STEP_INDEX.md)
- [Source pipeline README](../../src/gnn/README.md)
- [Framework integration guide](../gnn/integration/framework_integration_guide.md)
- [Setup](../SETUP.md)
- [Troubleshooting](../troubleshooting/README.md)
