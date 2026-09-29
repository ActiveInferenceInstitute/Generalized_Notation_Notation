# Framework Availability Guide

GNN has two related framework inventories:

- **Render registry** (`src/gnn/render/framework_registry.py`): 9 targets, including Stan.
- **Step 12 executor** (`src/gnn/execute/processor.py` + the per-framework runner
  packages): 10 executable framework families — every render target (bnlearn via
  `src/gnn/execute/bnlearn/`). Stan executes via the cmdstanpy driver
  (`src/gnn/execute/stan/`) and is reported `skipped` when cmdstanpy/CmdStan is absent;
  bnlearn programs are reported `skipped` with an install hint until their runtime is
  present (`uv sync --extra bnlearn`, or Rscript plus the R `bnlearn` package for
  `.R` scripts).

PyTorch and bnlearn are supported render/execute paths; both are intentionally absent
from the default lock (heavy optional runtimes). Julia targets require
their committed project environments.

## Model kinds and execution paths

`render.pomdp_contract.detect_model_kind` classifies each model as either
**discrete categorical** (A/B/C/D[/E]) or **continuous linear-Gaussian**
(F/H/Q/R with `prior_mean`/`prior_cov`). The kind decides which backends
apply:

- **Discrete categorical** models execute on PyMDP, RxInfer.jl,
  ActiveInference.jl, JAX (factorized Kronecker lane), NumPyro, PyTorch, and
  DisCoPy. Stan executes discrete models as HMM programs via the cmdstanpy
  driver (`src/gnn/execute/stan/`).
- **Continuous linear-Gaussian** models render to native LGSSM programs via
  `src/gnn/render/continuous_script.py` and execute on JAX, NumPyro, PyTorch,
  Stan (LGSSM), RxInfer.jl 5.5, and cpomdp (its own template, `src/gnn/render/cpomdp/`). Categorical backends that cannot express
  them (PyMDP, ActiveInference.jl, DisCoPy, bnlearn) report an explicit
  `unsupported` render status rather than failing silently.

Step 12 executes whatever rendered scripts exist for the requested
frameworks; it does not filter by model kind. When a framework produced only
`unsupported` statuses for a continuous model, the per-framework execution
summary simply has nothing to run — check the render summary before
interpreting an empty result as an executor failure.

## Check availability

Use the unified CLI before a run:

```bash
uv run gnn health
uv run gnn preflight
```

For a direct Python status report:

```bash
PYTHONPATH=src uv run python - <<'PY'
from gnn.execute import collect_doctor_report

report = collect_doctor_report()
for name in report["frameworks_available"]:
    print(f"{name}: available")
for name in report["frameworks_missing"]:
    print(f"{name}: unavailable")
PY
```

## Runtime checks

### Core Python targets

```bash
uv run python -c "from pymdp import Agent; print('PyMDP available')"
uv run python -c "import jax, numpyro, discopy; print('JAX, NumPyro, and DisCoPy available')"
```

### Julia targets

```bash
julia --startup-file=no --project=src/gnn/execute/rxinfer \
  -e 'using RxInfer; println("RxInfer.jl available")'
julia --startup-file=no --project=src/gnn/execute/activeinference_jl \
  -e 'using ActiveInference; println("ActiveInference.jl available")'
```

### Optional Python targets

```bash
uv run python -c "import torch; print('PyTorch available')"
uv run python -c "import bnlearn; print('bnlearn available')"  # or: uv sync --extra bnlearn
```

For `.R` bnlearn scripts, the R lane needs `Rscript` plus the R `bnlearn`
package (`install.packages('bnlearn')`); probe it with
`Rscript -e 'library(bnlearn)'`.

The matching `--project` is required. The executor uses the same project-specific
environments when launching rendered scripts.

## Selection examples

```bash
# Python-only quick preset.
uv run python src/gnn/12_execute.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --frameworks lite \
  --verbose

# Explicit requested frameworks. Missing requested frameworks are reported clearly.
uv run python src/gnn/12_execute.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --render-output-dir output/11_render_output \
  --frameworks "pymdp,jax" \
  --verbose
```

The executor has no `--dry-run` flag. Use `gnn health` and `gnn preflight` for
non-execution checks.

## Interpret the result

Inspect `output/12_execute_output/` and the pipeline summary. Distinguish:

- **Succeeded**: a rendered script ran and returned a successful result.
- **Skipped/unavailable**: a dependency or runtime was not present.
- **Failed**: an available/requested script ran and returned an error.

Do not report a fixed `N/M` success count in documentation. Counts depend on the
input corpus, selected frameworks, and local runtimes; use the generated execution
summary for a specific run.

## Related references

- [Setup](../SETUP.md)
- [Pipeline](../pipeline/README.md)
- [Render registry](../../src/gnn/render/framework_registry.py)
- [Execute module](../../src/gnn/execute/AGENTS.md)
- [Troubleshooting](../troubleshooting/README.md)
