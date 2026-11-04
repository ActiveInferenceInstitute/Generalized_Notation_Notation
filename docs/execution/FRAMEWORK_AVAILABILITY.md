# Framework Availability Guide

Backend inventories have separate owners:

- The [renderer registry](../../src/gnn/render/framework_registry.py) declares
  code generators, model-kind support and default/experimental selection.
- [Executor specifications](../../src/gnn/execute/executor_specs.py) declare direct
  runner wiring. [Pipeline discovery](../../src/gnn/execute/detection.py) detects
  generated scripts, including cpomdp, and resolves execution presets.
- Lean is a document-checking route against the pinned companion; it is separate
  from numerical simulation. See [FEP bridge guidance](../other/fep_lean/README.md).

`--frameworks all` selects the default preset. Experimental cpomdp and THRML
require explicit selection and their documented admission/resource contracts.

PyTorch, bnlearn and other optional backends need their declared extras. Stan uses
its cmdstanpy driver plus CmdStan; bnlearn uses the Python package or Rscript plus
R's bnlearn package for `.R` files. Missing runtimes produce skipped/unavailable
outcomes. Julia targets require their committed project environments. See the
[setup guide](../../SETUP_GUIDE.md) for installation and
[runtime exceptions](../../README.md#runtime-defaults-and-required-profiles)
for the locked Python/Gradio/Pillow compatibility profiles.

## Model kinds and execution paths

`render.pomdp_contract.detect_model_kind` classifies each model as either
**discrete categorical** (A/B/C/D[/E]) or **continuous linear-Gaussian**
(F/H/Q/R with `prior_mean`/`prior_cov`). The kind decides which backends
apply:

- **Discrete categorical** models execute on PyMDP, RxInfer.jl,
  ActiveInference.jl, JAX (factorized Kronecker lane), NumPyro, PyTorch,
  DisCoPy, bnlearn, and Stan. Stan executes discrete models as HMM programs
  via the cmdstanpy driver (`src/gnn/execute/stan/`).
- **Continuous linear-Gaussian** models render to native LGSSM programs via
  `src/gnn/render/continuous_script.py` and execute on JAX, NumPyro, PyTorch,
  Stan (LGSSM), RxInfer.jl, and ngc-learn. Categorical backends that cannot
  express them (PyMDP, ActiveInference.jl, DisCoPy, bnlearn) report an explicit
  `unsupported` render status. ngc-learn is continuous-only; explicitly
  selected cpomdp has bounded continuous filtering/control admission.

Step 12 admits rendered scripts through the current-run selection and artifact
indexes. An `unsupported` render is never executed for that model. Inspect both
render and execution summaries for admission, missing-runtime and failure reasons;
an empty directory cannot establish either success or failure. See the
[current-run migration guide](../development/run_ownership_migration.md).

## Check availability

Use the unified CLI before a run:

```bash
uv run --frozen --no-sync gnn health
uv run --frozen --no-sync gnn preflight
```

For a direct Python status report:

```bash
PYTHONPATH=src uv run --frozen --no-sync python - <<'PY'
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
uv run --frozen --no-sync python -c "from pymdp import Agent; print('PyMDP available')"
uv run --frozen --no-sync python -c "import jax, numpyro, discopy; print('JAX, NumPyro, and DisCoPy available')"
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
uv run --frozen --no-sync python -c "import torch; print('PyTorch available')"
uv run --frozen --no-sync python -c "import bnlearn; print('bnlearn available')"  # or: uv sync --extra bnlearn
```

For `.R` bnlearn scripts, the R lane needs `Rscript` plus the R `bnlearn`
package (`install.packages('bnlearn')`); probe it with
`Rscript -e 'library(bnlearn)'`.

The matching `--project` is required. The executor uses the same project-specific
environments when launching rendered scripts.

## Selection examples

```bash
# Python-only quick preset.
uv run --frozen --no-sync python src/gnn/12_execute.py \
  --target-dir input/gnn_files \
  --output-dir output/execution-example \
  --frameworks lite \
  --verbose

# Explicit requested frameworks. Missing requested frameworks are reported clearly.
uv run --frozen --no-sync python src/gnn/12_execute.py \
  --target-dir input/gnn_files \
  --output-dir output/execution-example \
  --render-output-dir output/execution-example/11_render_output \
  --frameworks "pymdp,jax" \
  --verbose
```

The executor has no `--dry-run` flag. Use `gnn health` and `gnn preflight` for
non-execution checks.

## Interpret the result

Inspect the selected run’s `12_execute_output/` and pipeline summary. Distinguish:

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
