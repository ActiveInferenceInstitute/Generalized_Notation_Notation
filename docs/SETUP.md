# GNN Project Setup Guide

This guide assumes commands are run from the repository root. The supported package
manager is `uv`; use `uv run` for Python commands so the project environment is used.
Use the committed lock for reproducible work. Required and optional dependencies,
interpreter markers and platform constraints are declared in `pyproject.toml` and
`uv.lock`; installation is separate from native backend acceptance.

## Fast path

```bash
UV_PYTHON=3.14 uv sync --frozen --extra dev
uv run --frozen --no-sync gnn health
```

`health` returns 2 for degraded readiness, including unprovisioned optional
tools. Inspect its diagnoses; generator imports do not certify native execution.

Then validate a model or run a focused pipeline path:

```bash
uv run --frozen --no-sync gnn validate input/gnn_files/discrete/two_state_bistable.md
uv run --frozen --no-sync python src/gnn/main.py \
  --target-dir input/gnn_files/basics \
  --output-dir /tmp/gnn-setup-example \
  --only-steps "3,5" \
  --verbose
```

The pipeline returns 2 for completed work with warnings and 1 for failure.
Inspect the summary and step outcomes rather than discarding a nonzero status.

The checked-in `input/config.yaml` is loaded automatically by `src/gnn/main.py`; see the
[configuration guide](configuration/README.md) for its supported sections.

## Requirements

- Python `>=3.11,<3.15` is the declared package range. See the current
  [CI workflow](../.github/workflows/ci.yml) for tested interpreter splits;
  a compatible dependency install does not establish every native backend's
  readiness on that interpreter or OS.
- `uv >=0.13.6,<0.14` is the declared tool range. CI pins 0.13.6.
- Installed-wheel acceptance covers declared Linux, macOS and Windows lanes; see
  the [platform guide](development/installed_platform_acceptance.md) for exact
  interpreter/OS scope. Optional Julia, GUI, audio and Graphviz paths need their
  own provisioning and acceptance.
- Allow space for the interpreter, locked dependencies, caches and generated
  artifacts. Optional backends and manuscript rendering add substantial storage.

Optional system tools:

- Julia for RxInfer.jl and ActiveInference.jl.
- Graphviz for graph layouts that invoke the Graphviz executable.
- A local Ollama installation for the default Step 13 local-LLM path.

## Installation

### 1. Clone and enter the repository

```bash
git clone https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation.git
cd Generalized_Notation_Notation
# Select the published release for a versioned install.
git checkout v5.0.3
```

### 2. Install the project environment

```bash
# Runtime only.
uv sync --frozen --no-dev --python 3.14

# Runtime and the default development, test, lint, and documentation group.
uv sync --frozen --python 3.14

# Every declared optional group, using the Python 3.12 profile.
uv sync --frozen --all-extras --python 3.12
```

The GUI and all-extras profiles require uv's committed Pillow 13 override while
Gradio 6.30.0 still declares Pillow `<13`. Plain pip cannot resolve those extras;
core pip installs retain the secure Pillow minimum. See the
[runtime exceptions and removal checks](../README.md#runtime-defaults-and-required-profiles)
for compatibility scope and the required native GUI/image checks.

Do not create a second `src/.venv`; the project environment belongs at the repository
root and is managed by `uv`.

### 3. Run the setup step when needed

Step 1 wraps environment checks and optional-group installation:

```bash
uv run python src/gnn/1_setup.py --dev --verbose
uv run python src/gnn/1_setup.py --install-all-extras
uv run python src/gnn/1_setup.py \
  --install-optional \
  --optional-groups "audio,gui,graphs"
```

To recreate the UV environment through the supported flag:

```bash
uv run python src/gnn/1_setup.py --recreate-uv-env --dev
```

The equivalent main-pipeline flag is `--recreate-uv-env`. The old
`--recreate-venv`, `--install_optional`, and `--optional_groups` spellings are not
supported.

## Dependency groups

The declared groups are visible in `pyproject.toml`:

| Group | Intended surface |
|---|---|
| `dev` | CmdStanPy extra. The separate default `dev` dependency group supplies test, lint, typing, and docs tools. |
| `api` | FastAPI and Uvicorn |
| `audio` | Librosa, SoundFile, and Pedalboard |
| `gui` | Gradio. Use Python 3.12 for interactive callbacks. |
| `graphs` | Python Graphviz bindings |
| `ml-ai` | SciPy, scikit-learn, and Transformers |
| `research` | Jupyter, SymPy, Numba, and research utilities |
| `scaling` | Dask, Distributed, and Ray |
| `stan` | cmdstanpy driver for rendered Stan programs (needs a local CmdStan toolchain) |
| `torch` | PyTorch backend |
| `bnlearn` | bnlearn runtime for rendered `.py`/`.R` Bayesian network programs |
| `geo-infer` | H3 geospatial indexing used by GEO_INFER integration surfaces |
| `all` | The optional groups above combined as a manually maintained extra |

Install a group directly with `uv sync --frozen --extra GROUP`; use `--all-extras` for all
optional groups.

## Framework Selection Strategies

Use the `--frameworks` option on Step 11 and Step 12 to select `all`, `lite`, or a
comma-separated list. The same selection is available through `src/gnn/main.py`.

## Framework boundaries

The [renderer registry](../src/gnn/render/framework_registry.py) owns generators
and model-kind support. The [direct executor specifications](../src/gnn/execute/executor_specs.py)
and [pipeline discovery](../src/gnn/execute/detection.py) own execution wiring and
presets. See [framework availability](execution/FRAMEWORK_AVAILABILITY.md) for
admission and readiness; registration alone does not certify native execution.

Stan needs the `stan` extra plus CmdStan. bnlearn needs its Python extra or
Rscript and R's bnlearn package for `.R` programs. Julia targets use their committed
project environments. Experimental cpomdp and THRML require explicit selection;
ngc-learn has a continuous-only contract. Other composition and model-kind
restrictions remain adapter-specific. Missing runtimes retain skipped/unavailable
outcomes, and unsupported renders are never executed for that model.

### Python targets

The normal sync installs the core Python targets:

```bash
uv run python -c "import jax, numpyro, discopy; print('JAX, NumPyro, and DisCoPy OK')"
uv run python -c "from pymdp import Agent; print('PyMDP OK')"
```

For PyTorch or bnlearn, install their declared extra and consult
`src/gnn/render/framework_registry.py` for model-kind admission and readiness.
They are optional dependencies; a rendered script is not evidence of native
execution or a successful current run.

### Julia targets

RxInfer.jl uses the committed environment:

```bash
julia --startup-file=no --project=src/gnn/execute/rxinfer \
  -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=src/gnn/execute/rxinfer \
  -e 'using RxInfer; println(pkgversion(RxInfer))'
```

ActiveInference.jl uses its committed environment:

```bash
julia --startup-file=no --project=src/gnn/execute/activeinference_jl \
  -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=src/gnn/execute/activeinference_jl \
  -e 'using ActiveInference; println("ActiveInference.jl OK")'
```

Always pass the matching `--project` when checking or running generated Julia code.
Do not use a bare global `Pkg.add` command as the normal setup path.

## Common setup commands

```bash
# Check renderer and dependency status.
uv run gnn health
uv run gnn preflight

# Inspect all CLI surfaces.
uv run python src/gnn/main.py --help
uv run gnn --help

# Run tests.
uv run --extra dev python -m pytest tests/ -q \
  --ignore=tests/llm/test_llm_ollama.py \
  --ignore=tests/llm/test_llm_ollama_integration.py
```

There is no main-pipeline `--dry-run`, `--debug`, `--memory-efficient`,
`--force-regenerate`, or `--config-file` option. Use `--verbose`, targeted step
selection, explicit output directories, and the health/preflight commands instead.

## Environment variables

Set provider and runtime variables in the shell or a local ignored `.env` file. Never
commit secrets.

- `OPENAI_API_KEY`: cloud LLM access.
- `OLLAMA_MODEL`, `OLLAMA_TEST_MODEL`: local model selection.
- `GNN_JAX_PLATFORM`: JAX device selection for PyMDP subprocesses.
- `GNN_SANDBOX`: Step 12 sandbox mode (`off`, `prefer`, or `require`).
- `GNN_ALLOW_UNSAFE_EXEC=1`: explicitly bypasses the Step 12 safety gate; use only
  for a deliberate, reviewed local test.

## Troubleshooting setup

- If imports fail after changing extras, run `uv sync --extra dev` again and inspect
  `uv run gnn health`.
- If Julia packages are not found, instantiate the matching committed environment
  with `--project` as shown above.
- If the type checker finds no models, pass a directory containing `.md` files via
  `--target-dir`; a single file path is not a directory discovery target.
- If a framework is unavailable, read the execution summary's skipped status before
  treating it as a pipeline failure.

## Related documentation

- [Configuration guide](configuration/README.md)
- [Quickstart](quickstart.md)
- [Framework availability](execution/FRAMEWORK_AVAILABILITY.md)
- [Troubleshooting](troubleshooting/README.md)
- [Dependency inventory](dependencies/OPTIONAL_DEPENDENCIES.md)
