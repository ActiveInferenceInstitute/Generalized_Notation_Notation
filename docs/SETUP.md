# GNN Project Setup Guide

This guide assumes commands are run from the repository root. The supported package
manager is `uv`; use `uv run` for Python commands so the project environment is used.

## Fast path

```bash
uv sync --extra dev
uv run python src/gnn/1_setup.py --target-dir input/gnn_files --output-dir output --dev --verbose
```

Then validate a model or run a focused pipeline path:

```bash
uv run gnn validate input/gnn_files/discrete/actinf_pomdp_agent.md --strict
uv run python src/gnn/main.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --only-steps "3,5,11,12" \
  --verbose
```

The checked-in `input/config.yaml` is loaded automatically by `src/gnn/main.py`; see the
[configuration guide](configuration/README.md) for its supported sections.

## Requirements

- Python `>=3.11,<3.14` (declared in `pyproject.toml`).
- `uv` with support for the lockfile format in this repository.
- Linux is the primary tested platform. macOS is supported for Python-only paths;
  Julia, GUI, audio, and system Graphviz paths have additional local requirements.
- At least 2 GB free disk space for a normal environment; rendering and generated
  artifacts can require considerably more.

Optional system tools:

- Julia for RxInfer.jl and ActiveInference.jl.
- Graphviz for graph layouts that invoke the Graphviz executable.
- A local Ollama installation for the default Step 13 local-LLM path.

## Installation

### 1. Clone and enter the repository

```bash
git clone https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation.git
cd Generalized_Notation_Notation
```

### 2. Install the project environment

```bash
# Runtime/core dependencies.
uv sync

# Runtime plus development, test, lint, and documentation tooling.
uv sync --extra dev

# Every declared optional group. This is the heaviest install.
uv sync --all-extras
```

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
| `dev` | Tests, coverage, linting, typing, docs, and development tools |
| `api` | FastAPI and Uvicorn |
| `audio` | Librosa, SoundFile, and Pedalboard |
| `gui` | Gradio |
| `graphs` | Python Graphviz bindings |
| `ml-ai` | SciPy, scikit-learn, and Transformers |
| `research` | Jupyter, SymPy, Numba, and research utilities |
| `scaling` | Dask, Distributed, and Ray |
| `stan` | cmdstanpy driver for rendered Stan programs (needs a local CmdStan toolchain) |
| `torch` | PyTorch backend |
| `bnlearn` | bnlearn runtime for rendered `.py`/`.R` Bayesian network programs |
| `geo-infer` | H3 geospatial indexing used by GEO_INFER integration surfaces |
| `all` | The optional groups above combined as a manually maintained extra |

Install a group directly with `uv sync --extra GROUP`; use `--all-extras` for all
optional groups.

## Framework Selection Strategies

Use the `--frameworks` option on Step 11 and Step 12 to select `all`, `lite`, or a
comma-separated list. The same selection is available through `src/gnn/main.py`.

## Framework boundaries

Step 11 has **9 render targets**. Step 12 executes **10 framework families** — every
render target plus Lean. Stan executes through the
cmdstanpy driver (`src/gnn/execute/stan/`) since v3.2.0; it needs `uv sync --extra stan`
plus a CmdStan toolchain and is reported skipped when either is absent. bnlearn
executes through `src/gnn/execute/bnlearn/` and is reported skipped until its runtime
is present (`uv sync --extra bnlearn`, or Rscript plus the R `bnlearn` package for
`.R` scripts). PyTorch and bnlearn are intentionally not part of the default lock
(heavy optional runtimes). The runtime reports skipped or unavailable frameworks rather
than pretending that every target is installed.

| Target | Language | Default environment status | Surface |
|---|---|---|---|
| PyMDP | Python | Core | Render + execute |
| JAX | Python | Core | Render + execute |
| NumPyro | Python | Core | Render + execute |
| DisCoPy | Python | Core | Render + execute |
| RxInfer.jl | Julia | Committed project environment | Render + execute |
| ActiveInference.jl | Julia | Committed project environment | Render + execute |
| PyTorch | Python | Intentionally not locked | Render + execute when installed manually |
| Stan | Stan | Optional extra (`uv sync --extra stan`) | Render + execute via cmdstanpy |
| bnlearn | Python | Optional extra (`uv sync --extra bnlearn`) | Render + execute |
| cpomdp | Python | Optional extra (`uv sync --extra cpomdp`) | Continuous models only, render + execute |

Model kind also matters: **discrete categorical** models (A/B/C/D[/E]) run on
every backend above except cpomdp, while **continuous linear-Gaussian** models
(F/H/Q/R + `prior_mean`/`prior_cov`) render to native LGSSM programs and
execute only on JAX, NumPyro, PyTorch, Stan, RxInfer.jl and cpomdp — the
categorical backends report them `unsupported`, and cpomdp reports discrete
models `unsupported`. See
[framework availability](execution/FRAMEWORK_AVAILABILITY.md).

### Python targets

The normal sync installs the core Python targets:

```bash
uv run python -c "import jax, numpyro, discopy; print('JAX, NumPyro, and DisCoPy OK')"
uv run python -c "from pymdp import Agent; print('PyMDP OK')"
```

For PyTorch or bnlearn, consult the registry explanation in
`src/gnn/render/framework_registry.py` and make the security decision explicitly before
installing them. Do not document them as core dependencies.

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
