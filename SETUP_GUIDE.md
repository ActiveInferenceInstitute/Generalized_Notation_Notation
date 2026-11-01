# GNN Pipeline - Complete Setup Guide

## Quick Start (Cold Start Installation)

### Prerequisites

- **Python `>=3.11,<3.15`** installed (default 3.14.8; see the root README for named compatibility exceptions)
- **uv 0.13.6** package manager installed (required by `uv.toml`)

  ```bash
  curl -LsSf https://astral.sh/uv/0.13.6/install.sh | sh
  ```

### Basic Setup (Core Dependencies)

```bash
# Clone the repository
git clone https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation.git
cd Generalized_Notation_Notation

# Run basic setup
uv run python src/gnn/1_setup.py --verbose
```

A normal `uv sync` / core install includes:

- Scientific stack: numpy, matplotlib, networkx, PyYAML, psutil, httpx
- Active Inference: `inferactively-pymdp`
- **LLM (Step 13+)**: `openai`, `ollama` (client), `python-dotenv`, `aiohttp` (core dependencies — no extra required)
- Development tooling from the default PEP 735 `dev` group: pytest, ruff, mypy, documentation and profiling tools. Use `--no-dev` for runtime only; the separate `--extra dev` adds CmdStanPy.

### Complete Setup (With Optional Packages)

For full functionality including Active Inference, machine learning, and visualization:

```bash
# Install all optional packages
uv sync --all-extras
```

Or install specific groups (canonical list in `pyproject.toml` → `[project.optional-dependencies]`):

```bash
# Install machine-learning extras (transformers, scipy, scikit-learn)
uv sync --extra ml-ai

# Install audio processing extras
uv sync --extra audio

# Install GUI interfaces
uv sync --extra gui
```

## Optional Package Groups

The GNN pipeline declares the following optional package groups (all installable
with `uv sync --extra <group>` or together with `uv sync --all-extras`):

### 1. **dev** - Development Profile

- **Default PEP 735 group**: pytest, pytest-cov, pytest-xdist, Ruff, mypy,
  Bandit, Sphinx, JupyterLab, profiling tools and related development packages.
  Ruff supplies the formatter/import/lint roles; Black, isort, Flake8 and
  Pylint are not separate declared tooling dependencies.
- **Separate `dev` extra**: `cmdstanpy`, mirroring the `stan` driver extra.
- **Installation**: `uv sync` includes the default tooling group;
  `uv sync --extra dev` also includes CmdStanPy. Use `uv sync --no-dev` for
  runtime dependencies without the default development group.

### 2. **api** - REST API Server

- **Packages**: `fastapi`, `uvicorn[standard]`
- **Use case**: Running the GNN REST API server (`src/gnn/api/`)
- **Installation**: `uv sync --extra api`

### 3. **ml-ai** - Machine Learning

- **Packages**: `transformers`, `scipy`, `scikit-learn`
- **Use case**: Deep learning and scientific computing beyond the core Step 12
  backends (PyTorch lives in its own `torch` extra — see group 10 — not here)
- **Installation**: `uv sync --extra ml-ai`

### 4. **audio** - Audio Processing & Sonification

- **Packages**: `librosa`, `soundfile`, `pedalboard`
- **Use case**: Audio analysis, sonification of model dynamics (Step 15)
- **Installation**: `uv sync --extra audio`

### 5. **gui** - Interactive GUI

- **Packages**: `gradio`
- **Use case**: GUI interfaces for model construction (Step 22)
- **Installation**: `uv sync --extra gui`. The committed uv override selects
  Pillow 13 despite Gradio 6.30.0's upstream `<13` cap; plain pip GUI/all installs
  cannot resolve this combination. See the
  [runtime exceptions](README.md#runtime-defaults-and-required-profiles).

### 6. **graphs** - Graphviz Bindings

- **Packages**: `graphviz`
- **Use case**: System Graphviz graph layouts (Steps 8-9)
- **Installation**: `uv sync --extra graphs`

### 7. **research** - Research & Notebooks

- **Packages**: `sympy`, `numba`, `cython`, `jupyterlab`, `jupyter-server`, `bleach`
- **Use case**: Notebook-based research workflows for non-developer users
- **Installation**: `uv sync --extra research`

### 8. **scaling** - Distributed Execution

- **Packages**: `dask`, `distributed`, `ray`
- **Use case**: Distributed execution of parameter sweeps (`execute/distributed.py`)
- **Installation**: `uv sync --extra scaling`

### 9. **stan** - Stan Backend Executor

- **Packages**: `cmdstanpy` (also included in `dev` and `all`)
- **Use case**: Step 12 execution of the Stan programs rendered by Step 11
  (HMM forward-algorithm models and continuous linear-Gaussian models)
- **Installation**: `uv sync --extra stan`, then install the CmdStan toolchain
  once: `uv run python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"`
  (needs a C++ compiler and `make`). Without a toolchain Step 12 reports Stan
  scripts as `skipped`, never as failed.

### 10. **torch** - PyTorch Backend

- **Packages**: `torch` (>= 2.13.0, also included in `all`)
- **Use case**: Step 11 rendering and Step 12 execution of the PyTorch
  backend (discrete tensor POMDPs and native continuous linear-Gaussian
  models). torch>=2.13.0 resolves GHSA-rrmf-rvhw-rf47 — the advisory that
  previously kept torch out of the default lock.
- **Installation**: `uv sync --extra torch`

### 11. **bnlearn** - Bayesian Networks

- **Packages**: `bnlearn` and its resolved dependencies.
- **Use case**: Native Bayesian-network programs generated by the bnlearn backend.
- **Installation**: `uv sync --extra bnlearn`.

### 12. **cpomdp** - Experimental Continuous POMDP

- **Packages**: `cpomdp==0.4.4`.
- **Installation**: `uv sync --extra cpomdp`; installing the extra is separate
  from qualifying a particular continuous model.

### 13. **thrml** - Experimental Categorical Sampling

- **Packages**: `thrml==0.1.4`.
- **Installation**: `uv sync --extra thrml`.

### 14. **ngclearn** - Experimental Neural Simulation

- **Packages**: `ngclearn`, `ngcsimlib` on Python >=3.12.
- **Installation**: `uv sync --extra ngclearn`; its requirements are inactive
  on Python 3.11, so that interpreter does not establish ngclearn readiness.

### 15. **geo-infer** - Geospatial Indexing

- **Packages**: `h3` in the declared 4.x range.
- **Installation**: `uv sync --extra geo-infer`.

### 16. **all** - Everything

- **Packages**: union of every group above
- **Installation**: `uv sync --all-extras`

> **Core vs optional**: `inferactively-pymdp`, `jax[cpu]`, `jaxlib`, `flax`,
> `optax`, `numpyro`, and `discopy` are **core** dependencies installed by a
> plain `uv sync` — no extra is required for the Step 12 Python backends.
> Julia backends (RxInfer.jl, ActiveInference.jl) additionally require a local
> Julia installation (the current qualified profile uses Julia 1.13.2 and
> RxInfer 6.1.0). Their project environments are committed
> under `src/gnn/execute/rxinfer/` and `src/gnn/execute/activeinference_jl/`; run
> `julia --project=<dir> -e 'using Pkg; Pkg.instantiate()'` once per project.
> `activeinference_jl/Project.toml` pins `Distributions < 0.25.126` because
> DistributionsAD 0.6.58 does not precompile against newer releases.
> PyTorch ships in the `torch` extra (`uv sync --extra torch`; torch>=2.13.0
> resolves GHSA-rrmf-rvhw-rf47); rendering is codegen-only and Step 12 skips
> the backend gracefully when the package is absent.

## Installation Methods

**Note**: The recommended installation method uses UV's built-in extras system via `pyproject.toml`.

### Method 1: Using UV Extras (Recommended)

```bash
# List available groups (shown in pyproject.toml [project.optional-dependencies])
# Read the authoritative list from pyproject.toml or OPTIONAL_GROUPS.

# Install all optional packages
uv sync --all-extras

# Install specific groups
uv sync --extra ml-ai --extra audio --extra gui
```

### Method 2: Using Setup Module

```bash
# Install via the setup step
uv run python src/gnn/1_setup.py --verbose
```

### Method 3: Using UV Directly

```bash
# Reproduce the committed core and default development profile.
uv sync --frozen
# Add a selected optional backend through its declared extra.
uv sync --frozen --extra torch
```

## Platform-Specific Notes

### Linux (Tested on Parrot OS / Debian)

- All packages install successfully using UV
- No additional system dependencies required for core functionality
- PyMDP and JAX work perfectly with CPU backend

### macOS

- All packages install successfully using UV
- Same behavior as Linux
- Native M1/M2 support with JAX

### Windows

- Core packages install successfully
- Some optional packages may require Windows-specific builds
- Recommended: Use WSL2 for best compatibility

## Verification

After installation, verify that packages are working:

```python
# Test PyMDP
from pymdp.agent import Agent

print("✅ PyMDP working!")

# Test JAX
import jax.numpy as jnp

print("✅ JAX working!")

# Test visualization
import plotly.express as px

print("✅ Plotly working!")
```

## Running the Pipeline

### Full Pipeline

```bash
uv run python src/gnn/main.py --target-dir input/gnn_files --verbose
```

### Specific Steps

```bash
# GNN parsing, rendering, and execution
uv run python src/gnn/main.py --only-steps "3,11,12" --verbose
```

### Individual Steps

```bash
# Just GNN parsing
uv run python src/gnn/3_gnn.py --target-dir input/gnn_files --verbose
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'psutil'"

**Solution**: Run setup to install core dependencies

```bash
uv run python src/gnn/1_setup.py --verbose
```

### Issue: "externally-managed-environment" error with pip

**Solution**: Use the supported project environment rather than the system Python:

```bash
uv sync --frozen
uv run --frozen --no-sync python --version
```

### Issue: JAX not using GPU

The frozen default is a CPU profile. GPU execution requires a separately
selected and qualified JAX/CUDA profile, matching hardware and native drivers;
a successful CPU installation does not establish GPU readiness. Keep such
experiments separate from the frozen release environment.

### Issue: PyMDP import errors

The distribution is named `inferactively-pymdp`; its Python module is `pymdp`.
Restore the frozen version split before diagnosing native dependencies:

```bash
uv sync --frozen
uv run --frozen --no-sync python -c "import pymdp; print(pymdp.__file__)"
```

## Package Versions (Current)

Versions are pinned in `uv.lock`; `pyproject.toml` declares the floor
constraints. Notable floors as of 2026-10-20:

### Core Dependencies

- Python: 3.11+ (`>=3.11,<3.15`)
- `inferactively-pymdp` 1.0.4 on Python 3.11 and 1.0.3 on 3.12+ (JAX-first rewrite; 1.0.4 caps `jax<0.11.1`)
- `jax[cpu]>=0.7.0,<0.13`, `jaxlib>=0.7.0,<0.13`, `flax`, `optax`, `numpyro>=0.22` (0.22 carries the jax 0.11 compatibility fix)
- `pillow>=13.0.0` and `aiohttp>=3.14.4` (lock: 13.0.0 and 3.14.5).
- numpy, matplotlib, networkx, pyyaml, pandas, plotly, seaborn, h5py

### Optional Dependencies

- See `[project.optional-dependencies]` in `pyproject.toml` and `uv.lock`
  for the complete optional-extra registry and its resolved versions; the profiles
  above include the native scientific, GUI, distributed and geospatial extras.

> Run `uv lock --check` and `uv sync --frozen` for the authoritative resolved
> set.

## References

- **UV Documentation**: <https://docs.astral.sh/uv/>
- **PyMDP Repository**: <https://github.com/infer-actively/pymdp>
- **JAX Documentation**: <https://jax.readthedocs.io/>
- **GNN Pipeline Documentation**: See `README.md` and `ARCHITECTURE.md`

**Last Updated**: 2026-10-20
**Pipeline Version**: [pyproject.toml](pyproject.toml) (canonical)
**Status**: Maintained (Linux & macOS)
**Validation authority**: use the command of record and latest dated receipt in
[`README.md`](README.md). Julia RxInfer execution uses the committed
`Project.toml` under `src/gnn/execute/rxinfer/`; local Ollama tests remain an
explicit opt-in surface.
