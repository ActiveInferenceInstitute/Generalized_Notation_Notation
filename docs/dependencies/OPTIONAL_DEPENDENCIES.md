# Optional Dependencies Guide

This document provides a comprehensive guide to optional dependencies in the GNN Pipeline. Understanding which dependencies are optional helps you install only what you need for your specific use case.

## Quick Reference

| Framework | Status | Purpose | Install Command | Pipeline Step |
|-----------|--------|---------|-----------------|---|
| PyMDP | Core | POMDP agent simulation (pymdp 1.0.0; discrete categorical models) | `uv sync` | 12 (Execute) |
| JAX + Flax | Core | JAX neural networks; discrete factorized and continuous LGSSM programs | `uv sync` | 12 (Execute) |
| NumPyro | Core | Probabilistic programming; discrete and continuous LGSSM | `uv sync` | 12 (Execute) |
| DisCoPy | Core | Category theory / string diagrams (discrete categorical models) | `uv sync` | 12 (Execute) |
| Stan | Optional | cmdstanpy driver for HMM (discrete) and LGSSM (continuous) programs | `uv sync --extra stan` + local CmdStan | 12 (Execute) |
| RxInfer.jl | Optional (Julia) | Julia probabilistic inference (discrete + continuous LGSSM) | `julia --startup-file=no --project=src/gnn/execute/rxinfer -e 'using Pkg; Pkg.instantiate()'` | 12 (Execute) |
| ActiveInference.jl | Optional (Julia) | Julia Active Inference (discrete categorical models) | `julia --startup-file=no --project=src/gnn/execute/activeinference_jl -e 'using Pkg; Pkg.instantiate()'` | 12 (Execute) |
| PyTorch | Optional | Deep learning backend; discrete factorized and continuous LGSSM | `uv sync --extra torch` | 12 (Execute) |
| Lean | Optional | fep_lean bridge for proof-carrying model verification | sibling fep_lean checkout (see `docs/other/fep_lean/README.md`) | 12 (Execute) |
| bnlearn | Optional | Bayesian network learning (render/execute; runtime optional) | `uv sync --extra bnlearn` | 12 (Execute) |
| Plotly | Core | Interactive visualizations | `uv sync` | 8-9 (Visualization) |
| GraphViz | Optional | Advanced graph layouts | `brew install graphviz` / `apt-get install graphviz` | 8-9 (Visualization) |

> Core status reflects `[project.dependencies]` in `pyproject.toml` (2026-08-02).
> Julia backends additionally require a local Julia install.

---

## Detailed Framework Information

### PyMDP - Core Simulation Framework

**Status**: Installed by the core `uv sync` environment.

**What it does**: Enables execution of rendered Python code for POMDP simulations using the PyMDP library

**Used in**: Step 12 (Execute) - Simulation execution

**Installation**:

```bash
# Via UV core dependency sync
uv sync

# Or individual package
uv pip install inferactively-pymdp
```

**Error if missing**:

```
ERROR:gnn.execute.pymdp.executor:PyMDP import failed: No module named 'pymdp.agent'
```

**Impact on pipeline**:

- ✅ Pipeline continues successfully
- ✅ DisCoPy and ActiveInference.jl still work
- ⚠️ PyMDP simulations are skipped
- Result: Python core backends (JAX, NumPyro, DisCoPy) still run; only the PyMDP backend is skipped

**When to install**:

- You need PyMDP-based POMDP simulations
- You're testing all supported frameworks
- You want comprehensive framework coverage

**When you don't need it**:

- You only need Julia (ActiveInference.jl, RxInfer) simulations
- You only need Python-based simulations (JAX, DisCoPy)
- You want a minimal installation

### Flax - Core JAX Dependency

**Status**: Installed by the core `uv sync` environment.

**What it does**: Enables execution of JAX-based neural network code for Active Inference models

**Used in**: Step 12 (Execute) - JAX simulation execution

**Installation**:

```bash
# Via UV core dependency sync
uv sync

# Or individual package
uv pip install flax
```

**Error if missing**:

```
ModuleNotFoundError: No module named 'flax'
```

**Impact on pipeline**:

- ✅ Pipeline continues successfully
- ✅ JAX rendering still works (code generation succeeds)
- ⚠️ JAX simulations fail to execute
- Result: other core backends (PyMDP, NumPyro, DisCoPy) still run; only the JAX backend is skipped

**When to install**:

- You need JAX neural network simulations
- You're using JAX-based Active Inference implementations
- You want to execute JAX-rendered code

**When you don't need it**:

- You don't use JAX for simulations
- You prefer Julia or Python alternatives
- You want a lighter installation

### RxInfer.jl - Optional Julia Framework

**Status**: ❌ Requires Julia; RxInfer is pinned by a committed Julia environment (not installed by default)

**What it does**: Enables execution of Julia code for probabilistic inference simulations using RxInfer via a genuine `@model` + `infer()` variational message-passing pipeline.

**Used in**: Step 12 (Execute) - RxInfer execution

**Installation**: RxInfer 5.5.0 and all Julia dependencies are pinned by the
committed environment under `src/gnn/execute/rxinfer/` (`Project.toml` +
`Manifest.toml`). Install by activating + instantiating that environment — there
is no runtime `Pkg.add`:

```bash
julia --startup-file=no --project=src/gnn/execute/rxinfer -e 'using Pkg; Pkg.instantiate()'
```

The runner invokes `julia --startup-file=no --project=src/gnn/execute/rxinfer <script>`.

**Prerequisites**: Julia must be installed (`julia --version`)

**Error if missing**:

```
ERROR: LoadError: ArgumentError: Package RxInfer not found in current path.
Run `julia --startup-file=no --project=src/gnn/execute/rxinfer -e 'using Pkg; Pkg.instantiate()'` to instantiate the committed RxInfer environment.
```

**Impact on pipeline**:

- ✅ Pipeline continues successfully
- ✅ RxInfer code rendering still works
- ⚠️ RxInfer simulations fail to execute
- Result: Python core backends run; Julia backends (RxInfer.jl, ActiveInference.jl) are skipped without Julia

**When to install**:

- You need Julia-based probabilistic inference
- You're testing RxInfer simulations
- You work primarily with Julia

**When you don't need it**:

- You don't use Julia
- You prefer Python frameworks (PyMDP, JAX)
- You want quick prototyping without Julia overhead

---

## Visualization Dependencies

### Plotly - Core Interactive Visualization Dependency

**Status**: Installed by the core `uv sync` environment.

**What it does**: Enables interactive 3D visualizations and web-based charts

**Used in**: Step 9 (Advanced Visualization) - Advanced visualization features

**Installation**:

```bash
# Via UV core dependency sync
uv sync

# Or individual package
uv pip install plotly
```

**Impact on pipeline**:

- ✅ Static visualizations still work (matplotlib-based)
- ⚠️ Interactive visualizations are skipped
- Result: Basic visualization features work, advanced features disabled

**When to install**:

- You need interactive 3D plots
- You want web-based visualization dashboards
- You're sharing visualizations as HTML files

### GraphViz - Optional Graph Layout Engine

**Status**: ❌ Not installed by default (OPTIONAL)

**What it does**: Provides advanced graph layout algorithms for network visualizations

**Used in**: Step 8-9 (Visualization) - Graph layout optimization

**Installation**:

**macOS**:

```bash
brew install graphviz
```

**Linux (Ubuntu/Debian)**:

```bash
apt-get install graphviz
```

**Windows**:
Download from: <https://graphviz.org/download/>

**Impact on pipeline**:

- ✅ Basic visualizations work (using networkx layouts)
- ⚠️ Advanced graph layouts fall back to simpler algorithms
- Result: Visualizations still generated, possibly less optimal

**When to install**:

- You need Graphviz-specific layout algorithms
- Your graphs have complex structures
- You want publication-quality layouts

---

## Installation Strategies

### Strategy 1: Minimal Installation (Fastest)

**Best for**: Quick testing, CI/CD pipelines, resource-constrained environments

```bash
# Core-only installation — includes PyMDP, JAX/Flax, NumPyro, DisCoPy
uv sync
```

**Result**:

- ✅ Full pipeline works
- ✅ All rendering (code generation) works
- ✅ Step 12 Python backends run (PyMDP, JAX, NumPyro, DisCoPy)
- ⚠️ Julia backends (RxInfer.jl, ActiveInference.jl) and PyTorch require manual installs

**Framework support**: PyMDP ✅, JAX+Flax ✅, NumPyro ✅, DisCoPy ✅

### Strategy 2: Standard Installation (Balanced)

**Best for**: Most users, development, standard workloads

```bash
uv sync
# Optional: Julia frameworks
# RxInfer.jl — instantiate the committed env (no runtime Pkg.add)
julia --startup-file=no --project=src/gnn/execute/rxinfer -e 'using Pkg; Pkg.instantiate()'
# ActiveInference.jl — instantiate the committed env (no runtime Pkg.add)
julia --startup-file=no --project=src/gnn/execute/activeinference_jl -e 'using Pkg; Pkg.instantiate()'
```

**Result**:

- ✅ Full pipeline works
- ✅ Python core frameworks available
- ✅ Julia backends available when Julia is installed

**Framework support**: PyMDP ✅, JAX+Flax ✅, NumPyro ✅, DisCoPy ✅, RxInfer.jl ✅ (with Julia), ActiveInference.jl ✅ (with Julia)

### Strategy 3: Complete Installation (All Features)

**Best for**: Comprehensive testing, research, all frameworks

```bash
# Core Python runtime dependencies
uv sync

# Optional extras (see pyproject.toml for the canonical group list)
uv sync --all-extras

# Julia packages
# RxInfer.jl — instantiate the committed env (no runtime Pkg.add)
julia --startup-file=no --project=src/gnn/execute/rxinfer -e 'using Pkg; Pkg.instantiate()'
# ActiveInference.jl — instantiate the committed env (no runtime Pkg.add)
julia --startup-file=no --project=src/gnn/execute/activeinference_jl -e 'using Pkg; Pkg.instantiate()'

# System dependencies
# macOS:
brew install graphviz
# Linux:
sudo apt-get install graphviz

# PyTorch backend (torch extra; torch>=2.13.0 resolves GHSA-rrmf-rvhw-rf47)
uv sync --extra torch
```

**Result**:

- ✅ Full pipeline with all features
- ✅ All frameworks available
- ✅ All visualization features enabled

**Framework support**: All ✅

---

## Checking What's Installed

### Check installed packages

```bash
# List installed Python packages
uv pip list | grep -E "pymdp|flax|plotly"

# Check for system dependencies
which julia  # Julia installed?
which dot    # GraphViz installed?
```

### Check Framework Availability at Runtime

```bash
# Run the setup step with verbose mode
uv run python src/gnn/1_setup.py --verbose

# Or check within Python
uv run python -c "
try:
    import pymdp
    print('PyMDP: Available')
except:
    print('PyMDP: Not installed')
    
try:
    import flax
    print('Flax: Available')
except:
    print('Flax: Not installed')
"
```

### Check MCP Tool Registration

```bash
# See which modules successfully loaded
uv run python src/gnn/21_mcp.py --verbose

# Grep output for successfully loaded modules
grep -i "successfully loaded" output/21_mcp_output/*.log
```

---

## Troubleshooting

### Issue: "PyMDP not available" warning during execution

**Cause**: PyMDP not installed

**Solutions**:

1. Install PyMDP through the core environment: `uv sync`
2. Or use other frameworks: Execution continues with available frameworks
3. Check installation: `uv run python -c "import pymdp; print(pymdp.__version__)"`

### Issue: JAX simulations fail but rendering works

**Cause**: Flax not installed (JAX is installed but Flax is missing)

**Solutions**:

1. Install Flax through the core environment: `uv sync`
2. Or skip JAX simulations in Step 12 settings
3. Check installation: `uv run python -c "import flax; print(flax.__version__)"`

### Issue: RxInfer simulations skipped

**Cause**: RxInfer.jl environment not instantiated / Julia not found

**Solutions**:

1. Instantiate the committed RxInfer env: `julia --startup-file=no --project=src/gnn/execute/rxinfer -e 'using Pkg; Pkg.instantiate()'` (no runtime `Pkg.add`)
2. Verify Julia: `julia --version`
3. Check installation: `julia --startup-file=no --project=src/gnn/execute/rxinfer -e 'using RxInfer'`

### Issue: Advanced visualizations not generating

**Cause**: GraphViz or Plotly not installed

**Solutions**:

1. Install GraphViz (system): See installation steps above
2. Install Plotly through the core environment: `uv sync`
3. Run Step 9 with verbose: `uv run python src/gnn/9_advanced_viz.py --verbose`

---

## Recommendations by Use Case

### For Research & Publication

```bash
# Install everything for maximum features
uv sync
# Plus Julia packages
# RxInfer.jl — instantiate the committed env (no runtime Pkg.add)
julia --startup-file=no --project=src/gnn/execute/rxinfer -e 'using Pkg; Pkg.instantiate()'
# Plus system dependencies
brew install graphviz  # or apt-get on Linux
```

### For Development/Testing

```bash
# Install core + testing frameworks
uv sync
```

### For Quick Prototyping

```bash
# Install only core
uv sync
# Add optional deps as needed
```

### For CI/CD Pipelines

```bash
# Install core - skip optional for speed
uv sync
# CI will test that pipeline works without optional deps
```

### For Production Deployment

```bash
# Install known-working set
uv sync
# Skip rarely-used optional deps
```

---

## FAQ

**Q: Do I need to install all optional dependencies?**  
A: No. The pipeline works without them. Rendering always works; execution uses whatever frameworks are installed.

**Q: Will the pipeline fail if I don't install optional dependencies?**  
A: No. The pipeline completes successfully with status SUCCESS_WITH_WARNINGS. Missing frameworks are skipped gracefully.

**Q: How many frameworks do I need for success?**  
A: The Step 12 outcome contract classifies each run: any failed script marks the
run `failed`; skipped optional frameworks yield `success_with_skips`; all
executed scripts succeeding yields `success`. Explicit (non-preset) framework
requests are strict — a failed or skipped requested framework fails the run.
See `src/gnn/execute/processor/summary.py::_classify_execute_outcome`.

**Q: Can I install dependencies later?**  
A: Yes. The pipeline detects available frameworks at execution time. Install a new framework and re-run Step 12.

**Q: What's the best framework to start with?**  
A: PyMDP, JAX, NumPyro, and DisCoPy are installed with the core `uv sync` — start with those. Add the Julia frameworks (RxInfer.jl, ActiveInference.jl) if you have Julia installed.

---

## Performance Impact

### Installation Size

- **Core only**: ~500MB (+ dependencies)
- **Core + PyMDP + Flax**: ~1.2GB
- **Full installation**: ~3-5GB (with Julia)

### Installation Time

- **Core only**: 2-5 minutes
- **Core + PyMDP + Flax**: 5-10 minutes
- **Full installation**: 15-30 minutes (includes Julia)

### Runtime Performance

- **Optional deps loading**: <100ms per framework
- **Missing framework skip**: ~10ms per framework
- **Overall impact**: Negligible - pipeline still completes in 3-4 minutes

---

## Contributing

If you encounter issues with optional dependencies:

1. Check version compatibility: Some frameworks require specific versions
2. Try fresh installation: `uv pip install --upgrade --force-reinstall inferactively-pymdp`
3. Report issues with: Python version, OS, and error messages
4. See [TROUBLESHOOTING.md](../troubleshooting/README.md) for more help

---
**Status**: ✅ Current for Pipeline v3.6.0 (verified 2026-09-26 — framework statuses reflect `[project.dependencies]` in `pyproject.toml`)
