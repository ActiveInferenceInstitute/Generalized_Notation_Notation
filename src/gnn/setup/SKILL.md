---
name: gnn-environment-setup
description: GNN environment setup and dependency management. Use when configuring the development environment, installing dependencies, managing virtual environments, or troubleshooting dependency issues for the GNN pipeline.
---

# GNN Environment Setup (Step 1)

## Purpose

Manages environment configuration, virtual environment creation, and dependency installation for the GNN pipeline. All package operations go through `uv`; there is no pip path.

## Key Commands

```bash
# Run setup step
python src/gnn/1_setup.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/gnn/main.py --only-steps 1 --verbose

# Using uv (recommended)
uv sync                    # Core dependencies plus the default dev tooling group
uv sync --no-dev           # Runtime dependencies without development tooling
uv sync --extra dev        # Development profile plus the CmdStanPy dev extra
uv sync --extra torch      # Supported optional PyTorch backend
uv sync --extra bnlearn    # Supported optional Bayesian-network backend
uv sync --all-extras       # Every declared optional extra
```

## Optional Dependency Groups

The default PEP 735 `dev` dependency group supplies test/lint/docs tools.
The separate `dev` extra adds CmdStanPy. Optional extras are discovered from
installed `Provides-Extra` metadata, with a source-manifest fallback;
`pyproject.toml` is the authoritative registry. Main profiles include:

| Group | Key Packages | Purpose |
| ----- | ------------ | ------- |
| Core `uv sync` | pymdp, jax, numpyro, discopy, LLM clients, visualization | Standard pipeline runtime |
| `audio` | librosa, soundfile, pedalboard | Audio generation |
| `gui` | gradio | GUI interface |
| `ml-ai` | transformers, scipy, scikit-learn | Machine learning extensions |
| `torch`, `bnlearn` | torch, bnlearn | Optional backends in the universal lock |
| `api` | fastapi, uvicorn | HTTP API |
| `stan` | cmdstanpy | Stan driver; native CmdStan is separate |
| `cpomdp`, `thrml`, `ngclearn` | cpomdp, thrml, ngclearn | Explicit experimental backends; ngclearn needs Python >=3.12 |
| `geo-infer` | h3 | Geospatial indexing |
| `graphs` | graphviz | Graphviz bindings |
| Default `dev` group | pytest-*, mypy, ruff | Development tools |
| `dev` extra | cmdstanpy | Development profile Stan driver |
| `research` | jupyterlab, sympy, numba, cython | Research tools |
| `scaling` | dask, distributed, ray | Parallel processing |
| `all` | Named optional groups above | Full optional installation |

## Environment Requirements

- **Python**: `>=3.11,<3.15`; default 3.14.8, with named compatibility profiles in the root README
- **Package manager**: uv 0.13.6 (required by `uv.toml`)
- **Core deps**: the scientific/backend/client requirements in `pyproject.toml`; the frozen lock selects their versions

The GUI/all-extras profiles use the documented Gradio/Pillow 13 uv override.
See [runtime exceptions](../../../README.md#runtime-defaults-and-required-profiles);
plain pip cannot resolve those extras while upstream Gradio caps Pillow below 13.

## Troubleshooting

| Issue | Solution |
| ----- | ------- |
| `uv: command not found` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| PyMDP import fails | `uv sync` |
| matplotlib missing | `uv sync` |
| CUDA not detected | Install the `torch` extra, inspect `torch.cuda.is_available()`, and qualify any separately selected CUDA build |


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `check_uv_project_status`
- `ensure_directory_exists`
- `find_project_gnn_files`
- `get_standard_output_paths`
- `get_uv_environment_info`
- `install_uv_dependency`
- `setup_uv_project_structure`
- `sync_uv_dependencies`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification
- [../../../SETUP_GUIDE.md](../../../SETUP_GUIDE.md) — Full setup guide
- [../../../pyproject.toml](../../../pyproject.toml) — Dependency definitions


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
