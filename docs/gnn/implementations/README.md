# GNN Implementations Documentation

**Version**: v3.2.0 Engine (Bundle v2.0.0)  
**Last Updated**: 2026-04-14  
**Status**: ✅ Production Ready  
**Modules**: 38+ · **Pipeline steps**: 25 · **Renderers**: 11 backends wired under `src/gnn/render/` (9 with dedicated guides in this directory; `bnlearn` and `ngclearn` have none) · **Tests**: see [../../../README.md](../../../README.md)

This directory contains documentation and references for the Implementations domain of Generalized Notation Notation (GNN).

## Available Documents

- **[PyMDP](pymdp.md)**: The canonical reference implementation for discrete True POMDP simulation (`pymdp`). Reached 1.0 Correlation Baseline.
- **[NumPyro](numpyro.md)**: Probabilistic programming for continuous distributions, uncertainty mechanics, and MCMC/SVI (`numpyro >= 0.14`). Verified as Fully Operational in March 2026.
- **[cpomdp](cpomdp.md)**: Continuous active inference: exact Kalman filtering plus expected-free-energy policy search on JAX (`uv sync --extra cpomdp`). Continuous models only. Discrete models report `unsupported`.
- **[PyTorch](pytorch.md)**: Neural Active Inference with learnable parameters, differentiable gradients, and GPU acceleration (`torch >= 2.0`). Verified as Fully Operational in March 2026.
- **[JAX](jax.md)**: High-performance numerical computing and XLA vector-space compilation (`jax`). Reached 1.0 Correlation Baseline.
- **[RxInfer.jl](rxinfer.md)**: Reactive message passing and declarative probabilistic programming in Julia (`RxInfer.jl`). Reached 1.0 Correlation Baseline.
- **[ActiveInference.jl](activeinference_jl.md)**: Dedicated discrete-state Active Inference simulation in Julia (`ActiveInference.jl`).
- **[DisCoPy](discopy.md)**: Categorical string diagrams enabling advanced symmetry representations and compositional verification semantics for Multi-Agent Topologies (`discopy`).
- **[Stan](stan.md)**: Runnable HMM forward-algorithm programs (Dirichlet-prior A_est, NUTS or L-BFGS MAP) for discrete models and Kalman marginal-likelihood programs for continuous linear-Gaussian models, each with a cmdstanpy driver executed by Step 12 (`src/gnn/execute/stan/`).

`bnlearn` (Bayesian network structure/parameter learning) is also a wired `src/gnn/render/` backend (see `generate_bnlearn_code` in `src/gnn/render/bnlearn/bnlearn_renderer.py`) but does not yet have a dedicated guide in this directory.

### Related integration (not a render backend)

- **[CatColab](catcolab.md)**: Topos Institute framework mapping GNN's `Step 7` export output into Schema/Stock-and-Flow/Olog categorical structures. This is an export-layer (`src/gnn/export/`) integration, not a `src/gnn/render/` renderer backend — there is no CatColab entry in `src/gnn/render/framework_registry.py` and no `--frameworks catcolab` render/execute path.

## Navigation

- [← Back to GNN Main Index](../README.md)
- [← Back to Master START_HERE](../../START_HERE.md)

---
*GNN: A text-based language for Active Inference generative models.*
