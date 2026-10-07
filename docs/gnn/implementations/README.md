# GNN Implementations Documentation

**Version**: 4.0.0 (canonical: [pyproject.toml](../../../pyproject.toml))

**Last Updated**: 2026-10-02

**Status**: Maintained guides; experimental status and acceptance are backend-specific

**Pipeline steps**: 25 · **Renderers**: declared by `src/gnn/render/framework_registry.py`; cpomdp and THRML require explicit experimental selection · **Tests**: see [../../../README.md](../../../README.md)

This directory contains documentation and references for the Implementations domain of Generalized Notation Notation (GNN).

## Available Documents

- **[PyMDP](pymdp.md)**: Native categorical Active Inference through the maintained `pymdp` Agent API; admitted composition and inference settings come from the renderer contract.
- **[NumPyro](numpyro.md)**: Discrete and continuous probabilistic programs with preserved numeric parameters; continuous traces distinguish online filtering from optional NUTS inference.
- **[PyTorch](pytorch.md)**: Discrete and continuous tensor programs with round-trippable scientific parameters and covariance-aware continuous output.
- **[JAX](jax.md)**: Native categorical and linear-Gaussian programs, with explicitly represented factor and independent-agent routes.
- **[RxInfer.jl](rxinfer.md)**: Native Julia reactive inference over categorical and Gaussian model families; posterior traces retain factor/agent identity and their declared filtering or smoothing interpretation.
- **[ActiveInference.jl](activeinference_jl.md)**: Dedicated discrete-state Active Inference simulation in Julia (`ActiveInference.jl`).
- **[DisCoPy](discopy.md)**: Categorical string diagrams enabling advanced symmetry representations and compositional verification semantics for Multi-Agent Topologies (`discopy`).
- **[Stan](stan.md)**: Runnable HMM forward-algorithm programs (Dirichlet-prior A_est, NUTS or L-BFGS MAP) for discrete models and Kalman marginal-likelihood programs for continuous linear-Gaussian models, each with a cmdstanpy driver executed by Step 12 (`src/gnn/execute/stan/`).
- **[cpomdp](cpomdp.md)**: Experimental released-wheel linear-Gaussian filtering and admitted EFE control; explicit selection and resource receipts.
- **[THRML](thrml.md)**: Experimental released-wheel finite categorical Gibbs smoothing under fixed actions, independent components and bounded joint factor/modality composition, supervised execution, and sample-bound analysis. Structural zeros, continuous models, coupled agents, and hardware execution are outside this contract.

`bnlearn` (Bayesian network structure/parameter learning) and `ngclearn`
(predictive processing) also have maintained render and execution modules under
`src/gnn/render/` and `src/gnn/execute/`. Their availability and supported model
kinds come from the live registries; a backend name alone does not establish
acceptance for every model composition.

### Related integration (not a render backend)

- **[CatColab](catcolab.md)**: Topos Institute framework mapping GNN's `Step 7` export output into Schema/Stock-and-Flow/Olog categorical structures. This is an export-layer (`src/gnn/export/`) integration, not a `src/gnn/render/` renderer backend — there is no CatColab entry in `src/gnn/render/framework_registry.py` and no `--frameworks catcolab` render/execute path.

## Navigation

- [← Back to GNN Main Index](../README.md)
- [← Back to Master START_HERE](../../START_HERE.md)

---
*GNN: A text-based language for Active Inference generative models.*
