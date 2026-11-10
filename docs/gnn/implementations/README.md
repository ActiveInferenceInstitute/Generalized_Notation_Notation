# GNN Implementations Documentation

**Version**: canonical in [pyproject.toml](../../../pyproject.toml)

**Last Updated**: 2026-10-02

**Status**: Maintained guides; experimental status and acceptance are backend-specific

**Pipeline steps**: 25 · **Renderers**: [framework registry](../../../src/gnn/render/framework_registry.py) · **Tests**: see [repository README](../../../README.md)

This directory contains documentation and references for the Implementations domain of Generalized Notation Notation (GNN).

## Backend admission

Choose a model from the [exemplar index](../../../input/gnn_files/INDEX.md),
then check the renderer's declared state space, model composition, dynamics,
and any named execution contract. The
[framework registry](../../../src/gnn/render/framework_registry.py) declares
these capabilities. Its maintained default set excludes experimental
`cpomdp` and `thrml`, which require explicit selection.

| Source contract | Backend boundary |
|---|---|
| Categorical | PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, Stan and bnlearn accept categorical state space. Each backend has its own composition and inference limits. |
| Single linear-Gaussian block | JAX, RxInfer.jl, PyTorch, NumPyro, Stan and ngc-learn admit the state space. Consult the backend guide for filtering, sampling, and controller semantics. |
| Independent per-factor linear-Gaussian blocks | JAX admits the [factored exemplar](../../../input/gnn_files/continuous/factored_continuous_lgssm.md). |
| Explicit independent linear-Gaussian agents | JAX and RxInfer admit the [independent-agent exemplar](../../../input/gnn_files/continuous/independent_gaussian_agents.md). A shared flat Gaussian block plus an agent count leaves the agent semantics unspecified and is refused. |
| Mixed categorical and linear-Gaussian parameters | The [hybrid exemplar](../../../input/gnn_files/continuous/hybrid_discrete_continuous.md) is classified as a composition and receives `unsupported-composition` from all current backends. |
| Nonstationary categorical transition | The Step 11/12 PyMDP route admits time-indexed and regime-switched transitions. The standalone PyMDP renderer does not implement the switching runner. |
| `block_reset_v1`, `timed_soft_controller_v1`, `episodic_contingent_v1` | The named block-reset hierarchy, timed soft controller, and contingent T-maze contracts require JAX. |

Step 11 records a refused model/backend pair as `unsupported` in
`render_processing_summary.json`, separately from render failures. Step 12
does not execute refused pairs. An explicit backend request can still leave
a run's required work incomplete. Even when a renderer emits a program,
execution depends on runtime readiness and the current run's execution
receipt.

## Available Documents

- **[PyMDP](pymdp.md)**: Native categorical Active Inference through the maintained `pymdp` Agent API; admitted composition and inference settings come from the renderer contract.
- **[NumPyro](numpyro.md)**: Discrete and continuous probabilistic programs with preserved numeric parameters; the continuous generator emits an online filter and a separate NUTS trajectory.
- **[PyTorch](pytorch.md)**: Discrete and continuous tensor programs with round-trippable scientific parameters and covariance-aware continuous output.
- **[JAX](jax.md)**: Native categorical and linear-Gaussian programs, with explicitly represented factor and independent-agent routes.
- **[RxInfer.jl](rxinfer.md)**: Native Julia reactive inference over categorical and Gaussian model families; posterior traces retain factor/agent identity and their declared filtering or smoothing interpretation.
- **[ActiveInference.jl](activeinference_jl.md)**: Dedicated discrete-state Active Inference simulation in Julia (`ActiveInference.jl`).
- **[DisCoPy](discopy.md)**: Categorical string diagrams enabling advanced symmetry representations and compositional verification semantics for Multi-Agent Topologies (`discopy`).
- **[Stan](stan.md)**: Runnable HMM forward-algorithm programs (Dirichlet-prior A_est, NUTS or L-BFGS MAP) for discrete models and Kalman marginal-likelihood programs for continuous linear-Gaussian models, each with a cmdstanpy driver executed by Step 12 (`src/gnn/execute/stan/`).
- **[cpomdp](cpomdp.md)**: Experimental released-wheel linear-Gaussian filtering and admitted EFE control; explicit selection and resource receipts.
- **[THRML](thrml.md)**: Experimental released-wheel finite categorical Gibbs smoothing under fixed actions, independent components and bounded joint factor/modality composition, supervised execution, and sample-bound analysis. Structural zeros, continuous models, coupled agents, and hardware execution are outside this contract.

`bnlearn` and `ngclearn` also have maintained render and execution modules under
`src/gnn/render/` and `src/gnn/execute/`. The bnlearn route includes finite-model
conditional filtering over source conditional probability tables. The
ngc-learn route covers continuous predictive processing.

### Related integration (not a render backend)

- **[CatColab](catcolab.md)**: Topos Institute framework mapping GNN's `Step 7` export output into Schema/Stock-and-Flow/Olog categorical structures. This is an export-layer (`src/gnn/export/`) integration, not a `src/gnn/render/` renderer backend — there is no CatColab entry in `src/gnn/render/framework_registry.py` and no `--frameworks catcolab` render/execute path.

## Navigation

- [← Back to GNN Main Index](../README.md)
- [← Back to Master START_HERE](../../START_HERE.md)

---
*GNN: A text-based language for Active Inference generative models.*
