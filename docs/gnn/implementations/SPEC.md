# Specification: Framework Implementation Docs

## Scope
Documentation for every supported rendering / execution backend. Each
document in this subtree describes how a GNN specification is translated
into framework-native code (PyMDP, RxInfer.jl, JAX, etc.) and what the
backend can / cannot express.

## Contents
| File | Backend | AI semantics |
|------|---------|--------------|
| `pymdp.md` | PyMDP (Python) | Full Active Inference |
| `rxinfer.md` | RxInfer.jl (Julia) | Full Active Inference |
| `activeinference_jl.md` | ActiveInference.jl (Julia) | Full Active Inference |
| `jax.md` | JAX (Python) | Full Active Inference |
| `numpyro.md` | NumPyro (Python) | Full probabilistic |
| `cpomdp.md` | cpomdp (Python) | Continuous Active Inference (expected-free-energy policy search) |
| `pytorch.md` | PyTorch (Python) | ML integration |
| `discopy.md` | DisCoPy (Python) | Categorical diagrams |
| `stan.md` | Stan | HMM forward-algorithm + LGSSM Kalman programs (cmdstanpy driver; no online AI loop) |
| `catcolab.md` | CatColab | Categorical tooling |

## Versioning
Each backend doc carries a frontmatter version matching the render module
version. Backends may lag the language version when experimental features
aren't yet implemented in that backend.

## Document Structure
Every backend doc should follow the structure in
[`rxinfer.md`](rxinfer.md) (the canonical template): Overview,
Architecture, GNN Parameter Ingestion, Generated Code Example, Limitations,
Usage, Source References, Navigation.

## Status
Maintained. `stan.md` documents runnable inference programs (Stan infers over a
simulated trajectory rather than running an online Active Inference loop); all
backends execute through the full pipeline.
