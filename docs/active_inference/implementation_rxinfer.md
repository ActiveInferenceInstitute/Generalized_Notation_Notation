# RxInfer.jl Implementation Reference

> **📋 Document Metadata**  
> **Type**: Implementation Reference | **Audience**: Developers | **Complexity**: Intermediate  
> **Cross-References**: [RxInfer Documentation](../rxinfer/README.md) | [Variational Inference](variational_inference.md) | [Computational Patterns](computational_patterns.md)

## Overview

**RxInfer.jl** is a Julia package for Bayesian inference using genuine `@model` + `infer()` variational message-passing on factor graphs. This document provides signposting to GNN source code and documentation.

The pipeline's canonical integration uses a committed Julia environment
(`Project.toml` + `Manifest.toml` pinning RxInfer 5.5.0 under
`src/gnn/execute/rxinfer/`) and a genuine `@model` + `infer()` pipeline:
`src/gnn/render/rxinfer/rxinfer_renderer.py` emits
`@model function pomdp_model(y, A, B, D, u, T)` (`Categorical` /
`DiscreteTransition` nodes) solved with `infer()` (`free_energy = true`),
populating `variational_free_energy` with real values (previously `Float64[]`).

RxInfer is the only one of the three Active Inference implementation frameworks (PyMDP, RxInfer, ActiveInference.jl) with continuous support (`supports_continuous=True`; the other registry backends with continuous support — JAX, PyTorch, NumPyro, Stan — are covered in their own docs). The discrete categorical POMDP path above is unaffected. When a GNN spec declares a continuous linear-Gaussian `initialparameterization` — F/H/Q/R system matrices plus `prior_mean`/`prior_cov` — [`src/gnn/render/rxinfer/_strategies_continuous.py`](../../src/gnn/render/rxinfer/_strategies_continuous.py) emits a native linear-Gaussian `@model` script (`continuous_pomdp_model`, precompiled in the pinned `GnnRxInferModels` package) instead of the categorical POMDP model:

```julia
x[1] ~ MvNormal(prior_mean, prior_cov)
x[t] = F * x[t-1] + u[t-1] + N(0, Q)
y[t] = H * x[t]           + N(0, R)
```

When `goal_mean`/`control_gain` are declared, the forward simulation closes the loop on beliefs with proportional control (`u[t] = gain * (goal - mu[t])`, `mu[t]` the online Kalman-filtered mean); otherwise `u` is zero and the dynamics run passively. The fully conjugate model is solved with the same `infer()` (`free_energy = true`) pipeline, and the renderer refuses to derive continuous parameters from discrete A/B/C/D stand-ins.

**Status**: ✅ Production Ready  
**Version**: 1.0

---

## Source Code Signposting

### Execution Engine

| Component | Path | Description |
|-----------|------|-------------|
| **RxInfer Runner** | [`src/gnn/execute/rxinfer/`](../../src/gnn/execute/rxinfer/) | Main execution scripts |
| **Julia Setup** | [`src/gnn/execute/julia_setup.py`](../../src/gnn/execute/julia_setup.py) | Julia environment config |

### Integration

| Component | Path | Description |
|-----------|------|-------------|
| **Executor** | [`src/gnn/execute/executor.py`](../../src/gnn/execute/executor.py) | Multi-engine dispatcher |
| **Processor** | [`src/gnn/execute/processor/`](../../src/gnn/execute/processor/) | Step 12 processing (facade package) |

---

## Documentation Signposting

### RxInfer Documentation

| Document | Path | Description |
|----------|------|-------------|
| **README** | [`docs/rxinfer/README.md`](../rxinfer/README.md) | Overview |
| **GNN RxInfer Guide** | [`docs/rxinfer/gnn_rxinfer.md`](../rxinfer/gnn_rxinfer.md) | Integration guide |
| **Multiagent** | [`docs/rxinfer/Multiagent_GNN_RxInfer.jl`](../rxinfer/Multiagent_GNN_RxInfer.jl) | Multi-agent models |
| **Engineering Guide** | [`docs/rxinfer/engineering_rxinfer_gnn.md`](../rxinfer/engineering_rxinfer_gnn.md) | Engineering details |

---

## Quick Reference

### Installation

Within the GNN pipeline, RxInfer 5.5.0 and all Julia dependencies are pinned by the
committed environment under `src/gnn/execute/rxinfer/` (`Project.toml` + `Manifest.toml`).
`setup_environment.jl` activates and instantiates it — there is **no runtime
`Pkg.add`**. The runner invokes:

```bash
julia --startup-file=no --project=src/gnn/execute/rxinfer <script>
```

For local experimentation outside the repo, install RxInfer directly:

```julia
using Pkg
Pkg.add("RxInfer")
```

### Basic Model

```julia
using RxInfer

@model function active_inference_model(A, B, C, D, T)
    # Initial state
    s_0 ~ Categorical(D)
    
    # State-observation sequence
    s = Vector{Any}(undef, T)
    o = Vector{Any}(undef, T)
    
    s[1] ~ Categorical(B * s_0)
    o[1] ~ Categorical(A * s[1])
    
    for t in 2:T
        s[t] ~ Categorical(B * s[t-1])
        o[t] ~ Categorical(A * s[t])
    end
    
    return s, o
end
```

### Inference

```julia
# Create model
model = active_inference_model(A, B, C, D, T)

# Run inference
result = inference(
    model = model,
    data = (o = observations,),
    returnvars = (s = KeepLast(),),
    iterations = 10
)

# Get beliefs
beliefs = result.posteriors[:s]
```

---

## Key Concepts

### Factor Graphs

RxInfer represents models as factor graphs:

```
    s[t-1] ────── B ────── s[t] ────── A ────── o[t]
       │                     │
       └──────── B ──────────┘
                 ↓
              s[t+1]
```

### Message Passing

Belief propagation via messages:

```julia
@rule Categorical(:out, Marginalisation) (
    m_p::PointMass,
) = begin
    Categorical(mean(m_p))
end
```

### Reactive Inference

Continuous, streaming inference:

```julia
subscription = subscribe!(
    inference_result,
    on_next = (result) -> handle_update(result)
)
```

---

## GNN Integration

### Model Mapping

```
GNN Syntax              →    RxInfer
─────────────────────────────────────────
A[obs, states]          →    A matrix
B[states, states, actions] → B[action] matrices
C[obs]                  →    C vector (via goal prior)
D[states]               →    D Categorical prior
```

### Execution Pipeline

```mermaid
graph LR
    GNN[GNN Model] --> Export[GNN Export]
    Export --> Julia[Julia Script]
    Julia --> RxInfer[RxInfer Model]
    RxInfer --> Inference[Message Passing]
    Inference --> Results[Posterior Beliefs]
```

---

## Advanced Features

### Hierarchical Models

```julia
@model function hierarchical_ai(T_high, T_low)
    # High-level (slow) dynamics
    for t in 1:T_high
        s_high[t] ~ transition_high(s_high[t-1])
        
        # Low-level (fast) dynamics
        for τ in 1:T_low
            s_low[t, τ] ~ transition_low(s_low[t, τ-1], s_high[t])
        end
    end
end
```

### Custom Nodes

```julia
# Define custom factor
@node MyFactor Stochastic [out, in1, in2]

# Implement update rules
@rule MyFactor(:out, Marginalisation) (...) = begin
    # Custom message computation
end
```

---

## Related Resources

### Theory
- **[Variational Inference](variational_inference.md)**: Message passing theory
- **[Active Inference Theory](active_inference_theory.md)**: Core concepts
- **[Generative Models](generative_models.md)**: Model specification

### Implementation
- **[PyMDP Implementation](implementation_pymdp.md)**: Python alternative
- **[ActiveInference.jl](implementation_activeinference_jl.md)**: Julia alternative
- **[Computational Patterns](computational_patterns.md)**: Common patterns

### External
- **[RxInfer.jl GitHub](https://github.com/biaslab/RxInfer.jl)**
- **[RxInfer Documentation](https://docs.rxinfer.com/stable/)**

---

**Status**: ✅ Production Ready  
**Compliance**: GNN documentation standards
