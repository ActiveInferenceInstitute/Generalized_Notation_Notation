# GNN Syntax Quick Reference

**GNN language version**: v1.1
**Status**: Cheatsheet

---

## File Structure

```gnn
## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1.1

## ModelName
Static Perception Sketch

## StateSpaceBlock
A[3,3, type=float]
D[3, type=float]
s[3, type=float, default=uniform]
o[3, type=int]

## Connections
A>o
s>A:observation_mapping
D>s:prior_initialization

## InitialParameterization
A={(0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9)}
D={(0.3, 0.4, 0.3)}
```

---

## Variable Declaration Syntax

```
NAME[dim₁, dim₂, …, type=<type>, default=<default>]
```

| Component | Example | Required? |
|-----------|---------|-----------|
| Name | `A`, `s`, `π` | ✅ |
| Dimensions | `[3,3]`, `[num_obs, num_states]` | ✅ |
| Type | `type=float`, `type=int`, `type=bool` | Optional, defaults to `float` |
| Default | `default=uniform`, `default=zeros` | ❌ |

**Defaults**: `uniform`, `zeros`, `ones`, `eye`, `random`

---

## Connection Syntax

| Syntax | Meaning |
|--------|---------|
| `A>B` | Directed edge: A → B |
| `A-B` | Undirected edge |
| `A>B:label` | Annotated directed edge |
| `A-B:label` | Annotated undirected edge |

---

## Initial Parameterization Keys

| Family | Keys | Typical kind |
|--------|------|--------------------|
| Discrete POMDP | `A` `B` `C` `D` `E` | FLAT (or FACTORED if `num_factors > 1`) |
| Linear-Gaussian | `F` `H` `Q` `R` `prior_mean` `prior_cov` | CONTINUOUS |
| Dirichlet priors | `dirichlet_A` … `dirichlet_E` | LEARNING |
| Per-level | `A_level1`, `B_level2`, … | HIERARCHICAL |
| Per-agent | `A_agent1`, `B_agent2`, … | MULTI_AGENT |
| Time-varying transitions | `B_t`, `B_regime` | NONSTATIONARY |

The canonical `B` order is `(next_state, previous_state, action)`.
The [kind rules](../gnn_syntax.md#model-kind-detection) cover structural
specs and compositions. Combining categorical parameters with a complete
linear-Gaussian block creates a hybrid composition. Check
[backend admission](../implementations/README.md#backend-admission)
before rendering a composition.

---

## ModelParameters

```gnn
## ModelParameters
num_hidden_states: 4
num_obs: 3
num_actions: 2
num_timesteps: 20
inference_mode: online      # batch (default) | online
inference_iterations: 40    # default 20
```

---

## Multi-Model Files (v1.1)

Separate models with `---` on its own line:

```gnn
## StateSpaceBlock
A[3,3, type=float]
## Connections
A>B
---
## StateSpaceBlock
C[2,2, type=float]
## Connections
C>D
```

---

## YAML Front-Matter (v1.1)

```yaml
---
author: Jane Doe
version: "1.0"
framework_targets: [pymdp, rxinfer]
tags:
  - active-inference
  - pomdp
---
```

---

## Error Codes

| Code | Meaning |
|------|---------|
| `GNN-E001` | Missing required section |
| `GNN-E002` | Dimension mismatch |
| `GNN-E003` | Unknown variable in connection (reserved — emitted as `GNN-W002`) |
| `GNN-E004` | Duplicate variable declaration |
| `GNN-E005` | Unparseable connection syntax |

---

*See [GNN v1.1 Syntax Specification](../gnn_syntax.md) for the canonical reference.*
