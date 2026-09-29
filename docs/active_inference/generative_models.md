# Generative Models: A, B, C, D Matrices

> **📋 Document Metadata**  
> **Type**: Technical Reference | **Audience**: Developers, Researchers | **Complexity**: Intermediate  
> **Cross-References**: [GNN Integration](gnn_integration.md) | [Active Inference Theory](active_inference_theory.md) | [PyMDP Implementation](implementation_pymdp.md)

## Overview

This document describes the **generative model** structure used in discrete Active Inference, focusing on the A, B, C, D, and E matrices. These matrices define how an agent believes the world works and what it prefers.

**Status**: ✅ Production Ready  
**Version**: 1.0

---

## The Generative Model

A generative model specifies the joint probability:

$$P(o_{1:T}, s_{1:T}, \pi) = P(\pi) \cdot P(s_1) \cdot \prod_{t=2}^{T} P(s_t|s_{t-1}, \pi) \cdot \prod_{t=1}^{T} P(o_t|s_t)$$

In matrix form, this becomes the A, B, C, D, E system.

> **Continuous models**: GNN also supports continuous linear-Gaussian state-space models, which declare F/H/Q/R system matrices plus `prior_mean`/`prior_cov` in an `initialparameterization` block instead of the discrete A/B/C/D/E matrices (detection: `detect_model_kind` in `src/gnn/render/pomdp_contract.py`). Continuous rendering is supported by RxInfer, JAX, NumPyro, PyTorch, Stan, and cpomdp; PyMDP and ActiveInference.jl are discrete-only.

---

## Matrix Definitions

### A Matrix: Likelihood (Observation Model)

**What it represents**: $P(o|s)$ — probability of observations given hidden states

**Dimensions**: `[num_observations, num_states]`

**Properties**:
- Columns sum to 1 (each state generates a probability distribution over observations)
- Each entry $A_{ij}$ = probability of observation $i$ given state $j$

**Example**: A 3-state, 3-observation model

```python
A = np.array(
    [
        [0.9, 0.1, 0.0],  # P(o=0|s=0,1,2)
        [0.1, 0.8, 0.2],  # P(o=1|s=0,1,2)
        [0.0, 0.1, 0.8],  # P(o=2|s=0,1,2)
    ]
)
# Column 0: state 0 → 90% observe o=0, 10% observe o=1
# Column 1: state 1 → 10% o=0, 80% o=1, 10% o=2
# Column 2: state 2 → 20% o=1, 80% o=2
```

**GNN Syntax**:
```
A[3,3] = likelihood_matrix
```

---

### B Matrix: Transitions (State Dynamics)

**What it represents**: $P(s'|s, a)$ — probability of next state given current state and action

**Dimensions**: `[num_states, num_states, num_actions]`

**Properties**:
- For each action, columns sum to 1
- $B_{ij}^{(a)}$ = probability of transitioning to state $i$ from state $j$ under action $a$

**Example**: 3 states, 2 actions

```python
B = np.zeros((3, 3, 2))

# Action 0: Stay in place (identity-ish)
B[:, :, 0] = np.array(
    [
        [0.9, 0.1, 0.0],
        [0.1, 0.8, 0.1],
        [0.0, 0.1, 0.9],
    ]
)

# Action 1: Move right
B[:, :, 1] = np.array(
    [
        [0.2, 0.0, 0.0],
        [0.8, 0.2, 0.0],
        [0.0, 0.8, 1.0],
    ]
)
```

**GNN Syntax**:
```
B[3,3,2] = transition_matrix
```

---

### C Matrix: Preferences (Prior Over Observations)

**What it represents**: $\ln P(o)$ — log prior preferences over observations

**Dimensions**: `[num_observations]` or `[num_observations, T]` for time-varying

**Properties**:
- Higher values = more preferred observations
- Often in log-probability (can be unnormalized)
- Drives goal-directed behavior through pragmatic value

**Example**: Prefer observation 2, avoid observation 0

```python
C = np.array([-2.0, 0.0, 2.0])
# o=0: strongly avoided (ln P = -2)
# o=1: neutral (ln P = 0)
# o=2: preferred (ln P = 2)
```

**Time-varying preferences**:
```python
C = np.array(
    [
        [-2.0, -2.0, 0.0, 2.0],  # o=0 preferences over time
        [0.0, 0.0, 0.0, 0.0],  # o=1 preferences over time
        [2.0, 2.0, 2.0, 2.0],  # o=2 preferences over time
    ]
)
# Goal becomes more important at later timesteps
```

**GNN Syntax**:
```
C[3] = preference_vector
```

---

### D Matrix: Initial State Prior

**What it represents**: $P(s_0)$ — prior belief about initial hidden state

**Dimensions**: `[num_states]`

**Properties**:
- Sums to 1 (probability distribution)
- Represents agent's initial uncertainty about world state

**Example**: Uncertain start, slight belief in state 0

```python
D = np.array([0.5, 0.3, 0.2])
# 50% initial belief in state 0
# 30% initial belief in state 1
# 20% initial belief in state 2
```

**GNN Syntax**:
```
D[3] = initial_prior
```

---

### E Matrix: Habit Prior (Policy Prior)

**What it represents**: $P(\pi)$ — prior probability over policies

**Dimensions**: `[num_policies]`

**Properties**:
- Default to uniform if not specified
- Can encode habitual or instinctive behaviors
- Combined with EFE to select policies

**Example**: Slight preference for policy 0

```python
E = np.array([0.4, 0.3, 0.3])
# Policy 0 has 40% prior probability
# Policies 1,2 have 30% each
```

**Effect on policy selection**:
$$P(\pi) \propto E(\pi) \cdot \exp(-\gamma G(\pi))$$

**GNN Syntax**:
```
E[3] = policy_prior
```

---

## Multi-Factor Models

For complex environments, use multiple state factors:

### Factorized States

```python
# Factor 1: Location (4 states)
# Factor 2: Object status (2 states)
# Total: 4 × 2 = 8 combined states

num_states = [4, 2]  # List of factor dimensions
```

### Factorized A Matrix

```python
A = []
# A[0]: How location affects observations
A.append(np.array([...]))  # [num_obs, 4]

# A[1]: How object status affects observations
A.append(np.array([...]))  # [num_obs, 2]
```

### Factorized B Matrix

```python
B = []
# B[0]: How actions affect location
B.append(np.array([...]))  # [4, 4, num_actions]

# B[1]: How actions affect object status
B.append(np.array([...]))  # [2, 2, num_actions]
```

---

## Learning the Matrices

### Dirichlet Priors

Learning uses Dirichlet distributions as conjugate priors:

```python
# Concentration parameters for A
a = np.ones((num_obs, num_states))  # Uniform prior

# After observing o when in state s:
a[o, s] += 1

# Expected A matrix:
A = a / a.sum(axis=0, keepdims=True)
```

### Learning Rate

Control learning speed with decay:

```python
a[o, s] += learning_rate * Q[s]  # Weighted by belief
```

### Forgetting

Implement forgetting for non-stationary environments:

```python
a = forgetting_rate * a + (1 - forgetting_rate) * a_prior
```

---

## Precision Weighting

### Sensory Precision

Confidence in observations:

```python
# Scale A matrix by precision
A_precise = softmax(beta_A * log(A))
# Higher beta_A → more confident in A
```

### Belief Precision

Confidence in beliefs:

```python
# Policy selection precision
gamma = 16.0  # Higher → more deterministic
P_pi = softmax(-gamma * G)
```

---

## Example: T-Maze Model

Classic Active Inference benchmark:

```python
# States: [center, left_arm, right_arm, cue_left, cue_right]
# Observations: [null, reward_left, reward_right, cue_left, cue_right]
# Actions: [stay, go_left, go_right]

# A Matrix: What do I observe in each state?
A = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0],  # null observation
        [0.0, 0.9, 0.1, 0.0, 0.0],  # reward_left
        [0.0, 0.1, 0.9, 0.0, 0.0],  # reward_right
        [0.0, 0.0, 0.0, 0.9, 0.1],  # cue_left
        [0.0, 0.0, 0.0, 0.1, 0.9],  # cue_right
    ]
)

# B Matrix: How do actions change state?
# B[:,:,0] = stay, B[:,:,1] = go_left, B[:,:,2] = go_right

# C Vector: Prefer rewards
C = np.array([0.0, 2.0, 2.0, 0.0, 0.0])  # Prefer reward observations

# D Vector: Start in center
D = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # Certain start in center
```

---

## Source Code References

### GNN Specification

| Component | Path |
|-----------|------|
| Model Parser | [`src/gnn/`](../../src/gnn/) |
| Type Checker | [`src/gnn/type_checker/`](../../src/gnn/type_checker/) |
| Validation | [`src/gnn/validation/`](../../src/gnn/validation/) |

### Execution

| Engine | Path |
|--------|------|
| PyMDP | [`src/gnn/execute/pymdp/`](../../src/gnn/execute/pymdp/) |
| RxInfer | [`src/gnn/execute/rxinfer/`](../../src/gnn/execute/rxinfer/) |

### Documentation

| Resource | Path |
|----------|------|
| PyMDP Guide | [`docs/pymdp/gnn_pymdp.md`](../pymdp/gnn_pymdp.md) |
| GNN Syntax | [`docs/gnn/reference/gnn_syntax.md`](../gnn/reference/gnn_syntax.md) |
| Examples | [`docs/gnn/tutorials/gnn_examples_doc.md`](../gnn/tutorials/gnn_examples_doc.md) |

---

## Related Documentation

- **[GNN Integration](gnn_integration.md)**: GNN syntax for models
- **[Active Inference Theory](active_inference_theory.md)**: How matrices are used
- **[Expected Free Energy](expected_free_energy.md)**: C matrix in EFE
- **[PyMDP Implementation](implementation_pymdp.md)**: Python usage
- **[POMDP Foundations](pomdp_foundations.md)**: Theoretical background

---

## Matrix Summary

| Matrix | Represents | Dimensions | Role |
|--------|------------|------------|------|
| **A** | $P(o|s)$ | `[obs, states]` | Observation model |
| **B** | $P(s'|s,a)$ | `[states, states, actions]` | Transition model |
| **C** | $\ln P(o)$ | `[obs]` or `[obs, T]` | Preferences |
| **D** | $P(s_0)$ | `[states]` | Initial prior |
| **E** | $P(\pi)$ | `[policies]` | Habit prior |

---

**Status**: ✅ Production Ready  
**Compliance**: GNN documentation standards  
**Maintenance**: Regular updates with new model patterns
