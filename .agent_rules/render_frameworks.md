# Render Frameworks

> Step 11 generates simulation code for multiple Active Inference frameworks.
> Run: `uv run python src/gnn/11_render.py --target-dir input/gnn_files`

## Supported Frameworks

| Framework | Language | Continuous | Step 12 Execution | Runtime Gating |
|----------------------|--------|--------------------------|-------------------|---------------------------------|
| **PyMDP** | Python | Unsupported | Yes | none |
| **RxInfer.jl** | Julia | Native | Yes | none |
| **ActiveInference.jl** | Julia | Unsupported | Yes | none |
| **JAX** | Python | Native | Yes | none |
| **DisCoPy** | Python | Unsupported | Yes | none |
| **PyTorch** | Python | Native | Yes | `torch` extra |
| **NumPyro** | Python | Native | Yes | none |
| **Stan** | Stan | Native | Yes | `uv sync --extra stan` |
| **bnlearn** | Python | Unsupported | Yes | `bnlearn` extra |
| **ngc-learn** | Python | Native (continuous-only) | Yes | `ngclearn` extra (py3.12 marker) |
| **cpomdp** | Python | Native (continuous-only) | Yes | `cpomdp` extra |

All eleven rows trace to `FRAMEWORK_REGISTRY` in
`src/gnn/render/framework_registry.py`: Continuous is `supports_continuous`,
Step 12 Execution is `supports_execution`, and runtime gating mirrors the
`FRAMEWORK_IMPORT_CHECK` probes in
`gnn.utils.runtime_safety.framework_availability` (backends listed `none` ship
in the default environment). ngc-learn and cpomdp are
continuous-only (`continuous_only: true`): discrete POMDP models are refused at Step 11. The
execution-only `lean` backend (`gnn.frameworks.ALL_FRAMEWORKS`, no render
entry) runs via the executor registry-runner path.

**Execution selection** (Step 12): run the requested frameworks and report explicit skipped or failed statuses when a requested backend is unavailable or fails.

---

## JAX ⭐ (Pure JAX — No Flax Required)

```python
"""Pure JAX Active Inference — no Flax dependency."""

import jax
import jax.numpy as jnp
from jax import random, jit

# Model matrices (extracted from GNN)
A_MATRIX = jnp.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
B_MATRIX = jnp.array([[[0.9, 0.1], [0.1, 0.9]], [[0.5, 0.5], [0.5, 0.5]]])
C_VECTOR = jnp.array([0.0, 1.0, 0.0])
D_VECTOR = jnp.array([0.5, 0.5])


@jit
def belief_update(prior, obs, A):
    posterior = prior * A[obs, :]
    return posterior / jnp.sum(posterior)


@jit
def expected_free_energy(qs, A, B, C, action):
    qs_next = B[:, :, action] @ qs
    qo = A @ qs_next
    H_qo = -jnp.sum(qo * jnp.log(qo + 1e-10))  # Epistemic
    pragmatic = jnp.sum(qo * C)  # Pragmatic
    return -pragmatic + H_qo
```

**Key**: Generated code uses only `jax`, `jax.numpy`, `optax` — never `flax` or `flax.linen`.

---

## PyMDP

```python
"""PyMDP Active Inference."""

try:
    from pymdp.agent import Agent
    from pymdp import utils

    PYMDP_AVAILABLE = True
except ImportError:
    PYMDP_AVAILABLE = False

# POMDP matrices
A = [[0.9, 0.1], [0.1, 0.9]]  # Likelihood
B = [[[0.9, 0.1], [0.1, 0.9]]]  # Transition
C = [0.0, 1.0]  # Preferences
D = [0.5, 0.5]  # Initial prior


def run_simulation(n_steps=10):
    if not PYMDP_AVAILABLE:
        raise ImportError("Install: uv pip install inferactively-pymdp")
    agent = Agent(A=A, B=B, C=C, D=D)
    obs = agent.reset()
    for t in range(n_steps):
        qs = agent.infer_states(obs)
        action = agent.sample_action()
        print(f"Step {t}: obs={obs}, action={action}")
```

---

## RxInfer.jl (Julia)

```julia
# Generated Julia code — genuine @model + infer() pipeline
using RxInfer, Distributions, LinearAlgebra, Random, SHA, JSON

@model function pomdp_model(y, A, B, D, u, T)
    s[1] ~ Categorical(D)
    y[1] ~ DiscreteTransition(s[1], A)
    for t in 2:T
        s[t] ~ DiscreteTransition(s[t-1], B[:, :, u[t-1]])
        y[t] ~ DiscreteTransition(s[t], A)
    end
end

result = infer(
    model = pomdp_model(A=A, B=B, D=D, u=actions, T=T),
    data = (y = observations,),
    iterations = 20,
    free_energy = true  # real VFE traces
)
```

**Setup**: `julia --startup-file=no --project=src/gnn/execute/rxinfer src/gnn/execute/rxinfer/setup_environment.jl --verbose`

---

## Matrix Extraction from GNN

All frameworks use the same extraction:

```python
def extract_matrices(gnn_model: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract POMDP matrices from parsed GNN model."""
    init = gnn_model.get("initialparameterization", {})
    return {
        "A": np.array(init.get("A", np.eye(3))),  # Likelihood
        "B": np.array(init.get("B", np.eye(3))),  # Transition
        "C": np.array(init.get("C", np.zeros(3))),  # Preferences
        "D": np.array(init.get("D", np.ones(3) / 3)),  # Initial prior
    }
```

---

## Output Structure

```
output/11_render_output/
└── model_name/
    ├── pymdp/model_name_pymdp.py
    ├── jax/model_name_jax.py
    ├── rxinfer/model_name_rxinfer.jl
    ├── activeinference_jl/model_name_activeinference.jl
    └── discopy/model_name_discopy.py
```

---

**Last Updated**: 2026-09-22 | **Status**: Maintained Standard
