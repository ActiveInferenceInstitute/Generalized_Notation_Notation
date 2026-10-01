# GNN Quick Reference Guide

A concise reference for common GNN operations, syntax patterns, and frequently used commands.

---

## Quick Start Commands

### Running the Pipeline

```bash
# Full pipeline (all 25 steps)
uv run python src/gnn/main.py --target-dir input/gnn_files --verbose

# Run specific steps
uv run python src/gnn/main.py --only-steps "3,5,7,8,11,12" --verbose
```

---

## GNN Syntax Quick Reference

### Basic Model Structure

```markdown
## GNNSection
MyModel

## GNNVersionAndFlags
GNN v1

## ModelName
My Model Name

## StateSpaceBlock
A[3,3,type=float]   # Matrix
B[3,3,3,type=float] # 3D tensor
C[3,type=float]     # Vector

## Connections
D>s                # Directed flow
s-A                # Bidirectional

## InitialParameterization
A={(0.9,0.05,0.05), (0.05,0.9,0.05), (0.05,0.05,0.9)}
```

---

## Model Kinds

A GNN file denotes a generative model whose kind is declared by its notation
blocks and classified by the pipeline (`render.pomdp_contract.detect_model_kind`):

- **Discrete categorical** (POMDP/HMM): `A/B/C/D[/E]` blocks — the syntax shown above
- **Continuous linear-Gaussian**: `F/H/Q/R` dynamics blocks with `prior_mean/prior_cov`,
  optional closed-loop `goal_mean/control_gain`

Continuous models render and execute on JAX, NumPyro, PyTorch, Stan,
RxInfer.jl, and cpomdp; the categorical backends report `unsupported` for them. See the
README section *"Model Kinds and Framework Support"* and the
[GNN Syntax Reference](gnn/reference/gnn_syntax.md).

---

## Directory Reference

| Directory | Purpose |
|-----------|---------|
| `docs/gnn/` | GNN language spec |
| `docs/pymdp/` | PyMDP integration |
| `docs/rxinfer/` | RxInfer.jl integration |
| `docs/cognitive_phenomena/` | Example models |

---

## Active Inference Glossary

| Term | Definition |
|------|------------|
| **GNN** | Generalized Notation Notation |
| **A-matrix** | Likelihood matrix P(o|s) |
| **B-matrix** | Transition matrix P(s'|s,a) |
| **C-vector** | Preferences |
| **D-vector** | Initial state prior |

---

*For comprehensive documentation, see [docs/INDEX.md](./INDEX.md)*
