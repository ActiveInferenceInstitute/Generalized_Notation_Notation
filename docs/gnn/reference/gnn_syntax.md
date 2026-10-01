# GNN Syntax Reference

**GNN language version**: v1.1
**Last Updated**: 2026-08-19
**Status**: Maintained  

Quick reference for GNN syntax with working examples. The normative specification lives at [../gnn_syntax.md](../gnn_syntax.md).

## Syntax Validation

GNN syntax is validated through the pipeline type checker:

- **`src/gnn/5_type_checker.py`** → Syntax validation and type checking
  - See: **[src/gnn/type_checker/AGENTS.md](../../../src/gnn/type_checker/AGENTS.md)** for validation details
- **`src/gnn/6_validation.py`** → Advanced consistency checking

**Quick Start:**

```bash
# Validate GNN syntax
uv run python src/gnn/5_type_checker.py --target-dir input/gnn_files --strict --verbose
```

For complete pipeline documentation, see **[src/gnn/AGENTS.md](../../../src/gnn/AGENTS.md)**.

---

## Canonical Section Inventory

A valid GNN file uses the following sections in the order below. Two distinct
levels of obligation apply, and it is worth keeping them apart:

- **Enforced** — listed in `src/gnn/schema/parser.py::REQUIRED_SECTIONS`. Absence is
  a hard error: `validate_required_sections` emits `GNN-E001`
  (*missing required section*) and the file fails type checking.
- **Expected** — not checked by the validator, but every sample in
  `input/gnn_files/` supplies them and downstream renderers and analyzers
  read them. Omitting one does not fail the type checker; it degrades or
  skips the steps that consume it.

| # | Section | Obligation | Parser hook |
|---|---------|------------|-------------|
| 1 | `## GNNSection` | **Enforced** | `_parse_gnn_section` |
| 2 | `## GNNVersionAndFlags` | **Enforced** | `_parse_version_section` |
| 3 | `## ModelName` | **Enforced** | `_parse_model_name` |
| 4 | `## ModelAnnotation` | Optional | `_parse_annotation` |
| 5 | `## StateSpaceBlock` | **Enforced** | `_parse_state_space` |
| 6 | `## Connections` | **Enforced** | `_parse_connections` |
| 7 | `## InitialParameterization` | Expected | `_parse_parameters` |
| 8 | `## Equations` | Expected | `_parse_equations` |
| 9 | `## Time` | Expected | `_parse_time` |
| 10 | `## ActInfOntologyAnnotation` | Expected | `_parse_ontology` |
| 11 | `## ModelParameters` | Expected | `_parse_model_parameters` |
| 12 | `## Footer` | Expected | `_parse_footer` |
| 13 | `## Signature` | Expected | `_parse_signature` |

Sections 7–13 are what earlier revisions of this page called "v1.5
extensions". Treat them as mandatory when authoring — a model missing
`InitialParameterization` cannot render, and one missing
`ActInfOntologyAnnotation` is invisible to Steps 10, 13, and 24 — but do not
expect the validator to catch their absence for you.

### `## Equations`

Free-form block documenting the generative equations of the model. Lines
starting with `#` are treated as commentary; lines containing `=` are
extracted as equation literals. Consumed by Step 13 (LLM analysis) for
prompt context and by Step 24 (intelligent analysis) for traceability. Not
dimensionally validated — the type checker skips this section.

```gnn
## Equations
# Generative Model
Q(s) = softmax(ln(D) + ln(A^T * o))
# Policy Posterior
Q(π) = softmax(-G(π))
```

### `## Time`

Temporal regime. Exactly one of the two values below, followed by optional
modifiers on subsequent non-comment lines. (The markdown parser recognises
only `Static` and `Dynamic`; continuous-state exemplars use `Dynamic` with a
`Discrete` time index and are classified CONTINUOUS from their
parameterization family, not from this section.)

| Value | Meaning |
|-------|---------|
| `Static` | Perception-only / single-step inference, no dynamics. |
| `Dynamic` | Time-indexed variables; B matrix required for discrete models. |

Optional modifiers (Dynamic only):
- `DiscreteTime=t` — declare the time index variable
- `ContinuousTime=τ`
- `ModelTimeHorizon=<int|Unbounded>`

```gnn
## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=20
```

### `## ModelParameters`

Key-value scalar parameters consumed by renderers (`render/pymdp`,
`render/rxinfer`, `render/jax`) to size the code they emit. Syntax:
`key: value` one per line. Scalar values only (int / float / string).
Dimensions declared here may be referenced by name in `StateSpaceBlock`
(e.g., `A[num_obs, num_hidden_states]`).

Canonical keys expected by renderers (not all are required for every model):

| Key | Type | Used by |
|-----|------|---------|
| `num_hidden_states` | int | all renderers |
| `num_obs` | int | all renderers |
| `num_actions` / `num_controls` | int | all renderers |
| `num_timesteps` | int | all renderers |
| `num_modalities` | int | PyMDP, JAX |
| `num_factors` | int | PyMDP, RxInfer; `> 1` also selects the FACTORED model kind |
| `nr_agents` | int | `> 1` selects the MULTI_AGENT model kind |
| `learning_rate` | float | JAX, PyTorch |
| `inference_mode` | `batch` (default) / `online` | RxInfer FLAT models — see below |
| `inference_iterations` | int (default `20`) | RxInfer — variational iterations per `infer()` call |

Omitting a canonical key is not an error; renderers fall back on the
dimensions parsed from `StateSpaceBlock`.

`inference_mode` chooses how a FLAT model is run. `batch`, the default,
smooths over the whole observation sequence. `online` runs `infer()` once per
timestep on the observation prefix, so the filtered posterior — rather than a
retrospective one — drives expected-free-energy and habit-prior action
selection. Any value other than `batch` or `online` is rejected. The same
setting can be passed as a render option, but declaring it here keeps the
file self-describing. Per-kind strategy details:
[`src/gnn/render/rxinfer/README.md`](../../../src/gnn/render/rxinfer/README.md).

```gnn
## ModelParameters
num_hidden_states: 4
num_obs: 3
num_actions: 2
num_timesteps: 20
learning_rate: 0.01
inference_mode: online
inference_iterations: 40
```

### `## Footer`

Human-readable closing block. Free-form Markdown, typically model name +
version + one-line disposition. Not dimensionally validated; consumed by
Step 23 (report) for audit trails.

```gnn
## Footer
Simple POMDP Agent v1.0 — deterministic transition, partial observability.
```

### `## Signature`

Provenance / cryptographic digest. Free-form single line — current
convention is a literal `pending` string or a hex digest. Reserved for
future provenance tooling (Step 18 security). Presence is expected but not
enforced by the type checker; the value may be marked as `pending`.

```gnn
## Signature
Cryptographic signature goes here
```

---

## Variable Declaration

```gnn
## StateSpaceBlock
s[2,1,type=int]      # 2D state vector, integer type
o[3,1,type=float]    # 3D observation vector, float type
A[3,2,type=float]    # 3×2 matrix, float type
t[1,type=int]        # Time scalar, integer type
```

## Subscripts and Superscripts

```gnn
s_t[2,1,type=float]      # s with subscript t
s_t+1[2,1,type=float]    # s with subscript t+1
X^observed[3,1,type=int] # X with superscript observed
π_f1[3,type=float]       # π with subscripts f1
```

## Connections

Identifiers in **Connections** must match variable names declared in **StateSpaceBlock** (case-sensitive): use `s>o` if the state variable is `s`, not `S`.

```gnn
## Connections
s>o          # Directed: s causes o
s-A          # Undirected: s relates to A
s_t>s_t+1    # Temporal: current state to next state
(s,u)>B      # Multiple inputs to B
```

## Dimensions and Types

```gnn
X[2]           # Vector of length 2
X[2,3]         # 2×3 matrix
X[2,3,4]       # 3D tensor: 2×3×4
X[len(π)]      # Dynamic size based on policy length
X[1,type=int]  # Explicit type declaration
```

## Initial Values

```gnn
## InitialParameterization
D={0.5,0.5}                    # Vector
A={(0.9,0.1),(0.2,0.8)}       # Matrix rows
B={((1,0),(0,1)),((0,1),(1,0))} # 3D tensor
```

Three further key families are recognized alongside the discrete POMDP
matrices `A`/`B`/`C`/`D`/`E`, and a file may carry more than one at once:

```gnn
## InitialParameterization
# Linear-Gaussian system (native continuous rendering)
F={(1.0,0.0),(0.0,1.0)}        # state transition
H={(1.0,0.0),(0.0,1.0)}        # observation readout
Q={(0.05,0.0),(0.0,0.05)}      # process noise
R={(0.1,0.0),(0.0,0.1)}        # observation noise
prior_mean={(0.0,0.0)}
prior_cov={(0.5,0.0),(0.0,0.5)}

# Dirichlet pseudo-counts — A becomes a learned latent, not a constant
dirichlet_A={(8.0,1.0,1.0),(1.0,8.0,1.0),(1.0,1.0,8.0)}

# Per-level (hierarchical) and per-agent (multi-agent) matrix suffixes
A_level2={(0.8,0.2),(0.2,0.8)}
B_agent1={((1,0),(0,1)),((0,1),(1,0))}
```

The `continuous/` exemplars are pure linear-Gaussian models: `x`, `y`,
optional `u`, `F`/`H`/`Q`/`R`, `prior_mean`/`prior_cov`, optional
`goal_mean`/`control_gain`, with no discrete A/B/C/D stand-in and every
symbol declared in `StateSpaceBlock` (so the dimension check reports no
`GNN-W003`/`GNN-E002` findings). Which family is present determines the model
kind the renderer dispatches on — `detect_model_kind` routes these files as
CONTINUOUS; the normative rules and precedence order are in
[`gnn_syntax.md` § Parameterization families](../gnn_syntax.md#parameterization-families).
JAX, NumPyro, PyTorch, Stan, RxInfer.jl and cpomdp render and execute them; PyMDP,
ActiveInference.jl, DisCoPy and bnlearn report render status `unsupported`.

## Canonical matrix orientation (B)

The transition tensor `B` is stored with semantic axis order
**(next_state, previous_state, action)**: `B[s', s, a]` is the probability of
moving to next state `s'` from previous state `s` under action `a`. This is
the native order of pymdp 1.0.0 (`B[f][s, v, u]`) and matches the
`StateSpaceBlock` declaration comment.

In `InitialParameterization`, `B` is written in **declaration order**: the
outer axis is `next_state`, and each next-state slice is a
`(previous_state, action)` matrix with **rows = previous states** and
**columns = actions**:

```gnn
## StateSpaceBlock
B[3,3,2,type=float]   # Transition tensor: next_state, previous_state, action

## InitialParameterization
# B: The transition tensor B is stored as (next_state, previous_state, action):
# the outer axis is the next state; within each slice, rows are previous
# states and columns are actions.
B={
  ( (0.9, 0.1),
    (0.2, 0.0),
    (0.0, 0.0) ),   # next_state 0
  ( (0.1, 0.0),
    (0.6, 0.2),
    (0.0, 0.2) ),   # next_state 1
  ( (0.0, 0.9),
    (0.2, 0.8),
    (1.0, 0.8) )    # next_state 2
}
```

The tensor is **column-stochastic over the `next_state` axis**: for every
`(previous_state, action)` pair, the entries along the outer axis sum to 1
(in the example above, summing the three slices position-by-position yields
all ones). `input/gnn_files/pomdp_gridworld/pomdp_gridworld_3x3.md` is the
canonical exemplar of this convention.

**Accepted alternative (must be self-described).** A file may instead write
`B` as one slice per **action**, with rows = next states and columns =
previous states — an axis permutation of the same semantic tensor, not a
different one. Such a file MUST say so in its `InitialParameterization`
comment (as `input/gnn_files/discrete/actinf_pomdp_agent.md` does); the
extractor records the claimed convention in `matrix_provenance["B"]` and
downstream consumers normalize it.

Consistency between the declaration and the parameterization is enforced by
the extractor and the type checker:

- The extractor parses the **declared** axis order (the `StateSpaceBlock`
  comment), the **claimed** convention (the `InitialParameterization`
  comment), and the orientation **detected** from slice row/column sums, and
  records all three in `matrix_provenance["B"]`. Matrix data is stored exactly
  as written; `canonicalize_pomdp(spec)` returns a copy re-expressed in the
  canonical order for consumers that need it.
- A contradiction between the declaration and the parameterization comment —
  or data that is only row-stochastic — is a validation **ERROR** under strict
  validation and a warning otherwise. **Doubly-stochastic data is accepted**:
  when both row and column sums equal 1 the data is valid under either
  orientation, so no verdict is issued.

> **🔗 Cross-Reference**: terminology for `B` and the other discrete POMDP
> matrices lives in the [Active Inference glossary](../../../docs/active_inference/glossary.md).

## C vector semantics

`C` is a **real-valued preference vector over observations** on a
log-preference scale:

- **Not a probability distribution.** Entries are log-preference weights, not
  normalized probabilities. Any real values are valid; only *differences*
  between entries are meaningful.
- **Unnormalized by design.** Consumers that need a distribution over
  observations apply `softmax(C)` to it (the ActiveInference.jl, NumPyro and
  PyMDP consumers do exactly this).
- A `C` that happens to be non-negative and sum to 1 is accepted, with a
  type-checker note — it is read under the same semantics as any other
  preference vector.

```gnn
## StateSpaceBlock
C[3,type=float]       # Log-preferences over observations

## InitialParameterization
# C: 3 observations. Log-preference weights: any real values,
# only differences matter; consumers softmax where a distribution is needed.
C={(0.0, -1.0, -2.0)}
```

> **🔗 Cross-Reference**: see
> [C Matrix/Vector](../../../docs/active_inference/glossary.md#c-matrixvector)
> in the Active Inference glossary.

## Mathematical Operations

```gnn
P(X|Y)    # Conditional probability
X+Y       # Addition
X*Y       # Multiplication  
X/Y       # Division
X^2       # Power
```

## Comments

```gnn
s[2,1,type=float]  # Hidden state vector
### This is a full-line comment
A[2,2,type=float]  ### Recognition matrix
```

## Time Specifications

```gnn
## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=10
```

## Complete Minimal Example

This example includes every **Enforced** and **Expected** section from the
table above (`GNNSection`, `GNNVersionAndFlags`, `ModelName`,
`StateSpaceBlock`, `Connections`, `InitialParameterization`, `Equations`,
`Time`, `ActInfOntologyAnnotation`, `ModelParameters`, `Footer`,
`Signature`):

```gnn
## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
Simple Static Model

## StateSpaceBlock
s[2,1,type=float]
o[2,1,type=float]
A[2,2,type=float]

## Connections
s>A
A>o

## InitialParameterization
A={(0.9,0.1),(0.1,0.9)}

## Equations
# Observation likelihood: o = A @ s

## Time
Static

## ActInfOntologyAnnotation
s=HiddenState
o=Observation
A=RecognitionMatrix

## ModelParameters
num_hidden_states: 2
num_obs: 2

## Footer
Simple Static Model - GNN Representation.

## Signature
Cryptographic signature goes here
```

This parses cleanly and executes in the GNN pipeline. Only the five
**Enforced** sections are strictly checked, per the obligation table above;
the rest are supplied because downstream steps read them.

## Variable naming conventions

Use factor/modality indices consistently (`s_f0`, `o_m0`, `u_c0`) as in the [Variable Declaration](#variable-declaration) examples above.

## Connection notation

Directed edges use `>`; undirected compatibility uses `-` (see [Connections](#connections)).

## Mathematical expressions

Matrix literals and tuples belong in `InitialParameterization` and equation blocks; keep shapes aligned with the [type checker](../../../src/gnn/type_checker/AGENTS.md).

## Multi-agent extensions

Model multiple agents by declaring additional state factors and control indices; see [GNN Multi-Agent](../advanced/gnn_multiagent.md).
