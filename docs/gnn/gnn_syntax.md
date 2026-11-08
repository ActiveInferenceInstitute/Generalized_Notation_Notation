# GNN v1.1 Syntax Specification

> **Status**: Living document · Last updated 2026-04-14
> **Canonical reference for parsers, validators, and editor support.**
> For the condensed example-driven companion, see [reference/gnn_syntax.md](reference/gnn_syntax.md).

---

## 1  File Structure

A GNN file is a UTF-8 Markdown file (`.md`) consisting of **ordered sections**, each
introduced by a level-2 header (`## SectionName`).

### Required sections (in order)

| Section | Purpose |
|---------|---------|
| `## GNNSection` | Short identifier (no spaces; e.g. `ActInfPOMDP`) |
| `## GNNVersionAndFlags` | `GNN v1`, `GNN v1.0`, or `GNN v1.1` with optional flags |
| `## ModelName` | Human-readable model title |
| `## StateSpaceBlock` | Variable and matrix declarations |
| `## Connections` | Edge list between state-space variables |

### Parser and validator behavior

Two code paths are used in this repository:

- **Strict schema validator**: `src/gnn/schema/parser.py` enforces required sections, declaration formats, and connection grammar.
- **Permissive markdown parser**: `src/gnn/parsers/markdown_parser.py` is more tolerant when loading mixed or partially structured markdown.

For CI, type-checking, and pipeline validation, prefer examples that satisfy the strict schema validator.

### Optional sections

| Section | Purpose |
|---------|---------|
| `## ModelAnnotation` | Free-text description of the model |
| `## InitialParameterization` | Concrete matrix / vector values |
| `## Equations` | LaTeX-rendered formulas defining model dynamics and relationships between variables |
| `## Time` | Static/Dynamic setting, discrete/continuous time variable, and model time horizon |
| `## ActInfOntologyAnnotation` | Variable semantic bindings to Ontological domains |
| `## ModelParameters` | Model settings such as counts, horizons, and execution-contract parameters |
| `## Footer` | Closes the file and allows read-in from either end |
| `## Signature` | Cryptographic signature block / provenance information |

### `ModelParameters` keys read by renderers

Write one `key: value` entry per line. Dimension keys such as
`num_hidden_states` and `num_obs` supply integer dimensions for declarations
such as `A[num_obs, num_hidden_states]`. Each consumer defines the type of
its settings. For example, the [contingent T-maze](../../input/gnn_files/discrete/tmaze_epistemic.md)
declares the list `terminal_locations: [1, 2]`, which its execution contract
validates as absorbing location indices.

| Key | Values | Effect |
|-----|--------|--------|
| `num_factors` | int | `> 1` adds FACTORED to the detected kind set |
| `nr_agents` | int | `> 1` adds MULTI_AGENT to the detected kind set |
| `inference_mode` | `batch` (default) or `online` | RxInfer FLAT models: `batch` runs smoothing over the whole observation sequence; `online` runs per-timestep `infer()` on the observation prefix, so the filtered posterior drives expected-free-energy and habit-prior action selection. Any other value is rejected. |
| `inference_iterations` | int (default `20`) | Variational iterations per `infer()` call in generated RxInfer scripts. |

`inference_mode` may also be supplied as a render option; the value declared
in `ModelParameters` is what makes a file self-describing. See
[`src/gnn/render/rxinfer/README.md`](../../src/gnn/render/rxinfer/README.md) for the
per-kind strategy table.

---

## 2  Variable Declarations (`StateSpaceBlock`)

Each non-comment, non-blank line declares one variable:

```
NAME[dim₁, dim₂, …, key=value, …]   # optional comment
```

### Name rules

- Alphanumeric plus `_`, `π`, `'` (prime).
- Case-sensitive: `s` ≠ `S`.

### Dimension rules

- Comma-separated positive integers: `A[3,3]`.
- A trailing key-value pair `type=<type>` is optional and defaults to `float`; allowed values: `float`, `int`, `bool`.
- Dimensions may use named references: `A[num_obs, num_states]`.

### v1.1 Extensions — Default Values

Variables may carry a default-value hint after dimensions:

```
D[3, type=float, default=uniform]
W[4,4, type=float, default=zeros]
I[3,3, type=float, default=eye]
B[3,3,3, type=float, default=ones]
```

The parser stores `default=<value>` verbatim without validating the name; conventional initializers are `uniform`, `zeros`, `ones`, `eye`, `random`.

---

## 3  Connections

Each line in the `## Connections` section defines one directed or undirected edge:

| Syntax | Meaning | Example |
|--------|---------|---------|
| `A>B` | Causal / directed: A → B | `D>s` |
| `A-B` | Undirected / bidirectional | `s-A` |
| `A>B:label` | Annotated directed edge | `π>u:select_action` |
| `A-B:label` | Annotated undirected edge | `s-A:likelihood` |

### v1.1 Extension — Connection Annotations

Annotations appear after a colon following the edge:

```
D>s:prior_initialization
A-o:observation_mapping
G>π:policy_selection
```

Annotations are arbitrary strings (alphanumeric + `_`). They serve as labels
for rendering and documentation; parsers **must** accept and preserve them,
but **may** ignore them for structural validation.

---

## 4  Initial Parameterization

Matrix values use brace-delimited, comma-separated notation:

```
A={
  (0.9, 0.05, 0.05),
  (0.05, 0.9,  0.05),
  (0.05, 0.05, 0.9)
}
```

### Rules

1. Outer braces `{…}` wrap the full tensor.
2. Inner parentheses `(…)` group rows or slices.
3. Values are numeric literals (ints or floats).
4. Matrix dimensions **must** match the variable declaration in `StateSpaceBlock`
   (validator emits `GNNParseError` on mismatch).

### Parameterization families

These are common parameter families. A file may declare several families,
and the backend must admit the resulting contract.

| Family | Keys | Meaning |
|--------|------|---------|
| Discrete POMDP | `A`, `B`, `C`, `D`, `E` | Likelihood, transition, preferences, initial prior, habit prior. `B` is ordered `(next_state, previous_state, action)`. |
| Linear-Gaussian | `F`, `H`, `Q`, `R`, `prior_mean`, `prior_cov` | State transition, observation readout, process-noise covariance, observation-noise covariance, and the Gaussian prior over the initial latent state. |
| Dirichlet priors | `dirichlet_A` … `dirichlet_E` | Pseudo-counts for a matrix treated as a latent variable rather than a fixed constant. |
| Per-level / per-agent | `A_level1`, `B_level2`, …; `A_agent1`, `B_agent2`, … | Suffixed copies of the POMDP matrices, one set per hierarchy level or per agent. |
| Nonstationary transition | `B_t`, `B_regime` | Time-indexed or regime-indexed transitions. A regime-indexed model declares the active regimes through a schedule in `ModelParameters`. |

#### Continuous-state (linear-Gaussian) files

The [continuous navigation exemplar](../../input/gnn_files/continuous/continuous_navigation.md)
is a complete linear-Gaussian model with continuous states at discrete time
steps. It declares one variable per line in `StateSpaceBlock`, including each
matrix and vector supplied in `InitialParameterization`. In this model `F`
is the state-transition matrix. Some categorical exemplars use `F[1]` for a
scalar variational free energy readout.

The generative model is `x_1 ~ N(prior_mean, prior_cov)`,
`x_t ~ N(F x_{t-1} + u_{t-1}, Q)`, and `y_t ~ N(H x_t, R)`. With
`goal_mean` and `control_gain` the exemplar uses the filtered state mean
`μ_t` for proportional control, `u_t = control_gain · (goal_mean − μ_t)`.
Passive filtering uses zero control input.

The [continuous exemplar index](../../input/gnn_files/INDEX.md#continuous)
also covers factored, independent-agent, and mixed-family specifications.
See [backend admission](implementations/README.md#backend-admission) before
selecting a renderer for these compositions.

#### Dirichlet pseudo-counts

Declare the pseudo-count array in `StateSpaceBlock` alongside the matrix it
governs, then give its values in `InitialParameterization`:

```gnn
## StateSpaceBlock
dirichlet_A[3,3,type=float]   # prior counts for q(A)

## InitialParameterization
# rows = observations, columns = states; each column is one Dirichlet
dirichlet_A={
  (3.0, 1.0, 1.0),
  (1.0, 3.0, 1.0),
  (1.0, 1.0, 3.0)
}
```

Each column is a prior over the observation probabilities for one state.
Its concentration of 3 on the diagonal and 1 elsewhere favors a matching
observation while allowing other outcomes. Learning requires a backend
strategy that implements the declared prior. For example, the RxInfer
learning strategy treats `A` as a latent `DirichletCollection` and infers
it jointly with hidden states. The declaration is drawn from
[`input/gnn_files/learning/dirichlet_likelihood_learning.md`](../../input/gnn_files/learning/dirichlet_likelihood_learning.md).

#### Model-kind detection

The full kind set comes from typed fields such as the `GNNSection` value,
parameterization keys and model settings. `ModelName` and prose annotations
have no effect on classification.

| Kind | Selected by |
|------|-------------|
| `MULTI_AGENT` | `nr_agents > 1` or recognized per-agent categorical or continuous keys |
| `HYBRID` | A categorical contract key alongside a full `F/H/Q/R` block or both Gaussian prior keys |
| `HIERARCHICAL` | `hierarchical` in `GNNSection`, or any `[ABCDE]_level<N>` key |
| `CONTINUOUS` | `continuous` in `GNNSection`, a full `F/H/Q/R` block, both Gaussian prior keys, or recognized per-factor or per-agent continuous keys |
| `NONSTATIONARY` | `nonstationary` in `GNNSection`, time-indexed or regime-indexed transition keys, or a recognized schedule setting |
| `LEARNING` | `learning` in `GNNSection`, or any `dirichlet_[ABCDE]` key |
| `FACTORED` | `num_factors > 1` |
| `STRUCTURAL` | An explicit structural classification from the producer, or no matching family and no categorical contract keys |
| `FLAT` | At least one categorical contract key and no more specific kind |

`detect_model_kinds` preserves every matching family, except when the
producer has explicitly classified the spec as structural. The single-kind
helper `detect_model_kind` returns the highest-precedence member, using
the table order. Backend admission also checks the full set. For example,
`{HYBRID, CONTINUOUS}` identifies a model that the current renderers refuse
with `unsupported-composition`.

The implementation is in
[`pomdp_contract.py`](../../src/gnn/render/pomdp_contract.py).

#### B-tensor orientation (canonical vs. textbook POMDP)

The canonical transition tensor order is
`B[next_state, previous_state, action]` (= pymdp 1.0.0 `B[s',s,a]`). Each
**per-action slice is column-stochastic**: rows are *next states*, columns
are *previous states*, and each column (one previous state) sums to 1 over
next states. This matches the declared order in every canonical exemplar
under `input/gnn_files/` (some store the action axis innermost, some
outermost — the orientation contract is about the per-action slices, not
where the action axis sits).

Textbook POMDP literature frequently writes transition matrices the other
way around: rows = previous state `s_t`, columns = next state `s_{t+1}`,
each **row** summing to 1 (row-stochastic). A file imported in that
layout is read *transposed* by every canonical consumer — each transition
probability `P(next | prev)` is silently swapped for `P(prev | next)`,
which changes the model's dynamics without any error.

Step 6 (`src/gnn/6_validation.py`) therefore ships a default-on
orientation diagnostic (module `gnn.validation.orientation`):

| Per-action slices | Step 6 finding |
|-------------------|----------------|
| Column-stochastic (canonical) | silent |
| Row-stochastic only (textbook) | **warning** naming the tensor, state factor, and flipped slice indices |
| Doubly stochastic (rows *and* columns sum to 1) | informational note — orientation cannot be determined from the data |
| Neither | silent here; the existing stochasticity error paths own it |

Row margins use the type checker's stochasticity tolerance (`1e-6`), the
same constant behind the Step 5 `GNN-E002` orientation checks.

Fixing a flagged file means transposing each per-action slice
(rows ↔ columns) and storing the action axis last, so slices become
rows = next states, columns = previous states. To validate the fix
without editing the file, run the step with the opt-in transposition:

```bash
uv run python src/gnn/6_validation.py --target-dir input/gnn_files --transpose-b
```

`--transpose-b` applies the canonical transposition in memory (the same
mapping as `canonicalize_pomdp`: `canonical[n][p][a] = stored[a][p][n]`
for action-outer textbook literals) and records it per tensor in the
validation receipt (`transposed: true, previous_orientation:
row_stochastic, canonical_after_transpose: true`). Source files are never
modified. The same option is available on the `process_validation` MCP
tool.

---

## 5  Comments

- Lines starting with `#` (after optional whitespace) are comments.
- Inline comments: any `#` after a declaration.

---

## 6  Multi-Model Files (v1.1)

A single `.md` file may contain multiple models separated by a `---` (horizontal rule)
on its own line. Each model block must contain its own `## StateSpaceBlock`
(conventionally with its own `## GNNSection`).

---

## 7 ActInfOntologyAnnotation (v1.5)

To bind internal variables to external semantic meaning, construct a list mapping variables to CamelCase ontological states. This is requisite for the Neurosymbolic LLM Context Analysis features in `13_llm.py` and heuristics tracking in `24_intelligent_analysis.py`.

```gnn
## ActInfOntologyAnnotation
s=HiddenState
o=Observation
A=LikelihoodMatrix
B=TransitionMatrix
```

Terms are validated against
[`src/gnn/ontology/act_inf_ontology_terms.json`](../../src/gnn/ontology/act_inf_ontology_terms.json)
by Step 10. Note the conventional binding: `A` is the **likelihood**
(observation) matrix and `B` is the **transition** matrix — not the reverse.

---

## 8  Error Taxonomy

| Error Code | Meaning |
|------------|---------|
| `GNN-E001` | Missing required section |
| `GNN-E002` | Variable dimension mismatch (declaration vs parameterization) |
| `GNN-E003` | Unknown variable in connection (**reserved, not yet enforced** — `src/gnn/schema/parser.py` reports this condition as the `GNN-W002` warning instead) |
| `GNN-E004` | Duplicate variable declaration |
| `GNN-E005` | Unparseable connection syntax |

| Warning Code | Meaning |
|--------------|---------|
| `GNN-W001` | Variable declared but never used in connections (**planned, not yet enforced** — no emitting code in `src/gnn/schema/parser.py`) |
| `GNN-W002` | Connection references undeclared variable |
| `GNN-W003` | Parameterization provided for undeclared variable |
