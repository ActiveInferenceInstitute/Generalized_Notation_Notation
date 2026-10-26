# E4-IG1: independent passive Gaussian composition

This additive contract extends independent numerical extraction to exactly two
source-declared Gaussian agents. It preserves the released single-factor witness
contract. It does not admit coupled dynamics, shared noise, closed-loop control,
smoothing, or general translation equivalence.

The authored source uses the same seven explicit witness sections as the
single-factor contract and JSON assignments, with `_agent1` and `_agent2` keys.
Its `NumericalWitness` declaration names
`gnn-independent-gaussian-composition-v1`, `model: gaussian_product`,
`question: filtering`, float64, prior-first timing, canonical F/H axes,
ordered agent IDs and per-agent state/observation labels. ModelParameters must
declare exactly two agents, independent coupling, the common timestep count and
both native seeds. At most eight prefixes and four total state/observation axes
are admitted. Every agent must declare a complete passive F/H/Q/R/prior block,
finite nonboolean rectangular values and positive definite covariances.

The independent reference assembles block-diagonal prior/dynamics/noise and
conditions the full joint Gaussian at each observation prefix. It compares every
per-agent mean and full within-agent covariance against native JAX filtering.
Cross-agent zero covariance belongs to the explicitly declared independent
reference; it is not represented as an observed native cross-covariance.
The JAX wrapper preserves complete native child results in `agent_results`, in
addition to its existing array projections. No expected metadata is copied into
native output.

The `gnn-semantic-witness-v3` receipt retains the exact source, result and optional
program bytes, independent reference, component identity/parameter comparisons
and numerical diagnostics. Replay rebuilds the whole receipt. Execution-source
verification remains false and formal evidence remains unavailable. Existing
v2 receipts keep their original representation and replay behavior.

Acceptance requires asymmetric agent dimensions and observation maps, agreement
with a separate recursive Kalman/Joseph oracle, a real emitted JAX program
when the native profile is scheduled, complete raw-artifact replay, agent
permutation consistency, and preservation of the existing single-factor tests.
Negative controls include duplicate/missing/extra agent blocks, shared/coupled
parameters, controls, smoothing, changed seeds/parameters/timing/precision,
missing/altered child metadata, nonfinite/boolean values, invalid covariance,
future-observation leakage, source/result/reference edits and resource bounds.

## Complete authored source

The following source is the asymmetric native acceptance fixture: agent 1 has
one state and two observations; agent 2 has two states and one observation.
Save it as a model source before rendering. The witness grammar intentionally
requires JSON literals and the explicit product equations; general GNN tuple
notation, commentary sections and undeclared dynamics remain outside this
contract.

```markdown
## ModelName
Asymmetric independent Gaussian witness

## GNNSection
ActInfContinuousMultiAgent

## StateSpaceBlock
x_agent1[1,1,type=float]
y_agent1[2,1,type=float]
x_agent2[2,1,type=float]
y_agent2[1,1,type=float]

## InitialParameterization
F_agent1 = [[0.85]]
H_agent1 = [[1.25], [-0.4]]
Q_agent1 = [[0.04]]
R_agent1 = [[0.12, 0.015], [0.015, 0.2]]
prior_mean_agent1 = [-1.0]
prior_cov_agent1 = [[0.3]]
F_agent2 = [[0.9, 0.15], [0.0, 0.8]]
H_agent2 = [[0.75, -1.1]]
Q_agent2 = [[0.03, 0.005], [0.005, 0.06]]
R_agent2 = [[0.2]]
prior_mean_agent2 = [2.0, -0.5]
prior_cov_agent2 = [[0.4, 0.08], [0.08, 0.5]]

## ModelParameters
nr_agents: 2
agent_coupling: independent
num_timesteps: 4
random_seed_agent1: 17
random_seed_agent2: 93

## Equations
p(x_agent1,x_agent2,y_agent1,y_agent2)=p(x_agent1,y_agent1)*p(x_agent2,y_agent2); x_agent1_0~N(prior_mean_agent1,prior_cov_agent1); x_agent1_(t+1)=F_agent1*x_agent1_t+N(0,Q_agent1); y_agent1_t=H_agent1*x_agent1_t+N(0,R_agent1); x_agent2_0~N(prior_mean_agent2,prior_cov_agent2); x_agent2_(t+1)=F_agent2*x_agent2_t+N(0,Q_agent2); y_agent2_t=H_agent2*x_agent2_t+N(0,R_agent2)

## NumericalWitness
{
  "schema": "gnn-independent-gaussian-composition-v1",
  "model": "gaussian_product",
  "question": "filtering",
  "dtype": "float64",
  "initial_time": "first_observation_uses_prior",
  "axes": {
    "F": [
      "next_state",
      "previous_state"
    ],
    "H": [
      "observation",
      "state"
    ]
  },
  "agents": [
    "agent1",
    "agent2"
  ],
  "state_ids": {
    "agent1": [
      "x0"
    ],
    "agent2": [
      "x0",
      "x1"
    ]
  },
  "observation_ids": {
    "agent1": [
      "y0",
      "y1"
    ],
    "agent2": [
      "y0"
    ]
  }
}
```

Use the existing `gnn witness SOURCE --results RESULT --program PROGRAM`
command on the emitted JAX result and program. `gnn witness --verify RECEIPT`
recomputes retained numerical evidence; it does not execute the program.
The focused native test is
`tests/validation/test_semantic_composition.py::test_native_jax_composition_retains_children_and_matches_independent_reference`.
Its optional `GNN_NUMERICAL_WITNESS_DIR` destination retains a complete v3 receipt
with source, program and result bytes for separate qualification.
