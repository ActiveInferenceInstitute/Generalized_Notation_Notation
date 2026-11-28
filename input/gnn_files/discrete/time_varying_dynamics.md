# GNN Example: Time-Varying Transition Dynamics

# GNN Version: 1.0

# Demonstrates a model whose transition matrix B changes over time
# (e.g., a non-stationary environment where dynamics evolve).

## GNNSection

ActInfPOMDP

## GNNVersionAndFlags

GNN v1

## ModelName

Time-Varying Transition Dynamics Agent

## ModelAnnotation

A POMDP agent operating in a non-stationary environment. The transition
model `B_t` is indexed by time: a 4-D tensor whose leading axis enumerates
declared transition phases over the planning horizon — e.g., shifting wind
patterns for a sailing agent, or changing opponent strategy in a
sequential game.

- 3 hidden states, 3 observations, 2 actions
- B_t: 4-D time-indexed tensor (declared phases × next_state × current_state × action)
- Four declared phases; timesteps beyond the declared span hold the last
  phase (documented hold-last semantics, recorded in execution results)
- The pymdp executor applies the phase sequence with a per-step Agent
  rebuild; renderers that cannot express time variation receipt the spec
  `unsupported-nonstationary` instead of rendering a static B

This sample pushes the language extensions around time-indexed tensors
and tests downstream code generation when matrix literals are
timestep-dependent. The regime-switched companion exemplar is
`regime_switched_dynamics.md`.

## StateSpaceBlock

# Generative model with time-varying dynamics

A[3,3,type=float]         # Observation model (time-invariant)
B_t[4,3,3,2,type=float]   # Transition model: 4 phases × (next × prev × action), indexed by t
C[3,1,type=float]         # Preference vector
D[3,1,type=float]         # Initial state prior

# Hidden state trajectory

s_t[3,1,type=float]       # Hidden state at time t
s_t+1[3,1,type=float]     # Hidden state at time t+1

# Observation and action

o_t[3,1,type=int]         # Observation at time t
u_t[2,1,type=int]         # Action at time t

## Connections

D>s_t
(s_t, u_t)>B_t
B_t>s_t+1
s_t-A
A-o_t
C-o_t

## InitialParameterization

# Time-invariant observation model (identity-like, mild noise)

A={
  (0.85, 0.10, 0.05),
  (0.10, 0.80, 0.10),
  (0.05, 0.10, 0.85)
}

# B_t: four declared transition phases (t=0 exploration-biased, t=1
# exploitation, t=2 reversed drift, t=3 quiescent). Timesteps beyond the
# declared span hold the last phase (documented hold-last semantics).
# The former phase-0 conditionals for action 1 and previous states 1 and 2
# had weights (0.1, 0.6, 0.1) and (0.8, 0.3, 0.1), with masses 0.8 and 1.2.
# Their declared probabilities preserve the relative weights.

B_t={
  (
    ((0.6, 0.1), (0.3, 0.125), (0.1, 0.6666666666666667)),
    ((0.3, 0.1), (0.6, 0.75), (0.1, 0.25)),
    ((0.1, 0.8), (0.1, 0.125), (0.8, 0.08333333333333333))
  ),
  (
    ((0.2, 0.1), (0.1, 0.1), (0.1, 0.1)),
    ((0.2, 0.1), (0.2, 0.1), (0.1, 0.1)),
    ((0.6, 0.8), (0.7, 0.8), (0.8, 0.8))
  ),
  (
    ((0.1, 0.2), (0.1, 0.2), (0.1, 0.2)),
    ((0.8, 0.2), (0.7, 0.2), (0.2, 0.2)),
    ((0.1, 0.6), (0.2, 0.6), (0.7, 0.6))
  ),
  (
    ((0.34, 0.34), (0.34, 0.34), (0.34, 0.34)),
    ((0.33, 0.33), (0.33, 0.33), (0.33, 0.33)),
    ((0.33, 0.33), (0.33, 0.33), (0.33, 0.33))
  )
}

# Preferences: goal state is state 2

C={(0.0, 0.0, 1.0)}

# Uniform prior

D={(0.33, 0.33, 0.34)}

## Equations

# Belief update under time-varying dynamics:
# Q(s_{t+1}) = softmax(ln(B_t[:, :, u_t] * Q(s_t)) + ln(A^T * o_{t+1}))
#
# Policy posterior accounts for B_t's evolution:
# Q(π) = softmax(-G(π))
# G(π) = Σ_t [KL(Q(s_t|π) || P(s_t|B_{t-1})) - E[ln(P(o_t|C))]]

## Time

Dynamic
DiscreteTime=t
ModelTimeHorizon=10

## ActInfOntologyAnnotation

A=LikelihoodMatrix
B_t=TimeVaryingTransitionMatrix
C=PreferenceVector
D=Prior
s_t=HiddenState
o_t=Observation
u_t=Action

## ModelParameters

num_hidden_states: 3
num_obs: 3
num_actions: 2
num_timesteps: 10
b_tensor_order_t: next_state_previous_state_action

## Footer

Time-Varying Transition Dynamics Agent v1.0 — demonstrates a non-stationary
time-indexed B_t tensor (4 declared phases) in a 3-state POMDP. Tests
per-phase transition handling and hold-last scheduling in the executor.

## Signature

Cryptographic signature goes here
