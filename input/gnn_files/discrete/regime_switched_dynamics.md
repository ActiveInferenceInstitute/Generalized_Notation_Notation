# GNN Example: Regime-Switched Transition Dynamics

# GNN Version: 1.0

# Demonstrates a model whose transition matrix B switches between a finite
# set of regimes according to an explicit schedule.

## GNNSection

ActInfPOMDP

## GNNVersionAndFlags

GNN v1

## ModelName

Regime-Switched Transition Dynamics Agent

## ModelAnnotation

A POMDP agent in an environment that switches between two transition
regimes. The transition tensor `B_regime` holds one
(next_state × current_state × action) tensor per regime, and
`b_regime_schedule` in ModelParameters names the active regime for every
timestep of the planning horizon — the switching exemplar of the
NONSTATIONARY model kind.

- 3 hidden states, 3 observations, 2 actions
- B_regime: 4-D tensor (2 regimes × 3 × 3 × 2)
- b_regime_schedule: regime 0 for t=0..3, regime 1 for t=4..7
- The regime index is exogenous (declared, not inferred)
- The pymdp executor applies the schedule with a per-step Agent rebuild;
  renderers that cannot express time variation receipt the spec
  `unsupported-nonstationary` instead of rendering a static B

## StateSpaceBlock

# Generative model with regime-switched dynamics

A[3,3,type=float]             # Observation model (time-invariant)
B_regime[2,3,3,2,type=float]  # Transition model: one tensor per regime
C[3,1,type=float]             # Preference vector
D[3,1,type=float]             # Initial state prior

# Hidden state trajectory

s_t[3,1,type=float]       # Hidden state at time t
s_t+1[3,1,type=float]     # Hidden state at time t+1

# Observation and action

o_t[3,1,type=int]         # Observation at time t
u_t[2,1,type=int]         # Action at time t

## Connections

D>s_t
(s_t, u_t)>B_regime
B_regime>s_t+1
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

# B_regime: regime 0 (calm) and regime 1 (storm). Each regime slice is
# (next_state × current_state × action) and column-stochastic.
# The two state axes are transposed from the earlier stored row-stochastic
# tables. This preserves every probability and puts next state first.

B_regime={
  (
    ((0.7, 0.1), (0.2, 0.1), (0.1, 0.1)),
    ((0.2, 0.1), (0.7, 0.1), (0.1, 0.1)),
    ((0.1, 0.8), (0.1, 0.8), (0.8, 0.8))
  ),
  (
    ((0.2, 0.4), (0.4, 0.2), (0.3, 0.1)),
    ((0.6, 0.2), (0.2, 0.4), (0.3, 0.1)),
    ((0.2, 0.4), (0.4, 0.4), (0.4, 0.8))
  )
}

# Preferences: goal state is state 2

C={(0.0, 0.0, 1.0)}

# Uniform prior

D={(0.33, 0.33, 0.34)}

## Equations

# Belief update under the active regime r(t):
# Q(s_{t+1}) = softmax(ln(B_regime[r(t)][:, :, u_t] * Q(s_t)) + ln(A^T * o_{t+1}))
#
# The regime index is exogenous: r(t) = b_regime_schedule[t], not inferred.

## Time

RegimeSwitched
DiscreteTime=t
ModelTimeHorizon=8

## ActInfOntologyAnnotation

A=LikelihoodMatrix
B_regime=RegimeSwitchedTransitionMatrix
C=PreferenceVector
D=Prior
s_t=HiddenState
o_t=Observation
u_t=Action

## ModelParameters

num_hidden_states: 3
num_obs: 3
num_actions: 2
num_timesteps: 8
b_tensor_order_regime: next_state_previous_state_action
b_regime_schedule: 0,0,0,0,1,1,1,1

## Footer

Regime-Switched Transition Dynamics Agent v1.0 — demonstrates B_regime +
b_regime_schedule switching in a 3-state POMDP. The pymdp executor applies
the schedule with a per-step Agent rebuild.

## Signature

Cryptographic signature goes here
