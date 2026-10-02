# GNN Example: Two-State Bistable POMDP

# GNN Version: 1.0

# The minimal possible POMDP: 2 hidden states, 2 observations, 2 actions

# Tests edge-case handling for minimal state spaces

## GNNSection

ActInfPOMDP

## GNNVersionAndFlags

GNN v1

## ModelName

Two State Bistable POMDP

## ModelAnnotation

This model describes a minimal 2-state bistable POMDP:

- 2 hidden states: "left" and "right" in a symmetric bistable potential.
- 2 noisy observations: the agent gets a noisy readout of which side it is on.
- 2 actions: push-left or push-right.
- The agent prefers observation 1 ("right") over observation 0 ("left").
- Tests the absolute smallest POMDP with full active inference structure.

## StateSpaceBlock

# Likelihood matrix: A[observations, hidden_states]

A[2,2,type=float]     # Noisy observation of state

# Transition matrix: B[next_state, previous_state, actions]

B[2,2,2,type=float]   # Action-dependent transitions

# Preference vector: C[observations]

C[2,type=float]       # Prefer right side

# Prior vector: D[states]

D[2,type=float]       # Prior over initial states

# Habit vector: E[actions]

E[2,type=float]       # Prior over actions

# Hidden State

s[2,1,type=float]     # Current state belief
s_prime[2,1,type=float] # Next state belief

# Observation

o[2,1,type=int]       # Current observation

# Policy and Control

π[2,type=float]       # Policy over actions
u[1,type=int]         # Chosen action
G[π,type=float]       # Expected Free Energy

# Time

t[1,type=int]         # Discrete time step

## Connections

D>s
s-A
A-o
s>s_prime
s-B
C>G
E>π
G>π
π>u
B>u
u>s_prime

## InitialParameterization

# A: 2x2 with noise — 80% accurate observation

A={
  (0.8, 0.2),
  (0.2, 0.8)
}

# B: 2 actions. Action 0 = push left, action 1 = push right. The outer B blocks are actions, each matrix stores (next_state, previous_state); ModelParameters declares action_next_state_previous_state explicitly. per-action slices are column-stochastic: rows are next states, columns are previous states, and each column sums to 1 over next states.

B={
  ( (0.8, 0.3), (0.2, 0.7) ),
  ( (0.3, 0.8), (0.7, 0.2) )
}

# C: Prefer observation 1 (right)

C={(0.0, 2.0)}

# D: Start uncertain

D={(0.5, 0.5)}

# E: No habit bias

E={(0.5, 0.5)}

## Equations

# Active Inference POMDP updates

# qs = infer_states(observation) — Bayesian belief update

# G(pi) = EFE(pi) — Expected Free Energy per policy

# u ~ softmax(-G) — Action selection

## Time

Time=t
Dynamic
Discrete
ModelTimeHorizon=Unbounded

## ActInfOntologyAnnotation

A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates
E=Habit
G=ExpectedFreeEnergy
s=HiddenState
s_prime=NextHiddenState
o=Observation
π=PolicyVector
u=Action
t=Time

## ModelParameters

# Axis correction: preserve every numeric transition value and transpose only
# from the declared action blocks; no probability-mass repair is allowed.
b_tensor_order: action_next_state_previous_state

num_hidden_states: 2
num_obs: 2
num_actions: 2
num_timesteps: 20

## Footer

Two State Bistable POMDP v1 - GNN Representation.
Minimal 2x2x2 POMDP. Tests edge cases for smallest possible active inference agent.

## Signature

Cryptographic signature goes here
