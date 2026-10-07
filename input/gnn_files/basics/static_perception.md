# GNN Example: Static Perception Model

# GNN Version: 1.0

# Simplest possible GNN model: perception with a minimal action/transition component
# (action/transition added for POMDP/pymdp rendering compatibility)

## GNNSection

ActInfPOMDP

## GNNVersionAndFlags

GNN v1

## ModelName

Static Perception Model

## ModelAnnotation

The simplest Active Inference model demonstrating pure perception:

- 2 hidden states mapped to 2 observations via a likelihood matrix A
- Prior D encodes initial beliefs over hidden states
- Minimal 2-action transition component B so the model is a complete POMDP
  for optional compatibility; perception itself requires no action planning
- Suitable as a minimal baseline and for testing perception-only inference

## StateSpaceBlock

# Generative model parameters

A[2,2,type=float]    # Likelihood matrix: P(observation | hidden state)
B[2,2,2,type=float]  # Transition matrix: B[next_state, previous_state, actions]
C[2,type=float]      # Preference vector over observations
D[2,1,type=float]    # Prior belief over hidden states

# Hidden state

s[2,1,type=float]    # Hidden state (posterior belief)

# Observation

o[2,1,type=int]      # Observation (one-hot encoded)

# Action

u[1,type=int]        # Action taken

## Connections

D>s
s-A
A-o
s-B
B>u
u>s

## InitialParameterization

# Asymmetric sensor: P(o=0|s=0)=0.9; P(o=1|s=1)=0.8

A={
  (0.9, 0.2),
  (0.1, 0.8)
}

# Minimal transitions: action 0 = stay, action 1 = flip state. The transition tensor B is stored as (next_state, previous_state, action); per-action slices are column-stochastic: rows are next states, columns are previous states, and each column sums to 1 over next states.

B={
  ( (0.95, 0.05), (0.05, 0.95) ),
  ( (0.05, 0.95), (0.95, 0.05) )
}

# Neutral preference over observations

C={(0.0, 0.0)}

# Uniform prior over hidden states

D={(0.5, 0.5)}

## Equations

A[o,s] = P(o | s). For the single observation o, Bayesian perception is
q(s) = D[s] A[o,s] / sum_j D[j] A[o,j]. Equivalently, for one-hot o,
q = softmax(log D + log(A^T o)). A is a likelihood, not a posterior Q(s|o).

The optional transition component uses B[next,previous,action]. Action 0
stays with probability 0.95; action 1 flips with probability 0.95.
For a chosen action u, the predicted next-state belief is
q_next = B[:,:,u] q.

An actual next state is sampled according to
P(s_next | s,u) = B[:,s,u].
Neutral C does not make
active planning necessary for this static perception baseline.

## Time

Static

## ActInfOntologyAnnotation

A=LikelihoodMatrix
B=TransitionMatrix
C=PreferenceVector
D=Prior
s=HiddenState
o=Observation
u=Action

## ModelParameters
b_tensor_order: next_state_previous_state_action

num_hidden_states: 2
num_obs: 2
num_actions: 2
num_timesteps: 5

## Footer

Static Perception Model v1 - GNN Representation.
Simplest possible Active Inference model.
Minimal action/transition component added for POMDP/pymdp compatibility.

## Signature

Cryptographic signature goes here
