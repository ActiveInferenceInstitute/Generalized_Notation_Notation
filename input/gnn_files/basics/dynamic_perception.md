# GNN Example: Dynamic Perception Model

# GNN Version: 1.0

# Passive temporal inference without action selection

## GNNSection

ActiveInferencePerception

## GNNVersionAndFlags

GNN v1

## ModelName

Dynamic Perception Model

## ModelAnnotation

A dynamic perception model extending the static model with temporal dynamics:

- 2 hidden states evolving over discrete time via transition matrix B
- 2 observations generated from states via likelihood matrix A
- Prior D constrains the initial hidden state
- No action selection — the agent passively observes a changing world
- Demonstrates belief updating (state inference) across time steps
- Suitable for tracking hidden sources from noisy observations

## StateSpaceBlock

# Generative model parameters

A[2,2,type=float]        # Likelihood matrix: P(observation | hidden state)
B[2,2,type=float]        # Transition matrix: P(s_{t+1} | s_t) — no action dependence
D[2,1,type=float]        # Prior over initial hidden states

# Hidden states

s_t[2,1,type=float]      # Hidden state belief at time t
s_prime[2,1,type=float]  # Hidden state belief at time t+1

# Observation

o_t[2,1,type=int]        # Observation at time t

# Inference quantities

F[1,type=float]          # Variational Free Energy (negative ELBO)

# Time

t[1,type=int]            # Discrete time index

## Connections

D>s_t
s_t-A
A-o_t
s_t-B
B>s_prime
s_t-F
o_t-F

## InitialParameterization

# Asymmetric sensor: P(o=0|s=0)=0.9; P(o=1|s=1)=0.8

A={
  (0.9, 0.2),
  (0.1, 0.8)
}

# Mildly persistent transitions (states tend to persist)

B={
  (0.7, 0.3),
  (0.3, 0.7)
}

# Uniform prior

D={(0.5, 0.5)}

## Equations

Exact online Bayesian filtering uses A[o,s] and B[next,previous].

r_0 = D
q_0(s) = A[o_0,s] D[s] / sum_j A[o_0,j] D[j]
r_t(s) = sum_j B[s,j] q_{t-1}(j), for t >= 1
q_t(s) = A[o_t,s] r_t(s) / sum_j A[o_t,j] r_t(j)

Thus q_t = P(s_t | o_0,...,o_t). Future observations do not enter this
filter. Forward-backward smoothing is a separate estimand, not this update.
The per-observation variational objective is
F_t(q) = sum_s q(s) log(q(s)/r_t(s)) - sum_s q(s) log A[o_t,s].
At the exact posterior its value is -log P(o_t | o_0,...,o_{t-1}); at t=0
the predictive prior is D. There is no factor of 1/2 in the filtering update.

## Time

Time=t
Dynamic
Discrete
ModelTimeHorizon=10

## ActInfOntologyAnnotation

A=LikelihoodMatrix
B=TransitionMatrix
D=Prior
s_t=HiddenState
s_prime=NextHiddenState
o_t=Observation
F=VariationalFreeEnergy
t=Time

## ModelParameters

num_hidden_states: 2
num_obs: 2
num_timesteps: 10

## Footer

Dynamic Perception Model v1 - GNN Representation.
Passive observer — no actions, no policies.
Demonstrates exact online Bayesian filtering without future evidence.

## Signature

Cryptographic signature goes here
