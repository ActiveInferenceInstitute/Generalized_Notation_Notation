# GNN Example: Hidden Markov Model Baseline
# GNN Version: 1.0
# Simple HMM for comparison with Active Inference POMDP variants.

## GNNSection
HiddenMarkovModel

## GNNVersionAndFlags
GNN v1

## ModelName
Hidden Markov Model Baseline

## ModelAnnotation
A standard discrete Hidden Markov Model with:
- 4 hidden states with Markovian dynamics
- 6 mutually exclusive observation symbols, one sampled per timestep
- An 80% identity channel: 70% correct state label, 10% each other label
- A 20% coarse-group channel: 80% correct group, 20% opposite group
- Symbol 4 denotes states 2/3; symbol 5 denotes states 0/1
- Group symbols are alternatives to identity symbols, not simultaneous observations
- Fixed transition and emission matrices
- No action selection (passive inference only)
- Suitable for sequence modeling and state estimation tasks

## StateSpaceBlock
# HMM parameters
A[6,4,type=float]      # Emission matrix: observations x hidden states
B[4,4,type=float]      # Transition matrix (no action dependence)
D[4,type=float]        # Initial state distribution (prior)

# State and observations
s[4,1,type=float]      # Hidden state belief (posterior)
s_prime[4,1,type=float] # Next hidden state
o[6,1,type=int]        # Current observation (one-hot)

# Inference quantities
F[1,type=float]        # Variational Free Energy (negative ELBO)
alpha[4,1,type=float]  # Forward variable (belief propagation)
beta[4,1,type=float]   # Backward variable

# Time
t[1,type=int]          # Discrete time step

## Connections
D>s
s-A
s>s_prime
A-o
B>s_prime
s-B
s-F
o-F
s-alpha
o-alpha
alpha>s_prime
s_prime-beta

## InitialParameterization
# Emission: 6 observations, 4 states
A={
  (0.56, 0.08, 0.08, 0.08),
  (0.08, 0.56, 0.08, 0.08),
  (0.08, 0.08, 0.56, 0.08),
  (0.08, 0.08, 0.08, 0.56),
  (0.04, 0.04, 0.16, 0.16),
  (0.16, 0.16, 0.04, 0.04)
}

# Transition: 4x4 column stochastic
B={
  (0.7, 0.1, 0.1, 0.1),
  (0.1, 0.7, 0.2, 0.1),
  (0.1, 0.1, 0.6, 0.2),
  (0.1, 0.1, 0.1, 0.6)
}

D={(0.25, 0.25, 0.25, 0.25)}

## Equations

The emission mechanism is a mixture. P(o=i|s)=0.8 times the identity-channel
conditional for i=0,...,3, and 0.2 times the group-channel conditional for
i=4,5. Each column of A sums to one; B[next,previous] is column-stochastic.

For observations o_0,...,o_{T-1}, unnormalized forward messages are
alpha_0(s) = D[s] A[o_0,s]
alpha_t(s) = A[o_t,s] sum_j B[s,j] alpha_{t-1}(j).
Filtering is q_filter,t(s) = alpha_t(s) / sum_j alpha_t(j).

Backward messages are beta_{T-1}(s)=1 and
beta_t(s) = sum_j B[j,s] A[o_{t+1},j] beta_{t+1}(j).
Smoothing is q_smooth,t(s) = alpha_t(s) beta_t(s) / sum_j alpha_t(j) beta_t(j).
Filtering conditions only on observations through t; smoothing uses the full
sequence. Each requires its own independent reference when testing an adapter.
Messages may be rescaled to avoid underflow if evidence normalizers are retained.
F = -log P(o_0,...,o_{T-1}) = -log sum_s alpha_{T-1}(s) for unscaled alpha.
No actions, policies, or control objective are introduced.

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=Unbounded

## ActInfOntologyAnnotation
A=EmissionMatrix
B=TransitionMatrix
D=InitialStateDistribution
s=HiddenState
s_prime=NextHiddenState
o=Observation
F=VariationalFreeEnergy
alpha=ForwardVariable
beta=BackwardVariable
t=Time

## ModelParameters
num_hidden_states: 4
num_observations: 6
num_timesteps: 50

## Footer
Hidden Markov Model Baseline v1 - GNN Representation.
No action selection — passive observer only.
Use as baseline comparison for POMDP Active Inference variants.

## Signature
Cryptographic signature goes here
