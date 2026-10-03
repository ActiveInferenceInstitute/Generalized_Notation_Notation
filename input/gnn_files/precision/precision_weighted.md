# GNN Example: Precision-Weighted Active Inference Agent
# GNN Version: 1.0
# POMDP agent with explicit sensory and policy precision parameters.

## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
Precision-Weighted Active Inference Agent

## ModelAnnotation
An Active Inference agent with explicit precision parameters:
- ω (omega): sensory precision weighting likelihood confidence
- γ (gamma): policy precision controlling action randomness
- β (beta): inverse temperature for policy selection (softmax)
- 3 hidden states, 3 observations, 3 actions (same topology as base POMDP)
- Precision parameters enable modeling of attention and confidence

## StateSpaceBlock
# Core model matrices
A[3,3,type=float]      # Likelihood matrix (modulated by ω)
B[3,3,3,type=float]    # Transition matrix
C[3,type=float]        # Log-preferences over observations
D[3,type=float]        # Prior over hidden states
E[3,type=float]        # Habit (prior over actions)

# State and observation
s[3,1,type=float]      # Hidden state distribution
s_prime[3,1,type=float] # Next hidden state
o[3,1,type=int]        # Current observation
F[π,type=float]        # Variational Free Energy

# Precision parameters
ω[1,type=float]        # Sensory precision (modulates A matrix confidence)
γ[1,type=float]        # Policy precision (temperature for action selection)
β[1,type=float]        # Inverse temperature (β = 1/γ, controls randomness)

# Policy and EFE
π[3,type=float]        # Policy distribution
u[1,type=int]          # Selected action
G[π,type=float]        # Expected Free Energy

## Connections
D>s
s-A
A-o
ω>A
s>s_prime
C>G
G>π
γ>π
β>π
E>π
π>u
B>u
u>s_prime
s-F
o-F
ω-F

## InitialParameterization
A={
  (0.9, 0.05, 0.05),
  (0.05, 0.9, 0.05),
  (0.05, 0.05, 0.9)
}

B={
  ( (1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0) ),
  ( (0.0,1.0,0.0), (1.0,0.0,0.0), (0.0,0.0,1.0) ),
  ( (0.0,0.0,1.0), (0.0,1.0,0.0), (1.0,0.0,0.0) )
}

C={(0.1, 0.1, 1.0)}
# v4 precision correction: the three equal entries declare uniform prior and
# habit distributions. The previous (0.333, 0.333, 0.333) had mass 0.999 and
# was not a probability distribution. Use repr(1/3) in both vectors; do not
# normalize malformed source distributions silently at render time.
D={(0.3333333333333333, 0.3333333333333333, 0.3333333333333333)}
E={(0.3333333333333333, 0.3333333333333333, 0.3333333333333333)}

# High sensory precision: agent trusts its observations
ω={(4.0)}

# Moderate policy precision: neither too random nor too deterministic
γ={(2.0)}
β={(0.5)}

## Equations
# Precision-weighted likelihood: A_ω = softmax(ω * log A)
# Precision-weighted policy: π = softmax(-β * G + log E)
# Standard VFE: F = D_KL[Q(s)||D] - E_Q[ω * log P(o|s)]

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
s=HiddenState
s_prime=NextHiddenState
o=Observation
π=PolicyVector
u=Action
G=ExpectedFreeEnergy
F=VariationalFreeEnergy
ω=SensoryPrecision
γ=PolicyPrecision
β=InverseTemperature
t=Time

## ModelParameters
num_hidden_states: 3
num_obs: 3
num_actions: 3
sensory_precision: 4.0
policy_precision: 2.0
num_timesteps: 30

## Footer
Precision-Weighted Active Inference Agent v1 - GNN Representation.
Demonstrates attentional modulation via sensory precision ω.
Policy precision γ controls exploitation vs exploration trade-off.

## Signature
Cryptographic signature goes here
