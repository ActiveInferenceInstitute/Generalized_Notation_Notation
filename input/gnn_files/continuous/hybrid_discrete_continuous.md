# GNN Example: Hybrid Discrete-Continuous POMDP

# GNN Version: 1.0

# Composed spec: a minimal discrete POMDP declared alongside a 2-dim
# continuous linear-Gaussian block — the exemplar of the hybrid ×
# continuous kind set

## GNNSection

ActInfPOMDP

## GNNVersionAndFlags

GNN v1

## ModelName

Hybrid Discrete-Continuous POMDP

## ModelAnnotation

This model describes a minimal hybrid agent: a 2-state discrete POMDP
declared in the SAME parameterization as a passive 2-dim continuous
linear-Gaussian (LGSSM) block:

- 2 hidden states: "left" and "right" in a symmetric bistable potential.
- 2 noisy observations: the agent gets a noisy readout of which side it is on.
- 1 action: a single push that perturbs the discrete transition.
- A continuous block (F/H/Q/R with a Gaussian prior) is declared alongside
  the discrete A/B/C/D matrices; the continuous state evolves passively and
  shares no variables with the discrete POMDP.

Hybrid specs classify under BOTH families — `detect_model_kinds` returns
{HYBRID, CONTINUOUS} — and no framework renders the mix whole: the render
step reports every framework as `unsupported-composition` (continuous x
hybrid) rather than silently rendering the spec as discrete-only or
continuous-only. Per-framework hybrid rendering is future work on the
renderer side: the mixed-family composition dimension is declared by no
frame today, so this refusal is a known backend gap, not a missing
declaration.

## StateSpaceBlock

# Likelihood matrix: A[observations, hidden_states]

A[2,2,type=float]     # Noisy observation of state

# Transition matrix: B[next_state, previous_state, actions]

B[2,2,1,type=float]   # Action-dependent transitions

# Preference vector: C[observations]

C[2,type=float]       # Prefer right side

# Prior vector: D[states]

D[2,type=float]       # Prior over initial states

# Hidden State

s[2,1,type=float]     # Current state belief
s_prime[2,1,type=float] # Next state belief

# Observation

o[2,1,type=int]       # Current observation

# Policy and Control

π[2,type=float]       # Policy over actions
u[1,type=int]         # Chosen action
G[π,type=float]       # Expected Free Energy

# Continuous linear-Gaussian block: declared alongside the discrete POMDP;
# this is what makes the spec hybrid. Passive: no continuous control input.

x[2,1,type=float]        # continuous latent state
y[2,1,type=float]        # continuous observation
F[2,2,type=float]        # state transition
H[2,2,type=float]        # observation matrix
Q[2,2,type=float]        # process-noise covariance
R[2,2,type=float]        # observation-noise covariance
prior_mean[2,type=float] # prior mean over the initial latent state
prior_cov[2,2,type=float]# prior covariance over the initial latent state

# Time

t[1,type=int]         # Discrete time step

## Connections

D>s
s-A
A-o
s>s_prime
s-B
C>G
G>π
π>u
B>u
u>s_prime
prior_mean>x
F>x
x>y
H>y
Q>x
R>y

## InitialParameterization

# A: 2x2 with noise — 80% accurate observation

A={
  (0.8, 0.2),
  (0.2, 0.8)
}

# B: 1 action. The transition tensor B is stored as (next_state, previous_state, action); per-action slices are column-stochastic: rows are next states, columns are previous states, and each column sums to 1 over next states.

B={
  ( (0.8, 0.3), (0.2, 0.7) )
}

# C: Prefer observation 1 (right)

C={(0.0, 2.0)}

# D: Start uncertain

D={(0.5, 0.5)}

# F: Euler discretization (dt = 0.1) of dx/dt = F x with mild velocity damping.

F={
  (1.0, 0.1),
  (0.0, 0.9)
}

# H: two noisy position readouts (both observation channels read position).

H={
  (1.0, 0.0),
  (1.0, 0.0)
}

# Q: process-noise covariance = gamma_state^-1 * I (gamma_state = 10.0).

Q={
  (0.1, 0.0),
  (0.0, 0.1)
}

# R: observation-noise covariance = gamma_obs^-1 * I (gamma_obs = 5.0).

R={
  (0.2, 0.0),
  (0.0, 0.2)
}

# Gaussian prior over the initial continuous state: position fairly certain,
# velocity less so.

prior_mean={(0.0, 0.0)}
prior_cov={
  (0.5, 0.0),
  (0.0, 1.0)
}

## Equations

# Discrete POMDP updates

# qs = infer_states(observation) — Bayesian belief update

# G(pi) = EFE(pi) — Expected Free Energy per policy

# u ~ softmax(-G) — Action selection

# Continuous linear-Gaussian block (passive, independent of the POMDP):

#   x_1 ~ N(prior_mean, prior_cov)

#   x_t = F x_{t-1} + N(0, Q)      (passive: no control input)

#   y_t = H x_t + N(0, R)

## Time

Time=t
Dynamic
Discrete
ModelTimeHorizon=10

## ActInfOntologyAnnotation

A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates
G=ExpectedFreeEnergy
s=HiddenState
s_prime=NextHiddenState
o=Observation
π=PolicyVector
u=Action
F=StateTransitionMatrix
H=ObservationMatrix
Q=ProcessNoiseCovariance
R=ObservationNoiseCovariance
prior_mean=PriorMean
prior_cov=PriorCovariance
x=ContinuousHiddenState
y=ContinuousObservation
t=Time

## ModelParameters

num_states: 2
num_observations: 2
num_actions: 1
num_timesteps: 10

## Footer

Hybrid Discrete-Continuous POMDP v1 - a composed hybrid x continuous GNN
model: the minimal 2-state POMDP is declared alongside a passive 2-dim
linear-Gaussian block. detect_model_kinds classifies it {HYBRID,
CONTINUOUS}; every framework receipts it unsupported-composition
(continuous x hybrid) instead of silently rendering one family.

## Signature

Cryptographic signature goes here
