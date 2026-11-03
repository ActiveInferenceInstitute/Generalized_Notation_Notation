# GNN Example: Composed Multi-Agent Continuous Navigation
# GNN Version: 1.0
# Composed spec: a linear-Gaussian (LGSSM) navigation model declared alongside
# nr_agents: 2 — the exemplar of the composed continuous × multi-agent kind set

## GNNSection
ActInfContinuous

## GNNVersionAndFlags
GNN v1

## ModelName
Composed Multi-Agent Continuous Navigation

## ModelAnnotation
A composed model: the continuous linear-Gaussian navigation block (F/H/Q/R
system matrices with a Gaussian prior and closed-loop goal control) is
declared together with an explicit multi-agent count (nr_agents: 2), so the
spec classifies as the composed kind set {CONTINUOUS, MULTI_AGENT}:

- Hidden state x = (x, y): the continuous 2D position shared by the declared
  agent population.
- Observation y: noisy readings of the 2D position (identity readout).
- Control input u: a goal-seeking command added to the state each step;
  u_t = control_gain * (goal_mean - mu_t) closes the loop on beliefs.
- nr_agents: 2 declares a two-agent composition of this continuous model.

Composed specs classify under BOTH families — `detect_model_kinds` returns
{CONTINUOUS, MULTI_AGENT} — and no framework renders the composition whole:
the render step reports every framework as `unsupported-composition` rather
than silently rendering the single-winner family with the other dropped.
Per-agent continuous parameterization exists for the independent-coupling
shape: `continuous/independent_gaussian_agents.md` declares
`agent_coupling: independent` with complete per-agent F/H/Q/R and prior
blocks and renders on JAX and RxInfer.jl. This exemplar deliberately keeps
the shared-block form, because the absence of a composer is itself the
boundary it documents: the mixed-family composition dimension is declared
by no frame, so every frame refuses the composed kind set instead of
guessing a per-agent decomposition.

## StateSpaceBlock
# Continuous latent state x = (x, y) position
x[2,1,type=float]        # continuous latent state
# Continuous observation y = noisy position readings
y[2,1,type=float]        # continuous observation
# Control input added to the state each step
u[2,1,type=float]        # control input
# State transition matrix (position persists between steps)
F[2,2,type=float]        # state transition
# Observation matrix (identity readout of the position)
H[2,2,type=float]        # observation matrix
# Process-noise covariance
Q[2,2,type=float]        # process-noise covariance
# Observation-noise covariance
R[2,2,type=float]        # observation-noise covariance
# Gaussian prior over the initial position
prior_mean[2,type=float] # prior mean over the initial latent state
prior_cov[2,2,type=float]# prior covariance over the initial latent state
# Preferred state the controller steers toward
goal_mean[2,type=float]  # preferred state (goal)
# Scalar proportional control gain
control_gain[1,type=float] # scalar proportional gain
# Time index
t[1,type=int]            # discrete time step

## Connections
prior_mean>x
F>x
x>y
H>y
Q>x
R>y
u>x
goal_mean>u
control_gain>u

## InitialParameterization
# F: position persists between steps; movement enters through the control input u.
F={
  (1.0, 0.0),
  (0.0, 1.0)
}

# H: identity readout — the observation reads the 2D position directly.
H={
  (1.0, 0.0),
  (0.0, 1.0)
}

# Q: process (motion) noise — modest slip per step.
Q={
  (0.05, 0.0),
  (0.0, 0.05)
}

# R: observation noise on the position readings.
R={
  (0.1, 0.0),
  (0.0, 0.1)
}

# Gaussian prior over the initial position: start at the origin, fairly certain.
prior_mean={(0.0, 0.0)}
prior_cov={
  (0.5, 0.0),
  (0.0, 0.5)
}

# goal_mean: the preferred position the closed-loop controller steers toward.
goal_mean={(2.0, 2.0)}

# control_gain: scalar proportional gain on (goal_mean - posterior_mean).
control_gain={(0.3)}

## Equations
# Generative model (linear-Gaussian state-space):
#   x_1 ~ N(prior_mean, prior_cov)
#   x_t = F x_{t-1} + u_{t-1} + N(0, Q)
#   y_t = H x_t + N(0, R)
# Closed-loop control on beliefs:
#   u_t = control_gain * (goal_mean - mu_t),  mu_t = current filtered posterior mean.

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=15

## ActInfOntologyAnnotation
F=StateTransitionMatrix
H=ObservationMatrix
Q=ProcessNoiseCovariance
R=ObservationNoiseCovariance
prior_mean=PriorMean
prior_cov=PriorCovariance
goal_mean=PreferredState
control_gain=ControlGain
x=ContinuousHiddenState
y=ContinuousObservation
u=ControlInput
t=Time

## ModelParameters
nr_agents: 2
num_timesteps: 15
dt: 0.1
random_seed: 42
num_states: 2
num_observations: 2

## Footer
Composed Multi-Agent Continuous Navigation v1 - a composed continuous x
multi-agent GNN model: the linear-Gaussian navigation block is declared
alongside nr_agents: 2. detect_model_kinds classifies it {CONTINUOUS,
MULTI_AGENT}; the render step receipts it as unsupported-composition on
every framework instead of silently rendering one family.

## Signature
Cryptographic signature goes here