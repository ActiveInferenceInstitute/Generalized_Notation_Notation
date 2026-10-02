# GNN Example: Independent Asymmetric Gaussian Agents

## GNNSection
ActInfContinuousMultiAgent

## GNNVersionAndFlags
GNN v1

## ModelName
Independent Asymmetric Gaussian Agents

## ModelAnnotation
Two independent Gaussian agents with distinct dimensions, dynamics, priors,
observations, and goals. Agent 1 has one scalar state; agent 2 has two states
and one scalar observation. Each receives its own noise stream and computes
its own proportional control from its own posterior. There is no shared
state, interaction, communication, or joint categorical distribution.

## StateSpaceBlock
x_agent1[1,type=float]
y_agent1[1,type=float]
u_agent1[1,type=float]
F_agent1[1,1,type=float]
H_agent1[1,1,type=float]
Q_agent1[1,1,type=float]
R_agent1[1,1,type=float]
prior_mean_agent1[1,type=float]
prior_cov_agent1[1,1,type=float]
goal_mean_agent1[1,type=float]
control_gain_agent1[1,type=float]
x_agent2[2,type=float]
y_agent2[1,type=float]
u_agent2[2,type=float]
F_agent2[2,2,type=float]
H_agent2[1,2,type=float]
Q_agent2[2,2,type=float]
R_agent2[1,1,type=float]
prior_mean_agent2[2,type=float]
prior_cov_agent2[2,2,type=float]
goal_mean_agent2[2,type=float]
control_gain_agent2[1,type=float]
t[1,type=int]

## Connections
prior_mean_agent1>x_agent1
F_agent1>x_agent1
x_agent1>y_agent1
H_agent1>y_agent1
goal_mean_agent1>u_agent1
u_agent1>x_agent1
prior_mean_agent2>x_agent2
F_agent2>x_agent2
x_agent2>y_agent2
H_agent2>y_agent2
goal_mean_agent2>u_agent2
u_agent2>x_agent2

## InitialParameterization
F_agent1={((0.85))}
H_agent1={((1.0))}
Q_agent1={((0.04))}
R_agent1={((0.12))}
prior_mean_agent1={(-1.0)}
prior_cov_agent1={((0.3))}
goal_mean_agent1={(1.5)}
control_gain_agent1={(0.25)}
F_agent2={ (0.9, 0.15), (0.0, 0.8) }
H_agent2={((1.0, -0.4))}
Q_agent2={ (0.03, 0.005), (0.005, 0.06) }
R_agent2={((0.2))}
prior_mean_agent2={(2.0, -0.5)}
prior_cov_agent2={ (0.4, 0.08), (0.08, 0.5) }
goal_mean_agent2={(-1.0, 0.75)}
control_gain_agent2={(0.15)}

## Equations
# p(x_agent1, x_agent2, y_agent1, y_agent2) factorizes over agents.
# x_i,1 ~ Normal(prior_mean_i, prior_cov_i)
# x_i,t = F_i x_i,t-1 + u_i,t-1 + Normal(0, Q_i)
# y_i,t = H_i x_i,t + Normal(0, R_i)
# u_i,t = control_gain_i * (goal_mean_i - posterior_mean_i,t)

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=4

## ActInfOntologyAnnotation
x_agent1=ContinuousHiddenState
x_agent2=ContinuousHiddenState
y_agent1=ContinuousObservation
y_agent2=ContinuousObservation
u_agent1=ControlInput
u_agent2=ControlInput

## ModelParameters
nr_agents: 2
agent_coupling: independent
num_timesteps: 4
dt: 0.1
random_seed: 42
random_seed_agent1: 17
random_seed_agent2: 93
inference_iterations: 4

## Footer
Independent agent Gaussian parameters are source declarations. The result
schema preserves per-agent means, posterior covariances, observations, truth,
and controls; JAX computes online filters, RxInfer computes batch smoothing.
These two posterior estimands need not be numerically equal.

## Signature
Unsigned reproducible exemplar.
