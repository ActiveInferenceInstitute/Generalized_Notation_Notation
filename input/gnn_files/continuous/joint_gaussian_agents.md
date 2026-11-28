# Two coupled Gaussian agents

## GNNSection

ActInfContinuousMultiAgent

## GNNVersionAndFlags

GNN v1

## ModelName

Two coupled Gaussian agents

## ModelAnnotation

Centralized filtering of a source-declared linear-Gaussian system with
asymmetric agent dimensions, correlated noise and a fixed control sequence.

## StateSpaceBlock

x[3,1,type=float]
y[3,1,type=float]
u[2,1,type=float]

## Connections

u>x
x>y

## InitialParameterization

joint_agents = [{"id": "north", "state_ids": ["position"], "observation_ids": ["direct", "offset"], "control_ids": ["input"]}, {"id": "south", "state_ids": ["position", "velocity"], "observation_ids": ["combined"], "control_ids": ["input"]}]
F = [[0.75, 0.0, 0.0], [0.3, 0.9, 0.08], [-0.1, 0.0, 0.8]]
H = [[1.1, -0.13, 0.0], [0.5, 0.0, 0.2], [0.15, 1.0, -0.6]]
Q = [[0.1, 0.03, -0.02], [0.03, 0.14, 0.01], [-0.02, 0.01, 0.09]]
R = [[0.2, 0.02, -0.03], [0.02, 0.25, 0.04], [-0.03, 0.04, 0.15]]
prior_mean = [-0.7, 0.8, -0.4]
prior_cov = [[0.4, 0.08, -0.04], [0.08, 0.5, 0.06], [-0.04, 0.06, 0.3]]
control_matrix = [[0.2, -0.35], [0.3, 0.5], [0.1, -0.25]]
control_sequence = [[0.4, 0.0], [-0.3, 0.5], [0.1, -0.6]]
observation_sequence = [[-0.7, -0.2, 1.0], [-0.2, -0.1, 0.8], [0.2, 0.1, 0.3], [0.5, 0.2, -0.2]]

## Equations

x(0) ~ Normal(prior_mean, prior_cov)
y(t) = H x(t) + epsilon(t), epsilon(t) ~ Normal(0, R)
x(t+1) = F x(t) + control_matrix u(t) + eta(t), eta(t) ~ Normal(0, Q)

## Time

Dynamic

## ModelParameters

nr_agents: 2
agent_coupling: gnn_joint_gaussian_filter_v1
num_timesteps: 4
dt: 1.0

## Footer

This exemplar uses the declared control and observation sequences to compare
the centralized filter's joint posterior mean and covariance with independent
Gaussian conditioning at each observation prefix.
