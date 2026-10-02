# GNN Example: Stigmergic Swarm Active Inference

# GNN Version: 1.0

# Three agents coordinating via environmental traces (stigmergy)

## GNNSection

ActInfPOMDP_MultiAgent

## GNNVersionAndFlags

GNN v1

## ModelName

Stigmergic Swarm Active Inference

## ModelAnnotation

Three Active Inference agents coordinating via stigmergy (environmental traces):

- No direct communication between agents — coordination emerges from environment
- Agents deposit and sense environmental signals (pheromone analogy)
- Shared 3x3 grid environment with signal intensity at each cell
- Each agent navigates independently while responding to accumulated signals
- Signal deposition: actions leave traces that other agents can observe
- Signal decay: environmental signals decay over time (volatility)
- Demonstrates emergent collective behavior from individual free energy minimization
- Models ant colony foraging, distributed robotics, and decentralized coordination

## StateSpaceBlock

# Agent 1

A_agent1[4,9,type=float]         # Agent 1 likelihood: P(obs | position on 3x3 grid)
B_agent1[9,9,4,type=float]       # Agent 1 transitions: (9 positions × 4 actions: N/S/E/W)
C_agent1[4,type=float]           # Agent 1 preferences over observations
D_agent1[9,type=float]           # Agent 1 position prior
s_agent1[9,1,type=float]         # Agent 1 position belief (9 grid cells)
o_agent1[4,1,type=int]           # Agent 1 observation: (empty, signal_low, signal_high, goal)
pi1[4,type=float]          # Agent 1 policy
u_agent1[1,type=int]             # Agent 1 action
G1[pi1,type=float]         # Agent 1 EFE

# Agent 2

A_agent2[4,9,type=float]         # Agent 2 likelihood
B_agent2[9,9,4,type=float]       # Agent 2 transitions
C_agent2[4,type=float]           # Agent 2 preferences
D_agent2[9,type=float]           # Agent 2 position prior
s_agent2[9,1,type=float]         # Agent 2 position belief
o_agent2[4,1,type=int]           # Agent 2 observation
pi2[4,type=float]          # Agent 2 policy
u_agent2[1,type=int]             # Agent 2 action
G2[pi2,type=float]         # Agent 2 EFE

# Agent 3

A_agent3[4,9,type=float]         # Agent 3 likelihood
B_agent3[9,9,4,type=float]       # Agent 3 transitions
C_agent3[4,type=float]           # Agent 3 preferences
D_agent3[9,type=float]           # Agent 3 position prior
s_agent3[9,1,type=float]         # Agent 3 position belief
o_agent3[4,1,type=int]           # Agent 3 observation
pi3[4,type=float]          # Agent 3 policy
u_agent3[1,type=int]             # Agent 3 action
G3[pi3,type=float]         # Agent 3 EFE

# Shared environment (stigmergic signals)

env_signal[9,1,type=float]     # Signal intensity at each grid cell (0.0 to 1.0)
signal_decay[1,type=float]     # Signal decay rate per timestep

# Env-conditioned observation likelihood + latent signal prior (MAJ-03)
# Each agent infers the local environmental signal level as a latent from
# its observations (empty / signal_low / signal_high / goal) via this likelihood,
# and conditions its action selection (signal-seeking) on the inferred signal.
env_obs_likelihood[4,3,type=float]  # P(obs category | local signal level: none/low/high)
env_signal_prior[3,type=float]      # prior over local signal level (none/low/high)
signal_seek[1,type=float]           # signal-seeking gain applied to action selection

# Time

t[1,type=int]                  # Discrete time step

## Connections

# Agent 1

D_agent1>s_agent1
s_agent1-A_agent1
A_agent1-o_agent1
C_agent1>G1
G1>pi1
pi1>u_agent1
B_agent1>u_agent1

# Agent 2

D_agent2>s_agent2
s_agent2-A_agent2
A_agent2-o_agent2
C_agent2>G2
G2>pi2
pi2>u_agent2
B_agent2>u_agent2

# Agent 3

D_agent3>s_agent3
s_agent3-A_agent3
A_agent3-o_agent3
C_agent3>G3
G3>pi3
pi3>u_agent3
B_agent3>u_agent3

# Stigmergic coupling via environment

u_agent1>env_signal
u_agent2>env_signal
u_agent3>env_signal
env_signal-A_agent1
env_signal-A_agent2
env_signal-A_agent3
signal_decay>env_signal

## InitialParameterization

# All agents have identical generative models (homogeneous swarm)

# Likelihood: observation depends on position and environmental signal

# At each cell, agent observes: empty (no signal), signal_low, signal_high, or goal

A_agent1={
  (0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1),
  (0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1),
  (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
  (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7)
}

A_agent2={
  (0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1),
  (0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1),
  (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
  (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7)
}

A_agent3={
  (0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1),
  (0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1),
  (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
  (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7)
}

# Shared preferences: follow signals, seek goal

C_agent1={(-0.5, 0.5, 1.5, 3.0)}
C_agent2={(-0.5, 0.5, 1.5, 3.0)}
C_agent3={(-0.5, 0.5, 1.5, 3.0)}

# Starting positions: agents start at different corners

D_agent1={(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}
D_agent2={(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}
D_agent3={(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)}



# B: per-agent transitions on the 3x3 grid (N/S/E/W)

B_agent1={
  ( (1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1) ),
  ( (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0) ),
  ( (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0) ),
  ( (1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1) )
}

B_agent2={
  ( (1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1) ),
  ( (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0) ),
  ( (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0) ),
  ( (1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1) )
}

B_agent3={
  ( (1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1) ),
  ( (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0) ),
  ( (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0) ),
  ( (1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1) )
}


# Initial environment: no signals

env_signal={(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}

# Signal decay: 10% per timestep

signal_decay={(0.9)}

# Env-conditioned observation likelihood (MAJ-03):
# rows = observation categories (empty, signal_low, signal_high, goal);
# columns = local signal level (none, low, high). Each column is normalized so
# P(obs | signal level) is a valid likelihood, and the agent uses it to infer
# the latent signal level from each observation.
env_obs_likelihood={
  (0.70, 0.10, 0.05),
  (0.15, 0.70, 0.15),
  (0.10, 0.15, 0.75),
  (0.05, 0.05, 0.05)
}

# Prior over local signal level (none, low, high): mostly signal-free initially.
env_signal_prior={(0.70, 0.20, 0.10)}

# Signal-seeking gain: scales how strongly the inferred latent signal biases
# action selection toward signal-rich observations (trophotaxis / stigmergy).
signal_seek={(2.0)}

## Equations

# Each agent independently minimizes VFE and selects actions via EFE

# Stigmergic coupling mechanism

# env_signal[cell] += deposit_rate when agent visits cell

# env_signal[cell] *= signal_decay each timestep

# Agent likelihood A is modulated by env_signal

# P(signal_obs | cell) increases with env_signal[cell]

# No direct communication — coordination emerges from shared environment

## Time

Time=t
Dynamic
Discrete
ModelTimeHorizon=30

## ActInfOntologyAnnotation

A_agent1=Agent1LikelihoodMatrix
C_agent1=Agent1PreferenceVector
D_agent1=Agent1PositionPrior
s_agent1=Agent1PositionState
o_agent1=Agent1Observation
pi1=Agent1PolicyVector
u_agent1=Agent1Action
G1=Agent1ExpectedFreeEnergy
A_agent2=Agent2LikelihoodMatrix
C_agent2=Agent2PreferenceVector
D_agent2=Agent2PositionPrior
s_agent2=Agent2PositionState
o_agent2=Agent2Observation
pi2=Agent2PolicyVector
u_agent2=Agent2Action
G2=Agent2ExpectedFreeEnergy
A_agent3=Agent3LikelihoodMatrix
C_agent3=Agent3PreferenceVector
D_agent3=Agent3PositionPrior
s_agent3=Agent3PositionState
o_agent3=Agent3Observation
pi3=Agent3PolicyVector
u_agent3=Agent3Action
G3=Agent3ExpectedFreeEnergy
env_signal=EnvironmentalSignal
signal_decay=SignalDecayRate
t=Time

## ModelParameters

# v4 axis clarification: numeric B_agentN values are unchanged. Each outer
# block is an action; rows are next states and columns are previous states.
b_tensor_order: action_next_state_previous_state

num_hidden_states: 729
num_obs: 64
num_actions: 4
num_agents: 3
grid_size: 9
num_obs: 4
num_actions: 4
signal_decay_rate: 0.9
signal_deposit_rate: 0.3
num_timesteps: 30

## Footer

Stigmergic Swarm Active Inference v1 - GNN Representation.
3 agents coordinating via environmental traces only.
No direct communication — emergent coordination from shared signals.
Models ant colony foraging and decentralized robotic coordination.

## Signature

Cryptographic signature goes here
