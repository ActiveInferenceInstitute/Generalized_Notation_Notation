# GNN Example: Multi-Agent Cooperative Active Inference

# GNN Version: 1.0

# Two agents cooperating via shared observation space

## GNNSection

ActInfPOMDP_MultiAgent

## GNNVersionAndFlags

GNN v1

## ModelName

Multi-Agent Cooperative Active Inference

## ModelAnnotation

Two Active Inference agents cooperating on a joint task:

- Agent 1 and Agent 2 each maintain independent beliefs
- Shared observation space: agents observe each other's actions
- Joint task state includes both agents' positions (4x4 = 16 joint states)
- Cooperative preferences: both agents prefer the same goal configuration
- Models social cognition and coordination without explicit communication

## StateSpaceBlock

# Agent 1

A_agent1[4,4,type=float]     # Agent 1 likelihood
B_agent1[4,4,3,type=float]   # Agent 1 transitions (3 actions)
C_agent1[4,type=float]       # Agent 1 preferences
D_agent1[4,type=float]       # Agent 1 prior
s_agent1[4,1,type=float]     # Agent 1 hidden state
x_next1[4,1,type=float] # Agent 1 next hidden state
o_agent1[4,1,type=int]       # Agent 1 observations (includes Agent 2 obs)
π1[3,type=float]       # Agent 1 policy
u1[1,type=int]         # Agent 1 action
G1[π1,type=float]      # Agent 1 EFE

# Agent 2

A_agent2[4,4,type=float]     # Agent 2 likelihood
B_agent2[4,4,3,type=float]   # Agent 2 transitions (3 actions)
C_agent2[4,type=float]       # Agent 2 preferences
D_agent2[4,type=float]       # Agent 2 prior
s_agent2[4,1,type=float]     # Agent 2 hidden state
x_next2[4,1,type=float] # Agent 2 next hidden state
o_agent2[4,1,type=int]       # Agent 2 observations (includes Agent 1 obs)
π2[3,type=float]       # Agent 2 policy
u2[1,type=int]         # Agent 2 action
G2[π2,type=float]      # Agent 2 EFE

# Shared environment state

s_joint[16,1,type=float]  # Joint state (Agent1_pos x Agent2_pos)
o_joint[4,1,type=int]     # Joint observation (goal achievement)

# Time

t[1,type=int]

## Connections

D_agent1>s_agent1
s_agent1-A_agent1
A_agent1-o_agent1
s_agent1>x_next1
C_agent1>G1
G1>π1
π1>u1
B_agent1>u1
D_agent2>s_agent2
s_agent2-A_agent2
A_agent2-o_agent2
s_agent2>x_next2
C_agent2>G2
G2>π2
π2>u2
B_agent2>u2
u1>s_joint
u2>s_joint
s_joint-o_joint
o_agent1-s_joint
o_agent2-s_joint

## InitialParameterization

A_agent1={
  (0.85, 0.05, 0.05, 0.05),
  (0.05, 0.85, 0.05, 0.05),
  (0.05, 0.05, 0.85, 0.05),
  (0.05, 0.05, 0.05, 0.85)
}

A_agent2={
  (0.85, 0.05, 0.05, 0.05),
  (0.05, 0.85, 0.05, 0.05),
  (0.05, 0.05, 0.85, 0.05),
  (0.05, 0.05, 0.05, 0.85)
}

# Shared cooperative preference: goal = state 4 (index 3)

C_agent1={(-1.0, -1.0, -1.0, 2.0)}
C_agent2={(-1.0, -1.0, -1.0, 2.0)}
D_agent1={(0.25, 0.25, 0.25, 0.25)}
D_agent2={(0.25, 0.25, 0.25, 0.25)}

B_agent1={
  ( (0.9,0.1,0.0,0.0), (0.0,0.9,0.1,0.0), (0.0,0.0,0.9,0.1), (0.1,0.0,0.0,0.9) ),
  ( (0.9,0.0,0.0,0.1), (0.1,0.9,0.0,0.0), (0.0,0.1,0.9,0.0), (0.0,0.0,0.1,0.9) ),
  ( (0.8,0.1,0.1,0.0), (0.1,0.8,0.0,0.1), (0.1,0.0,0.8,0.1), (0.0,0.1,0.1,0.8) )
}

B_agent2={
  ( (0.9,0.1,0.0,0.0), (0.0,0.9,0.1,0.0), (0.0,0.0,0.9,0.1), (0.1,0.0,0.0,0.9) ),
  ( (0.9,0.0,0.0,0.1), (0.1,0.9,0.0,0.0), (0.0,0.1,0.9,0.0), (0.0,0.0,0.1,0.9) ),
  ( (0.8,0.1,0.1,0.0), (0.1,0.8,0.0,0.1), (0.1,0.0,0.8,0.1), (0.0,0.1,0.1,0.8) )
}

## Equations

# Each agent independently minimizes their own VFE

# Coordination emerges from shared observation space and aligned preferences

# Agent 1 observes both own state and Agent 2's last action

# No explicit communication channel — implicit coordination only

## Time

Time=t
Dynamic
Discrete
ModelTimeHorizon=20

## ActInfOntologyAnnotation

A_agent1=LikelihoodMatrix
B_agent1=TransitionMatrix
C_agent1=LogPreferenceVector
D_agent1=PriorOverHiddenStates
s_agent1=Agent1HiddenState
x_next1=Agent1NextHiddenState
o_agent1=Agent1Observation
π1=Agent1PolicyVector
u1=Agent1Action
G1=Agent1ExpectedFreeEnergy
A_agent2=LikelihoodMatrix
B_agent2=TransitionMatrix
C_agent2=LogPreferenceVector
D_agent2=PriorOverHiddenStates
s_agent2=Agent2HiddenState
x_next2=Agent2NextHiddenState
o_agent2=Agent2Observation
π2=Agent2PolicyVector
u2=Agent2Action
G2=Agent2ExpectedFreeEnergy
s_joint=JointState
o_joint=JointObservation
t=Time

## ModelParameters

# v4 axis clarification: numeric B_agentN values are unchanged. Each outer
# block is an action; rows are next states and columns are previous states.
b_tensor_order: action_next_state_previous_state

num_hidden_states: 16
num_obs: 16
num_actions: 3
num_agents: 2
num_hidden_states_per_agent: 4
num_obs_per_agent: 4
num_actions_per_agent: 3
num_timesteps: 20

## Footer

Multi-Agent Cooperative Active Inference v1 - GNN Representation.
Implicit coordination via shared observation space.
No explicit communication — emergent cooperation from aligned preferences.

## Signature

Cryptographic signature goes here
