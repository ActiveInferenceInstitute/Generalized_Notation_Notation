# GNN Example: Three-Level Temporal Hierarchy Agent

# GNN Version: 1.0

# Hierarchical Active Inference with three temporal scales

## GNNSection

ActInfPOMDP_Hierarchical

## GNNVersionAndFlags

GNN v1

## ModelName

Three-Level Temporal Hierarchy Agent

## ModelAnnotation

An approximate three-level message-passing controller with distinct clocks:

- Level 0: four fast states, three external sensor outcomes, three actions;
  fast transitions occur every 0.1 seconds.
- Level 1: three tactical belief categories and four soft-message categories;
  updates follow every 10 fast transitions (1 second).
- Level 2: two strategic belief categories and three soft-message categories;
  updates follow every 100 fast transitions (10 seconds).
- Only fast external states and observations are sampled. Upper actions control
  internal predictive-belief transitions, not separately observed physical actuators.
- Bottom-up messages average lower posterior means with total evidence weight one.
  Upper A matrices are hypothetical categorical models for counterfactual scoring,
  while actual assimilation receives soft summaries. This is an approximate
  controller, not exact inference in a joint latent-state generative hierarchy.
- Top-down maps add preferences from frozen upper beliefs. A separate stochastic
  map initializes the tactical prior; it does not overwrite running posteriors.

Fast state 3 is absorbing and unreachable from other states. Its three outcomes
are equiprobable; their distribution can still distinguish it from other states.
With uniform D_level0, 25% of trajectories start there and remain there forever.
The parameter literal 0.3333333333333333 represents mathematical 1/3 in the
numeric grammar. Upper likelihoods allocate probability 0.1 to their last,
ambiguous predicted category; these are not additional sampled external reports.
Products of per-level cardinalities do not define a 24-state joint world or
36 external symbols. Native support requires this controller's timing and
objective, beyond parsing or normalizing its tables.

## StateSpaceBlock

# Level 0: Fast sensorimotor (4 states, 3 obs, 3 actions)

A_level0[3,4,type=float]         # Level 0 likelihood: P(fast_obs | fast_state)
B_level0[4,4,3,type=float]       # Level 0 transitions: P(fast_state' | fast_state, fast_action)
C_level0[3,type=float]           # Base raw utilities; effective C0 adds M10 q1
D_level0[4,type=float]           # Level 0 prior over initial states
s_level0[4,1,type=float]         # Level 0 hidden state belief
o_level0[3,1,type=int]           # Level 0 observation
pi0[3,type=float]          # Level 0 policy
u_level0[1,type=int]             # Level 0 action
G0[pi0,type=float]         # Negative one-step utility-plus-information score

# Level 1: Medium tactical (3 states, 4 obs, 3 actions)

A_level1[4,3,type=float]         # Level 1 likelihood: P(tactic_obs | tactic_state)
B_level1[3,3,3,type=float]       # Level 1 transitions
C_level1[4,type=float]           # Base raw utilities; effective C1 adds M21 q2
D_level1[3,type=float]           # Derived initialization-only prior N21 D_level2
s_level1[3,1,type=float]         # Level 1 hidden state belief
o_level1[4,1,type=float]         # Level 1 observation (= summary of Level 0 state trajectory)
pi1[3,type=float]          # Level 1 policy
u_level1[1,type=int]             # Level 1 action
G1[pi1,type=float]         # Negative one-step utility-plus-information score

# Level 2: Slow strategic (2 states, 3 obs, 2 actions)

A_level2[3,2,type=float]         # Level 2 likelihood: P(strategy_obs | strategy_state)
B_level2[2,2,2,type=float]       # Level 2 transitions
C_level2[3,type=float]           # Fixed base raw utilities
D_level2[2,type=float]           # Level 2 prior over strategies
s_level2[2,1,type=float]         # Level 2 hidden state belief
o_level2[3,1,type=float]         # Level 2 observation (= summary of Level 1 outcomes)
pi2[2,type=float]          # Level 2 policy
u_level2[1,type=int]             # Level 2 action
G2[pi2,type=float]         # Negative one-step utility-plus-information score

# Cross-level maps: M maps are preference increments, N21 is stochastic

M10[3,3,type=float]             # Tactical beliefs to fast preference increments
M21[4,2,type=float]             # Strategic beliefs to tactical preference increments
N21[3,2,type=float]             # Initialization-only strategic-to-tactical prior

# Timescale parameters

tau_level0[1,type=float]         # Level 0 time constant (0.1s)
tau_level1[1,type=float]         # Level 1 time constant (1.0s)
tau_level2[1,type=float]         # Level 2 time constant (10.0s)

# Time

t[1,type=int]              # Global discrete time counter

## Connections

# Level 0 (fast) internal loop

D_level0>s_level0
s_level0-A_level0
A_level0-o_level0
C_level0>G0
G0>pi0
pi0>u_level0
B_level0>u_level0

# Level 1 (medium) internal loop

D_level1>s_level1
s_level1-A_level1
A_level1-o_level1
C_level1>G1
G1>pi1
pi1>u_level1
B_level1>u_level1

# Level 2 (slow) internal loop

D_level2>s_level2
s_level2-A_level2
A_level2-o_level2
C_level2>G2
G2>pi2
pi2>u_level2
B_level2>u_level2

# Top-down causal flow (context modulates subordinate levels)

s_level2>M21
M21>C_level1
s_level1>M10
M10>C_level0
s_level2>N21
N21>D_level1

# Bottom-up evidential flow (observations inform superior levels)

s_level0>o_level1
s_level1>o_level2

## InitialParameterization

# Level 0: Sensorimotor (fast, reflexive)

A_level0={
  (0.9, 0.05, 0.05, 0.3333333333333333),
  (0.05, 0.9, 0.05, 0.3333333333333333),
  (0.05, 0.05, 0.9, 0.3333333333333333)
}

C_level0={(0.0, -1.0, 1.0)}
D_level0={(0.25, 0.25, 0.25, 0.25)}

# Level 1: Tactical

A_level1={
  (0.7, 0.1, 0.1),
  (0.1, 0.7, 0.1),
  (0.1, 0.1, 0.7),
  (0.1, 0.1, 0.1)
}

C_level1={(-0.5, 1.0, 1.5, -1.0)}
D_level1={
  (0.4, 0.2, 0.4)
}

# Level 2: Strategic

A_level2={
  (0.8, 0.1),
  (0.1, 0.8),
  (0.1, 0.1)
}

C_level2={(-1.0, 2.0, 0.5)}
D_level2={(0.5, 0.5)}

# Timescale constants

tau_level0={(0.1)}
tau_level1={(1.0)}
tau_level2={(10.0)}

# B[next,previous,action]: fast/medium actions persistent, swap 0/1, cycle 0→1→2→0; slow actions hold, switch

B_level0={
  ((0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9), (0.0, 0.0, 0.0)),
  ((0.05, 0.9, 0.9), (0.9, 0.05, 0.05), (0.05, 0.05, 0.05), (0.0, 0.0, 0.0)),
  ((0.05, 0.05, 0.05), (0.05, 0.05, 0.9), (0.9, 0.9, 0.05), (0.0, 0.0, 0.0)),
  ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
}

B_level1={
  ((0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9)),
  ((0.05, 0.9, 0.9), (0.9, 0.05, 0.05), (0.05, 0.05, 0.05)),
  ((0.05, 0.05, 0.05), (0.05, 0.05, 0.9), (0.9, 0.9, 0.05))
}

B_level2={
  ((0.95, 0.05), (0.05, 0.95)),
  ((0.05, 0.95), (0.95, 0.05))
}

M10={
  (0.8, 0.1, 0.1),
  (0.1, 0.8, 0.1),
  (0.1, 0.1, 0.8)
}
M21={
  (1.0, 0.0),
  (0.5, 0.5),
  (0.0, 1.0),
  (0.0, 0.0)
}
N21={
  (0.7, 0.1),
  (0.2, 0.2),
  (0.1, 0.7)
}

## Equations

Write q0,q1,q2 for level beliefs. The action slice is
Bk_action = Bk[:,:,a] for canonical B[next,previous,action].
At fast and medium levels action 0 is persistent,
action 1 swaps states 0/1 with probability 0.9, and action 2 cycles
0→1→2→0 with probability 0.9 (0.05 for each other destination among 0/1/2).
Fast state 3 remains absorbing under all actions. Slow action 0 holds context
with probability 0.95; action 1 switches with probability 0.95.

Initialize at transition count t=0:
q2=D_level2; q1=N21 q2=D_level1=(0.4,0.2,0.4); q0=D_level0.
Sample the initial fast state and observation and update
q0(s) ∝ D_level0[s] A_level0[o_0,s] before choosing the first fast action.
N21 is used only at initialization, never to reset an ongoing tactical posterior.

Effective raw utility scores, recomputed from base without accumulation, are
C0=C_level0+M10 q1; C1=C_level1+M21 q2; C2=C_level2.
The top-down gain is one log-score unit. M10 and M21 are preference maps,
not likelihoods or probability conditionals. M21's last row contributes zero.

For each level k and candidate action a define
r_a(s)=sum_j Bk[s,j,a] qk(j); p_a(o)=sum_s Ak[o,s] r_a(s).
U_k(a)=sum_o p_a(o) Ck[o].
I_k(a)=sum_{s,o} r_a(s) Ak[o,s] log(Ak[o,s]/p_a(o)), with zero-mass terms zero.
Gk(a)=-U_k(a)-I_k(a). Choose argmin_a Gk(a), ties by lowest action index.
All logarithms are natural; information coefficient is one. The pi vectors
are one-hot selections of these actions, not softmax policies. Choose initial
fast, medium and slow actions at t=0 for their next respective intervals.

For every subsequent fast transition, sample the fast state and observation
according to the following conditionals, then update the fast belief.

P(x_t | x_{t-1},a0) = B_level0[:,x_{t-1},a0].
P(o_t | x_t) = A_level0[:,x_t].
q0_t(s) ∝ A_level0[o_t,s] sum_j B_level0[s,j,a0] q0_{t-1}(j).
Freeze upper beliefs between their boundaries. At t=10k+10 form
h1=(1/10) sum_{r=10k+1}^{10k+10} q0_r, excluding the block-initial q0.
Predict r1=B_level1[:,:,a1] q1 using the action chosen at the previous medium
boundary, then q1(s) ∝ r1(s) exp(sum_i h1[i] log A_level1[i,s]).
o_level1=h1 has four categories and total weight one, not ten observations.

At t=100k+100 form h2 from the mean of the ten updated q1 vectors at
100k+10,...,100k+100. Predict r2=B_level2[:,:,a2] q2 using the previously
selected slow action, then q2(s) ∝ r2(s) exp(sum_i h2[i] log A_level2[i,s]).
o_level2=h2 has three categories and total weight one.

At a shared boundary, perform fast observation update, medium prediction/message
update, then slow prediction/message update, in that order. Only then recompute
C1 and C0 from the new upper beliefs and choose actions for the next intervals.
Choose a0 after every fast observation, a1 at medium boundaries, and a2 at slow
boundaries. Held upper actions are used once in their next predictive update.
All posterior updates normalize over their own level. The soft log-likelihood
messages define an approximate controller, not an exact generative likelihood.

The authored demonstration has 100 transitions plus its initial observation.
10 medium updates and one slow update after initialization. Acceptance must
separately run at least 200 transitions to witness two slow updates and verify
message weights, frozen beliefs, update order and nonaccumulating preferences.

## Time

Time=t
Dynamic
Discrete
ModelTimeHorizon=100

## ActInfOntologyAnnotation

A_level0=FastLikelihoodMatrix
B_level0=FastTransitionMatrix
C_level0=FastPreferenceVector
D_level0=FastPrior
s_level0=FastHiddenState
o_level0=FastObservation
pi0=FastPolicyVector
u_level0=FastAction
G0=NegativeUtilityInformationScore
A_level1=TacticalLikelihoodMatrix
B_level1=TacticalTransitionMatrix
C_level1=TacticalPreferenceVector
D_level1=TacticalPrior
s_level1=TacticalHiddenState
o_level1=SoftPosteriorSummary
pi1=TacticalPolicyVector
u_level1=TacticalAction
G1=NegativeUtilityInformationScore
A_level2=StrategicLikelihoodMatrix
B_level2=StrategicTransitionMatrix
C_level2=StrategicPreferenceVector
D_level2=StrategicPrior
s_level2=StrategicHiddenState
o_level2=SoftPosteriorSummary
pi2=StrategicPolicyVector
u_level2=StrategicAction
G2=NegativeUtilityInformationScore
M10=PreferenceIncrementMap
M21=PreferenceIncrementMap
N21=InitialPriorMap
tau_level0=FastTimeConstant
tau_level1=TacticalTimeConstant
tau_level2=StrategicTimeConstant
t=Time

## ModelParameters
execution_contract: timed_soft_controller_v1
b_tensor_order: next_state_previous_state_action

num_levels: 3
num_states_l0: 4
num_obs_l0: 3
num_actions_l0: 3
num_states_l1: 3
num_obs_l1: 4
num_actions_l1: 3
num_states_l2: 2
num_obs_l2: 3
num_actions_l2: 2
timescale_ratio_1_0: 10
timescale_ratio_2_1: 10
num_timesteps: 100
num_fast_transitions: 100
num_external_observations: 101

## Footer

Three-Level Temporal Hierarchy Agent v1 - GNN Representation.
Fast (100ms sensorimotor), Medium (1s tactical), Slow (10s strategic).
Top-down: strategy → tactics → sensorimotor preferences.
Bottom-up: sensory evidence → tactical summaries → strategic outcomes.

## Signature

Cryptographic signature goes here
