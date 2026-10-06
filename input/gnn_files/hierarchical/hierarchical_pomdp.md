# GNN Example: Hierarchical Active Inference POMDP
# GNN Version: 1.0
# Two-level hierarchical POMDP with slow higher-level and fast lower-level dynamics.

## GNNSection
ActInfPOMDP_Hierarchical

## GNNVersionAndFlags
GNN v1

## ModelName
Hierarchical Active Inference POMDP

## ModelAnnotation

A two-level hierarchical POMDP with four lower states and two contexts:

- Level 1 has four external sensor outcomes and three controlled permutations.
- Level 2 modulates the lower-state prior at block boundaries; A_level1 is fixed.
- Each block has five observations and four controlled transitions. At the
  boundary, context transitions passively and the lower state is reinitialized.
- A_level2 mixes a 50% context-sensitive channel (90% reliable preference for
  lower state 0 or 1) with a 50% context-independent channel (equal states 2/3).
- D_level1 is the derived marginal A_level2 D_level2, not a competing prior.
- C_level1 governs lower-level behavior. C_level2 has four outcome-based
  log-preference scores and is diagnostic only; it cannot change context
  inference or lower-level policy scores. Scores need not sum to one.

The joint state (lower state, context) has eight possibilities and four external
observation symbols. A block likelihood message is not an additional sensor.
These semantics require explicit backend support and a timing/reset witness;
normalized probability tables alone do not establish native support.

## StateSpaceBlock
# Level 1 (fast dynamics)
A_level1[4,4,type=float]     # Level 1 likelihood: observations x hidden states
B_level1[4,4,3,type=float]   # Level 1 transitions: next x prev x actions
C_level1[4,type=float]       # Level 1 preferences over observations
D_level1[4,type=float]       # Level 1 prior over hidden states
s_level1[4,1,type=float]     # Level 1 hidden state distribution
x_next1[4,1,type=float] # Level 1 next hidden state
o_level1[4,1,type=int]       # Level 1 observations
π1[3,type=float]       # Level 1 policy (actions)
u_level1[1,type=int]         # Level 1 action
G1[π1,type=float]      # Level 1 Expected Free Energy

# Level 2 (slow dynamics)
A_level2[4,2,type=float]     # Level 2 likelihood: maps context to Level 1 hidden state prior
B_level2[2,2,1,type=float]   # Level 2 transitions (context switches)
C_level2[4,type=float]       # Diagnostic log-preference scores over lower-state outcomes
D_level2[2,type=float]       # Level 2 prior over contextual states
s_level2[2,1,type=float]     # Level 2 contextual hidden state
o_level2[2,1,type=float]     # Derived block likelihood for each context; not a sampled observation
G2[1,type=float]       # Optional expected preference diagnostic; no control authority

# Time
t1[1,type=int]         # Fast timescale counter
t2[1,type=int]         # Slow timescale counter

## Connections
D_level1>s_level1
s_level1-A_level1
s_level1>x_next1
A_level1-o_level1
C_level1>G1
G1>π1
π1>u_level1
B_level1>u_level1
u_level1>x_next1
o_level1>o_level2
u_level1>o_level2
o_level2>s_level2
D_level2>s_level2
s_level2-A_level2
A_level2>D_level1
s_level2-B_level2
C_level2>G2

## InitialParameterization
A_level1={
  (0.85, 0.05, 0.05, 0.05),
  (0.05, 0.85, 0.05, 0.05),
  (0.05, 0.05, 0.85, 0.05),
  (0.05, 0.05, 0.05, 0.85)
}

B_level1={
  ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
  ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
  ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
  ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
}

C_level1={(0.1, 0.1, 0.1, 1.0)}
D_level1={(0.25, 0.25, 0.25, 0.25)}

A_level2={
  (0.45, 0.05),
  (0.05, 0.45),
  (0.25, 0.25),
  (0.25, 0.25)
}

B_level2={
  ((0.9,), (0.1,)),
  ((0.1,), (0.9,))
}

C_level2={(0.0, 0.5, 0.0, 0.5)}
D_level2={(0.5, 0.5)}

## Equations

All transition tensors use B[next,previous,action]. Lower action 0 is identity,
action 1 swaps 0/1 and 2/3, and action 2 swaps 0/2 and 1/3. The context has one
passive action. Let c_b be context in block b, x_{b,k} its lower state,
y_{b,k} its observation, and a_{b,k} its action, with k=0,...,4 for states
and k=0,...,3 for actions.

P(c_0)=D_level2; P(x_{b,0}|c_b)=A_level2[:,c_b].
P(c_{b+1}|c_b)=B_level2[:,:,0].
P(x_{b,k+1}|x_{b,k},a_{b,k})=B_level1[:,:,a_{b,k}].
P(y_{b,k}|x_{b,k})=A_level1[:,x_{b,k}].
D_level1 = A_level2 D_level2 = (0.25,0.25,0.25,0.25).

For a fixed context c initialize f_{b,0}(x,c)=A_level1[y_{b,0},x] A_level2[x,c].
Then f_{b,k+1}(x,c)=A_level1[y_{b,k+1},x]
    * sum_j B_level1[x,j,a_{b,k}] f_{b,k}(j,c).
The derived message o_level2[c]=L_b(c)=sum_x f_{b,4}(x,c) is the likelihood
of the five observations conditional on context and the four executed actions.
Actions are conditioning inputs, not additional observations.

r_0(c)=D_level2[c]; q_b(c)=r_b(c)L_b(c)/sum_d r_b(d)L_b(d).
r_{b+1}(c)=sum_d B_level2[c,d,0] q_b(d).
Within a block, normalize r_b(c) f_{b,k}(x,c) jointly to obtain the current
joint posterior; marginalize it over c for lower-state inference and control.
Do not assimilate that lower posterior again as an independent observed sample.

After observation k=4, no lower action is selected or applied: transition context
and reset the lower state from A_level2. At the next block, the marginal lower
prior is A_level2 r_{b+1}. The old lower state is not carried across the reset.
The optional diagnostic G2=-sum_c q_b(c) sum_x A_level2[x,c] C_level2[x]
has no causal role in inference or action choice. Lower policy evaluation uses
C_level1 alone; higher preferences do not impose additional control.

## Time
Time=t1
Dynamic
Discrete
ModelTimeHorizon=Unbounded

## ActInfOntologyAnnotation
A_level1=LikelihoodMatrix
B_level1=TransitionMatrix
C_level1=LogPreferenceVector
D_level1=PriorOverHiddenStates
s_level1=HiddenState
o_level1=Observation
π1=PolicyVector
u_level1=Action
G1=ExpectedFreeEnergy
A_level2=HigherLevelLikelihoodMatrix
B_level2=ContextTransitionMatrix
s_level2=ContextualHiddenState
o_level2=BlockLikelihoodMessage
G2=ExpectedPreferenceDiagnostic

## ModelParameters
execution_contract: block_reset_v1
b_tensor_order: next_state_previous_state_action
num_joint_states: 8
num_external_obs: 4
num_actions: 3
num_timesteps: 20
num_hidden_states_l1: 4
num_obs_l1: 4
num_actions_l1: 3
num_context_states_l2: 2
timescale_ratio: 5

## Footer
Hierarchical Active Inference POMDP v1 - GNN Representation.
Each block contains five observations and four controlled transitions; boundaries reset the lower state.
Context modulates block-initial lower-state priors; the external likelihood remains fixed.

## Signature
Cryptographic signature goes here
