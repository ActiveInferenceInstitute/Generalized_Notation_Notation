# GNN Example: T-Maze Epistemic Foraging Agent

# GNN Version: 1.0

# Classic Active Inference T-maze demonstrating epistemic foraging behavior

## GNNSection

ActInfPOMDP

## GNNVersionAndFlags

GNN v1

## ModelName

T-Maze Epistemic Foraging Agent

## ModelAnnotation

An episodic T-maze with four locations (center, left arm, right arm, cue) and
two fixed reward contexts (reward_left, reward_right). Location is observed
exactly. The second modality has three outcomes: no_reward_or_no_signal,
reward, and left_cue_signal. At cue, a left signal has probability 0.8 in the
left context and 0.2 in the right context. Under the uniform prior, a signal
therefore yields P(reward_left)=0.8 and its absence yields 0.2.

From center or cue, actions go_left, go_right and go_cue move directly to their
named destinations. Both arms are absorbing under every action. Stay is the
identity at every location. Episodes reset externally to center with uniform
context; no arm transition silently resets the agent.

The objective combines cumulative raw utility with information about context
across a contingent two-transition policy. The second action may depend on the
first observed cue. Cue information can improve choice before commitment; it
does not guarantee cue-first behavior for other priors, horizons or objectives.
Adapters must implement the episodic objective and contingent policy semantics
explicitly or report them unsupported; corrected inputs alone do not establish
native support.

## StateSpaceBlock

# Hidden state factors

s_loc[4,1,type=float]        # Location state: (0:center, 1:left_arm, 2:right_arm, 3:cue_location)
s_ctx[2,1,type=float]        # Context state: (0:reward_left, 1:reward_right)

# Observation modalities

o_loc[4,1,type=int]          # Location observation: (0:center, 1:left, 2:right, 3:cue)
o_rew[3,1,type=int]          # Reward/cue observation: (0:no_reward_or_no_signal, 1:reward, 2:left_cue_signal)

# Generative model matrices

A_loc[4,4,type=float]        # Location likelihood: P(o_loc | s_loc) — identity
A_rew[3,4,2,type=float]      # Reward likelihood: P(o_rew | s_loc, s_ctx) — context-dependent

# Transition matrices

B_loc[4,4,4,type=float]      # Location transitions: P(s_loc' | s_loc, action)
B_ctx[2,2,1,type=float]      # Context transitions: identity (context doesn't change)

# Preferences

C_loc[4,type=float]          # No location preference (agent doesn't prefer a location per se)
C_rew[3,type=float]          # Raw utilities: penalize no reward/no signal, reward success, neutral cue

# Priors

D_loc[4,type=float]          # Prior: agent starts at center
D_ctx[2,type=float]          # Prior: uncertain about which arm has reward

# Policy and action

pi[4,type=float]             # First-action marginal of selected contingent policy; order left,right,cue,stay
u[1,type=int]                # Selected action
G[pi,type=float]             # Negative utility-plus-information score per contingent policy
G_epi[pi,type=float]         # Epistemic value (information gain about context)
G_ins[pi,type=float]         # Instrumental value (expected reward)

# Inference

F[1,type=float]              # Variational Free Energy

# Time

t[1,type=int]                # Discrete time step

## Connections

D_loc>s_loc
D_ctx>s_ctx
s_loc-A_loc
A_loc-o_loc
s_loc-A_rew
s_ctx-A_rew
A_rew-o_rew
s_loc-B_loc
s_ctx-B_ctx
C_rew>G_ins
G_epi>G
G_ins>G
G>pi
pi>u
B_loc>u
s_loc-F
s_ctx-F
o_loc-F
o_rew-F

## InitialParameterization

# Location likelihood: identity mapping (agent knows its location)

A_loc={
  (1.0, 0.0, 0.0, 0.0),
  (0.0, 1.0, 0.0, 0.0),
  (0.0, 0.0, 1.0, 0.0),
  (0.0, 0.0, 0.0, 1.0)
}

# Reward likelihood A_rew[outcome,location,context]. Center always gives no reward.
# Left/right arms deterministically reveal whether that arm matches context.
# At cue, reward is impossible; left_cue_signal has probability 0.8 in
# reward_left and 0.2 in reward_right. Absence has probabilities 0.2 and 0.8.

A_rew={
  ((1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (0.2, 0.8)),
  ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
  ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.8, 0.2))
}

# B_loc[next,previous,action], action order left,right,cue,stay; absorbing arms

B_loc={
  ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
  ((1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
  ((0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0), (0.0, 1.0, 0.0, 0.0)),
  ((0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0))
}

# Context is static

B_ctx={
  ((1.0,), (0.0,)),
  ((0.0,), (1.0,))
}

# Raw utilities: no_reward_or_no_signal=-1, reward=3, left_cue_signal=0

C_loc={(0.0, 0.0, 0.0, 0.0)}
C_rew={(-1.0, 3.0, 0.0)}

# Prior: starts at center, uncertain about context

D_loc={(1.0, 0.0, 0.0, 0.0)}
D_ctx={(0.5, 0.5)}

## Equations

A_loc[o,l]=P(o_loc=o|location=l) is identity. A_rew[o,l,c] is a
conditional over reward/cue outcomes. B_loc[next,previous,action] has four
actions indexed 0 (left), 1 (right), 2 (cue), 3 (stay).
identity = B_ctx[:,:,0].
Bayesian context updates use q'(c) ∝ q(c) A_rew[o_rew,l,c]; location is known.

Enumerate contingent policies pi=(a0, a1(first observation)) for at most two
action transitions. The three authored observation instants include the initial
center observation, which is excluded from scoring. Let tau be first arm
arrival or the two-transition horizon, whichever comes first, and Y the complete
observed post-action trajectory through tau, including both modalities.

U(pi)=E_pi[sum_{t=1}^{tau} C_rew[o_rew,t]].
G_ins(pi)=-U(pi).
G_epi(pi)=-I(context;Y | pi,current belief).
G(pi)=G_ins(pi)+G_epi(pi); select the smallest G, ties by lowest action index
(at each decision; compare contingent action tables in outcome-index order).
Information uses natural logarithms and coefficient one.
I(context;Y)=sum_{c,Y} q(c) P(Y|c,pi) log(P(Y|c,pi)/P(Y|pi)).
Use the joint trajectory information once, or conditional information increments;
never sum repeated marginal gains. Expected likelihood entropy alone is not
information gain.

C_rew=(-1,3,0) contains raw utilities, not log-softmax-normalized preferences.
A center wait costs -1. At cue, absence costs -1 and left_cue_signal costs 0.
First arm arrival gives +3 for reward or -1 otherwise; terminate scoring
immediately. Unused horizon steps score zero, with no terminal bonus and no
repeated reward from subsequent absorbing-arm observations. Per-step preference
normalization would change this variable-length episodic objective.

Analytic references under the uniform prior follow. Immediate arm choice has U=1;
cue followed by left after a left signal and right after absence has U=1.7;
a fixed cue→left sequence has U=0.5. Cue information alone is
log(2)+0.8 log(0.8)+0.2 log(0.2)=0.192745 nats. An arm's deterministic
reward outcome reveals context completely, giving log(2)=0.693147 nats for
both immediate-arm and complete cue-plus-arm trajectories. Their total scores
U+I are respectively 1.693147 and 2.393147 (fixed cue→left has score 1.193147).
The cue's value is information before commitment, not extra total trajectory
information. These are analytic references, not measured native results;
acceptance must compare all contingent policy scores to independent enumeration.

## Time

Time=t
Dynamic
Discrete
ModelTimeHorizon=3

## ActInfOntologyAnnotation

A_loc=LocationLikelihoodMatrix
A_rew=RewardLikelihoodMatrix
B_loc=LocationTransitionMatrix
B_ctx=ContextTransitionMatrix
C_loc=LocationPreferenceVector
C_rew=RewardPreferenceVector
D_loc=LocationPrior
D_ctx=ContextPrior
s_loc=LocationHiddenState
s_ctx=ContextHiddenState
o_loc=LocationObservation
o_rew=RewardObservation
pi=PolicyVector
u=Action
G=ExpectedFreeEnergy
G_epi=EpistemicValue
G_ins=InstrumentalValue
F=VariationalFreeEnergy
t=Time

## ModelParameters
execution_contract: episodic_contingent_v1
terminal_locations: [1, 2]
b_tensor_order: next_state_previous_state_action

num_locations: 4
num_contexts: 2
num_location_obs: 4
num_reward_obs: 3
num_actions: 4
num_timesteps: 3
policy_horizon: 2
num_modalities: 2
num_state_factors: 2

## Footer

T-Maze Epistemic Foraging Agent v1 - GNN Representation.
Episodic exploration and exploitation with a partial cue and absorbing arms.
Two action transitions, contingent second actions, and once-per-episode reward scoring.

## Signature

Cryptographic signature goes here
