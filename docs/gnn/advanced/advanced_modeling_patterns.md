# Advanced GNN Modeling Patterns

**Version**: v3.6.0 Engine (Bundle v2.0.0)
**Last Updated**: 2026-09-23
**Status**: Maintained
**Scope**: Advanced GNN modeling patterns. See [framework implementations](../implementations/README.md) for current backend coverage.

Comprehensive guide to sophisticated Active Inference modeling techniques using GNN.

## Pipeline Processing for Advanced Models

Advanced GNN models benefit from the full pipeline processing capabilities:

### Parsing & Validation (Steps 3, 5, 6)

- Complex hierarchical and multi-agent models are validated for consistency
- See: **[src/gnn/AGENTS.md](../../../src/gnn/AGENTS.md)**, **[src/gnn/type_checker/AGENTS.md](../../../src/gnn/type_checker/AGENTS.md)**

### Code Generation & Execution (Steps 11, 12)

- Advanced patterns rendered to framework-specific implementations  
- See: **[src/gnn/render/AGENTS.md](../../../src/gnn/render/AGENTS.md)**, **[src/gnn/execute/AGENTS.md](../../../src/gnn/execute/AGENTS.md)**

### Analysis & Reporting (Steps 13, 16, 23)

- Advanced statistical analysis and LLM-enhanced interpretation
- See: **[src/gnn/llm/AGENTS.md](../../../src/gnn/llm/AGENTS.md)**, **[src/gnn/analysis/AGENTS.md](../../../src/gnn/analysis/AGENTS.md)**

**Quick Start:**

```bash
# Process advanced models through full pipeline
uv run python src/gnn/main.py --target-dir input/gnn_files --verbose
```

For complete pipeline documentation, see **[src/gnn/AGENTS.md](../../../src/gnn/AGENTS.md)**.

---

## 🎯 Overview

This guide covers advanced patterns for modeling complex cognitive and behavioral systems using GNN. Each pattern includes theory, implementation, and practical examples.

> **Read the `Connections` blocks below as topology sketches, not as literal GNN.**
> To keep the structure of each pattern readable, the connection lines in this document
> use an arrow-and-grouping shorthand — `(s_f0) -> (A_m0) -> (o_m0)`,
> `(s_f0, u_c0) -> (B_f0)`. Real GNN `Connections` syntax has no arrows, parentheses, or
> grouping: it is one edge per line using `>` for directed and `-` for undirected, with no
> spaces, as in
>
> ```text
> D>s
> s-A
> s>s_prime
> C>G
> ```
>
> Copying a `Connections` block from this document straight into a `.md` model will not
> parse. Translate each grouped arrow into individual `>` / `-` edges first, and see
> [`docs/gnn/reference/gnn_syntax.md`](../reference/gnn_syntax.md) for the authoritative
> grammar plus [`input/gnn_files/`](../../../input/gnn_files/) for models that do parse.
> Everything else here — `StateSpaceBlock` dimensions, `InitialParameterization`, `Time`,
> and the ontology annotations — is literal GNN.

## 📚 Table of Contents

1. [Hierarchical Modeling](#hierarchical-modeling)

   ### Hierarchical Active Inference

    Pattern for modeling nested levels of abstraction and temporal scales.

   ### Factorial State Spaces

    Pattern for modeling multidimensional, independent state factors.

   ### Dependency Injection pattern

```python
# Example of Dependency Injection in GNN
# This pattern is more about code structure than GNN model structure
# but can be used to manage complex GNN model components.


class GNNModel:
    def __init__(self, likelihood_module, transition_module):
        self.A = likelihood_module
        self.B = transition_module

    def predict(self, state):
        # Use injected modules
        pass


class LikelihoodModule:
    def calculate(self, obs, state):
        pass


class TransitionModule:
    def update(self, state, action):
        pass


# Usage:
# likelihood = LikelihoodModule()
# transition = TransitionModule()
# model = GNNModel(likelihood, transition)
```

### Policy Tree Optimization

**Planned — not yet implemented.** There is no dedicated policy-tree-optimization pattern or module in `src/` today. The closest existing capability is the `policy_tree` output type in `src/gnn/advanced_visualization` (`src/gnn/advanced_visualization/mcp.py`), which visualizes the action-sequence tree of a planning agent but does not perform optimization over it.

2. [Multi-Agent Systems](#multi-agent-systems)
3. [Learning and Adaptation](#learning-and-adaptation)
4. [Temporal Dynamics](#temporal-dynamics)
5. [Uncertainty and Robustness](#uncertainty-and-robustness)
6. [Compositional Modeling](#compositional-modeling)
7. [Graceful Degradation in the Pipeline](#7-graceful-degradation-in-the-pipeline)
8. [Domain-Specific Patterns](#domain-specific-patterns)

---

## 1. Hierarchical Modeling

### Pattern: Temporal Hierarchies

**Use Case**: Different cognitive processes operating at different timescales.

```gnn
## ModelName
Hierarchical Temporal Agent v1.0

## StateSpaceBlock
# Fast level: Immediate sensorimotor responses (100ms)
s_f0[4,1,type=int]     # Fast_action_state (0:attend, 1:move, 2:grasp, 3:release)
o_m0[3,1,type=int]     # Fast_observations (0:clear, 1:obstacle, 2:target)

# Medium level: Tactical planning (1-10s)
s_f1[6,1,type=int]     # Medium_goal_state (0:explore, 1:approach, 2:manipulate, 3:avoid, 4:wait, 5:retreat)
o_m1[4,1,type=int]     # Medium_observations (0:safe, 1:risky, 2:opportunity, 3:completion)

# Slow level: Strategic objectives (minutes-hours)
s_f2[3,1,type=int]     # Slow_strategy_state (0:gathering, 1:building, 2:resting)
o_m2[2,1,type=int]     # Slow_observations (0:insufficient, 1:sufficient)

# Hierarchical control variables
pi_c0[4,type=float]    # Fast_policy
pi_c1[6,type=float]    # Medium_policy  
pi_c2[3,type=float]    # Slow_policy

# Cross-level matrices
A_m0[3,4,type=float]   # Fast_likelihood: P(fast_obs | fast_state)
A_m1[4,6,type=float]   # Medium_likelihood: P(medium_obs | medium_state)
A_m2[2,3,type=float]   # Slow_likelihood: P(slow_obs | slow_state)

B_f0[4,4,4,type=float] # Fast_transitions: P(fast_state' | fast_state, fast_action)
B_f1[6,6,6,type=float] # Medium_transitions: P(medium_state' | medium_state, medium_action)
B_f2[3,3,3,type=float] # Slow_transitions: P(slow_state' | slow_state, slow_action)

# Hierarchical preferences (goals flow down)
C_m0[3,type=float]     # Fast_preferences (context-dependent)
C_m1[4,type=float]     # Medium_preferences (goal-dependent) 
C_m2[2,type=float]     # Slow_preferences (strategic)

# Time constants for each level
tau_0[1,type=float]    # Fast_time_constant (0.1)
tau_1[1,type=float]    # Medium_time_constant (1.0)
tau_2[1,type=float]    # Slow_time_constant (10.0)

## Connections
# Within-level connections
(s_f0) -> (A_m0) -> (o_m0)
(s_f1) -> (A_m1) -> (o_m1) 
(s_f2) -> (A_m2) -> (o_m2)

# Hierarchical influence (top-down context)
(s_f2) -> (C_m1)  # Strategy sets medium-level goals
(s_f1) -> (C_m0)  # Medium goals set fast-level preferences

# Bottom-up information flow
(o_m0) -> (s_f1)  # Fast observations inform medium planning
(o_m1) -> (s_f2)  # Medium observations inform strategy

# Time-dependent transitions
(s_f0, pi_c0, tau_0) -> (B_f0)
(s_f1, pi_c1, tau_1) -> (B_f1)
(s_f2, pi_c2, tau_2) -> (B_f2)

## InitialParameterization
# Fast level optimizes for immediate safety/efficiency
C_m0={(0.0, -1.0, 1.0)}  # Prefer target, avoid obstacle

# Medium level balances exploration vs exploitation
C_m1={(-0.5, 1.0, 1.5, -1.0)}  # Prefer approach and manipulate

# Slow level optimizes for long-term resource accumulation
C_m2={(-1.0, 2.0)}  # Strongly prefer sufficient resources

# Time constants reflect cognitive timescales
tau_0={(0.1)}   # 100ms sensorimotor
tau_1={(1.0)}   # 1s tactical
tau_2={(10.0)}  # 10s strategic

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=20
HierarchicalLevels=3
```

### Pattern: Spatial Hierarchies

**Use Case**: Multi-scale spatial reasoning (local → global).

```gnn
## StateSpaceBlock
# Local spatial attention (immediate vicinity)
s_f0[9,1,type=int]     # Local_position (3x3 grid around agent)
s_f1[8,1,type=int]     # Local_objects (adjacent cells)

# Regional navigation (neighborhood)
s_f2[25,1,type=int]    # Regional_position (5x5 regions)
s_f3[16,1,type=int]    # Regional_landmarks (4x4 landmark grid)

# Global planning (entire environment)
s_f4[100,1,type=int]   # Global_position (10x10 world map)
s_f5[20,1,type=int]    # Global_objectives (distributed goals)

# Cross-scale observation modalities
o_m0[4,1,type=int]     # Immediate_percept (N,S,E,W)
o_m1[8,1,type=int]     # Regional_survey (cardinal+diagonal directions)
o_m2[4,1,type=int]     # Global_compass (rough direction to goal)

## Connections
# Scale-specific perception
(s_f0) -> (A_m0) -> (o_m0)  # Local detailed perception
(s_f2) -> (A_m1) -> (o_m1)  # Regional survey
(s_f4) -> (A_m2) -> (o_m2)  # Global orientation

# Hierarchical spatial embedding
(s_f0) -> (s_f2)  # Local position informs regional
(s_f2) -> (s_f4)  # Regional position informs global

# Cross-scale object recognition
(s_f1, s_f3, s_f5) -> landmark_consistency_constraint
```

---

## 2. Multi-Agent Systems

### Pattern: Distributed Coordination

**Use Case**: Multiple agents coordinating without centralized control.

```gnn
## ModelName
Distributed_Coordination_Agent_i v1.0

## StateSpaceBlock
# Self-model
s_f0[4,1,type=int]     # Own_state (position/role)
s_f1[3,1,type=int]     # Own_capability (what this agent can do)
s_f2[2,1,type=int]     # Own_resources (current resource level)

# Other agents model (Theory of Mind)
s_f3[16,1,type=int]    # Others_states (4 agents × 4 states each)
s_f4[12,1,type=int]    # Others_intentions (4 agents × 3 intentions each)
s_f5[8,1,type=int]     # Others_resources (4 agents × 2 resource levels each)

# Shared environment
s_f6[10,1,type=int]    # Environment_state (shared world state)
s_f7[5,1,type=int]     # Collective_progress (team task progress)

# Communication states
s_f8[4,1,type=int]     # Message_received (last message content)
s_f9[4,1,type=int]     # Message_to_send (outgoing message)

# Observations
o_m0[4,1,type=int]     # Direct_observation (what agent directly sees)
o_m1[3,1,type=int]     # Social_observation (observed agent behaviors)
o_m2[2,1,type=int]     # Communication_channel (received messages)

# Actions
u_c0[5,1,type=int]     # Physical_action (move, manipulate, wait, etc.)
u_c1[4,1,type=int]     # Communication_action (message type to send)
u_c2[3,1,type=int]     # Coordination_action (propose, accept, decline)

## Connections
# Self-perception and action
(s_f0, s_f1, s_f2) -> (A_m0) -> (o_m0)
(s_f0, u_c0) -> (B_f0)

# Theory of Mind: predicting others
(s_f3, s_f4) -> predicted_other_actions
(o_m1) -> (s_f3, s_f4)  # Update beliefs about others

# Communication dynamics
(s_f8) -> (o_m2)  # Receive messages
(s_f9, u_c1) -> outgoing_communication

# Coordination constraints
(s_f0, s_f3, s_f7) -> coordination_utility
coordination_utility -> (C_m0, C_m1, C_m2)

# Environment shared by all agents
(s_f6) -> (A_m0)  # Environment affects observations
(u_c0) -> (s_f6)  # Actions affect shared environment

## InitialParameterization
# Self-interest vs. collective benefit trade-off
C_m0={(1.0, 0.5, 0.5, 0.2)}  # Prefer beneficial actions

# Communication preferences (truth-telling, coordination)
C_m1={(0.8, 0.2, 0.0, -0.5)}  # Prefer honest, helpful communication

# Coordination preferences (consensus, efficiency)
C_m2={(1.5, 1.0, 0.0)}  # Prefer accept > propose > decline

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=15
```

### Pattern: Leader-Follower Dynamics

**Use Case**: Asymmetric coordination with emergent leadership.

```gnn
## StateSpaceBlock
s_top[1, type=int]
s_mid[1, type=int]
s_low[1, type=int]
s_f0[3,1,type=int]     # Leadership_role (0:follower, 1:candidate, 2:leader)
s_f1[4,1,type=int]     # Authority_level (influence over others)
s_f2[5,1,type=int]     # Group_consensus (level of agreement in group)

# Task allocation
s_f3[6,1,type=int]     # Task_assignment (which task this agent has)
s_f4[3,1,type=int]     # Task_competence (how well agent can do tasks)
s_f5[4,1,type=int]     # Task_progress (current progress on assigned task)

# Social influence model
s_f6[8,1,type=int]     # Influence_network (who influences whom)
s_f7[4,1,type=int]     # Reputation (credibility with others)

## Connections
# Leadership emergence
(s_f1, s_f7, s_f2) -> leadership_transition_probability
leadership_transition_probability -> (B_f0)

# Task allocation by leader
(s_f0) -> task_allocation_authority
(task_allocation_authority, s_f4) -> optimal_task_assignment

# Follower compliance
(s_f0, s_f1) -> compliance_probability
(compliance_probability, task_assignment) -> (B_f3)

# Reputation dynamics
(s_f5) -> task_performance_signal
(task_performance_signal) -> (s_f7)  # Performance affects reputation
```

---

## 3. Learning and Adaptation

### Pattern: Bayesian Model Selection

**Use Case**: Agent learns which model of the world is correct.

```gnn
## StateSpaceBlock
# Model space
s_f0[3,1,type=int]     # Active_model (0:model_A, 1:model_B, 2:model_C)
s_f1[3,1,type=float]   # Model_evidence (posterior over models)

# Model-specific parameters
s_f2[4,1,type=float]   # ModelA_parameters (if model A is true)
s_f3[6,1,type=float]   # ModelB_parameters (if model B is true)  
s_f4[5,1,type=float]   # ModelC_parameters (if model C is true)

# Observations for model comparison
o_m0[4,1,type=int]     # Data_observation (evidence for model selection)
o_m1[2,1,type=int]     # Meta_observation (higher-order patterns)

# Learning control
u_c0[3,1,type=int]     # Information_seeking_action (gather evidence)
u_c1[2,1,type=int]     # Exploitation_action (act under best model)

# Model-dependent matrices
A_m0_modelA[4,4,type=float]  # Likelihood under model A
A_m0_modelB[4,6,type=float]  # Likelihood under model B  
A_m0_modelC[4,5,type=float]  # Likelihood under model C

## Connections
# Model selection dynamics
(s_f1) -> (s_f0)  # Posterior determines active model

# Model-conditional observation
(s_f0) -> model_selector
(model_selector, s_f2) -> A_m0_modelA
(model_selector, s_f3) -> A_m0_modelB
(model_selector, s_f4) -> A_m0_modelC

# Evidence accumulation
(o_m0) -> observation_likelihood
(observation_likelihood, s_f1) -> bayesian_update -> s_f1_next

# Information seeking behavior
(s_f1) -> information_value
(information_value) -> (C_m0)  # Prefer informative actions

## InitialParameterization
# Prior over models (uniform initially)
s_f1={(0.33, 0.33, 0.34)}

# Information seeking preference
C_m0={(1.0, 0.5, 0.8)}  # Prefer actions that disambiguate models

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=25
LearningRate=0.1
```

### Pattern: Habit Formation

**Use Case**: Gradual transition from deliberative to automatic behavior.

```gnn
## StateSpaceBlock
# Behavioral control systems
s_f0[4,1,type=int]     # Deliberative_state (conscious planning state)
s_f1[4,1,type=int]     # Habitual_state (automatic response state)
s_f2[1,1,type=float]   # Control_weight (deliberative vs habitual)

# Context and cues
s_f3[6,1,type=int]     # Context_state (environmental context)
s_f4[8,1,type=int]     # Cue_state (habit triggers)

# Action values and frequencies  
s_f5[5,5,type=float]   # Action_values (learned Q-values)
s_f6[5,5,type=int]     # Action_frequencies (how often actions taken)

# Observations
o_m0[4,1,type=int]     # Environmental_cue
o_m1[3,1,type=int]     # Reward_signal

# Actions  
u_c0[5,1,type=int]     # Available_actions

## Connections
# Dual control systems
(s_f0, s_f2) -> deliberative_contribution
(s_f1, s_f2) -> habitual_contribution
(deliberative_contribution + habitual_contribution) -> action_tendency

# Habit strength learning
(s_f6) -> habit_strength
(habit_strength) -> (s_f2)  # Frequent actions become more automatic

# Context-dependent cuing
(s_f3) -> (s_f4)  # Context activates cues
(s_f4) -> (s_f1)  # Cues trigger habitual responses

# Value learning
(o_m1, u_c0) -> reward_prediction_error
(reward_prediction_error) -> (s_f5)  # Update action values

# Frequency tracking
(u_c0) -> action_execution_signal
(action_execution_signal) -> (s_f6)  # Track action frequencies

## InitialParameterization
# Start with high deliberative control
s_f2={(0.9)}  # 90% deliberative, 10% habitual initially

# Neutral action values (to be learned)
s_f5={((0.0, 0.0, 0.0, 0.0, 0.0),
       (0.0, 0.0, 0.0, 0.0, 0.0),
       (0.0, 0.0, 0.0, 0.0, 0.0),
       (0.0, 0.0, 0.0, 0.0, 0.0),
       (0.0, 0.0, 0.0, 0.0, 0.0))}

## Time
Dynamic  
DiscreteTime=t
ModelTimeHorizon=100
HabitFormationRate=0.01
```

---

## 4. Temporal Dynamics

### Pattern: Predictive Coding

**Use Case**: Forward models for prediction and control.

```gnn
## StateSpaceBlock
# Predictive hierarchy
s_f0[6,1,type=int]     # Sensory_prediction (what we expect to sense)
s_f1[4,1,type=int]     # Motor_prediction (predicted action outcomes)
s_f2[3,1,type=int]     # State_prediction (predicted next state)

# Prediction errors
s_f3[6,1,type=float]   # Sensory_error (prediction - observation)
s_f4[4,1,type=float]   # Motor_error (predicted - actual outcome)
s_f5[3,1,type=float]   # State_error (predicted - actual state)

# Forward models
s_f6[12,1,type=float]  # Sensory_forward_model (parameters)
s_f7[16,1,type=float]  # Motor_forward_model (parameters)
s_f8[9,1,type=float]   # State_forward_model (parameters)

# Observations
o_m0[6,1,type=int]     # Actual_sensory_input
o_m1[4,1,type=int]     # Actual_motor_outcome
o_m2[3,1,type=int]     # Actual_state_transition

# Actions
u_c0[4,1,type=int]     # Motor_command

## Connections
# Forward model predictions
(s_f6, t) -> (s_f0)           # Sensory forward model
(s_f7, u_c0, t) -> (s_f1)    # Motor forward model  
(s_f8, u_c0, t) -> (s_f2)    # State forward model

# Prediction error computation
(s_f0, o_m0) -> (s_f3)       # Sensory prediction error
(s_f1, o_m1) -> (s_f4)       # Motor prediction error
(s_f2, o_m2) -> (s_f5)       # State prediction error

# Forward model learning (minimize prediction error)
(s_f3) -> sensory_model_update -> (s_f6)
(s_f4) -> motor_model_update -> (s_f7)
(s_f5) -> state_model_update -> (s_f8)

# Error-driven attention and control
(s_f3, s_f4, s_f5) -> total_prediction_error
(total_prediction_error) -> attention_allocation
(total_prediction_error) -> action_selection_bias

## InitialParameterization
# Preference for predictable outcomes (minimize surprise)
C_m0={(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)}  # Penalize prediction errors

# Learning rates for different forward models
sensory_learning_rate={(0.1)}
motor_learning_rate={(0.05)}
state_learning_rate={(0.02)}

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=10
PredictionWindow=3
```

### Pattern: Memory and Temporal Context

**Use Case**: Working memory and temporal context effects.

```gnn
## StateSpaceBlock
# Working memory slots
s_f0[12,1,type=int]    # Memory_slot_1 (4 time steps × 3 features)
s_f1[12,1,type=int]    # Memory_slot_2 (4 time steps × 3 features)
s_f2[12,1,type=int]    # Memory_slot_3 (4 time steps × 3 features)

# Memory control
s_f3[3,1,type=float]   # Memory_attention (attention to each slot)
s_f4[1,1,type=int]     # Active_memory_slot (which slot to update)
s_f5[1,1,type=float]   # Memory_decay_rate (forgetting parameter)

# Temporal context
s_f6[5,1,type=int]     # Context_buffer (recent context history)
s_f7[3,1,type=float]   # Context_weights (importance of recent contexts)

# Current processing
s_f8[4,1,type=int]     # Current_state (present moment state)
s_f9[3,1,type=int]     # Current_goal (active goal)

## Connections
# Memory encoding 
(s_f8, s_f4) -> memory_write_operation
(memory_write_operation) -> (s_f0, s_f1, s_f2)

# Memory retrieval
(s_f9, s_f3) -> memory_read_operation  
(memory_read_operation, s_f0, s_f1, s_f2) -> retrieved_memory

# Context-dependent processing
(s_f6, s_f7) -> temporal_context
(temporal_context, retrieved_memory) -> context_modulated_state

# Memory decay
(s_f5, t) -> decay_function
(decay_function) -> (s_f0, s_f1, s_f2)  # Apply forgetting

# Context buffer update
(s_f8) -> context_update
(context_update) -> (s_f6)  # Shift buffer, add current state

## InitialParameterization
# Equal attention to memory slots initially
s_f3={(0.33, 0.33, 0.34)}

# Moderate memory decay
s_f5={(0.05)}  # 5% decay per time step

# Recent context more important
s_f7={(0.5, 0.3, 0.2)}  # Decreasing weights for older context

## Time
Dynamic
DiscreteTime=t  
ModelTimeHorizon=20
MemoryCapacity=3
ContextWindow=5
```

---

## 5. Uncertainty and Robustness

### Pattern: Ambiguity Resolution

**Use Case**: Dealing with perceptual ambiguity and conflicting evidence.

```gnn
## StateSpaceBlock
# Competing interpretations
s_f0[4,1,type=int]     # Interpretation_A (one way to parse the scene)
s_f1[4,1,type=int]     # Interpretation_B (alternative interpretation)  
s_f2[4,1,type=int]     # Interpretation_C (third interpretation)

# Interpretation confidence
s_f3[3,1,type=float]   # Confidence_levels (posterior over interpretations)
s_f4[1,1,type=float]   # Ambiguity_level (entropy over interpretations)

# Evidence accumulation
s_f5[6,1,type=float]   # Evidence_A (support for interpretation A)
s_f6[6,1,type=float]   # Evidence_B (support for interpretation B)
s_f7[6,1,type=float]   # Evidence_C (support for interpretation C)

# Attention and exploration
s_f8[8,1,type=float]   # Attention_allocation (where to look for evidence)
s_f9[3,1,type=int]     # Exploration_strategy (how to gather information)

## Connections
# Evidence integration
(s_f5, s_f6, s_f7) -> evidence_comparison
(evidence_comparison) -> (s_f3)  # Update confidence

# Ambiguity monitoring
(s_f3) -> entropy_computation -> (s_f4)

# Attention guidance by uncertainty
(s_f4) -> uncertainty_driven_attention
(uncertainty_driven_attention) -> (s_f8)

# Active information seeking
(s_f8) -> optimal_exploration_action
(optimal_exploration_action) -> (s_f9)

# Interpretation-dependent action
(s_f3) -> interpretation_weighted_action
(interpretation_weighted_action) -> action_policy

## InitialParameterization
# Start with uniform interpretation priors
s_f3={(0.33, 0.33, 0.34)}

# High preference for reducing ambiguity
C_ambiguity_reduction={(2.0)}  # Strong drive to resolve uncertainty

## Equations
# Ambiguity (entropy) computation:
# H = -∑ p_i log(p_i) where p_i are interpretation probabilities

# Information gain for action a:
# IG(a) = H_current - E[H_after_action_a]

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=15
```

### Pattern: Risk-Sensitive Decision Making

**Use Case**: Decisions under uncertainty with risk preferences.

```gnn
## StateSpaceBlock
# Outcome uncertainty
s_f0[5,1,type=int]     # Possible_outcomes (different result scenarios)
s_f1[5,1,type=float]   # Outcome_probabilities (likelihood of each outcome)
s_f2[5,1,type=float]   # Outcome_utilities (value of each outcome)

# Risk assessment
s_f3[1,1,type=float]   # Variance_estimate (outcome uncertainty level)
s_f4[1,1,type=float]   # Downside_risk (probability of bad outcomes)
s_f5[1,1,type=float]   # Upside_potential (probability of good outcomes)

# Risk preferences
s_f6[1,1,type=float]   # Risk_tolerance (risk-seeking vs risk-averse)
s_f7[1,1,type=float]   # Loss_aversion (asymmetric value for losses)

# Decision variables
s_f8[4,1,type=int]     # Available_actions
s_f9[4,1,type=float]   # Action_expected_utilities
s_f10[4,1,type=float]  # Action_risk_adjustments

## Connections
# Uncertainty quantification
(s_f1, s_f2) -> variance_computation -> (s_f3)
(s_f1, s_f2) -> downside_calculation -> (s_f4)
(s_f1, s_f2) -> upside_calculation -> (s_f5)

# Risk-adjusted utility
(s_f9, s_f6, s_f7) -> risk_adjustment -> (s_f10)
(s_f10) -> risk_sensitive_action_selection

# Learning risk preferences from experience
(observed_outcomes, s_f2) -> outcome_prediction_error
(outcome_prediction_error) -> risk_preference_update -> (s_f6, s_f7)

## InitialParameterization
# Moderate risk aversion
s_f6={(0.3)}  # 0 = risk-neutral, <0 = risk-averse, >0 = risk-seeking

# Typical loss aversion
s_f7={(2.0)}  # Losses weighted 2x more than equivalent gains

## Equations
# Risk-adjusted utility:
# U_adj(a) = EU(a) - risk_tolerance × Var(a) - loss_aversion × P(loss|a)

# Where EU(a) is expected utility, Var(a) is variance, P(loss|a) is loss probability

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=12
```

---

## 6. Compositional Modeling

### Pattern: Modular Cognitive Architecture

**Use Case**: Decomposing complex cognition into reusable modules.

```gnn
## ModelName
Modular_Cognitive_Agent v1.0

## StateSpaceBlock
# Perception module
s_perception[8,1,type=int]    # Perceptual_features
o_perception[6,1,type=int]    # Sensory_inputs
A_perception[6,8,type=float]  # Perception_likelihood

# Attention module  
s_attention[4,1,type=int]     # Attention_focus
C_attention[6,type=float]     # Attention_preferences

# Memory module
s_memory[12,1,type=int]       # Memory_contents
u_memory[3,1,type=int]        # Memory_operations (read/write/forget)

# Planning module
s_planning[6,1,type=int]      # Planning_state
u_planning[4,1,type=int]      # Planning_actions

# Motor module
s_motor[5,1,type=int]         # Motor_preparation
u_motor[3,1,type=int]         # Motor_execution

# Control module (coordinates other modules)
s_control[8,1,type=int]       # Control_state
C_control[8,type=float]       # Control_priorities

## Connections
# Information flow between modules
(s_perception) -> (s_attention)  # Perception drives attention
(s_attention) -> attention_signal -> (C_perception)  # Attention modulates perception

(s_perception) -> (s_memory)     # Perceptual input to memory
(s_memory) -> memory_retrieval -> (s_planning)  # Memory informs planning

(s_planning) -> planning_output -> (s_motor)  # Planning drives motor preparation
(s_motor) -> (u_motor)           # Motor preparation leads to action

# Control coordination
(s_control) -> module_arbitration
(module_arbitration) -> (C_attention, C_memory, C_planning, C_motor)

# Modular interfaces (standardized communication)
perception_output = (s_perception, confidence_perception)
attention_output = (s_attention, attention_weights)
memory_output = (s_memory, memory_availability)
planning_output = (s_planning, plan_confidence)
motor_output = (s_motor, action_readiness)

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=10
ModuleUpdateSchedule={perception:1, attention:2, memory:3, planning:5, motor:1}
```

### Pattern: Nested Compositional Models

**Use Case**: Hierarchical composition of cognitive processes.

```gnn
## StateSpaceBlock
# Outer composition (system level)
s_system[3,1,type=int]        # System_mode (explore/exploit/maintain)

# Middle composition (subsystem level)  
s_navigation[4,1,type=int]    # Navigation_subsystem
s_manipulation[5,1,type=int]  # Manipulation_subsystem
s_communication[3,1,type=int] # Communication_subsystem

# Inner composition (component level)
s_nav_perception[6,1,type=int]      # Navigation_perception_component
s_nav_planning[8,1,type=int]        # Navigation_planning_component
s_manip_perception[4,1,type=int]    # Manipulation_perception_component
s_manip_control[6,1,type=int]       # Manipulation_control_component

## Connections
# Hierarchical control flow
(s_system) -> system_mode_signal
(system_mode_signal) -> (subsystem_activation_levels)
(subsystem_activation_levels) -> (s_navigation, s_manipulation, s_communication)

# Subsystem to component communication
(s_navigation) -> nav_subsystem_signal
(nav_subsystem_signal) -> (s_nav_perception, s_nav_planning)

(s_manipulation) -> manip_subsystem_signal  
(manip_subsystem_signal) -> (s_manip_perception, s_manip_control)

# Cross-cutting concerns (e.g., shared perception)
shared_perceptual_features -> (s_nav_perception, s_manip_perception)

# Component interaction within subsystems
(s_nav_perception) -> (s_nav_planning)
(s_manip_perception) -> (s_manip_control)

## InitialParameterization
# System-level preferences
C_system={(0.4, 0.4, 0.2)}  # Balanced explore/exploit, some maintenance

# Subsystem activation thresholds
subsystem_activation_threshold={(0.3, 0.5, 0.2)}  # Different activation levels
```

---

## 7. Graceful Degradation in the Pipeline

There is no model-level "solver escalation" construct in GNN — no circuit breaker, no
heuristic-override topology, no dual formal/heuristic matrix pair. Degradation is handled
by the pipeline, not encoded in your spec, and it works in two deliberately different
ways depending on what went wrong.

**A missing optional dependency skips a backend.** Step 12 runs a pre-flight check before
executing Python framework scripts (`src/gnn/execute/processor/`). If the backend's package
is absent — bnlearn being the standing case, with PyTorch absent on a plain
`uv sync` (the package ships in the `torch` extra) — that backend is recorded as *skipped* with a
dependency reason rather than failed. The run continues, downstream visualization and
reporting receive partial results from the backends that did run, and the logs say
exactly which were skipped and why.

**A failed inference does not degrade — it crashes.** This is the opposite policy, and
the contrast is the point. Every RxInfer strategy emits its `infer()` call without a
`try`/`catch`, carrying the comment *"NO try/catch — if infer() fails, the script crashes
with a clear error. This is deliberate: real RxInfer inference or nothing."* A result
silently produced by hand-rolled Bayesian updating would still be reported as RxInfer
inference, so the pipeline refuses to produce one. Step 12 likewise exits non-zero when
`validation.all_valid` is false.

The rule underneath both: **absence is reported, wrongness is refused.** A backend you
could not run is a known gap; a number you cannot trust is worse than no number.

For the analysis side of this, Step 24 (`src/gnn/intelligent_analysis/`) scores pipeline
health, performs failure root-cause analysis, detects per-step warning flags and
bottlenecks, and writes executive reports. It analyzes execution logs after the fact — it
does not participate in solver selection or route between model paths.

If you genuinely want to model an agent that switches between a precise and an
approximate strategy, that is a policy-selection problem, not a fallback mechanism:
represent both strategies as actions in a single control factor and let expected free
energy choose between them, as in the policy patterns earlier in this document.

---

## 8. Domain-Specific Patterns

### Pattern: Social Cognition

**Use Case**: Theory of Mind and social interaction modeling.

```gnn
## StateSpaceBlock
# Self-model
s_f0[4,1,type=int]     # Own_mental_state
s_f1[3,1,type=int]     # Own_intentions
s_f2[2,1,type=int]     # Own_emotions

# Other-model (Theory of Mind)
s_f3[4,1,type=int]     # Other_mental_state (what I think they think)
s_f4[3,1,type=int]     # Other_intentions (what I think they want)
s_f5[2,1,type=int]     # Other_emotions (what I think they feel)

# Recursive modeling (what I think they think I think)
s_f6[4,1,type=int]     # Recursive_mental_state
s_f7[1,1,type=int]     # Recursion_depth (how many levels deep)

# Social context
s_f8[6,1,type=int]     # Social_situation (formal/informal, competitive/cooperative, etc.)
s_f9[5,1,type=int]     # Social_roles (parent/child, leader/follower, etc.)
s_f10[3,1,type=int]    # Group_dynamics (cohesion, conflict, etc.)

# Communication and signaling
o_m0[8,1,type=int]     # Verbal_communication (what others say)
o_m1[6,1,type=int]     # Nonverbal_signals (body language, tone, etc.)
o_m2[4,1,type=int]     # Social_feedback (approval, disapproval, etc.)

u_c0[8,1,type=int]     # Verbal_response
u_c1[6,1,type=int]     # Nonverbal_behavior
u_c2[3,1,type=int]     # Social_action (cooperate, compete, withdraw)

## Connections
# Theory of Mind updating
(o_m0, o_m1) -> social_observation_integration
(social_observation_integration) -> (s_f3, s_f4, s_f5)

# Recursive modeling
(s_f3, s_f0) -> perspective_taking
(perspective_taking) -> (s_f6)

# Social context influences interpretation
(s_f8, s_f9) -> social_context_modulation
(social_context_modulation) -> interpretation_bias -> (A_m0, A_m1)

# Strategic communication
(s_f3, s_f4) -> strategic_communication_planning
(strategic_communication_planning) -> (C_m0, C_m1)  # Prefer actions that influence others' beliefs

## InitialParameterization
# Preference for positive social feedback
C_m2={(2.0, -1.0, 0.0, -0.5)}  # Strong preference for approval

# Theory of Mind accuracy (starts uncertain)
ToM_confidence={(0.6)}  # 60% confidence in reading others' minds

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=20
```

### Pattern: Language and Communication

**Use Case**: Natural language understanding and generation.

```gnn
## StateSpaceBlock
# Linguistic representation levels
s_f0[20,1,type=int]    # Phonological_features
s_f1[15,1,type=int]    # Morphological_features  
s_f2[25,1,type=int]    # Syntactic_structure
s_f3[30,1,type=int]    # Semantic_representation
s_f4[10,1,type=int]    # Pragmatic_context

# Language processing
s_f5[8,1,type=int]     # Working_memory_linguistic
s_f6[12,1,type=int]    # Language_attention
s_f7[6,1,type=int]     # Speech_motor_plan

# Compositional semantics
s_f8[40,1,type=int]    # Compositional_meaning (built from parts)
s_f9[20,1,type=int]    # Discourse_model (conversation state)

# Language production
s_f10[15,1,type=int]   # Message_intention
s_f11[18,1,type=int]   # Linguistic_formulation
s_f12[12,1,type=int]   # Speech_articulation

## Connections
# Bottom-up language comprehension
(o_phonetic) -> (s_f0) -> (s_f1) -> (s_f2) -> (s_f3)

# Top-down prediction and context
(s_f4, s_f9) -> contextual_predictions
(contextual_predictions) -> (s_f3, s_f2, s_f1, s_f0)

# Compositional meaning construction
(s_f2, s_f3) -> compositional_semantics -> (s_f8)

# Language production pipeline
(s_f10) -> (s_f11) -> (s_f7) -> (speech_output)

# Pragmatic reasoning
(s_f3, s_f4, social_context) -> pragmatic_inference
(pragmatic_inference) -> (s_f8)  # Update meaning based on context

## InitialParameterization
# Language comprehension preferences
C_comprehension={(1.0, 0.8, 0.6, 0.9, 0.7)}  # Weight different linguistic levels

# Production fluency vs accuracy trade-off
production_accuracy_weight={(0.8)}
production_fluency_weight={(0.6)}

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=30
LanguageProcessingWindow=5
```

---

## 9. Continuous Linear-Gaussian Models

Every pattern above is **discrete categorical**: states, observations, and
actions live on small supports, and the parameterization declares
`A`/`B`/`C`/`D`/`E` keys. GNN treats a second model kind as first-class: the
**continuous linear-Gaussian state-space model (LGSSM)**. A spec that declares
`F` (state transition), `H` (observation matrix), `Q`/`R` (process and
observation noise covariances), and `prior_mean`/`prior_cov` (Gaussian prior)
is classified as continuous and rendered natively on JAX, NumPyro, PyTorch,
Stan, and RxInfer.jl. The categorical backends (PyMDP, ActiveInference.jl,
DisCoPy, bnlearn) report such models `unsupported` — a status, not a failure.

### Pattern: Passive Filtering (LGSSM)

```text
## StateSpaceBlock
x[2,1,type=float]        # continuous latent state
y[2,1,type=float]        # continuous observation
F[2,2,type=float]        # state transition
H[2,2,type=float]        # observation matrix
Q[2,2,type=float]        # process-noise covariance
R[2,2,type=float]        # observation-noise covariance
prior_mean[2,type=float] # prior mean over the initial latent state
prior_cov[2,2,type=float]# prior covariance over the initial latent state
```

Inference is Gaussian filtering/smoothing — no softmax perception, no policy
enumeration. Runnable exemplars: `input/gnn_files/continuous/stochastic_dynamics.md`
(minimal drift-plus-noise LGSSM) and `input/gnn_files/continuous/predictive_coding_agent.md`
(passive predictive coding in generalized coordinates).

### Pattern: Closed-Loop Control on Beliefs (LGSSM + goal)

Adding `goal_mean` (preferred state) and a scalar `control_gain` closes the
loop: a control input `u` is added to the state each step, steering the
filtered posterior mean toward the goal,

$$u_t = \text{control\_gain} \times (\text{goal\_mean} - \mu_t)$$

with connections `goal_mean>u` and `control_gain>u`. Runnable exemplar:
`input/gnn_files/continuous/continuous_navigation.md` — a 2D navigator whose
controller pushes the position belief mean toward `goal_mean = (2.0, 2.0)`.

When to reach for the continuous kind: the latent quantity is genuinely
metric (position, angle, intensity, generalized coordinates of motion), the
observation channel is a noisy linear readout, and dynamics are well
approximated locally by a linear Gaussian map. For discrete choices on top of
continuous latents, keep the hybrid decomposition explicit — declare the
categorical layer with the standard `A`/`B`/`C`/`D`/`E` keys and treat the
continuous layer as a separate model or submodel rather than mixing the two
parameterization families in one block.

---

## 📚 Implementation Guidelines

### 1. Pattern Selection

### Observer Pattern example

```python
# The Observer pattern can be used to notify different parts of a GNN system
# when a state changes, e.g., a belief update or a policy change.


class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def notify(self, state_change):
        for observer in self._observers:
            observer.update(state_change)


class Observer:
    def update(self, state_change):
        print(f"State changed: {state_change}")


class BeliefStateMonitor(Observer):
    def update(self, state_change):
        print(f"Belief state updated: {state_change}")


class PolicyExecutor(Observer):
    def update(self, state_change):
        if "policy_ready" in state_change:
            print("Executing new policy based on state change.")


# Usage:
# belief_subject = Subject()
# monitor = BeliefStateMonitor()
# executor = PolicyExecutor()

# belief_subject.attach(monitor)
# belief_subject.attach(executor)

# # When GNN belief state changes
# belief_subject.notify({"belief_state": "updated", "policy_ready": True})
```

**Choose patterns based on**:

- **Problem complexity**: Start simple, add complexity gradually
- **Available data**: Some patterns require more training data
- **Computational constraints**: Complex patterns need more resources
- **Domain requirements**: Some domains favor certain patterns

### 2. Pattern Combination

**Combining patterns effectively**:

```gnn
# Example: Hierarchical + Multi-agent + Learning
## StateSpaceBlock
# Individual agent hierarchy
s_individual_fast[4,1,type=int]
s_individual_slow[2,1,type=int]

# Multi-agent coordination
s_coordination[8,1,type=int]

# Learning across levels
s_learning_fast[12,1,type=float]
s_learning_slow[6,1,type=float]
s_learning_coordination[16,1,type=float]

## Connections
# Cross-pattern interactions
(s_individual_slow) -> (s_coordination)  # Slow planning coordinates
(s_coordination) -> (s_individual_fast)  # Coordination affects fast responses
(s_learning_coordination) -> coordination_learning_signal
```

### 3. Testing and Validation

**Pattern validation checklist**:

- [ ] **Mathematical consistency**: Probability constraints satisfied
- [ ] **Behavioral plausibility**: Produces reasonable agent behavior  
- [ ] **Computational efficiency**: Runs in acceptable time
- [ ] **Empirical validation**: Matches relevant experimental data
- [ ] **Robustness**: Works across different parameter settings
- [ ] **Interpretability**: Model components have clear meanings

### 4. Common Pitfalls

**Avoid these mistakes**:

- **Over-engineering**: Don't add complexity without clear benefit
- **Disconnected components**: Ensure all model parts interact meaningfully
- **Scale mismatches**: Match temporal and spatial scales appropriately
- [ ] **Ignored constraints**: Respect computational and biological plausibility
- [ ] **Poor modularity**: Design for reusability and composability

---

## 🚀 Future Directions

### Emerging Patterns

- **Continual learning**: Models that learn continuously without forgetting
- **Meta-learning**: Learning to learn from few examples
- **Causal reasoning**: Understanding and manipulating causal relationships
- **Embodied cognition**: Tight coupling between body, brain, and environment

### Research Opportunities

- **Pattern discovery**: Automated identification of useful patterns

- ### Factory Pattern implementation

```python
# The Factory pattern can be used to create different types of GNN agents
# or modules based on configuration, without specifying the exact class.


class AgentFactory:
    @staticmethod
    def create_agent(agent_type, config):
        if agent_type == "simple":
            return SimpleGNNAgent(config)
        elif agent_type == "hierarchical":
            return HierarchicalGNNAgent(config)
        elif agent_type == "multi_agent":
            return MultiAgentGNNAgent(config)
        else:
            raise ValueError("Unknown agent type")


class SimpleGNNAgent:
    def __init__(self, config):
        self.config = config
        print(f"Creating Simple GNN Agent with config: {config}")


class HierarchicalGNNAgent:
    def __init__(self, config):
        self.config = config
        print(f"Creating Hierarchical GNN Agent with config: {config}")


class MultiAgentGNNAgent:
    def __init__(self, config):
        self.config = config
        print(f"Creating Multi-Agent GNN Agent with config: {config}")


# Usage:
# agent1 = AgentFactory.create_agent("simple", {"learning_rate": 0.01})
# agent2 = AgentFactory.create_agent("hierarchical", {"levels": 3, "time_constants": [0.1, 1.0, 10.0]})
```

- **Pattern optimization**: Learning optimal pattern combinations
- **Cross-domain transfer**: Adapting patterns across different domains
- **Biological validation**: Testing patterns against neuroscience data

---

## Category theory

Compositional semantics for GNN ↔ DisCoPy: see [DisCoPy integration](../../discopy/gnn_discopy.md) and [Mathematical Foundations](#mathematical-foundations).

## Causal inference

Explicit causal structure appears in GNN `## Connections` blocks and validation (Steps 5-6); pair with [Hierarchical Modeling](#1-hierarchical-modeling) and other patterns above.

## Decision theory

Risk and utility: see [Pattern: Risk-Sensitive Decision Making](#pattern-risk-sensitive-decision-making).

## Geometric foundations

Spatial embeddings: see [Pattern: Spatial Hierarchies](#pattern-spatial-hierarchies).

## Hierarchical architectures

Layered models: see [Hierarchical Modeling](#1-hierarchical-modeling) and [Pattern: Temporal Hierarchies](#pattern-temporal-hierarchies).

## Learning algorithms

Inference and learning hooks are framework-specific after Step 11 render; see [Pipeline Processing for Advanced Models](#pipeline-processing-for-advanced-models).

## Mathematical foundations

Priors, transitions, and constraints are expressed through `A`/`B`/`C`/`D` blocks throughout this guide—start from [Overview](#overview).

## Social cognition

Multi-agent and ToM-style patterns: see [2. Multi-Agent Systems](#2-multi-agent-systems) and [Pattern: Language and Communication](#pattern-language-and-communication).

## Temporal hierarchy

Slow–fast decompositions: see [Pattern: Temporal Hierarchies](#pattern-temporal-hierarchies).

**This guide provides a foundation for sophisticated GNN modeling. Start with simpler patterns and gradually incorporate complexity as needed for your specific application domain.**
