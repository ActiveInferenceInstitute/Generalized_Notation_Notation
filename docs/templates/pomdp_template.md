# POMDP Active Inference Template

This template provides a starting point for Partially Observable Markov Decision Process (POMDP) models using Active Inference principles.

## GNNSection
[YourGNNSection]

## GNNVersionAndFlags
GNN v1.0
ProcessingFlags: active_inference,pomdp

## ModelName
[YourPOMDPModelName]

## ModelAnnotation
[Describe your POMDP model - what domain does it address? What does the agent need to accomplish?
Example: A navigation agent that must reach a goal location while avoiding obstacles in a partially observable grid world.
Key features to mention: observation uncertainty, action consequences, learning requirements, temporal dynamics.]

## StateSpaceBlock
### Hidden State Factors
s_f0[4,1,type=categorical]      ### Agent position: North=0, South=1, East=2, West=3
s_f1[3,1,type=categorical]      ### Environmental context: Safe=0, Dangerous=1, Goal=2

### Observation Modalities
o_m0[3,1,type=categorical]      ### Visual sensor: Clear=0, Obstacle=1, Goal=2
o_m1[2,1,type=categorical]      ### Audio sensor: Quiet=0, Alarm=1

### Control Factors (Actions)
u_c0[4,1,type=categorical]      ### Movement actions: Forward=0, Back=1, Left=2, Right=3
u_c1[2,1,type=categorical]      ### Communication: Listen=0, Send=1

### Policy Factors
π_c0[4,1,type=categorical]      ### Movement policies: Conservative=0, Exploratory=1, Direct=2, Random=3
π_c1[2,1,type=categorical]      ### Communication policies: Silent=0, Active=1

## Connections
### Observation Dependencies (Likelihood model)
s_f0 > o_m0                     ### Position influences visual observations
s_f1 > o_m0                     ### Context influences what is seen
s_f1 > o_m1                     ### Context influences audio (alarms in dangerous areas)

### Transition Dependencies (Dynamics model)
(s_f0, u_c0) > s_f0             ### Position changes based on movement actions
(s_f1, u_c1) > s_f1             ### Context may change based on communication

### Policy Dependencies
π_c0 > u_c0                     ### Movement policy determines movement action
π_c1 > u_c1                     ### Communication policy determines communication action

### Cross-factor Dependencies (if applicable)
s_f0 - s_f1                     ### Position and context may be correlated

## InitialParameterization
### Likelihood Matrices (A matrices) - P(observation|states)
A_m0 = [
    [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]],   # Clear, Obstacle, Goal | (Position, Context) combinations
    [[0.8, 0.2, 0.0], [0.2, 0.7, 0.1], [0.0, 0.2, 0.8]],   # Different observation probabilities per state combination
    [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.1, 0.1, 0.8]],
    [[0.9, 0.1, 0.0], [0.3, 0.6, 0.1], [0.0, 0.0, 1.0]]
]  ### P(o_m0|s_f0,s_f1)

A_m1 = [
    [0.9, 0.1], [0.3, 0.7], [0.1, 0.9]                      # Audio observations depend mainly on context
]  ### P(o_m1|s_f1)

### Transition Matrices (B matrices) - P(next_state|current_state,action)
B_f0 = [
    # Position transitions for each movement action
    [[0.8, 0.1, 0.05, 0.05], [0.1, 0.8, 0.05, 0.05], [0.05, 0.05, 0.8, 0.1], [0.05, 0.05, 0.1, 0.8]],  # Forward
    [[0.8, 0.1, 0.05, 0.05], [0.1, 0.8, 0.05, 0.05], [0.05, 0.05, 0.8, 0.1], [0.05, 0.05, 0.1, 0.8]],  # Back  
    [[0.7, 0.2, 0.05, 0.05], [0.2, 0.7, 0.05, 0.05], [0.05, 0.05, 0.7, 0.2], [0.05, 0.05, 0.2, 0.7]],  # Left
    [[0.7, 0.2, 0.05, 0.05], [0.2, 0.7, 0.05, 0.05], [0.05, 0.05, 0.7, 0.2], [0.05, 0.05, 0.2, 0.7]]   # Right
]  ### P(s_f0'|s_f0,u_c0)

B_f1 = [
    # Context transitions (mostly stable, slight changes possible)
    [[0.9, 0.1, 0.0], [0.05, 0.9, 0.05], [0.0, 0.1, 0.9]],  # Listen action
    [[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]] # Send action  
]  ### P(s_f1'|s_f1,u_c1)

### Preference Vectors (C vectors) - Log preferences over observations
C_m0 = [0.0, -2.0, 4.0]        ### Neutral about clear, avoid obstacles, strongly prefer goal
C_m1 = [1.0, -1.0]             ### Slight preference for quiet, avoid alarms

### Prior Beliefs (D vectors) - Initial state beliefs
D_f0 = [0.4, 0.3, 0.2, 0.1]    ### Initial position uncertainty (higher prob for certain positions)
D_f1 = [0.7, 0.2, 0.1]         ### Start in safe context most likely

### Precision Parameters
gamma = 2.0                     ### Policy precision (higher = more deterministic)
alpha = 1.0                     ### Action precision
beta = 1.0                      ### Expected free energy precision

## Equations
### Expected Free Energy (G) for policy evaluation
G_π = \sum_{\tau=1}^{T} \mathbb{E}_{Q(s_\tau|π)} \left[ D_{KL}[Q(o_\tau|s_\tau) || P(o_\tau|s_\tau)] - \mathbb{E}_{Q(o_\tau|s_\tau)}[\ln P(o_\tau)] \right]

### Posterior beliefs over states
Q(s_t|o_{1:t}) = \sigma\left( \ln A^T o_t + \ln B^T Q(s_{t+1}|o_{1:t}) + \ln D \right)

### Policy posterior (softmax over negative expected free energy)
Q(π) = \sigma(-\gamma G_π)

### Action sampling from selected policy
P(u_t|π) = \text{according to policy } π \text{ at time } t

## Time
Dynamic
DiscreteTime = t
ModelTimeHorizon = 10

## ActInfOntologyAnnotation
### Map variables to Active Inference Ontology
s_f0 = HiddenStateFactor0_Position
s_f1 = HiddenStateFactor1_Context
o_m0 = ObservationModality0_Visual
o_m1 = ObservationModality1_Audio
u_c0 = ControlFactor0_Movement
u_c1 = ControlFactor1_Communication
π_c0 = PolicyFactor0_MovementPolicy
π_c1 = PolicyFactor1_CommunicationPolicy
A_m0 = LikelihoodMatrix_Visual
A_m1 = LikelihoodMatrix_Audio
B_f0 = TransitionMatrix_Position
B_f1 = TransitionMatrix_Context
C_m0 = PreferenceVector_Visual
C_m1 = PreferenceVector_Audio
D_f0 = PriorBelief_Position
D_f1 = PriorBelief_Context
G = ExpectedFreeEnergy
gamma = PolicyPrecision
alpha = ActionPrecision

## Footer
Created: [Date]
LastModified: [Date]
Version: 1.0

## Signature
ModelCreator: [Your Name]
Institution: [Your Institution]
Email: [Your Email]
License: [License type]

---

## Usage Instructions

1. **Copy this template** to create your POMDP model

2. **Customize the state space**:
   - Define what hidden states your agent needs to track
   - Specify what observations are available
   - Design the action space for your domain

3. **Set up the matrices**:
   - **A matrices**: How do states relate to observations? (likelihood)
   - **B matrices**: How do actions change states? (dynamics)
   - **C vectors**: What does the agent prefer? (goals/rewards)
   - **D vectors**: What does the agent initially believe?

4. **Configure precision parameters**:
   - **gamma**: Higher values make policy selection more deterministic
   - **alpha**: Higher values make action selection more deterministic  
   - **beta**: Controls balance between exploitation and exploration

5. **Test your model**:
   ```bash
   uv run python src/gnn/main.py --target-dir path/to/your/pomdp_models
   ```

## POMDP Design Patterns

### Navigation Tasks
- **States**: Position, velocity, orientation
- **Observations**: Sensor readings (vision, lidar, GPS)
- **Actions**: Movement commands (forward, turn, stop)
- **Preferences**: Reach goal, avoid obstacles, minimize energy

### Object Recognition
- **States**: Object identity, object presence, attention location
- **Observations**: Visual features, spatial information
- **Actions**: Eye movements, attention shifts, classification decisions
- **Preferences**: Correct classification, minimize uncertainty

### Resource Management
- **States**: Resource levels, market conditions, system state
- **Observations**: Sensor readings, market signals, user feedback
- **Actions**: Resource allocation, purchasing, selling
- **Preferences**: Maintain resources, maximize utility, minimize cost

### Communication/Social Interaction
- **States**: Other agent intentions, communication state, relationship state
- **Observations**: Messages received, behavioral cues, environmental signals
- **Actions**: Send messages, behavioral responses, alliance formation
- **Preferences**: Successful communication, positive relationships, goal achievement

## Common POMDP Challenges

### Partial Observability
- Ensure A matrices properly capture observation uncertainty
- Use multiple observation modalities when appropriate
- Consider temporal correlations in observations

### Action Selection
- Balance exploration vs exploitation through C vectors and precision
- Use hierarchical policies for complex action spaces
- Consider action costs in preference specification

### State Space Design
- Include only necessary state factors to avoid curse of dimensionality
- Use factorized representations when states are independent
- Consider temporal dependencies in state transitions

### Parameter Tuning
- Start with simple matrices and gradually increase complexity
- Use symmetry and structure to reduce parameter space
- Validate parameters through simulation before deployment

## Validation Checklist

- [ ] All probability matrices sum to 1 along appropriate dimensions
- [ ] State space captures all relevant aspects of the environment
- [ ] Observation model reflects realistic sensor characteristics
- [ ] Action effects are properly modeled in transition matrices
- [ ] Preferences align with desired behavior
- [ ] Model validates without errors in type checker
- [ ] Generated code runs successfully in target framework
- [ ] Simulation results match expected behavior patterns

## Related Templates

- [Basic GNN Template](basic_gnn_template.md) - For simpler models
- [Multi-agent Template](multiagent_template.md) - For multiple interacting agents
- [Hierarchical Template](hierarchical_template.md) - For multi-level architectures

## References

- [Active Inference Tutorial](../gnn/about_gnn.md)
- [POMDP Solving Methods](../gnn/integration/gnn_implementation.md)
- [PyMDP POMDP Examples](../pymdp/gnn_pymdp.md)
- [RxInfer POMDP Models](../rxinfer/gnn_rxinfer.md) 