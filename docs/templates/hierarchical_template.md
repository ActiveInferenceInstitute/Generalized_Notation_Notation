# Hierarchical Active Inference Template

This template provides a framework for hierarchical Active Inference models with multiple temporal and abstraction levels.

## GNNSection
[YourGNNSection]

## GNNVersionAndFlags
GNN v1.0
ProcessingFlags: active_inference,hierarchical,temporal_abstraction

## ModelName
[YourHierarchicalModelName]

## ModelAnnotation
[Describe your hierarchical model - what are the different levels of abstraction? How do they interact?
Example: A cognitive architecture with high-level goal planning, mid-level action sequencing, and low-level motor control.
Each level operates at different timescales and abstractions.
Key features: temporal hierarchy, abstraction levels, top-down/bottom-up information flow, multi-scale optimization.]

## StateSpaceBlock
### High-Level States (Abstract Goals/Context)
s_high_f0[4,1,type=categorical]     ### Abstract goals: Explore=0, Exploit=1, Rest=2, Communicate=3
s_high_f1[3,1,type=categorical]     ### Context level: Local=0, Regional=1, Global=2
s_high_f2[2,1,type=categorical]     ### Planning horizon: Short=0, Long=1

### Mid-Level States (Action Sequences/Subgoals)
s_mid_f0[8,1,type=categorical]      ### Action sequences: Seq0=0, Seq1=1, ..., Seq7=7
s_mid_f1[5,1,type=categorical]      ### Sequence progress: Start=0, Quarter=1, Half=2, ThreeQuarter=3, End=4
s_mid_f2[3,1,type=categorical]      ### Sequence type: Motor=0, Cognitive=1, Social=2

### Low-Level States (Immediate Actions/Sensorimotor)
s_low_f0[6,1,type=categorical]      ### Immediate actions: Move_N=0, Move_S=1, Move_E=2, Move_W=3, Stay=4, Act=5
s_low_f1[4,1,type=categorical]      ### Motor state: Idle=0, Moving=1, Acting=2, Transitioning=3
s_low_f2[3,1,type=categorical]      ### Sensory focus: Visual=0, Auditory=1, Tactile=2

### Hierarchical Observations
o_high_m0[3,1,type=categorical]     ### High-level feedback: Success=0, Partial=1, Failure=2
o_high_m1[4,1,type=categorical]     ### Context observations: Stable=0, Changing=1, Novel=2, Familiar=3

o_mid_m0[5,1,type=categorical]      ### Mid-level progress: NotStarted=0, InProgress=1, Completed=2, Interrupted=3, Error=4
o_mid_m1[3,1,type=categorical]      ### Sequence feedback: OnTrack=0, Deviation=1, Correction=2

o_low_m0[7,1,type=categorical]      ### Low-level sensory: Empty=0, Object=1, Agent=2, Obstacle=3, Target=4, Boundary=5, Unknown=6
o_low_m1[4,1,type=categorical]      ### Immediate feedback: Success=0, Failure=1, Blocked=2, Partial=3

### Hierarchical Actions/Policies
u_high_c0[4,1,type=categorical]     ### High-level decisions: Continue=0, Switch=1, Interrupt=2, Restart=3
u_high_c1[3,1,type=categorical]     ### Context switching: Maintain=0, Expand=1, Contract=2

u_mid_c0[6,1,type=categorical]      ### Mid-level control: Execute=0, Pause=1, Skip=2, Repeat=3, Modify=4, Abort=5
u_mid_c1[2,1,type=categorical]      ### Sequence selection: Current=0, Alternative=1

u_low_c0[6,1,type=categorical]      ### Low-level actions: Move_N=0, Move_S=1, Move_E=2, Move_W=3, Stay=4, Act=5
u_low_c1[3,1,type=categorical]      ### Execution mode: Normal=0, Careful=1, Fast=2

### Cross-Level Policy Factors
π_high[3,1,type=categorical]        ### High-level policies: Conservative=0, Balanced=1, Aggressive=2
π_mid[4,1,type=categorical]         ### Mid-level policies: Sequential=0, Parallel=1, Adaptive=2, Reactive=3
π_low[2,1,type=categorical]         ### Low-level policies: Precise=0, Flexible=1

## Connections
### Within-Level Dependencies
(s_high_f0, s_high_f1) > o_high_m0 ### High-level observations depend on goals and context
(s_mid_f0, s_mid_f1) > o_mid_m0    ### Mid-level observations depend on sequences and progress
(s_low_f0, s_low_f1) > o_low_m0    ### Low-level observations depend on actions and motor state

### Cross-Level Top-Down Dependencies (Higher levels influence lower)
s_high_f0 > s_mid_f0               ### High-level goals influence mid-level sequences
s_high_f2 > s_mid_f1               ### Planning horizon affects sequence progress
s_mid_f0 > s_low_f0                ### Mid-level sequences determine low-level actions
s_mid_f2 > s_low_f2                ### Sequence type affects sensory focus

### Cross-Level Bottom-Up Dependencies (Lower levels inform higher)
o_low_m1 > s_mid_f1                ### Low-level feedback updates sequence progress  
o_mid_m0 > s_high_f0               ### Mid-level progress informs goal states
(s_low_f1, s_mid_f1) > o_high_m0   ### Combined lower-level states create high-level observations

### Transition Dependencies
(s_high_f0, u_high_c0, o_high_m0) > s_high_f0  ### High-level state transitions
(s_mid_f0, u_mid_c0, s_high_f0) > s_mid_f0     ### Mid-level influenced by high-level
(s_low_f0, u_low_c0, s_mid_f0) > s_low_f0      ### Low-level influenced by mid-level

### Policy Dependencies  
π_high > u_high_c0                 ### High-level policy determines high-level actions
π_mid > u_mid_c0                   ### Mid-level policy determines mid-level actions
π_low > u_low_c0                   ### Low-level policy determines low-level actions

π_high > π_mid                     ### Higher-level policies influence lower-level policies
π_mid > π_low                      ### Hierarchical policy influence

## InitialParameterization
### High-Level Likelihood Matrices
A_high_m0 = [
    [[0.8, 0.15, 0.05], [0.3, 0.6, 0.1], [0.1, 0.3, 0.6], [0.6, 0.3, 0.1]],  # Goal-context combinations
    [[0.7, 0.2, 0.1], [0.4, 0.5, 0.1], [0.2, 0.4, 0.4], [0.5, 0.4, 0.1]],
    [[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.7, 0.2, 0.1]]
]  ### P(o_high_m0|s_high_f0,s_high_f1)

A_high_m1 = [
    [0.7, 0.2, 0.05, 0.05], [0.1, 0.6, 0.2, 0.1], [0.05, 0.05, 0.8, 0.1]
]  ### P(o_high_m1|s_high_f1)

### Mid-Level Likelihood Matrices
A_mid_m0 = [
    # Progress observations based on sequence state and progress
    [[[0.9, 0.1, 0.0, 0.0, 0.0], [0.2, 0.7, 0.1, 0.0, 0.0], [0.0, 0.2, 0.7, 0.1, 0.0], [0.0, 0.0, 0.2, 0.7, 0.1], [0.0, 0.0, 0.0, 0.3, 0.7]],
     # Additional sequences...
     [[0.85, 0.1, 0.05, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0, 0.0], [0.0, 0.1, 0.8, 0.1, 0.0], [0.0, 0.0, 0.1, 0.8, 0.1], [0.0, 0.0, 0.0, 0.2, 0.8]]]
]  ### P(o_mid_m0|s_mid_f0,s_mid_f1)

A_mid_m1 = [
    [0.8, 0.15, 0.05], [0.6, 0.3, 0.1], [0.4, 0.4, 0.2]
]  ### P(o_mid_m1|s_mid_f2)

### Low-Level Likelihood Matrices
A_low_m0 = [
    # Sensory observations based on actions and motor state  
    [[[0.7, 0.1, 0.05, 0.1, 0.03, 0.01, 0.01], [0.6, 0.15, 0.1, 0.1, 0.03, 0.01, 0.01]],  # Action-motor combinations
     [[0.5, 0.2, 0.1, 0.15, 0.03, 0.01, 0.01], [0.4, 0.25, 0.15, 0.15, 0.03, 0.01, 0.01]],
     # More action-motor combinations...
     [[0.8, 0.05, 0.05, 0.05, 0.03, 0.01, 0.01], [0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01]]]
]  ### P(o_low_m0|s_low_f0,s_low_f1)

A_low_m1 = [
    [0.8, 0.1, 0.05, 0.05], [0.2, 0.6, 0.15, 0.05], [0.1, 0.3, 0.5, 0.1], [0.3, 0.4, 0.2, 0.1]
]  ### P(o_low_m1|s_low_f1)

### High-Level Transition Matrices
B_high_f0 = [
    # High-level goal transitions (slow timescale)
    [[[0.9, 0.05, 0.03, 0.02], [0.1, 0.8, 0.05, 0.05], [0.02, 0.08, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85]],  # Continue
     [[0.3, 0.4, 0.2, 0.1], [0.2, 0.5, 0.2, 0.1], [0.1, 0.2, 0.6, 0.1], [0.2, 0.2, 0.2, 0.4]],              # Switch
     [[0.5, 0.2, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1], [0.3, 0.3, 0.3, 0.1], [0.3, 0.2, 0.2, 0.3]],              # Interrupt
     [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]]]               # Restart
]  ### P(s_high_f0'|s_high_f0,u_high_c0,o_high_m0)

B_high_f1 = [
    # Context transitions
    [[[0.85, 0.1, 0.05], [0.3, 0.6, 0.1], [0.2, 0.3, 0.5]],   # Maintain
     [[0.2, 0.7, 0.1], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]],   # Expand  
     [[0.8, 0.15, 0.05], [0.6, 0.3, 0.1], [0.4, 0.4, 0.2]]]   # Contract
]  ### P(s_high_f1'|s_high_f1,u_high_c1)

### Mid-Level Transition Matrices (influenced by high-level)
B_mid_f0 = [
    # Sequence transitions depend on mid-level control and high-level state
    # This creates a complex tensor: P(s_mid_f0'|s_mid_f0,u_mid_c0,s_high_f0)
    [[[[0.8, 0.05, 0.05, 0.05, 0.02, 0.01, 0.01, 0.01] for _ in range(8)] for _ in range(6)] for _ in range(4)]
]  ### P(s_mid_f0'|s_mid_f0,u_mid_c0,s_high_f0) - Simplified representation

B_mid_f1 = [
    # Progress transitions (faster timescale than high-level)
    [[[0.8, 0.15, 0.03, 0.01, 0.01], [0.1, 0.7, 0.15, 0.03, 0.02], [0.0, 0.2, 0.6, 0.15, 0.05], [0.0, 0.0, 0.3, 0.6, 0.1], [0.0, 0.0, 0.0, 0.2, 0.8]],  # Execute
     [[0.9, 0.1, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0]]]       # Pause
]  ### P(s_mid_f1'|s_mid_f1,u_mid_c0)

### Low-Level Transition Matrices (influenced by mid-level)
B_low_f0 = [
    # Low-level action transitions depend on low-level control and mid-level state
    [[[[0.7 if i==j else 0.05 for j in range(6)] for i in range(6)] for _ in range(8)] for _ in range(6)]
]  ### P(s_low_f0'|s_low_f0,u_low_c0,s_mid_f0) - Simplified representation

B_low_f1 = [
    # Motor state transitions (fastest timescale)
    [[[0.8, 0.1, 0.05, 0.05], [0.2, 0.6, 0.15, 0.05], [0.1, 0.3, 0.5, 0.1], [0.3, 0.3, 0.3, 0.1]],  # Normal execution
     [[0.9, 0.05, 0.03, 0.02], [0.1, 0.8, 0.05, 0.05], [0.05, 0.1, 0.8, 0.05], [0.1, 0.1, 0.1, 0.7]]]  # Careful execution
]  ### P(s_low_f1'|s_low_f1,u_low_c1)

### Hierarchical Preference Vectors
C_high_m0 = [2.0, 1.0, -1.0]       ### High-level: Prefer success, neutral partial, avoid failure
C_high_m1 = [1.0, 0.0, -0.5, 0.5]  ### Context: Slight preference for stable, avoid changing

C_mid_m0 = [0.0, 0.5, 2.0, -1.0, -2.0]  ### Mid-level: Neutral start, slight progress preference, strong completion preference, avoid interruption/error
C_mid_m1 = [1.0, 0.0, -0.5]             ### Sequence: Prefer on track, neutral deviation, slight avoid correction

C_low_m0 = [0.0, 0.5, -0.5, -2.0, 3.0, 0.0, -1.0]  ### Low-level: Target seeking, obstacle avoidance
C_low_m1 = [2.0, -2.0, -1.0, 0.5]                   ### Immediate: Prefer success, avoid failure/blocking

### Hierarchical Prior Beliefs
D_high_f0 = [0.4, 0.3, 0.2, 0.1]   ### Start with exploration preference
D_high_f1 = [0.7, 0.2, 0.1]        ### Start local
D_high_f2 = [0.6, 0.4]             ### Slight preference for short-term planning

D_mid_f0 = [0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1]  ### Uniform over sequences initially
D_mid_f1 = [0.8, 0.1, 0.05, 0.03, 0.02]                ### Start at beginning of sequences
D_mid_f2 = [0.4, 0.4, 0.2]                             ### Balanced sequence types

D_low_f0 = [0.2, 0.2, 0.2, 0.2, 0.15, 0.05]  ### Slight preference for movement over staying/acting
D_low_f1 = [0.7, 0.15, 0.1, 0.05]            ### Start idle
D_low_f2 = [0.5, 0.3, 0.2]                   ### Visual focus initially

### Temporal Precision Parameters
gamma_high = 1.0                    ### Lower precision for high-level (more exploration)
gamma_mid = 1.5                     ### Medium precision for mid-level
gamma_low = 2.0                     ### Higher precision for low-level (more deterministic)

alpha_high = 0.8                    ### Lower action precision for high-level
alpha_mid = 1.2                     ### Medium action precision for mid-level  
alpha_low = 1.8                     ### Higher action precision for low-level

### Cross-Level Coupling Parameters
top_down_strength = 1.5             ### How much higher levels influence lower levels
bottom_up_strength = 0.8            ### How much lower levels inform higher levels
temporal_coupling = 1.2             ### Strength of temporal dependencies across levels

## Equations
### Hierarchical Expected Free Energy
G_high = \sum_{\tau=1}^{T_{high}} \mathbb{E}_{Q(s_{high,\tau}|π_{high})} \left[ D_{KL}[Q(o_{high,\tau}|s_{high,\tau}) || P(o_{high,\tau}|s_{high,\tau})] - \mathbb{E}_{Q(o_{high,\tau}|s_{high,\tau})}[\ln P(o_{high,\tau})] \right]

G_mid = \sum_{\tau=1}^{T_{mid}} \mathbb{E}_{Q(s_{mid,\tau}|π_{mid},s_{high})} \left[ D_{KL}[Q(o_{mid,\tau}|s_{mid,\tau}) || P(o_{mid,\tau}|s_{mid,\tau})] - \mathbb{E}_{Q(o_{mid,\tau}|s_{mid,\tau})}[\ln P(o_{mid,\tau})] \right]

G_low = \sum_{\tau=1}^{T_{low}} \mathbb{E}_{Q(s_{low,\tau}|π_{low},s_{mid})} \left[ D_{KL}[Q(o_{low,\tau}|s_{low,\tau}) || P(o_{low,\tau}|s_{low,\tau})] - \mathbb{E}_{Q(o_{low,\tau}|s_{low,\tau})}[\ln P(o_{low,\tau})] \right]

### Cross-Level Coupling Terms
G_{td} = top\_down\_strength \cdot \sum_{\tau} D_{KL}[Q(s_{lower,\tau}|s_{higher,\tau}) || Q(s_{lower,\tau})]

G_{bu} = bottom\_up\_strength \cdot \sum_{\tau} D_{KL}[Q(s_{higher,\tau}|o_{lower,\tau}) || Q(s_{higher,\tau})]

### Total Hierarchical Free Energy
G_{total} = G_{high} + G_{mid} + G_{low} + G_{td} + G_{bu}

### Temporal Abstraction
T_{high} : T_{mid} : T_{low} = 1 : 5 : 25  \quad \text{(Different timescales)}

## Time
Dynamic
DiscreteTime = t
ModelTimeHorizon_High = 5
ModelTimeHorizon_Mid = 25  
ModelTimeHorizon_Low = 125

## ActInfOntologyAnnotation
### High-Level Mappings
s_high_f0 = AbstractGoalFactor
s_high_f1 = ContextLevelFactor
s_high_f2 = PlanningHorizonFactor
o_high_m0 = AbstractFeedbackModality
u_high_c0 = HighLevelDecisionFactor
π_high = AbstractPolicyFactor

### Mid-Level Mappings  
s_mid_f0 = ActionSequenceFactor
s_mid_f1 = SequenceProgressFactor
s_mid_f2 = SequenceTypeFactor
o_mid_m0 = ProgressFeedbackModality
u_mid_c0 = SequenceControlFactor
π_mid = SequencePolicyFactor

### Low-Level Mappings
s_low_f0 = ImmediateActionFactor
s_low_f1 = MotorStateFactor
s_low_f2 = SensoryFocusFactor
o_low_m0 = SensoriMotorModality
u_low_c0 = LowLevelActionFactor
π_low = ExecutionPolicyFactor

### Hierarchical Structure Mappings
G_total = HierarchicalExpectedFreeEnergy
top_down_strength = TopDownCouplingParameter
bottom_up_strength = BottomUpCouplingParameter
temporal_coupling = TemporalCouplingParameter

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

1. **Copy this template** and adapt for your hierarchical architecture

2. **Design hierarchy levels**:
   - Define abstraction levels (what, how, when)
   - Set temporal scales (slow to fast)
   - Specify information flow patterns

3. **Configure cross-level interactions**:
   - Set top-down influence strength
   - Configure bottom-up information flow
   - Balance autonomy vs integration

4. **Tune temporal dynamics**:
   - Align timescales with problem structure
   - Set precision parameters appropriately
   - Configure temporal coupling strength

5. **Test hierarchical behavior**:
   ```bash
   uv run python src/gnn/main.py --target-dir path/to/your/hierarchical_models
   ```

## Hierarchical Design Patterns

### Cognitive Architecture
- **High**: Goals, intentions, long-term planning
- **Mid**: Action sequences, subgoal management, skill coordination  
- **Low**: Motor control, immediate sensorimotor processing
- **Timescales**: Minutes → Seconds → Milliseconds

### Robot Control Systems
- **High**: Mission planning, task allocation, strategic decisions
- **Mid**: Path planning, behavior coordination, skill execution
- **Low**: Motor commands, sensor processing, reflex responses
- **Timescales**: Hours → Minutes → Real-time

### Economic/Market Models
- **High**: Market regulations, long-term trends, policy decisions
- **Mid**: Company strategies, resource allocation, investment decisions
- **Low**: Individual transactions, price adjustments, immediate responses
- **Timescales**: Years → Months → Days

### Biological Systems
- **High**: Life history strategies, behavioral ecology, adaptation
- **Mid**: Daily routines, foraging strategies, social behaviors  
- **Low**: Movement patterns, sensory processing, reflexes
- **Timescales**: Seasons → Days → Seconds

## Key Hierarchical Principles

### Temporal Abstraction
- Higher levels operate on longer timescales
- Lower levels provide high-frequency updates
- Cross-scale coordination through temporal coupling

### Functional Decomposition
- Each level handles appropriate abstraction
- Higher levels set context/goals for lower levels
- Lower levels provide detailed implementation

### Information Flow
- **Top-Down**: Goals, context, constraints, biases
- **Bottom-Up**: Feedback, state updates, error signals, opportunities
- **Lateral**: Coordination within levels, parallel processing

### Precision Hierarchy
- Higher levels: Lower precision (more exploration/flexibility)
- Lower levels: Higher precision (more deterministic/reliable)
- Adaptive precision based on context and performance

## Common Hierarchical Challenges

### Scale Separation
- Ensure timescales are sufficiently separated
- Avoid interference between fast and slow dynamics
- Solution: Careful temporal parameter tuning

### Information Bottlenecks
- Limited communication bandwidth between levels
- Critical information may be lost or delayed
- Solution: Selective information filtering, importance weighting

### Coordination Complexity
- Multiple levels making simultaneous decisions
- Potential for conflicts or suboptimal coordination
- Solution: Clear hierarchical precedence, coordination mechanisms

### Computational Scaling
- Hierarchical models can become computationally expensive
- Multiple levels of optimization and coordination
- Solution: Approximation methods, factorization, efficient inference

## Validation Strategies

### Level Isolation Testing
- Test each level independently first
- Verify individual level functionality
- Ensure proper matrix dimensions and constraints

### Cross-Level Integration
- Test two-level interactions before full hierarchy
- Verify information flow in both directions
- Check temporal coordination mechanisms

### Emergent Behavior Analysis
- Monitor for expected hierarchical behaviors
- Look for appropriate scale separation in dynamics
- Validate that higher levels actually influence lower levels

### Performance Scaling
- Test computational performance with increasing hierarchy depth
- Monitor memory usage and processing time
- Optimize bottlenecks in cross-level communication

## Validation Checklist

- [ ] Each hierarchical level functions correctly in isolation
- [ ] Cross-level dependencies are properly specified
- [ ] Temporal scales are appropriately separated
- [ ] Information flows correctly top-down and bottom-up
- [ ] Precision parameters reflect hierarchical structure
- [ ] Model validates without errors in type checker
- [ ] Emergent behaviors show hierarchical organization
- [ ] Computational performance is acceptable

## Related Templates

- [Basic GNN Template](basic_gnn_template.md) - For single-level models
- [POMDP Template](pomdp_template.md) - For uncertainty handling at each level
- [Multi-agent Template](multiagent_template.md) - For hierarchical multi-agent systems

## References

- [Hierarchical Active Inference](../gnn/advanced/advanced_modeling_patterns.md)
- [Temporal Abstraction in AI](../gnn/advanced/advanced_modeling_patterns.md#temporal-hierarchy)
- [Multi-Scale Modeling](../gnn/tutorials/gnn_examples_doc.md#hierarchical-examples)
- [Cognitive Architectures](../cognitive_phenomena/README.md)