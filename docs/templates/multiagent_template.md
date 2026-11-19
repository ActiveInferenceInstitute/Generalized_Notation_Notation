# Multi-Agent Active Inference Template

This template provides a framework for modeling multiple interacting Active Inference agents. It supports coordination, communication, and shared environments.

## GNNSection
[YourGNNSection]

## GNNVersionAndFlags
GNN v1.0
ProcessingFlags: active_inference,multiagent,coordination

## ModelName
[YourMultiAgentModelName]

## ModelAnnotation
[Describe your multi-agent system - what domain does it address? How do agents interact?
Example: A coordinated search and rescue scenario where multiple autonomous agents must explore an unknown environment, 
communicate discoveries, and collaborate to achieve shared objectives.
Key features: agent coordination, information sharing, emergent behaviors, scalability considerations.]

## StateSpaceBlock
### Agent 1 State Factors
s1_f0[4,1,type=categorical]      ### Agent 1 position: North=0, South=1, East=2, West=3  
s1_f1[3,1,type=categorical]      ### Agent 1 internal state: Searching=0, Found=1, Helping=2
s1_f2[2,1,type=categorical]      ### Agent 1 communication state: Idle=0, Transmitting=1

### Agent 2 State Factors  
s2_f0[4,1,type=categorical]      ### Agent 2 position: North=0, South=1, East=2, West=3
s2_f1[3,1,type=categorical]      ### Agent 2 internal state: Searching=0, Found=1, Helping=2
s2_f2[2,1,type=categorical]      ### Agent 2 communication state: Idle=0, Transmitting=1

### Shared Environment State
env_f0[9,1,type=categorical]     ### Grid locations: 3x3 grid positions
env_f1[3,1,type=categorical]     ### Environment status: Empty=0, Target=1, Obstacle=2
env_f2[2,1,type=categorical]     ### Discovery state: Unknown=0, Explored=1

### Agent 1 Observations
o1_m0[5,1,type=categorical]      ### Agent 1 local vision: Empty=0, Agent=1, Target=2, Obstacle=3, Boundary=4
o1_m1[3,1,type=categorical]      ### Agent 1 communication received: None=0, TargetFound=1, NeedHelp=2

### Agent 2 Observations
o2_m0[5,1,type=categorical]      ### Agent 2 local vision: Empty=0, Agent=1, Target=2, Obstacle=3, Boundary=4  
o2_m1[3,1,type=categorical]      ### Agent 2 communication received: None=0, TargetFound=1, NeedHelp=2

### Agent 1 Actions
u1_c0[4,1,type=categorical]      ### Agent 1 movement: Forward=0, Back=1, Left=2, Right=3
u1_c1[3,1,type=categorical]      ### Agent 1 communication: Silent=0, Broadcast_Found=1, Request_Help=2

### Agent 2 Actions
u2_c0[4,1,type=categorical]      ### Agent 2 movement: Forward=0, Back=1, Left=2, Right=3
u2_c1[3,1,type=categorical]      ### Agent 2 communication: Silent=0, Broadcast_Found=1, Request_Help=2

### Policy Factors
π1_c0[3,1,type=categorical]      ### Agent 1 movement policy: Explore=0, Direct=1, Cooperative=2
π1_c1[2,1,type=categorical]      ### Agent 1 communication policy: Minimal=0, Active=1
π2_c0[3,1,type=categorical]      ### Agent 2 movement policy: Explore=0, Direct=1, Cooperative=2  
π2_c1[2,1,type=categorical]      ### Agent 2 communication policy: Minimal=0, Active=1

## Connections
### Agent 1 Internal Dependencies
(s1_f0, s1_f1) > o1_m0          ### Agent 1 observations depend on position and internal state
(s1_f2, s2_f2) > o1_m1          ### Agent 1 receives communications from Agent 2

### Agent 2 Internal Dependencies  
(s2_f0, s2_f1) > o2_m0          ### Agent 2 observations depend on position and internal state
(s2_f2, s1_f2) > o2_m1          ### Agent 2 receives communications from Agent 1

### Environment Dependencies
(env_f0, env_f1) > o1_m0        ### Environment affects what Agent 1 sees
(env_f0, env_f1) > o2_m0        ### Environment affects what Agent 2 sees

### Transition Dependencies
(s1_f0, u1_c0, env_f1) > s1_f0  ### Agent 1 movement constrained by environment
(s1_f1, o1_m0, o1_m1) > s1_f1   ### Agent 1 internal state updated by observations
(s1_f2, u1_c1) > s1_f2          ### Agent 1 communication state

(s2_f0, u2_c0, env_f1) > s2_f0  ### Agent 2 movement constrained by environment
(s2_f1, o2_m0, o2_m1) > s2_f1   ### Agent 2 internal state updated by observations
(s2_f2, u2_c1) > s2_f2          ### Agent 2 communication state

### Cross-Agent Dependencies
(s1_f0, s2_f0) > env_f2         ### Joint exploration updates environment discovery
(s1_f1, s2_f1) > env_f1         ### Collaborative target identification

### Policy Dependencies
π1_c0 > u1_c0                   ### Agent 1 movement policy determines actions
π1_c1 > u1_c1                   ### Agent 1 communication policy determines actions
π2_c0 > u2_c0                   ### Agent 2 movement policy determines actions
π2_c1 > u2_c1                   ### Agent 2 communication policy determines actions

## InitialParameterization
### Agent 1 Likelihood Matrices
A1_m0 = [
    # Visual observations based on agent position, internal state, and environment
    [[[0.9, 0.1, 0.0, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0, 0.0]], # Position 0, different internal states
     [[0.8, 0.1, 0.1, 0.0, 0.0], [0.2, 0.7, 0.1, 0.0, 0.0]], # Position 1
     [[0.7, 0.2, 0.1, 0.0, 0.0], [0.1, 0.6, 0.2, 0.1, 0.0]], # Position 2  
     [[0.9, 0.1, 0.0, 0.0, 0.0], [0.3, 0.5, 0.2, 0.0, 0.0]]] # Position 3
]  ### P(o1_m0|s1_f0,s1_f1,env_f0,env_f1)

A1_m1 = [
    [0.9, 0.05, 0.05], [0.3, 0.6, 0.1], [0.1, 0.1, 0.8]      # Communication reception
]  ### P(o1_m1|s1_f2,s2_f2)

### Agent 2 Likelihood Matrices (similar structure)
A2_m0 = [
    [[[0.9, 0.1, 0.0, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0, 0.0]],
     [[0.8, 0.1, 0.1, 0.0, 0.0], [0.2, 0.7, 0.1, 0.0, 0.0]],
     [[0.7, 0.2, 0.1, 0.0, 0.0], [0.1, 0.6, 0.2, 0.1, 0.0]],
     [[0.9, 0.1, 0.0, 0.0, 0.0], [0.3, 0.5, 0.2, 0.0, 0.0]]]
]  ### P(o2_m0|s2_f0,s2_f1,env_f0,env_f1)

A2_m1 = [
    [0.9, 0.05, 0.05], [0.3, 0.6, 0.1], [0.1, 0.1, 0.8]
]  ### P(o2_m1|s2_f2,s1_f2)

### Agent 1 Transition Matrices
B1_f0 = [
    # Movement transitions (4 actions x 4 positions x 4 positions) 
    [[[0.8, 0.1, 0.05, 0.05], [0.1, 0.8, 0.05, 0.05], [0.05, 0.05, 0.8, 0.1], [0.05, 0.05, 0.1, 0.8]],  # Forward
     [[0.8, 0.1, 0.05, 0.05], [0.1, 0.8, 0.05, 0.05], [0.05, 0.05, 0.8, 0.1], [0.05, 0.05, 0.1, 0.8]],  # Back
     [[0.7, 0.2, 0.05, 0.05], [0.2, 0.7, 0.05, 0.05], [0.05, 0.05, 0.7, 0.2], [0.05, 0.05, 0.2, 0.7]],  # Left  
     [[0.7, 0.2, 0.05, 0.05], [0.2, 0.7, 0.05, 0.05], [0.05, 0.05, 0.7, 0.2], [0.05, 0.05, 0.2, 0.7]]]  # Right
]  ### P(s1_f0'|s1_f0,u1_c0,env_f1)

B1_f1 = [
    # Internal state transitions based on observations
    [[[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.0, 0.3, 0.7]],    # Different observation contexts
     [[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]],
     [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]]]
]  ### P(s1_f1'|s1_f1,o1_m0,o1_m1)

B1_f2 = [
    # Communication state transitions  
    [[[0.9, 0.1], [0.3, 0.7]],                                # Silent action
     [[0.2, 0.8], [0.1, 0.9]],                                # Broadcast action
     [[0.3, 0.7], [0.2, 0.8]]]                                # Request action
]  ### P(s1_f2'|s1_f2,u1_c1)

### Agent 2 Transition Matrices (symmetric to Agent 1)
B2_f0 = B1_f0  ### Same movement dynamics
B2_f1 = B1_f1  ### Same internal state dynamics  
B2_f2 = B1_f2  ### Same communication dynamics

### Environment Transition Matrices
B_env_f0 = [
    # Environment locations are generally static
    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],         # Location transitions (mostly identity)
     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
]  ### P(env_f0'|env_f0)

B_env_f1 = [
    # Environment status can change through agent actions
    [[[0.9, 0.1, 0.0], [0.8, 0.15, 0.05], [0.85, 0.1, 0.05]],  # Different transition probabilities
     [[0.1, 0.8, 0.1], [0.05, 0.9, 0.05], [0.1, 0.85, 0.05]],   # based on agent activities
     [[0.05, 0.05, 0.9], [0.0, 0.1, 0.9], [0.0, 0.0, 1.0]]]
]  ### P(env_f1'|env_f1,s1_f1,s2_f1)

B_env_f2 = [
    # Discovery state updated by joint exploration
    [[[0.8, 0.2], [0.3, 0.7]],                                 # Unknown -> Explored transitions
     [[0.1, 0.9], [0.0, 1.0]]]                                 # based on agent positions
]  ### P(env_f2'|env_f2,s1_f0,s2_f0)

### Preference Vectors
C1_m0 = [0.0, -1.0, 3.0, -2.0, 0.0]   ### Agent 1: Neutral, avoid agents, seek targets, avoid obstacles, neutral boundaries
C1_m1 = [0.0, 2.0, 1.0]               ### Agent 1: Neutral silence, prefer target found, slight preference for help

C2_m0 = [0.0, -1.0, 3.0, -2.0, 0.0]   ### Agent 2: Same visual preferences as Agent 1
C2_m1 = [0.0, 2.0, 1.0]               ### Agent 2: Same communication preferences

### Prior Beliefs
D1_f0 = [0.4, 0.3, 0.2, 0.1]          ### Agent 1 initial position beliefs
D1_f1 = [0.8, 0.1, 0.1]               ### Agent 1 starts in searching state
D1_f2 = [0.9, 0.1]                    ### Agent 1 starts communication idle

D2_f0 = [0.1, 0.2, 0.3, 0.4]          ### Agent 2 initial position beliefs (different from Agent 1)
D2_f1 = [0.8, 0.1, 0.1]               ### Agent 2 starts in searching state  
D2_f2 = [0.9, 0.1]                    ### Agent 2 starts communication idle

D_env_f0 = [0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111]  ### Uniform over grid
D_env_f1 = [0.7, 0.2, 0.1]            ### Environment mostly empty initially
D_env_f2 = [0.9, 0.1]                 ### Environment starts mostly unknown

### Coordination Parameters
coordination_strength = 1.5           ### How much agents consider others' actions
communication_noise = 0.1             ### Reliability of inter-agent communication  
shared_goal_weight = 2.0              ### Importance of collective objectives

### Precision Parameters
gamma1 = 2.0                          ### Agent 1 policy precision
gamma2 = 2.0                          ### Agent 2 policy precision
alpha1 = 1.0                          ### Agent 1 action precision
alpha2 = 1.0                          ### Agent 2 action precision
coordination_precision = 1.5          ### Precision for coordination behaviors

## Equations
### Individual Expected Free Energy
G1_π = \sum_{\tau=1}^{T} \mathbb{E}_{Q(s1_\tau|π1)} \left[ D_{KL}[Q(o1_\tau|s1_\tau) || P(o1_\tau|s1_\tau)] - \mathbb{E}_{Q(o1_\tau|s1_\tau)}[\ln P(o1_\tau)] \right]

G2_π = \sum_{\tau=1}^{T} \mathbb{E}_{Q(s2_\tau|π2)} \left[ D_{KL}[Q(o2_\tau|s2_\tau) || P(o2_\tau|s2_\tau)] - \mathbb{E}_{Q(o2_\tau|s2_\tau)}[\ln P(o2_\tau)] \right]

### Coordination Free Energy (joint term)
G_{coord} = \sum_{\tau=1}^{T} coordination\_strength \cdot D_{KL}[Q(s1_\tau,s2_\tau|π1,π2) || Q(s1_\tau|π1)Q(s2_\tau|π2)]

### Joint Policy Evaluation  
G_{joint} = G1_π + G2_π + G_{coord}

### Communication Channel Model
P(o1_{m1}|s2_{f2}) = (1-communication\_noise) \cdot P_{ideal}(o1_{m1}|s2_{f2}) + communication\_noise \cdot P_{uniform}(o1_{m1})

## Time
Dynamic
DiscreteTime = t
ModelTimeHorizon = 15

## ActInfOntologyAnnotation
### Agent 1 Mappings
s1_f0 = HiddenStateFactor0_Agent1Position
s1_f1 = HiddenStateFactor1_Agent1InternalState  
s1_f2 = HiddenStateFactor2_Agent1CommunicationState
o1_m0 = ObservationModality0_Agent1Vision
o1_m1 = ObservationModality1_Agent1Communication
u1_c0 = ControlFactor0_Agent1Movement
u1_c1 = ControlFactor1_Agent1Communication

### Agent 2 Mappings
s2_f0 = HiddenStateFactor0_Agent2Position
s2_f1 = HiddenStateFactor1_Agent2InternalState
s2_f2 = HiddenStateFactor2_Agent2CommunicationState  
o2_m0 = ObservationModality0_Agent2Vision
o2_m1 = ObservationModality1_Agent2Communication
u2_c0 = ControlFactor0_Agent2Movement
u2_c1 = ControlFactor1_Agent2Communication

### Environment Mappings
env_f0 = SharedEnvironmentFactor0_Locations
env_f1 = SharedEnvironmentFactor1_Status
env_f2 = SharedEnvironmentFactor2_Discovery

### Coordination Mappings
coordination_strength = CoordinationParameter
G_coord = CoordinationFreeEnergy
G_joint = JointExpectedFreeEnergy

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

1. **Copy this template** and customize for your multi-agent scenario

2. **Design agent roles**:
   - Define what each agent needs to accomplish
   - Specify individual vs shared objectives
   - Design communication protocols

3. **Configure interactions**:
   - Set up environment shared between agents
   - Define coordination mechanisms
   - Specify information sharing channels

4. **Balance autonomy and coordination**:
   - Adjust `coordination_strength` parameter
   - Configure communication reliability
   - Set individual vs joint preferences

5. **Test coordination**:
   ```bash
   uv run python src/gnn/main.py --target-dir path/to/your/multiagent_models
   ```

## Multi-Agent Design Patterns

### Cooperative Search
- **Agents**: Specialized searchers with different capabilities
- **Environment**: Large space with hidden targets
- **Coordination**: Information sharing, area division, joint planning
- **Challenges**: Exploration vs exploitation, communication overhead

### Competitive Markets
- **Agents**: Trading agents with different strategies
- **Environment**: Market with dynamic prices and resources
- **Coordination**: Strategic interactions, coalition formation
- **Challenges**: Nash equilibria, mechanism design, information asymmetry

### Swarm Robotics
- **Agents**: Simple robots with limited sensing/computation
- **Environment**: Physical space with obstacles and objectives
- **Coordination**: Local interaction rules, emergent behaviors
- **Challenges**: Scalability, robustness, distributed decision making

### Collaborative Planning
- **Agents**: Specialists with complementary expertise
- **Environment**: Complex problem space requiring diverse skills
- **Coordination**: Task allocation, knowledge integration, joint execution
- **Challenges**: Load balancing, skill matching, communication complexity

## Coordination Mechanisms

### Information Sharing
- **Broadcast**: All agents receive same information
- **Selective**: Targeted communication based on relevance
- **Hierarchical**: Multi-level information filtering
- **Emergent**: Self-organizing communication networks

### Task Allocation
- **Central Assignment**: Leader assigns tasks to followers
- **Market-Based**: Agents bid for tasks based on capabilities
- **Consensus**: Joint decision making through negotiation  
- **Role-Based**: Predefined specializations and responsibilities

### Conflict Resolution
- **Priority Systems**: Predetermined hierarchies for conflicts
- **Negotiation**: Real-time bargaining and compromise
- **Voting**: Democratic decision making mechanisms
- **Mediation**: Third-party conflict resolution

## Performance Considerations

### Scalability
- **Linear**: Agent complexity grows linearly with team size
- **Quadratic**: All-to-all communication creates quadratic complexity
- **Hierarchical**: Tree structures for better scaling
- **Distributed**: Local interactions only for massive scale

### Communication Efficiency  
- **Bandwidth**: Limit information exchange to essential data
- **Timing**: Synchronous vs asynchronous coordination
- **Protocols**: Standardized message formats and meanings
- **Compression**: Efficient encoding of shared information

### Computational Complexity
- **Centralized**: Single agent computes joint policies (exponential)
- **Factored**: Exploit independence for computational savings
- **Approximate**: Use heuristics and sampling for large teams
- **Distributed**: Each agent computes local policies only

## Validation Checklist

- [ ] Individual agent models work correctly in isolation
- [ ] Communication channels function bidirectionally  
- [ ] Coordination mechanisms produce sensible joint behaviors
- [ ] Shared environment state updates correctly
- [ ] Preference structures balance individual and collective goals
- [ ] Model validates without errors in type checker
- [ ] Emergent behaviors align with design objectives
- [ ] Performance scales acceptably with team size

## Common Multi-Agent Challenges

### Coordination Dilemmas
- Agents may prefer individual optimization over team performance
- Solution: Adjust preference weights, add coordination incentives

### Communication Overhead
- Too much communication can overwhelm agents
- Solution: Selective communication, importance filtering, hierarchical structures

### Emergent Behaviors
- Complex interactions may produce unexpected team behaviors
- Solution: Extensive simulation, gradual complexity increase, monitoring mechanisms

### Scalability Issues
- Coordination complexity often grows exponentially with team size
- Solution: Hierarchical organization, local interaction rules, approximation methods

## Related Templates

- [Basic GNN Template](basic_gnn_template.md) - For individual agent design
- [POMDP Template](pomdp_template.md) - For single-agent uncertainty handling  
- [Hierarchical Template](hierarchical_template.md) - For multi-level agent architectures

## References

- [Multi-Agent Active Inference](../gnn/advanced/gnn_multiagent.md)
- [Coordination Mechanisms](../gnn/integration/gnn_implementation.md#multi-agent-coordination)
- [Distributed Active Inference](../other/federated_inference_friston2023.md)
- [Communication Protocols](../gnn/reference/gnn_syntax.md#multi-agent-extensions) 