# Basic GNN Model Template

Copy this template and modify it to create your own GNN models. Replace the bracketed fields with your specific model details.

## GNNSection
[YourGNNSection]

## GNNVersionAndFlags
GNN v1.0
ProcessingFlags: default

## ModelName
[YourModelName]

## ModelAnnotation
[Brief description of what your model does and its purpose.
Include key features, assumptions, and intended use cases.]

## StateSpaceBlock
### Hidden State Factors
s_f0[2,1,type=categorical]  ### [Description of state factor 0]
s_f1[3,1,type=categorical]  ### [Description of state factor 1]

### Observation Modalities  
o_m0[2,1,type=categorical]  ### [Description of observation modality 0]
o_m1[3,1,type=categorical]  ### [Description of observation modality 1]

### Control Factors (if applicable)
u_c0[2,1,type=categorical]  ### [Description of control factor 0]

### Policy Factors (if applicable)
π_c0[2,1,type=categorical]  ### [Description of policy factor 0]

## Connections
### Basic Dependencies
s_f0 > o_m0  ### [Describe this causal relationship]
s_f1 > o_m1  ### [Describe this causal relationship]

### Control Dependencies (if applicable)
u_c0 > s_f0  ### [Describe how control affects states]

### State Interactions (if applicable)
s_f0 - s_f1  ### [Describe state correlations]

## InitialParameterization
### Likelihood Matrices (A matrices)
A_m0 = [[0.8, 0.2], [0.3, 0.7]]  ### P(o_m0|s_f0)
A_m1 = [[0.9, 0.1, 0.0], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]]  ### P(o_m1|s_f1)

### Transition Matrices (B matrices, for dynamic models)
B_f0 = [[[0.9, 0.1], [0.2, 0.8]], [[0.1, 0.9], [0.8, 0.2]]]  ### P(s_f0'|s_f0,u_c0)

### Preference Vectors (C vectors)
C_m0 = [2.0, 0.0]  ### Log preferences over o_m0
C_m1 = [1.0, 0.5, 0.0]  ### Log preferences over o_m1

### Prior Beliefs (D vectors)
D_f0 = [0.5, 0.5]  ### Uniform prior over s_f0
D_f1 = [0.33, 0.33, 0.34]  ### Uniform prior over s_f1

### Precision Parameters (optional)
gamma = 1.0  ### Policy precision
alpha = 1.0  ### Action precision

## Equations
### Expected Free Energy (if applicable)
G = \sum_{\tau} \mathbb{E}_{Q(s_\tau|o_{1:\tau})} \left[ D_{KL}[Q(o_\tau|s_\tau) || P(o_\tau|s_\tau)] - \mathbb{E}_{Q(o_\tau|s_\tau)}[\ln P(o_\tau)] \right]

### Belief Update (if applicable)
Q(s_t|o_{1:t}) \propto P(o_t|s_t) \sum_{s_{t-1}} Q(s_{t-1}|o_{1:t-1}) P(s_t|s_{t-1}, u_{t-1})

## Time
Static  ### Change to "Dynamic" for temporal models
### For dynamic models, also specify:
### DiscreteTime
### ModelTimeHorizon: [number of time steps]

## ActInfOntologyAnnotation
### Map to Active Inference Ontology terms
hasStateSpace: [relevant ontology terms]
hasObservationSpace: [relevant ontology terms]
hasActionSpace: [relevant ontology terms]
implementsProcess: [relevant process terms]

## Footer
Created: [Date]
LastModified: [Date]
Version: 1.0

## Signature
ModelCreator: [Your Name]
Institution: [Your Institution]
Email: [Your Email]
License: [License type, e.g., MIT, CC-BY]

---

## Usage Instructions

1. **Copy this template** to a new file with a descriptive name (e.g., `my_robot_navigation_model.md`)

2. **Replace all bracketed fields** with your specific model details

3. **Validate your model** using the GNN type checker:
   ```bash
   uv run python src/gnn/main.py --target-dir path/to/your/models --only-steps 4
   ```

4. **Test your model** by running the full pipeline:
   ```bash
   uv run python src/gnn/main.py --target-dir path/to/your/models
   ```

## Common Modifications

### For Static Models
- Remove control factors (u_c0, π_c0) and their connections
- Remove B matrices (transition dynamics)
- Keep only A matrices, C vectors, and D vectors

### For Dynamic Models  
- Change `Time` section to `Dynamic`
- Add `DiscreteTime` and `ModelTimeHorizon`
- Include B matrices for state transitions
- Add control factors and policies if applicable

### For Multi-Modal Models
- Add more observation modalities (o_m1, o_m2, etc.)
- Include corresponding A matrices for each modality
- Consider cross-modal connections

### For Hierarchical Models
- Use state factors with different levels (s_f0 for low level, s_f1 for high level)
- Add connections between hierarchical levels
- Include appropriate precision parameters

## Validation Checklist

- [ ] All variable names follow GNN conventions (s_fX, o_mX, u_cX, π_cX)
- [ ] Matrix dimensions match variable dimensions
- [ ] All probability matrices are properly normalized (rows sum to 1)
- [ ] All required sections are present
- [ ] Connections reference existing variables
- [ ] Model validates without errors using type checker 
