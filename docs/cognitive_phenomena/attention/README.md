# Attention in Active Inference

## Overview

Attention in Active Inference emerges from precision optimization - the allocation of computational resources to minimize prediction error where it matters most. Rather than being a separate cognitive mechanism, attention represents the brain's strategy for efficient information processing through precision-weighted prediction error minimization.

## Core Mechanisms

### Precision-Weighted Prediction Error
Attention modulates the precision (inverse variance) of prediction errors, amplifying signals that are relevant for current goals while suppressing irrelevant information.

### Hierarchical Attention Control
Attention operates across multiple hierarchical levels, from low-level sensory selection to high-level executive control of cognitive processes.

### Active Sampling
Attention guides active sampling of the environment through eye movements, head turns, and other exploratory behaviors to reduce uncertainty about relevant aspects of the world.

## Key Phenomena Modeled

### Selective Attention
- **Cocktail Party Effect**: Focusing on one conversation in a noisy environment
- **Visual Search**: Finding targets among distractors
- **Spatial Attention**: Attending to specific locations in visual space
- **Feature-Based Attention**: Attending to specific features (color, motion, etc.)

### Attention Capture and Control
- **Bottom-Up Capture**: Automatic attention to salient stimuli
- **Top-Down Control**: Voluntary direction of attention based on goals
- **Inhibition of Return**: Reduced likelihood of returning attention to previously attended locations
- **Attentional Blink**: Temporary impairment in detecting targets following another target

### Divided Attention and Multitasking
- **Dual-Task Performance**: Managing multiple concurrent tasks
- **Attention Switching**: Costs associated with changing attentional focus
- **Resource Allocation**: Distribution of limited attentional resources

## Computational Models

### Core Attention Parameters
- **Precision**: Determines the strength of attentional modulation
- **Scope**: Controls the breadth vs. focus of attention
- **Persistence**: How long attention remains allocated to a location/feature
- **Switching Cost**: Metabolic cost of changing attentional focus

### Attention-Precision Relationship
```
Effective_Precision = Base_Precision × Attention_Weight
Attention_Weight = f(goal_relevance, salience, competition)
```

### Hierarchical Control
Top-down attention emerges from higher-level goals and expectations, while bottom-up attention responds to prediction errors and stimulus salience.

## Clinical Applications

### Attention Disorders
- **ADHD**: Reduced sustained attention and increased distractibility
- **Autism**: Atypical attentional patterns and sensory processing
- **Anxiety**: Hypervigilance and threat-biased attention
- **Depression**: Reduced attentional control and rumination

### Assessment and Intervention
- **Attention Training**: Protocols for improving attentional control
- **Neurofeedback**: Real-time training of attention-related brain states
- **Pharmacological Interventions**: Modulating neurotransmitter systems that affect attention

## Research Applications

### Experimental Paradigms
- **Posner Cueing Task**: Spatial attention and validity effects
- **Flanker Task**: Selective attention and conflict resolution
- **Attention Network Test**: Measuring efficiency of attention networks
- **Change Blindness**: Failures of attention and awareness

### Neural Correlates
- **Frontal-Parietal Networks**: Top-down attentional control
- **Temporal-Parietal Junction**: Bottom-up attention and reorienting
- **Superior Colliculus**: Oculomotor attention and spatial selection
- **Pulvinar**: Attentional modulation and coordination

## Future Directions

1. **Predictive Attention**: How attention anticipates future events
2. **Social Attention**: Joint attention and theory of mind
3. **Embodied Attention**: Integration with motor systems and active sampling
4. **Computational Psychiatry**: Precision-based models of attention disorders
5. **Artificial Attention**: Implementing attention mechanisms in AI systems

## 🏗️ Precision as Attention in GNN

In GNN models, attention is not a separate block but is implemented via **Precision Modulators**:
1. **Likelihood Precision**: Use the `@precision` flag on an `A_m0` matrix to dynamically scale the influence of sensory input.
2. **Prior Precision**: Adjust the `D_f0` concentration to model expected uncertainty.
3. **Policy Precision (Gamma)**: Scale the expected free energy ($\gamma$) to control the exploration-exploitation trade-off, effectively "attending" to future goals.

## References

### Core Papers
- Posner, M. I., & Petersen, S. E. (1990). The attention system of the human brain
- Corbetta, M., & Shulman, G. L. (2002). Control of goal-directed and stimulus-driven attention
- Reynolds, J. H., & Heeger, D. J. (2009). The normalization model of attention

### Active Inference Applications
- Feldman, H., & Friston, K. J. (2010). Attention, uncertainty, and free-energy
- Parr, T., & Friston, K. J. (2017). Uncertainty, epistemics and active inference 


## Documents in this folder

- [Spatial Attention Model](attention_model.md)
