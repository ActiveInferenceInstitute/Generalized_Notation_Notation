# Memory in Active Inference

## Overview

Memory in Active Inference emerges from the brain's need to maintain and update probabilistic beliefs about the world across time. Rather than being a passive storage system, memory represents the brain's dynamic model of temporal dependencies and learned associations that minimize future prediction error.

## Core Mechanisms

### Predictive Memory Models
Memory systems maintain generative models that predict future observations based on past experience. These models are continuously updated through prediction error minimization.

### Hierarchical Temporal Representation
Memory operates across multiple timescales - from millisecond sensory buffering to lifelong episodic memories - each optimized for different temporal prediction horizons.

### Precision-Weighted Encoding
Memory formation depends on the precision (confidence) assigned to experiences. High-precision events are more likely to be encoded and retained.

## Memory Systems

### Working Memory
**Function**: Active maintenance of information for immediate cognitive operations
**Mechanism**: Recurrent precision dynamics maintaining high-precision representations
**Capacity**: Limited by metabolic costs of maintaining high precision
**Duration**: Seconds to minutes

### Short-term Memory
**Function**: Bridge between immediate perception and long-term storage
**Mechanism**: Temporary increase in synaptic strength and local recurrent activity
**Capacity**: Miller's 7±2 items, but varies with chunking and attention
**Duration**: Minutes to hours

### Long-term Memory

#### Episodic Memory
**Function**: Memory for specific personal experiences and their temporal context
**Mechanism**: Hippocampal binding of distributed neocortical representations
**Characteristics**: Rich contextual detail, conscious recollection, vulnerable to interference

#### Semantic Memory
**Function**: General knowledge about the world, concepts, and facts
**Mechanism**: Distributed neocortical representations strengthened through repetition
**Characteristics**: Context-independent, gradually acquired, relatively stable

#### Procedural Memory
**Function**: Motor skills and cognitive procedures
**Mechanism**: Basal ganglia and cerebellar learning circuits
**Characteristics**: Implicit, gradual acquisition, highly practiced automaticity

## Key Phenomena

### Encoding Processes
- **Attention and Encoding**: Attended information receives higher precision encoding
- **Levels of Processing**: Deeper semantic processing enhances memory formation
- **Elaborative Encoding**: Rich associations improve memory strength
- **Spacing Effect**: Distributed practice enhances long-term retention

### Storage and Consolidation
- **Memory Consolidation**: Gradual strengthening and stabilization of memory traces
- **Systems Consolidation**: Transfer from hippocampus to neocortex
- **Reconsolidation**: Memory updating during retrieval
- **Memory Interference**: Competition between similar memories

### Retrieval Processes
- **Cue-Dependent Retrieval**: Context and cues guide memory access
- **Recognition vs. Recall**: Different retrieval mechanisms and thresholds
- **Retrieval Practice**: Testing enhances memory strength
- **False Memories**: Constructive nature of memory retrieval

### Forgetting and Interference
- **Decay Theory**: Passive loss of memory strength over time
- **Interference Theory**: Competition from similar memories
- **Retrieval-Induced Forgetting**: Active suppression of competing memories
- **Motivated Forgetting**: Goal-directed memory inhibition

## Computational Models

### Memory Encoding
```
Encoding_Strength = Attention_Weight × Prediction_Error × Novelty × Emotional_Salience
Memory_Formation = f(Encoding_Strength, Existing_Associations, Consolidation_Time)
```

### Memory Retrieval
```
Retrieval_Probability = f(Memory_Strength, Cue_Match, Interference, Retrieval_Context)
Retrieved_Content = Reconstructive_Process(Memory_Trace, Current_Context, Expectations)
```

### Memory Updates
```
Updated_Memory = (1 - Learning_Rate) × Old_Memory + Learning_Rate × New_Information
Prediction_Error = New_Information - Predicted_Information
Learning_Rate = f(Prediction_Error, Uncertainty, Attention)
```

## Clinical Applications

### Memory Disorders

#### Alzheimer's Disease
- **Pathology**: Amyloid plaques and tau tangles affecting hippocampus and cortex
- **Symptoms**: Progressive episodic memory loss, semantic memory decline
- **Model**: Reduced precision in memory encoding and retrieval networks

#### Amnesia
- **Anterograde**: Inability to form new memories (hippocampal damage)
- **Retrograde**: Loss of old memories (temporal gradient)
- **Model**: Disrupted binding mechanisms and consolidation processes

#### Depression and Memory
- **Mood-Congruent Memory**: Enhanced recall for negative information
- **Rumination**: Repetitive retrieval of negative memories
- **Model**: Biased precision allocation favoring negative content

#### PTSD and Trauma
- **Intrusive Memories**: Involuntary, vivid recollections
- **Avoidance**: Motivated forgetting of trauma-related content
- **Model**: Dysregulated precision and failed memory integration

### Assessment and Intervention
- **Neuropsychological Testing**: Standardized memory assessments
- **Memory Training**: Techniques to improve encoding and retrieval
- **Cognitive Rehabilitation**: Compensatory strategies for memory impairment
- **Pharmacological Interventions**: Cholinesterase inhibitors, cognitive enhancers

## Research Applications

### Experimental Paradigms
- **Free Recall**: Retrieval without external cues
- **Recognition Memory**: Identification of previously encountered items
- **Working Memory Tasks**: N-back, change detection, digit span
- **Paired Associate Learning**: Formation of arbitrary associations

### Neural Correlates
- **Hippocampus**: Pattern completion and separation, binding
- **Prefrontal Cortex**: Working memory, strategic retrieval
- **Medial Temporal Lobe**: Episodic memory formation
- **Default Mode Network**: Memory consolidation and retrieval

### Advanced Techniques
- **fMRI**: Spatial patterns of memory representation
- **EEG**: Temporal dynamics of memory processes
- **Intracranial Recording**: Single-cell memory mechanisms
- **Computational Modeling**: Mechanistic theories of memory

## Future Directions

1. **Predictive Memory**: How memory serves future prediction rather than past storage
2. **Social Memory**: Memory for social interactions and relationships
3. **Embodied Memory**: Integration of memory with motor and sensory systems
4. **Memory and Consciousness**: Relationship between memory and conscious experience
5. **Artificial Memory**: Implementing memory mechanisms in AI systems
6. **Memory Enhancement**: Technologies for augmenting human memory

## References

### Foundational Works
- Atkinson, R. C., & Shiffrin, R. M. (1968). Human memory: A proposed system
- Baddeley, A. (1992). Working memory
- Tulving, E. (1972). Episodic and semantic memory

### Active Inference Applications
- Friston, K. J., & Buzsáki, G. (2016). The functional anatomy of time
- Pezzulo, G., et al. (2014). Active inference, homeostatic regulation and adaptive behavioural control

### Contemporary Research
- Squire, L. R., & Kandel, E. R. (2009). Memory: From mind to molecules
- Buckner, R. L., & Carroll, D. C. (2007). Self-projection and the brain 


## Documents in this folder

- [Working Memory Model](working_memory_model.md)
