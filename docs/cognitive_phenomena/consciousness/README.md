# Consciousness in Active Inference

## Overview

Consciousness emerges from hierarchical predictive processing, attention, and self-modeling. In Active Inference, consciousness represents the unified, coherent model of the self and world that arises from integrated belief states across multiple hierarchical levels.

## Core Components

### 1. Global Workspace and Conscious Access

```gnn
## ModelName
GlobalWorkspaceModel

## ModelAnnotation
Models consciousness through global accessibility of information and workspace dynamics.
Conscious access emerges when local processors broadcast to a global workspace.

## GNNVersionAndFlags
GNN_v1.0
ProcessingFlags: global_workspace, conscious_access, attention_integration

## StateSpaceBlock
# Global workspace level
s_global_f0[20,1,type=categorical]       ### Global workspace contents: various information types
s_global_f1[4,1,type=categorical]        ### Workspace state: Empty=0, Competing=1, Coherent=2, Switching=3
s_global_f2[3,1,type=categorical]        ### Access level: Unconscious=0, Preconscious=1, Conscious=2

# Local processors
s_visual_f0[10,1,type=categorical]       ### Visual processing: various visual features
s_auditory_f0[8,1,type=categorical]      ### Auditory processing: various auditory features
s_semantic_f0[15,1,type=categorical]     ### Semantic processing: conceptual content

# Attention and control
s_attn_f0[5,1,type=categorical]          ### Attention focus: Visual=0, Auditory=1, Semantic=2, Motor=3, Internal=4
s_attn_f1[4,1,type=categorical]          ### Attention strength: Weak=0, Moderate=1, Strong=2, Focused=3

## Observations
o_m0[10,1,type=categorical]              ### Visual input: matches s_visual_f0
o_m1[8,1,type=categorical]               ### Auditory input: matches s_auditory_f0

## Actions
u_c0[5,1,type=categorical]               ### Attention deployment: matches s_attn_f0
u_c1[4,1,type=categorical]               ### Workspace access: Broadcast=0, Inhibit=1, Integrate=2, Switch=3

## Connections
# Local to global broadcasting
s_visual_f0 > s_global_f0                ### Visual processors broadcast to workspace
s_auditory_f0 > s_global_f0              ### Auditory processors broadcast to workspace
s_semantic_f0 > s_global_f0              ### Semantic processors broadcast to workspace

# Attention-mediated access
s_attn_f0, s_attn_f1 > s_global_f2       ### Attention determines conscious access

## InitialParameterization
# Broadcasting parameters
broadcast_threshold = 0.7                ### Threshold for global broadcasting
workspace_capacity = 3                   ### Limited workspace capacity
conscious_access_threshold = 0.8         ### Threshold for conscious access

# Attention parameters
attention_precision = 2.0                ### High precision for attended information
attention_capacity = 4                   ### Limited attention capacity

## Time
Dynamic
DiscreteTime = t
ModelTimeHorizon = 100

## Footer
This model captures consciousness through global workspace dynamics and attention.
```

### 2. Self-Model and Metacognition

```gnn
## ModelName
SelfModelConsciousness

## ModelAnnotation
Models consciousness through hierarchical self-modeling and metacognitive awareness.
The self-model provides the sense of being a unified conscious agent.

## StateSpaceBlock
# Self-model hierarchy
s_self_f0[8,1,type=categorical]          ### Basic self-properties: Embodied=0, Temporal=1, Agential=2, etc.
s_self_f1[6,1,type=categorical]          ### Self-awareness: Body=0, Mind=1, Emotions=2, Thoughts=3, Intentions=4, Beliefs=5
s_self_f2[4,1,type=categorical]          ### Self-reflection: None=0, Monitoring=1, Evaluating=2, Regulating=3

# Metacognitive states
s_meta_f0[5,1,type=categorical]          ### Metacognitive type: Knowing=0, Feeling=1, Judging=2, Controlling=3, Experiencing=4
s_meta_f1[4,1,type=categorical]          ### Confidence: Low=0, Medium=1, High=2, Certain=3
s_meta_f2[3,1,type=categorical]          ### Monitoring: Absent=0, Present=1, Active=2

# Narrative self
s_narr_f0[10,1,type=categorical]         ### Life story elements: Past=0, Present=1, Future=2, Goals=3, etc.
s_narr_f1[4,1,type=categorical]          ### Narrative coherence: Fragmented=0, Partial=1, Coherent=2, Integrated=3

## Observations
o_m0[8,1,type=categorical]               ### Interoceptive signals: Body awareness
o_m1[6,1,type=categorical]               ### Mental state observations: Current thoughts/feelings
o_m2[4,1,type=categorical]               ### Memory cues: Autobiographical memory triggers

## Actions
u_c0[4,1,type=categorical]               ### Self-reflection: Introspect=0, Evaluate=1, Plan=2, Remember=3
u_c1[3,1,type=categorical]               ### Metacognitive control: Monitor=0, Regulate=1, Redirect=2

## InitialParameterization
# Self-model parameters
self_coherence = 0.8                     ### Coherence of self-model
self_update_rate = 0.1                   ### Rate of self-model updating
embodiment_strength = 1.2                ### Strength of embodied self-sense

# Metacognitive parameters
metacognitive_sensitivity = 0.7          ### Sensitivity to metacognitive signals
confidence_calibration = 0.6             ### Accuracy of confidence judgments
introspective_accuracy = 0.5             ### Accuracy of introspection

## Equations
# Self-model coherence
self_coherence(t) = consistency(beliefs_about_self(t))

# Metacognitive confidence
confidence(t) = f(processing_fluency(t), memory_strength(t), social_feedback(t))
```

### 3. Phenomenal Consciousness and Qualia

```gnn
## ModelName
PhenomenalConsciousnessModel

## ModelAnnotation
Models phenomenal aspects of consciousness through precision-weighted prediction and sensory integration.
Qualia emerge from high-precision prediction errors in sensory hierarchies.

## StateSpaceBlock
# Sensory qualities (qualia)
s_visual_qual[12,1,type=continuous]      ### Visual qualia: color, brightness, texture, etc.
s_auditory_qual[8,1,type=continuous]     ### Auditory qualia: pitch, timbre, loudness, etc.
s_tactile_qual[6,1,type=continuous]      ### Tactile qualia: texture, temperature, pressure
s_emotional_qual[5,1,type=continuous]    ### Emotional qualia: valence, arousal, dominance

# Binding and integration
s_binding_f0[4,1,type=categorical]       ### Feature binding: Unbound=0, Partial=1, Bound=2, Integrated=3
s_unity_f0[3,1,type=categorical]         ### Unity of consciousness: Fragmented=0, Unified=1, Integrated=2
s_temporal_f0[4,1,type=categorical]      ### Temporal consciousness: Past=0, Present=1, Future=2, Timeless=3

# Awareness levels
s_aware_f0[5,1,type=categorical]         ### Awareness level: Unconscious=0, Subliminal=1, Preconscious=2, Conscious=3, Reflective=4
s_clarity_f0[4,1,type=categorical]       ### Clarity: Vague=0, Unclear=1, Clear=2, Vivid=3

## Observations
o_m0[12,1,type=continuous]               ### Visual sensory input
o_m1[8,1,type=continuous]                ### Auditory sensory input
o_m2[6,1,type=continuous]                ### Tactile sensory input

## InitialParameterization
# Phenomenal parameters
qualitative_precision = 2.5              ### High precision for conscious sensory states
binding_strength = 1.5                   ### Strength of feature binding
unity_threshold = 0.8                    ### Threshold for unified consciousness

# Temporal parameters
present_moment_window = 0.5               ### Duration of experienced present
temporal_integration = 0.7                ### Integration across time

## Equations
# Qualia intensity
qualia_intensity(t) = precision(t) * prediction_error(t) * attention_weight(t)

# Unity of consciousness
unity(t) = coherence(bound_features(t)) * integration_strength(t)
```

## Clinical Applications

### Disorders of Consciousness

```gnn
## ModelName
DisordersConsciousnessModel

## ModelAnnotation
Models various disorders of consciousness through altered connectivity and reduced integration.

## ModifiedParameters
# Vegetative state - minimal cortical integration
global_integration = 0.1                 ### Severely reduced from normal 0.8
workspace_capacity = 1                   ### Reduced from normal 3
conscious_access_threshold = 0.95        ### Increased from normal 0.8

# Minimally conscious state - partial integration
global_integration = 0.3                 ### Reduced from normal 0.8
workspace_capacity = 2                   ### Reduced from normal 3
conscious_access_threshold = 0.9         ### Increased from normal 0.8

# Locked-in syndrome - preserved consciousness, impaired motor output
global_integration = 0.8                 ### Normal
workspace_capacity = 3                   ### Normal
motor_output_precision = 0.1             ### Severely reduced motor precision
```

### Schizophrenia and Reality Monitoring

```gnn
## ModelName
SchizophreniaConsciousnessModel

## ModelAnnotation
Models schizophrenia through altered self-other boundaries and impaired reality monitoring.

## ModifiedParameters
# Altered precision allocation
sensory_precision = 0.6                  ### Reduced from normal 1.0
self_model_precision = 0.4               ### Reduced from normal 0.8
reality_monitoring_precision = 0.3       ### Severely reduced

# Boundary disturbances
self_other_boundary = 0.4                ### Reduced from normal 0.8
internal_external_boundary = 0.3         ### Reduced from normal 0.8
thought_insertion_susceptibility = 1.8   ### Increased susceptibility

# Metacognitive deficits
metacognitive_sensitivity = 0.3          ### Reduced from normal 0.7
confidence_calibration = 0.2             ### Poor calibration
```

### Dissociative Disorders

```gnn
## ModelName
DissociativeConsciousnessModel

## ModelAnnotation
Models dissociation through fragmented self-models and reduced integration.

## ModifiedParameters
# Self-model fragmentation
self_coherence = 0.3                     ### Severely reduced from normal 0.8
identity_integration = 0.2               ### Severely reduced
memory_integration = 0.4                 ### Reduced integration

# Altered consciousness
present_moment_awareness = 0.5           ### Reduced awareness
embodiment_strength = 0.4                ### Reduced embodiment
emotional_integration = 0.3              ### Poor emotional integration
```

## Altered States of Consciousness

### Meditation and Contemplative States

```gnn
## ModelName
MeditativeConsciousnessModel

## ModelAnnotation
Models meditative states through altered attention, self-model, and awareness.

## ModifiedParameters
# Enhanced attention
attention_precision = 3.0                ### Increased from normal 2.0
attention_stability = 1.5                ### Increased stability
meta_attention = 1.8                     ### Enhanced metacognitive monitoring

# Altered self-model
ego_dissolution = variable               ### Variable ego dissolution
present_moment_focus = 2.0               ### Enhanced present-moment awareness
conceptual_elaboration = 0.3             ### Reduced conceptual thinking

# Enhanced awareness
introspective_accuracy = 1.2             ### Enhanced introspective accuracy
emotional_clarity = 1.5                  ### Enhanced emotional awareness
```

### Psychedelic States

```gnn
## ModelName
PsychedelicConsciousnessModel

## ModelAnnotation
Models psychedelic states through altered precision, reduced self-model, and enhanced connectivity.

## ModifiedParameters
# Altered precision hierarchy
top_down_precision = 0.4                 ### Reduced top-down control
bottom_up_precision = 1.8                ### Enhanced bottom-up processing
sensory_precision = 2.5                  ### Enhanced sensory processing

# Self-model dissolution
ego_boundaries = 0.2                     ### Dissolved ego boundaries
self_other_distinction = 0.3             ### Reduced self-other distinction
narrative_coherence = 0.2                ### Reduced narrative coherence

# Enhanced connectivity
cross_modal_integration = 2.0            ### Enhanced cross-modal processing
temporal_binding = 0.4                   ### Altered temporal binding
novel_connections = 2.5                  ### Enhanced novel associations
```

## Computational Implementations

### Python Implementation

```python
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ConsciousnessLevel(Enum):
    UNCONSCIOUS = 0
    PRECONSCIOUS = 1
    CONSCIOUS = 2
    REFLECTIVE = 3


class AttentionType(Enum):
    VISUAL = 0
    AUDITORY = 1
    SEMANTIC = 2
    MOTOR = 3
    INTERNAL = 4


@dataclass
class ConsciousnessState:
    """Comprehensive consciousness state representation"""

    global_workspace: np.ndarray
    local_processors: Dict[str, np.ndarray]
    attention_focus: AttentionType
    attention_strength: float
    consciousness_level: ConsciousnessLevel
    self_model: np.ndarray
    metacognitive_state: np.ndarray
    phenomenal_qualities: np.ndarray


class GlobalWorkspaceModel:
    """
    Active Inference implementation of consciousness through global workspace theory
    """

    def __init__(self, config: Dict):
        self.config = config
        self.initialize_parameters()

    def initialize_parameters(self):
        """Initialize consciousness-specific parameters"""
        # Global workspace parameters
        self.broadcast_threshold = self.config.get("broadcast_threshold", 0.7)
        self.workspace_capacity = self.config.get("workspace_capacity", 3)
        self.integration_time = self.config.get("integration_time", 0.3)

        # Attention parameters
        self.attention_precision = self.config.get("attention_precision", 2.0)
        self.attention_capacity = self.config.get("attention_capacity", 4)

        # Consciousness parameters
        self.conscious_access_threshold = self.config.get(
            "conscious_access_threshold", 0.8
        )
        self.conscious_duration = self.config.get("conscious_duration", 0.5)

        # Local processors
        self.visual_processor = self.initialize_visual_processor()
        self.auditory_processor = self.initialize_auditory_processor()
        self.semantic_processor = self.initialize_semantic_processor()

    def update_consciousness(
        self, sensory_input: Dict[str, np.ndarray], current_state: ConsciousnessState
    ) -> ConsciousnessState:
        """Update consciousness through global workspace dynamics"""

        # Update local processors
        local_states = self.update_local_processors(sensory_input, current_state)

        # Determine broadcasting candidates
        broadcast_candidates = self.identify_broadcast_candidates(local_states)

        # Global workspace competition and integration
        workspace_contents = self.update_global_workspace(
            broadcast_candidates, current_state
        )

        # Determine conscious access
        consciousness_level = self.determine_consciousness_level(workspace_contents)

        # Update attention based on workspace contents
        new_attention = self.update_attention(workspace_contents, current_state)

        # Update self-model and metacognition
        self_model = self.update_self_model(workspace_contents, current_state)
        metacognitive_state = self.update_metacognition(
            workspace_contents, current_state
        )

        # Compute phenomenal qualities
        phenomenal_qualities = self.compute_phenomenal_qualities(
            workspace_contents, new_attention
        )

        return ConsciousnessState(
            global_workspace=workspace_contents,
            local_processors=local_states,
            attention_focus=new_attention["focus"],
            attention_strength=new_attention["strength"],
            consciousness_level=consciousness_level,
            self_model=self_model,
            metacognitive_state=metacognitive_state,
            phenomenal_qualities=phenomenal_qualities,
        )

    def identify_broadcast_candidates(
        self, local_states: Dict[str, np.ndarray]
    ) -> List[Tuple[str, np.ndarray]]:
        """Identify which local processors should broadcast to global workspace"""
        candidates = []

        for processor_name, state in local_states.items():
            # Check if state exceeds broadcast threshold
            activation_strength = np.max(state)
            if activation_strength > self.broadcast_threshold:
                candidates.append((processor_name, state))

        # Sort by activation strength
        candidates.sort(key=lambda x: np.max(x[1]), reverse=True)

        return candidates

    def update_global_workspace(
        self,
        candidates: List[Tuple[str, np.ndarray]],
        current_state: ConsciousnessState,
    ) -> np.ndarray:
        """Update global workspace through competition and integration"""

        # Initialize workspace
        workspace = np.zeros(20)  # Assuming 20-dimensional workspace

        # Competition among candidates
        if len(candidates) > self.workspace_capacity:
            # Select top candidates based on strength and coherence
            selected_candidates = self.select_workspace_contents(
                candidates, current_state
            )
        else:
            selected_candidates = candidates

        # Integrate selected contents
        for i, (processor, state) in enumerate(selected_candidates):
            if i < self.workspace_capacity:
                # Map processor state to workspace representation
                workspace_representation = self.map_to_workspace(processor, state)
                workspace += workspace_representation

        # Apply temporal dynamics
        workspace = self.apply_workspace_dynamics(
            workspace, current_state.global_workspace
        )

        return workspace

    def determine_consciousness_level(
        self, workspace_contents: np.ndarray
    ) -> ConsciousnessLevel:
        """Determine level of consciousness based on workspace state"""

        workspace_activation = np.linalg.norm(workspace_contents)
        workspace_coherence = self.compute_workspace_coherence(workspace_contents)

        # Combined measure of consciousness
        consciousness_strength = workspace_activation * workspace_coherence

        if consciousness_strength > self.conscious_access_threshold:
            if workspace_coherence > 0.8:
                return ConsciousnessLevel.REFLECTIVE
            else:
                return ConsciousnessLevel.CONSCIOUS
        elif consciousness_strength > 0.4:
            return ConsciousnessLevel.PRECONSCIOUS
        else:
            return ConsciousnessLevel.UNCONSCIOUS

    def compute_phenomenal_qualities(
        self, workspace_contents: np.ndarray, attention_state: Dict
    ) -> np.ndarray:
        """Compute phenomenal qualities (qualia) from workspace and attention"""

        # Qualia emerge from precision-weighted prediction errors
        attention_weight = attention_state["strength"]
        workspace_strength = np.linalg.norm(workspace_contents)

        # Different modalities contribute to different qualia
        visual_qualia = self.compute_visual_qualia(workspace_contents, attention_weight)
        auditory_qualia = self.compute_auditory_qualia(
            workspace_contents, attention_weight
        )
        emotional_qualia = self.compute_emotional_qualia(
            workspace_contents, attention_weight
        )

        # Combine all qualitative dimensions
        phenomenal_qualities = np.concatenate(
            [visual_qualia, auditory_qualia, emotional_qualia]
        )

        return phenomenal_qualities
```

## Research Applications

### Consciousness Studies
- Measuring consciousness in unresponsive patients
- Understanding altered states of consciousness
- Investigating the neural correlates of consciousness

### Artificial Intelligence
- Building conscious AI systems
- Implementing global workspace architectures
- Developing metacognitive AI

### Clinical Applications
- Diagnosis of disorders of consciousness
- Monitoring consciousness during anesthesia
- Treatment of dissociative disorders

## Future Directions

1. **Integrated Information Theory**: Incorporating IIT measures of consciousness
2. **Predictive Processing Consciousness**: Advanced predictive models of consciousness
3. **Social Consciousness**: Modeling shared and collective consciousness
4. **Artificial Consciousness**: Building truly conscious artificial systems
5. **Therapeutic Applications**: Consciousness-based therapeutic interventions

## References

### Core Papers
- Dehaene, S. (2014). Consciousness and the brain: Deciphering how the brain codes our thoughts
- Seth, A. K. (2021). Being you: A new science of consciousness
- Tononi, G. (2008). Integrated information theory

### Active Inference Applications
- Friston, K. J. (2018). Am I self-conscious? (Or does self-organization entail self-consciousness?)
- Hohwy, J. (2013). The predictive mind and the puzzle of consciousness 


## Documents in this folder

- [Global Workspace Consciousness Model](global_workspace_model.md)
