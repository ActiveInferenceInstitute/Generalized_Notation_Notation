# Learning and Adaptation in Active Inference

## Overview

Learning and adaptation emerge from prediction error minimization and precision optimization across multiple timescales. In Active Inference, learning represents the continuous updating of generative models to better predict and control environmental contingencies.

## Core Components

### 1. Hierarchical Learning Architecture

```gnn
## ModelName
HierarchicalLearningModel

## ModelAnnotation
Models learning through hierarchical belief updating and precision optimization.
Learning occurs at multiple levels from fast perceptual adaptation to slow conceptual change.

## GNNVersionAndFlags
GNN_v1.0
ProcessingFlags: hierarchical_learning, precision_optimization, adaptive_plasticity

## StateSpaceBlock
# Fast learning (perceptual/motor)
s_fast_f0[20,1,type=categorical]         ### Perceptual features: rapidly updated representations
s_fast_f1[10,1,type=categorical]         ### Motor patterns: skill acquisition states
s_fast_f2[5,1,type=categorical]          ### Attention allocation: learning focus areas

# Medium learning (associations/habits)
s_med_f0[50,1,type=categorical]          ### Associative links: stimulus-response mappings
s_med_f1[15,1,type=categorical]          ### Habit strength: behavioral pattern strength
s_med_f2[8,1,type=categorical]           ### Context sensitivity: situational dependencies

# Slow learning (concepts/schemas)
s_slow_f0[30,1,type=categorical]         ### Conceptual knowledge: abstract representations
s_slow_f1[12,1,type=categorical]         ### Schema structures: organized knowledge frameworks
s_slow_f2[6,1,type=categorical]          ### Metacognitive knowledge: learning about learning

## Observations
o_m0[20,1,type=categorical]              ### Sensory input: environmental stimuli
o_m1[10,1,type=categorical]              ### Feedback signals: reward, error, success indicators
o_m2[8,1,type=categorical]               ### Context cues: situational information

## Actions
u_c0[6,1,type=categorical]               ### Learning actions: Attend=0, Practice=1, Explore=2, etc.
u_c1[4,1,type=categorical]               ### Information seeking: Query=0, Test=1, Observe=2, Experiment=3

## Connections
# Bottom-up learning
s_fast_f0 > s_med_f0                     ### Fast learning feeds into associative learning
s_med_f0 > s_slow_f0                     ### Associations contribute to concept formation

# Top-down guidance
s_slow_f2 > s_fast_f2                    ### Metacognitive knowledge guides attention

## InitialParameterization
# Learning rates (hierarchical)
fast_learning_rate = 0.3                 ### High plasticity for fast adaptation
medium_learning_rate = 0.1               ### Moderate plasticity for associations
slow_learning_rate = 0.02                ### Low plasticity for stable concepts

# Precision parameters
perceptual_precision = 2.0               ### High precision for sensory learning
associative_precision = 1.5              ### Medium precision for associations
conceptual_precision = 1.0               ### Lower precision allows flexibility

## Time
Dynamic
DiscreteTime = t
ModelTimeHorizon = 1000

## Footer
This model captures hierarchical learning across multiple timescales and precision levels.
```

### 2. Skill Acquisition and Expertise

```gnn
## ModelName
SkillAcquisitionModel

## ModelAnnotation
Models skill development from novice to expert through precision refinement and automatization.

## StateSpaceBlock
s_f0[5,1,type=categorical]               ### Skill level: Novice=0, Advanced_Beginner=1, Competent=2, Proficient=3, Expert=4
s_f1[4,1,type=categorical]               ### Processing mode: Effortful=0, Mixed=1, Automatic=2, Intuitive=3
s_f2[6,1,type=categorical]               ### Skill components: Component_A through Component_F
s_f3[3,1,type=categorical]               ### Integration level: Independent=0, Coordinated=1, Integrated=2

## InitialParameterization
# Skill development parameters
practice_effect = 0.05                   ### Improvement per practice session
automatization_threshold = 0.8           ### Threshold for automatic processing
expertise_threshold = 0.95               ### Threshold for expert performance

# Precision development
precision_novice = 0.5                   ### Low precision for novices
precision_expert = 2.5                   ### High precision for experts
precision_growth_rate = 0.1              ### Rate of precision increase

## Equations
# Skill improvement
skill_level(t+1) = skill_level(t) + practice_effect * effort(t) * feedback_quality(t)

# Automatization
automatization(t) = practice_time(t) / (practice_time(t) + automatization_constant)
```

## Clinical Applications

### Learning Disabilities

```gnn
## ModelName
LearningDisabilityModel

## ModelAnnotation
Models learning disabilities through altered precision, impaired connectivity, or processing deficits.

## ModifiedParameters
# Dyslexia - phonological processing deficits
phonological_precision = 0.6             ### Reduced from normal 1.5
orthographic_mapping = 0.4               ### Impaired mapping between sounds and letters
reading_automatization = 0.3             ### Delayed automatization

# ADHD - attention and learning difficulties
attention_stability = 0.4                ### Reduced sustained attention
working_memory_capacity = 2              ### Reduced from normal 4
learning_rate_variability = 2.0          ### Increased variability in learning

# Autism - altered learning patterns
social_learning_precision = 0.3          ### Reduced social learning
detail_focused_learning = 2.0            ### Enhanced detail-focused processing
generalization_difficulty = 0.4          ### Difficulty with generalization
```

### Memory Disorders

```gnn
## ModelName
MemoryDisorderModel

## ModelAnnotation
Models memory-related learning impairments through altered consolidation and retrieval.

## ModifiedParameters
# Alzheimer's disease
consolidation_rate = 0.02                ### Severely reduced from normal 0.1
retrieval_precision = 0.3                ### Impaired retrieval
interference_susceptibility = 2.0        ### Increased interference

# Amnesia
episodic_learning = 0.1                  ### Severely impaired episodic learning
semantic_learning = 0.6                  ### Relatively preserved semantic learning
working_memory = 0.4                     ### Impaired working memory
```

## Developmental Models

### Learning Across the Lifespan

```gnn
## ModelName
LifespanLearningModel

## ModelAnnotation
Models learning capacity and characteristics across different life stages.

## DevelopmentalParameters
# Childhood - high plasticity, rapid learning
plasticity_child = 2.0                   ### High neural plasticity
learning_rate_child = 0.4                ### Rapid learning rate
curiosity_drive_child = 1.8              ### High curiosity drive

# Adolescence - continued plasticity with social focus
plasticity_adolescent = 1.5              ### Moderate plasticity
social_learning_weight = 2.0             ### Enhanced social learning
risk_taking_learning = 1.6               ### Learning through risk-taking

# Adulthood - stable learning with experience
plasticity_adult = 1.0                   ### Standard plasticity
expertise_utilization = 1.5              ### Better use of existing knowledge
strategic_learning = 1.8                 ### Enhanced strategic learning

# Aging - compensatory mechanisms
plasticity_aging = 0.7                   ### Reduced plasticity
compensation_strategies = 1.4            ### Enhanced compensation
crystallized_knowledge = 2.0             ### Accumulated knowledge advantage

## Equations
# Age-dependent learning
learning_capacity(age) = base_capacity * plasticity_factor(age) * experience_factor(age)

# Cognitive reserve
reserve_effect(age) = education_level * social_engagement * physical_activity
```

## Social Learning

### Observational Learning and Imitation

```gnn
## ModelName
SocialLearningModel

## ModelAnnotation
Models learning through observation, imitation, and social interaction.

## StateSpaceBlock
s_f0[6,1,type=categorical]               ### Social learning mode: Observe=0, Imitate=1, Collaborate=2, etc.
s_f1[4,1,type=categorical]               ### Model credibility: Low=0, Medium=1, High=2, Expert=3
s_f2[5,1,type=categorical]               ### Social attention: Teacher=0, Peer=1, Group=2, Media=3, Self=4

## InitialParameterization
# Social learning parameters
imitation_fidelity = 0.8                 ### Accuracy of imitation
social_motivation = 1.2                  ### Motivation for social learning
peer_influence = 0.9                     ### Susceptibility to peer influence

# Model selection
expertise_weighting = 1.5                ### Weight given to expert models
similarity_bias = 0.8                    ### Preference for similar models
prestige_bias = 0.6                      ### Preference for high-status models

## Equations
# Social learning effectiveness
social_learning_rate = base_rate * model_credibility * attention_to_model * motivation

# Model selection
model_value = expertise_weight * expertise + similarity_weight * similarity + prestige_weight * prestige
```

## Computational Implementation

### Python Implementation

```python
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


class LearningMode(Enum):
    PASSIVE = 0
    ACTIVE = 1
    STRATEGIC = 2
    METACOGNITIVE = 3


@dataclass
class LearningState:
    """Comprehensive learning state representation"""

    fast_representations: np.ndarray
    associative_links: np.ndarray
    conceptual_knowledge: np.ndarray
    skill_levels: np.ndarray
    learning_mode: LearningMode
    attention_allocation: np.ndarray
    exploration_level: float
    metacognitive_awareness: np.ndarray


class HierarchicalLearningModel:
    """
    Active Inference implementation of hierarchical learning
    """

    def __init__(self, config: Dict):
        self.config = config
        self.initialize_parameters()

    def initialize_parameters(self):
        """Initialize learning-specific parameters"""
        # Learning rates for different levels
        self.fast_learning_rate = self.config.get("fast_learning_rate", 0.3)
        self.medium_learning_rate = self.config.get("medium_learning_rate", 0.1)
        self.slow_learning_rate = self.config.get("slow_learning_rate", 0.02)

        # Precision parameters
        self.perceptual_precision = self.config.get("perceptual_precision", 2.0)
        self.associative_precision = self.config.get("associative_precision", 1.5)
        self.conceptual_precision = self.config.get("conceptual_precision", 1.0)

        # Exploration parameters
        self.curiosity_drive = self.config.get("curiosity_drive", 0.8)
        self.uncertainty_tolerance = self.config.get("uncertainty_tolerance", 0.6)

        # Forgetting parameters
        self.fast_decay = self.config.get("fast_decay_rate", 0.1)
        self.medium_decay = self.config.get("medium_decay_rate", 0.02)
        self.slow_decay = self.config.get("slow_decay_rate", 0.005)

    def update_learning(
        self, experience: Dict[str, np.ndarray], current_state: LearningState
    ) -> LearningState:
        """Update learning state through hierarchical belief updating"""

        # Compute prediction errors at each level
        fast_pe = self.compute_fast_prediction_error(experience, current_state)
        medium_pe = self.compute_medium_prediction_error(experience, current_state)
        slow_pe = self.compute_slow_prediction_error(experience, current_state)

        # Update representations based on prediction errors
        new_fast_repr = self.update_fast_representations(fast_pe, current_state)
        new_assoc_links = self.update_associative_links(medium_pe, current_state)
        new_concepts = self.update_conceptual_knowledge(slow_pe, current_state)

        # Update skill levels
        new_skills = self.update_skill_levels(experience, current_state)

        # Determine new learning mode
        new_mode = self.select_learning_mode(experience, current_state)

        # Update attention allocation
        new_attention = self.update_attention_allocation(fast_pe, medium_pe, slow_pe)

        # Compute exploration level
        exploration = self.compute_exploration_level(current_state, experience)

        # Update metacognitive awareness
        metacog_awareness = self.update_metacognitive_awareness(
            current_state, experience
        )

        return LearningState(
            fast_representations=new_fast_repr,
            associative_links=new_assoc_links,
            conceptual_knowledge=new_concepts,
            skill_levels=new_skills,
            learning_mode=new_mode,
            attention_allocation=new_attention,
            exploration_level=exploration,
            metacognitive_awareness=metacog_awareness,
        )

    def compute_fast_prediction_error(
        self, experience: Dict, state: LearningState
    ) -> np.ndarray:
        """Compute prediction error for fast perceptual learning"""
        sensory_input = experience["sensory"]
        predicted_sensory = self.predict_sensory(state.fast_representations)

        pe = self.perceptual_precision * (sensory_input - predicted_sensory)
        return pe

    def update_fast_representations(
        self, prediction_error: np.ndarray, state: LearningState
    ) -> np.ndarray:
        """Update fast perceptual representations"""
        learning_rate = self.modulate_learning_rate(self.fast_learning_rate, state)

        # Update with forgetting
        updated_repr = state.fast_representations * (1 - self.fast_decay)
        updated_repr += learning_rate * prediction_error

        return updated_repr

    def modulate_learning_rate(self, base_rate: float, state: LearningState) -> float:
        """Modulate learning rate based on attention and surprise"""
        attention_weight = np.mean(state.attention_allocation)
        surprise_weight = 1.0  # Would be computed from prediction error magnitude
        motivation_weight = self.get_motivation_level(state)

        modulated_rate = (
            base_rate * attention_weight * surprise_weight * motivation_weight
        )
        return np.clip(modulated_rate, 0.0, 1.0)

    def select_learning_mode(
        self, experience: Dict, state: LearningState
    ) -> LearningMode:
        """Select appropriate learning mode based on context and metacognition"""

        # Consider task difficulty, prior knowledge, and metacognitive state
        task_difficulty = self.assess_task_difficulty(experience)
        prior_knowledge = self.assess_prior_knowledge(state)
        metacog_level = np.mean(state.metacognitive_awareness)

        if metacog_level > 0.8 and task_difficulty > 0.7:
            return LearningMode.METACOGNITIVE
        elif prior_knowledge > 0.6 and task_difficulty > 0.5:
            return LearningMode.STRATEGIC
        elif task_difficulty > 0.3:
            return LearningMode.ACTIVE
        else:
            return LearningMode.PASSIVE
```

## Future Directions

1. **Lifelong Learning**: Continuous adaptation across extended time periods
2. **Transfer Learning**: Generalization across domains and contexts
3. **Meta-Learning**: Learning how to learn more effectively
4. **Social Learning Networks**: Collective intelligence and distributed learning
5. **Personalized Learning**: Adaptive systems tailored to individual learners

## References

### Core Papers
- Anderson, J. R. (1982). Acquisition of cognitive skill
- Ericsson, K. A., & Lehmann, A. C. (1996). Expert and exceptional performance
- Roediger, H. L., & Butler, A. C. (2011). The critical role of retrieval practice

### Active Inference Applications
- Friston, K. J., et al. (2016). Active inference, curiosity and insight
- Sajid, N., et al. (2021). Active inference: Demystified and compared 


## Documents in this folder

- [Hierarchical Learning and Adaptation Model](hierarchical_learning_model.md)
