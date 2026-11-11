# Emotion and Affect in Active Inference

## Overview

Emotion and affect emerge from interoceptive prediction, allostatic regulation, and precision-weighted prediction error minimization. In Active Inference, emotions represent policies for maintaining homeostatic balance and optimizing long-term survival and wellbeing.

## Core Components

### 1. Interoceptive Inference and Emotion

```gnn
## ModelName
InteroceptiveEmotionModel

## ModelAnnotation
Models emotion through interoceptive prediction and allostatic regulation.
Emotions emerge from prediction errors about bodily states and their homeostatic implications.

## GNNVersionAndFlags
GNN_v1.0
ProcessingFlags: interoception, allostasis, emotion_regulation

## StateSpaceBlock
# Physiological level
s_physio_f0[8,1,type=continuous]         ### Autonomic states: HR, BP, GSR, Temp, Cortisol, etc.
s_physio_f1[4,1,type=categorical]        ### Arousal level: Low=0, Medium=1, High=2, Very_High=3
s_physio_f2[3,1,type=categorical]        ### Valence: Negative=0, Neutral=1, Positive=2

# Interoceptive level
s_intero_f0[6,1,type=categorical]        ### Interoceptive signals: Hunger=0, Thirst=1, Pain=2, Fatigue=3, Arousal=4, Visceral=5
s_intero_f1[4,1,type=categorical]        ### Interoceptive accuracy: Poor=0, Fair=1, Good=2, Excellent=3
s_intero_f2[5,1,type=categorical]        ### Body awareness: Minimal=0, Low=1, Moderate=2, High=3, Hypervigilant=4

# Emotional level
s_emo_f0[8,1,type=categorical]           ### Basic emotions: Joy=0, Sadness=1, Anger=2, Fear=3, Disgust=4, Surprise=5, Interest=6, Shame=7
s_emo_f1[5,1,type=categorical]           ### Emotion intensity: Very_Low=0, Low=1, Medium=2, High=3, Very_High=4
s_emo_f2[4,1,type=categorical]           ### Emotion regulation: Unregulated=0, Suppression=1, Reappraisal=2, Acceptance=3

## Observations
o_m0[8,1,type=continuous]                ### Physiological measurements: matches s_physio_f0
o_m1[6,1,type=categorical]               ### Interoceptive sensations: matches s_intero_f0
o_m2[10,1,type=categorical]              ### Environmental stimuli: Threat=0, Reward=1, Social=2, Novel=3, etc.

## Actions
u_c0[8,1,type=categorical]               ### Emotional expression: matches s_emo_f0
u_c1[6,1,type=categorical]               ### Regulation strategies: Suppression=0, Reappraisal=1, Distraction=2, etc.

## Connections
# Bottom-up interoceptive processing
s_physio_f0 > s_intero_f0                ### Physiological states generate interoceptive signals
s_intero_f0 > s_emo_f0                   ### Interoceptive signals contribute to emotions

# Top-down regulation
s_emo_f2 > s_emo_f1                     ### Regulation modulates emotion intensity
u_c1 > s_physio_f0                       ### Regulation actions affect physiology

## InitialParameterization
# Interoceptive precision
interoceptive_precision = 1.2            ### Moderate precision for interoceptive signals
physiological_precision = 2.0            ### High precision for physiological states

# Emotion parameters
emotion_threshold = 0.5                  ### Threshold for emotion activation
regulation_strength = 0.7                ### Effectiveness of emotion regulation

# Allostatic parameters
homeostatic_set_points = [70, 120, 0.5, 37, 10, 0.3, 0.4, 0.6]  ### Target values for physiology

## Time
Dynamic
DiscreteTime = t
ModelTimeHorizon = 200

## Footer
This model captures emotion through interoceptive inference and allostatic regulation.
```

### 2. Appraisal and Emotion Generation

```gnn
## ModelName
AppraisalEmotionModel

## ModelAnnotation
Models emotion generation through cognitive appraisal processes and their interaction with physiological arousal.

## StateSpaceBlock
s_f0[6,1,type=categorical]               ### Primary appraisal: Threat=0, Challenge=1, Benefit=2, Loss=3, Irrelevant=4, Novel=5
s_f1[4,1,type=categorical]               ### Secondary appraisal: Can_Cope=0, Cannot_Cope=1, Uncertain=2, No_Coping_Needed=3
s_f2[5,1,type=categorical]               ### Attribution: Internal=0, External=1, Stable=2, Unstable=3, Controllable=4
s_f3[8,1,type=categorical]               ### Resulting emotion: matches basic emotions

o_m0[10,1,type=categorical]              ### Stimulus characteristics: Intensity, Novelty, Predictability, etc.
o_m1[4,1,type=categorical]               ### Context: Social=0, Achievement=1, Relationship=2, Survival=3

u_c0[6,1,type=categorical]               ### Appraisal focus: Relevance=0, Consequences=1, Coping=2, Agency=3, Norms=4, Self_Esteem=5

## InitialParameterization
# Appraisal biases
threat_sensitivity = 1.2                 ### Heightened threat detection
optimism_bias = 0.8                      ### Slight optimistic bias in appraisals
control_beliefs = 1.0                    ### Belief in personal control

# Individual differences
neuroticism = 1.0                        ### Emotional stability (0-2 scale)
extraversion = 1.0                       ### Social orientation (0-2 scale)
trait_anxiety = 1.0                      ### Baseline anxiety level

## Equations
# Appraisal integration
appraisal_outcome(t) = primary_appraisal(t) × secondary_appraisal(t) × attribution(t)

# Emotion probability
P(emotion_i|appraisal) = softmax(appraisal_emotion_weights_i × appraisal_outcome(t))
```

### 3. Emotion Regulation and Control

```gnn
## ModelName
EmotionRegulationModel

## ModelAnnotation
Models various emotion regulation strategies and their effectiveness in modulating emotional responses.

## StateSpaceBlock
s_f0[6,1,type=categorical]               ### Regulation strategy: Reappraisal=0, Suppression=1, Distraction=2, Acceptance=3, Problem_Solving=4, Social_Support=5
s_f1[4,1,type=categorical]               ### Regulation timing: Antecedent=0, Response=1, Ongoing=2, Post_Response=3
s_f2[3,1,type=categorical]               ### Regulation success: Failed=0, Partial=1, Successful=2
s_f3[5,1,type=categorical]               ### Cognitive load: Very_Low=0, Low=1, Medium=2, High=3, Very_High=4

o_m0[8,1,type=categorical]               ### Emotion to regulate: matches basic emotions
o_m1[4,1,type=categorical]               ### Regulation demand: Low=0, Medium=1, High=2, Extreme=3

u_c0[6,1,type=categorical]               ### Choose regulation strategy: matches s_f0
u_c1[3,1,type=categorical]               ### Regulation effort: Low=0, Medium=1, High=2

## InitialParameterization
# Strategy effectiveness
reappraisal_effectiveness = 0.8          ### High effectiveness for reappraisal
suppression_effectiveness = 0.4          ### Lower effectiveness for suppression
distraction_effectiveness = 0.6          ### Moderate effectiveness
acceptance_effectiveness = 0.7           ### Good effectiveness for acceptance

# Cognitive costs
reappraisal_cost = 0.6                   ### Moderate cognitive cost
suppression_cost = 0.8                   ### High cognitive cost
distraction_cost = 0.4                   ### Low cognitive cost
acceptance_cost = 0.2                    ### Very low cognitive cost

## Equations
# Regulation effectiveness
effectiveness(t) = base_effectiveness × practice_factor × situational_match × individual_capacity

# Cognitive load
cognitive_load(t) = strategy_cost × regulation_effort × emotion_intensity
```

## Clinical Applications

### Depression and Mood Disorders

```gnn
## ModelName
DepressionEmotionModel

## ModelAnnotation
Models depression through altered interoceptive processing, negative appraisal biases, and impaired emotion regulation.

## ModifiedParameters
# Altered interoceptive processing
interoceptive_precision = 0.8            ### Reduced from normal 1.2
negative_interoceptive_bias = 1.5        ### Increased sensitivity to negative signals

# Appraisal biases
threat_sensitivity = 1.8                 ### Increased from normal 1.2
optimism_bias = 0.3                      ### Reduced from normal 0.8
hopelessness_bias = 1.6                  ### Strong hopelessness bias

# Emotion regulation deficits
regulation_strength = 0.4                ### Reduced from normal 0.7
rumination_tendency = 1.8                ### Increased rumination
avoidance_preference = 1.4               ### Increased avoidance behaviors

# Physiological alterations
cortisol_baseline = 1.5                  ### Elevated baseline cortisol
autonomic_reactivity = 0.6               ### Reduced autonomic flexibility

## Equations
# Negative mood maintenance
mood_persistence_depression = 0.95       ### Higher than normal 0.8

# Cognitive bias
negative_interpretation_bias(t) = depression_severity × appraisal_negativity
```

### Anxiety Disorders

```gnn
## ModelName
AnxietyEmotionModel

## ModelAnnotation
Models anxiety through hypervigilant threat detection, catastrophic appraisals, and avoidance behaviors.

## ModifiedParameters
# Hypervigilant threat detection
threat_sensitivity = 2.0                 ### Highly increased from normal 1.2
threat_threshold = 0.3                   ### Lowered from normal 0.5
attention_bias_threat = 1.8              ### Strong attentional bias to threats

# Catastrophic appraisals
probability_overestimation = 1.7         ### Overestimate threat probability
cost_overestimation = 1.8                ### Overestimate threat consequences
control_underestimation = 0.4            ### Underestimate personal control

# Avoidance and safety behaviors
avoidance_preference = 2.0               ### Strong avoidance tendency
safety_behavior_reliance = 1.6          ### High reliance on safety behaviors
intolerance_uncertainty = 1.9           ### High intolerance of uncertainty

# Physiological hyperarousal
arousal_baseline = 1.4                   ### Elevated baseline arousal
arousal_reactivity = 1.8                 ### Heightened arousal response
recovery_rate = 0.6                      ### Slower autonomic recovery
```

### Emotion Dysregulation

```gnn
## ModelName
EmotionDysregulationModel

## ModelAnnotation
Models emotion dysregulation through impaired regulation strategies and heightened emotional reactivity.

## ModifiedParameters
# Heightened emotional reactivity
emotion_threshold = 0.3                  ### Lowered from normal 0.5
emotion_intensity_multiplier = 1.6       ### Increased emotional intensity
emotional_lability = 1.8                ### Increased emotional instability

# Impaired regulation
regulation_strength = 0.3                ### Severely reduced from normal 0.7
strategy_repertoire = 0.5                ### Limited regulation strategies
regulation_flexibility = 0.4             ### Poor strategy switching

# Maladaptive strategies
suppression_overuse = 1.8                ### Overreliance on suppression
rumination_tendency = 2.0                ### Strong rumination tendency
emotional_avoidance = 1.7                ### High emotional avoidance
```

## Developmental Models

### Emotional Development

```gnn
## ModelName
EmotionalDevelopmentModel

## ModelAnnotation
Models the development of emotion understanding, regulation, and expression across the lifespan.

## DevelopmentalParameters
# Age-dependent emotion regulation
regulation_infant = 0.1                  ### Minimal self-regulation in infancy
regulation_child = 0.4                   ### Developing regulation in childhood
regulation_adolescent = 0.6              ### Improving but still developing
regulation_adult = 0.8                   ### Mature regulation capabilities

# Emotional complexity
basic_emotions_infant = 4                ### Few basic emotions in infancy
emotion_vocabulary_child = 20            ### Limited emotion words
emotion_vocabulary_adult = 200           ### Rich emotion vocabulary
emotional_granularity_development = progressive  ### Increasing emotional differentiation

# Social emotion development
empathy_development = [0.2, 0.5, 0.8, 1.0]      ### Ages 2, 5, 10, adult
theory_of_mind_emotions = [0.0, 0.3, 0.7, 1.0]  ### Understanding others' emotions
social_emotion_regulation = [0.1, 0.4, 0.7, 0.9] ### Regulation in social contexts

## Equations
# Emotional learning
emotion_concept_formation(t) = experience_frequency(t) × social_feedback(t) × cognitive_maturity(t)

# Regulation skill development
regulation_skill(age) = base_capacity(age) × practice_time × scaffolding_quality
```

## Social and Cultural Dimensions

### Cultural Emotion Models

```gnn
## ModelName
CulturalEmotionModel

## ModelAnnotation
Models cultural variations in emotion concepts, expression rules, and regulation strategies.

## CulturalParameters
# Display rules
emotional_expression_appropriateness = culture_specific_matrix
intensity_norms = culture_specific_values
context_sensitivity = high_vs_low_context_cultures

# Emotion concepts
emotion_categories = culture_specific_categories
emotion_valuation = culture_specific_preferences
social_vs_individual_emotions = cultural_emphasis

# Regulation strategies
preferred_strategies = culture_specific_preferences
social_regulation_emphasis = collectivist_vs_individualist
emotion_suppression_norms = cultural_norms

## Equations
# Cultural fit
cultural_emotion_fit(t) = |expressed_emotion(t) - cultural_norm(t)|

# Social consequences
social_approval(t) = cultural_appropriateness(t) × context_match(t)
```

## Computational Implementations

### Python Implementation

```python
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class EmotionType(Enum):
    JOY = 0
    SADNESS = 1
    ANGER = 2
    FEAR = 3
    DISGUST = 4
    SURPRISE = 5
    INTEREST = 6
    SHAME = 7


class RegulationStrategy(Enum):
    REAPPRAISAL = 0
    SUPPRESSION = 1
    DISTRACTION = 2
    ACCEPTANCE = 3
    PROBLEM_SOLVING = 4
    SOCIAL_SUPPORT = 5


@dataclass
class EmotionalState:
    """Comprehensive emotional state representation"""

    physiological_arousal: np.ndarray
    interoceptive_signals: np.ndarray
    current_emotions: np.ndarray
    emotion_intensities: np.ndarray
    regulation_strategy: RegulationStrategy
    mood_state: float
    appraisal_pattern: np.ndarray


class InteroceptiveEmotionModel:
    """
    Active Inference implementation of emotion through interoceptive processing
    """

    def __init__(self, config: Dict):
        self.config = config
        self.initialize_parameters()

    def initialize_parameters(self):
        """Initialize emotion-specific parameters"""
        # Interoceptive precision
        self.interoceptive_precision = self.config.get("interoceptive_precision", 1.2)
        self.physiological_precision = self.config.get("physiological_precision", 2.0)

        # Homeostatic set points
        self.homeostatic_targets = np.array(
            [
                70,  # Heart rate
                120,  # Blood pressure
                0.5,  # Galvanic skin response
                37,  # Temperature
                10,  # Cortisol (normalized)
                0.3,  # Breathing rate (normalized)
                0.4,  # Muscle tension (normalized)
                0.6,  # Autonomic balance (normalized)
            ]
        )

        # Emotion parameters
        self.emotion_threshold = self.config.get("emotion_threshold", 0.5)
        self.regulation_strength = self.config.get("regulation_strength", 0.7)

    def update_emotional_state(
        self,
        interoceptive_input: np.ndarray,
        environmental_context: np.ndarray,
        current_state: EmotionalState,
    ) -> EmotionalState:
        """Update emotional state through interoceptive inference"""

        # Interoceptive prediction and error
        predicted_interoception = self.predict_interoceptive_signals(current_state)
        interoceptive_pe = self.compute_interoceptive_pe(
            interoceptive_input, predicted_interoception
        )

        # Physiological regulation (allostasis)
        physiological_state = self.update_physiological_state(
            current_state, interoceptive_pe
        )

        # Appraisal processes
        appraisal = self.appraise_situation(
            environmental_context, physiological_state, current_state
        )

        # Emotion generation
        emotions = self.generate_emotions(
            interoceptive_pe, appraisal, physiological_state
        )

        # Emotion regulation
        regulated_emotions = self.apply_emotion_regulation(
            emotions, current_state.regulation_strategy, current_state
        )

        return EmotionalState(
            physiological_arousal=physiological_state,
            interoceptive_signals=interoceptive_input,
            current_emotions=regulated_emotions,
            emotion_intensities=self.compute_emotion_intensities(regulated_emotions),
            regulation_strategy=self.select_regulation_strategy(appraisal, emotions),
            mood_state=self.update_mood(current_state.mood_state, emotions),
            appraisal_pattern=appraisal,
        )

    def compute_interoceptive_pe(
        self, observed: np.ndarray, predicted: np.ndarray
    ) -> np.ndarray:
        """Compute interoceptive prediction error"""
        return self.interoceptive_precision * (observed - predicted)

    def generate_emotions(
        self,
        interoceptive_pe: np.ndarray,
        appraisal: np.ndarray,
        physiological_state: np.ndarray,
    ) -> np.ndarray:
        """Generate emotions from interoceptive PE and appraisals"""

        # Combine interoceptive signals with cognitive appraisals
        arousal_component = np.linalg.norm(
            physiological_state - self.homeostatic_targets
        )
        valence_component = self.compute_valence(appraisal)

        # Map to basic emotions
        emotion_activations = np.zeros(8)  # 8 basic emotions

        # Joy: positive valence, moderate arousal
        emotion_activations[EmotionType.JOY.value] = max(
            0, valence_component * (1 - abs(arousal_component - 0.5))
        )

        # Fear: negative valence, high arousal, threat appraisal
        emotion_activations[EmotionType.FEAR.value] = max(
            0, -valence_component * arousal_component * appraisal[0]
        )

        # Sadness: negative valence, low arousal, loss appraisal
        emotion_activations[EmotionType.SADNESS.value] = max(
            0, -valence_component * (1 - arousal_component) * appraisal[2]
        )

        # Anger: negative valence, high arousal, obstacle appraisal
        emotion_activations[EmotionType.ANGER.value] = max(
            0, -valence_component * arousal_component * appraisal[1]
        )

        # Apply threshold
        emotion_activations[emotion_activations < self.emotion_threshold] = 0

        return emotion_activations

    def apply_emotion_regulation(
        self,
        emotions: np.ndarray,
        strategy: RegulationStrategy,
        current_state: EmotionalState,
    ) -> np.ndarray:
        """Apply emotion regulation strategy"""

        if strategy == RegulationStrategy.REAPPRAISAL:
            # Reappraisal reduces negative emotions through cognitive change
            regulation_effect = 0.8 * self.regulation_strength
            emotions[emotions < 0] *= 1 - regulation_effect

        elif strategy == RegulationStrategy.SUPPRESSION:
            # Suppression reduces expression but not feeling
            regulation_effect = 0.4 * self.regulation_strength
            emotions *= 1 - regulation_effect

        elif strategy == RegulationStrategy.DISTRACTION:
            # Distraction reduces attention to emotional stimuli
            regulation_effect = 0.6 * self.regulation_strength
            emotions *= 1 - regulation_effect

        elif strategy == RegulationStrategy.ACCEPTANCE:
            # Acceptance doesn't change emotion but reduces struggle
            # Implementation would involve changing relationship to emotion
            pass

        return emotions
```

## Research Applications

### Affective Computing
- Emotion recognition from physiological signals
- Adaptive user interfaces based on emotional state
- Therapeutic virtual agents for emotion regulation training

### Clinical Assessment
- Computational phenotyping of mood disorders
- Objective measures of emotion regulation capacity
- Personalized treatment selection based on emotional profiles

### Educational Applications
- Social-emotional learning curricula
- Emotion regulation training programs
- Understanding emotional factors in learning

## Future Directions

1. **Interpersonal Emotion Dynamics**: Modeling emotional contagion and co-regulation
2. **Cultural Emotion Models**: Cross-cultural variations in emotional processing
3. **Developmental Trajectories**: Emotional development across the lifespan
4. **Emotion-Cognition Integration**: Bidirectional effects between emotion and cognition
5. **Therapeutic Applications**: Emotion-focused computational interventions

## References

### Core Papers
- Barrett, L. F. (2017). How emotions are made: The secret life of the brain
- Gross, J. J. (2015). Emotion regulation: Current status and future prospects
- Seth, A. K. (2013). Interoceptive inference, emotion, and the embodied self

### Active Inference Applications
- Allen, M., & Friston, K. J. (2018). From cognitivism to autopoiesis: Towards a computational framework for the embodied mind
- Smith, R., et al. (2019). The hierarchical basis of neurovisceral integration 


## Documents in this folder

- [Interoceptive Emotion and Affect Model](interoceptive_emotion_model.md)
