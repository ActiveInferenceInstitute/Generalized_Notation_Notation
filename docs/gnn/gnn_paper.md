# GNN Paper: Generalized Notation Notation

**Version**: 3.6.0
**Last Updated**: 2026-09-23
**Status**: Maintained
**Scope**: GNN language, pipeline, and renderer architecture. See [implementations/README.md](implementations/README.md) and the repository test commands for current inventory and evidence.

Published on April 5, 2023 at: <https://zenodo.org/records/7803328>

Generalized Notation Notation for Active Inference Models
Jakub Smekal [0000−0003−4989−4968] and Daniel Ari Friedman [0000−0001−6232−9096]
Active Inference Institute, USA <https://www.activeinference.institute/>
<blanket@activeinference.institute>

Abstract. This paper introduces Generalized Notation Notation (GNN),
a novel approach to generative model representation that facilitates communication, understanding, and application of Active Inference across
various domains. GNN complements the Active Inference Ontology as a
flexible and expressive language for education and modeling, by providing a standardized method for describing cognitive models. In this paper
we introduce GNN, and provide a Step-by-Step example of what GNN
looks like in practice. We then explore ”the Triple Play”, a pragmatic
approach to expressing GNN in linguistic, visual, and executable cognitive models. By situating GNN within the broader context of cognitive
modeling and Active Inference, this work aims to bridge and respect the
gaps among different modeling settings. The goal of this work is to facilitate interdisciplinary research and application, ultimately promoting
the advancement of the field.
Keywords: Active Inference · cognitive models · model representation
· hierarchical cognitive models · Bayesian statistics · generative models.

## 1 Introduction

### 1.1 Communicating Active Inference: Challenges and opportunities

In recent years, the cognitive sciences have made significant strides towards understanding the complex nature of human cognition and behavior. One promising approach is Active Inference, a unifying theoretical framework that combines
perception, action, and learning in a coherent manner [1] [2]. Despite the potential value of models within this framework, the widespread adoption of Active
Inference has been hindered by the lack of a standardized method for effectively
representing and communicating them. In this paper, we address this challenge
by introducing a novel approach to cognitive model representation called Generalized Notation Notation (GNN), which aims to facilitate communication and
understanding of Active Inference models across various domains and settings.

### 1.2 Comprehensively described cognitive models are key

Developing comprehensive, accessible, reproducible, interoperable cognitive models is crucial for the advancement of the Active Inference field. Without a standardized language for describing cognitive models, researchers experience friction when collaborating, sharing, and building upon existing work. In Active
Inference research, models are often conveyed through assemblages of natural
language, pseudocode, programming languages, analytical formulas, and pictorial representations. In this paper, we present GNN as a flexible and expressive
language tailored for expressing Active Inference models, and encompassing various relevant aspects of languages, including ontology, morphology, grammar,
and pragmatics. By leveraging GNN as an Active Inferlingua (or Interlingua, Infralingua, or Supralingua), we aim to bridge and respect the gaps among different
modeling approaches in order to facilitate interdisciplinary research.

### 1.3 Goals and structure of the paper

In this paper, first we present the current specification of the GNN language
and method (Section 2). To provide an example of GNN in action, we provide the GNN representation of a recent step-by-step Active Inference tutorial
[3] [4] (Section 3). We point towards the practical utility of GNN by exploring
”the Triple Play” – a pragmatic approach to expressing GNN in three distinct
modalities: text-based models, statistical graphical models, and executable cognitive models (Section 4). Lastly, we discuss the implications of our findings,
in terms of philosophical and taxonomic perspectives on the diversity of Active
Inference generative models, and suggest future research directions (Section 5).
By situating GNN within the broader context of hierarchical cognitive models
and Active Inference, we hope to inspire further exploration and development of
this promising approach to understanding and communicating complex cognitive
processes.

## 2 Active Inference Linguistics: Ontology, Morphology, and Grammar

### 2.1 Generalized Notation Notation (GNN) overview

GNN describes Active Inference models with ASCII letters and punctuation,
structured in a source file that accords with the principles of Markdown. In this
section we provide a snapshot of the current specification of GNN.
Updated information on GNN can be found at Github [10] or in Coda [11].

### 2.2 GNN punctuation

GNN utilizes standard ASCII characters to represent Active Inference generative models. Key punctuation rules include:
- **Variable Declarations**: Formatted as `NAME[dim_1, dim_2, ..., key=value]`. Variables are alphanumeric and case-sensitive. The `type` key is required (e.g., `A[3,3, type=float]`).
- **Default Value Hints**: Extended syntax supports `default=uniform`, `zeros`, `ones`, or `eye` (e.g., `W[4,4, type=float, default=zeros]`).
- **Connection Operators**: Relationships between state-space variables are denoted by specific operators:
  - `>` signifies a causal or directed edge (e.g., `D>s` for prior to hidden state).
  - `-` signifies an undirected or bidirectional edge (e.g., `s-A` for state to likelihood mapping).
- **Annotations**: Colons `:` can be used to annotate edges for documentation and rendering (e.g., `G>π:policy_selection`).
- **Initial Parameterization**: Matrices and tensors are parameterized using outer braces `{...}` for the full tensor and inner parentheses `(...)` for rows or slices.

### 2.3 GNN source file structure

A GNN source file is a UTF-8 Markdown file (`.md`) organized into ordered sections, denoted by level-2 headers (`##`). The strict schema validator requires the following sections; the order below is the authoring convention, but validation itself checks only that each is present:
1. `## GNNSection`: A short, space-free identifier for the model (e.g., `ActInfPOMDP`).
2. `## GNNVersionAndFlags`: Indicates the GNN specification version, such as `GNN v1.1`.
3. `## ModelName`: A human-readable title for the generative model.
4. `## StateSpaceBlock`: The core declaration block where all variables, matrices, and dimensions are defined.
5. `## Connections`: A structured list of directed and undirected edges forming the causal structure of the model.

Optional sections, such as `## ModelAnnotation`, `## InitialParameterization`, `## ModelParameters`, and `## ActInfOntologyAnnotation`, can be included to provide further semantic, mathematical, and historical context.

## 3 Step-by-Step GNN

### 3.1 A Step-by-Step example of GNN applied to Smith et al. 2022

Here we provide several examples of increasing expressiveness of GNN, as defined
in the prior section. The examples are directly drawn from ”A step-by-step
tutorial on active inference and its application to empirical data” by Smith,
Friston, and Whyte 2022 [3] [4]. Just as the goal of the initial step-by-step paper
was to start simple and progressively add model features till one has arrived at a
full Active Inference generative model, here we use that exact same progression
to demonstrate the flexibility of GNN.

For instance, the classic T-Maze Epistemic Foraging Agent from this tutorial can be formally represented in GNN as follows. This snippet highlights the structural mapping of the POMDP state space and its connections:

```gnn
## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1.1

## ModelName
T-Maze Epistemic Foraging Agent

## StateSpaceBlock
# Hidden state factors
s_loc[4,1,type=float]        # Location state: (0:center, 1:left, 2:right, 3:cue)
s_ctx[2,1,type=float]        # Context state: (0:reward_left, 1:reward_right)

# Observation modalities
o_loc[4,1,type=int]          # Location observation
o_rew[3,1,type=int]          # Reward/cue observation

# Generative model matrices
A_loc[4,4,type=float]        # Location likelihood: P(o_loc | s_loc)
A_rew[3,4,2,type=float]      # Reward likelihood: P(o_rew | s_loc, s_ctx)

# Policy and action
pi[4,type=float]             # Policy over 4 actions
u[1,type=int]                # Selected action
G[pi,type=float]             # Expected Free Energy per policy

## Connections
s_loc-A_loc
A_loc-o_loc
s_loc-A_rew
s_ctx-A_rew
A_rew-o_rew
G>pi
pi>u
```
This plain-text scaffolding provides both humans and machines with a concise, unambiguous representation of the generative model's epistemic and instrumental properties.

## 4 Expressing GNN: The Triple Play

### 4.1 Every GNN expression has a pragmatic and epistemic context

As with communication more broadly, it is key to consider the pragmatic and
epistemic context associated with the use of any given Generalized Notation
Notation (GNN) expressions. Here, the pragmatic context refers to the practical aspects of communication, such as the preferences and behavioral policies of
the communicators, while the epistemic context pertains to the beliefs of those
involved in the ”ecosystem of shared intelligence” [5]. By taking agent-scale pragmatic and epistemic contexts into account, GNN expressions can be tailored to
effectively convey complex concepts using a linguistics that translates seamlessly
across different modalities.
Various strategies and tactics may be helpful when employing GNN expressions in different settings and using a spectrum of model precisions (e.g. from
informal conversation, to a beautiful presentation, to a fully-documented reproducible research product). Model precision should be balanced against the need
for comprehensibility and ease of communication.
Broadly, strategic considerations for expressing GNN include the overall approach to communication, such as the choice of modality and level of technical
rigor applied for a given audience. Tactical considerations for expressing GNN include specific techniques for enhancing clarity and understanding, such as using
visual aids, examples, and analogies.
Below we highlight the “Triple Play”, gesturing towards the high-fidelity
rendering of a GNN expression (of a small motif or complete ecosystem-scale
generative model) across plain-text, graphical, and executable (computational)
forms.

### 4.2 Text-based models

At its core, GNN is a text-based model that can be rendered into different formats, including mathematical notation, visualized figures, natural language descriptions, algorithmic pseudocode, and executable simulations. The plain-text
basis of GNN provides a flexible framework for communicating these computational models, allowing for the integration of different representations to better
convey the underlying concepts and relationships of active inference linguistics.
Additionally the plain-text basis of GNN enables the use of tools such as Regular
Expressions and Large Language Models.

### 4.3 Graphical models

Graphical models can be visualized using GNN. In the context of Bayesian statistics, graphical models offer a powerful way to represent complex relationships
and dependencies. GNN enables the detailing and rendering of clear and informative visual representations that can be easily understood by different audiences.
Here GNN can be applied as post hoc documentation, by deriving plain-text of
Figures drawn in papers. Additionally GNN can be used pre hoc or in medias res
during the process of designing and implementing Active Inference models. By
incorporating graphical models into GNN expressions, we can on one hand utilize
the power of Bayesian statistics, and on the other hand benefit from improved
visual communication of intricate cognitive models and concepts.

### 4.4 Executable Cognitive models

Lastly, cognitive models can be executed using GNN. Cognitive models represent
the mental processes and structures described by Active Inference. GNN provides
a means to specify these models in a formal and precise manner, allowing for their
implementation and testing in computational simulations. Critically, GNN as a
pseudocode does not restrict which programming language or package ultimately
implements the particular generative model in question. 

### 4.5 Neurosymbolic Context Analysis

The Triple Play formulation is augmented by the pipeline's research and intelligent-analysis stages (Steps 19 and 24). The integration operates via:
1. **Structural Semantic Extraction:** Mapping matrix nodes into Active Inference ontological databases.
2. **LLM Synthesis:** Translating complex categorical topologies into heuristically approachable natural language streams, generating self-documenting executable architectures.
3. **Execution Recovery Heuristics:** Automatically triggering heuristic code recovery when downstream solvers fail, driving a continuous reinforcement iteration loop against explicit execution environments.

```mermaid
graph LR
    A[GNN Structural State] --> B{Ontological Mapping}
    B --> C[Edge-Based LLM Parsing]
    C --> D[Executable Source Code generation]
    D --> E{Execution Success?}
    E -->|No| F[Heuristic Circuit Breaker]
    F --> C
    E -->|Yes| G[Intelligent Analysis Report]
```

### 4.6 Execution and Metric Logging

To advance the pragmatics of cognitive modeling, the GNN ecosystem provides built-in mechanisms for executing rendered models and instrumenting them to extract native behavioral metrics. When models are executed via the pipeline (e.g., Step 12 Execute), variables such as Variational Free Energy (VFE) and Expected Free Energy (EFE) are logged continuously over the simulation horizon.

These execution traces allow researchers to empirically investigate the theoretical properties of Active Inference. Validation testing over uniform priors across $N \in \{2, 4, 8\}$ confirms that entropy — and hence an agent's initial VFE prior to active uncertainty resolution — scales as $\ln(N)$, matching analytical flat-prior expectations. The automated analysis layer (Step 16) directly consumes these traces to generate trajectory and energy convergence plots, rendering complex behavioral phenomena legible without manual metric extraction. Performance scaling is exercised by the PyMDP scaling orchestrator (`scripts/experiments/run_pymdp_gnn_scaling_analysis.py`) over a configured grid of $N \in \{2, 4, 8, 16, 32, 64, 128\}$ against $T \in \{10, 50, 100\}$, with a per-model wall-clock timeout of 20 minutes. A representative subset — one spec per state-space size from $N=4$ to $N=64$, each at $T=100$ — is retained under `input/gnn_files/pymdp_scaling_study/`; the full grid is generated rather than hand-authored.

## 5 Discussion

### 5.1 Summary of key findings and developments

In recent years, the field of cognitive science has experienced significant epistemic
advancements and pragmatic developments, especially in the context of Active
Inference generative models [2]. These models, which encompass wide ranges of
statistical and symbolic model types [7], have been instrumental in shaping our
understanding of cognitive processes.

### 5.2 Philosophical and Taxonomic perspectives on Active Inference model diversity

Ongoing discussions among various factions (e.g. scientific realism and instrumentalism) have raised important questions about the nature, function, and
purpose of Active Inference models [8].
The realism-instrumentalism distinction plays a crucial role in shaping our
understanding of generative models in Active Inference. Scientific realism posits
that these models are approximations of universal truths about reality, implying
that they provide veridical representations of the cognitive processes they aim
to describe. On the other hand, scientific instrumentalism suggests that these
models are intellectual structures that facilitate predictions and problem-solving
in specific domains, without necessarily corresponding to any underlying reality.
This distinction raises the question of how these models might be viewed as tools
for representing objective cognitive truths, and/or for subjective understanding
and prediction.
There are significant implications of the realism-instrumentalism conversation in the context of Active Inference cognitive models. Both the realist and
instrumentalist perspectives can inform the development of more comprehensive and accurate models, by emphasizing the need for a balance among factors
such as explanatory power and predictive utility. By taking into account the
strengths and limitations of specific perspectives as applied to particular generative models, researchers can develop methods that are better suited to capture
and respect the complexity of cognitive processes, while also providing useful
frameworks for problem-solving and prediction.

### 5.3 Some future research directions

Various future research directions have emerged related to GNN, and will be the
subject of ongoing work at the Active Inference Institute and elsewhere. Some
salient directions are briefly listed here. In terms of realizing the utility of GNN,
improved automatic rendering software and better integration with the Active
Inference Ontology [9] will enable the ”Triple Play” outlined above. Complex
Systems Engineering frameworks such as cadCAD will be useful for specifying
the execution order of GNN expressions [6] as well as for mapping the complete
landscape of the particular model in question (parameter sweeps across the statespace/belief configurations), for example in the context of cyberphysical systems.
From a linguistics perspective it would be interesting to explore possible grammatical case systems, morphology, and dialects associated with GNN. Better
integration with natural language processing and formal semiotic methods will
enable new kinds of analyses on, with, and for the generative models described
by GNN.

### 5.4 Final thoughts on GNN and Beyond

By adhering to scientific principles such as rigor, accessibility, and plurality,
Active Inference researchers can develop and use versatile methods that point
towards the intricacies of cognitive processes while also providing practical solutions to real-world problems. As the field continues to evolve, it is crucial to
remain open to new ideas, methodologies, action policies, and inferred beliefs, in
order to foster a more comprehensive understanding of minds and their workings
[1].

## References

1. Friston, KJ.: The free-energy principle: a unified brain theory?. Nature Reviews
Neuroscience 11, pages 127–138 (2010) <https://doi.org/10.1038/nrn2787>
2. Parr, T., Pezzulo, G., Friston, KJ.: Active Inference: The Free Energy Principle in
Mind, Brain, and Behavior. MIT Press (2022) 9780262045353
3. Smith, R., Friston, KJ, Whyte, CJ: A step-by-step tutorial on active inference and
its application to empirical data. Journal of Mathematical Psychology, Volume 107
(2022) <https://doi.org/10.1016/j.jmp.2021.102632>

4. Smith, R., Whyte, CJ., Murphy, M., Friedman, DA: ActInf ModelStream 001.1,
001.2, 001.3, 001.4: ”A Step-by-Step Tutorial on Active Inference”. Active Inference
Journal (2022) <https://doi.org/10.5281/zenodo.7452789>
5. Friston, KJ., Ramstead, MJD, Kiefer, AB, Tschantz, A, Buckley CL, et al.: Designing Ecosystems of Intelligence from First Principles. arXiv:2212.01354 (2022)
<https://doi.org/10.48550/arXiv.2212.01354>
6. Friedman, DA, Applegate-Swanson, S, Balbuena, JA, Choudhury, A, Cordes, RJ:
An Active Inference Ontology for Decentralized Science: from Situated Sensemaking
to the Epistemic Commons. Zenodo (2022) <https://doi.org/10.5281/zenodo.7484994>
7. Cloutier, J-F: Towards a symbolic implementation of Active Inference for Lego
robots. Zenodo (2022) <https://doi.org/10.5281/zenodo.6862626>
8. Andrews, M: The math is not the territory: navigating the free energy principle.
Biology Philosophy 36, 30 (2021) <https://doi.org/10.1007/s10539-021-09807-0>
9. Active Inference Institute: Active Inference Ontology. (2022)
<https://doi.org/10.5281/zenodo.7430333>
10. Active Inference Institute: Generalized Notation Notation (GNN) Github
repo: github.com/ActiveInferenceInstitute/Generalized_Notation_Notation
<https://doi.org/10.5281/zenodo.7803314>
11. Active Inference Institute: Generalized Notation Notation (GNN) Coda:
<https://coda.io/@active-inference-institute/generalized-notation-notation>
