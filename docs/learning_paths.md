# GNN Learning Paths

> **Role**: Step-by-step curricula (beginner → advanced). **Hub**: [README.md](README.md). **Flat index**: [INDEX.md](INDEX.md). **Topic graph**: [CROSS_REFERENCE_INDEX.md](CROSS_REFERENCE_INDEX.md).

This document outlines structured learning paths for users of varying expertise levels in Generalized Notation Notation (GNN) and Active Inference. Each path includes key resources, prerequisites, and progression steps. Paths are designed to be modular, example-driven, and tied to the project's pipeline for reproducibility.

**See also**: [docs/SPEC.md](SPEC.md) (how `docs/` versioning relates to the GNN language and the Python package), [CLAUDE.md](../CLAUDE.md) (commands and measured test expectations).

## Research-focused path

Theory-first track: see [Advanced Path: Research and Custom Extensions](#advanced-path-research-and-custom-extensions) and the [Research](CROSS_REFERENCE_INDEX.md#research) entry in the cross-reference index.

## Developer-focused path

Implementation focus: [Intermediate Path](#intermediate-path-building-and-integrating-models), [Development Guide](development/README.md), and [Practitioner Path](CROSS_REFERENCE_INDEX.md#practitioner-path).

## Academic learning path

Structured progression: [Beginner Path](#beginner-path-getting-started-with-gnn) → [Intermediate](#intermediate-path-building-and-integrating-models) → [Advanced](#advanced-path-research-and-custom-extensions).

## Quick exploration path

Fast orientation: [Quickstart](quickstart.md), [INDEX](INDEX.md), and [CROSS_REFERENCE_INDEX](CROSS_REFERENCE_INDEX.md).

## Beginner path

Same content as [Beginner Path: Getting Started with GNN](#beginner-path-getting-started-with-gnn).

## Intermediate path

Same content as [Intermediate Path: Building and Integrating Models](#intermediate-path-building-and-integrating-models).

## Beginner Path: Getting Started with GNN

**Target Audience**: New users with basic programming knowledge (Python/Julia) but no Active Inference experience.

1. **Introduction to GNN**:
   - Read [GNN Overview](gnn/gnn_overview.md) for core concepts.
   - Review [About GNN](gnn/about_gnn.md) for project context.

2. **Quickstart Tutorial**:
   - Follow [Quickstart](quickstart.md) to set up your environment (see [SETUP.md](SETUP.md)).
   - Build your first model using [Basic GNN Template](templates/basic_gnn_template.md).

3. **Basic Syntax and Examples**:
   - Study [GNN Syntax](gnn/reference/gnn_syntax.md).
   - Explore simple examples in [GNN Examples](gnn/tutorials/gnn_examples_doc.md).

4. **Run Your First Pipeline**:
   - Execute via `src/gnn/main.py` (details in [Pipeline Architecture](gnn/reference/architecture_reference.md)).
   - Visualize results (see [Visualization Docs](visualization/README.md)).

**Next Steps**: Move to Intermediate Path once comfortable with basic models.

### Beginner Skill Checkpoints

- [ ] Can you explain the "triple play" approach?
- [ ] Have you successfully generated visualization output for a basic model?
- [ ] Can you identify the difference between `s_f0` and `o_m0` in a GNN file?

## Intermediate Path: Building and Integrating Models

**Target Audience**: Users familiar with GNN basics, seeking integrations and advanced patterns.

**Prerequisites**: Complete Beginner Path.

1. **Advanced Modeling**:
   - Dive into [Advanced Modeling Patterns](gnn/advanced/advanced_modeling_patterns.md).
   - Run continuous linear-Gaussian models: [`input/gnn_files/continuous/`](../input/gnn_files/continuous/) on JAX, NumPyro, PyTorch, Stan, RxInfer.jl, or cpomdp (see [continuous LGSSM patterns](gnn/advanced/advanced_modeling_patterns.md#9-continuous-linear-gaussian-models)).
   - Learn multi-agent systems in [GNN Multiagent](gnn/advanced/gnn_multiagent.md).

2. **Integrations**:
   - PyMDP: [GNN PyMDP](pymdp/gnn_pymdp.md).
   - RxInfer: [GNN RxInfer](rxinfer/gnn_rxinfer.md).
   - DisCoPy: [GNN DisCoPy](discopy/gnn_discopy.md).

3. **Tools and APIs**:
   - Use MCP tools ([MCP Docs](mcp/gnn_mcp_model_context_protocol.md)).
   - Explore LLM enhancements ([LLM Integration](llm/README.md)).
   - Visualization tools: [GUI Oxdraw](gui_oxdraw/gnn_oxdraw.md) for visual model construction.
   - Analysis tools: [OneFileLLM](other/onefilellm/onefilellm_gnn.md), [Vec2Text](other/vec2text/vec2text_gnn.md).

4. **Testing and Troubleshooting**:
   - Run tests ([Testing Docs](testing/README.md)).
   - Handle errors ([Common Errors](troubleshooting/common_errors.md)).

**Next Steps**: Proceed to Advanced Path for research-level applications.

### Intermediate Skill Checkpoints

- [ ] Have you modified a POMDP template for a custom domain?
- [ ] Can you run the same GNN model across two different frameworks (e.g., PyMDP and RxInfer)?
- [ ] Have you run a continuous (`F`/`H`/`Q`/`R`) model from `input/gnn_files/continuous/` on a Gaussian-capable backend?
- [ ] Do you understand how to use the MCP tools to query your model structure?

## Advanced Path: Research and Custom Extensions

**Target Audience**: Experienced researchers extending GNN for novel Active Inference applications.

**Prerequisites**: Complete Intermediate Path.

1. **Domain-Specific Applications**:
   - Cognitive Phenomena: Explore subdirs in [/docs/cognitive_phenomena/](cognitive_phenomena/README.md) (e.g., [Meta-Awareness](cognitive_phenomena/meta-awareness/)).
   - Ontology: [Ontology System](gnn/advanced/ontology_system.md).

2. **Custom Development**:
   - Extend pipeline ([Development Docs](development/README.md)).
   - Integrate with external tools (e.g., [SymPy](sympy/gnn_sympy.md), [DSPy](other/dspy/gnn_dspy.md)).
   - Formal methods: [Axiom](other/axiom/axiom_gnn.md), [Petri Nets](petri_nets/README.md), [Nock](other/nock/nock-gnn.md).
   - Audio processing: [SAPF](sapf/sapf_gnn.md), [Pedalboard](pedalboard/pedalboard_gnn.md).
   - Visualization: [D2 Diagramming](d2/gnn_d2.md), [Glowstick](other/glowstick/glowstick_gnn.md), [GUI Oxdraw](gui_oxdraw/gnn_oxdraw.md).

3. **Performance and Optimization**:
   - Metrics: [Resource Metrics](gnn/operations/resource_metrics.md).
   - Optimization: [Performance Guide](pymdp/pymdp_performance_guide.md).

4. **Contribution and Research**:
   - Read [GNN Paper](gnn/gnn_paper.md).
   - Contribute via [Releases](releases/README.md) and [Security Framework](security/security_framework.md).

### Advanced Skill Checkpoints

- [ ] Have you implemented a custom cognitive phenomenon model using GNN?
- [ ] Can you explain the categorical foundations of your model using the DisCoPy output?
- [ ] Have you integrated a new scientific library into the GNN pipeline?

**Additional Resources**:

- Full index: [Cross-Reference Index](CROSS_REFERENCE_INDEX.md).
- Style Guide: [Style Guide](style_guide.md) for contributions.
- For updates, check project [README](../README.md) and CHANGELOG.

These paths emphasize hands-on examples and reproducibility, aligning with GNN's scientific standards.
