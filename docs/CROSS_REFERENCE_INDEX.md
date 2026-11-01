# GNN Documentation Cross-Reference Index

This page maps the main learning, implementation, and support relationships. It is not
a second pipeline or directory inventory; use [INDEX.md](INDEX.md) for flat links and
[README.md](README.md) for onboarding.

## Learning paths

### Beginner

[GNN Overview](gnn/gnn_overview.md) → [Quickstart Tutorial](gnn/tutorials/quickstart_tutorial.md)
→ [Syntax Reference](gnn/reference/gnn_syntax.md) → [Examples](gnn/tutorials/gnn_examples_doc.md)
→ [Templates](templates/README.md)

### Practitioner

[Syntax Reference](gnn/reference/gnn_syntax.md) → [File Structure](gnn/reference/gnn_file_structure_doc.md)
→ [Implementation Guide](gnn/integration/gnn_implementation.md) →
[Framework Integration](gnn/integration/framework_integration_guide.md) →
[Testing](testing/README.md)

### Developer

[Pipeline Architecture](../src/gnn/README.md) → [Step Index](../src/gnn/STEP_INDEX.md) →
[Pipeline Scripts](PIPELINE_SCRIPTS.md) → [API](api/README.md) →
[Development](development/README.md)

### Researcher

[Active Inference](active_inference/README.md) → [GNN Paper](gnn/gnn_paper.md) →
[Advanced Modeling](gnn/advanced/advanced_modeling_patterns.md) →
[Cognitive Phenomena](cognitive_phenomena/README.md)

## Framework network

All framework-specific paths are mediated by Step 11 rendering and, where an executor
exists, Step 12 execution. See the [framework integration guide](gnn/integration/framework_integration_guide.md)
for the current render/execute split.

- **PyMDP**: [guide](pymdp/gnn_pymdp.md) · [templates](templates/pomdp_template.md)
- **RxInfer.jl**: [guide](rxinfer/gnn_rxinfer.md) · [engineering](rxinfer/engineering_rxinfer_gnn.md)
- **ActiveInference.jl**: [guide](activeinference_jl/activeinference-jl.md)
- **JAX**: [implementation guide](gnn/implementations/jax.md)
- **DisCoPy**: [guide](discopy/gnn_discopy.md)
- **PyTorch**: [implementation guide](gnn/implementations/pytorch.md); manually enabled and not in the default lock
- **NumPyro**: [implementation guide](gnn/implementations/numpyro.md)
- **Stan**: [implementation guide](gnn/implementations/stan.md); executable in Step 12 via the cmdstanpy driver
- **bnlearn**: [renderer inventory](../src/gnn/render/AGENTS.md); manually enabled and not in the default lock
- **Lean**: proof verification via the FEP_Formal bridge (no render target; see [FEP bridge docs](other/fep_lean/README.md))

## Operations network

- [Setup](SETUP.md) → [Configuration](configuration/README.md) → [Pipeline](pipeline/README.md)
- [Framework Availability](execution/FRAMEWORK_AVAILABILITY.md) → [Troubleshooting](troubleshooting/README.md)
- [Testing](testing/README.md) → [Development](development/README.md)
- [Security](security/README.md) → [Deployment](deployment/README.md)

## Source of truth

- Pipeline order: `src/gnn/pipeline/step_registry.py`
- CLI options: `src/gnn/utils/arguments/arg_parsing.py` and `src/gnn/cli/__init__.py`
- Automatic YAML path: `input/config.yaml` and `src/gnn/utils/config_io/config_loader.py`
- Required GNN sections: `src/gnn/schema/parser.py`
- Render inventory: `src/gnn/render/framework_registry.py`
- Execute inventory: `src/gnn/execute/processor.py::parse_frameworks_parameter`

## Inbound topic anchors

Older topic pages link to these anchors. They remain compatibility headings; the
current relationships are documented in the sections above.

### activeinference_jl
### advanced-visualization
### api-reference-integration
### audio-and-sonification
### d2
### discopy
### export
### gui_oxdraw
### research
### practitioner-path
### other
### arc-agi
### autogenlib
### formal-methods-and-verification
### axiom
### cerebrum
### dspy
### glowstick
### distributed-systems-and-networking
### iroh
### kit
### klong
### performance-optimization
### muscle-mem
### nock
### mathematical-foundations
### data-processing
### onefilellm
### poe-world
### spatial-modeling
### quadray
### embedding-systems
### privacy-and-interpretability
### vec2text
### x402
### pkl
### pomdp
### release-management
### rxinferjl
### sapf
### neuroscience
### spm
### sympy
### type-inference-zoo
### visualization
### timep
### pedalboard
### advanced-topics
### configuration-and-performance
### security-and-compliance
### deployment-operations
### troubleshooting-support
### learning-resources
### api-reference-integration
