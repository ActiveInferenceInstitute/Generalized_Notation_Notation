# GNN Documentation Index

**Version**: v3.6.0 Engine (Bundle v2.0.0)
**Last Updated**: 2026-09-23
**Status**: Maintained
**Pipeline Steps**: 25

Complete navigation guide for all GNN (Generalized Notation Notation) documentation. For the machine-oriented tree manifest (directory counts and new-module index), see **[AGENTS.md](AGENTS.md)**. Parent doc hub: [docs/README.md](../README.md), [docs/INDEX.md](../INDEX.md). **Versioning policy** (GNN language vs doc bundle vs Python package): [docs/SPEC.md](../SPEC.md).

## 🚀 Quick Start

**New to GNN?** Start here:

1. [GNN Overview](gnn_overview.md) - Introduction to GNN and its purpose
2. [Quickstart Tutorial](tutorials/quickstart_tutorial.md) - Build your first model in 15 minutes
3. [GNN File Structure](reference/gnn_file_structure_doc.md) - Understanding GNN file organization

## 📋 Pipeline Documentation

**GNN Processing Pipeline** (25 Steps, 0-24):

**Core Documentation:**

- **[src/gnn/AGENTS.md](../../src/gnn/AGENTS.md)** - Master agent scaffolding and complete 25-step registry
- **[src/gnn/README.md](../../src/gnn/README.md)** - Pipeline architecture and thin orchestrator pattern
- **[src/gnn/main.py](../../src/gnn/main.py)** - Pipeline orchestrator implementation
- [Architecture Reference](reference/architecture_reference.md) - Implementation patterns and cross-module data flow
- [Module Reference](modules/README.md) - Per-module documentation for all 25 steps, with architecture, CLI args, and output details
- [GNN Tools and Resources](operations/gnn_tools.md) - Complete pipeline usage and examples
- [Technical Reference](reference/technical_reference.md) - Round-trip data flow and entry points
- **[Testing Hub](testing/README.md)** — Test architecture, patterns, MCP audit (see root [README.md](../../README.md) for latest `pytest` counts)

**Quick Start:**

```bash
# Run full pipeline
uv run python src/gnn/main.py --target-dir input/gnn_files --verbose

# Run specific steps
uv run python src/gnn/main.py --only-steps "3,5,8,11,12" --verbose
```

## 📖 Language Specification

**GNN Syntax and Structure:**

- **[Language Hub](language/README.md)** — Syntax cheatsheet, variable grammar, connection grammar
- **[GNN v1.1 Syntax Specification](gnn_syntax.md)** — Normative spec for sections, parsers, and validators (living document)
- [GNN Syntax Reference](reference/gnn_syntax.md) — Examples, validation commands, and quick patterns (read with the spec above)
- [GNN DSL Manual](reference/gnn_dsl_manual.md) - Domain-Specific Language specification
- [GNN Schema Specification](reference/gnn_schema.md) - Parsing and validation schemas
- [GNN File Structure](reference/gnn_file_structure_doc.md) - File organization and sections
- [GNN Standards](reference/gnn_standards.md) - Domain knowledge and coding standards
- [About GNN](about_gnn.md) - Detailed GNN specification
- [GNN Troubleshooting](operations/gnn_troubleshooting.md) - Common issues and solutions

The keys a model declares in `InitialParameterization` decide which **model
kind** the renderers dispatch on — discrete `A`/`B`/`C`/`D`/`E`,
linear-Gaussian `F`/`H`/`Q`/`R`, `dirichlet_*` pseudo-counts, or per-level and
per-agent suffixes. Detection rules and precedence are specified in
[gnn_syntax.md § Parameterization families](gnn_syntax.md#parameterization-families);
the per-kind rendering strategies are documented in
[src/gnn/render/rxinfer/README.md](../../src/gnn/render/rxinfer/README.md).

## 🎯 Modeling and Examples

**Learning to Model:**

- [Quickstart Tutorial](tutorials/quickstart_tutorial.md) - Step-by-step first model
- [GNN Examples](tutorials/gnn_examples_doc.md) - Model progression from simple to complex
- [Advanced Modeling Patterns](advanced/advanced_modeling_patterns.md) - Hierarchical and sophisticated techniques
- [Multi-Agent Systems](advanced/gnn_multiagent.md) - Multi-agent modeling specification
- [Continuous Linear-Gaussian Models](advanced/advanced_modeling_patterns.md#9-continuous-linear-gaussian-models) - LGSSM kind (`F`/`H`/`Q`/`R`) with runnable exemplars in [`input/gnn_files/continuous/`](../../input/gnn_files/continuous/)

**Cognitive Phenomena:**

- [Cognitive Phenomena Models](../cognitive_phenomena/) - Attention, memory, learning, emotion models

## 🔧 Implementation and Integration

**Framework Integration:**

- **[MCP Hub](mcp/README.md)** — tool inventory for the **whole pipeline** MCP surface (order-of-magnitude 130+ tools; not Step 3 only), client setup, tool development guide
- [Framework Integration Guide](integration/framework_integration_guide.md) - PyMDP, RxInfer, ActiveInference.jl, DisCoPy, JAX, PyTorch, NumPyro, Stan
- [GNN Implementation Guide](integration/gnn_implementation.md) - Implementation workflows and patterns
- **[Framework Implementations](implementations/README.md)** — Framework documentation for PyMDP, JAX, RxInfer, ActiveInference.jl, DisCoPy, PyTorch, NumPyro, Stan, and CatColab
- [PyMDP Integration](../pymdp/gnn_pymdp.md) - Python POMDP framework
- [RxInfer Integration](../rxinfer/gnn_rxinfer.md) - Julia Bayesian inference
- [ActiveInference.jl Integration](../activeinference_jl/activeinference-jl.md) - Julia Active Inference
- [DisCoPy Integration](../discopy/gnn_discopy.md) - Category theory and string diagrams
- [CatColab Integration](../other/catcolab/catcolab_gnn.md) - Categorical compositional modeling

## 🧠 Advanced Topics

**AI and Analysis:**

- [LLM and Neurosymbolic Active Inference](advanced/gnn_llm_neurosymbolic_active_inference.md) - LLM integration (Step 13)
- [Ontology System](advanced/ontology_system.md) - Active Inference ontology annotations (Step 10)
- [Resource Metrics](operations/resource_metrics.md) - Computational resource estimation (Step 5)

**Quality and Improvement:**

- [Improvement Analysis](operations/improvement_analysis.md) - Pipeline improvement opportunities
- [REPO Coherence Check](operations/REPO_COHERENCE_CHECK.md) - Quality standards and validation

## 📊 Complete Module Registry

### Core Processing (Steps 0-9)

| Step | Script | Module | Purpose |
| ---- | ------ | ------ | ------- |
| 0 | `0_template.py` | [template/](../../src/gnn/template/AGENTS.md) | Pipeline initialization |
| 1 | `1_setup.py` | [setup/](../../src/gnn/setup/AGENTS.md) | Environment setup |
| 2 | `2_tests.py` | [tests/](../../tests/AGENTS.md) | Test suite |
| 3 | `3_gnn.py` | [gnn/](../../src/gnn/AGENTS.md) | GNN parsing |
| 4 | `4_model_registry.py` | [model_registry/](../../src/gnn/model_registry/AGENTS.md) | Model versioning |
| 5 | `5_type_checker.py` | [type_checker/](../../src/gnn/type_checker/AGENTS.md) | Type validation |
| 6 | `6_validation.py` | [validation/](../../src/gnn/validation/AGENTS.md) | Consistency checking |
| 7 | `7_export.py` | [export/](../../src/gnn/export/AGENTS.md) | Multi-format export |
| 8 | `8_visualization.py` | [visualization/](../../src/gnn/visualization/AGENTS.md) | Graph visualization |
| 9 | `9_advanced_viz.py` | [advanced_visualization/](../../src/gnn/advanced_visualization/AGENTS.md) | Advanced plots |

### Simulation and Analysis (Steps 10-16)

| Step | Script | Module | Purpose |
| ---- | ------ | ------ | ------- |
| 10 | `10_ontology.py` | [ontology/](../../src/gnn/ontology/AGENTS.md) | Ontology processing |
| 11 | `11_render.py` | [render/](../../src/gnn/render/AGENTS.md) | Code generation |
| 12 | `12_execute.py` | [execute/](../../src/gnn/execute/AGENTS.md) | Simulation execution |
| 13 | `13_llm.py` | [llm/](../../src/gnn/llm/AGENTS.md) | Neurosymbolic LLM Analysis |
| 14 | `14_ml_integration.py` | [ml_integration/](../../src/gnn/ml_integration/AGENTS.md) | ML integration |
| 15 | `15_audio.py` | [audio/](../../src/gnn/audio/AGENTS.md) | Audio generation |
| 16 | `16_analysis.py` | [analysis/](../../src/gnn/analysis/AGENTS.md) | Statistical analysis |

### Integration and Output (Steps 17-24)

| Step | Script | Module | Purpose |
| ---- | ------ | ------ | ------- |
| 17 | `17_integration.py` | [integration/](../../src/gnn/integration/AGENTS.md) | System integration |
| 18 | `18_security.py` | [security/](../../src/gnn/security/AGENTS.md) | Security validation |
| 19 | `19_research.py` | [research/](../../src/gnn/research/AGENTS.md) | Research tools |
| 20 | `20_website.py` | [website/](../../src/gnn/website/AGENTS.md) | Website generation |
| 21 | `21_mcp.py` | [mcp/](../../src/gnn/mcp/AGENTS.md) | MCP processing |
| 22 | `22_gui.py` | [gui/](../../src/gnn/gui/AGENTS.md) | GUI interface |
| 23 | `23_report.py` | [report/](../../src/gnn/report/AGENTS.md) | Report generation |
| 24 | `24_intelligent_analysis.py` | [intelligent_analysis/](../../src/gnn/intelligent_analysis/AGENTS.md) | Intelligent Heuristic Recovery |

## 🔍 Find What You Need

### By Task

**I want to...**

- **Learn GNN basics** → [GNN Overview](gnn_overview.md) → [Quickstart Tutorial](tutorials/quickstart_tutorial.md)
- **Understand syntax** → [GNN Syntax](reference/gnn_syntax.md) → [GNN DSL Manual](reference/gnn_dsl_manual.md)
- **Build a model** → [Quickstart Tutorial](tutorials/quickstart_tutorial.md) → [GNN Examples](tutorials/gnn_examples_doc.md)
- **Use advanced patterns** → [Advanced Modeling Patterns](advanced/advanced_modeling_patterns.md)
- **Integrate frameworks** → [Framework Integration Guide](integration/framework_integration_guide.md)
- **Process models** → [GNN Tools](operations/gnn_tools.md) → [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md)
- **Understand pipeline** → [src/gnn/README.md](../../src/gnn/README.md) → [Architecture Reference](reference/architecture_reference.md)
- **Debug issues** → [Technical Reference](reference/technical_reference.md) → [Improvement Analysis](operations/improvement_analysis.md)

### By Audience

**Beginners:**

1. [GNN Overview](gnn_overview.md)
2. [Quickstart Tutorial](tutorials/quickstart_tutorial.md)
3. [GNN Examples](tutorials/gnn_examples_doc.md)
4. [GNN Syntax](reference/gnn_syntax.md)

**Researchers:**

1. [Advanced Modeling Patterns](advanced/advanced_modeling_patterns.md)
2. [Multi-Agent Systems](advanced/gnn_multiagent.md)
3. [LLM and Neurosymbolic AI](advanced/gnn_llm_neurosymbolic_active_inference.md)
4. [Ontology System](advanced/ontology_system.md)

**Developers:**

1. [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md)
2. [Architecture Reference](reference/architecture_reference.md)
3. [Technical Reference](reference/technical_reference.md)
4. [Framework Integration Guide](integration/framework_integration_guide.md)
5. [GNN Implementation Guide](integration/gnn_implementation.md)

**System Architects:**

1. [src/gnn/README.md](../../src/gnn/README.md)
2. [Architecture Reference](reference/architecture_reference.md)
3. [GNN Standards](reference/gnn_standards.md)
4. [Improvement Analysis](operations/improvement_analysis.md)
5. [REPO Coherence Check](operations/REPO_COHERENCE_CHECK.md)

## 🛠️ Additional Resources

**Related Documentation:**

- [GNN Troubleshooting](operations/gnn_troubleshooting.md) - Common issues and solutions
- [API Reference](../api/) - API documentation
- [Configuration Guide](../configuration/) - Configuration options
- [Troubleshooting](../troubleshooting/) - Common issues and solutions
- [Development Guide](../development/) - Development standards

**External Links:**

- [GitHub Repository](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation)
- [Active Inference Institute](https://activeinference.org)
- [Community Discussions](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/discussions)

## 📝 Documentation Standards

All GNN documentation follows these principles:

- **Understated**: Concrete examples over promotional language
- **Show Not Tell**: Working code and real outputs
- **Evidence-Based**: Specific metrics and measurable results
- **Functional**: Emphasis on what the code actually does
- **Professional**: Clear, technical, and precise

## 📋 Documentation Status

- Documentation covers all pipeline steps and major module groups.
- Framework docs cover PyMDP, RxInfer, JAX, NumPyro, Stan, PyTorch, ActiveInference.jl, and DisCoPy.
- CLI and API entrypoints are documented with links to module-level references.
- Cross-references are maintained across language, integration, operations, and tutorial sections.

## 📖 Document Index (Alphabetical)

- [About GNN](about_gnn.md)
- [Advanced Modeling Patterns](advanced/advanced_modeling_patterns.md)
- [Architecture Reference](reference/architecture_reference.md)
- [Framework Integration Guide](integration/framework_integration_guide.md)
- [GNN Connection Grammar](language/gnn_connection_grammar.md)
- [GNN DSL Manual](reference/gnn_dsl_manual.md)
- [GNN Examples](tutorials/gnn_examples_doc.md)
- [GNN Export Guide](integration/gnn_export.md)
- [GNN File Structure](reference/gnn_file_structure_doc.md)
- [GNN Implementation Guide](integration/gnn_implementation.md)
- [GNN LLM and Neurosymbolic AI](advanced/gnn_llm_neurosymbolic_active_inference.md)
- [GNN Multi-Agent](advanced/gnn_multiagent.md)
- [GNN Ontology Guide](advanced/gnn_ontology.md)
- [GNN Overview](gnn_overview.md)
- [GNN Paper](gnn_paper.md)
- [GNN Schema](reference/gnn_schema.md)
- [GNN Standards](reference/gnn_standards.md)
- [GNN Syntax Reference](reference/gnn_syntax.md)
- [GNN Syntax Quick Reference](language/gnn_syntax_quickref.md)
- [GNN Tools and Resources](operations/gnn_tools.md)
- [GNN v1.1 Syntax Specification](gnn_syntax.md) — normative
- [GNN Variable Grammar](language/gnn_variable_grammar.md)
- [GNN Troubleshooting](operations/gnn_troubleshooting.md)
- [GNN Type System](reference/gnn_type_system.md)
- [GNN Visualization Guide](integration/gnn_visualization.md)
- [Improvement Analysis](operations/improvement_analysis.md)
- [Ontology System](advanced/ontology_system.md)
- [Quickstart Tutorial](tutorials/quickstart_tutorial.md)
- [REPO Coherence Check](operations/REPO_COHERENCE_CHECK.md)
- [Resource Metrics](operations/resource_metrics.md)
- [Technical Reference](reference/technical_reference.md)

---

**GNN documentation bundle**: v3.6.0 Engine (Bundle v2.0.0) (this tree and front matter on major pages)
**GNN syntax standard**: v1.1 (see [gnn_syntax.md](gnn_syntax.md))
**Python package** (`generalized-notation-notation`): **3.6.0** ([pyproject.toml](../../pyproject.toml))
**Total Pipeline Steps**: 25 (0–24)
**Last Updated**: 2026-09-23

**Status**: Maintained (re-run `uv run --extra dev python docs/development/docs_audit.py` after link or tree changes)
