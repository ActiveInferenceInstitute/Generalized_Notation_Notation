# dspy

## Overview

This directory contains comprehensive documentation and resources for the DSPy subsystem, providing programmatic LLM integration for GNN (Generalized Notation Notation).

**Status**: ✅ Documentation Module  
**Version**: 2.0  
**Last Updated**: February 2026

---

## Purpose

DSPy integration for structured prompting and LLM coordination within the GNN ecosystem. This subsystem is part of the broader GNN documentation ecosystem, integrated with the 25-step processing pipeline.

---

### Documentation Files

| File | Description | Lines |
|------|-------------|-------|
| [README.md](README.md) | Directory overview and quick start | ~200 |
| [AGENTS.md](AGENTS.md) | Technical scaffolding (this file) | ~150 |
| [gnn_dspy.md](gnn_dspy.md) | Core DSPy-GNN integration theory | ~550 |
| [dspy_modules_reference.md](dspy_modules_reference.md) | Comprehensive module catalog | ~400 |
| [dspy_agents_guide.md](dspy_agents_guide.md) | Building agents with ReAct | ~450 |
| [dspy_optimizers_guide.md](dspy_optimizers_guide.md) | Optimization with MIPROv2 | ~500 |
| [dspy_assertions_guide.md](dspy_assertions_guide.md) | Output validation and constraints | ~400 |
| [dspy_retrieval_guide.md](dspy_retrieval_guide.md) | RAG and vector databases | ~450 |
| [dspy_typed_predictors.md](dspy_typed_predictors.md) | Pydantic and structured output | ~400 |
| [dspy_gnn_integration_patterns.md](dspy_gnn_integration_patterns.md) | Practical integration patterns | ~500 |

---

## Quick Navigation

### This Directory

- **[README.md](README.md)**: Directory overview (start here)
- **[gnn_dspy.md](gnn_dspy.md)**: Complete DSPy-GNN integration guide

### Core DSPy Documentation

- **[dspy_modules_reference.md](dspy_modules_reference.md)**: All DSPy modules
- **[dspy_agents_guide.md](dspy_agents_guide.md)**: Agent development with ReAct
- **[dspy_optimizers_guide.md](dspy_optimizers_guide.md)**: MIPROv2, BootstrapFinetune

### Advanced Topics

- **[dspy_assertions_guide.md](dspy_assertions_guide.md)**: Output validation
- **[dspy_retrieval_guide.md](dspy_retrieval_guide.md)**: RAG and retrieval
- **[dspy_typed_predictors.md](dspy_typed_predictors.md)**: Structured output
- **[dspy_gnn_integration_patterns.md](dspy_gnn_integration_patterns.md)**: GNN patterns

### Related Documentation

- **[GNN Documentation](../../gnn/AGENTS.md)**: Core GNN docs
- **[Main Documentation](../README.md)**: Documentation hub
- **[Pipeline Reference](../../../src/gnn/AGENTS.md)**: Pipeline details

---

## Documentation Structure

This module is organized as follows:

```
docs/dspy/
│
├── Overview & Navigation
│   ├── README.md              # Entry point
│   └── AGENTS.md              # This scaffolding
│
├── Core Theory
│   └── gnn_dspy.md            # DSPy-GNN integration theory
│
├── Module Reference
│   └── dspy_modules_reference.md
│
├── Guides
│   ├── dspy_agents_guide.md
│   ├── dspy_optimizers_guide.md
│   ├── dspy_assertions_guide.md
│   ├── dspy_retrieval_guide.md
│   └── dspy_typed_predictors.md
│
└── Integration
    └── dspy_gnn_integration_patterns.md
```

---

## Integration with Pipeline

This documentation is integrated with the 25-step GNN processing pipeline:

### Core Processing (Steps 0-9)

- GNN parsing, validation, export
- DSPy can enhance GNN parsing with LLM assistance
- Structured prompting for model interpretation

### Simulation (Steps 10-16)

- Model execution and analysis
- DSPy-optimized LLM analysis (Step 13: LLM)
- Automated prompt optimization for model interpretation

### Integration (Steps 17-24)

- System coordination and output
- DSPy results integrated into comprehensive outputs
- LLM-enhanced documentation generation

See [src/gnn/AGENTS.md](../../../src/gnn/AGENTS.md) for complete pipeline documentation.

---

## Key Topics Covered

### DSPy Fundamentals

- Signatures and declarative specifications
- Modules (Predict, ChainOfThought, ReAct, etc.)
- Optimizers (MIPROv2, BootstrapFewShot, etc.)
- Configuration and LM providers

### GNN Integration

- Observation processing with DSPy modules
- Policy evaluation using semantic understanding
- GNN model authoring assistance
- Explanation generation for Active Inference

### Advanced Patterns

- Multi-hop reasoning architectures
- Typed outputs with Pydantic
- RAG with ColBERT and vector databases
- Agent development with tools

---

## Standards and Guidelines

All documentation in this module adheres to professional standards:

- **Clarity**: Concrete, technical writing
- **Functionality**: Describes actual capabilities
- **Completeness**: Comprehensive coverage
- **Consistency**: Uniform structure and style
- **Code Examples**: Syntactically correct, tested

---

## Related Resources

### Main GNN Documentation

- [GNN Overview](../../gnn/gnn_overview.md)
- [GNN Quickstart](../../gnn/tutorials/quickstart_tutorial.md)
- [GNN Examples](../../gnn/tutorials/gnn_examples_doc.md)

### Pipeline Architecture

- [Pipeline AGENTS](../../../src/gnn/AGENTS.md)
- [Pipeline README](../../../src/gnn/README.md)

### External Resources

- [DSPy Official](https://dspy.ai)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
