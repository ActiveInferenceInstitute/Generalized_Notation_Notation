# ActiveInference.jl Documentation Agents

## Overview

This document provides agent scaffolding for the ActiveInference.jl documentation subsystem. The documentation is organized into specialized agent modules, each responsible for specific aspects of the ActiveInference.jl system.

**Status**: ✅ Documentation  
**Version**: 1.0  

---

## Purpose

Documentation for the Julia implementation of Active Inference algorithms and frameworks.

**Coverage**: The GNN ActiveInference.jl renderer supports **discrete
categorical models only** (`A`/`B`/`C`/`D[/E]`, multi-modality and
multi-factor supported; continuous linear-Gaussian models are not supported —
`supports_continuous=False`). Rendered scripts are **executable at Step 12**
via `src/gnn/execute/activeinference_jl/`.

## Documentation Organization

The ActiveInference.jl documentation system is part of the broader GNN (Generalized Notation Notation) pipeline and ecosystem.

### Related Systems
- **Main Documentation**: [docs/README.md](../README.md)
- **GNN Documentation**: [gnn/AGENTS.md](../gnn/AGENTS.md)
- **Pipeline Reference**: [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md)
- **Active Inference**: [activeinference.org](https://activeinference.org)

## Subsystems

- **activeinference-jl.md**: Main ActiveInference.jl implementation guide
- **activeinference-jl_source_code.md**: Complete source code documentation
- **enhanced_analysis_documentation.md**: Advanced analysis and features
- **actinf_jl_src/**: Julia source code examples and implementations

## Quick Navigation

- **Get Started**: See [README.md](README.md)
- **Pipeline Overview**: [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md)
- **Full Documentation Index**: [docs/README.md](../README.md)

## Documentation Standards

All documentation in this subsystem follows the professional standards established in the GNN documentation system:

- **Clarity**: Concrete examples, technical accuracy
- **Functionality**: Shows what code actually does
- **Evidence-Based**: Specific metrics and real examples
- **Professional**: Clear structure, no promotional language

## Integration

This documentation integrates with the 25-step GNN processing pipeline:

1. **Pipeline Initialization** (Step 0): Template setup
2. **Environment Setup** (Step 1): Dependency installation
3. **Testing** (Step 2): Quality assurance
4. **Core Processing** (Steps 3-9): GNN parsing and validation
5. **Simulation** (Steps 10-16): Model execution and analysis
6. **Integration** (Steps 17-24): Output and reporting

See [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md) for complete pipeline documentation.
