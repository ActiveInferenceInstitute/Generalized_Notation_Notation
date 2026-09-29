# SPM Integration for GNN

> **📋 Document Metadata**  
> **Type**: Neuroscientific Integration Guide | **Audience**: Neuroscientists, AI Researchers | **Complexity**: Advanced  
> **Cross-References**: [AGENTS.md](AGENTS.md) | [SPM GNN Guide](spm_gnn.md) | [SPM Overview](spm.md) | [Cognitive Phenomena](../cognitive_phenomena/README.md) | [Main Documentation](../README.md)

## Overview

This directory contains documentation, resources, and implementation guides for integrating **SPM** (Statistical Parametric Mapping) with GNN (Generalized Notation Notation). SPM provides established statistical frameworks for neuroimaging analysis, enabling translation of neuroimaging insights into computational cognitive architectures.

> **Scope note**: SPM is a *research/integration-notes* capability documented here. It is **not** a render or execution framework — it does not appear in `src/gnn/render/framework_registry.py` (whose 10 render frameworks are pymdp, rxinfer, activeinference_jl, jax, discopy, pytorch, numpyro, cpomdp, stan, and bnlearn) — and there is no SPM implementation code under `src/gnn/`. The documents in this directory analyze how SPM results could inform GNN model specification and validation.

**Status**: Documentation module — research/integration notes (no SPM implementation in `src/gnn/`)  
**Version**: 1.0

## Quick Navigation

### This Directory
- **[README.md](README.md)**: Directory overview (this file)
- **[AGENTS.md](AGENTS.md)**: Technical documentation and agent scaffolding
- **[spm_gnn.md](spm_gnn.md)**: Complete SPM-GNN integration guide
- **[spm.md](spm.md)**: SPM framework overview

### Main Documentation
- **[docs/README.md](../README.md)**: Main documentation hub
- **[CROSS_REFERENCE_INDEX.md](../CROSS_REFERENCE_INDEX.md)**: Complete cross-reference index
- **[learning_paths.md](../learning_paths.md)**: Learning pathways

### Related Directories
- **[Cognitive Phenomena](../cognitive_phenomena/README.md)**: Cognitive modeling applications
- **[Neuroscience](../CROSS_REFERENCE_INDEX.md#neuroscience)**: Neuroscientific methods
- **[Research Tools](../research/README.md)**: Research workflow tools

### Pipeline Integration
- **[Pipeline Documentation](../gnn/operations/gnn_tools.md)**: Complete pipeline guide
- **[src/gnn/AGENTS.md](../../src/gnn/AGENTS.md)**: Implementation details

## Contents

**Files**: 3 | **Subdirectories**: 0

### Core Files

- **`spm_gnn.md`**: Complete SPM-GNN integration guide
  - Statistical Parametric Mapping integration
  - Neuroimaging analysis frameworks
  - Brain-inspired AI development
  - Cognitive architecture translation

- **`spm.md`**: SPM framework overview
  - SPM statistical frameworks
  - Neuroimaging analysis methods
  - Brain region identification

- **`AGENTS.md`**: Technical documentation and agent scaffolding
  - Complete documentation structure
  - Integration with pipeline
  - Cross-references and navigation

- **`README.md`**: Directory overview (this file)

## SPM Overview

SPM provides:

### Neuroimaging Analysis
- **Mass Univariate Analysis**: Statistical analysis at each voxel
- **General Linear Models**: GLMs for hypothesis testing
- **Random Field Theory**: Multiple comparisons correction
- **Dynamic Causal Modeling**: Effective connectivity analysis

### Key Features
- **Brain Region Identification**: Statistical activation maps
- **Temporal Dynamics**: Event-related designs
- **Connectivity Analysis**: DCM and functional connectivity
- **Population-Level Inference**: Group statistics

## Integration with GNN

SPM integration enables:

- **Brain-Inspired AI**: Translation of neuroimaging insights to computational models
- **Model Calibration**: SPM results inform GNN state space design
- **Connectivity Mapping**: DCM connectivity matrices guide GNN transition models
- **Temporal Dynamics**: SPM temporal dynamics constrain GNN time horizons

## Integration with Pipeline

This directory documents SPM at the *conceptual and research* level only. No pipeline
step invokes SPM today: there is no SPM code under `src/gnn/`, no SPM render target
in `framework_registry.py`, and no SPM Step 12 executor. The integration pathways
described in [spm_gnn.md](spm_gnn.md) are proposals for how SPM analysis results
(activation maps, DCM connectivity, temporal dynamics) could inform GNN model
specification and validation.


See [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md) for complete pipeline documentation.

## Related Resources

### Main GNN Documentation
- **[GNN Overview](../gnn/gnn_overview.md)**: Core GNN concepts
- **[GNN Quickstart](../gnn/tutorials/quickstart_tutorial.md)**: Getting started guide
- **[Cognitive Phenomena](../cognitive_phenomena/README.md)**: Cognitive modeling applications

### Neuroscientific Resources
- **[Cognitive Phenomena](../cognitive_phenomena/README.md)**: Cognitive modeling applications
- **[Neuroscience](../CROSS_REFERENCE_INDEX.md#neuroscience)**: Neuroscientific methods
- **[Research Tools](../research/README.md)**: Research workflow tools

### Pipeline Architecture
- **[Pipeline Documentation](../gnn/operations/gnn_tools.md)**: Complete pipeline guide
- **[Pipeline AGENTS](../../src/gnn/AGENTS.md)**: Implementation details
- **[Pipeline README](../../src/gnn/README.md)**: Pipeline overview

## Standards and Guidelines

All documentation in this module adheres to professional standards:

- **Clarity**: Concrete, technical writing with neuroscientific foundations
- **Functionality**: Describes actual SPM integration capabilities
- **Completeness**: Comprehensive coverage of neuroimaging integration
- **Consistency**: Uniform structure and style with GNN documentation ecosystem

## See Also

- **[SPM Cross-Reference](../CROSS_REFERENCE_INDEX.md#spm)**: Cross-reference index entry
- **[Cognitive Phenomena](../cognitive_phenomena/README.md)**: Cognitive modeling applications
- **[Neuroscience](../CROSS_REFERENCE_INDEX.md#neuroscience)**: Neuroscientific methods
- **[Main Index](../README.md)**: Return to main documentation

---

**Status**: Documentation module — research/integration notes (no SPM implementation in `src/gnn/`)  
**Compliance**: Professional documentation standards  
**Maintenance**: Regular updates with new SPM features and integration capabilities
