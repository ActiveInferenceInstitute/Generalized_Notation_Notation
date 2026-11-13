# SPM Documentation Agent

> **📋 Document Metadata**  
> **Type**: Neuroscientific Integration Agent | **Audience**: Neuroscientists, AI Researchers | **Complexity**: Advanced  
> **Cross-References**: [README.md](README.md) | [SPM GNN Guide](spm_gnn.md) | [Cognitive Phenomena](../cognitive_phenomena/README.md) | [Main Documentation](../README.md)

## Overview

This directory contains comprehensive documentation, resources, and implementation guides for integrating **SPM** (Statistical Parametric Mapping) with GNN (Generalized Notation Notation). SPM provides established statistical frameworks for neuroimaging analysis, enabling translation of neuroimaging insights into computational cognitive architectures.

> **Scope note**: SPM is documented here as a research/integration-notes capability. It is **not** a render or execution framework — it is not an entry in `src/gnn/render/framework_registry.py`, and no SPM code exists under `src/gnn/`. Pipeline references below describe attachment points a future integration could take; nothing here is wired into the pipeline today.

**Status**: Documentation module — research/integration notes (no SPM implementation in `src/gnn/`)  
**Version**: 1.0

## Purpose

SPM integration enables:

- **Brain-Inspired AI**: Translation of neuroimaging insights to computational models
- **Model Calibration**: SPM results inform GNN state space design
- **Connectivity Mapping**: DCM connectivity matrices guide GNN transition models
- **Temporal Dynamics**: SPM temporal dynamics constrain GNN time horizons
- **Population-Level Inference**: Group statistics for model validation

## Contents

Browse the [current directory](./) for files and subdirectories; use the task links below for navigation.

## Quick Navigation

- **README.md**: [Directory overview](README.md)
- **GNN Documentation**: [gnn/AGENTS.md](../gnn/AGENTS.md)
- **Main Documentation**: [docs/README.md](../README.md)
- **Pipeline Reference**: [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md)

## Integration with Pipeline

This documentation references the 25-step GNN processing pipeline at the conceptual level:

The integration pathways below are **proposals**: no code realizes any of them today. No
pipeline step invokes SPM today (no SPM code under `src/gnn/`; SPM is not in
`framework_registry.py`, so there is no Step 11 render target and no Step 12
executor for it):

- **Step 3 (GNN)**: where SPM-informed model specification *could* attach
- **Step 6 (Validation)**: where SPM-based statistical validation *could* attach
- **Step 11 (Render)** / **Step 12 (Execute)**: no SPM target or executor exists
- **Step 16 (Analysis)**: where SPM-style statistical analysis *could* integrate


See [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md) for complete pipeline documentation.

## Function Signatures and API (proposed signatures only)

The signatures below are **illustrative sketches** of a future SPM bridge; no such
functions exist anywhere in the codebase.

### Neuroimaging Analysis Functions

```python
def calibrate_gnn_from_spm(gnn_model: GNNModel, spm_results: SPMResults) -> GNNModel:
    """
    Calibrate GNN model using SPM neuroimaging results.

    Parameters:
        gnn_model: Parsed GNN model structure
        spm_results: SPM statistical analysis results

    Returns:
        Calibrated GNNModel with SPM-informed parameters
    """


def extract_connectivity_matrix(spm_results: SPMResults) -> ConnectivityMatrix:
    """
    Extract connectivity matrix from SPM DCM results.

    Parameters:
        spm_results: SPM analysis results

    Returns:
        ConnectivityMatrix for GNN transition model
    """
```

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

## See Also

- **[SPM Cross-Reference](../CROSS_REFERENCE_INDEX.md#spm)**: Cross-reference index entry
- **[Cognitive Phenomena](../cognitive_phenomena/README.md)**: Cognitive modeling applications
- **[Neuroscience](../CROSS_REFERENCE_INDEX.md#neuroscience)**: Neuroscientific methods
- **[Main Index](../README.md)**: Return to main documentation

---

**Status**: Documentation module — research/integration notes (no SPM implementation in `src/gnn/`)  
**Maintenance**: Regular updates with new SPM features and integration capabilities
