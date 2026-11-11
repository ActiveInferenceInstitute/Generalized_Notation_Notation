# Pedalboard Integration for GNN

> **📋 Document Metadata**  
> **Type**: Audio Processing Integration Guide | **Audience**: Researchers, Audio Engineers | **Complexity**: Intermediate-Advanced  
> **Cross-References**: [AGENTS.md](AGENTS.md) | [Pedalboard GNN Guide](pedalboard_gnn.md) | [Audio Processing](../audio/README.md) | [SAPF Integration](../sapf/sapf_gnn.md) | [Main Documentation](../README.md)

## Overview

This directory contains documentation, resources, and implementation guides for integrating **Pedalboard** (Spotify's Audio Processing Library) with GNN (Generalized Notation Notation). Pedalboard provides high-performance DSP capabilities, VST3/AU plugin ecosystem, and Python-native API for sophisticated audio representations and real-time sonification of Active Inference models.

**Role**: Pedalboard is an **audio post-processing dependency**, not a render or execution framework. It is not an entry in `src/gnn/render/framework_registry.py`, generates no simulation code, and has no Step 12 executor. In the live tree it is an optional audio backend probed by `src/gnn/audio/__init__.py` and listed in `src/gnn/STEP_INDEX.md` as a Step 15 audio-sonification dependency (alongside soundfile); the Step 15 audio script accepts `--audio-backend` values including `pedalboard`.

**Status**: 🛠️ Scaffold / Planned  
**Version**: 1.0

## Quick Navigation

### This Directory
- **[README.md](README.md)**: Directory overview (this file)
- **[AGENTS.md](AGENTS.md)**: Technical documentation and agent scaffolding
- **[pedalboard_gnn.md](pedalboard_gnn.md)**: Complete Pedalboard-GNN integration guide

### Main Documentation
- **[docs/README.md](../README.md)**: Main documentation hub
- **[CROSS_REFERENCE_INDEX.md](../CROSS_REFERENCE_INDEX.md)**: Complete cross-reference index
- **[learning_paths.md](../learning_paths.md)**: Learning pathways

### Related Directories
- **[Audio Processing](../../src/gnn/audio/README.md)**: Audio generation and sonification
- **[SAPF Integration](../sapf/sapf_gnn.md)**: Sound As Pure Form framework
- **[Sonification](../CROSS_REFERENCE_INDEX.md#audio-and-sonification)**: Audio sonification tools

### Pipeline Integration
- **[Pipeline Documentation](../gnn/operations/gnn_tools.md)**: Complete pipeline guide
- **[src/gnn/AGENTS.md](../../src/gnn/AGENTS.md)**: Implementation details

## Contents

Browse the [current directory](./) for files and subdirectories; use the task links below for navigation.

### Core Files

- **`pedalboard_gnn.md`**: Complete Pedalboard-GNN integration guide
  - Audio processing for Active Inference models
  - Real-time sonification
  - VST3/AU plugin integration
  - GNN-to-audio mapping strategies

- **`AGENTS.md`**: Technical documentation and agent scaffolding
  - Complete documentation structure
  - Integration with pipeline
  - Cross-references and navigation

- **`README.md`**: Directory overview (this file)

## Pedalboard Overview

Pedalboard provides:

### High-Performance Audio Processing
- **DSP Capabilities**: High-performance digital signal processing
- **VST3/AU Plugins**: Professional audio plugin ecosystem
- **Python-Native API**: Seamless integration with Python workflows
- **Real-Time Processing**: Low-latency audio processing

### Key Features
- **Audio Effects**: Reverb, delay, distortion, and more
- **Plugin Support**: VST3 and AU plugin loading
- **Batch Processing**: Efficient audio file processing
- **Real-Time Streaming**: Live audio processing capabilities

## Integration with GNN

Pedalboard integration enables:

- **Model Sonification**: Auditory representation of Active Inference models
- **Real-Time Audio**: Real-time sonification of model dynamics
- **Audio Analysis**: Audio-based model analysis and debugging
- **Professional Quality**: High-quality audio processing for research and presentation

## Integration with Pipeline

This documentation is integrated with the 25-step GNN processing pipeline:

1. **Core Processing** (Steps 0-9): GNN parsing, validation, export
   - No audio processing occurs in these steps; parsed models are simply prepared for downstream sonification

2. **Simulation** (Steps 10-16): Model execution and analysis
   - Real-time sonification (Step 15: Audio)
   - Audio-based model analysis

3. **Integration** (Steps 17-24): System coordination and output
   - Pedalboard results integrated into comprehensive outputs
   - Audio visualization and analysis

See [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md) for complete pipeline documentation.

## Related Resources

### Main GNN Documentation
- **[GNN Overview](../gnn/gnn_overview.md)**: Core GNN concepts
- **[GNN Quickstart](../gnn/tutorials/quickstart_tutorial.md)**: Getting started guide
- **[Audio Processing](../audio/README.md)**: Audio generation and sonification

### Audio Resources
- **[SAPF Integration](../sapf/sapf_gnn.md)**: Sound As Pure Form framework
- **[Audio Processing](../audio/README.md)**: Audio generation tools
- **[Sonification](../CROSS_REFERENCE_INDEX.md#audio-and-sonification)**: Audio sonification overview

### Pipeline Architecture
- **[Pipeline Documentation](../gnn/operations/gnn_tools.md)**: Complete pipeline guide
- **[Pipeline AGENTS](../../src/gnn/AGENTS.md)**: Implementation details
- **[Pipeline README](../../src/gnn/README.md)**: Pipeline overview

## Standards and Guidelines

All documentation in this module adheres to professional standards:

- **Clarity**: Concrete, technical writing with audio processing foundations
- **Functionality**: Describes actual Pedalboard integration capabilities
- **Completeness**: Comprehensive coverage of audio processing integration
- **Consistency**: Uniform structure and style with GNN documentation ecosystem

## See Also

- **[Pedalboard Cross-Reference](../CROSS_REFERENCE_INDEX.md#pedalboard)**: Cross-reference index entry
- **[Audio Processing](../audio/README.md)**: Audio generation and sonification
- **[SAPF Integration](../sapf/sapf_gnn.md)**: Sound As Pure Form framework
- **[Main Index](../README.md)**: Return to main documentation

---

**Status**: 🛠️ Scaffold / Planned  
**Maintenance**: Regular updates with new Pedalboard features and integration capabilities


## Documents in this folder

- [Pedalboard: Technical Deep-Dive](pedalboard.md)
