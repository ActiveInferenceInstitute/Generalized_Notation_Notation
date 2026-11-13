# Pedalboard Documentation Agent

> **📋 Document Metadata**  
> **Type**: Audio Processing Integration Agent | **Audience**: Researchers, Audio Engineers | **Complexity**: Intermediate-Advanced  
> **Cross-References**: [README.md](README.md) | [Pedalboard GNN Guide](pedalboard_gnn.md) | [Audio Processing](../audio/README.md) | [Main Documentation](../README.md)

## Overview

This directory contains comprehensive documentation, resources, and implementation guides for integrating **Pedalboard** (Spotify's Audio Processing Library) with GNN (Generalized Notation Notation). Pedalboard provides high-performance DSP capabilities, VST3/AU plugin ecosystem, and Python-native API for sophisticated audio representations and real-time sonification of Active Inference models.

**Status**: ✅ Documentation Module  
**Version**: 1.0  

---

## Purpose

Pedalboard integration enables:

- **Model Sonification**: Auditory representation of Active Inference models
- **Real-Time Audio**: Real-time sonification of model dynamics
- **Audio Analysis**: Audio-based model analysis and debugging
- **Professional Quality**: High-quality audio processing for research and presentation
- **VST3/AU Plugin Support**: Professional audio plugin ecosystem integration

## Contents

Browse the [current directory](./) for files and subdirectories; use the task links below for navigation.

## Quick Navigation

- **README.md**: [Directory overview](README.md)
- **GNN Documentation**: [gnn/AGENTS.md](../gnn/AGENTS.md)
- **Main Documentation**: [docs/README.md](../README.md)
- **Pipeline Reference**: [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md)

## Integration with Pipeline

This documentation is integrated with the 25-step GNN processing pipeline:

### Simulation (Steps 10-16)
- **Step 15 (Audio)**: Pedalboard is an optional audio backend in the sonification step (`src/gnn/audio/` probes it; `src/gnn/15_audio.py` accepts `--audio-backend pedalboard`; `src/gnn/STEP_INDEX.md` lists it as a Step 15 dependency alongside soundfile) — this is the only pipeline step that consumes Pedalboard, and it also runs audio analysis via `--full-analysis`

No other pipeline step (Steps 0-9, 16, 20, 23) consumes Pedalboard or audio output: Steps 3/7 are parsing/export only, and the analysis/website/report steps have no audio references.

See [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md) for complete pipeline documentation.

## Function Signatures and API

> ⚠️ **Planned / Not Yet Implemented**: `src/gnn/audio/pedalboard/` is currently a scaffold only (no `.py` source). The signatures below are planned and do not describe shipped functions.

### Audio Processing Functions

```python
def generate_audio_from_gnn(gnn_model: GNNModel, audio_params: dict) -> AudioFile:
    """
    Generate audio representation from GNN model using Pedalboard.

    Parameters:
        gnn_model: Parsed GNN model structure
        audio_params: Audio generation parameters

    Returns:
        AudioFile with sonified model representation
    """


def apply_audio_effects(audio: AudioFile, effects: List[Plugin]) -> AudioFile:
    """
    Apply Pedalboard audio effects to generated audio.

    Parameters:
        audio: Input audio file
        effects: List of Pedalboard plugins to apply

    Returns:
        Processed AudioFile with effects applied
    """
```

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

## See Also

- **[Pedalboard Cross-Reference](../CROSS_REFERENCE_INDEX.md#pedalboard)**: Cross-reference index entry
- **[Audio Processing](../audio/README.md)**: Audio generation and sonification
- **[SAPF Integration](../sapf/sapf_gnn.md)**: Sound As Pure Form framework
- **[Main Index](../README.md)**: Return to main documentation

---

**Status**: 🛠️ Scaffold / Planned  
**Maintenance**: Regular updates with new Pedalboard features and integration capabilities
