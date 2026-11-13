# SAPF Documentation Agent

> **📋 Document Metadata**  
> **Type**: Audio Framework Integration Agent | **Audience**: Researchers, Audio Engineers | **Complexity**: Advanced  
> **Cross-References**: [README.md](README.md) | [SAPF GNN Guide](sapf_gnn.md) | [Audio Processing](../audio/README.md) | [Pedalboard Integration](../pedalboard/pedalboard_gnn.md) | [Main Documentation](../README.md)

## Overview

This directory contains comprehensive documentation, resources, and implementation guides for integrating **SAPF** (Sound As Pure Form) with GNN (Generalized Notation Notation). SAPF provides a concatenative programming paradigm for auditory representation and real-time sonification of Active Inference generative models.

**Version**: 1.0

## Purpose

SAPF integration enables:

- **Auditory Model Representation**: Understanding models through sound
- **Real-Time Sonification**: Real-time sonification of Active Inference dynamics
- **Multi-Dimensional Audio**: Multi-dimensional audio synthesis for complex models
- **Debugging Through Sound**: Audio-based model debugging and analysis
- **Concatenative Programming**: Lazy evaluation and infinite sequences

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
- **Step 15 (Audio)**: SAPF is the audio sonification backend (`src/gnn/audio/sapf/`; `src/gnn/15_audio.py` accepts `--audio-backend sapf`) — this is the only pipeline step that consumes SAPF, and it also runs audio analysis via `--full-analysis`

No other pipeline step (Steps 0-9, 16, 20, 23) consumes SAPF or audio output: Steps 3/7 are parsing/export only, and the analysis/website/report steps have no audio references.

See [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md) for complete pipeline documentation.

## Function Signatures and API

### Audio Synthesis Functions

```python
def sonify_gnn_model(gnn_model: GNNModel, audio_params: dict) -> AudioSequence:
    """
    Generate audio sequence from GNN model using SAPF.

    Parameters:
        gnn_model: Parsed GNN model structure
        audio_params: Audio generation parameters

    Returns:
        AudioSequence with sonified model representation
    """


def apply_sapf_processing(
    audio: AudioSequence, operations: List[SAPFOp]
) -> AudioSequence:
    """
    Apply SAPF concatenative operations to audio sequence.

    Parameters:
        audio: Input audio sequence
        operations: List of SAPF operations to apply

    Returns:
        Processed AudioSequence
    """
```

## Related Resources

### Main GNN Documentation
- **[GNN Overview](../gnn/gnn_overview.md)**: Core GNN concepts
- **[GNN Quickstart](../gnn/tutorials/quickstart_tutorial.md)**: Getting started guide
- **[Audio Processing](../audio/README.md)**: Audio generation and sonification

### Audio Resources
- **[Pedalboard Integration](../pedalboard/pedalboard_gnn.md)**: Audio processing library
- **[Audio Processing](../audio/README.md)**: Audio generation tools
- **[Sonification](../CROSS_REFERENCE_INDEX.md#audio-and-sonification)**: Audio sonification overview

### Pipeline Architecture
- **[Pipeline Documentation](../gnn/operations/gnn_tools.md)**: Complete pipeline guide
- **[Pipeline AGENTS](../../src/gnn/AGENTS.md)**: Implementation details
- **[Pipeline README](../../src/gnn/README.md)**: Pipeline overview

## See Also

- **[SAPF Cross-Reference](../CROSS_REFERENCE_INDEX.md#sapf)**: Cross-reference index entry
- **[Audio Processing](../audio/README.md)**: Audio generation and sonification
- **[Pedalboard Integration](../pedalboard/pedalboard_gnn.md)**: Audio processing library
- **[Main Index](../README.md)**: Return to main documentation

---

**Maintenance**: Regular updates with new SAPF features and integration capabilities
