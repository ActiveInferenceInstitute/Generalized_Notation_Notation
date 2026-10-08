# Step 15: Audio

## Architectural Mapping

**Orchestrator**: `src/gnn/15_audio.py` (78 lines)
**Implementation Layer**: `src/gnn/audio/`

## Module Description

This module provides comprehensive audio generation capabilities for GNN models, including sonification, audio synthesis, and multi-backend audio processing with support for SAPF (Synthetic Audio Processing Framework) and Pedalboard.


```
src/gnn/audio/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
├── generator.py                    # Main audio generation system
├── mcp.py                         # Model Context Protocol integration
├── pedalboard/                    # Pedalboard audio processing
│   └── [pedalboard components]
└── sapf/                          # SAPF audio framework
    ├── __init__.py               # SAPF module initialization

## Agent Identity & Capabilities

# Audio Module - Agent Scaffolding

## Module Overview

**Purpose**: Generate audio representations and sonifications of GNN models using SAPF, Pedalboard, and other audio backends

**Pipeline Step**: Step 15: Audio processing (15_audio.py)

**Category**: Audio Generation / Sonification

**Status**: 🔄 In Development

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-01-21

---

## Core Functionality

### Primary Responsibilities
1. Convert GNN specifications to audio representations
2. Generate SAPF (Structured Audio Processing Format) code
3. Apply audio effects with Pedalboard
4. Create sonifications of model dynamics
5. Support multiple audio backends

### Key Capabilities
- SAPF code generation from GNN models
- Audio synthesis and processing
- Model sonification (state transitions, observations)
- Multi-backend support (SAPF, Pedalboard, Pure Data)
- WAV file generation

---

## API Reference

### Public Functions

#### `process_audio(target_dir, output_dir, verbose=False, logger=None, **kwargs) -> bool`
**Description**: Main audio processing function called by orchestrator (15_audio.py)

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for audio files
- `audio_backend` (str): Audio backend ("auto", "sapf", "pedalboard")
- `duration` (float): Audio duration in seconds
- `**kwargs**: Additional options

**Returns**: `True` if audio generation succeeded

#### `generate_audio_from_gnn(gnn_model, duration=30.0) -> Path`
**Description**: Generate audio file from GNN model

**Returns**: Path to generated WAV file

#### `create_sonification(gnn_model, **kwargs) -> bytes`
**Description**: Create sonification of model dynamics

**Returns**: Audio data as bytes

---

## Configuration

### Configuration Options

#### Audio Backend Selection
- `audio_backend` (str): Audio backend to use (default: `"auto"`)
  - `"auto"`: Automatically select best available backend
  - `"sapf"`: Use SAPF backend
  - `"pedalboard"`: Use Pedalboard backend
  - `"pure_data"`: Use Pure Data backend

#### Audio Generation Parameters
- `duration` (float): Audio duration in seconds (default: `30.0`)
- `sample_rate` (int): Audio sample rate in Hz (default: `44100`)
- `channels` (int): Number of audio channels (default: `1` for mono, `2` for stereo)

#### Sonification Strategy
- `sonification_strategy` (str): Strategy for model-to-sound mapping (default: `"default"`)
  - `"default"`: Standard mapping (states→pitch, observations→timbre)
  - `"harmonic"`: Emphasize harmonic relationships
  - `"rhythmic"`: Emphasize temporal patterns
  - `"textural"`: Emphasize timbral variations

#### SAPF Configuration
- `sapf_output_format` (str): SAPF output format (default: `"python"`)
  - Options: `"python"`, `"json"`, `"yaml"`

---

## Dependencies

### Required Dependencies
- `numpy` - Audio sample generation
- `soundfile` - WAV file I/O

### Optional Dependencies
- `librosa` - Audio analysis (recovery: basic generation)
- `pedalboard` - Audio effects (recovery: skip effects)
- `sapf` - SAPF backend (recovery: skip SAPF)

---

## Usage Examples

### Basic Usage
```python
from gnn.audio import process_audio

success = process_audio(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/15_audio_output"),
    audio_backend="auto",
    duration=30.0
)
```

### Generate Specific Audio
```python
from gnn.audio import generate_audio_from_gnn
from gnn import load_parsed_model

model = load_parsed_model("actinf_pomdp_agent.md")
audio_file = generate_audio_from_gnn(model, duration=60.0)
```

---

## Output Specification

### Output Products
- `*.wav` - Generated audio files
- `*_sapf.py` - SAPF code files
- `audio_processing_summary.json` - Processing summary

### Output Directory Structure
```
output/15_audio_output/
├── model_name_sonification.wav
└── audio_processing_summary.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 114ms
- **Memory**: 19.09 MB
- **Status**: SUCCESS
- **Files Processed**: 1
- **Audio Generated**: 0 (module in development)

### Audio Generation Times
- **SAPF Generation**: ~50ms per model
- **WAV Rendering**: ~1-5s for 30s audio
- **Effects Processing**: +500ms-2s with Pedalboard

---

## Sonification Strategies

### Model-to-Sound Mapping
1. **States → Pitch**: State values map to musical pitches
2. **Observations → Timbre**: Observation probabilities affect tone
3. **Actions → Rhythm**: Action selection creates rhythmic patterns
4. **Free Energy → Volume**: Lower FE = louder (more confident)
5. **Connections → Harmonies**: Connected variables create harmonies

---

## Error Handling

### Graceful Degradation
- **No SAPF**: Skip SAPF generation, log warning, continue with other backends
- **No Pedalboard**: Skip effects processing, use basic audio generation
- **No soundfile**: Return error, cannot generate WAV files
- **Invalid GNN model**: Return structured error, skip model

### Error Categories
1. **Backend Unavailable**: Framework not installed (recovery: skip backend)
2. **Audio Generation Failure**: Cannot generate audio (return error)
3. **File I/O Errors**: Cannot write WAV files (return error)
4. **Model Parsing Errors**: Invalid GNN structure (skip model, log error)

### Error Recovery
- **Backend Recovery**: Automatically try next available backend
- **Partial Generation**: Generate what's possible, report failures
- **Resource Cleanup**: Proper cleanup of audio resources on errors

---

## Integration Points

### Pipeline Integration
- **Input**: Re-parses GNN `.md` files from the target directory; optionally loads Step 12 execution telemetry from `12_execute_output/`
- **Output**: Writes audio files and results JSON to `output/15_audio_output/`, included in Step 20 (website) and Step 23 (report) output-dir scans
- **Dependencies**: No step artifacts required; optional Step 12 telemetry

### Module Dependencies
- **gnn/**: Reads raw GNN file content for sonification
- **sapf/**: Uses SAPF module for structured audio generation
- **export/**: Uses export formats for audio metadata

### External Integration
- **SAPF Backend**: Integrates with SAPF audio processing framework
- **Pedalboard**: Optional integration for audio effects
- **Pure Data**: Optional integration for advanced audio processing

### Data Flow
```
input/gnn_files (re-parsed by 15_audio.py)
  ↓
12_execute_output/ execution telemetry [optional]
  ↓
15_audio.py (Audio generation)
  ↓
  └→ output/15_audio_output/ (Standalone audio files; scanned by Steps 20 and 23)
```

---

## Testing

### Test Files
- `tests/audio/test_audio_integration.py`
- `tests/audio/test_audio_sapf.py`

### Test Coverage

- Measure: `uv run --extra dev python -m pytest tests/audio/ --cov=gnn.audio --cov-report=term-missing` (do not treat fixed percentages in this doc as canonical).

### Key Test Scenarios
1. Audio generation from GNN models
2. SAPF code generation
3. Audio backend validation
4. Sonification strategies

---

## MCP Integration

### Tools Registered

Registered in `register_tools` (`src/gnn/audio/mcp.py`):

- `process_audio` - Run Step 15 over a directory
- `check_audio_backends` - Report available audio backends
- `get_audio_generation_options` - Return generation options
- `analyze_audio_characteristics` - Analyze audio content
- `validate_audio_content` - Validate audio content
- `get_audio_module_info` - Module metadata

### MCP File Location
- `src/gnn/audio/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Audio backend not available
**Symptom**: Audio generation fails with backend errors  
**Cause**: Required audio libraries not installed  
**Solution**: 
- Install audio dependencies: `uv pip install soundfile librosa pedalboard`
- Check backend availability: `python -c "import soundfile; print('OK')"`
- Use `--audio-backend auto` for automatic selection

#### Issue 2: WAV file generation fails
**Symptom**: Audio processing completes but no WAV files created  
**Cause**: File permissions or disk space issues  
**Solution**:
- Check output directory permissions
- Verify sufficient disk space
- Check file system format supports WAV files

#### Issue 3: Sonification produces silence
**Symptom**: Generated audio files are silent  
**Cause**: Model dynamics not extracted or sonification strategy mismatch  
**Solution**:
- Verify GNN model has state transitions or dynamics
- Try different sonification strategies
- Check audio sample rate and duration settings

---

## Version History

### Current module status

**Features**:
- SAPF code generation
- Audio synthesis and processing
- Model sonification
- Multi-backend support

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced sonification strategies
- **Future**: Real-time audio streaming

---

## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [SAPF Documentation](../../../docs/sapf/)
- [Pedalboard Documentation](../../../docs/pedalboard/)

### External Resources
- [SAPF Specification](../../sapf/README.md)
- [Pedalboard Documentation](https://github.com/spotify/pedalboard)
- [Librosa Documentation](https://librosa.org/)

---

**Last Updated**: 2026-01-21
**Maintainer**: GNN Pipeline Team
**Status**: 🔄 In Development
**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern


---
## Documentation
- **[README](../../../src/gnn/audio/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/audio/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/audio/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/audio/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/audio](../../../src/gnn/audio)
