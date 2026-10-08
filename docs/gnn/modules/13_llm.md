# Step 13: Llm

## Architectural Mapping

**Orchestrator**: `src/gnn/13_llm.py` (52 lines)
**Implementation Layer**: `src/gnn/llm/`

## Module Description

This module provides comprehensive LLM (Large Language Model) integration capabilities for GNN models, including model analysis, interpretation, AI assistance, and enhanced understanding through multiple LLM providers.


```
src/gnn/llm/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
├── analyzer.py                    # LLM analysis system
├── llm_operations.py             # Core LLM operations
├── llm_processor.py              # LLM processing system
├── mcp.py                        # Model Context Protocol integration
├── prompts.py                    # LLM prompt templates
└── providers/                    # LLM provider implementations

## Agent Identity & Capabilities

# LLM Processing Module - Agent Scaffolding

## Module Overview

**Purpose**: LLM-enhanced analysis, model interpretation, and AI assistance for GNN models

**Pipeline Step**: Step 13: LLM processing (13_llm.py)

**Category**: AI Enhancement / Analysis

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-04-10

---

## Core Functionality

### Primary Responsibilities
1. LLM-based model analysis and interpretation
2. Natural language explanations of GNN structures
3. Active Inference concept clarification
4. Model optimization suggestions
5. Automated documentation generation

### Key Capabilities
- Multi-provider LLM support (Ollama, OpenAI, OpenRouter, Perplexity; Anthropic keys appear in the provider matrix when present)
- Automated preference for local Ollama when no cloud API keys are set (`LLMProcessor`)
- Context-aware prompt generation
- Structured output parsing
- Rate limiting and error handling

---

## API Reference

### Public Functions

#### `process_llm(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`
**Description**: Main corpus processor: deterministic structural analysis for every selected model, then summaries for every model, then fair prompt rounds on the configured provider/model.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to analyze
- `output_dir` (Path): Output directory for LLM analyses
- `verbose` (bool): Enable verbose logging (default: False)
- `analysis_type` (str, optional): Type of analysis ("comprehensive", "summary", "explain", "optimize") (default: "comprehensive")
- `provider` (str, optional): Exact provider ("openai", "openrouter", "perplexity", "ollama"); the corpus default is "ollama"
  - `"openai"`: Use OpenAI API (requires OPENAI_API_KEY; implemented in `src/gnn/llm/providers/openai_provider.py`)
  - `"openrouter"`: Use OpenRouter API (requires OPENROUTER_API_KEY; implemented in `src/gnn/llm/providers/openrouter_provider.py`)
  - `"perplexity"`: Use Perplexity API (requires PERPLEXITY_API_KEY; implemented in `src/gnn/llm/providers/perplexity_provider.py`)
  - `"ollama"`: Use local Ollama (requires Ollama installation)
  - Note: `src/gnn/llm/providers/` has no `anthropic_provider.py`. `ANTHROPIC_API_KEY` only surfaces as an availability flag in the diagnostic provider matrix (see below) and in auth-error attribution; there is no wired Anthropic provider to select via this parameter.
- `llm_tasks` (str, optional): Specific tasks ("all", "summarize", "explain", "optimize") (default: "all")
- `llm_timeout` (float, optional): Total corpus budget; automatic default is 600 seconds per selected model
- `max_prompt_timeout` (float, optional): Per-request ceiling, default 45 seconds; generated responses are capped at 512 tokens
- `model` (str, optional): Specific model to use (provider-specific)
- `**kwargs`: Additional LLM processing options

**Returns**: `bool` - True if processing succeeded, False otherwise

**Example**:
```python
from llm import process_llm
from pathlib import Path
import logging

success = process_llm(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/13_llm_output"),
    verbose=True,
    analysis_type="comprehensive",
    provider="auto",
    llm_tasks="all"
)
```

#### `analyze_gnn_file_with_llm(file_path: Path, verbose: bool = False, ollama_model: Optional[str] = None) -> Dict[str, Any] | Coroutine`
**Description**: Analyze a GNN file (heuristic extractors + optional LLM summary). When `ollama_model` is set (e.g. from `process_llm` after `_select_best_ollama_model`), summarization uses that tag on Ollama via `LLMOperations.summarize_gnn`.

**Parameters**:
- `file_path` (Path): Path to the GNN `.md` file
- `verbose` (bool): Verbose logging
- `ollama_model` (str, optional): Resolved Ollama tag for the per-file summary; if omitted, summarization follows `LLMProcessor` defaults

**Returns**: Result dict, or a coroutine if called while an event loop is already running

#### `extract_variables(content: str) -> List[Dict[str, Any]]`
**Description**: Extract variable definitions from GNN content.

**Parameters**:
- `content` (str): GNN content string

**Returns**: `List[Dict[str, Any]]` - List of variable dictionaries with name, type, dimensions

#### `extract_connections(content: str) -> List[Dict[str, Any]]`
**Description**: Extract connection definitions from GNN content.

**Parameters**:
- `content` (str): GNN content string

**Returns**: `List[Dict[str, Any]]` - List of connection dictionaries with source, target, type

#### `generate_model_insights(gnn_content: str, analysis_results: Dict[str, Any] = None) -> Dict[str, Any]`
**Description**: Generate insights from GNN model analysis.

**Parameters**:
- `gnn_content` (str): GNN content string
- `analysis_results` (Dict[str, Any], optional): Previous analysis results

**Returns**: `Dict[str, Any]` - Insights dictionary with complexity, patterns, recommendations

#### `generate_documentation(gnn_content: str, model_name: str = None) -> str`
**Description**: Generate comprehensive documentation for GNN model using LLM.

**Parameters**:
- `gnn_content` (str): GNN content string
- `model_name` (str, optional): Name of the model

**Returns**: `str` - Generated documentation as markdown string

---

## LLM Providers

### Supported Providers (`LLMProcessor` / `ProviderType`)
1. **Ollama** — local inference through bounded HTTP chat on Ollama 0.35 or later. Whole messages and role/generation options are preserved; input truncation and context shifting are disabled. Default model tag `smollm2:135m-instruct-q4_K_S` (`llm.defaults.DEFAULT_OLLAMA_MODEL`; overridable via `OLLAMA_MODEL`, `OLLAMA_TEST_MODEL`, or `input/config.yaml` `llm.model`).
2. **OpenAI** — cloud API when `OPENAI_API_KEY` is set.
3. **OpenRouter** — when `OPENROUTER_API_KEY` is set.
4. **Perplexity** — when `PERPLEXITY_API_KEY` is set.

### Ollama (local) — implementation surface

| Location | Role |
|----------|------|
| `llm/providers/ollama_provider.py` | `OllamaProvider`: `initialize`, `validate_config`, `generate_response`, `generate_stream`, `analyze`, `close` |
| `llm/llm_processor.py` | `LLMProcessor` merges `get_default_provider_configs()` into `provider_configs` so env vars apply even when the caller passes `None`/`{}` |
| `llm/processor.py` | `_start_ollama_if_needed`, `_select_best_ollama_model`, `_model_is_cached`; step-13 orchestration and `provider_matrix` |

**Whole-context acceptance**: Ollama runtime capability is checked within the request budget. A source that exceeds the configured model context remains unfinished with a failure receipt. The pipeline does not shorten the source or select another model. HTTP responses are bounded at 8 MiB, with 1 MiB per streaming record. The supervised request worker provides hard containment; standalone DNS and transport threads are not a process sandbox.

**Step-13 model selection**: explicit `model` → `OLLAMA_MODEL`/`OLLAMA_TEST_MODEL` → resolved `llm.model` → `DEFAULT_OLLAMA_MODEL`. Provider selection is explicit `provider` → configured `llm.provider` → `DEFAULT_PROVIDER` → Ollama. The pipeline never substitutes providers/models or starts/pulls a daemon automatically. Separate selection helpers remain callable separately.

Every selected model receives structural analysis before provider initialization. Summary requests cover the corpus before additional prompts. Each request runs in a killable subprocess with the 45-second ceiling intersected with the run's remaining budget. Full messages, model, provider and generation settings bind caches; source bytes, prompt definitions, effective selection/budget options and the complete-context policy bind version-two checkpoints. `resume=True` reuses only verified matching responses.

`llm_results.json` records structural/prompt/summary coverage and unfinished work. Required incomplete work returns `False` and reports `partial` or `timed_out`; an empty selection records `skipped`. Recovery text never counts as a successful response. Credentials remain in the inherited environment and are excluded from request payloads and receipts.

---

## Configuration

### Configuration Options

#### LLM Provider Selection
- `provider` (str): LLM provider to use (default: `"auto"`)
  - `"auto"`: Automatically select best available provider
  - `"openai"`: Use OpenAI API (requires OPENAI_API_KEY)
  - `"openrouter"`: Use OpenRouter API (requires OPENROUTER_API_KEY)
  - `"perplexity"`: Use Perplexity API (requires PERPLEXITY_API_KEY)
  - `"ollama"`: Use local Ollama (requires Ollama installation)
  - No `"anthropic"` provider is implemented (`src/gnn/llm/providers/` has no Anthropic provider file); `ANTHROPIC_API_KEY` only appears as an availability flag in the provider matrix, not as a selectable provider

#### Analysis Type
- `analysis_type` (str): Type of analysis to perform (default: `"comprehensive"`)
  - `"comprehensive"`: Full model analysis
  - `"summary"`: Brief summary only
  - `"explain"`: Concept explanations
  - `"optimize"`: Optimization suggestions

#### LLM Tasks
- `llm_tasks` (str): Specific tasks to perform (default: `"all"`)
  - `"all"`: All available tasks
  - `"summarize"`: Generate model summary
  - `"explain"`: Explain Active Inference concepts
  - `"optimize"`: Suggest optimizations

#### Performance Settings
- `llm_timeout` (float): Total corpus budget; automatic default is 600 seconds per selected model
- `max_prompt_timeout` (float): Per-request ceiling, default 45 seconds
- Corpus responses use at most 512 tokens and temperature 0.2; standalone provider calls accept `LLMConfig`

#### Environment Variables (Ollama)
- `OLLAMA_MODEL`: Model tag for requests (default `llm.defaults.DEFAULT_OLLAMA_MODEL`)
- `OLLAMA_TEST_MODEL`: Overrides model name in tests when set
- `OLLAMA_MAX_TOKENS`: Default `num_predict` cap (default `256` in processor config)
- `OLLAMA_TIMEOUT`: Standalone HTTP request budget in seconds (default `60`); pipeline requests remain capped at 45 seconds
- `OLLAMA_HOST`: HTTP(S) daemon URL or host:port; default `http://127.0.0.1:11434`, without URL credentials
- `OLLAMA_DISABLED`: Set to `1` or `true` to skip registering Ollama as a provider

#### Environment Variables (cloud)
- `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `PERPLEXITY_API_KEY`, `ANTHROPIC_API_KEY` (Anthropic appears in summaries/matrix when present)
- `DEFAULT_PROVIDER`: e.g. `ollama` (see `llm_processor.get_preferred_providers_from_env`)

---

## Dependencies

### Required Dependencies
- `json` - Configuration and output
- `pathlib` - File operations

### Optional Dependencies
- `openai` — OpenAI API (also backs the OpenRouter provider, which uses the OpenAI-compatible client)
- `ollama` (PyPI) — SDK remains a declared dependency; maintained chat transport uses the standard library HTTP API and needs neither SDK imports nor CLI recovery
- Note: there is no `anthropic` package dependency in use — no code in `src/gnn/llm/` imports `anthropic`; `ANTHROPIC_API_KEY` is only checked for presence in the provider matrix

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Logging utilities
- `pipeline.config` - Configuration management
- `llm.processor` - Core LLM logic

---

## Usage Examples

### Basic Usage (Auto-detect Provider)
```python
from llm import process_llm

success = process_llm(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/13_llm_output"),
    logger=logger,
    analysis_type="comprehensive",
)
```

### Specific Provider
```python
success = process_llm(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/13_llm_output"),
    logger=logger,
    provider="ollama",  # Force Ollama
    llm_tasks="all",
)
```

---

## Output Specification

### Output Products
- `llm_results.json` - Detailed results: per-file analyses, prompt outputs, cache stats, errors
- `llm_summary.md` - Generated summary report
- `prompts_{model_stem}/{prompt_key}.md` - Per-file prompt/response transcripts (one directory per analyzed model)

### Output Directory Structure
```
output/13_llm_output/
├── llm_results.json
├── llm_summary.md
└── prompts_actinf_pomdp_agent/
    ├── technical_description.md
    └── ... (one file per prompt key)
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 3.73 seconds
- **Memory**: 28.8 MB
- **Status**: SUCCESS
- **Provider Used**: Ollama (recovery)

---

## Recent Improvements

### Ollama Recovery Enhancement ✅
**Added**: Automatic Ollama availability check
```python
# Check if Ollama is available
result = subprocess.run(['ollama', 'list'], 
                       capture_output=True, timeout=5)
if result.returncode == 0:
    # Use Ollama as recovery
```

---

## Testing

### Test Files
- `tests/llm/test_llm_overall.py` - Module-level tests
- `tests/llm/test_llm_functional.py` - Functional tests
- `tests/llm/test_llm_ollama.py` - Ollama-specific tests
- `tests/llm/test_llm_ollama_integration.py` - Ollama integration tests

### Test Coverage
- Measure: `uv run --extra dev python -m pytest tests/llm/ --cov=gnn.llm --cov-report=term-missing` (do not treat fixed percentages in this doc as canonical).

### Key Test Scenarios
1. Ollama detection and availability check
2. Model selection and prioritization
3. LLM processing with Ollama integration
4. Recovery mode when Ollama unavailable
5. Error handling and recovery
6. Timeout management for LLM calls

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Ollama Not Found
**Symptom**: "Ollama not found in PATH" message

**Solution**:
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com
```

**Verification**:
```bash
ollama --version
which ollama
```

#### 2. Ollama Service Not Running
**Symptom**: "Ollama is installed but may not be running"

**Solution**:
```bash
# Start Ollama service
ollama serve

# In a separate terminal, verify it's running
ollama list
```

**Alternative**: Run Ollama in background
```bash
# macOS/Linux
nohup ollama serve > /dev/null 2>&1 &

# Or use system service (if configured)
systemctl start ollama
```

#### 3. No Models Installed
**Symptom**: "Ollama is running but no models are installed"

**Solution**:
```bash
# Default pipeline tag (see llm.defaults.DEFAULT_OLLAMA_MODEL)
ollama pull smollm2:135m-instruct-q4_K_S

# Alternates
ollama pull gemma3:4b
ollama pull tinyllama
ollama pull llama2:7b
```

**Suggested models** (latency and RAM depend on CPU/GPU — measure locally):
- **Default / small instruct**: `smollm2:135m-instruct-q4_K_S`
- **Balanced**: `tinyllama`, `gemma3:4b`
- **Larger**: `llama2:7b` and up

**View Installed Models**:
```bash
ollama list
```

#### 4. LLM Timeout Issues
**Symptom**: "Prompt execution timed out" or slow responses

**Solution**:
- ✅ **Automatic**: Module now uses adaptive timeouts based on prompt complexity
- Environment variable override:
  ```bash
  export OLLAMA_TIMEOUT=120  # Increase timeout to 120 seconds
  ```

- Use a smaller/faster tag:
  ```bash
  export OLLAMA_MODEL=smollm2:135m-instruct-q4_K_S
  ```

**Performance Tips**:
- Use GPU acceleration if available (Ollama detects automatically)
- Close other applications to free memory
- Use smaller models for routine analysis

#### 5. Model Selection Issues
**Symptom**: Wrong model being used or "model not found" errors

**Solution**:
```bash
# Override model selection via environment variable
export OLLAMA_MODEL=tinyllama

# Or specify in command
OLLAMA_MODEL=tinyllama uv run python src/gnn/13_llm.py --target-dir input/gnn_files
```

**Automatic Selection**:
- ✅ `_select_best_ollama_model` picks from installed tags using the ordered preference list in `llm/processor.py` (starts with `smollm2`, `tinyllama`, `gemma3:4b`, `gemma2:2b`, …)

**Check Which Model Was Used**:
```bash
# View LLM results
cat output/13_llm_output/llm_results.json | grep "selected_model"
```

#### 6. Recovery Mode Warnings
**Symptom**: "Proceeding with recovery LLM analysis" messages

**Explanation**: This is expected when Ollama is not available. The module provides basic analysis without live LLM interaction.

**Solution** (if you want LLM features):
1. Install and start Ollama (see issues #1 and #2)
2. Install at least one model (see issue #3)
3. Re-run the LLM step

**Recovery Capabilities**:
- ✅ Basic pattern extraction
- ✅ Variable and connection identification
- ✅ Structure analysis
- ❌ No natural language generation
- ❌ No model interpretation

#### 7. Slow LLM Processing
**Symptom**: Step 13 takes several minutes (3m+ per model)

**Causes**:
- Large models (llama2:70b, etc.)
- CPU-only inference (no GPU)
- Complex/long prompts
- Multiple GNN files being processed

**Solutions**:
1. **Use a smaller tag** (see `llm.defaults.DEFAULT_OLLAMA_MODEL`):
   ```bash
   export OLLAMA_MODEL=smollm2:135m-instruct-q4_K_S
   ```

2. **Reduce prompt complexity**:
   ```bash
   export OLLAMA_MAX_TOKENS=256  # Shorter responses
   ```

3. **Enable GPU acceleration** (if available):
   ```bash
   # Ollama uses GPU automatically if detected
   ollama run llama2 --gpu
   ```

4. **Process files individually**:
   ```bash
   # Process one file at a time
   uv run python src/gnn/13_llm.py --target-dir input/gnn_files --verbose
   ```

**Performance**: Measure with your hardware; smaller instruct models are usually faster on CPU.

#### 8. Memory Issues
**Symptom**: System slowdown or "out of memory" errors

**Solution**:
1. **Use smaller models** (e.g. default `smollm2:135m-instruct-q4_K_S`).

2. **Limit concurrent processing**:
   - Process files one at a time
   - Close other applications

3. **Monitor resource usage**:
   ```bash
   # Monitor Ollama memory usage
   ps aux | grep ollama
   htop  # or top
   ```

**Memory**: Check `ollama show <tag>` and system monitor while loading models.

### Ollama Integration Features

#### ✅ Enhanced Detection (October 2025)
- Automatic Ollama availability check
- Model listing and validation
- Service health monitoring (port 11434)
- Helpful installation instructions when not found

#### ✅ Intelligent Model Selection
- Prioritizes small, fast models for quick execution
- Automatic recovery chain
- Environment variable override support
- Logs selected model for transparency

#### ✅ Progress Tracking
- File-by-file progress indicators
- Prompt-by-prompt completion tracking
- Detailed logging with emoji indicators 📝
- Clear success/failure indicators ✅/❌

#### ✅ Error Recovery
- Graceful recovery when Ollama unavailable
- Per-prompt error handling
- Timeout protection with retry logic
- Comprehensive error messages

### Best Practices

1. **Install and Start Ollama Before Running**:
   ```bash
   # Terminal 1: Start Ollama
   ollama serve
   
   # Terminal 2: Run pipeline
   uv run python src/gnn/main.py --only-steps "13" --verbose
   ```

2. **Use Appropriate Model for Task**:
   - **Quick / default**: `smollm2:135m-instruct-q4_K_S`
   - **Balanced**: `tinyllama`, `gemma3:4b`
   - **Deep Analysis**: `llama2:7b` and larger

3. **Monitor Performance**:
   ```bash
   # Run with verbose logging
   uv run python src/gnn/13_llm.py --verbose --target-dir input/gnn_files
   
   # Check timing in results
   cat output/13_llm_output/llm_results.json
   ```

4. **Optimize for Speed**:
   ```bash
   export OLLAMA_MODEL=smollm2:135m-instruct-q4_K_S
   export OLLAMA_MAX_TOKENS=256
   export OLLAMA_TIMEOUT=30
   ```

5. **Check Results Quality**:
   ```bash
   # View generated analyses
   cat output/13_llm_output/prompts_*/technical_description.md
   cat output/13_llm_output/llm_summary.md
   ```

### Advanced Configuration

#### Environment Variables
```bash
# Model selection
export OLLAMA_MODEL=tinyllama           # Override automatic selection
export OLLAMA_TEST_MODEL=smollm2:135m-instruct-q4_K_S   # Test/CI model

# Performance tuning
export OLLAMA_MAX_TOKENS=512            # Maximum response length
export OLLAMA_TIMEOUT=60                # Request timeout (seconds)
export OLLAMA_HOST=http://localhost:11434  # Ollama server URL

# Behavior
export OLLAMA_DISABLED=1                # Skip Ollama provider registration
export DEFAULT_PROVIDER=ollama          # Prefer Ollama when keys allow
```

#### Custom Model Configuration
```python
# In your code or config
from llm.llm_processor import get_default_provider_configs

configs = get_default_provider_configs()
configs["ollama"]["default_model"] = "my-custom-model"
configs["ollama"]["default_max_tokens"] = 1024
```

---

## Error Handling

### Graceful Degradation
- **No API Keys**: Automatically recovery to Ollama if available
- **Ollama Unavailable**: Skip LLM analysis, log informative message, continue pipeline
- **LLM Timeout**: Retry with shorter timeout, then skip if still fails
- **Invalid Response**: Parse what's possible, log warning

### Error Categories
1. **Provider Unavailable**: No API keys and Ollama not available (recovery: skip analysis)
2. **API Errors**: Rate limits, network errors (recovery: retry with backoff)
3. **Timeout Errors**: LLM response too slow (recovery: use faster model or skip)
4. **Parsing Errors**: Invalid LLM response format (recovery: use raw response)

### Error Recovery
- **Automatic Recovery**: Try next available provider automatically
- **Partial Analysis**: Generate what's possible, report failures
- **Resource Cleanup**: Proper cleanup of LLM connections on errors
- **Informative Messages**: Clear error messages with recovery suggestions

---

## Integration Points

### Pipeline Integration
- **Input**: Re-parses GNN files from the target directory; optionally injects `10_ontology_output/ontology_results.json` when Step 10 has run
- **Output**: Writes standalone analyses to `output/13_llm_output/`, consumed by Step 20 (website) and Step 23 (report) through their output-dir scans
- **Dependencies**: No step artifacts required; optional read of Step 10's `ontology_results.json`

### Module Dependencies
- **gnn/**: GNN content is re-read from raw files for analysis
- **ontology/**: Optionally injects `ontology_results.json` into prompts (Step 10 → Step 13)
- **analysis/**: Step 16 does not read `13_llm_output`; the documented 'LLM insights → analysis' edge does not exist at runtime
- **report/website**: Consume Step 13 outputs only via their uniform output-dir scans

### External Integration
- **OpenAI API**: Cloud-based LLM analysis
- **OpenRouter / Perplexity APIs**: Cloud-based LLM analysis
- **Ollama**: Local LLM execution for privacy and offline use

### Data Flow
```
input/gnn_files (re-parsed by 13_llm.py)
  ↓
10_ontology_output/ontology_results.json [optional injection]
  ↓
13_llm.py (LLM analysis)
  ↓
  └→ output/13_llm_output/ (Standalone analyses; scanned by Steps 20 and 23)
```

---

## Version History

### Current module status

**Features**:
- Multi-provider LLM support (OpenAI, OpenRouter, Perplexity, Ollama)
- Automatic Ollama recovery
- Context-aware prompt generation
- Structured output parsing
- Rate limiting and error handling

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced prompt optimization
- **Future**: Multi-modal LLM support

---

## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [Ollama Integration Guide](../../../docs/llm/)
- [LLM Configuration](../../../.agent_rules#ollama-llm-integration-standards)

### External Resources
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com)
- [Ollama Documentation](https://ollama.ai/docs)

---

**Last Updated**: 2026-04-10
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern


---
## Documentation
- **[README](../../../src/gnn/llm/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/llm/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/llm/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/llm/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/llm](../../../src/gnn/llm)
