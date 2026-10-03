# LLM Processing Module - Agent Scaffolding

## Module Overview

**Purpose**: LLM-enhanced analysis, model interpretation, and AI assistance for GNN models

**Pipeline Step**: Step 13: LLM processing (src/gnn/13_llm.py)

**Category**: AI Enhancement / Analysis

**Status**: Maintained; live provider/model acceptance is explicit

**Last Updated**: 2026-09-24


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
- `llm_timeout` (int, optional): Timeout budget for the step in seconds (automatic default: 600 per selected model; also settable via `input/config.yaml` `llm.timeout_seconds`)
- `max_files` (int, optional): Explicit optional cap on files; no default sampling (also settable via `llm.max_files`)
- `custom_prompts` (list, optional): Custom prompt payloads; each becomes a per-file output
- `max_prompt_timeout` (int, optional): Per-prompt timeout in seconds (default: 45; also `llm.prompt_timeout`)
- `**kwargs`: Additional LLM processing options

`resolved_config`/`input_config` or `llm_config` inject configuration; pipeline calls use the immutable run context. Explicit `provider`/`model` kwargs win over configured defaults.

**Returns**: `bool` - True if processing succeeded, False otherwise

**Example**:
```python
from gnn.llm import process_llm
from pathlib import Path
import logging

success = process_llm(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/13_llm_output"),
    verbose=True,
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

#### `generate_model_insights(file_analysis: Dict[str, Any]) -> Dict[str, Any]`
**Description**: Generate insights about the GNN model.

**Parameters**:
- `file_analysis` (Dict[str, Any]): File analysis dict with `variables`, `connections`, `patterns`, and `complexity_metrics`

**Returns**: `Dict[str, Any]` - Insights dictionary with complexity, patterns, recommendations

#### `generate_documentation(file_analysis: Dict[str, Any]) -> Dict[str, Any]`
**Description**: Generate documentation for the GNN model.

**Parameters**:
- `file_analysis` (Dict[str, Any]): File analysis dict with `file_name`, `variables`, `connections`, and `complexity_metrics`

**Returns**: `Dict[str, Any]` - Documentation dictionary with `model_overview`, `variable_documentation`, `connection_documentation`, and `usage_examples`

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

Runtime knobs come from `process_llm` kwargs and `input/config.yaml` (`llm:` section); provider/model selection comes from environment variables.

#### `process_llm` kwargs
- `llm_timeout` (int): Step timeout budget in seconds (automatic default: `600` per selected model; explicit `llm.timeout_seconds` wins)
- `max_files` (int): Cap on GNN files processed per run (`llm.max_files`)
- `custom_prompts` (list): Extra prompts run per file (in addition to the structured prompt set)
- `max_prompt_timeout` (int): Per-prompt timeout in seconds (default: `45`; `llm.prompt_timeout`)

#### Environment Variables (Ollama)
- `OLLAMA_MODEL`: Model tag for requests (default `llm.defaults.DEFAULT_OLLAMA_MODEL`)
- `OLLAMA_TEST_MODEL`: Overrides model name in tests when set
- `OLLAMA_MAX_TOKENS`: Default `num_predict` cap (default `256` in processor config)
- `OLLAMA_TIMEOUT`: Standalone HTTP request budget in seconds (default `60`); pipeline requests remain capped at 45 seconds
- `OLLAMA_HOST`: HTTP(S) daemon URL or host:port; default `http://127.0.0.1:11434`, without URL credentials
- `OLLAMA_DISABLED`: Set to `1` or `true` to skip registering Ollama as a provider
- `OLLAMA_AUTO_START`: Set to `1` to allow starting the Ollama daemon when installed but not running
- `OLLAMA_AUTO_PULL`: Set to `1` to allow pulling the default model when no model is installed

#### Environment Variables (cloud)
- `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `PERPLEXITY_API_KEY`, `ANTHROPIC_API_KEY` (Anthropic appears in summaries/matrix when present)
- `DEFAULT_PROVIDER`: e.g. `ollama` (see `llm_processor.get_preferred_providers_from_env`)

---

## Dependencies

### Required Dependencies
- `json` - Configuration and output
- `pathlib` - File operations

### Core Dependencies (installed by `uv sync`; no dedicated pip extra for this module)
- `openai` — OpenAI/OpenRouter API access
- `ollama` (PyPI) — SDK remains a declared dependency; maintained chat transport uses the standard library HTTP API and needs neither SDK imports nor CLI recovery

Local inference additionally requires the Ollama 0.35 or later daemon (from https://ollama.com), which is not a pip package.

There is no Anthropic provider module; `ANTHROPIC_API_KEY` only influences the provider matrix and error attribution.

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Logging utilities
- `pipeline.config` - Configuration management
- `llm.processor` - Core LLM logic

---

## Usage Examples

### Basic Usage
```python
from gnn.llm import process_llm

success = process_llm(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/13_llm_output"),
    verbose=True,
)
```

### Local-Only Run (force Ollama)
```bash
# Provider preference is environment-driven; unset cloud keys and point at Ollama
unset OPENAI_API_KEY OPENROUTER_API_KEY PERPLEXITY_API_KEY
export OLLAMA_MODEL=smollm2:135m-instruct-q4_K_S
python src/gnn/13_llm.py --target-dir input/gnn_files --output-dir output --verbose
```

---

## Output Specification

### Output Products
- `llm_results.json` - Processing results
- `llm_summary.md` - Human-readable processing summary

### Output Directory Structure
```
output/13_llm_output/
├── llm_results.json
└── llm_summary.md
```

---

## Performance Characteristics

### Latest Execution
See `output/13_llm_output/` and the pipeline summary for the current run's duration,
memory, and provider status; this document does not track them.

---

## Testing

### Test Files
- `tests/llm/test_llm_overall.py` - Module-level tests
- `tests/llm/test_llm_functional.py` - Functional tests
- `tests/llm/test_llm_ollama.py` - Ollama-specific tests
- `tests/llm/test_llm_ollama_integration.py` - Ollama integration tests
- `tests/llm/test_llm_analyzer_extractors.py` - Extractor/pattern pure-logic tests
- `tests/llm/test_llm_prompts_registry.py` - Prompt registry completeness (all PromptType)
- `tests/llm/test_llm_cache.py` - Cache key/roundtrip/corruption/clear tests
- `tests/llm/test_llm_generator.py` - Generator heuristic outputs
- `tests/llm/test_llm_processor_helpers.py` - Step-13 helpers + shared prompt funnel
- `tests/llm/test_llm_sync_wrappers.py` - Sync wrapper error contracts (offline)
- `tests/llm/test_llm_mcp_tools.py` - MCP tool happy paths + parameter honoring
- `tests/llm/test_llm_cache_configurable.py` - Cache dir override behavior
- `tests/llm/test_llm_provider_contract.py` - Provider structural contract
- `tests/llm/test_llm_pipeline_model_wiring.py` - Model/cache/budget wiring
- `tests/llm/test_llm_mcp_security.py` - MCP path-sandbox security

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest tests/test_llm*.py \
    --cov=src/gnn/llm --cov-report=term-missing
```
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
- **Automatic**: Module uses adaptive timeouts based on prompt complexity
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
OLLAMA_MODEL=tinyllama python src/gnn/13_llm.py --target-dir input/gnn_files
```

**Automatic Selection**:
`_select_best_ollama_model` picks from installed tags using the ordered preference list in `llm/processor.py` (starts with `smollm2`, `tinyllama`, `gemma3:4b`, `gemma2:2b`, …)

**Check Which Model Was Used**:
```bash
# View LLM results
cat output/13_llm_output/llm_results/llm_results.json | grep "selected_model"
```

#### 6. Recovery Mode Warnings
**Symptom**: "Proceeding with recovery LLM analysis" messages

**Explanation**: This is expected when Ollama is not available. The module provides basic analysis without live LLM interaction.

**Solution** (if you want LLM features):
1. Install and start Ollama (see issues #1 and #2)
2. Install at least one model (see issue #3)
3. Re-run the LLM step

**Recovery Capabilities**:
- Basic pattern extraction
- Variable and connection identification
- Structure analysis
- No natural language generation
- No model interpretation

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

3. **GPU acceleration** (if available):
   ```bash
   # Ollama uses the GPU automatically when detected
   ollama run llama2
   ```

4. **Limit files per run**:
   ```bash
   # Cap the number of files processed
   python src/gnn/13_llm.py --target-dir input/gnn_files --max-files 5
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

#### Enhanced Detection (October 2025)
- Automatic Ollama availability check
- Model listing and validation
- Service health monitoring (port 11434)
- Helpful installation instructions when not found

#### Intelligent Model Selection
- Prioritizes small, fast models for quick execution
- Automatic recovery chain
- Environment variable override support
- Logs selected model for transparency

#### Progress Tracking
- File-by-file progress indicators
- Prompt-by-prompt completion tracking
- Detailed progress logging
- Clear success/failure indicators in logs

#### Error Recovery
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
   python src/gnn/main.py --only-steps "13" --verbose
   ```

2. **Use Appropriate Model for Task**:
   - **Quick / default**: `smollm2:135m-instruct-q4_K_S`
   - **Balanced**: `tinyllama`, `gemma3:4b`
   - **Deep Analysis**: `llama2:7b` and larger

3. **Monitor Performance**:
   ```bash
   # Run with verbose logging
   python src/gnn/13_llm.py --verbose --target-dir input/gnn_files
   
   # Check timing in results
   cat output/13_llm_output/llm_results/llm_results.json
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
   cat output/13_llm_output/llm_results/prompts_*/technical_description.md
   cat output/13_llm_output/llm_results/llm_summary.md
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
from gnn.llm.llm_processor import get_default_provider_configs

configs = get_default_provider_configs()
configs["ollama"]["default_model"] = "my-custom-model"
configs["ollama"]["default_max_tokens"] = 1024
```

---

## Error Handling

### Graceful Degradation
- **No API Keys**: Automatic recovery to Ollama if available
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
- **Input**: Re-parses GNN files from the target directory; optionally injects `10_ontology_output/ontology_results.json` when Step 10 has run, and a compact cross-framework comparison summary when `{model}_comparison.html` exists under `12_execute_output/` (Step 12 → Step 13)
- **Output**: Writes standalone analyses to `output/13_llm_output/`, consumed by Step 20 (website) and Step 23 (report) through their output-dir scans
- **Dependencies**: No step artifacts required; optional reads of Step 10's `ontology_results.json` and the Step 12 cross-framework comparison artifact

### Module Dependencies
- **gnn/**: GNN content is re-read from raw files for analysis
- **execute/**: Optionally injects a compact cross-framework comparison status summary into prompts (Step 12 → Step 13)
- **analysis/**: Step 16 does not read `13_llm_output`; the documented 'LLM insights → analysis' edge does not exist at runtime
- **report/website**: Consume Step 13 outputs only via their uniform output-dir scans

### External Integration
- **OpenAI / OpenRouter / Perplexity APIs**: Cloud-based LLM analysis (OpenRouter and Perplexity via their own provider modules)
- **Ollama**: Local LLM execution for privacy and offline use

### Data Flow
```
input/gnn_files (re-parsed by 13_llm.py)
  ↓
10_ontology_output/ontology_results.json [optional injection]
  ↓
12_execute_output/{model}_comparison.html [optional injection]
  ↓
13_llm.py (LLM analysis)
  ↓
  └→ output/13_llm_output/ (Standalone analyses; scanned by Steps 20 and 23)
```

---

## 2026-09-04 Composability Refactor (API deltas)

All changes are additive or behavior-preserving unless noted. Consumer grep
(`src/gnn/13_llm.py`, `intelligent_analysis/`, `research/`, tests) verified before
any signature change.

### New public surface

| Symbol | Location | Purpose |
|---|---|---|
| `LLMProcessor.get_default_provider() -> Optional[BaseLLMProvider]` | `llm/llm_processor.py` | Peek the provider that `get_response`/`analyze_gnn` would route to without making a request (mirrors default routing incl. circuit breakers). |
| `variable_type_counts(variables) -> Dict[str, int]` | `llm/analyzer.py` | Shared variable-type census used by `perform_semantic_analysis` and `generator.generate_code_suggestions`. |
| `_classify_auth_error(error_str) -> str \| None` | `llm/processor.py` | Pure auth-failure classifier with provider attribution (openai/openrouter/perplexity/ollama/unknown). |
| `_prompt_fallback_text(label, custom) -> str` | `llm/processor.py` | Single source for recovery text recorded per failed prompt. |
| `_execute_prompt(...)` | `llm/processor.py` | Async funnel shared by the structured `PromptType` loop and custom prompts: cache lookup, timeout, auth fail-fast, response caching. |
| `_error_result(exc) -> Dict[str, Any]` | `llm/llm_processor.py` | Shared never-raises failure payload for the synchronous GNN wrappers. |

### Behavior changes

1. **Auth fail-fast is now enforced.** After an auth error (401/403/invalid
   key) is attributed to a provider, later prompts routed to that provider
   short-circuit to the recovery text instead of burning retries — matching
   what the logs previously claimed. Auth errors are reported once per
   provider in `results["auth_errors"]`; non-auth failures still yield the
   fallback text and never abort the run.
2. **Circuit-breaker parity.** `LLMProcessor.analyze_gnn` now trips the same
   per-provider circuit breaker as `get_response`, and `_try_fallback_analysis`
   resets breakers on success (previously only `get_response` did).
3. **MCP `analyze_gnn_with_llm_mcp` honors its documented params.**
   Unknown `analysis_type` values now fail with `success: False` (previously
   silently ignored); `provider="ollama"` pins `DEFAULT_OLLAMA_MODEL` for the
   per-file summary call, any other value uses standard multi-provider
   routing. Both parameters are echoed in the result payload.
4. **`get_available_providers()` is environment-driven** (was import-probe
   based, which always listed openai/openrouter and never perplexity). Ollama
   counts unless `OLLAMA_DISABLED` is truthy; each cloud provider requires its
   API key. Delegates to `load_api_keys_from_env()`.
5. **All `PromptType` members are answerable.** Added missing prompt configs
   for `COMPARE_MODELS` and `VALIDATE_SYNTAX` (previously `get_prompt` raised
   `ValueError` for them despite `get_all_prompt_types()` advertising them).
6. **Module-level `llm_operations.summarize_gnn` accepts `ollama_model`**,
   aligning the convenience wrapper with `LLMOperations.summarize_gnn`.
7. **`generate_explanation` / `enhance_model` delegate** to
   `analyze_gnn_model(content, "summary"/"enhancement")` — identical flow,
   one implementation.
8. **MCP registration is table-driven** (single `tool_specs` tuple + loop),
   and `get_llm_providers_mcp` now reports openrouter/perplexity alongside
   the previous four.

### Removed (dead weight)

- `llm/llm_system_demo.py` and `llm/demo_llm_features.py`: zero references
  repo-wide; the former asserted against in-memory provider doubles (fabricated "tests"),
  the latter called live LLM providers when run. Demo usage lives in
  AGENTS.md/README examples instead.

---

## Version History

### Current Version: [pyproject.toml](../../../pyproject.toml) (canonical)

**Features**:
- Multi-provider LLM support (Ollama, OpenAI, OpenRouter, Perplexity)
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
- [LLM Configuration](../../../AGENTS.md)

### External Resources
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com)
- [Ollama Documentation](https://ollama.ai/docs)

---

**Last Updated**: 2026-09-24
**Maintainer**: GNN Pipeline Team
**Status**: Maintained; live provider/model acceptance is explicit
**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: Thin Orchestrator Pattern


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
