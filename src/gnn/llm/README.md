# LLM Module

Optional model interpretation through local Ollama or cloud OpenAI, OpenRouter
and Perplexity adapters. Deterministic structural analysis is always recorded
separately from native LLM responses. Cloud keys are unnecessary for local use.

## Local Mac quick start

Provision a stable Ollama runtime >=0.35.0 and the exact model before running:

```bash
ollama pull gemma3:12b
just llm-local-setup
OLLAMA_TEST_MODEL=gemma3:12b just test-llm-local
just llm-local --output-dir output/local_llm
just llm-local --output-dir output/local_llm --resume
```

The explicit local profile selects Gemma 3 12B, 131,072 context tokens,
4,096 response tokens and byte-preserving source partitions. It runs all
selected model sources and all nine default prompts. The ordinary pipeline
retains the lightweight `smollm2:135m-instruct-q4_K_S` default and whole-source
mode. Use `--skip-llm` for a pipeline without optional LLM calls.

See the [Mac setup, configuration and evidence runbook](../../../docs/llm/README.md)
for daemon setup, GPU verification, full-pipeline commands and limitations.

## Public API

```python
from pathlib import Path
from gnn.llm import process_llm

success = process_llm(
    Path("input/gnn_files"),
    Path("output/local_llm"),
    provider="ollama",
    model="gemma3:12b",
    max_tokens=4096,
    source_chunk_bytes=65536,
    llm_config={"ollama_options": {"num_ctx": 131072, "seed": 42}},
)
```

`gnn.llm.UnifiedLLMProcessor` provides async `get_response`, `analyze_gnn`,
provider lookup and cleanup; the same class is named `LLMProcessor` inside
`gnn.llm.llm_processor`. The package-level `gnn.llm.LLMProcessor` and
`LLMAnalyzer` are deterministic structural compatibility facades.
Explicit `provider_type=ProviderType.OLLAMA` stays pinned on failure.
Generic convenience calls without a model resolve the selected provider's
model. `LLMOperations.summarize_gnn(..., ollama_model=...)` pins the local tag;
unfinished summaries return an error rather than appearing complete.

`analyze_gnn_file_with_llm` combines structural extraction and an optional
summary. `attempt_llm=False` performs structural work only. It returns a dict
outside an event loop and a coroutine inside one. `analyze_gnn_model` is the
synchronous structural analyzer. See `__init__.__all__` for public exports.

## Processing contract

Every selected model receives structural analysis first, then all summaries
precede other prompt rounds. No default file sampling occurs. Explicit empty
selection is `skipped`. Automatic total budget is 600 seconds per selected
model; explicit finite positive budgets win. Each isolated request is bounded
by its prompt budget (default 45 seconds) and the remaining invocation time.

Whole-source mode preserves messages and forbids input truncation/context
shifting. Context overflow is an explicit failure. Optional multipart mode
retains every original UTF-8 byte and submits every source part for every
prompt. Results retain every part response in order and explicitly identify
fragment analysis. Complete multipart coverage **does not establish joint
analysis of cross-part relationships**. Byte limits bound payloads, not token
counts, model quality or total memory. Full synthesis remains roadmap S1.

Native completion requires the provider's terminal `stop` reason. Token-limit,
filtered or tool responses retain partial text, remain unfinished, and do not
enter completed checkpoints or response caches. Structural results and fallback
messages never count as native inference. Required incomplete work returns False.

Cache request version 4 binds source content, complete messages, provider/model,
endpoint, token/context/seed settings, runtime/model digest and completion policy.
Checkpoint version 3 also binds partition byte/hash receipts, prompt definitions
and budgets. Explicit resume verifies matching runtime/configuration/source and
response digests; a changed tag digest cannot reuse old native responses.

## Configuration

| Setting | Resolution / default |
|---|---|
| Provider | Explicit kwarg, `llm.provider`, `DEFAULT_PROVIDER`, then Ollama |
| Model | Explicit kwarg, `OLLAMA_MODEL`, `OLLAMA_TEST_MODEL`, `llm.model`, then `DEFAULT_OLLAMA_MODEL` |
| Total budget | `llm_timeout`, `llm.timeout_seconds`, then automatic |
| Prompt budget | `max_prompt_timeout`, `llm.prompt_timeout`, then 45 seconds |
| Response budget | `max_tokens`, `GNN_LLM_MAX_TOKENS`, `llm.max_tokens`, then prompt template (custom: 2048) |
| Source partition | `source_chunk_bytes`, `GNN_LLM_CHUNK_BYTES`, `llm.source_chunk_bytes`; unset means whole source |
| Local generation | `llm.ollama_options` with `num_ctx`/`seed`, or `OLLAMA_NUM_CTX`/`OLLAMA_SEED` |
| Local endpoint | `OLLAMA_HOST`, default `http://127.0.0.1:11434` |

The maintained Ollama transport uses standard-library HTTP, not the Python SDK
or CLI recovery. The daemon and exact model must be available. Pipeline
processing never starts/pulls a model or substitutes a provider. Separate helper
functions for provisioning are not part of the corpus execution path.

## Outputs and integrations

- `llm_results.json`: source identities, runtime/model digest, whole/multipart modes, measured structural/native coverage, per-prompt/part outcomes and unfinished work.
- `llm_checkpoint.json`: byte/configuration-bound completed responses for explicit resume.
- `llm_summary.md`: human-readable status and coverage distinctions.
- `prompts_<artifact_stem>/`: per-prompt reports; multipart reports also retain `<label>.parts/` responses.
- `source_parts/<artifact_stem>/`: exact multipart inputs and byte/hash manifests.

Steps 20 and 23 scan Step 13 artifacts for publication. Step 16 does not consume
LLM insights. Step 24 separately reads the current pipeline summary and can
create a local LLM executive report; incomplete/invalid reports are explicitly
`rule_based`. Its cache binds local endpoint, context/seed and request policy.
MCP tools validate paths, honor `OLLAMA_MODEL`, and distinguish configuration
from actual runtime readiness. There is no Anthropic, MLX or llama.cpp GNN adapter.

## Testing

```bash
uv run --frozen --no-sync python -m pytest tests/llm -m 'not ollama' -q
OLLAMA_TEST_MODEL=gemma3:12b just test-llm-local
```

The opt-in native tests require actual chat/stream responses and complete corpus
prompt coverage/resume. See [maintenance contracts](AGENTS.md) and
[testing guidance](../../../tests/llm/AGENTS.md).

## Documentation

- [README](README.md): module overview
- [AGENTS](AGENTS.md): maintenance contracts
- [SPEC](SPEC.md): architectural specification
- [SKILL](SKILL.md): capability API

The Mac driver uses concise named-answer bundles after the summary round, with
a 180-second explicit request budget. All required answers must validate before
completion. Shared request identities distinguish task answers from provider
calls. `--no-batch-prompts` disables bundling. The dedicated `.venv/local_llm/` profile
uses frozen Python 3.14 and preserves other native environments. See the runbook
for context admission measurements and optional Qwen3-Coder provisioning.
