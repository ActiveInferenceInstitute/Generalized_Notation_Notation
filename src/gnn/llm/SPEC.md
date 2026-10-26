# LLM Module Specification

## Overview
LLM (Large Language Model) integration for GNN processing.

## Components

### Core
- `processor.py` - LLM processor

### Providers
- Ollama (local), OpenAI, OpenRouter, Perplexity provider modules (no Anthropic module; its key only appears in the provider matrix)

## Features
- GNN to natural language
- LLM-assisted validation
- Model explanation generation

## Key Exports
```python
from gnn.llm import process_llm
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API

## Corpus scheduling and evidence

`corpus_runner.py` consumes frozen run selection/configuration, completes structural analysis before summaries, then schedules additional prompts fairly across the corpus. `None` means discover standalone inputs; `selected_files=[]` means no work. No default sampling is applied.

Automatic budget is 600 seconds per selected model; explicit positive finite budgets win. `request_worker.py` runs a pinned provider/model in a killable process with a configured per-request budget (default 45 seconds). Model/provider mismatches fail. Auth failures retain only a safe typed category.

Full request definitions bind caches. Version-four checkpoints bind source hashes, model/provider, prompt definitions and resolved effective options; response digests verify resumed work. Required unfinished work returns False with `partial`/`timed_out` and exact coverage. Empty selection is `skipped`. Ollama requires a verified 0.35-or-later daemon and complete-context HTTP chat; oversized source inputs fail without truncation or model replacement. Explicit multipart mode preserves every original UTF-8 byte, submits each part for each prompt, and records independent fragment analyses without claiming joint cross-part reasoning. Response budgets use prompt templates or explicit overrides rather than a fixed 512-token cap. Context/seed options and runtime/model digest bind cache and resume identity. Structural analysis or hashes do not prove runtime/model validity.

## Opt-in cross-part synthesis

`synthesis_mode` is `none` by default. `joint_source` submits a real final
request containing every original source byte plus ordered fragment answers;
`part_evidence` submits derived answers only and keeps that limitation explicit.
Each multipart task has its own checkpointed reducer. Required unfinished
reducers prevent success; no model or mode fallback shortens source.
`synthesis_max_tokens` is a positive output reserve (default 4096), with
explicit `llm.ollama_options.num_ctx` required. The reviewed context contract is
limited to GGUF Ollama 0.35.0/b11081 and 0.40.11/b11845. Proof records distinguish
provider-contract admission from unavailable independent token/attention
attestation. Opt-in checkpoint5/cache-request6/cache-record3 preserve actual
context observations; existing default schemas remain separate. See the
[complete interface and evidence limits](../../../docs/llm/README.md#explicit-cross-part-synthesis-and-resume).
