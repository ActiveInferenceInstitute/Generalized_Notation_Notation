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

Automatic budget is 600 seconds per selected model; explicit positive finite budgets win. `request_worker.py` runs a pinned provider/model in a killable process with a 45-second request ceiling. Model/provider mismatches fail. Auth failures retain only a safe typed category.

Full request definitions bind caches. Version-two checkpoints bind source hashes, model/provider, prompt definitions and resolved effective options; response digests verify resumed work. Required unfinished work returns False with `partial`/`timed_out` and exact coverage. Empty selection is `skipped`. Ollama requires a verified 0.35-or-later daemon and complete-context HTTP chat; oversized source inputs fail without truncation or model replacement. Structural analysis or hashes do not prove runtime/model validity.
