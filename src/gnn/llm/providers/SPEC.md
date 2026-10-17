# LLM provider specification

[`BaseLLMProvider`](base_provider.py) owns the common interface:

| Member | Contract |
|---|---|
| `initialize() -> bool` | Synchronous provider configuration initialization |
| `validate_config(config: LLMConfig) -> bool` | Validate request settings |
| `generate_response(messages, config) -> LLMResponse` | Await one response |
| `generate_stream(messages, config)` | Consume an asynchronous iterator of text |
| `provider_type`, `default_model`, `available_models` | Provider identity and advertised defaults |
| `get_provider_info()` | Configuration/initialization metadata |
| `close()` | Await cleanup |

Messages are `LLMMessage` objects; settings use `LLMConfig`. A provider response
retains its native completion reason, usage, model/provider and metadata. Text
alone is insufficient for complete native acceptance. Ollama's maintained
completion policy requires terminal `stop`, including streaming.

There is no common `generate`, `health_check` or `get_available_models` method.
OpenRouter's additional model-list method is adapter-specific. Defaults and
cloud model names live in each adapter; the advertised list is not a live
availability probe.

## Selection and local context

`gnn.llm.UnifiedLLMProcessor` (also
`gnn.llm.llm_processor.LLMProcessor`) coordinates native adapters. Explicit
`provider_type` requests remain pinned on failure; generic unpinned operations
can recover through initialized providers according to configured preferences.
There is no universal fixed provider chain. The package-level
`gnn.llm.LLMProcessor` is a deterministic structural compatibility facade.

Ollama uses `OLLAMA_HOST`, optional `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`,
`OLLAMA_NUM_CTX` and `OLLAMA_SEED`. The portable tag is
`smollm2:135m-instruct-q4_K_S`; the explicit Mac corpus profile uses `gemma3:12b`.
Stable runtime >=0.35.0 and the exact installed model are required. Native
preflight records the runtime version, tag digest and generation settings.
Complete messages use `truncate=false` and `shift=false`; context overflow is
refused. Analysis never implicitly starts a daemon, downloads weights, changes
models or redirects explicit local requests to cloud adapters.

OpenAI, OpenRouter and Perplexity use their separate provider configuration and
API keys. No Anthropic, MLX or standalone llama.cpp adapter exists in GNN;
Ollama's underlying runner is recorded by its runtime.

See the [executable request example](README.md), [module evidence
contracts](../AGENTS.md), and [Mac configuration guide](../../../../docs/llm/README.md).
