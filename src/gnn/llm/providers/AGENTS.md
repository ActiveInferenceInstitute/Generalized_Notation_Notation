# LLM Provider Sub-module

Providers implement `BaseLLMProvider`: `initialize`, `validate_config`, async
`generate_response`, async-generator `generate_stream`, `close`, and
`provider_type`, `default_model`, `available_models` properties. The advertised
model list is not a live daemon/model readiness probe. There is no common
`health_check` or `get_available_models` interface; OpenRouter has its own
additional model-list method.

The maintained adapters are Ollama, OpenAI, OpenRouter and Perplexity. Explicit
provider requests remain pinned even on unavailable providers or circuit-breaker
failure. Generic unpinned operations retain the shared processor's recovery
behavior. See [parent contracts](../AGENTS.md).

Ollama uses bounded standard-library HTTP chat, preserving whole messages with
input truncation and shifting disabled. `preflight_model` verifies stable runtime
>=0.35.0 and the exact model tag, and records its digest and context/seed options.
Generation uses one total request deadline, including runtime capability checks;
pipeline workers supply hard process containment. Missing SDK/CLI is irrelevant
to this transport. Never start/pull a daemon or change the model implicitly.

Native completion is distinct from receiving text. Preserve finish reason,
usage and transport metadata in `LLMResponse`. Token-limit and unfinished replies
cannot satisfy complete prompt coverage or enter a completed response cache.

See [provider overview](README.md), [specification](SPEC.md), and
[Mac runbook](../../../../docs/llm/README.md).
