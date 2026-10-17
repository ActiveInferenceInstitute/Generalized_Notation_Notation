# LLM providers

Ollama, OpenAI, OpenRouter and Perplexity implement the shared
[`BaseLLMProvider`](base_provider.py) interface. Ollama uses standard-library
HTTP to a configured daemon; neither its Python SDK nor CLI is required by this
transport. Provision exact model weights separately.

## Native local request

Run inside the installed GNN environment after provisioning `gemma3:12b`:

```python
import asyncio
from gnn.llm import LLMMessage, ProviderType, UnifiedLLMProcessor

async def local_reply():
    processor = UnifiedLLMProcessor(preferred_providers=[ProviderType.OLLAMA])
    try:
        if not await processor.initialize():
            raise RuntimeError("Local provider initialization failed")
        response = await processor.get_response(
            [LLMMessage(role="user", content="Reply with only the word ok.")],
            provider_type=ProviderType.OLLAMA,
            model_name="gemma3:12b",
            max_tokens=32,
        )
        if response.finish_reason != "stop":
            raise RuntimeError("Local response did not finish")
        return response.content
    finally:
        await processor.close()

print(asyncio.run(local_reply()))
```

`UnifiedLLMProcessor` is also named `LLMProcessor` in `gnn.llm.llm_processor`.
The package-level `gnn.llm.LLMProcessor` is a separate structural compatibility
facade. Import the asynchronous processor explicitly for native generation.

There is no `get_provider("ollama")` module factory or common `generate(prompt)`
method. Lazy class accessors such as `get_ollama_provider_class` return a class;
`processor.get_provider(ProviderType.OLLAMA)` returns an initialized instance.
Explicit provider requests do not substitute cloud providers after failure.

## Extend a provider

Implement `initialize`, `validate_config`, async `generate_response`,
`generate_stream`, and the properties `provider_type`, `default_model` and
`available_models`. Responses carry content, provider/model, usage, finish reason
and native transport metadata. `close` performs asynchronous cleanup. Wire class
creation and configuration into `llm_processor.py` and lazy access into
`providers/__init__.py`; verify routing, error and completion contracts.

The advertised model list does not prove runtime readiness. Ollama's separate
`preflight_model` checks runtime capabilities and exact installed tag/digest.
See [interface details](SPEC.md), [maintenance rules](AGENTS.md), and the
[Mac runbook](../../../../docs/llm/README.md).
