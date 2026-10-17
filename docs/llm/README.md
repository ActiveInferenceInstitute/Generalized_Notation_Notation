# Local and cloud LLM processing

Local inference uses Ollama's HTTP chat API. Cloud keys are unnecessary. Ollama
and downloaded model weights remain optional runtime components; ordinary
parsing, inference backends and rule-based reports work without them. Use
`--skip-llm` for a pipeline run without LLM calls.

## Apple Silicon Mac setup

Install Ollama from [the official download](https://ollama.com/download), open
its application, and verify the daemon. The app manages its own server; avoid
starting a second Homebrew server on the same port.

```bash
ollama --version
ollama list
curl -f http://127.0.0.1:11434/api/version
ollama pull smollm2:135m-instruct-q4_K_S
ollama pull gemma3:12b
```

GNN requires a verified stable Ollama runtime >=0.35.0 for the maintained
whole-context controls. The small default tag supports inexpensive transport
smoke checks. A larger instruction model is preferable for detailed reports.
The explicit Mac profile uses [Gemma 3 12B](https://ollama.com/library/gemma3:12b),
131,072 context tokens and a 4,096-token response budget. Measure memory and
latency on the actual host; available system memory alone does not prove GPU
use. `ollama ps` reports processor allocation and effective context.

Create the separate Python 3.14 local-LLM profile (the existing native environment is preserved):

```bash
just llm-local-setup
```

Then run:

```bash
# Real chat, streaming, provider selection and complete pipeline responses.
OLLAMA_TEST_MODEL=gemma3:12b just test-llm-local

# All selected example models, every default prompt, local provider only.
just llm-local --output-dir output/local_llm

# Explicit resume, with identical model/source/options and verified responses.
just llm-local --output-dir output/local_llm --resume
```

Without `just`, set `UV_PROJECT_ENVIRONMENT=.venv/local_llm` and run `uv sync --frozen --extra dev --python 3.14`, then `uv run --frozen --no-sync --python 3.14 python scripts/run_local_llm.py`.
The driver accepts `--model`, `--target-dir`, `--output-dir`, `--context-tokens`,
`--source-chunk-bytes`, `--max-tokens`, `--prompt-timeout`, `--llm-timeout`, and
`--response-words`, `--no-batch-prompts`, and `--resume`. It never starts a daemon, downloads weights, samples files, or
changes the selected model. Provisioning is an explicit separate operation.
Failures exit nonzero and retain coverage, partial text and checkpoints.

## Full pipeline and single-file use

The ordinary pipeline keeps the portable small default. To explicitly enable
this Mac's larger local configuration:

```bash
export DEFAULT_PROVIDER=ollama
export OLLAMA_MODEL=gemma3:12b
export OLLAMA_NUM_CTX=131072
export OLLAMA_SEED=42
export GNN_LLM_MAX_TOKENS=4096
export GNN_LLM_CHUNK_BYTES=65536
export GNN_LLM_PROMPT_TIMEOUT=180
export GNN_LLM_BATCH_PROMPTS=1
export GNN_LLM_RESPONSE_WORDS=160
UV_PROJECT_ENVIRONMENT=.venv/local_llm uv run --frozen --no-sync --python 3.14 python src/gnn/main.py --only-steps '13,23,24' --verbose
```

Step 24 uses the selected local tag for its executive report. Missing providers,
unfinished responses and invalid report structure produce an explicitly
`rule_based` report. An explicitly selected provider does not switch to a cloud
provider after failure. Single-file MCP analysis with `provider="ollama"` honors
`OLLAMA_MODEL` and reports an unavailable summary as failure while preserving
structural results.

For Python, call `gnn.llm.process_llm` with `provider="ollama"`, `model=...`,
`max_tokens=4096`, `source_chunk_bytes=65536`, and
`llm_config={"ollama_options": {"num_ctx": 131072, "seed": 42}}`.
Use `gnn.llm.UnifiedLLMProcessor.get_response` with
`provider_type=ProviderType.OLLAMA` for a pinned single request; its `LLMResponse`
includes completion reason, usage and native transport metadata. This processor
is also named `LLMProcessor` in `gnn.llm.llm_processor`; the package-level
`gnn.llm.LLMProcessor` is a separate structural compatibility facade. See the
[native request example](../../src/gnn/llm/providers/README.md).
Generic convenience calls without a model use the
selected provider's default, rather than sending a cloud model name to Ollama.

## Context, completion and evidence

Whole-source mode is the default. The provider sends intact messages with
`truncate=false` and `shift=false`; it refuses oversized context instead of
shortening the source. HTTP reads, initialization, inference and worker cleanup
share the request deadline. The default per-request budget is 45 seconds;
explicit finite positive `max_prompt_timeout`/`llm.prompt_timeout` values override
it, still bounded by the invocation deadline. A large cold model may need a
larger explicit budget. Automatic corpus budget is 600 seconds per selected
model; `llm_timeout` imposes an explicit total limit.

The optional multipart mode partitions UTF-8 source bytes without dropping,
reordering or normalizing them. Every part is submitted independently for every
required prompt, with at most 1,024 response tokens and instructions for a short
fragment analysis. Original part files and byte/hash manifests are retained.
A prompt completes only when every part ends with the provider's `stop` reason.
The combined report preserves all part responses in order. It **does not claim
joint reasoning about relationships across parts**. This distinction matters for
the large dense scaling examples; full cross-part synthesis remains roadmap S1.
A byte limit bounds payload size, not tokenizer context or peak memory; the
runtime must still admit every intact part.

`llm_results.json` records source identities, whole/multipart modes, structural,
prompt, summary and source-part coverage, unfinished work, provider/model,
runtime version/model digest and cache statistics. `llm_checkpoint.json` version
3 binds source bytes, partitions, prompt definitions, endpoint, model, generation
options and budgets. Resume verifies response digests and the live runtime/model
identity. Cache request version 4 binds full messages, response budgets, context,
seed and runtime identity. A token-limit response remains inspectable but does
not enter completed checkpoints or response caches.

## Other providers and optionality

OpenAI, OpenRouter and Perplexity have separate adapters, enabled by their API
keys. `ANTHROPIC_API_KEY` appears only in the configuration matrix; there is no
Anthropic adapter. Explicit provider choices remain pinned. Generic unpinned
multi-provider operations retain their documented recovery behavior.

Neither the Python Ollama SDK nor the Ollama CLI is required by the maintained
chat transport. A configured HTTP daemon and exact installed model are required
for live local inference. `OLLAMA_HOST` selects the endpoint (default
`http://127.0.0.1:11434`); URL credentials are rejected. The pipeline does not
implicitly invoke the separate daemon-start/model-pull helper functions.

## Verification and troubleshooting

- Run offline contracts with `uv run --frozen --no-sync python -m pytest tests/llm -m 'not ollama' -q`.
- Run native tests only after provisioning the exact `OLLAMA_TEST_MODEL` tag.
- Inspect `llm_results.json`, not the presence of Markdown files, for completion.
- Missing model: pull the exact configured tag; the pipeline never substitutes one.
- Context refusal: choose an admitted context or explicit multipart configuration.
- Incomplete response: increase the response budget or simplify the requested output.
- Timeout: inspect `ollama ps`, warm the model, and set an explicit measured budget.
- GPU validation: retain `ollama ps` and native request timing/usage alongside results.

The maintained local backend is Ollama. MLX and llama.cpp are not separate GNN
provider adapters; Ollama's own runner selection is reported by the runtime.

See [module contracts](../../src/gnn/llm/README.md),
[provider interface](../../src/gnn/llm/providers/README.md), and
[roadmap](../../TO-DO.md).

## Named answer bundles

The explicit Mac driver keeps the summary round first, then bundles the other
eight prompts into one structured request per source part. The source appears
once in that request. Every named answer must be present, nonempty and textual;
missing, extra or duplicate keys fail, and unfinished JSON never becomes cached
completion. Native batch text is retained alongside per-task answers. Shared
`request_sha256` values identify answers from the same provider call: count
request groups separately from task/part answers, and deduplicate native usage
by that identity. `--no-batch-prompts` restores independent requests.

The Mac profile explicitly uses a 180-second request budget and concise answer
targets (160 words, or 120 per fragment). The ordinary pipeline keeps its
45-second default and template response lengths. Large numeric source fragments
can tokenize almost one token per byte; source-byte size is not a token count.
A measured 64 KiB fragment needed 67,132 tokens and was correctly refused by a
65,536-token context. The 131,072-token profile admits that fragment; complete
eight-answer generation was measured separately at approximately 118 seconds.
These are host observations, not performance guarantees.

For larger-memory Macs, `ollama pull qwen3-coder:30b` provisions the optional
[Qwen3-Coder sparse coding model](https://ollama.com/library/qwen3-coder:30b).
Use `just llm-local --model qwen3-coder:30b --output-dir output/local_llm_qwen`.
Installed tags and runtime measurements determine readiness; the catalog alone
does not establish local acceptance.

The local profile lives in ignored `.venv/local_llm/`; model weights live in Ollama's
managed store. `just llm-local-setup` installs the frozen dev profile once; run
commands use frozen, no-sync Python 3.14. Ollama remains an explicit optional
service. MCP documentation generation is labeled structural, and single-file
MCP analysis supports explicit Ollama or default routing. Other explicit
providers are available through `UnifiedLLMProcessor`, rather than silently ignored
provider names in the MCP wrapper. Sync entrypoints also work inside an active
event loop through an owned loop bridge.
