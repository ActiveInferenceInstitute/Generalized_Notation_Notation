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
131,072 context tokens and an 8,192-token response budget. Measure memory and
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
`--response-words`, `--answer-max-chars`, `--no-batch-prompts`, and `--resume`. It never starts a daemon, downloads weights, samples files, or
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
export OLLAMA_MAX_TOKENS=8192
export GNN_LLM_MAX_TOKENS=8192
export GNN_LLM_CHUNK_BYTES=65536
export GNN_LLM_PROMPT_TIMEOUT=240
export GNN_LLM_BATCH_PROMPTS=1
export GNN_LLM_RESPONSE_WORDS=60
export GNN_LLM_ANSWER_MAX_CHARS=1000
export OLLAMA_TIMEOUT=240
UV_PROJECT_ENVIRONMENT=.venv/local_llm uv run --frozen --no-sync --python 3.14 python src/gnn/main.py --only-steps '13,23,24' --verbose
```

Step 24 uses the selected local tag for its executive report. Missing providers,
unfinished responses and invalid report structure produce an explicitly
`rule_based` report. An explicitly selected provider does not switch to a cloud
provider after failure. Single-file MCP analysis with `provider="ollama"` honors
`OLLAMA_MODEL` and reports an unavailable summary as failure while preserving
structural results.

For Python, call `gnn.llm.process_llm` with `provider="ollama"`, `model=...`,
`max_tokens=8192`, `source_chunk_bytes=65536`, `max_prompt_timeout=240`,
`batch_prompts=True`, `response_words=60`, `answer_max_chars=1000`, and
`llm_config={"ollama_options": {"num_ctx": 131072, "seed": 42}}`.
Use `gnn.llm.UnifiedLLMProcessor.get_response` with
`provider_type=ProviderType.OLLAMA` for a pinned single request; its `LLMResponse`
includes completion reason, usage and native transport metadata. This processor
is also named `LLMProcessor` in `gnn.llm.llm_processor`; the package-level
`gnn.llm.LLMProcessor` is a separate structural compatibility facade. See the
[native request example](../../src/gnn/llm/providers/README.md).
Generic convenience calls without a model use the
selected provider's default, rather than sending a cloud model name to Ollama.


## Explicit cross-part synthesis and resume

Python callers can select `synthesis_mode="joint_source"` or
`synthesis_mode="part_evidence"`; the default `"none"` preserves independent
fragment reports. These options also live in the resolved `llm` configuration.
Set an explicit `ollama_options.num_ctx` and positive `synthesis_max_tokens`.
For example, add `synthesis_mode="joint_source", synthesis_max_tokens=256`
to a bounded `process_llm(...)` invocation with `source_chunk_bytes` and
`llm_config={"ollama_options": {"num_ctx": 8192, "seed": 42}}`.
No new command-line flag is implied by this Python/configuration interface.

For each multipart logical task, `joint_source` makes a final real request with
the exact original UTF-8 source and every ordered fragment answer. Its request
ledger identifies the original byte range, all part hashes, child request and
response hashes, and complete messages. The v2 layout separates task instructions,
exact source and fallible answers; hashes stay in the ledger. Its full plan hash
also binds the cache when two plans have identical visible messages.
`response_words` applies to both fragment and final request instructions.
`part_evidence` instead supplies all
fragment answers and source identities; it explicitly does not present the raw
source jointly. Neither mode silently substitutes for the other. Joint source
must fit the selected model's context; splitting the map requests cannot make
an arbitrarily large original fit the final request.

One final request is required per task and multipart file, even when map tasks
are batched. With six default structured tasks, no custom tasks, two parts and
batching disabled, this is twelve map requests plus six synthesis requests.
All summaries, including their synthesis, finish before other task rounds.
The invocation and per-request deadlines also cover the added requests.

S1 context admission is deliberately narrower than generic Ollama support:
only reviewed Ollama **0.35.0** (llama.cpp **b11081**) and **0.40.11**
(**b11845**) GGUF contracts are admitted. Unsupported versions, missing model
metadata, model changes or unavailable loaded capacity refuse acceptance.
The adapter sends `truncate=false` and `shift=false`, binds `/api/show` template
and maximum context, observes exact-model `/api/ps` capacity, and retains native
terminal usage. Acceptance requires `stop` and space for the observed prompt,
the full declared output reserve, and a terminal token. This is a post-response
bounded admission check under the reviewed provider contract, not a predictive
client tokenizer. Overflow and length termination retain unfinished outputs.
The official [Ollama request handling](https://github.com/ollama/ollama/blob/v0.40.11/server/routes.go),
[runner wrapper](https://github.com/ollama/ollama/blob/v0.40.11/llm/llama_server.go)
and [pinned runner](https://github.com/ggml-org/llama.cpp/blob/b11845/tools/server/server-context.cpp)
own the no-shortening/error behavior.

Receipts distinguish exact client byte coverage, native terminal completion,
and `admission_basis="reviewed_provider_contract"`. Token retention is
`provider_reported_not_independent`; `attention_verified=false`. Completion
does not establish scientific correctness or detailed model understanding.
Fixture protocol tests establish none of the native model claims.

Opt-in cache request version 6, proof-bearing cache record version 3 and
checkpoint version 5 bind this policy separately from existing requests. Resume
rebuilds part messages and synthesis dependencies, validates retained proofs and
native JSON, and reuses only exact completed nodes. A stopped run can preserve
finished parts while leaving a synthesis unfinished. A complete resume makes
no generation requests, but still checks current runtime/model identity.
The reports retain fragment answers, final answers, each `.request.json`
ledger and `.native.json` response under `prompts_<id>/synthesis/`.

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
the large dense scaling examples. Explicit cross-part modes below add a separate
provider request; they do not change this default fragment contract.
A byte limit bounds payload size, not tokenizer context or peak memory; the
runtime must still admit every intact part.

`llm_results.json` records source identities, whole/multipart modes, structural,
prompt, summary and source-part coverage, unfinished work, provider/model,
runtime version/model digest and cache statistics. In default `none` mode,
`llm_checkpoint.json` version 4 binds source bytes, partitions, prompt definitions, endpoint, model, generation
options and budgets. Resume verifies response digests and the live runtime/model
identity. Cache request version 5 binds full messages, response budgets, context,
seed and runtime identity. A token-limit response remains inspectable but does
not enter completed checkpoints or response caches.

Native requests verify the server version, exact 64-digit model digest and
generation options before and after generation within the same request deadline.
A change stops corpus generation and rejects the affected response. Finalization
checks the runtime again, including explicit cache-only resume. Step 24 also binds
and checks runtime/model identity when admitting generated or cached reports.
Older startup-only cache/checkpoint records cannot satisfy this policy. For long
runs, use a fixed local runtime and model store; a desktop runtime update may
interrupt an active request, which must remain unfinished until a fresh run.

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

The Mac profile explicitly uses a 240-second request budget and a 60-word answer target. The ordinary pipeline keeps its
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

The native JSON schema constrains each named string when `answer_max_chars` is
configured. Answers reaching that boundary remain unfinished even if the provider
reports `stop`; neither the application nor the cache clips text to manufacture
completion. The word target leaves space below the native boundary for complete
sentences. The Mac profile uses 60 words, a 1,000-character boundary, 8,192
response tokens and a 240-second request allowance. A former 4,096-token bundle
ran out of output space on a dense N64 fragment; the failed raw response remains
negative evidence. The revised fragment probe returned all eight answers below
the boundary with terminal completion. Complete-corpus acceptance still requires
every selected source and task. Changed schemas, limits or prompt definitions
require a new matching checkpoint; earlier runs are not silently adopted.

The GNN analysis/comparison convenience APIs retain the response and finish
reason, and report unsuccessful completion for empty, token-limited or unknown
finish reasons. Their analysis-type labels are JSON-serializable strings. The
Ollama plain-text `analyze` helper returns an explicit failure label when a
response is unfinished; it does not present partial text as a completed analysis.
