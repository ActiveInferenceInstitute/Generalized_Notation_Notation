# LLM Processing Module

## Module Overview

Step 13 adds optional native LLM interpretation to deterministic structural
analysis. Supported providers are Ollama, OpenAI, OpenRouter and Perplexity.
Anthropic keys are diagnostic configuration only; no Anthropic adapter exists.
The maintained local adapter is Ollama HTTP, not separate MLX/llama.cpp adapters.

Read [README](README.md), [SPEC](SPEC.md) and the
[local Mac runbook](../../../docs/llm/README.md) before changing these contracts.
Root [custody rules](../../../AGENTS.md#cross-repo-custody-fep_formal-geo-infer)
apply to every Python source edit.

## Architecture

| Owner | Responsibility |
|---|---|
| `processor.py` | Public `process_llm`, configuration/helper compatibility and request/cache funnel |
| `corpus_runner.py` | Frozen selection, structural-first processing, summary-first prompt rounds, coverage and checkpoints |
| `prompt_batch.py`, `sync_bridge.py` | Strict named answers and safe sync calls inside active loops |
| `context_admission.py` | Exact reviewed provider contracts and retained context bounds; no attention attestation |
| `cross_part_synthesis.py`, `corpus_synthesis.py` | Source-preserving joint/derived request plans and exact reducer checkpoints |
| `source_parts.py` | Exact UTF-8 partitions, byte receipts and explicitly fragment-scoped reports |
| `request_worker.py` | One pinned request per supervised, killable process |
| `providers/ollama_provider.py` | Bounded HTTP transport, version/model preflight, intact messages and streaming |
| `llm_processor.py` | Shared provider selection; explicit provider requests cannot substitute providers |
| `llm_operations.py` | Sync/async convenience operations using the selected provider's default model |
| `analyzer.py`, `generator.py` | Structural extraction and downstream documentation |
| `mcp.py` | Path-validated tools, configured-model honoring and truthful summary status |
| `scripts/run_local_llm.py` | Thin explicit local corpus profile |

## Public API

`process_llm(target_dir, output_dir, verbose=False, **kwargs) -> bool` selects all
model sources unless explicitly capped. `selected_files=[]` means an empty
selection. Structural analysis finishes for every source before provider
initialization; all summaries precede other prompts. Default work is six
structured prompts plus three custom prompts per model. Custom prompt labels
must be unique safe filenames.

Selection precedence: explicit `provider` → `llm.provider` → `DEFAULT_PROVIDER`
→ Ollama. Model: explicit `model` → `OLLAMA_MODEL` → `OLLAMA_TEST_MODEL` →
`llm.model` → `DEFAULT_OLLAMA_MODEL` (`smollm2:135m-instruct-q4_K_S`). The small
default is a transport smoke model, not evidence of detailed report quality.
No pipeline call starts/pulls a daemon or changes providers/models implicitly.
Separate selection/provisioning helper functions are not the corpus path.

## Configuration

`resolved_config`/`input_config` supply the immutable invocation config; direct
`llm_config` overrides its `llm` block. Explicit function options win.

| Option | Default / behavior |
|---|---|
| `llm_timeout` / `llm.timeout_seconds` | Automatic 600 seconds per selected model; finite positive explicit total wins |
| `max_prompt_timeout` / `llm.prompt_timeout` | 45 seconds, bounded by remaining run time |
| `max_files` / `llm.max_files` | All selected sources |
| `max_tokens` / `GNN_LLM_MAX_TOKENS` / `llm.max_tokens` | Prompt template budget (custom: 2048); explicit positive integer overrides |
| `source_chunk_bytes` / `GNN_LLM_CHUNK_BYTES` / `llm.source_chunk_bytes` | Whole-source mode unless explicitly partitioned; integer >=4 |
| `llm.ollama_options` | Validated `num_ctx` and `seed`; corresponding `OLLAMA_NUM_CTX`/`OLLAMA_SEED` env defaults |
| `synthesis_mode` | `none`; explicit `joint_source` or `part_evidence` for multipart reducers |
| `synthesis_max_tokens` | Positive reducer output reserve, default 4096 |
| `resume` | False; explicit matching checkpoint reuse only |

Standalone HTTP uses `OLLAMA_HOST` and `OLLAMA_TIMEOUT` (60 seconds); provider
`OLLAMA_MAX_TOKENS` defaults to 256 for generic requests. Step 13 supplies its
own declared prompt budgets. `OLLAMA_DISABLED` controls environment provider
listing; it does not override an explicitly requested Ollama operation.

## Context and completion contracts

Ollama requires stable runtime >=0.35.0. Full messages and role/generation options
are preserved; `truncate=false` and `shift=false` forbid context shortening.
Responses are bounded at 8 MiB and streaming records at 1 MiB. Standalone
transport threads are not a process sandbox; pipeline workers own hard timeout
and cancellation containment.

Whole-source mode refuses overflow. Explicit multipart mode preserves every
original UTF-8 byte exactly once, including CRLF and non-ASCII characters.
Every part receives every prompt. Part responses are retained in order and
labeled as independent fragment analyses. Multipart completion does not assert
joint cross-part reasoning or complete scientific interpretation. A byte bound
is not a tokenizer or memory admission proof. Never silently sample, truncate,
compress away source values, change model, or label structural work as inference.

Only terminal `stop` responses count as native completion. Token limits and
other finish reasons retain partial text and unfinished outcomes without caching
or checkpoint admission. In-memory provider doubles may omit finish reasons for
contract tests; they do not establish native acceptance.

Cache request version 5 binds all messages, provider/model/endpoint, generation
settings, native runtime/model digest and completion policy. Checkpoint version
4 also binds source hashes, exact partitions, prompt definitions and budgets.
Only explicit resume reuses verified matching responses. Old records cannot
satisfy the new fingerprint. Credentials stay in the environment and out of
request payloads and receipts.

Every native generation validates the exact model digest, server version and
generation options before and after the response under one total request deadline.
Identity drift refuses cache/checkpoint admission and stops further corpus native
dispatch. Finalization validates the runtime again, including cache-only resume.
Do not reuse records from the older startup-only validation policy.

## Outputs and failure handling

`llm_results.json` contains structural/native coverage, per-prompt and per-part
outcomes, runtime identity, unfinished work, errors and cache stats.
`llm_checkpoint.json` persists each completed part immediately.
`llm_summary.md` explains whole/multipart coverage. Prompt Markdown lives in
`prompts_<artifact_stem>/`; multipart outputs additionally contain `<label>.parts/`
and `source_parts/<artifact_stem>/` exact inputs plus byte/hash manifests.

Required unfinished work returns False with `partial` or `timed_out`. Empty
selection is `skipped`. Missing local inference retains structural outputs.
Single-file MCP local summaries respect `OLLAMA_MODEL`; unavailable summaries
are not reported as successful LLM analysis. Step 24 validates its complete
report topology and terminal response, otherwise labels its report `rule_based`.
Its cache binds endpoint/context/seed, verified runtime/model identity and full
request policy; identity drift also refuses cached reports.
Step 24 cache request version 4 binds the selected provider's configured endpoint
and generation options. Later environment changes cannot relabel an initialized
provider or cause cross-endpoint report reuse.

The asynchronous native class is `gnn.llm.UnifiedLLMProcessor`, aliased from
`gnn.llm.llm_processor.LLMProcessor`. Package-level `gnn.llm.LLMProcessor` and
`LLMAnalyzer` are structural compatibility facades. Keep their evidence boundary
explicit in examples and use the native class for provider requests.

## Testing

Offline: `uv run --frozen --no-sync python -m pytest tests/llm -m 'not ollama' -q`.
Native, explicitly provisioned: `OLLAMA_TEST_MODEL=gemma3:12b just test-llm-local`.
Full Mac corpus: `just llm-local --output-dir output/local_llm`.

Tests cover real transport controls, deadlines, overflow/refusal, exact byte
coverage, partial completion, cache isolation, provider pinning, response digest
validation and native pipeline completion/resume. Native acceptance requires
actual nonempty terminal responses and complete coverage, not merely files or
boolean return types. See [tests](../../../tests/llm/AGENTS.md).

## Documentation

- [README](README.md): module overview
- [AGENTS](AGENTS.md): maintenance contracts
- [SPEC](SPEC.md): architectural specification
- [SKILL](SKILL.md): capability API

## Explicit Mac profile

`just llm-local-setup` creates `.venv/local_llm/` with frozen Python 3.14; run recipes
preserve the existing native environment. The Mac driver uses a 131,072-token
context, 65,536-byte source parts, 8,192 output tokens, 240-second request budget
and concise answers. Summary requests still precede the bundled additional
tasks. Every task answer is required; strict JSON schema and parser validation
reject incomplete, duplicate, missing or extra fields before cache admission.
Native batch text and shared request identities distinguish provider calls from
logical task/part answers. `batch_prompts` defaults False outside the driver.
`response_words` is an explicit style target; neither it nor terminal completion
proves scientific quality. Both settings, response schema and full batch prompt
definitions bind checkpoints. Cache record version 2 verifies response digests
and identity and writes atomically.

Named bundle limits: `answer_max_chars` / `GNN_LLM_ANSWER_MAX_CHARS` is an optional
positive integer, bound into schema, caches and checkpoints. The Mac profile
uses 1,000 characters and a 60-word style target. Native schema decoding enforces
the boundary; the parser also rejects boundary hits (not just overflow), because
forced string closure cannot prove a complete answer. Keep these responses
unavailable to completed checkpoints and caches even with terminal `stop`.
Scope and brevity constraints in the system message override task-template
verbosity; task-template symbols are illustrative, not source declarations.

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
