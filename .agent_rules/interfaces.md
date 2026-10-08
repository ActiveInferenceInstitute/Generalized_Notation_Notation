# CLI, API, GUI and LLM interfaces

Use this guide for public adapters. Owners:
[CLI](../src/gnn/cli/AGENTS.md), [API](../src/gnn/api/AGENTS.md),
[GUI](../src/gnn/gui/AGENTS.md), [LLM](../src/gnn/llm/AGENTS.md).

## Shared contracts

Bind CLI, Python, REST, GUI and MCP options to canonical configuration and
implementation. Validate types/precedence, empty selections, nested paths and
experimental options. Update help/schemas/routes/docs/parity tests together.

Installed entrypoints need wheel/resource checks outside the checkout. Distinguish
imports/builders/health from live callback, worker, transport and scientific
acceptance. Name verified Python/dependency/OS combinations.

## API and GUI

API deliberately has job/tool and run/runs app factories. Preserve owned routes
and shared `{status,data,error,meta}` envelopes. Markdown/SSE retain their media
contracts. Reproduce is verify-only. Preserve path admission, bind/auth, bounded
work and redaction. Test worker start/cancellation/observation/cleanup.
In-memory state does not imply persistence.

GUI builders and callbacks require separate installed Gradio checks. Headless
HTML is not callback acceptance. Retain upstream interpreter/callback failures
with the bounded limitation instead of claiming support.

## LLM and Ollama

Defaults belong to [gnn.llm.defaults](../src/gnn/llm/defaults.py) and processor/
provider configuration. Preserve explicit provider/model precedence. Step 13
does not start/pull a daemon or substitute models/providers automatically.

Preflight the configured runtime/model within budget. Ollama complete-context
requests require documented runtime capability. Keep whole messages, roles,
source bytes and generation options; truncation/context shifting stay disabled.
Oversized context remains unfinished with a reason.

Supervised requests intersect prompt ceilings with the remaining deadline.
Cache full messages/options. Version-two checkpoints bind source, provider/model,
endpoint/configuration, prompts and response digests. Resume verified matches only.

Record structural, summary and additional-prompt coverage and unfinished work
separately. Recovery text is not a provider response. Credentials stay in the
environment, outside payloads/receipts.

## Verify

Exercise real adapters and malformed options, auth/path refusal, stale runs,
deadlines and cleanup failures. Provision services explicitly for native acceptance;
hosted unit and local provider evidence remain separate.
