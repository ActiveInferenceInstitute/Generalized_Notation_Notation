"""Corpus-wide structural coverage and fair, resumable LLM scheduling."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from gnn.pipeline._io import atomic_write_text
from gnn.pipeline.run_context import current_run_context, selected_model_sources

from .defaults import ollama_generation_options
from .prompt_batch import batch_prompt, parse_answers, response_schema
from .provider_identity import resolved_endpoint
from .providers.base_provider import ProviderType
from .source_parts import (
    combine_part_responses,
    multipart_instruction,
    partition_source,
)


def _positive_seconds(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return seconds


def _identity(
    path: Path, root: Path, context: Any, input_view: Path | None = None
) -> dict[str, str]:
    if context:
        assert input_view is not None
        for model in context.selected_models(13):
            staged = (
                input_view / f"{model.artifact_stem}{Path(model.source_path).suffix}"
            )
            if path.resolve() in {Path(model.source_path).resolve(), staged.resolve()}:
                return {
                    "model_id": model.model_id,
                    "source_path": model.relative_path,
                    "source_sha256": model.sha256,
                    "artifact_stem": model.artifact_stem,
                }
        raise ValueError("LLM source is outside the frozen selection")
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    digest = hashlib.sha256(relative.encode()).hexdigest()[:16]
    return {
        "model_id": digest,
        "source_path": relative,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "artifact_stem": f"{path.stem}_{digest}",
    }


async def run_corpus(
    target_dir: Path, output_dir: Path, verbose: bool, kwargs: dict[str, Any], api: Any
) -> bool:
    """Run structure first, summaries for all models, then prompt rounds.

    Checkpoints bind source bytes, provider/model and prompt definitions. Only
    explicitly requested resume reuses completed work; historical files never
    add coverage to the current result index.
    """
    context = current_run_context()
    supplied = kwargs.get("resolved_config", kwargs.get("input_config"))
    config = dict(
        supplied.get("llm", {}) if supplied is not None else api._get_llm_config()
    )
    if kwargs.get("llm_config") is not None:
        config = dict(kwargs["llm_config"])
    paths_arg = kwargs.get("selected_files")
    discovered = (
        selected_model_sources(target_dir, 13)
        if paths_arg is None
        else [Path(path) for path in paths_arg]
    )
    discovered.sort(key=api._llm_file_sort_key)
    cap = api._resolve_llm_max_files(kwargs, config)
    paths = discovered if cap is None else discovered[:cap]
    budget = api._resolve_llm_budget_seconds(kwargs, config, len(paths))
    _positive_seconds(budget, "LLM budget")
    deadline = time.monotonic() + budget
    prompt_ceiling = _positive_seconds(
        kwargs.get(
            "max_prompt_timeout",
            os.getenv("GNN_LLM_PROMPT_TIMEOUT") or config.get("prompt_timeout", 45),
        ),
        "Prompt timeout",
    )

    def remaining() -> float:
        local = max(0.0, deadline - time.monotonic())
        return local if not context else context.bounded_timeout(local)

    provider = ProviderType(
        kwargs.get("provider")
        or config.get("provider")
        or os.getenv("DEFAULT_PROVIDER")
        or "ollama"
    )
    endpoint_url, endpoint_sha256 = resolved_endpoint(provider)
    model = (
        kwargs.get("model")
        or os.getenv("OLLAMA_MODEL")
        or os.getenv("OLLAMA_TEST_MODEL")
        or config.get("model")
        or api.DEFAULT_OLLAMA_MODEL
    )
    if provider == ProviderType.OLLAMA:
        model = api._validate_model_name(model)
    batching = kwargs.get(
        "batch_prompts",
        config.get("batch_prompts", os.getenv("GNN_LLM_BATCH_PROMPTS", "0") == "1"),
    )
    if not isinstance(batching, bool) or (batching and provider != ProviderType.OLLAMA):
        raise ValueError("batch_prompts requires a boolean and the Ollama provider")
    response_words = kwargs.get(
        "response_words",
        os.getenv("GNN_LLM_RESPONSE_WORDS") or config.get("response_words"),
    )
    if isinstance(response_words, str) and response_words.isdecimal():
        response_words = int(response_words)
    if response_words is not None and (
        isinstance(response_words, bool)
        or not isinstance(response_words, int)
        or response_words <= 0
    ):
        raise ValueError("response_words must be a positive integer or null")
    generation_options = (
        ollama_generation_options(config.get("ollama_options"))
        if provider == ProviderType.OLLAMA
        else {}
    )
    token_limit = kwargs.get(
        "max_tokens", os.getenv("GNN_LLM_MAX_TOKENS") or config.get("max_tokens")
    )
    if isinstance(token_limit, str) and token_limit.isdecimal():
        token_limit = int(token_limit)
    if token_limit is not None and (
        isinstance(token_limit, bool)
        or not isinstance(token_limit, int)
        or token_limit <= 0
    ):
        raise ValueError("LLM max_tokens must be a positive integer or null")
    chunk_bytes = kwargs.get(
        "source_chunk_bytes",
        os.getenv("GNN_LLM_CHUNK_BYTES") or config.get("source_chunk_bytes"),
    )
    if isinstance(chunk_bytes, str) and chunk_bytes.isdecimal():
        chunk_bytes = int(chunk_bytes)
    input_view = context.input_view(13) if context else None
    identities = [_identity(path, target_dir, context, input_view) for path in paths]
    contents = {path: path.read_bytes().decode("utf-8") for path in paths}
    plans = {path: partition_source(contents[path], chunk_bytes) for path in paths}
    # Hash and prompt the same bytes, including original CRLF and Unicode.
    for path, identity in zip(paths, identities):
        if (
            hashlib.sha256(contents[path].encode("utf-8")).hexdigest()
            != identity["source_sha256"]
        ):
            raise ValueError("LLM source changed while freezing the selection")
    structured = [
        api.PromptType.SUMMARIZE_CONTENT,
        api.PromptType.EXPLAIN_MODEL,
        api.PromptType.IDENTIFY_COMPONENTS,
        api.PromptType.ANALYZE_STRUCTURE,
        api.PromptType.EXTRACT_PARAMETERS,
        api.PromptType.PRACTICAL_APPLICATIONS,
    ]
    custom = kwargs.get(
        "custom_prompts",
        [
            (
                "technical_description",
                "Describe this GNN model comprehensively, in technical detail.",
            ),
            (
                "nontechnical_description",
                "Describe this GNN model for a broad audience.",
            ),
            ("runtime_behavior", "Describe what happens when this GNN model runs."),
        ],
    )
    labels = [prompt.value for prompt in structured] + [key for key, _ in custom]
    tasks: list[tuple[str, Any, str | None]] = [
        (prompt.value, prompt, None) for prompt in structured
    ]
    tasks.extend((key, None, text) for key, text in custom)
    if len(set(labels)) != len(labels) or any(
        not label.replace("_", "").isalnum() for label in labels
    ):
        raise ValueError("Prompt labels must be unique safe file names")
    binding = {
        "schema_version": 3,
        "effective_options": {
            "max_files": cap,
            "budget_seconds": budget,
            "prompt_timeout": prompt_ceiling,
            "max_tokens": token_limit,
            "source_chunk_bytes": chunk_bytes,
            "part_max_tokens_ceiling": 1024,
            "generation_options": generation_options,
            "batch_prompts": batching,
            "batch_policy": "named_answers_v1",
            "response_words": response_words,
            "temperature": 0.2,
            "input_policy": "complete_context_v1",
            "completion_policy": "terminal_stop_v1",
        },
        "endpoint_sha256": endpoint_sha256,
        "models": identities,
        "provider": provider.value,
        "model": model,
        "prompts": labels,
        "custom_prompts": custom,
        "batch_response_schema": response_schema(labels[1:]) if batching else None,
        "batch_prompt_definitions": [
            hashlib.sha256(
                json.dumps(
                    batch_prompt(
                        api,
                        tasks[1:],
                        part.text,
                        scope=multipart_instruction(
                            part, index, len(plans[path]), word_limit=None
                        )
                        if len(plans[path]) > 1
                        else "This request contains the complete source.",
                        words=120 if len(plans[path]) > 1 else response_words or 160,
                    ),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            for path in paths
            for index, part in enumerate(plans[path])
        ]
        if batching
        else [],
        "source_parts": [[part.receipt() for part in plans[path]] for path in paths],
        "multipart_instructions": [
            [
                multipart_instruction(part, index, len(plans[path]))
                for index, part in enumerate(plans[path])
            ]
            for path in paths
            if len(plans[path]) > 1
        ],
        "prompt_definitions": [
            hashlib.sha256(
                json.dumps(
                    api.get_prompt(prompt, contents[path]),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            for path in paths
            for prompt in structured
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(binding, sort_keys=True).encode()
    ).hexdigest()
    checkpoint = output_dir / "llm_checkpoint.json"
    state: dict[str, Any] = {
        "binding": binding,
        "fingerprint": fingerprint,
        "completed": {},
        "completed_parts": {},
    }
    if kwargs.get("resume") and checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        if prior.get("fingerprint") != fingerprint:
            raise ValueError(
                "LLM resume identity/configuration does not match checkpoint"
            )
        state = prior
        for entry in [*state["completed"].values(), *state["completed_parts"].values()]:
            text = entry.get("response")
            if (
                not isinstance(text, str)
                or not text.strip()
                or hashlib.sha256(text.encode()).hexdigest() != entry.get("sha256")
            ):
                raise ValueError("LLM checkpoint response digest is invalid")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = api.LLMCache(
        cache_dir=output_dir
        / ".cache"
        / provider.value
        / "complete_context_v1"
        / endpoint_sha256
    )
    results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "run_id": context.run_id if context else fingerprint,
        "status": "running",
        "success": False,
        "processed_files": 0,
        "selected_files": len(paths),
        "total_files_discovered": len(discovered),
        "skipped_files": len(discovered) - len(paths),
        "budget_seconds": budget,
        "provider": provider.value,
        "selected_model": model,
        "endpoint_sha256": endpoint_sha256,
        "generation_options": generation_options,
        "errors": [],
        "auth_errors": [],
        "analysis_results": [],
        "model_insights": [],
        "code_suggestions": [],
        "documentation_generated": [],
        "file_selection": {
            "max_files": cap,
            "policy": "explicit selection or all model sources",
            "models": identities,
        },
    }

    def save() -> None:
        atomic_write_text(checkpoint, json.dumps(state, indent=2))
        atomic_write_text(
            output_dir / "llm_results.json", json.dumps(results, indent=2)
        )

    records: list[tuple[Path, dict[str, Any], str, dict[str, str]]] = []
    save()
    # Complete deterministic structure before any provider initialization or call.
    for path, identity in zip(paths, identities):
        if remaining() <= 0:
            break
        try:
            candidate = api.analyze_gnn_file_with_llm(path, verbose, attempt_llm=False)
            analysis = await candidate if inspect.isawaitable(candidate) else candidate
            analysis.update(identity)
            analysis.update(
                llm_prompt_outputs={},
                prompt_outcomes={},
                part_outcomes={},
                llm_summary_status="pending",
                source_context={
                    "mode": "multipart" if len(plans[path]) > 1 else "whole",
                    "source_bytes": len(contents[path].encode("utf-8")),
                    "parts": [part.receipt() for part in plans[path]],
                    "joint_context": len(plans[path]) == 1,
                },
            )
            results["analysis_results"].append(analysis)
            results["model_insights"].append(api.generate_model_insights(analysis))
            results["code_suggestions"].append(api.generate_code_suggestions(analysis))
            results["documentation_generated"].append(
                api.generate_documentation(analysis)
            )
            records.append((path, analysis, contents[path], identity))
            if len(plans[path]) > 1:
                source_dir = output_dir / "source_parts" / identity["artifact_stem"]
                source_dir.mkdir(parents=True, exist_ok=True)
                for index, part in enumerate(plans[path]):
                    # Write exact bytes; text-mode newline translation would
                    # invalidate byte coverage on Windows.
                    (source_dir / f"part_{index + 1:04d}.txt").write_bytes(
                        part.text.encode("utf-8")
                    )
                atomic_write_text(
                    source_dir / "manifest.json",
                    json.dumps(analysis["source_context"], indent=2),
                )
            results["processed_files"] += 1
        except Exception as exc:
            results["errors"].append({"file": str(path), "error": str(exc)})
        save()

    processor = None
    isolated = False
    ready = False
    runtime_identity: dict[str, Any] = {}
    failed_auth: set[str] = set()
    try:
        try:
            processor = api.LLMProcessor(preferred_providers=[provider])
            isolated = isinstance(processor, api._LIVE_PROCESSOR_CLASS)
        except (TypeError, ValueError) as exc:
            results["errors"].append({"stage": "initialization", "error": str(exc)})
        unfinished = any(
            f"{identity['model_id']}:{label}" not in state["completed"]
            for identity in identities
            for label in labels
        )
        if (
            processor is not None
            and records
            and (unfinished or isolated)
            and remaining() > 0
        ):
            try:
                if isolated:
                    from .request_worker import isolated_request

                    preflight = await isolated_request(
                        provider=provider.value,
                        model=model,
                        preflight=True,
                        endpoint_url=endpoint_url,
                        generation_options=generation_options,
                        timeout=min(prompt_ceiling, remaining()),
                    )
                    runtime_identity = preflight.metadata or {}
                    if (
                        kwargs.get("resume")
                        and state.get("runtime_identity") != runtime_identity
                    ):
                        raise ValueError(
                            "LLM resume runtime/model digest does not match checkpoint"
                        )
                    state["runtime_identity"] = runtime_identity
                    results["runtime_identity"] = runtime_identity
                    ready = True
                else:
                    # Injection point for deterministic provider-contract tests.
                    ready = await processor.initialize()
            except Exception as exc:
                results["errors"].append({"stage": "preflight", "error": str(exc)})
        results["provider_matrix"] = {
            provider.value: {"available": ready, "selected_model": model}
        }
        # Round-major scheduling guarantees every summary precedes any extra.
        prefetched: dict[str, tuple[str, dict[str, Any]]] = {}
        for label, prompt_type, custom_text in tasks:
            for path, analysis, content, identity in records:
                key = f"{identity['model_id']}:{label}"
                parts = plans[path]
                responses = []
                part_statuses = []
                for index, part in enumerate(parts):
                    part_key = f"{key}:part_{index + 1:04d}"
                    prior = state["completed_parts"].get(part_key)
                    if isolated and not ready:
                        prior = None  # Old parts cannot establish current runtime acceptance.
                    if prior:
                        text = prior["response"]
                        if (
                            not isinstance(text, str)
                            or not text.strip()
                            or hashlib.sha256(text.encode()).hexdigest()
                            != prior.get("sha256")
                        ):
                            raise ValueError(
                                "LLM checkpoint response digest is invalid"
                            )
                        outcome = {
                            **prior.get("outcome", {}),
                            "status": "success",
                            "resumed": part_key not in prefetched,
                        }
                    elif part_key in prefetched:
                        text, outcome = prefetched[part_key]
                    elif not ready or remaining() <= 0:
                        text = "Source part unfinished: provider unavailable or request budget exhausted."
                        outcome = {"status": "pending"}
                    else:
                        cfg = (
                            api.get_prompt(prompt_type, part.text)
                            if prompt_type
                            else {
                                "system_message": "You are an expert in Active Inference and GNN.",
                                "user_prompt": f"{custom_text}\n\nGNN Model Content:\n{part.text}",
                            }
                        )
                        prompt = cfg["user_prompt"]
                        tokens = (
                            token_limit
                            if token_limit is not None
                            else cfg.get("max_tokens", 2048)
                        )
                        if len(parts) > 1:
                            prompt = (
                                multipart_instruction(part, index, len(parts)) + prompt
                            )
                            tokens = min(tokens, 1024)
                        if response_words is not None:
                            prompt += f"\n\nComplete all requested topics concisely in at most {min(response_words, 120) if len(parts) > 1 else response_words} words."
                        output_fields = None
                        if batching and label != structured[0].value:
                            output_fields = [task[0] for task in tasks[1:]]
                            cfg = batch_prompt(
                                api,
                                tasks[1:],
                                part.text,
                                scope=multipart_instruction(
                                    part, index, len(parts), word_limit=None
                                )
                                if len(parts) > 1
                                else "This request contains the complete source.",
                                words=120 if len(parts) > 1 else response_words or 160,
                            )
                            prompt = cfg["user_prompt"]
                            tokens = token_limit or 4096
                        text = await api._execute_prompt(
                            processor,
                            cache,
                            cache_content=content,
                            model_name=model,
                            messages=[
                                api.LLMMessage(
                                    role="system", content=cfg["system_message"]
                                ),
                                api.LLMMessage(role="user", content=prompt),
                            ],
                            prompt_text=prompt,
                            label=part_key,
                            custom=prompt_type is None,
                            max_tokens=tokens,
                            max_prompt_timeout=min(prompt_ceiling, remaining()),
                            failed_auth_providers=failed_auth,
                            auth_errors=results["auth_errors"],
                            outcomes=analysis["part_outcomes"],
                            provider_type=provider,
                            isolated=isolated,
                            endpoint_url=endpoint_url,
                            generation_options=generation_options,
                            runtime_identity=runtime_identity,
                            output_fields=output_fields,
                            remaining_budget=remaining,
                        )
                        outcome = analysis["part_outcomes"][part_key]
                        if output_fields:
                            batch_dir = (
                                output_dir
                                / f"prompts_{identity['artifact_stem']}"
                                / "native_batches"
                            )
                            batch_dir.mkdir(parents=True, exist_ok=True)
                            atomic_write_text(
                                batch_dir / f"part_{index + 1:04d}.txt", text
                            )
                            answers = (
                                parse_answers(text, output_fields)
                                if outcome["status"] == "success"
                                else {}
                            )
                            for field in output_fields:
                                field_key = f"{identity['model_id']}:{field}:part_{index + 1:04d}"
                                field_text = answers.get(
                                    field,
                                    "Native batch unfinished; inspect the retained native batch response.",
                                )
                                field_outcome = {
                                    **outcome,
                                    "batched": True,
                                    "batch_fields": output_fields,
                                }
                                prefetched[field_key] = (field_text, field_outcome)
                                if outcome["status"] == "success":
                                    state["completed_parts"][field_key] = {
                                        "response": field_text,
                                        "sha256": hashlib.sha256(
                                            field_text.encode()
                                        ).hexdigest(),
                                        "outcome": field_outcome,
                                    }
                            text, outcome = prefetched[part_key]
                        if outcome["status"] == "success":
                            state["completed_parts"][part_key] = {
                                "response": text,
                                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                                "outcome": outcome,
                            }
                    analysis["part_outcomes"][part_key] = outcome
                    api.logger.info(
                        "%s %s part %d/%d: %s",
                        identity["source_path"],
                        label,
                        index + 1,
                        len(parts),
                        outcome["status"],
                    )
                    part_statuses.append(outcome["status"])
                    responses.append(text)
                    if len(parts) > 1:
                        part_dir = (
                            output_dir
                            / f"prompts_{identity['artifact_stem']}"
                            / f"{label}.parts"
                        )
                        part_dir.mkdir(parents=True, exist_ok=True)
                        atomic_write_text(part_dir / f"part_{index + 1:04d}.md", text)
                    save()
                response = combine_part_responses(parts, responses)
                completed_parts = part_statuses.count("success")
                # A completed multipart answer means every independent source
                # part completed. It does not assert joint attention/synthesis.
                status = (
                    "success"
                    if completed_parts == len(parts)
                    else "timed_out"
                    if "timed_out" in part_statuses
                    else "incomplete"
                    if "incomplete" in part_statuses
                    else "failed"
                    if "failed" in part_statuses
                    else "pending"
                )
                analysis["prompt_outcomes"][label] = {
                    "status": status,
                    "source_mode": analysis["source_context"]["mode"],
                    "parts_completed": completed_parts,
                    "parts_required": len(parts),
                    "resumed": all(
                        analysis["part_outcomes"][f"{key}:part_{i + 1:04d}"].get(
                            "resumed", False
                        )
                        for i in range(len(parts))
                    ),
                }
                if len(parts) == 1:
                    analysis["prompt_outcomes"][label].update(
                        analysis["part_outcomes"][f"{key}:part_0001"]
                    )
                if status == "success":
                    state["completed"][key] = {
                        "response": response,
                        "sha256": hashlib.sha256(response.encode()).hexdigest(),
                    }
                analysis["llm_prompt_outputs"][label] = response
                if label == structured[0].value:
                    success = analysis["prompt_outcomes"][label]["status"] == "success"
                    analysis["llm_summary_status"] = (
                        "available" if success else "unavailable"
                    )
                    if success:
                        analysis["llm_summary"] = response
                        analysis["analysis_method"] = "structural_with_llm_summary"
                folder = output_dir / f"prompts_{identity['artifact_stem']}"
                folder.mkdir(exist_ok=True)
                atomic_write_text(
                    folder / f"{label}.md",
                    f"# {label.replace('_', ' ').title()}\n\n{response}\n",
                )
                save()
    finally:
        if not isolated and processor is not None:
            await processor.close()
        completed = sum(
            analysis["prompt_outcomes"].get(label, {}).get("status") == "success"
            for _, analysis, _, _ in records
            for label in labels
        )
        required = len(paths) * len(labels)
        results["coverage"] = {
            "structural_completed": len(records),
            "structural_required": len(paths),
            "prompts_completed": completed,
            "prompts_required": required,
            "summaries_completed": sum(
                record[1]["llm_summary_status"] == "available" for record in records
            ),
            "summaries_required": len(paths),
            "source_parts_completed": sum(
                outcome["status"] == "success"
                for _, analysis, _, _ in records
                for outcome in analysis["part_outcomes"].values()
            ),
            "source_parts_required": sum(len(plans[path]) for path in paths)
            * len(labels),
            "request_groups_completed": len(
                {
                    outcome["request_sha256"]
                    for _, analysis, _, _ in records
                    for outcome in analysis["part_outcomes"].values()
                    if outcome["status"] == "success" and outcome.get("request_sha256")
                }
            ),
            "request_groups_required": sum(len(plans[path]) for path in paths)
            * (2 if batching else len(labels)),
            "new_provider_requests_completed": len(
                {
                    outcome["request_sha256"]
                    for _, analysis, _, _ in records
                    for outcome in analysis["part_outcomes"].values()
                    if outcome["status"] == "success"
                    and outcome.get("request_sha256")
                    and not outcome.get("cached")
                    and not outcome.get("resumed")
                }
            ),
        }
        by_id = {identity["model_id"]: analysis for _, analysis, _, identity in records}
        results["unfinished_source_parts"] = [
            f"{identity['model_id']}:{label}:part_{index + 1:04d}"
            for path, identity in zip(paths, identities)
            for label in labels
            for index in range(len(plans[path]))
            if by_id.get(identity["model_id"], {})
            .get("part_outcomes", {})
            .get(f"{identity['model_id']}:{label}:part_{index + 1:04d}", {})
            .get("status")
            != "success"
        ]
        results["unfinished_work"] = [
            f"{identity['model_id']}:{label}"
            for identity in identities
            for label in labels
            if by_id.get(identity["model_id"], {})
            .get("prompt_outcomes", {})
            .get(label, {})
            .get("status")
            != "success"
        ]
        results["success"] = (
            len(records) == len(paths)
            and completed == required
            and not results["errors"]
            and remaining() > 0
        )
        results["status"] = (
            "skipped"
            if not paths
            else "success"
            if results["success"]
            else "timed_out"
            if remaining() <= 0
            else "partial"
        )
        results["cache_stats"] = cache.summary()
        save()
        atomic_write_text(
            output_dir / "llm_summary.md", api.generate_llm_summary(results)
        )
        if paths and remaining() <= 0:
            results["success"] = False
            results["status"] = "timed_out"
            results["finalization_status"] = "timed_out"
            save()
            atomic_write_text(
                output_dir / "llm_summary.md", api.generate_llm_summary(results)
            )
    return bool(results["success"])
