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

from .provider_identity import resolved_endpoint
from .providers.base_provider import ProviderType


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
        kwargs.get("max_prompt_timeout", config.get("prompt_timeout", 45)),
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
    input_view = context.input_view(13) if context else None
    identities = [_identity(path, target_dir, context, input_view) for path in paths]
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
    if len(set(labels)) != len(labels) or any(
        not label.replace("_", "").isalnum() for label in labels
    ):
        raise ValueError("Prompt labels must be unique safe file names")
    binding = {
        "schema_version": 2,
        "effective_options": {
            "max_files": cap,
            "budget_seconds": budget,
            "prompt_timeout": prompt_ceiling,
            "max_tokens_ceiling": 512,
            "temperature": 0.2,
            "input_policy": "complete_context_v1",
        },
        "endpoint_sha256": endpoint_sha256,
        "models": identities,
        "provider": provider.value,
        "model": model,
        "prompts": labels,
        "custom_prompts": custom,
        "prompt_definitions": [
            hashlib.sha256(
                json.dumps(
                    api.get_prompt(prompt, path.read_text(encoding="utf-8")),
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
    }
    if kwargs.get("resume") and checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        if prior.get("fingerprint") != fingerprint:
            raise ValueError(
                "LLM resume identity/configuration does not match checkpoint"
            )
        state = prior
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
                llm_prompt_outputs={}, prompt_outcomes={}, llm_summary_status="pending"
            )
            results["analysis_results"].append(analysis)
            results["model_insights"].append(api.generate_model_insights(analysis))
            results["code_suggestions"].append(api.generate_code_suggestions(analysis))
            results["documentation_generated"].append(
                api.generate_documentation(analysis)
            )
            records.append((path, analysis, path.read_text(encoding="utf-8"), identity))
            results["processed_files"] += 1
        except Exception as exc:
            results["errors"].append({"file": str(path), "error": str(exc)})
        save()

    processor = api.LLMProcessor(preferred_providers=[provider])
    isolated = isinstance(processor, api._LIVE_PROCESSOR_CLASS)
    ready = False
    failed_auth: set[str] = set()
    try:
        unfinished = any(
            f"{identity['model_id']}:{label}" not in state["completed"]
            for identity in identities
            for label in labels
        )
        if records and unfinished and remaining() > 0:
            try:
                if isolated:
                    from .request_worker import isolated_request

                    await isolated_request(
                        provider=provider.value,
                        model=model,
                        preflight=True,
                        endpoint_url=endpoint_url,
                        timeout=min(prompt_ceiling, remaining()),
                    )
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
        tasks = [(prompt.value, prompt, None) for prompt in structured]
        tasks.extend((key, None, text) for key, text in custom)
        for label, prompt_type, custom_text in tasks:
            for path, analysis, content, identity in records:
                key = f"{identity['model_id']}:{label}"
                prior = state["completed"].get(key)
                if prior:
                    response = prior["response"]
                    if (
                        not isinstance(response, str)
                        or not response.strip()
                        or (
                            hashlib.sha256(response.encode()).hexdigest()
                            != prior.get("sha256")
                        )
                    ):
                        raise ValueError("LLM checkpoint response digest is invalid")
                    analysis["prompt_outcomes"][label] = {
                        "status": "success",
                        "resumed": True,
                    }
                else:
                    if not ready or remaining() <= 0:
                        continue
                    cfg = (
                        api.get_prompt(prompt_type, content)
                        if prompt_type
                        else {
                            "system_message": "You are an expert in Active Inference and GNN.",
                            "user_prompt": f"{custom_text}\n\nGNN Model Content:\n{content}",
                        }
                    )
                    response = await api._execute_prompt(
                        processor,
                        cache,
                        cache_content=content,
                        model_name=model,
                        messages=[
                            api.LLMMessage(
                                role="system", content=cfg["system_message"]
                            ),
                            api.LLMMessage(role="user", content=cfg["user_prompt"]),
                        ],
                        prompt_text=cfg["user_prompt"],
                        label=label,
                        custom=prompt_type is None,
                        max_tokens=min(512, cfg.get("max_tokens", 512)),
                        max_prompt_timeout=min(prompt_ceiling, remaining()),
                        failed_auth_providers=failed_auth,
                        auth_errors=results["auth_errors"],
                        outcomes=analysis["prompt_outcomes"],
                        provider_type=provider,
                        isolated=isolated,
                        endpoint_url=endpoint_url,
                        remaining_budget=remaining,
                    )
                    if analysis["prompt_outcomes"][label]["status"] == "success":
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
        if not isolated:
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
        }
        results["unfinished_work"] = [
            f"{identity['model_id']}:{label}"
            for identity in identities
            for label in labels
            if f"{identity['model_id']}:{label}" not in state["completed"]
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
