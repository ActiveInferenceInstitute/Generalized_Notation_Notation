"""Bounded admission through a reviewed provider contract, never attention proof.

The exact Ollama 0.35.0 and 0.40.11 GGUF contracts are reviewed against their
pinned llama.cpp b11081 and b11845 runners. False truncate/shift flags preserve messages or refuse overflow.
Native counters and loaded capacity corroborate that contract; neither is an
independent token-retention or model-cognition attestation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

POLICY = "reviewed_ollama_gguf_no_truncation_v1"
REVIEWED_VERSIONS = frozenset({"0.35.0", "0.40.11"})


def _positive(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def messages_sha256(messages: list[dict[str, str]]) -> str:
    """Identify exactly the role/content messages sent to native chat."""
    if not messages or any(
        set(row) != {"role", "content"}
        or row["role"] not in {"system", "user", "assistant"}
        or not isinstance(row["content"], str)
        for row in messages
    ):
        raise ValueError("Complete textual chat messages required")
    return hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def checked_model_context(
    show: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any]:
    """Validate an explicitly configured context against reviewed model metadata."""
    if identity.get("server_version") not in REVIEWED_VERSIONS:
        raise ValueError("S1 context admission unsupported for this Ollama version")
    if show.get("details", {}).get("format") != "gguf":
        raise ValueError("S1 context admission requires the reviewed GGUF runner")
    info = show.get("model_info", {})
    architecture = info.get("general.architecture")
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("Model architecture context metadata unavailable")
    maximum = _positive(info.get(architecture + ".context_length"), "Model context")
    requested = _positive(
        identity.get("generation_options", {}).get("num_ctx"), "Explicit num_ctx"
    )
    if requested > maximum:
        raise ValueError("Requested context exceeds advertised model context")
    template = show.get("template")
    if not isinstance(template, str):
        raise ValueError("Model template identity unavailable")
    return {
        "policy": POLICY,
        "architecture": architecture,
        "model_context_tokens": maximum,
        "requested_context_tokens": requested,
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
    }


def context_receipt(
    *,
    messages: list[dict[str, str]],
    identity: dict[str, Any],
    loaded: dict[str, Any],
    usage: dict[str, Any],
    max_tokens: int,
    transport: str = "ollama_http_chat",
) -> dict[str, Any]:
    """Check native runtime bounds under the reviewed no-truncation contract.

    A returned receipt does not independently observe tokenizer retention,
    attention, scientific accuracy, GPU placement, or memory requirements.
    """
    facts = identity.get("context_capability")
    if not isinstance(facts, dict) or facts.get("policy") != POLICY:
        raise ValueError("Reviewed model context capability unavailable")
    if identity.get("server_version") not in REVIEWED_VERSIONS:
        raise ValueError("S1 context admission unsupported for this Ollama version")
    if loaded.get("name") != identity.get("model") or loaded.get(
        "digest"
    ) != identity.get("model_digest"):
        raise ValueError("Loaded model context identity differs")
    capacity = _positive(loaded.get("context_length"), "Loaded context")
    requested = _positive(facts.get("requested_context_tokens"), "Requested context")
    if capacity != requested or capacity > _positive(
        facts.get("model_context_tokens"), "Model context"
    ):
        raise ValueError("Loaded context differs from the declared capacity")
    prompt = _positive(usage.get("prompt_tokens"), "Native prompt token count")
    completion = _positive(usage.get("completion_tokens"), "Native output token count")
    reserve = _positive(max_tokens, "Output token reserve")
    if completion > reserve or prompt + reserve + 1 > capacity:
        raise ValueError("Native prompt and declared output reserve exceed context")
    return {
        "policy": POLICY,
        "status": "admitted",
        "admission_basis": "reviewed_provider_contract",
        "transport_evidence": transport,
        "messages_sha256": messages_sha256(messages),
        "model": identity["model"],
        "model_digest": identity["model_digest"],
        "server_version": identity["server_version"],
        **facts,
        "loaded_context_tokens": capacity,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reserved_output_tokens": reserve,
        "reserved_terminal_tokens": 1,
        "input_controls": {"truncate": False, "shift": False},
        "token_retention_attestation": "provider_reported_not_independent",
        "attention_verified": False,
    }


def validate_context_receipt(
    receipt: Any,
    *,
    messages: list[dict[str, str]],
    identity: dict[str, Any],
    max_tokens: int,
    native: bool,
) -> None:
    """Reject missing, stale or relabeled request/cache/checkpoint context facts."""
    if not isinstance(receipt, dict):
        raise ValueError("Context admission receipt unavailable")
    rebuilt = context_receipt(
        messages=messages,
        identity=identity,
        loaded={
            "name": receipt.get("model"),
            "digest": receipt.get("model_digest"),
            "context_length": receipt.get("loaded_context_tokens"),
        },
        usage={
            "prompt_tokens": receipt.get("prompt_tokens"),
            "completion_tokens": receipt.get("completion_tokens"),
        },
        max_tokens=max_tokens,
        transport="ollama_http_chat" if native else "fixture_protocol",
    )
    if receipt != rebuilt:
        raise ValueError("Context admission receipt differs from the exact request")
