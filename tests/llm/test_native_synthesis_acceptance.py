"""Explicitly provisioned S1 acceptance; no model choice, setup or fake responses.

GNN_LLM_NATIVE_PROFILE names an owner-frozen JSON profile. Required fields:
provider='ollama', endpoint_url, runtime_identity (the complete approved S1
preflight identity), max_tokens, synthesis_max_tokens, max_prompt_timeout,
llm_timeout, response_words and answer_max_chars. Explicit num_ctx is required;
any seed is frozen without substitution. The corpus case also requires synthesis_mode,
source_chunk_bytes and corpus=[{path, relative_path, sha256}, ...]. Corpus paths
are absolute owner-selected inputs; relative paths are distinct portable .md
names, including duplicate stems with different bytes. No source is sampled.

An absent profile skips this opt-in harness, not its qualification requirement.
Any selected skip is unavailable acceptance. A supplied but unavailable or
mismatching runtime fails; no model pull/start, fallback, or weakened proof is
attempted. Native stop/context evidence is not answer-quality or attention proof.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from gnn.llm import processor as api
from gnn.llm import request_worker
from gnn.llm.context_admission import REVIEWED_VERSIONS, validate_context_receipt
from gnn.llm.defaults import OLLAMA_RUNTIME_VALIDATION_POLICY
from gnn.llm.provider_identity import resolved_endpoint

pytestmark = [pytest.mark.ollama, pytest.mark.timeout(1800)]
RAW = ("FIRST=α\r\n" + "x" * 12 + "\r\nLAST=β").encode("utf-8")


@pytest.fixture
def native_profile(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    location = os.environ.get("GNN_LLM_NATIVE_PROFILE")
    if not location:
        pytest.skip("Owner-frozen GNN_LLM_NATIVE_PROFILE required; not qualified")
    profile = json.loads(Path(location).read_bytes())
    assert profile["provider"] == "ollama"
    identity = profile["runtime_identity"]
    assert identity["server_version"] in REVIEWED_VERSIONS
    assert isinstance(identity["model"], str) and identity["model"]
    assert re.fullmatch("[0-9a-f]{64}", identity["model_digest"])
    options = identity["generation_options"]
    assert type(options["num_ctx"]) is int and options["num_ctx"] > 0
    monkeypatch.delenv("OLLAMA_SEED", raising=False)
    if "seed" in options:
        assert type(options["seed"]) is int
    assert "context_capability" in identity
    profile["endpoint_url"], _ = resolved_endpoint(
        api.ProviderType.OLLAMA, configured_url=profile["endpoint_url"]
    )
    monkeypatch.setenv("OLLAMA_HOST", profile["endpoint_url"])
    return profile


def _options(profile: dict[str, Any], mode: str, chunk: int) -> dict[str, Any]:
    return {
        "provider": "ollama",
        "model": profile["runtime_identity"]["model"],
        "llm_config": {
            "ollama_options": copy.deepcopy(
                profile["runtime_identity"]["generation_options"]
            )
        },
        "custom_prompts": [],
        "batch_prompts": False,
        "source_chunk_bytes": chunk,
        "synthesis_mode": mode,
        **{
            name: profile[name]
            for name in (
                "max_tokens",
                "synthesis_max_tokens",
                "max_prompt_timeout",
                "llm_timeout",
                "response_words",
                "answer_max_chars",
            )
        },
    }


@pytest.fixture
def native_dispatches(
    native_profile: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    actual_request = request_worker.isolated_request
    events: dict[str, Any] = {
        "generations": [],
        "preflights": [],
        "cancelled_before_dispatch": [],
        "interrupt_reducer": False,
    }
    evidence = tmp_path / "native-dispatches.json"

    def retain() -> None:
        evidence.write_text(
            json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    async def observed(**payload: Any) -> Any:
        if (
            events["interrupt_reducer"]
            and not payload.get("preflight")
            and payload.get("response_format")
        ):
            events["interrupt_reducer"] = False
            events["cancelled_before_dispatch"].append(copy.deepcopy(payload))
            retain()
            raise asyncio.CancelledError
        entry = {"payload": copy.deepcopy(payload), "status": "dispatched"}
        events["preflights" if payload.get("preflight") else "generations"].append(
            entry
        )
        retain()
        try:
            response = await actual_request(**payload)
        except BaseException as exc:
            entry.update(status="failed", exception_type=type(exc).__name__)
            retain()
            raise
        entry.update(
            status="returned",
            response={
                "content": response.content,
                "model": response.model_used,
                "provider": response.provider,
                "finish_reason": response.finish_reason,
                "usage": response.usage,
                "metadata": copy.deepcopy(response.metadata),
            },
        )
        retain()
        assert payload["model"] == native_profile["runtime_identity"]["model"]
        assert payload["endpoint_url"] == native_profile["endpoint_url"]
        if payload.get("preflight"):
            assert response.metadata == native_profile["runtime_identity"]
        else:
            assert payload["runtime_identity"] == native_profile["runtime_identity"]
        return response

    monkeypatch.setattr(request_worker, "isolated_request", observed)
    return events


def _assert_completed(
    target: Path,
    output: Path,
    originals: dict[str, bytes],
    profile: dict[str, Any],
    events: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    result = json.loads((output / "llm_results.json").read_bytes())
    checkpoint = json.loads((output / "llm_checkpoint.json").read_bytes())
    expected_identity = profile["runtime_identity"]
    assert result["status"] == "success" and result["success"] is True
    assert (
        result["runtime_identity"]
        == result["final_runtime_identity"]
        == expected_identity
    )
    assert result["unfinished_work"] == result["unfinished_source_parts"] == []
    native_answers: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in events["generations"]:
        response = entry["response"]
        admission = response["metadata"]["context_admission"]
        key = (
            hashlib.sha256(response["content"].encode("utf-8")).hexdigest(),
            admission["messages_sha256"],
        )
        native_answers.setdefault(key, []).append(admission)
    rows = result["analysis_results"]
    assert {row["source_path"] for row in rows} == set(originals)
    maps = reducers = 0
    for row in rows:
        raw = originals[row["source_path"]]
        assert (target / row["source_path"]).read_bytes() == raw
        assert row["source_sha256"] == hashlib.sha256(raw).hexdigest()
        context = row["source_context"]
        parts = context["parts"]
        assert context["source_bytes"] == len(raw)
        maps += 6 * len(parts)
        reducers += 6 if len(parts) > 1 else 0
        cursor = 0
        rebuilt = b""
        for index, part in enumerate(parts):
            assert part["start_byte"] == cursor
            end = part["end_byte"]
            content = raw[cursor:end]
            assert end > cursor and end <= len(raw)
            assert part["sha256"] == hashlib.sha256(content).hexdigest()
            if len(parts) > 1:
                saved = (
                    output
                    / "source_parts"
                    / row["artifact_stem"]
                    / f"part_{index + 1:04d}.txt"
                )
                assert saved.read_bytes() == content
            rebuilt += content
            cursor = end
        assert rebuilt == raw and cursor == len(raw)
        if len(parts) > 1:
            assert context["joint_context"] is False
            folder = output / f"prompts_{row['artifact_stem']}" / "synthesis"
            ledgers = list(folder.glob("*.request.json"))
            assert len(ledgers) == 6
            assert set(row["synthesis_outcomes"]) == set(row["prompt_outcomes"])
            assert all(
                outcome["status"] == "success"
                for outcome in [
                    *row["part_outcomes"].values(),
                    *row["synthesis_outcomes"].values(),
                    *row["prompt_outcomes"].values(),
                ]
            )
            for path in ledgers:
                ledger = json.loads(path.read_bytes())
                assert ledger["mode"] == mode
                assert ledger["source_sha256"] == hashlib.sha256(raw).hexdigest()
                prompt = ledger["messages"][1]["content"].encode("utf-8")
                bounds = ledger["exact_source_request_range"]
                if mode == "joint_source":
                    assert prompt[bounds["start_byte"] : bounds["end_byte"]] == raw
                else:
                    assert bounds is None
                    assert b"EXACT_ORIGINAL_SOURCE_BEGIN" not in prompt
                label = ledger["label"]
                assert len(ledger["source_parts"]) == len(parts)
                for index, dependency in enumerate(ledger["source_parts"]):
                    key = f"{row['model_id']}:{label}:part_{index + 1:04d}"
                    child = checkpoint["completed_parts"][key]
                    assert {
                        name: dependency[name]
                        for name in ("start_byte", "end_byte", "sha256")
                    } == parts[index]
                    admission = child["outcome"]["context_admission"]
                    assert (
                        admission
                        in native_answers[
                            (child["sha256"], admission["messages_sha256"])
                        ]
                    )
                    assert (
                        dependency["child_request_sha256"]
                        == child["outcome"]["request_sha256"]
                    )
                    assert (
                        dependency["child_response_sha256"]
                        == hashlib.sha256(child["response"].encode()).hexdigest()
                    )
                native_text = (folder / f"{label}.native.json").read_text(
                    encoding="utf-8"
                )
                answer = json.loads(native_text)
                assert set(answer) == {label} and answer[label].strip()
                assert any(
                    entry["payload"]["messages"] == ledger["messages"]
                    and entry.get("response", {}).get("content") == native_text
                    for entry in events["generations"]
                )
    assert (
        result["coverage"]["prompts_completed"]
        == result["coverage"]["prompts_required"]
        == 6 * len(originals)
    )
    assert (
        result["coverage"]["source_parts_completed"]
        == result["coverage"]["source_parts_required"]
        == maps
    )
    assert (
        result["coverage"]["syntheses_completed"]
        == result["coverage"]["syntheses_required"]
        == reducers
    )
    assert (
        result["coverage"]["request_groups_completed"]
        == result["coverage"]["request_groups_required"]
        == maps + reducers
    )
    assert len(events["generations"]) == maps + reducers
    for entry in events["generations"]:
        assert entry["status"] == "returned"
        response, payload = entry["response"], entry["payload"]
        assert response["finish_reason"] == "stop" and response["content"].strip()
        assert response["model"] == expected_identity["model"]
        assert response["provider"] == "ollama"
        metadata = response["metadata"]
        assert (
            metadata["runtime_identity"]
            == metadata["runtime_identity_after"]
            == expected_identity
        )
        assert metadata["runtime_validation_policy"] == OLLAMA_RUNTIME_VALIDATION_POLICY
        admission = metadata["context_admission"]
        validate_context_receipt(
            admission,
            messages=payload["messages"],
            identity=expected_identity,
            max_tokens=payload["max_tokens"],
            native=True,
        )
        assert admission["prompt_tokens"] == response["usage"]["prompt_tokens"]
        assert admission["completion_tokens"] == response["usage"]["completion_tokens"]
        assert admission["attention_verified"] is False
        assert (
            admission["token_retention_attestation"]
            == "provider_reported_not_independent"
        )
    return result


def _assert_zero_generation_resume(
    target: Path,
    output: Path,
    options: dict[str, Any],
    profile: dict[str, Any],
    events: dict[str, Any],
) -> None:
    before = len(events["generations"])
    probes = len(events["preflights"])
    assert api.process_llm(target, output, resume=True, **options)
    assert len(events["generations"]) == before
    assert len(events["preflights"]) >= probes + 2
    result = json.loads((output / "llm_results.json").read_bytes())
    assert result["status"] == "success"
    assert result["final_runtime_identity"] == profile["runtime_identity"]
    assert result["coverage"]["new_provider_requests_completed"] == 0


@pytest.mark.parametrize("mode", ["joint_source", "part_evidence"])
def test_native_exact_source_modes_and_zero_generation_resume(
    tmp_path: Path,
    native_profile: dict[str, Any],
    native_dispatches: dict[str, Any],
    mode: str,
) -> None:
    target = tmp_path / "models"
    target.mkdir()
    (target / "model.md").write_bytes(RAW)
    output = tmp_path / "output"
    options = _options(native_profile, mode, 20)
    assert api.process_llm(target, output, **options)
    result = _assert_completed(
        target, output, {"model.md": RAW}, native_profile, native_dispatches, mode
    )
    assert result["analysis_results"][0]["source_context"]["parts"] == [
        {
            "start_byte": 0,
            "end_byte": 20,
            "sha256": hashlib.sha256(RAW[:20]).hexdigest(),
        },
        {
            "start_byte": 20,
            "end_byte": 31,
            "sha256": hashlib.sha256(RAW[20:]).hexdigest(),
        },
    ]
    assert result["coverage"]["source_parts_completed"] == 12
    assert result["coverage"]["syntheses_completed"] == 6
    assert len(native_dispatches["generations"]) == 18
    _assert_zero_generation_resume(
        target, output, options, native_profile, native_dispatches
    )


@pytest.mark.parametrize("mode", ["joint_source", "part_evidence"])
def test_native_cancel_before_reducer_preserves_and_resumes_finished_prefix(
    tmp_path: Path,
    native_profile: dict[str, Any],
    native_dispatches: dict[str, Any],
    mode: str,
) -> None:
    target = tmp_path / "models"
    target.mkdir()
    (target / "model.md").write_bytes(RAW)
    output = tmp_path / "output"
    options = _options(native_profile, mode, 20)
    native_dispatches["interrupt_reducer"] = True
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(api._process_llm_async(target, output, False, **options))
    checkpoint = json.loads((output / "llm_checkpoint.json").read_bytes())
    prefix = copy.deepcopy(checkpoint["completed_parts"])
    assert len(prefix) == 2 and checkpoint["completed_syntheses"] == {}
    assert len(native_dispatches["generations"]) == 2
    prefix_messages = [
        entry["payload"]["messages"] for entry in native_dispatches["generations"]
    ]
    assert api.process_llm(target, output, resume=True, **options)
    assert all(
        entry["payload"]["messages"] not in prefix_messages
        for entry in native_dispatches["generations"][2:]
    )
    result = _assert_completed(
        target, output, {"model.md": RAW}, native_profile, native_dispatches, mode
    )
    assert result["coverage"]["new_provider_requests_completed"] == 16
    completed = json.loads((output / "llm_checkpoint.json").read_bytes())[
        "completed_parts"
    ]
    for key, prior in prefix.items():
        assert completed[key]["response"] == prior["response"]
        assert (
            completed[key]["outcome"]["request_sha256"]
            == prior["outcome"]["request_sha256"]
        )
        assert (
            completed[key]["outcome"]["context_admission"]
            == prior["outcome"]["context_admission"]
        )
    _assert_zero_generation_resume(
        target, output, options, native_profile, native_dispatches
    )


def test_native_complete_owner_corpus_preserves_duplicate_stems_and_full_bytes(
    tmp_path: Path, native_profile: dict[str, Any], native_dispatches: dict[str, Any]
) -> None:
    target = tmp_path / "models"
    target.mkdir()
    originals: dict[str, bytes] = {}
    for source in native_profile["corpus"]:
        relative = source["relative_path"]
        name = PurePosixPath(relative)
        assert name.as_posix() == relative and not name.is_absolute()
        assert ".." not in name.parts and "\\" not in relative and ":" not in relative
        assert name.suffix == ".md" and relative not in originals
        path = Path(source["path"])
        assert path.is_absolute()
        raw = path.read_bytes()
        raw.decode("utf-8")
        assert hashlib.sha256(raw).hexdigest() == source["sha256"]
        destination = target.joinpath(*name.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        originals[relative] = raw
    assert len(originals) >= 2
    assert any(
        PurePosixPath(left).name == PurePosixPath(right).name and left_raw != right_raw
        for left, left_raw in originals.items()
        for right, right_raw in originals.items()
    )
    chunk = native_profile["source_chunk_bytes"]
    assert type(chunk) is int and chunk >= 4
    assert any(len(raw) > max(31, chunk) for raw in originals.values())
    mode = native_profile["synthesis_mode"]
    assert mode in {"joint_source", "part_evidence"}
    options = _options(native_profile, mode, chunk)
    options["selected_files"] = [target / relative for relative in originals]
    output = tmp_path / "output"
    assert api.process_llm(target, output, **options)
    result = _assert_completed(
        target, output, originals, native_profile, native_dispatches, mode
    )
    rows = result["analysis_results"]
    assert len({row["model_id"] for row in rows}) == len(originals)
    assert len({row["artifact_stem"] for row in rows}) == len(originals)
    _assert_zero_generation_resume(
        target, output, options, native_profile, native_dispatches
    )
