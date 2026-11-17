"""Deterministic tests for the LLMCache content-addressed store."""

import errno
import json
import multiprocessing
import os
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.llm.cache import LLMCache

pytestmark = pytest.mark.unit


class TestCacheKey:
    def test_deterministic(self) -> None:
        assert LLMCache._make_key("a", "m", "p") == LLMCache._make_key("a", "m", "p")

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            (("a", "m", "p"), ("b", "m", "p")),
            (("a", "m", "p"), ("a", "n", "p")),
            (("a", "m", "p"), ("a", "m", "q")),
        ],
        ids=["content", "model", "prompt"],
    )
    def test_sensitive_to_each_component(
        self, first: tuple[str, str, str], second: tuple[str, str, str]
    ) -> None:
        assert LLMCache._make_key(*first) != LLMCache._make_key(*second)


class TestCacheRoundtrip:
    def test_put_then_get(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path)
        cache.put("content", "model", "prompt", "response-text")
        assert cache.get("content", "model", "prompt") == "response-text"
        assert cache.hits == 1
        assert cache.writes == 1

    def test_miss_returns_none_and_counts(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path)
        assert cache.get("content", "model", "prompt") is None
        assert cache.misses == 1

    def test_different_key_is_a_miss(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path)
        cache.put("content", "model", "prompt", "response")
        assert cache.get("content", "model", "other-prompt") is None


class TestCacheRobustness:
    def test_corrupt_entry_is_dropped(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path)
        key = LLMCache._make_key("content", "model", "prompt")
        (tmp_path / f"{key}.json").write_text("{not json", encoding="utf-8")
        assert cache.get("content", "model", "prompt") is None
        assert not (tmp_path / f"{key}.json").exists()

    def test_entry_missing_response_key_is_corrupt(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path)
        key = LLMCache._make_key("content", "model", "prompt")
        (tmp_path / f"{key}.json").write_text(
            json.dumps({"model": "x"}), encoding="utf-8"
        )
        assert cache.get("content", "model", "prompt") is None

    def test_clear_removes_entries(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path)
        cache.put("a", "m", "p", "1")
        cache.put("b", "m", "p", "2")
        assert cache.clear() == 2
        assert cache.summary()["entries_on_disk"] == 0


class TestCacheSummary:
    def test_summary_shape_and_ratio(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path)
        cache.put("a", "m", "p", "1")
        cache.get("a", "m", "p")
        cache.get("zzz", "m", "p")
        summary = cache.summary()
        assert summary["hits"] == 1
        assert summary["misses"] == 1
        assert summary["writes"] == 1
        assert summary["hit_ratio_pct"] == 50.0
        assert summary["entries_on_disk"] == 1
        assert summary["cache_dir"] == str(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("response", "changed"),
        ("model", "changed"),
        ("key", "wrong"),
        ("schema_version", 1),
        ("response_sha256", "wrong"),
    ],
)
def test_valid_json_with_changed_response_or_identity_is_a_miss(
    tmp_path: Path, field: str, value: object
) -> None:
    cache = LLMCache(cache_dir=tmp_path)
    cache.put("content", "model", "prompt", "response")
    path = next(tmp_path.glob("*.json"))
    entry = json.loads(path.read_text())
    entry[field] = value
    path.write_text(json.dumps(entry))
    assert cache.get("content", "model", "prompt") is None
    assert cache.hits == 0
    assert cache.misses == 1
    assert not path.exists()


@pytest.mark.parametrize("verified", [False, True])
def test_conditional_rollback_preserves_newer_and_other_finished_requests(
    tmp_path: Path, verified: bool
) -> None:
    cache = LLMCache(cache_dir=tmp_path)
    if verified:
        publication = cache.put_verified(
            "source", "model", "request", "old", {"proof": 1}
        )
        cache.put_verified("source", "model", "request", "new", {"proof": 2})
    else:
        publication = cache.put("source", "model", "request", "old")
        cache.put("source", "model", "request", "new")
    cache.put("other source", "model", "finished", "preserved")
    assert publication is not None
    cache.discard_publication("source", "model", "request", publication)
    if verified:
        assert cache.get_verified("source", "model", "request") == {
            "response": "new",
            "metadata": {"proof": 2},
        }
    else:
        assert cache.get("source", "model", "request") == "new"
    assert cache.get("other source", "model", "finished") == "preserved"


@pytest.mark.parametrize("verified", [False, True])
def test_rollback_distinguishes_identical_completed_publications(
    tmp_path: Path, verified: bool
) -> None:
    older = LLMCache(cache_dir=tmp_path)
    newer = LLMCache(cache_dir=tmp_path)
    if verified:
        first = older.put_verified("source", "model", "request", "same", {"proof": 1})
        second = newer.put_verified("source", "model", "request", "same", {"proof": 1})
    else:
        first = older.put("source", "model", "request", "same")
        second = newer.put("source", "model", "request", "same")
    assert first is not None and second is not None
    first_entry, second_entry = json.loads(first), json.loads(second)
    assert first_entry.pop("publication_id") != second_entry.pop("publication_id")
    if not verified:
        first_entry.pop("timestamp")
        second_entry.pop("timestamp")
    assert first_entry == second_entry
    older.discard_publication("source", "model", "request", first)
    if verified:
        assert newer.get_verified("source", "model", "request") == {
            "response": "same",
            "metadata": {"proof": 1},
        }
    else:
        assert newer.get("source", "model", "request") == "same"
    newer.discard_publication("source", "model", "request", second)
    if verified:
        assert newer.get_verified("source", "model", "request") is None
    else:
        assert newer.get("source", "model", "request") is None


def _publish_replacement_observing_contention(
    cache_dir: Path,
    verified: bool,
    on_attempt: Callable[[], None],
    on_contention: Callable[[], None],
    cache: LLMCache | None = None,
) -> None:
    """Probe the real native lock before the publisher's blocking acquisition."""
    publisher_thread = threading.get_ident()
    if os.name == "posix":
        import fcntl

        native_flock = fcntl.flock

        def observe_flock(descriptor: int, operation: int) -> None:
            if threading.get_ident() == publisher_thread and operation == fcntl.LOCK_EX:
                try:
                    native_flock(descriptor, operation | fcntl.LOCK_NB)
                except BlockingIOError:
                    on_contention()
                else:
                    native_flock(descriptor, fcntl.LOCK_UN)
                on_attempt()
            native_flock(descriptor, operation)

        observation = patch.object(fcntl, "flock", observe_flock)
    else:
        import msvcrt

        windows_lock = cast(Any, msvcrt)
        native_locking = windows_lock.locking

        def observe_locking(descriptor: int, operation: int, length: int) -> None:
            if (
                threading.get_ident() == publisher_thread
                and operation == windows_lock.LK_LOCK
            ):
                try:
                    native_locking(descriptor, windows_lock.LK_NBLCK, length)
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    on_contention()
                else:
                    native_locking(descriptor, windows_lock.LK_UNLCK, length)
                on_attempt()
            native_locking(descriptor, operation, length)

        observation = patch.object(msvcrt, "locking", observe_locking)
    try:
        with observation:
            writer = cache if cache is not None else LLMCache(cache_dir=cache_dir)
            if verified:
                publication = writer.put_verified(
                    "source", "model", "request", "new", {"proof": 2}
                )
            else:
                publication = writer.put("source", "model", "request", "new")
            assert publication is not None
    finally:
        # A publisher that bypasses the lock must fail the contention assertion,
        # rather than leave the parent waiting for a native call it never made.
        on_attempt()


@pytest.mark.parametrize("verified", [False, True])
@pytest.mark.parametrize("publisher", ["same_instance", "separate_instance", "process"])
@pytest.mark.parametrize(
    "mutation", ["rollback", "corrupt_eviction", "verified_corrupt_eviction", "clear"]
)
def test_same_key_mutations_serialize_with_newer_publications(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified: bool,
    publisher: str,
    mutation: str,
) -> None:
    """Attempt a real replacement after comparison, but before the actual unlink."""
    cache = LLMCache(cache_dir=tmp_path)
    if verified:
        publication = cache.put_verified(
            "source", "model", "request", "old", {"proof": 1}
        )
    else:
        publication = cache.put("source", "model", "request", "old")
    assert publication is not None
    path = cache._cache_path(cache._make_key("source", "model", "request"))
    if mutation in ("corrupt_eviction", "verified_corrupt_eviction"):
        path.write_text("{invalid json", encoding="utf-8")
    if mutation != "clear":
        cache.put("other source", "model", "finished", "preserved")

    before_unlink = threading.Event()
    release_unlink = threading.Event()
    unlink = Path.unlink

    def pause_unlink(candidate: Path, missing_ok: bool = False) -> None:
        if candidate == path and not before_unlink.is_set():
            # discard_publication has already read and compared the old bytes.
            # Eviction has already rejected the record; clear has selected it.
            before_unlink.set()
            assert release_unlink.wait(30), "The test did not release the unlink"
        unlink(candidate, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", pause_unlink)

    def remove_old_record() -> None:
        if mutation == "rollback":
            cache.discard_publication("source", "model", "request", publication)
        elif mutation == "corrupt_eviction":
            assert cache.get("source", "model", "request") is None
        elif mutation == "verified_corrupt_eviction":
            assert cache.get_verified("source", "model", "request") is None
        else:
            assert cache.clear() == 1

    context = multiprocessing.get_context("spawn")
    attempted = context.Event() if publisher == "process" else threading.Event()
    contended = context.Event() if publisher == "process" else threading.Event()
    process = None
    with ThreadPoolExecutor(max_workers=3) as executor:
        removal = executor.submit(remove_old_record)
        replacement = None
        try:
            assert before_unlink.wait(10), "The removal did not reach its unlink"
            if publisher == "process":
                process = context.Process(
                    target=_publish_replacement_observing_contention,
                    args=(tmp_path, verified, attempted.set, contended.set),
                )
                process.start()
            else:
                replacement = executor.submit(
                    _publish_replacement_observing_contention,
                    tmp_path,
                    verified,
                    attempted.set,
                    contended.set,
                    cache if publisher == "same_instance" else None,
                )
            assert attempted.wait(10), "The replacement did not attempt publication"
            # This is an actual nonblocking native-lock refusal, not a timed
            # guess that the publishing thread/process has not been scheduled.
            assert contended.is_set(), "The publication bypassed the removal's lock"
            if mutation != "clear":
                independent = executor.submit(
                    cache.put,
                    "other source",
                    "model",
                    "finished",
                    "independently updated",
                )
                assert independent.result(timeout=10) is not None
        finally:
            release_unlink.set()
            if process is not None and process.pid is not None:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
                    pytest.fail("The test-owned publishing process did not exit")
        removal.result(timeout=10)
        if process is not None:
            assert process.exitcode == 0
            process.close()
        else:
            assert replacement is not None
            replacement.result(timeout=10)

    newer_record = path.read_bytes()
    cache.discard_publication("source", "model", "request", publication)
    assert path.read_bytes() == newer_record
    if verified:
        assert cache.get_verified("source", "model", "request") == {
            "response": "new",
            "metadata": {"proof": 2},
        }
    else:
        assert cache.get("source", "model", "request") == "new"
    if mutation != "clear":
        assert cache.get("other source", "model", "finished") == "independently updated"


@pytest.mark.parametrize("verified", [False, True])
def test_key_lock_inode_survives_rollback_and_clear(
    tmp_path: Path, verified: bool
) -> None:
    cache = LLMCache(cache_dir=tmp_path)
    if verified:
        publication = cache.put_verified(
            "source", "model", "request", "old", {"proof": 1}
        )
    else:
        publication = cache.put("source", "model", "request", "old")
    assert publication is not None
    key = cache._make_key("source", "model", "request")
    lock_path = tmp_path / f".{key}.lock"
    identity = lock_path.stat()
    cache.discard_publication("source", "model", "request", publication)
    assert not cache._cache_path(key).exists()
    cache.put("source", "model", "request", "new")
    assert cache.clear() == 1
    assert cache.summary()["entries_on_disk"] == 0
    assert (lock_path.stat().st_dev, lock_path.stat().st_ino) == (
        identity.st_dev,
        identity.st_ino,
    )


@pytest.mark.parametrize("verified", [False, True])
def test_key_lock_failure_keeps_cache_reads_and_writes_best_effort(
    tmp_path: Path, verified: bool
) -> None:
    healthy = LLMCache(cache_dir=tmp_path / "healthy")
    if verified:
        publication = healthy.put_verified(
            "source", "model", "request", "old", {"proof": 1}
        )
    else:
        publication = healthy.put("source", "model", "request", "old")
    assert publication is not None
    cache = LLMCache(cache_dir=tmp_path / "unavailable")
    key = cache._make_key("source", "model", "request")
    path = cache._cache_path(key)
    path.write_text(publication, encoding="utf-8")
    # Real filesystem acquisition failure, with a complete record left intact.
    (cache.cache_dir / f".{key}.lock").mkdir()
    if verified:
        replacement = cache.put_verified(
            "source", "model", "request", "new", {"proof": 2}
        )
        assert replacement is None
        assert cache.get_verified("source", "model", "request") is None
    else:
        assert cache.put("source", "model", "request", "new") is None
        assert cache.get("source", "model", "request") is None
    assert path.read_text(encoding="utf-8") == publication
    assert cache.writes == 0
    assert cache.misses == 1
