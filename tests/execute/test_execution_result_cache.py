"""Unit tests for ``gnn.execute.result_cache.ExecutionResultCache``.

Pins the content-addressed on-disk envelope cache contract: NUL-joined
sha256 keys (timeout deliberately excluded), corrupt/missing entries are
misses, JSON round-trip alias isolation, success-only stores, explicit
``invalidate() -> int``, ``GNN_EXEC_CACHE`` opt-in enablement, lazy
directory creation, and CWD-relative default ``cache_dir`` resolution.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any, Dict

import pytest

from gnn.execute.result_cache import ExecutionResultCache, cache_key_for_script


def _key_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Canonical ``make_key`` inputs; every call site overrides one axis."""
    kwargs: Dict[str, Any] = {
        "interpreter": "python",
        "script_bytes": b"print('ok')\n",
        "args": (),
        "cwd": None,
        "env_overrides": None,
        "capture_output": True,
    }
    kwargs.update(overrides)
    return kwargs


def _success_envelope(**overrides: Any) -> Dict[str, Any]:
    """Canned successful envelope, including fields only some paths populate."""
    envelope: Dict[str, Any] = {
        "success": True,
        "return_code": 0,
        "stdout": "ok",
        "stderr": "",
        "duration_seconds": 0.01,
        "cancelled": False,
        "sandbox_mode": "off",
        "sandboxed": False,
    }
    envelope.update(overrides)
    return envelope


# ── key derivation ─────────────────────────────────────────────────────────


def test_make_key_is_deterministic_sha256_hex() -> None:
    key = ExecutionResultCache.make_key(**_key_kwargs())
    assert key == ExecutionResultCache.make_key(**_key_kwargs())
    assert re.fullmatch(r"[0-9a-f]{64}", key)


def test_make_key_distinguishes_every_input_axis() -> None:
    base = ExecutionResultCache.make_key(**_key_kwargs())
    variants: list[Dict[str, Any]] = [
        {"script_bytes": b"print('changed')\n"},
        {"args": ("--flag",)},
        {"cwd": "/somewhere"},
        {"env_overrides": {"GNN_VAR": "1"}},
        {"capture_output": False},
    ]
    for overrides in variants:
        keyed = ExecutionResultCache.make_key(**_key_kwargs(**overrides))
        assert keyed != base, f"key must change when {overrides} changes"


def test_make_key_excludes_timeout_from_the_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different wall-clock budget must not change the key (LLMCache style)."""
    monkeypatch.setenv("GNN_EXECUTE_DEFAULT_TIMEOUT", "99")
    key_default = ExecutionResultCache.make_key(**_key_kwargs())
    monkeypatch.setenv("GNN_EXECUTE_DEFAULT_TIMEOUT", "7")
    key_other = ExecutionResultCache.make_key(**_key_kwargs())
    assert key_default == key_other


def test_cache_key_for_script_is_content_addressed(tmp_path: Path) -> None:
    original = tmp_path / "m.py"
    original.write_text("print('v1')\n")
    twin = tmp_path / "twin.py"
    twin.write_text("print('v1')\n")

    assert cache_key_for_script(original, interpreter="python") == (
        cache_key_for_script(twin, interpreter="python")
    )

    original.write_text("print('v2')\n")
    assert cache_key_for_script(original, interpreter="python") != (
        cache_key_for_script(twin, interpreter="python")
    )


def test_cache_key_for_script_unreadable_file_falls_back_to_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "gone.py"
    other_missing = tmp_path / "also-gone.py"

    key = cache_key_for_script(missing, interpreter="python")
    assert re.fullmatch(r"[0-9a-f]{64}", key)
    assert key != cache_key_for_script(other_missing, interpreter="python")


# ── lookup / store / invalidate ────────────────────────────────────────────


def test_lookup_misses_when_no_entry_exists(tmp_path: Path) -> None:
    cache = ExecutionResultCache(cache_dir=tmp_path / "cache", enabled=True)

    assert cache.lookup("absent-key") is None
    assert cache.stats == {"hits": 0, "misses": 1, "writes": 0}
    assert not (tmp_path / "cache").exists()  # misses never mkdir


def test_lookup_treats_corrupt_entry_as_miss(tmp_path: Path) -> None:
    cache = ExecutionResultCache(cache_dir=tmp_path / "cache", enabled=True)
    key = ExecutionResultCache.make_key(**_key_kwargs())
    cache.cache_dir.mkdir(parents=True)
    (cache.cache_dir / f"{key}.json").write_bytes(b"not-json")

    assert cache.lookup(key) is None
    assert cache.stats == {"hits": 0, "misses": 1, "writes": 0}


def test_lookup_hit_is_alias_isolated(tmp_path: Path) -> None:
    cache = ExecutionResultCache(cache_dir=tmp_path / "cache", enabled=True)
    key = ExecutionResultCache.make_key(**_key_kwargs())
    envelope = _success_envelope(stdout="stored")

    assert cache.store(key, envelope) is True
    envelope["stdout"] = "mutated-after-store"  # caller keeps mutating its copy

    first_hit = cache.lookup(key)
    assert first_hit is not None
    assert first_hit["stdout"] == "stored"
    first_hit["stdout"] = "mutated-returned-copy"

    second_hit = cache.lookup(key)
    assert second_hit is not None
    assert second_hit["stdout"] == "stored"


def test_store_only_persists_successful_envelopes(tmp_path: Path) -> None:
    cache = ExecutionResultCache(cache_dir=tmp_path / "cache", enabled=True)
    key = ExecutionResultCache.make_key(**_key_kwargs())

    assert (
        cache.store(
            key,
            _success_envelope(
                success=False,
                return_code=3,
                error="boom",
                error_type="SystemExit",
            ),
        )
        is False
    )
    assert cache.lookup(key) is None
    assert not (tmp_path / "cache").exists()  # rejected store never runs lazy mkdir


def test_invalidate_reports_removed_count_and_empties_dir(tmp_path: Path) -> None:
    cache = ExecutionResultCache(cache_dir=tmp_path / "cache", enabled=True)
    assert not (tmp_path / "cache").exists()
    assert cache.invalidate() == 0  # nothing stored yet; missing dir is no error

    keys = [
        ExecutionResultCache.make_key(**_key_kwargs(script_bytes=bytes([i])))
        for i in range(2)
    ]
    for key in keys:
        assert cache.store(key, _success_envelope()) is True

    assert cache.invalidate() == 2
    assert list(cache.cache_dir.glob("*.json")) == []
    assert cache.invalidate() == 0


def test_stats_track_hits_misses_and_writes(tmp_path: Path) -> None:
    cache = ExecutionResultCache(cache_dir=tmp_path / "cache", enabled=True)
    assert cache.stats == {"hits": 0, "misses": 0, "writes": 0}
    key = ExecutionResultCache.make_key(**_key_kwargs())

    assert cache.lookup("no-such-key") is None
    assert cache.stats["misses"] == 1

    assert cache.store(key, _success_envelope()) is True
    assert cache.stats["writes"] == 1

    assert cache.lookup(key) is not None
    assert cache.stats["hits"] == 1

    assert (
        cache.store("other-key", _success_envelope(success=False, return_code=1))
        is False
    )
    assert cache.stats["writes"] == 1  # failed stores are not counted


# ── enablement ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("explicit", "env_value", "expected"),
    [
        (True, "1", True),
        (False, "1", False),
        (True, "0", True),
        (False, "0", False),
    ],
)
def test_enabled_explicit_flag_wins_over_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit: bool,
    env_value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("GNN_EXEC_CACHE", env_value)
    cache = ExecutionResultCache(cache_dir=tmp_path / "cache", enabled=explicit)
    assert cache.enabled is expected


@pytest.mark.parametrize("env_value", ["1", "TRUE", "yes", "on"])
def test_enabled_env_truthy_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_value: str
) -> None:
    monkeypatch.delenv("GNN_EXEC_CACHE", raising=False)
    cache = ExecutionResultCache(cache_dir=tmp_path / "cache")
    assert cache.enabled is False  # evaluated at access time, env unset → off

    monkeypatch.setenv("GNN_EXEC_CACHE", env_value)
    assert cache.enabled is True


@pytest.mark.parametrize("env_value", ["0", "", "bogus"])
def test_enabled_env_falsy_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_value: str
) -> None:
    monkeypatch.setenv("GNN_EXEC_CACHE", env_value)
    cache = ExecutionResultCache(cache_dir=tmp_path / "cache")
    assert cache.enabled is False


# ── lazy directory creation & default cache_dir ────────────────────────────


def test_default_off_cache_never_creates_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GNN_EXEC_CACHE", raising=False)
    cache_dir = tmp_path / "cache"
    cache = ExecutionResultCache(cache_dir=cache_dir)

    assert not cache_dir.exists()
    assert cache.lookup("some-key") is None
    assert not cache_dir.exists()


def test_enabled_store_creates_directory_and_entry(tmp_path: Path) -> None:
    cache = ExecutionResultCache(cache_dir=tmp_path / "cache", enabled=True)
    key = ExecutionResultCache.make_key(**_key_kwargs())

    assert cache.store(key, _success_envelope()) is True
    assert (tmp_path / "cache").is_dir()
    assert (tmp_path / "cache" / f"{key}.json").is_file()


def test_default_cache_dir_resolves_relative_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cache = ExecutionResultCache(enabled=True)
    assert cache.cache_dir == Path("output/12_execute_output/.cache")

    key = ExecutionResultCache.make_key(**_key_kwargs())
    assert cache.store(key, _success_envelope()) is True
    assert (tmp_path / "output/12_execute_output/.cache" / f"{key}.json").is_file()


# ── thread safety smoke ────────────────────────────────────────────────────


def test_concurrent_store_and_lookup_smoke(tmp_path: Path) -> None:
    cache = ExecutionResultCache(cache_dir=tmp_path / "cache", enabled=True)
    errors: list[BaseException] = []

    def _worker(thread_id: int) -> None:
        try:
            for i in range(50):
                key = ExecutionResultCache.make_key(
                    **_key_kwargs(script_bytes=f"thread-{thread_id}-{i}".encode())
                )
                assert cache.store(key, _success_envelope(stdout=f"t{thread_id}-{i}"))
                hit = cache.lookup(key)
                assert hit is not None
                assert hit["stdout"] == f"t{thread_id}-{i}"
        except BaseException as exc:  # noqa: BLE001 — collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(t,)) for t in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert cache.stats == {"hits": 100, "misses": 0, "writes": 100}


@pytest.mark.parametrize("method", ["safe", "direct"])
def test_cached_success_cannot_override_pre_cancel_or_invalid_budget(
    tmp_path, monkeypatch, method
):
    import sys

    from gnn.execute.executor import GNNExecutor, execute_script_safely
    from gnn.execute.subprocess_envelope import CancelToken

    script = tmp_path / "cached.py"
    script.write_text("raise RuntimeError('must not start')\n")
    cache = ExecutionResultCache(cache_dir=tmp_path / "cache", enabled=True)
    cache.store(
        cache_key_for_script(script, interpreter=sys.executable), _success_envelope()
    )
    monkeypatch.setenv("GNN_ALLOW_UNSAFE_EXEC", "1")
    token = CancelToken()
    token.cancel("already stopped")
    dispatch = (
        (lambda **kw: execute_script_safely(script, cache=cache, **kw))
        if method == "safe"
        else (
            lambda **kw: GNNExecutor(
                str(tmp_path / "out"), cache=cache
            )._execute_jax_script(str(script), **kw)
        )
    )
    result = dispatch(cancel_token=token)
    assert not result["success"] and result["error_type"] == "Cancelled"
    for timeout in [True, 0, float("nan"), float("inf"), -1]:
        result = dispatch(timeout=timeout)
        assert (
            not result["success"] and result["error_type"] == "InvalidExecutionTimeout"
        )
        assert not result.get("cache_hit")


def test_unrendered_pymdp_source_has_no_execution_success(tmp_path):
    from gnn.execute.executor import GNNExecutor

    source = tmp_path / "model.md"
    source.write_text("# Model source\n")
    result = GNNExecutor(str(tmp_path / "out"))._execute_pymdp_script(str(source))
    assert not result["success"] and result["skipped"]
    assert result["error_type"] == "RenderedScriptRequired"
