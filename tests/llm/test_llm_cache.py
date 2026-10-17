"""Deterministic tests for the LLMCache content-addressed store."""

import json
import sys
from pathlib import Path

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
