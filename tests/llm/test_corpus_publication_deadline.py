"""Independent deadline reproduction; provider replies are contract injections."""

import json
from types import SimpleNamespace

from gnn.llm import corpus_runner
from gnn.llm import processor as api
from gnn.llm.providers.base_provider import LLMResponse


def test_last_cache_publication_cannot_make_an_exhausted_corpus_success(
    tmp_path, monkeypatch
):
    root = tmp_path / "models"
    root.mkdir()
    (root / "case.md").write_text("## GNNSection\nActInfPOMDP\n## ModelName\ncase\n")
    clock = {"now": 100.0, "puts": 0}

    class Processor:
        async def initialize(self):
            return True

        async def get_response(self, **kwargs):
            return LLMResponse("response", "configured:exact", "ollama")

        async def close(self):
            return None

    original_cache = api.LLMCache

    class Cache(original_cache):
        def put(self, *args, **kwargs):
            value = super().put(*args, **kwargs)
            clock["puts"] += 1
            if clock["puts"] == 6:
                clock["now"] = 103.0
            return value

    monkeypatch.setattr(api, "LLMProcessor", lambda **kwargs: Processor())
    monkeypatch.setattr(api, "LLMCache", Cache)
    monkeypatch.setattr(
        corpus_runner, "time", SimpleNamespace(monotonic=lambda: clock["now"])
    )
    output = tmp_path / "output"
    success = api.process_llm(
        root,
        output,
        llm_config={"model": "configured:exact"},
        total_budget=2,
        custom_prompts=[],
    )
    results = json.loads((output / "llm_results.json").read_text())
    assert not success, results
    assert results["status"] == "timed_out"
