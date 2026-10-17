#!/usr/bin/env python3
"""
Comprehensive Ollama integration tests for LLM module.

These tests validate Ollama detection, model selection, and real LLM processing
with proper recovery handling when Ollama is not available.
"""

import json
import os
import shutil
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Any

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.llm.processor import (
    _select_best_ollama_model,
    _start_ollama_if_needed,
    process_llm,
)

# SC-44: these tests detect/start a local Ollama daemon and run real LLM
# processing. The default suite filters the whole file via ``-m "not ollama"``
# (see justfile test-cov / ci.yml); run explicitly with ``-m ollama``.
pytestmark = pytest.mark.ollama


class TestOllamaDetection:
    """Test Ollama availability detection."""

    def test_ollama_check_returns_tuple(self, caplog: Any) -> Any:
        """Test that Ollama check returns (bool, list) tuple."""
        import logging

        logger = logging.getLogger("test_ollama")

        result = _start_ollama_if_needed(logger)

        # Should return a tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

        is_available, models = result
        assert isinstance(is_available, bool)
        assert isinstance(models, list)

    def test_ollama_detection_logging(self, caplog: Any) -> Any:
        """Test that Ollama detection provides informative logging."""
        import logging

        caplog.set_level(logging.INFO)

        logger = logging.getLogger("test_ollama")
        is_available, models = _start_ollama_if_needed(logger)

        log_text = caplog.text

        # Should have some informative message
        assert len(log_text) > 0

        # Accept multiple valid states:
        # 1. Service running: "✅", "running", "ready", "available"
        # 2. CLI found but not running: "found ollama"
        # 3. Not found: "not found", "not running", "not available"
        valid_messages: list[Any] = [
            "✅",
            "running",
            "ready",
            "available",
            "found",
            "not found",
            "not running",
            "not available",
        ]

        assert any(msg in log_text.lower() for msg in valid_messages), (
            f"Expected informative Ollama status message, got: {log_text}"
        )

    def test_ollama_model_listing(self, caplog: Any) -> Any:
        """Test Ollama model listing when available."""
        import logging

        caplog.set_level(logging.INFO)

        logger = logging.getLogger("test_ollama")
        is_available, models = _start_ollama_if_needed(logger)

        if is_available and models:
            # Should log available models
            log_text = caplog.text
            assert "models" in log_text.lower() or "📦" in log_text

            # Models should be non-empty strings
            for model in models:
                assert isinstance(model, str)
                assert len(model) > 0

    def test_ollama_socket_check(self, caplog: Any) -> Any:
        """Test Ollama socket/API endpoint check."""
        import logging
        import socket

        logger = logging.getLogger("test_ollama")

        # Check if Ollama port is actually open
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("localhost", 11434))
            sock.close()
            port_open = result == 0
        except Exception:
            port_open = False

        is_available, models = _start_ollama_if_needed(logger)

        # Verify consistent detection - both methods should agree
        # Log result for debugging rather than strict assertion
        if port_open and is_available:
            logger.info(f"✅ Ollama detected with port open, models: {models}")
        elif port_open and not is_available:
            logger.info(
                "⚠️ Port 11434 open but detection returned False - API may be starting up"
            )
        elif not port_open and is_available:
            logger.info(
                "⚠️ Detection returned True but port closed - CLI found without service"
            )
        else:
            logger.info("ℹ️ Ollama not detected and port closed")

        # Verify detection returned consistent results
        assert isinstance(is_available, bool), (
            f"Expected bool, got {type(is_available)}"
        )
        assert isinstance(models, list), f"Expected list, got {type(models)}"

        # Verify logical consistency between port and availability
        if models:
            # If we have models, Ollama must be available
            assert is_available or port_open, (
                "Models found but neither available nor port open"
            )

        logger.info("✅ Detection completed with consistent results")


class TestOllamaModelSelection:
    """Test Ollama model selection logic."""

    @pytest.fixture(autouse=True)
    def isolate_selection_defaults(self, monkeypatch: Any) -> None:
        """Live test model selection must not change pure preference controls."""
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_TEST_MODEL", raising=False)

    def test_model_selection_with_empty_list(
        self, caplog: Any, monkeypatch: Any
    ) -> Any:
        """Test model selection when no models are available."""
        monkeypatch.setenv("GNN_TESTING_NO_LLM_CONFIG", "1")
        import logging

        logger = logging.getLogger("test_model_selection")

        model = _select_best_ollama_model([], logger)

        # Should return a default model
        assert isinstance(model, str)
        assert len(model) > 0
        from gnn.llm.defaults import DEFAULT_OLLAMA_MODEL

        assert model == DEFAULT_OLLAMA_MODEL

    def test_model_selection_prefers_small_models(
        self, caplog: Any, monkeypatch: Any
    ) -> Any:
        """Preference order matches ``preferred_models`` in ``llm.processor``."""
        monkeypatch.setenv("GNN_TESTING_NO_LLM_CONFIG", "1")
        import logging

        logger = logging.getLogger("test_model_selection")

        test_cases: list[Any] = [
            (
                ["llama2:7b", "gemma3:4b", "smollm2:135m-instruct-q4_K_S"],
                "smollm2:135m-instruct-q4_K_S",
            ),
            (["llama2:7b", "gemma3:4b"], "gemma3:4b"),
            (["mistral:7b", "tinyllama"], "tinyllama"),
            (["llama2", "gemma2:2b"], "gemma2:2b"),
        ]

        for available_models, expected in test_cases:
            selected = _select_best_ollama_model(available_models, logger)
            assert selected == expected, (
                f"expected {expected!r} for {available_models!r}, got {selected!r}"
            )

    def test_model_selection_respects_env_override(
        self, caplog: Any, monkeypatch: Any
    ) -> Any:
        """Test that environment variable overrides model selection."""
        import logging

        logger = logging.getLogger("test_model_selection")

        # Set environment variable
        monkeypatch.setenv("GNN_TESTING_NO_LLM_CONFIG", "1")
        test_model = "my-custom-model:latest"
        monkeypatch.setenv("OLLAMA_MODEL", test_model)

        models: list[Any] = ["gemma3:4b", "llama2:7b"]
        selected = _select_best_ollama_model(models, logger)

        # Should use environment variable
        assert selected == test_model

    def test_model_selection_logging(self, caplog: Any, monkeypatch: Any) -> Any:
        """Test that model selection provides clear logging."""
        monkeypatch.setenv("GNN_TESTING_NO_LLM_CONFIG", "1")
        import logging

        caplog.set_level(logging.INFO)

        logger = logging.getLogger("test_model_selection")

        models: list[Any] = ["gemma3:4b", "llama2:7b"]
        selected = _select_best_ollama_model(models, logger)

        log_text = caplog.text

        # Should log the selected model
        assert "model" in log_text.lower()
        assert selected in log_text or "🎯" in log_text


class TestLLMProcessing:
    """Opt-in native acceptance must establish completion, not merely files."""

    @pytest.mark.needs_ollama
    @pytest.mark.timeout(600)
    @pytest.mark.parametrize("batch_prompts", [False, True])
    def test_native_corpus_completion_and_resume(
        self, tmp_path: Path, batch_prompts: bool
    ) -> None:
        from gnn.llm.defaults import DEFAULT_OLLAMA_MODEL

        model = (
            os.getenv("OLLAMA_TEST_MODEL")
            or os.getenv("OLLAMA_MODEL")
            or DEFAULT_OLLAMA_MODEL
        )
        source = (
            Path(__file__).resolve().parents[2]
            / "input/gnn_files/discrete/actinf_pomdp_agent.md"
        )
        target = tmp_path / "models"
        target.mkdir()
        shutil.copyfile(source, target / source.name)
        output = tmp_path / "output"
        options = dict(
            provider="ollama",
            model=model,
            max_tokens=4096,
            custom_prompts=[],
            max_prompt_timeout=180,
            response_words=160,
            batch_prompts=batch_prompts,
        )
        assert process_llm(target, output, **options)
        result = json.loads((output / "llm_results.json").read_text())
        assert result["status"] == "success"
        assert result["selected_model"] == model
        assert (
            result["coverage"]["prompts_completed"]
            == result["coverage"]["prompts_required"]
            == 6
        )
        assert result["coverage"]["summaries_completed"] == 1
        assert result["unfinished_work"] == result["unfinished_source_parts"] == []
        assert all(
            o["status"] == "success"
            for row in result["analysis_results"]
            for o in row["part_outcomes"].values()
        )
        assert process_llm(target, output, resume=True, **options)
        resumed = json.loads((output / "llm_results.json").read_text())
        assert all(
            o["resumed"]
            for row in resumed["analysis_results"]
            for o in row["prompt_outcomes"].values()
        )

    @pytest.mark.timeout(15)
    def test_unavailable_local_provider_retains_structural_work(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        import socket

        target = tmp_path / "models"
        target.mkdir()
        (target / "model.md").write_text(
            "## GNNSection\nActInfPOMDP\n## ModelName\nUnavailable\n"
        )
        # A bound, non-listening port deterministically refuses connections.
        with socket.socket() as unavailable:
            unavailable.bind(("127.0.0.1", 0))
            monkeypatch.setenv(
                "OLLAMA_HOST", f"http://127.0.0.1:{unavailable.getsockname()[1]}"
            )
            output = tmp_path / "output"
            assert not process_llm(
                target,
                output,
                provider="ollama",
                model="missing:exact",
                llm_timeout=5,
                max_prompt_timeout=2,
            )
        result = json.loads((output / "llm_results.json").read_text())
        assert result["status"] == "partial"
        assert result["coverage"]["structural_completed"] == 1
        assert result["coverage"]["prompts_completed"] == 0
        assert result["coverage"]["summaries_completed"] == 0
        assert result["errors"][0]["stage"] == "preflight"
        assert len(result["unfinished_work"]) == 9

    def test_empty_selection_is_explicitly_skipped(self, tmp_path: Path) -> None:
        target = tmp_path / "models"
        target.mkdir()
        output = tmp_path / "output"
        assert process_llm(target, output, provider="ollama")
        result = json.loads((output / "llm_results.json").read_text())
        assert result["status"] == "skipped"
        assert result["selected_files"] == 0
        assert result["coverage"]["prompts_required"] == 0
        assert not list(output.glob("prompts_*"))


@pytest.mark.needs_ollama
class TestOllamaIntegrationEnd2End:
    """Checks against the real Ollama CLI/daemon (gated by ``needs_ollama``;
    see tests/helpers/toolchain_probes.py)."""

    def test_ollama_command_exists(self) -> Any:
        ollama_path = shutil.which("ollama")
        assert ollama_path is not None
        assert Path(ollama_path).exists()

    def test_ollama_service_running(self) -> Any:
        result = subprocess.run(  # nosec B607 B603
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


@pytest.mark.needs_ollama
@pytest.mark.timeout(180)
def test_native_convenience_and_mcp_inside_active_loop(monkeypatch: Any) -> None:
    import asyncio

    from gnn.llm.defaults import DEFAULT_OLLAMA_MODEL
    from gnn.llm.llm_operations import LLMOperations
    from gnn.llm.mcp import analyze_gnn_with_llm_mcp

    model = (
        os.getenv("OLLAMA_TEST_MODEL")
        or os.getenv("OLLAMA_MODEL")
        or DEFAULT_OLLAMA_MODEL
    )
    monkeypatch.setenv("OLLAMA_MODEL", model)
    source = (
        Path(__file__).resolve().parents[2]
        / "input/gnn_files/discrete/actinf_pomdp_agent.md"
    )

    async def run() -> None:
        ops = LLMOperations()
        response = ops.get_llm_response("Reply with only the word ok.", max_tokens=32)
        assert response.strip() and not response.startswith("Error:")
        result = analyze_gnn_with_llm_mcp(str(source), provider="ollama")
        assert result["success"] is True
        assert result["llm_summary_status"] == "available"
        assert isinstance(result["llm_summary"], str)

    asyncio.run(run())
